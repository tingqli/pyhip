"""pyhip.kit: an op is a family of plain functions that share a name prefix.

    ops = Registry()                                  # one per package (e.g. in _ops.py)
    op_register = ops.register

    @op_register
    def axpy_ref(x, y, *, alpha=1.0):                 # op "axpy": interface + reference
        return alpha * x + y

    @op_register
    @param("n", (64, 1000))
    def axpy_test(n, seed=0):                         # test inputs; vocabulary: n, seed
        g = torch.Generator().manual_seed(seed)
        return dict(x=torch.randn(n, generator=g), y=torch.randn(n, generator=g))

    @op_register
    def axpy_dims(x, y, *, alpha=1.0):                # vocabulary values of a real call
        return dict(n=x.numel())

    @op_register                                      # outermost, above the kit marks
    @param("block", (16, 64), id="b{}")
    @requires(lambda n, block: n % block == 0, "n={n} is not a multiple of {block}")
    def axpy_tiled(x, y, *, alpha=1.0, block=16):     # implementation "tiled", knob block
        ...

    ops.axpy.candidates(dict(n=64, seed=0))           # [{"kind": "tiled", "block": 16}, ...]

The vocabulary names are defined by the op's ``<op>_test`` function and describe a
concrete problem instance (for example ``n`` and ``seed`` in ``axpy_test``). Knobs are
defined by an implementation function such as ``<op>_<kind>`` and are the tuning
choices of that implementation (for example ``block`` in ``axpy_tiled``). In other
words, vocabulary answers "what problem am I solving?" and knobs answer "how is this
implementation configured?".

The vocabulary is used by ``<op>_dims``, ``@requires``, and ``@meta`` logic when a
real call or a test case is evaluated. The knobs are used by the implementation's
``@param`` grid, config naming/parsing, and runtime dispatch to choose a valid
configuration for a given vocabulary.

Registration: ``Registry.register`` (``@op_register``) adds one function to its op and
returns it unchanged, after checking it against the conventions below:

* It is the outermost decorator: the kit marks below it are complete when it checks
  them, and a mark applied above it raises.
* The name gives the role. ``<op>_ref`` defines op ``<op>`` and is registered first.
  Any other ``<op>_<suffix>`` joins the registered op with the longest such prefix
  (``moe_dyn_ep3k`` belongs to ``moe_dyn`` once ``moe_dyn_ref`` is registered): suffix
  ``test`` / ``check`` / ``dims`` is that role, any other suffix an implementation.
  ``<op>_test`` comes before the implementations (their limits are checked against its
  vocabulary).
* One registry per package (a small module holding ``ops = Registry()``, imported by
  the modules that register): a package imported under two names keeps two registries.

Conventions:

* ``<op>_ref`` defines op ``<op>``: its parameters are the interface (named parameters
  only), its result the expected answer.
* ``<op>_<kind>`` is implementation ``kind``; the longest op prefix wins (``moe_dyn_ep3k``
  belongs to ``moe_dyn`` when ``moe`` and ``moe_dyn`` both exist). Parameters: exactly
  the ref's, then keyword-only knobs with defaults, each with a ``@param``, then
  optionally ``**over`` (free overrides, e.g. for experiments).
* ``<op>_test(...)`` returns the interface arguments (a dict) of one problem. Its
  parameter names are the op's vocabulary, the names ``requires`` / ``meta`` / ``dims``
  speak; its ``@param`` grid gives the test cases.
* ``<op>_check(out, expected, **inputs)`` raises when ``out`` is wrong; it may return a
  dict of error metrics. Default: ``torch.testing.assert_close``. The first two
  arguments come by position: name them apart from the interface's parameters.
* ``<op>_dims(...)`` takes the ref's parameters and returns vocabulary values.
* Suffixes ``ref``, ``test``, ``check``, ``dims`` are reserved; an implementation name
  does not end in ``_test`` / ``_check`` / ``_dims`` either. Names starting with ``_``
  are not registered.

A config is a dict ``{"kind": kind, knob: value, ...}``. Its name joins the kind and the
``@param`` ids with spaces (``"tiled b64"``), overrides appended as ``":k=v,k=v"``.

The decorators only record themselves in ``f.kit_marks`` (in written order) and return
``f`` unchanged. ``pyhip.kit.testing`` turns the ops into pytest tests.
"""

import inspect
import itertools
import operator
import re
import string
import types
from collections.abc import Mapping

__all__ = ["param", "requires", "meta", "stack", "Registry", "Op"]

ROLES = ("ref", "test", "check", "dims")
_FIELD = re.compile(r"\{[^{}]*\}")
_NAMED = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


class _Mark:
    """Base of the decorators: record ``self`` in ``f.kit_marks`` and return ``f``."""

    def __call__(self, f):
        if not callable(f) or isinstance(f, _Mark):
            raise TypeError(f"{type(self).__name__} decorates a function, got {f!r}")
        if "kit_op" in getattr(f, "__dict__", ()):
            raise TypeError(f"{f.__name__}: @op_register must be the outermost decorator, "
                            f"above {type(self).__name__}")
        f.__dict__.setdefault("kit_marks", []).insert(0, self)  # applied bottom-up
        return f


def _marks(f, cls=_Mark):
    return [m for m in getattr(f, "kit_marks", ()) if isinstance(m, cls)]


def _getter(names):
    """ns -> tuple of ns[name] for ``names`` (C-level for 2+ names: limits are checked
    on every dispatch)."""
    if len(names) > 1:
        return operator.itemgetter(*names)
    if names:
        one = operator.itemgetter(names[0])
        return lambda ns: (one(ns),)
    return lambda ns: ()


class _Call:
    """A user function called with the values of its parameter names (``names``); a
    ``**kw`` parameter also receives the extras."""

    __slots__ = ("f", "names", "pos", "varkw", "plain", "get")

    def __init__(self, f):
        ps = inspect.signature(f).parameters.values()
        self.f = f
        self.names = tuple(p.name for p in ps if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD))
        self.pos = all(p.kind is not p.KEYWORD_ONLY for p in ps)
        self.varkw = any(p.kind is p.VAR_KEYWORD for p in ps)
        self.plain = self.pos and not self.varkw  # f(*values)
        self.get = _getter(self.names)

    def __call__(self, vals, extras):
        if self.pos:
            return self.f(*vals, **extras) if self.varkw else self.f(*vals)
        kw = dict(zip(self.names, vals))
        return self.f(**kw, **extras) if self.varkw else self.f(**kw)


def _guess(s):
    for t in (int, float):
        try:
            return t(s)
        except ValueError:
            pass
    return s


def _convert(s, default):
    """Config-name text -> value, typed like the knob's default."""
    if isinstance(default, bool):
        if s not in ("True", "False"):
            raise ValueError(s)
        return s == "True"
    return _guess(s) if default is None else type(default)(s)


def _unknown(names, known):
    return sorted(set(names) - set(known))


class param(_Mark):
    """One axis of a grid; stacked ``@param``s form their Cartesian product, the first
    written outermost.

        @param("bm", (32, 64), id="bm{}")             # a sequence of values
        @param("act", ("silu", "gelu"), id="{}")
        @param("fp8", {"bf16": False, "fp8": True})   # a dict: the keys are the ids
        @param("tokens, valid", {"v0": (512, 0), "v37": (512, 37)})   # several names
        @param("split", lambda dim: (1, 2, 4))        # a function of vocabulary names
                                                      # (implementation knobs only)

    ``id`` is a format string with one ``{}`` field (default ``name + "{}"``): the
    axis' part of config names and test ids. Config names must read back, so a
    value's text holds no space or ":".
    """

    def __init__(self, names, values, id=None):
        self.names = tuple(n.strip() for n in names.split(","))
        if not all(n.isidentifier() for n in self.names) or len(set(self.names)) < len(self.names):
            raise TypeError(f"param: bad names {names!r}")
        self.fmt = self.table = self.values = None
        if isinstance(values, str):
            raise TypeError(f"param {names!r}: values {values!r} is a string, not a sequence")
        if isinstance(values, dict):
            if id is not None:
                raise TypeError(f"param {names!r}: a dict's keys are the ids; drop id=")
            self.table = {}
            for k, v in values.items():
                v = tuple(v) if len(self.names) > 1 else (v,)
                if len(v) != len(self.names):
                    raise TypeError(f"param {names!r}: {k!r} has {len(v)} values")
                self.table[str(k)] = dict(zip(self.names, v))
        elif len(self.names) > 1:
            raise TypeError(f"param {names!r}: several names take a dict {{id: values}}")
        else:
            self.fmt = self.names[0] + "{}" if id is None else id
            if len(_FIELD.findall(self.fmt)) != 1:
                raise TypeError(f"param {names!r}: id {self.fmt!r} needs exactly one {{}} field")
            self.values = _Call(values) if callable(values) else tuple(values)

    def __repr__(self):
        return f"param({', '.join(self.names)!r})"

    def points(self, vocab=None):
        """[(id, {name: value})] of the axis (``vocab`` feeds a values function)."""
        if self.table is not None:
            return [(k, dict(d)) for k, d in self.table.items()]
        vals = self.values
        if isinstance(vals, _Call):
            vals = vals([vocab[n] for n in vals.names], {})
        return [(self.fmt.format(v), {self.names[0]: v}) for v in vals]

    def token(self, knobs):
        """The id of the axis point holding these values."""
        if self.table is None:
            return self.fmt.format(knobs[self.names[0]])
        for k, d in self.table.items():
            if all(knobs[n] == d[n] for n in self.names):
                return k
        raise ValueError(f"{self!r}: {[knobs[n] for n in self.names]} is not in "
                         f"{list(self.table)}")

    def pattern(self, group):
        """Regex of the axis' ids, the text in group ``group``."""
        if self.table is None:
            pre, post = _FIELD.split(self.fmt)
            return re.escape(pre) + f"(?P<{group}>[^\\s:]+?)" + re.escape(post)
        keys = sorted(self.table, key=len, reverse=True)
        return f"(?P<{group}>{'|'.join(map(re.escape, keys))})"

    def decode(self, text, defaults):
        """The inverse of ``token``: {name: value}, typed like ``defaults``."""
        if self.table is not None:
            return dict(self.table[text])
        return {self.names[0]: _convert(text, defaults[self.names[0]])}


def _fields(fmt):
    """The names a format string refers to (``"{a.b} {c[0]}"`` -> {"a", "c"})."""
    return {re.match(r"[^.\[]*", name)[0]
            for _, name, _, _ in string.Formatter().parse(fmt) if name is not None}


class requires(_Mark):
    """A limit of an implementation: a config cannot run unless ``pred`` is true.

        @requires(lambda dim, block: dim % block == 0, "dim={dim} % block {block}")

    ``pred`` and ``reason`` (a format string) use vocabulary names, knobs and
    ``capturing`` (True when asked for CUDA-graph capture); a ``**kw`` in ``pred``
    receives the config's overrides. The limit is skipped when a value it needs is
    None (unknown). One ``requires`` object can decorate several functions.
    """

    def __init__(self, pred, reason):
        self.pred, self.reason = _Call(pred), reason
        self.fields = _fields(reason)
        if not all(n.isidentifier() for n in self.fields):
            raise TypeError(f"requires: reason {reason!r} may only use named fields")

    def __repr__(self):
        return f"requires({self.reason!r})"


class meta(_Mark):
    """Metadata for tools, read with ``Op.meta``: constants, or functions of names (as in
    ``requires``). ``@meta(tile_m=64, cached=lambda bcm: bcm != 2)``."""

    def __init__(self, **items):
        self.items = {k: _Call(v) if callable(v) else v for k, v in items.items()}


class stack(_Mark):
    """Several marks as one, e.g. a configuration space shared by implementations:
    ``@stack(a, b)`` is ``@a`` followed by ``@b``."""

    def __init__(self, *marks):
        for m in marks:
            if not isinstance(m, _Mark):
                raise TypeError(f"stack takes kit marks, got {m!r}")
        self.marks = marks

    def __call__(self, f):
        for m in reversed(self.marks):
            f = m(f)
        return f


def _sig(op, role, f):
    return f"{op}_{role}{inspect.signature(f)}"


class _Impl:
    """An implementation as its op sees it."""

    __slots__ = ("kind", "f", "params", "requires", "meta", "defaults", "knobs", "keys", "varkw",
                 "regex")

    def __init__(self, op, kind, f, ref_params, vocab):
        where = f"{op}_{kind}"
        if not kind.isidentifier() or kind in ROLES:
            raise TypeError(f"{where}: bad implementation name {kind!r}")
        ps = list(inspect.signature(f).parameters.values())
        if ([(p.name, p.kind, p.default) for p in ps[:len(ref_params)]]
                != [(p.name, p.kind, p.default) for p in ref_params]):
            raise TypeError(f"{_sig(op, kind, f)}: the parameters must start with exactly "
                            f"those of {op}_ref{inspect.Signature(ref_params)}")
        self.kind, self.f, self.defaults, self.varkw = kind, f, {}, False
        for p in ps[len(ref_params):]:
            if p.kind is p.VAR_KEYWORD:
                self.varkw = True
            elif p.kind is p.KEYWORD_ONLY and p.default is not p.empty:
                self.defaults[p.name] = p.default
            else:
                raise TypeError(f"{where}: parameter '{p}' after the {op}_ref ones must be a "
                                "keyword-only knob with a default, or **over")
        knobs = list(self.defaults)
        clash = [k for k in knobs if k in vocab or k in ("kind", "capturing")]
        if clash:
            raise TypeError(f"{where}: knob(s) {clash} clash with the vocabulary of {op} "
                            f"{list(vocab)} or with 'kind' / 'capturing'")
        self.params = tuple(_marks(f, param))
        names = [n for p in self.params for n in p.names]
        if len(set(names)) < len(names) or set(names) != set(knobs):
            raise TypeError(f"{where}: each knob needs exactly one @param; knobs {knobs}, "
                            f"@param names {names}")
        self.knobs = frozenset(knobs)
        self.keys = self.knobs | {"kind"}
        self.requires = tuple(_marks(f, requires))
        self.meta = {}
        for m in _marks(f, meta):
            self.meta.update(m.items)
        known = (*vocab, *knobs, "capturing")
        for r in self.requires:
            bad = _unknown((*r.pred.names, *r.fields), known)
            if bad:
                raise TypeError(f"{where}: {r!r} uses unknown name(s) {bad}; known: {list(known)}")
        for k, v in self.meta.items():
            bad = _unknown(v.names, known) if isinstance(v, _Call) else ()
            if bad:
                raise TypeError(f"{where}: meta {k!r} uses unknown name(s) {bad}; known: "
                                f"{list(known)}")
        for p in self.params:
            bad = _unknown(p.values.names, vocab) if isinstance(p.values, _Call) else ()
            if bad:
                raise TypeError(f"{where}: the values of {p!r} use name(s) {bad} outside the "
                                f"vocabulary {list(vocab)}")
        rx = re.escape(kind) + "".join(" " + p.pattern(f"_p{i}") for i, p in enumerate(self.params))
        self.regex = re.compile(rx + ("(?::(?P<_over>.+))?" if self.varkw else "") + r"\Z")


class Op:
    """One op: its functions and the queries tools need. Built by ``Registry.register``
    one function at a time; read-only otherwise.

    Attributes: ``name``, ``ref``, ``test``, ``check``, ``dims`` (None when absent),
    ``impls`` ({kind: function}), ``vocab`` (the ``<op>_test`` parameter names). A
    ``vocab`` argument is a dict {name: value} holding every vocabulary name (None:
    unknown; the limits that need it are skipped).
    """

    __slots__ = ("name", "ref", "test", "check", "dims", "impls", "vocab", "_imp", "_fns",
                 "_axes", "_test_defaults", "_ref_params")

    def __init__(self, name, ref):
        ref_params = list(inspect.signature(ref).parameters.values())
        if not all(p.kind in _NAMED for p in ref_params):
            raise TypeError(f"{_sig(name, 'ref', ref)}: the interface takes named parameters "
                            "only (no '/', *args, **kwargs)")
        if _marks(ref):
            raise TypeError(f"{name}_ref: only implementations and {name}_test take kit marks")
        fns = {}
        for k, v in dict(name=name, ref=ref, test=None, check=None, dims=None,
                         impls=types.MappingProxyType(fns), vocab=(), _imp={}, _fns=fns,
                         _axes=(), _test_defaults={}, _ref_params=ref_params).items():
            object.__setattr__(self, k, v)

    def _add(self, suffix, f):
        """Add ``f`` as role ``suffix`` (test / check / dims) or as implementation
        ``suffix``; the op is unchanged when ``f`` breaks a convention."""
        if suffix in ROLES and getattr(self, suffix) is not None:
            raise TypeError(f"{self.name}_{suffix} is already registered")
        if suffix == "test":
            self._add_test(f)
        elif suffix in ROLES:
            self._add_helper(suffix, f)
        else:
            self._add_impl(suffix, f)

    def _add_test(self, test):
        name = self.name
        if self._imp:
            raise TypeError(f"{name}_test: register it before the implementations "
                            f"{list(self._imp)} (their limits use its vocabulary)")
        tps = list(inspect.signature(test).parameters.values())
        if not all(p.kind in _NAMED for p in tps):
            raise TypeError(f"{_sig(name, 'test', test)}: named parameters only (their "
                            "names are the vocabulary)")
        if any(not isinstance(m, param) for m in _marks(test)):
            raise TypeError(f"{name}_test: only @param marks")
        vocab = tuple(p.name for p in tps)
        axes = tuple(_marks(test, param))
        grid = [n for p in axes for n in p.names]
        if _unknown(grid, vocab) or len(set(grid)) < len(grid):
            raise TypeError(f"{name}_test: the @param names {grid} must be distinct "
                            f"parameters of {_sig(name, 'test', test)}")
        if any(isinstance(p.values, _Call) for p in axes):
            raise TypeError(f"{name}_test: @param values must be a sequence or a dict")
        missing = [p.name for p in tps if p.default is p.empty and p.name not in grid]
        if missing:
            raise TypeError(f"{name}_test: parameter(s) {missing} need a default or a @param")
        defaults = {p.name: p.default for p in tps if p.default is not p.empty}
        for k, v in dict(test=test, vocab=vocab, _axes=axes, _test_defaults=defaults).items():
            object.__setattr__(self, k, v)

    def _add_helper(self, role, f):
        """``<op>_check`` / ``<op>_dims``: no marks; they take the ref's arguments."""
        if _marks(f):
            raise TypeError(f"{self.name}_{role}: only implementations and {self.name}_test "
                            "take kit marks")
        inputs = {p.name: None for p in self._ref_params}
        try:
            inspect.signature(f).bind(*((None, None) if role == "check" else ()), **inputs)
        except TypeError as e:
            raise TypeError(f"{_sig(self.name, role, f)} does not take the arguments of "
                            f"{_sig(self.name, 'ref', self.ref)}: {e}") from None
        object.__setattr__(self, role, f)

    def _add_impl(self, kind, f):
        if kind in self._imp:
            raise TypeError(f"{self.name}_{kind} is already registered")
        imp = _Impl(self.name, kind, f, self._ref_params, self.vocab)
        self._imp[kind], self._fns[kind] = imp, f
        try:
            self._check_names(imp)
        except Exception:
            del self._imp[kind], self._fns[kind]
            raise

    def __setattr__(self, k, v):
        raise AttributeError(f"Op is read-only ({k})")

    def __repr__(self):
        return f"Op({self.name!r}, impls={list(self._fns)})"

    def _check_names(self, imp):
        """Every grid value prints into a config name that reads back to it."""
        static = [p.points() for p in imp.params if not isinstance(p.values, _Call)]
        base = {"kind": imp.kind}
        for pts in static:
            base.update(pts[0][1] if pts else {})
        for pts in static:
            for _, vals in pts:
                name = self.name_of({**base, **vals})
                try:
                    back = self.parse(name)
                except ValueError as e:
                    raise TypeError(f"{self.name}_{imp.kind}: {e}") from None
                if any(back[k] != v for k, v in vals.items()):
                    raise TypeError(f"{self.name}_{imp.kind}: {vals} prints as {name!r}, which "
                                    f"reads back as {back}")

    def cases(self):
        """[(id, case)] of the ``<op>_test`` grid: ``case`` holds every ``<op>_test``
        argument (defaults filled in), ``id`` joins the axes' ids with "-"."""
        if self.test is None:
            return []
        out = []
        for combo in itertools.product(*(p.points() for p in self._axes)):
            case = dict(self._test_defaults)
            for _, vals in combo:
                case.update(vals)
            out.append(("-".join(tok for tok, _ in combo), case))
        return out

    def _split(self, cfg):
        """cfg -> (implementation, knobs with defaults filled in, overrides)."""
        knobs = dict(cfg)
        kind = knobs.pop("kind", None)
        imp = self._imp.get(kind)
        if imp is None:
            raise KeyError(f"{self.name}: no implementation {kind!r}; have {list(self._imp)}")
        extras = {k: v for k, v in knobs.items() if k not in imp.knobs}
        if extras and not imp.varkw:
            raise TypeError(f"{self.name}_{kind}: unknown knob(s) {sorted(extras)}")
        return imp, {**imp.defaults, **{k: v for k, v in knobs.items() if k in imp.knobs}}, extras

    def _why(self, imp, knobs, extras, vocab, capturing):
        if not imp.requires:
            return None
        ns = {**vocab, **knobs, "capturing": capturing}
        for r in imp.requires:
            p = r.pred
            try:
                vals = p.get(ns)
            except KeyError:
                raise TypeError(f"{self.name}_{imp.kind}: {r!r} needs "
                                f"{_unknown(p.names, ns)}, missing from the vocabulary "
                                f"passed {sorted(vocab)}") from None
            if None in vals:
                continue
            if not (p.f(*vals) if p.plain else p(vals, extras)):
                return r.reason.format_map({**ns, **extras})
        return None

    def why_not(self, cfg, vocab, *, capturing=False):
        """None when ``cfg`` meets every ``requires`` of its implementation, else the
        reason of the first one it fails."""
        imp = self._imp.get(cfg.get("kind"))
        if imp is not None and cfg.keys() == imp.keys:  # fast path: all knobs, no overrides
            knobs = dict(cfg)
            del knobs["kind"]
            return self._why(imp, knobs, {}, vocab, capturing)
        imp, knobs, extras = self._split(cfg)
        return self._why(imp, knobs, extras, vocab, capturing)

    def configs(self, vocab, *, capturing=False):
        """(cfg, None or the reason it cannot run) for every grid point of every
        implementation, in definition order."""
        for imp in self._imp.values():
            try:
                axes = [p.points(vocab) for p in imp.params]
            except KeyError as e:
                raise TypeError(f"{self.name}_{imp.kind}: the vocabulary passed lacks "
                                f"{e}") from None
            for combo in itertools.product(*axes):
                knobs = {}
                for _, vals in combo:
                    knobs.update(vals)
                yield (dict(kind=imp.kind, **knobs),
                       self._why(imp, {**imp.defaults, **knobs}, {}, vocab, capturing))

    def candidates(self, vocab, *, capturing=False):
        """The configs that can run on ``vocab``."""
        return [c for c, why in self.configs(vocab, capturing=capturing) if why is None]

    def name_of(self, cfg):
        imp, knobs, extras = self._split(cfg)
        s = " ".join([imp.kind] + [p.token(knobs) for p in imp.params])
        return s + (":" + ",".join(f"{k}={v}" for k, v in extras.items()) if extras else "")

    def parse(self, name):
        """The inverse of ``name_of`` (override values come back as int, float or str)."""
        kind = re.match(r"[^\s:]*", name)[0]
        imp = self._imp.get(kind)
        m = imp.regex.match(name) if imp else None
        if m is None:
            raise ValueError(f"{self.name}: cannot parse config name {name!r}"
                             + (f" (pattern {imp.regex.pattern!r})" if imp else
                                f"; kinds: {list(self._imp)}"))
        cfg = dict(kind=kind)
        for i, p in enumerate(imp.params):
            text = m[f"_p{i}"]
            try:
                cfg.update(p.decode(text, imp.defaults))
            except ValueError:
                raise ValueError(f"{self.name}: {name!r}: bad {p!r} value {text!r}") from None
        if imp.varkw and m["_over"]:
            for kv in m["_over"].split(","):
                k, eq, v = kv.partition("=")
                if not eq or not k.isidentifier() or k in cfg:
                    raise ValueError(f"{self.name}: {name!r}: bad override {kv!r}")
                cfg[k] = _guess(v)
        return cfg

    def run(self, cfg, *args, **kwargs):
        """Call the implementation of ``cfg`` with its knobs (and overrides)."""
        knobs = dict(cfg)
        return self._fns[knobs.pop("kind")](*args, **kwargs, **knobs)

    def meta(self, cfg, key, vocab=None):
        """``@meta`` item ``key`` of ``cfg``'s implementation; a function is evaluated on
        ``vocab``, the knobs and the overrides."""
        imp, knobs, extras = self._split(cfg)
        try:
            v = imp.meta[key]
        except KeyError:
            raise KeyError(f"{self.name}_{imp.kind}: no meta {key!r}; it has "
                           f"{sorted(imp.meta)}") from None
        if not isinstance(v, _Call):
            return v
        ns = {**(vocab or {}), **knobs, "capturing": False}
        try:
            vals = [ns[n] for n in v.names]
        except KeyError:
            raise TypeError(f"{self.name}_{imp.kind}: meta {key!r} needs "
                            f"{_unknown(v.names, ns)}") from None
        return v(vals, extras)


class Registry(Mapping):
    """The ops of a package, built one function at a time by ``register`` (used as the
    decorator ``@op_register``, see the module docstring). Read it as a mapping {op
    name: Op} in registration order, or by attribute: ``ops.moe is ops["moe"]``. An
    ``Op`` is the same object from its ref's registration on; later registrations add
    to it."""

    def __init__(self):
        self._ops = {}

    def register(self, f):
        """Add ``f`` to its op, by its name (see the module docstring); return ``f``."""
        if not inspect.isfunction(f):
            raise TypeError(f"op_register registers a function, got {f!r}")
        name = f.__name__
        if "kit_op" in f.__dict__:
            raise TypeError(f"{name} is already registered (op {f.kit_op!r})")
        if name.startswith("_") or not name.isidentifier():
            raise TypeError(f"{name!r}: private or unnamed functions are not registered")
        if name.endswith("_ref") and len(name) > 4:
            op = name[:-4]
            self._new_op(op, f)
        else:
            op = self._owner(name)
            self._ops[op]._add(name[len(op) + 1:], f)
        f.kit_op = op
        return f

    def _new_op(self, op, ref):
        if op in self._ops:
            raise TypeError(f"{op}_ref: op {op!r} is already registered")
        if hasattr(Registry, op):
            raise TypeError(f"{op}_ref: the op name {op!r} is taken by Registry.{op}")
        for o in self._ops.values():
            pre = op[len(o.name) + 1:] + "_" if op.startswith(o.name + "_") else None
            early = [k for k in o.impls if pre and k.startswith(pre)]
            if early:
                raise TypeError(f"{op}_ref: {o.name}_{early[0]} was registered before it and "
                                f"went to op {o.name!r}; register {op}_ref first")
        self._ops[op] = Op(op, ref)

    def _owner(self, name):
        """The registered op with the longest prefix of ``name``."""
        op = max((o for o in self._ops if name.startswith(o + "_") and len(name) > len(o) + 1),
                 key=len, default=None)
        if op is None:
            raise TypeError(f"{name}: no registered op is a prefix of the name; register its "
                            f"<op>_ref first (registered: {list(self._ops)})")
        head, _, role = name[len(op) + 1:].rpartition("_")
        if head and role in ROLES:
            raise TypeError(f"{name}: no op {op}_{head} is registered; register "
                            f"{op}_{head}_ref first (implementation names do not end in "
                            f"_{role})")
        return op

    def __getitem__(self, name):
        return self._ops[name]

    def __iter__(self):
        return iter(self._ops)

    def __len__(self):
        return len(self._ops)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._ops[name]
        except KeyError:
            raise AttributeError(f"no op {name!r}; registered: {list(self._ops)} (import the "
                                 f"module that registers {name}_ref)") from None

    def __dir__(self):
        return [*super().__dir__(), *self._ops]

    def __repr__(self):
        return f"Registry({list(self._ops)})"


def collect(namespace):
    """Transition only (removed when a8w4 registers its functions): a Registry of the
    functions of ``namespace`` named ``<op>_...`` for some ``<op>_ref`` in it, the refs
    registered first, then the tests, then the rest in namespace order."""
    fns = {k: f for k, f in namespace.items() if not k.startswith("_") and inspect.isfunction(f)}
    reg = Registry()
    refs = [f for k, f in fns.items() if k.endswith("_ref") and len(k) > 4]
    for f in refs:
        reg.register(f)
    rest = [f for k, f in fns.items() if f not in refs
            and any(k.startswith(o + "_") and len(k) > len(o) + 1 for o in reg)]
    for f in sorted(rest, key=lambda f: not f.__name__.endswith("_test")):
        reg.register(f)
    return reg
