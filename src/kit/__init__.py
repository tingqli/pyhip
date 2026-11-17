"""pyhip.kit: an op is a family of plain functions that share a name prefix.

    def axpy_ref(x, y, *, alpha=1.0):                 # op "axpy": interface + reference
        return alpha * x + y

    @param("n", (64, 1000))
    def axpy_test(n, seed=0):                         # test inputs; vocabulary: n, seed
        g = torch.Generator().manual_seed(seed)
        return dict(x=torch.randn(n, generator=g), y=torch.randn(n, generator=g))

    def axpy_dims(x, y, *, alpha=1.0):                # vocabulary values of a real call
        return dict(n=x.numel())

    @param("block", (16, 64), id="b{}")
    @requires(lambda n, block: n % block == 0, "n={n} is not a multiple of {block}")
    def axpy_tiled(x, y, *, alpha=1.0, block=16):     # implementation "tiled", knob block
        ...

    OPS = collect(globals())                          # {"axpy": Op}
    OPS["axpy"].candidates(dict(n=64, seed=0))        # [{"kind": "tiled", "block": 16}, ...]

Conventions, checked when ``collect`` builds the ops:

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
  dict of error metrics. Default: ``torch.testing.assert_close``.
* ``<op>_dims(...)`` takes the ref's parameters and returns vocabulary values.
* Suffixes ``ref``, ``test``, ``check``, ``dims`` are reserved. Names starting with
  ``_``, classes and kit marks are not collected.

A config is a dict ``{"kind": kind, knob: value, ...}``. Its name joins the kind and the
``@param`` ids with spaces (``"tiled b64"``), overrides appended as ``":k=v,k=v"``.

The decorators only record themselves in ``f.kit_marks`` (in written order) and return
``f`` unchanged. ``pyhip.kit.testing`` turns the ops into pytest tests.
"""

import inspect
import itertools
import re
import string
import types

__all__ = ["param", "requires", "meta", "stack", "collect", "Op"]

ROLES = ("ref", "test", "check", "dims")
_FIELD = re.compile(r"\{[^{}]*\}")
_NAMED = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


class _Mark:
    """Base of the decorators: record ``self`` in ``f.kit_marks`` and return ``f``."""

    def __call__(self, f):
        if not callable(f) or isinstance(f, _Mark):
            raise TypeError(f"{type(self).__name__} decorates a function, got {f!r}")
        f.__dict__.setdefault("kit_marks", []).insert(0, self)  # applied bottom-up
        return f


def _marks(f, cls=_Mark):
    return [m for m in getattr(f, "kit_marks", ()) if isinstance(m, cls)]


class _Call:
    """A user function called with the values of its parameter names (``names``); a
    ``**kw`` parameter also receives the extras."""

    __slots__ = ("f", "names", "pos", "varkw")

    def __init__(self, f):
        ps = inspect.signature(f).parameters.values()
        self.f = f
        self.names = tuple(p.name for p in ps if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD))
        self.pos = all(p.kind is not p.KEYWORD_ONLY for p in ps)
        self.varkw = any(p.kind is p.VAR_KEYWORD for p in ps)

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
    """One op: its functions and the queries tools need. Built by ``collect``; read-only.

    Attributes: ``name``, ``ref``, ``test``, ``check``, ``dims`` (None when absent),
    ``impls`` ({kind: function}), ``vocab`` (the ``<op>_test`` parameter names). A
    ``vocab`` argument is a dict {name: value} holding every vocabulary name (None:
    unknown; the limits that need it are skipped).
    """

    __slots__ = ("name", "ref", "test", "check", "dims", "impls", "vocab", "_imp", "_fns",
                 "_axes", "_test_defaults")

    def __init__(self, name, ref, impls=None, *, test=None, check=None, dims=None):
        ref_params = list(inspect.signature(ref).parameters.values())
        if not all(p.kind in _NAMED for p in ref_params):
            raise TypeError(f"{_sig(name, 'ref', ref)}: the interface takes named parameters "
                            "only (no '/', *args, **kwargs)")
        for role, f in dict(ref=ref, check=check, dims=dims).items():
            if f is not None and _marks(f):
                raise TypeError(f"{name}_{role}: only implementations and {name}_test take "
                                "kit marks")
        vocab, axes, defaults = (), (), {}
        if test is not None:
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
                raise TypeError(f"{name}_test: parameter(s) {missing} need a default or a "
                                "@param")
            defaults = {p.name: p.default for p in tps if p.default is not p.empty}
        inputs = {p.name: None for p in ref_params}
        for role, f, args in (("check", check, (None, None)), ("dims", dims, ())):
            try:
                if f is not None:
                    inspect.signature(f).bind(*args, **inputs)
            except TypeError as e:
                raise TypeError(f"{_sig(name, role, f)} does not take the arguments of "
                                f"{_sig(name, 'ref', ref)}: {e}") from None
        fns = dict(impls or {})
        imp = {k: _Impl(name, k, f, ref_params, vocab) for k, f in fns.items()}
        for k, v in dict(name=name, ref=ref, test=test, check=check, dims=dims,
                         impls=types.MappingProxyType(fns), vocab=vocab, _imp=imp, _fns=fns,
                         _axes=axes, _test_defaults=defaults).items():
            object.__setattr__(self, k, v)
        for i in imp.values():
            self._check_names(i)

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
            try:
                vals = [ns[n] for n in r.pred.names]
            except KeyError:
                raise TypeError(f"{self.name}_{imp.kind}: {r!r} needs "
                                f"{_unknown(r.pred.names, ns)}, missing from the vocabulary "
                                f"passed {sorted(vocab)}") from None
            if None in vals:
                continue
            if not r.pred(vals, extras):
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


def collect(namespace):
    """{op name: Op} from the functions of ``namespace`` (a module's ``globals()``).

    ``<op>_ref`` defines op ``<op>``; every other function named ``<op>_<suffix>`` (the
    longest such op) is its test / check / dims, or implementation ``suffix``. Raises
    TypeError when a function breaks a convention (see the module docstring)."""
    fns = {k: v for k, v in namespace.items()
           if not k.startswith("_") and callable(v) and not isinstance(v, (_Mark, type))}
    refs = {k[:-4]: f for k, f in fns.items() if k.endswith("_ref") and len(k) > 4}
    by_len = sorted(refs, key=len, reverse=True)
    roles = {op: {} for op in refs}
    impls = {op: {} for op in refs}
    for k, f in fns.items():
        op = next((o for o in by_len if k.startswith(o + "_") and len(k) > len(o) + 1), None)
        if op is not None:
            suffix = k[len(op) + 1:]
            (roles if suffix in ROLES else impls)[op][suffix] = f
    return {op: Op(op, refs[op], impls[op], test=roles[op].get("test"),
                   check=roles[op].get("check"), dims=roles[op].get("dims")) for op in refs}
