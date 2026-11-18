"""pyhip.kit: an op is a family of plain functions that share a name prefix.

    ops = Registry()                                  # one per package (e.g. in _ops.py)
    op_register = ops.register

    @op_register
    def axpy_ref(x, y, *, alpha=1.0):                 # op "axpy": interface + reference
        return alpha * x + y

    @op_register
    @dim("n", (64, 1000))
    def axpy_problem(n, *, seed=0):                   # problem -> inputs; dims: n
        g = torch.Generator().manual_seed(seed)
        return dict(x=torch.randn(n, generator=g), y=torch.randn(n, generator=g))

    @op_register
    def axpy_dims(x, y, *, alpha=1.0):                # inputs -> the dims of a real call
        return dict(n=x.numel())

    @op_register                                      # outermost, above the kit marks
    @knob("block", (16, 64), id="b{}")
    def axpy_tiled(x, y, *, alpha=1.0, block=16):     # solver "tiled", knob block
        assert x.numel() % block == 0, f"n={x.numel()} % block {block}"  # a limit: raise
        ...

    ops.axpy.configs(dict(n=64))                      # [{"solver": "tiled", "block": 16}, ...]

Both sides of an op are declared the same way, by the parameters of a function and marks
giving their values. ``<op>_problem`` declares the problem: its parameters but ``seed``
(the random seed of the inputs, keyword-only) are the dims, ``@dim`` names values of
them, and a problem is a dict of dim values (``n`` in ``axpy_problem``). A solver
``<op>_<solver>`` declares one way to solve it: its keyword-only parameters are its
knobs, ``@knob`` gives their values, and a config is a solver with a value for each knob
(``block`` in ``axpy_tiled``). Dims answer "which problem?", knobs "how is this solver
set?".

``<op>_dims`` and ``@meta`` read the dims of a call or a problem; the tuned table is
keyed by dims. The knobs make the solver's configs, their names and what the dispatch
runs.

Limits are not declared: a solver raises (an ``assert`` is enough) when it
cannot run a call. The tuner runs every config and keeps those that ran and
passed ``<op>_check``; ``op(...)`` only picks those, so a call is guaranteed for a
tuned key only. ``@no_capture(reason)`` marks a solver that cannot run under
CUDA-graph capture: the dispatch skips it while the current stream captures.

Registration: ``Registry.register`` (``@op_register``) adds one function to its op and
returns it unchanged, after checking it against the conventions below:

* It is the outermost decorator: the kit marks below it are complete when it checks
  them, and a mark applied above it raises.
* The name gives the role. ``<op>_ref`` defines op ``<op>`` and is registered first.
  Any other ``<op>_<suffix>`` joins the registered op with the longest such prefix
  (``moe_dyn_ep3k`` belongs to ``moe_dyn`` once ``moe_dyn_ref`` is registered): suffix
  ``problem`` / ``check`` / ``dims`` is that role, any other suffix a solver.
  ``<op>_problem`` comes before the solvers (they are checked against its dims).
* One registry per package (a small module holding ``ops = Registry()``, imported by
  the modules that register): a package imported under two names keeps two registries.

Conventions:

* ``<op>_ref`` defines op ``<op>``: its parameters are the interface (named parameters
  only), its result the expected answer.
* ``<op>_<solver>`` is a solver, named by the suffix; the longest op prefix wins
  (``moe_dyn_ep3k`` belongs to ``moe_dyn`` when ``moe`` and ``moe_dyn`` both exist).
  Parameters: exactly the ref's, then keyword-only knobs with defaults, each with a
  ``@knob``, then optionally ``**over`` (free overrides, e.g. for experiments).
* ``<op>_problem(...)`` returns the inputs (the interface arguments, a dict) of one
  problem. Its parameter names but ``seed`` are the op's dims, the names ``meta`` /
  ``dims`` and the tuned table speak; its ``@dim`` marks name values of dims
  (``op.problems`` builds problems from them). ``seed``, when present, is keyword-only
  with a default.
* ``<op>_check(out, expected, **inputs)`` raises when ``out`` is wrong; it may return a
  dict of error metrics. Default: ``torch.testing.assert_close``. The first two
  arguments come by position: name them apart from the interface's parameters.
* ``<op>_dims(...)`` takes the ref's parameters and returns dim values.
  ``op(...)`` calls it on every call (``config=`` too): the place to check the
  arguments.
* Suffixes ``ref``, ``problem``, ``check``, ``dims`` are reserved; a solver name
  does not end in ``_problem`` / ``_check`` / ``_dims`` either. Names starting with
  ``_`` are not registered.

A config is a dict ``{"solver": name, knob: value, ...}``. Its name joins the solver name
and the ``@knob`` ids with spaces (``"tiled b64"``), overrides appended as
``":k=v,k=v"``.

Dispatch: ``Registry(name, table=path)`` names the tuned table of its ops (written by
``kit.tune``). ``op(*args, **kwargs)`` reads the dims of the call (``<op>_dims``, plus
keyword arguments that are dim names: hints such as ``routing=``), picks a config
(``op.pick``) and runs it; ``config=`` forces one, ``table=`` another table (None:
none). ``op.pick(problem, table=, capturing=)``:

* The table is a CSV. Its columns that are dims are the key, matched exactly; the
  bucket column, marked ``~<dim>`` (``~num_tokens``), is matched inexactly: the first
  tuned point >= the call's value (the largest point beyond). ``config`` holds a config
  name, ``us`` its time; other columns are ignored. Rows of other ops' solvers are
  skipped. A key dim that is None in ``problem`` matches the ``<op>_problem`` default.
* The rows of the key and the point of ``problem`` are ``op.tuned(problem)``, fastest
  first. ``pick`` returns the first, skipping ``@no_capture`` solvers when ``capturing``;
  no rows: ValueError (tune the key or pass ``config=``).
* Memoized per (table, its mtime, dim values, capturing): the returned dict is shared,
  do not modify it; ``op(...)`` memoizes its pick per (table, its mtime, capturing, the
  ``<op>_dims`` values and hints of the call). The mtime is read at most once a second
  per table: an edited table is used within a second.

The decorators only record themselves in ``f.kit_marks`` (in written order) and return
``f`` unchanged. Tests: ``pyhip.kit.testing.op_tests(op, problems)``; tuning:
``pyhip.kit.tune.Tuning``.
"""

import functools
import inspect
import itertools
import operator
import os
import re
import sys
import time
import types
from collections.abc import Mapping

from . import table as _table

__all__ = ["dim", "knob", "no_capture", "meta", "stack", "Registry", "Op"]

ROLES = ("ref", "problem", "check", "dims")
_FIELD = re.compile(r"\{[^{}]*\}")
_NAMED = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
_DEFAULT = object()  # table=: the registry's
_MEMO_MAX = 4096


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
    """ns -> tuple of ns[name] for ``names`` (C-level for 2+ names: memo keys are built
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


def _cell(s, default):
    """A table cell, typed like the ``<op>_problem`` default (int / float / str without)."""
    try:
        return _convert(s, default)
    except (TypeError, ValueError):
        return _guess(s)


_RECHECK_S = 1.0  # a table's mtime is read again after this long (os.stat: ~1 us)
_seen = {}  # path -> (monotonic time read, mtime or None)


def _version(path):
    """Cache key of a table file: its mtime (None: no table), read at most every
    ``_RECHECK_S`` seconds."""
    if not path:
        return None
    now = time.monotonic()
    got = _seen.get(path)
    if got is not None and now - got[0] < _RECHECK_S:
        return got[1]
    try:
        version = os.stat(path).st_mtime
    except OSError:
        version = None
    if len(_seen) >= 64:
        _seen.clear()
    _seen[path] = (now, version)
    return version


@functools.lru_cache(maxsize=16)
def _read_table(path, version):
    """(header, rows) of a table file; ``version`` (its mtime) only keys the cache."""
    del version
    return _table.read(path)


_cuda = None  # torch.cuda once CUDA is initialized


def _capturing():
    """Whether the current CUDA stream is capturing a graph (False without torch or
    before CUDA is initialized: nothing can capture then)."""
    global _cuda
    if _cuda is None:
        torch = sys.modules.get("torch")
        if torch is None or not torch.cuda.is_initialized():
            return False
        _cuda = torch.cuda
    return _cuda.is_current_stream_capturing()


class _Axis(_Mark):
    """One axis of a grid (``dim`` / ``knob``); stacked axes form their Cartesian
    product, the first written outermost.

        @knob("bm", (32, 64), id="bm{}")              # a sequence of values
        @dim("act", ("silu", "gelu"), id="{}")
        @knob("fp8", {"bf16": False, "fp8": True})    # a dict: the keys are the ids
        @dim("tokens, valid", {"v0": (512, 0), "v37": (512, 37)})   # several names
        @knob("split", lambda n: (1, 2, 4))           # a function of dims (knobs only)

    ``id`` is a format string with one ``{}`` field (default ``name + "{}"``): the
    axis' part of config names and test ids. Config names must read back, so a
    value's text holds no space or ":".
    """

    def __init__(self, names, values, id=None):
        what = f"{type(self).__name__} {names!r}"
        self.names = tuple(n.strip() for n in names.split(","))
        if not all(n.isidentifier() for n in self.names) or len(set(self.names)) < len(self.names):
            raise TypeError(f"{type(self).__name__}: bad names {names!r}")
        self.fmt = self.table = self.values = None
        if isinstance(values, str):
            raise TypeError(f"{what}: values {values!r} is a string, not a sequence")
        if isinstance(values, dict):
            if id is not None:
                raise TypeError(f"{what}: a dict's keys are the ids; drop id=")
            self.table = {}
            for k, v in values.items():
                v = tuple(v) if len(self.names) > 1 else (v,)
                if len(v) != len(self.names):
                    raise TypeError(f"{what}: {k!r} has {len(v)} values")
                self.table[str(k)] = dict(zip(self.names, v))
        elif len(self.names) > 1:
            raise TypeError(f"{what}: several names take a dict {{id: values}}")
        else:
            self.fmt = self.names[0] + "{}" if id is None else id
            if len(_FIELD.findall(self.fmt)) != 1:
                raise TypeError(f"{what}: id {self.fmt!r} needs exactly one {{}} field")
            self.values = _Call(values) if callable(values) else tuple(values)

    def __repr__(self):
        return f"{type(self).__name__}({', '.join(self.names)!r})"

    def points(self, problem=None):
        """[(id, {name: value})] of the axis (``problem`` feeds a values function)."""
        if self.table is not None:
            return [(k, dict(d)) for k, d in self.table.items()]
        vals = self.values
        if isinstance(vals, _Call):
            vals = vals([problem[n] for n in vals.names], {})
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


class dim(_Axis):
    """Values of problem dims. On ``<op>_problem``: named values of a dim (``op.problems``
    picks from them by the axis name); in ``op.problems``: an ad hoc axis.
    ``@dim("act", ("silu", "gelu"), id="{}")``; several dims at once take a dict, and on
    ``<op>_problem`` a ``name``: ``@dim("m, n", {"sq": (64, 64)}, name="shape")``. A
    sequence or a dict of values (see ``_Axis``)."""

    def __init__(self, names, values, id=None, name=None):
        if callable(values):
            raise TypeError(f"dim {names!r}: values must be a sequence or a dict")
        super().__init__(names, values, id)
        if name is not None and not str(name).isidentifier():
            raise TypeError(f"dim {names!r}: bad name {name!r}")
        self.name = name if name is not None or len(self.names) > 1 else self.names[0]


class knob(_Axis):
    """Values of a solver knob, on ``<op>_<solver>``: ``@knob("block", (16, 64),
    id="b{}")``. A sequence, a dict, or a function of dims (see ``_Axis``)."""


class no_capture(_Mark):
    """The solver cannot run under CUDA-graph capture (e.g. it passes a host
    counter by launch argument): ``op(...)`` / ``pick`` skip it while the current
    stream captures. ``@no_capture("host launch epoch")``."""

    def __init__(self, reason):
        self.reason = reason

    def __repr__(self):
        return f"no_capture({self.reason!r})"


class meta(_Mark):
    """Metadata for tools, read with ``Op.meta``: constants, or functions of dims and
    knobs. ``@meta(tile_m=64, cached=lambda bcm: bcm != 2)``."""

    def __init__(self, **items):
        self.items = {k: _Call(v) if callable(v) else v for k, v in items.items()}


class stack(_Mark):
    """Several marks as one, e.g. a configuration space shared by solvers:
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


class _Solver:
    """A solver as its op sees it."""

    __slots__ = ("name", "f", "params", "no_capture", "meta", "defaults", "knobs", "varkw",
                 "regex")

    def __init__(self, op, name, f, ref_params, dims):
        where = f"{op}_{name}"
        if not name.isidentifier() or name in ROLES:
            raise TypeError(f"{where}: bad solver name {name!r}")
        if _marks(f, dim):
            raise TypeError(f"{where}: @dim describes the problem; it goes on {op}_problem "
                            "(a solver's options are @knob)")
        ps = list(inspect.signature(f).parameters.values())
        if ([(p.name, p.kind, p.default) for p in ps[:len(ref_params)]]
                != [(p.name, p.kind, p.default) for p in ref_params]):
            raise TypeError(f"{_sig(op, name, f)}: the parameters must start with exactly "
                            f"those of {op}_ref{inspect.Signature(ref_params)}")
        self.name, self.f, self.defaults, self.varkw = name, f, {}, False
        for p in ps[len(ref_params):]:
            if p.kind is p.VAR_KEYWORD:
                self.varkw = True
            elif p.kind is p.KEYWORD_ONLY and p.default is not p.empty:
                self.defaults[p.name] = p.default
            else:
                raise TypeError(f"{where}: parameter '{p}' after the {op}_ref ones must be a "
                                "keyword-only knob with a default, or **over")
        knobs = list(self.defaults)
        clash = [k for k in knobs if k in dims or k == "solver"]
        if clash:
            raise TypeError(f"{where}: knob(s) {clash} clash with the dims of {op} "
                            f"{list(dims)} or with 'solver'")
        self.params = tuple(_marks(f, knob))
        names = [n for p in self.params for n in p.names]
        if len(set(names)) < len(names) or set(names) != set(knobs):
            raise TypeError(f"{where}: each knob needs exactly one @knob; knobs {knobs}, "
                            f"@knob names {names}")
        self.knobs = frozenset(knobs)
        caps = _marks(f, no_capture)
        self.no_capture = caps[0].reason if caps else None
        self.meta = {}
        for m in _marks(f, meta):
            self.meta.update(m.items)
        known = (*dims, *knobs)
        for k, v in self.meta.items():
            bad = _unknown(v.names, known) if isinstance(v, _Call) else ()
            if bad:
                raise TypeError(f"{where}: meta {k!r} uses unknown name(s) {bad}; known: "
                                f"{list(known)}")
        for p in self.params:
            bad = _unknown(p.values.names, dims) if isinstance(p.values, _Call) else ()
            if bad:
                raise TypeError(f"{where}: the values of {p!r} use name(s) {bad} outside the "
                                f"dims {list(dims)}")
        rx = re.escape(name) + "".join(" " + p.pattern(f"_p{i}") for i, p in enumerate(self.params))
        self.regex = re.compile(rx + ("(?::(?P<_over>.+))?" if self.varkw else "") + r"\Z")


class Op:
    """One op: its functions and the queries tools need. Built by ``Registry.register``
    one function at a time; read-only otherwise.

    Attributes: ``name``, ``ref``, ``problem``, ``check``, ``dims`` (None when absent),
    ``solvers`` ({name: function}), ``dim_names`` (the ``<op>_problem`` parameter names but
    ``seed``). A ``problem`` argument is a dict {dim: value} holding every dim name (None:
    unknown).
    """

    __slots__ = ("name", "ref", "problem", "check", "dims", "solvers", "dim_names", "_sol",
                 "_fns", "_axes", "_defaults", "_seeded", "_ref_params", "_ref_names",
                 "_registry", "_dget", "_blank", "_tables", "_memo", "_calls")

    def __init__(self, name, ref, registry=None):
        ref_params = list(inspect.signature(ref).parameters.values())
        if not all(p.kind in _NAMED for p in ref_params):
            raise TypeError(f"{_sig(name, 'ref', ref)}: the interface takes named parameters "
                            "only (no '/', *args, **kwargs)")
        if _marks(ref):
            raise TypeError(f"{name}_ref: only solvers and {name}_problem take kit marks")
        taken = [p.name for p in ref_params if p.name in ("config", "table")]
        if taken:
            raise TypeError(f"{_sig(name, 'ref', ref)}: {taken} are reserved (op(...) takes "
                            "config= and table=)")
        fns = {}
        for k, v in dict(name=name, ref=ref, problem=None, check=None, dims=None,
                         solvers=types.MappingProxyType(fns), dim_names=(), _sol={}, _fns=fns,
                         _axes={}, _defaults={}, _seeded=False, _ref_params=ref_params,
                         _ref_names=frozenset(p.name for p in ref_params),
                         _registry=registry if registry is not None else Registry(),
                         _dget=_getter(()), _blank={}, _tables={}, _memo={},
                         _calls={}).items():
            object.__setattr__(self, k, v)

    def _add(self, suffix, f):
        """Add ``f`` as role ``suffix`` (problem / check / dims) or as solver
        ``suffix``; the op is unchanged when ``f`` breaks a convention."""
        if suffix in ROLES and getattr(self, suffix) is not None:
            raise TypeError(f"{self.name}_{suffix} is already registered")
        if suffix == "problem":
            self._add_problem(f)
        elif suffix in ROLES:
            self._add_helper(suffix, f)
        else:
            self._add_solver(suffix, f)

    def _add_problem(self, f):
        name = self.name
        if self._sol:
            raise TypeError(f"{name}_problem: register it before the solvers "
                            f"{list(self._sol)} (they use its dims)")
        tps = list(inspect.signature(f).parameters.values())
        if not all(p.kind in _NAMED for p in tps):
            raise TypeError(f"{_sig(name, 'problem', f)}: named parameters only (their "
                            "names are the dims)")
        seed = next((p for p in tps if p.name == "seed"), None)
        if seed is not None and (seed.kind is not seed.KEYWORD_ONLY or seed.default is seed.empty):
            raise TypeError(f"{_sig(name, 'problem', f)}: seed (the random seed of the inputs, "
                            "not a dim) must be keyword-only with a default: *, seed=0")
        if any(not isinstance(m, dim) for m in _marks(f)):
            raise TypeError(f"{name}_problem: only @dim marks (a solver's options are @knob)")
        tps = [p for p in tps if p is not seed]
        dims = tuple(p.name for p in tps)
        axes = {}
        for p in _marks(f, dim):
            if p.name is None:
                raise TypeError(f"{name}_problem: {p!r} needs name= (problems pick its values "
                                "by the axis name)")
            if p.name in axes:
                raise TypeError(f"{name}_problem: two @dim axes named {p.name!r}")
            axes[p.name] = p
        names = [n for p in axes.values() for n in p.names]
        if _unknown(names, dims) or len(set(names)) < len(names):
            raise TypeError(f"{name}_problem: the @dim names {names} must be distinct dims of "
                            f"{_sig(name, 'problem', f)}")
        defaults = {p.name: p.default for p in tps if p.default is not p.empty}
        for k, v in dict(problem=f, dim_names=dims, _axes=types.MappingProxyType(axes),
                         _defaults=defaults, _seeded=seed is not None,
                         _dget=_getter(dims), _blank=dict.fromkeys(dims)).items():
            object.__setattr__(self, k, v)

    def _add_helper(self, role, f):
        """``<op>_check`` / ``<op>_dims``: no marks; they take the ref's arguments."""
        if _marks(f):
            raise TypeError(f"{self.name}_{role}: only solvers and {self.name}_problem "
                            "take kit marks")
        inputs = {p.name: None for p in self._ref_params}
        try:
            inspect.signature(f).bind(*((None, None) if role == "check" else ()), **inputs)
        except TypeError as e:
            raise TypeError(f"{_sig(self.name, role, f)} does not take the arguments of "
                            f"{_sig(self.name, 'ref', self.ref)}: {e}") from None
        object.__setattr__(self, role, f)

    def _add_solver(self, name, f):
        if name in self._sol:
            raise TypeError(f"{self.name}_{name} is already registered")
        s = _Solver(self.name, name, f, self._ref_params, self.dim_names)
        self._sol[name], self._fns[name] = s, f
        try:
            self._check_names(s)
        except Exception:
            del self._sol[name], self._fns[name]
            raise
        self._tables.clear()  # the table rows of this solver count from now on
        self._memo.clear()
        self._calls.clear()

    def __setattr__(self, k, v):
        raise AttributeError(f"Op is read-only ({k})")

    registry = property(lambda self: self._registry)

    def __repr__(self):
        return f"Op({self.name!r}, solvers={list(self._fns)})"

    def _check_names(self, s):
        """Every knob value prints into a config name that reads back to it."""
        static = [p.points() for p in s.params if not isinstance(p.values, _Call)]
        base = {"solver": s.name}
        for pts in static:
            base.update(pts[0][1] if pts else {})
        for pts in static:
            for _, vals in pts:
                name = self.name_of({**base, **vals})
                try:
                    back = self.parse(name)
                except ValueError as e:
                    raise TypeError(f"{self.name}_{s.name}: {e}") from None
                if any(back[k] != v for k, v in vals.items()):
                    raise TypeError(f"{self.name}_{s.name}: {vals} prints as {name!r}, which "
                                    f"reads back as {back}")

    def problems(self, *axes):
        """[(id, problem)]: the Cartesian product of ``axes``, the first outermost. An axis
        is the name of a ``@dim`` of ``<op>_problem`` (all its values), (name, [ids]) (those
        values, in that order) or a ``dim(...)`` (ad hoc). A problem holds every dim: the
        axes' values, else the ``<op>_problem`` default, else None. The id joins the axes'
        ids with "-"."""
        pts, seen = [], []
        for a in axes:
            if isinstance(a, dim):
                p = a
                pts.append(p.points())
            else:
                key, ids = (a, None) if isinstance(a, str) else a
                p = self._axes.get(key)
                if p is None:
                    raise KeyError(f"{self.name}: no @dim {key!r} on {self.name}_problem; "
                                   f"axes: {list(self._axes)}")
                got = dict(p.points())
                if ids is not None:
                    ids = [str(i) for i in ([ids] if isinstance(ids, (str, int)) else ids)]
                    bad = [i for i in ids if i not in got]
                    if bad:
                        raise KeyError(f"{self.name}: {key!r} has no value {bad}; ids: "
                                       f"{list(got)}")
                pts.append([(i, got[i]) for i in (ids if ids is not None else got)])
            seen += p.names
        bad = _unknown(seen, self.dim_names)
        if bad or len(set(seen)) < len(seen):
            raise TypeError(f"{self.name}: the axes set {seen}: each a dim of "
                            f"{self.name}_problem {list(self.dim_names)} at most once")
        out = []
        for combo in itertools.product(*pts):
            prob = dict(self._defaults)
            for _, vals in combo:
                prob.update(vals)
            out.append(("-".join(tok for tok, _ in combo), {**self._blank, **prob}))
        return out

    def _split(self, cfg):
        """cfg -> (solver, knobs with defaults filled in, overrides)."""
        knobs = dict(cfg)
        name = knobs.pop("solver", None)
        s = self._sol.get(name)
        if s is None:
            raise KeyError(f"{self.name}: no solver {name!r}; have {list(self._sol)}")
        over = {k: v for k, v in knobs.items() if k not in s.knobs}
        if over and not s.varkw:
            raise TypeError(f"{self.name}_{name}: unknown knob(s) {sorted(over)}")
        return s, {**s.defaults, **{k: v for k, v in knobs.items() if k in s.knobs}}, over

    def capturable(self, cfg):
        """Whether ``cfg``'s solver may run under CUDA-graph capture (no
        ``@no_capture``)."""
        return self._split(cfg)[0].no_capture is None

    def configs(self, problem):
        """Every config of every solver (each knob at each of its ``@knob`` values), in
        definition order (``problem`` feeds the ``@knob`` values functions). Whether one
        runs on ``problem``: run it."""
        out = []
        for s in self._sol.values():
            try:
                axes = [p.points(problem) for p in s.params]
            except KeyError as e:
                raise TypeError(f"{self.name}_{s.name}: the problem passed lacks "
                                f"{e}") from None
            for combo in itertools.product(*axes):
                cfg = dict(solver=s.name)
                for _, vals in combo:
                    cfg.update(vals)
                out.append(cfg)
        return out

    def name_of(self, cfg):
        s, knobs, over = self._split(cfg)
        text = " ".join([s.name] + [p.token(knobs) for p in s.params])
        return text + (":" + ",".join(f"{k}={v}" for k, v in over.items()) if over else "")

    def parse(self, name):
        """The inverse of ``name_of`` (override values come back as int, float or str)."""
        solver = re.match(r"[^\s:]*", name)[0]
        s = self._sol.get(solver)
        m = s.regex.match(name) if s else None
        if m is None:
            raise ValueError(f"{self.name}: cannot parse config name {name!r}"
                             + (f" (pattern {s.regex.pattern!r})" if s else
                                f"; solvers: {list(self._sol)}"))
        cfg = dict(solver=solver)
        for i, p in enumerate(s.params):
            text = m[f"_p{i}"]
            try:
                cfg.update(p.decode(text, s.defaults))
            except ValueError:
                raise ValueError(f"{self.name}: {name!r}: bad {p!r} value {text!r}") from None
        if s.varkw and m["_over"]:
            for kv in m["_over"].split(","):
                k, eq, v = kv.partition("=")
                if not eq or not k.isidentifier() or k in cfg:
                    raise ValueError(f"{self.name}: {name!r}: bad override {kv!r}")
                cfg[k] = _guess(v)
        return cfg

    def run(self, cfg, *args, **kwargs):
        """Call the solver of ``cfg`` with its knobs (and overrides)."""
        knobs = dict(cfg)
        return self._fns[knobs.pop("solver")](*args, **kwargs, **knobs)

    def meta(self, cfg, name, problem=None):
        """``@meta`` item ``name`` of ``cfg``'s solver; a function is evaluated on
        ``problem``, the knobs and the overrides."""
        s, knobs, over = self._split(cfg)
        try:
            v = s.meta[name]
        except KeyError:
            raise KeyError(f"{self.name}_{s.name}: no meta {name!r}; it has "
                           f"{sorted(s.meta)}") from None
        if not isinstance(v, _Call):
            return v
        ns = {**(problem or {}), **knobs}
        try:
            vals = [ns[n] for n in v.names]
        except KeyError:
            raise TypeError(f"{self.name}_{s.name}: meta {name!r} needs "
                            f"{_unknown(v.names, ns)}") from None
        return v(vals, over)

    # ---- dispatch -----------------------------------------------------------------

    def __call__(self, *args, config=None, table=_DEFAULT, **kwargs):
        """Run the op on the interface arguments: ``config`` (a dict or a name) if given,
        else ``pick`` on the dims of the call (``<op>_dims`` and the keyword arguments
        that are dim names, e.g. ``routing=``)."""
        hints = None
        if not kwargs.keys() <= self._ref_names:
            hints = {k: kwargs.pop(k) for k in [k for k in kwargs if k not in self._ref_names]}
            bad = _unknown(hints, self.dim_names)
            if bad:
                raise TypeError(f"{self.name}: unknown argument(s) {bad}: neither parameters "
                                f"of {self.name}_ref nor dims {list(self.dim_names)}")
        dims = self.dims
        if config is not None:
            if dims is not None:
                dims(*args, **kwargs)  # it checks the arguments
            if isinstance(config, str):
                config = self.parse(config)
            knobs = dict(config)
            return self._fns[knobs.pop("solver")](*args, **kwargs, **knobs)
        if dims is None:
            raise TypeError(f"{self.name}: picking a config needs {self.name}_dims; "
                            "pass config=")
        got = dims(*args, **kwargs)
        path = self._registry.table if table is _DEFAULT else table
        capturing = _capturing()
        # memo of the call: (solver, knobs) per dims of the call
        key = (path, _version(path), capturing, tuple(got.items()),
               tuple(hints.items()) if hints else None)
        hit = self._calls.get(key)
        if hit is None:
            knobs = dict(self.pick({**self._blank, **got, **(hints or {})}, table=path,
                                   capturing=capturing))
            if len(self._calls) >= _MEMO_MAX:
                self._calls.clear()
            hit = self._calls[key] = (self._fns[knobs.pop("solver")], knobs)
        fn, knobs = hit
        return fn(*args, **kwargs, **knobs)

    def pick(self, problem, *, table=_DEFAULT, capturing=False):
        """The config ``op(...)`` runs for ``problem`` (see the module docstring): the
        fastest of ``tuned(problem, table=)`` (``table``: a path, None for none; default the
        registry's) that may run in the capture state; ValueError when there is none."""
        path = self._registry.table if table is _DEFAULT else table
        version = _version(path)
        try:
            key = (path, version, bool(capturing), self._dget(problem))
        except KeyError:
            raise TypeError(f"{self.name}: the problem passed lacks "
                            f"{_unknown(self.dim_names, problem)}") from None
        cfg = self._memo.get(key)
        if cfg is None:
            if len(self._memo) >= _MEMO_MAX:
                self._memo.clear()
            cfg = self._memo[key] = self._pick(problem, path, version, bool(capturing))
        return cfg

    def points(self, problem, *, table=_DEFAULT):
        """[(bucket value, [(us, cfg), ...] fastest first)] of the table rows of
        the key of ``problem`` ([] without rows)."""
        path = self._registry.table if table is _DEFAULT else table
        version = _version(path)
        if version is None:
            return []
        cols, _, index = self._index(path, version)
        return index.get(self._key(cols, problem), [])

    def _key(self, cols, problem):
        d = self._defaults
        return tuple(d.get(c) if problem[c] is None else problem[c] for c in cols)

    def _index(self, path, version):
        """(key columns, bucket column or None, {key: [(bucket value, [(us, cfg), ...]
        fastest first), ...] by bucket value}) of the rows of ``path`` this op runs."""
        got = self._tables.get((path, version))
        if got is not None:
            return got
        header, rows = _read_table(path, version)
        if "config" not in header or "us" not in header:
            raise ValueError(f"{path}: a tuned table has the columns config and us; this one "
                             f"has {list(header)}")
        d = self._defaults
        bucket = _table.bucket(header)
        bucket = bucket if bucket in self.dim_names else None
        bcol = _table.BUCKET + bucket if bucket else None
        cols = tuple(n for n in self.dim_names if n in header)
        pts = {}
        for i, r in enumerate(rows, 2):
            name = r["config"]
            if re.match(r"[^\s:]*", name)[0] not in self._sol:
                continue  # another op's solver
            try:
                cfg = self.parse(name)
            except ValueError as e:
                raise ValueError(f"{path}:{i}: {e}") from None
            key = tuple(_cell(r[c], d.get(c)) for c in cols)
            b = _cell(r[bcol], d.get(bucket)) if bucket else None
            pts.setdefault(key, {}).setdefault(b, []).append((float(r["us"]), cfg))
        index = {k: sorted((b, sorted(c, key=lambda t: t[0])) for b, c in v.items())
                 for k, v in pts.items()}
        if len(self._tables) >= 8:
            self._tables.clear()
        got = self._tables[path, version] = (cols, bucket, index)
        return got

    def _table_configs(self, problem, path, version):
        """The configs of the tuned point for ``problem``, fastest first."""
        if version is None:
            return []
        cols, bucket, index = self._index(path, version)
        pts = index.get(self._key(cols, problem))
        if not pts:
            return []
        if bucket is not None:
            n = problem[bucket]
            if n is None:
                raise ValueError(f"{self.name}: picking from {path} needs {bucket}")
            pts = [next((p for p in pts if p[0] >= n), pts[-1])]
        return [cfg for _, cfg in pts[0][1]]

    def tuned(self, problem, *, table=_DEFAULT):
        """The configs of the tuned point for ``problem`` (``pick``'s choices), fastest
        first; [] when the table has no rows for its key."""
        path = self._registry.table if table is _DEFAULT else table
        return list(self._table_configs(problem, path, _version(path)))

    def _pick(self, problem, path, version, capturing):
        cfgs = self._table_configs(problem, path, version)
        for cfg in cfgs:
            if not capturing or self._sol[cfg["solver"]].no_capture is None:
                return cfg
        known = {k: v for k, v in problem.items() if v is not None}
        if not cfgs:
            raise ValueError(f"{self.name}: no tuned config for {known} "
                             + (f"in {path}" if path else "(no table)")
                             + ": tune it, or pass config=")
        why = "; ".join(f"{self.name_of(c)}: {self._sol[c['solver']].no_capture}" for c in cfgs)
        raise ValueError(f"{self.name}: no tuned config for {known} can run under CUDA-graph "
                         f"capture: {why}")


class Registry(Mapping):
    """The ops of a package, built one function at a time by ``register`` (used as the
    decorator ``@op_register``, see the module docstring). Read it as a mapping {op
    name: Op} in registration order, or by attribute: ``ops.moe is ops["moe"]``. An
    ``Op`` is the same object from its ref's registration on; later registrations add
    to it.

    ``name``: for reports. ``table``: the tuned table the ops dispatch from (see the
    module docstring). Tuning is declared apart: ``kit.tune.Tuning``."""

    def __init__(self, name=None, *, table=None):
        self._ops = {}
        self._name, self._table = name, table

    name = property(lambda self: self._name)
    table = property(lambda self: self._table)

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
            early = [k for k in o.solvers if pre and k.startswith(pre)]
            if early:
                raise TypeError(f"{op}_ref: {o.name}_{early[0]} was registered before it and "
                                f"went to op {o.name!r}; register {op}_ref first")
        self._ops[op] = Op(op, ref, self)

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
                            f"{op}_{head}_ref first (solver names do not end in "
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
        return f"Registry({'' if self._name is None else repr(self._name) + ', '}{list(self._ops)})"
