"""CPU tests of pyhip.kit (no GPU): registration, grids, config names, dispatch, the
checks a registration runs on the conventions, and the pytest tests ``kit.testing``
generates."""

import os
import textwrap

import pytest

from pyhip import kit
from pyhip.kit import Registry, meta, no_capture, param, stack
from pyhip.kit.testing import assert_close

pytest_plugins = ["pytester"]

TOY = '''
import random
from pyhip.kit import Registry, meta, no_capture, param, stack

ops = Registry()
op_register = ops.register


@op_register
def axpy_ref(x, y, *, alpha=1.0):
    return [alpha * a + b for a, b in zip(x, y)]


@op_register
@param("n", (4, 6), id="n{}")
@param("alpha", {"a1": 1.0, "a2": 2.0})
def axpy_test(n, alpha, seed=0):
    rng = random.Random(seed + n)
    return dict(x=[rng.random() for _ in range(n)], y=[rng.random() for _ in range(n)],
                alpha=alpha)


@op_register
def axpy_dims(x, y, *, alpha=1.0):
    return dict(n=len(x), alpha=alpha)


@op_register
def axpy_check(out, expected, **inputs):
    assert len(out) == len(expected), (len(out), len(expected))
    err = max((abs(a - b) for a, b in zip(out, expected)), default=0.0)
    assert err < 1e-12, err
    return {"err": err}


@op_register
def axpy_loop(x, y, *, alpha=1.0):
    return [alpha * a + b for a, b in zip(x, y)]


tiles = stack(param("block", (2, 4), id="b{}"))


@op_register
@tiles
@meta(rows=lambda block: block * 8, graph_safe=True)
def axpy_tiled(x, y, *, alpha=1.0, block=2, **over):
    assert len(x) % block == 0, f"n={len(x)} % block {block}"  # a limit: the impl raises
    out = []
    for i in range(0, len(x), block):
        out += [alpha * a + b for a, b in zip(x[i:i + block], y[i:i + block])]
    return out


@op_register
@param("mode", ("fast", "slow"), id="{}")
@no_capture("not graph-safe")
@meta(rows=None, graph_safe=False)
def axpy_queue(x, y, *, alpha=1.0, mode="fast"):
    return [alpha * a + b for a, b in zip(x, y)]


@op_register
def axpy_dyn_ref(x, y, valid, *, alpha=1.0):
    return [alpha * a + b if i < valid else 0.0 for i, (a, b) in enumerate(zip(x, y))]


@op_register
def axpy_dyn_loop(x, y, valid, *, alpha=1.0):
    return axpy_dyn_ref(x, y, valid, alpha=alpha)

'''


def _toy():
    ns = {"__name__": "toy"}
    exec(compile(TOY, "toy.py", "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def toy():
    return _toy()


def _ops(src, **registry):
    """A fresh Registry(**registry) holding the top-level functions of ``src``, each
    registered as it is defined (``@op_register`` inserted above its decorators)."""
    lines, out = textwrap.dedent(src).splitlines(), []
    for i, line in enumerate(lines):
        if line.startswith(("@", "def ")) and not (i and lines[i - 1].startswith("@")):
            out.append("@op_register")
        out.append(line)
    reg = Registry(**registry)
    exec("\n".join(out), {"op_register": reg.register, "param": param,
                          "no_capture": no_capture, "meta": meta, "stack": stack}, {})
    return reg


def _raises(src, match):
    with pytest.raises(TypeError, match=match):
        _ops(src)


def test_registration(toy):
    ops = toy["ops"]
    assert list(ops) == ["axpy", "axpy_dyn"] and len(ops) == 2 and "axpy" in ops
    op = ops["axpy"]
    assert ops.axpy is op
    assert (op.ref, op.test, op.check, op.dims) == (toy["axpy_ref"], toy["axpy_test"],
                                                     toy["axpy_check"], toy["axpy_dims"])
    assert list(op.impls) == ["loop", "tiled", "queue"]  # axpy_dyn_* go to the longer prefix
    assert op.vocab == ("n", "alpha", "seed")
    dyn = ops["axpy_dyn"]
    assert list(dyn.impls) == ["loop"] and dyn.test is None and dyn.vocab == ()
    assert (toy["axpy_tiled"].kit_op, toy["axpy_dyn_loop"].kit_op) == ("axpy", "axpy_dyn")
    assert repr(op) == "Op('axpy', impls=['loop', 'tiled', 'queue'])"
    assert repr(ops) == "Registry(['axpy', 'axpy_dyn'])"
    with pytest.raises(AttributeError):
        op.name = "x"
    with pytest.raises(AttributeError, match="no op 'nope'; registered: "):
        ops.nope  # noqa: B018
    with pytest.raises(AttributeError):
        ops._nope  # noqa: B018
    other = _toy()["ops"]  # a package imported twice: two independent registries
    assert other["axpy"] is not op and list(other["axpy"].impls) == list(op.impls)


def test_registration_rules():
    ref = "def f_ref(x, *, a=1): pass\n"
    _raises("def f_k(x): pass\n" + ref, r"register its <op>_ref first")
    _raises(ref + "def f_k(x, *, a=1): pass\ndef f_test(n=1): pass",
            "register it before the implementations")
    _raises(ref + "def f_dyn_test(n=1): pass", "register f_dyn_ref first")
    _raises(ref + "def f_dyn_k(x, *, a=1): pass\ndef f_dyn_ref(y): pass",
            "f_dyn_k was registered before it")
    _raises(ref + ref, "op 'f' is already registered")
    _raises(ref + "def f_dims(x, *, a=1): pass\n" * 2, "f_dims is already registered")
    _raises(ref + "def f_k(x, *, a=1): pass\n" * 2, "f_k is already registered")
    _raises("def items_ref(x): pass", r"taken by Registry.items")
    _raises("def _f_ref(x): pass", "private")
    reg = Registry()
    for bad in (int, len, param("a", (1,)), lambda x: x):
        with pytest.raises(TypeError):
            reg.register(bad)

    @reg.register
    def g_ref(x):
        pass

    with pytest.raises(TypeError, match="g_ref is already registered"):
        reg.register(g_ref)
    with pytest.raises(TypeError, match="outermost decorator, above meta"):
        @meta(k=1)
        @reg.register
        def g_k(x):
            pass
    with pytest.raises(TypeError, match="outermost"):
        stack(param("b", (1,)))(g_ref)

    def g_bad(x, *, b=1):  # a knob without its @param
        pass

    @param("b", ("x y", "z"), id="{}")
    def g_odd(x, *, b="z"):  # a value that cannot appear in a config name
        pass

    for f, match in ((g_bad, "each knob needs exactly one @param"), (g_odd, "cannot parse")):
        with pytest.raises(TypeError, match=match):
            reg.register(f)
        assert "kit_op" not in f.__dict__ and list(reg.g.impls) == ["k"]  # left no trace
    _raises("def f_ref(x, *, config=None): pass", r"\['config'\] are reserved")
    with pytest.raises(TypeError, match="relax is a dict"):
        Registry(relax=["act"])
    assert repr(Registry("lib")) == "Registry('lib', [])"


DISPATCH = '''
def f_ref(x, *, dt="a"):
    return ("ref",)

@param("n", (4,))
def f_test(n, dt="a", mode="s", seed=0):
    return dict(x=[0.0] * n, dt=dt)

def f_dims(x, *, dt="a"):
    return dict(n=len(x), dt=dt)

@param("b", (1, 2), id="b{}")
def f_k1(x, *, dt="a", b=1):
    return ("k1", b)

@no_capture("not graph-safe")
def f_k2(x, *, dt="a"):
    return ("k2",)

def f_k3(x, *, dt="a"):
    return ("k3",)
'''

TABLE = """dt,mode,n,config,us,err
a,s,4,k2,2.0,0
a,s,4,k1 b2,1.0,0
a,s,4,k3,3.0,0
a,s,4,zz,0.5,0
a,s,16,k1 b1,2.0,0
a,s,16,k3,1.0,0
a,t,4,k3,1.0,0
b,s,4,k2,1.0,0
c,s,4,k2,1.0,0
c,s,4,k3,2.0,0
"""


def test_dispatch(tmp_path):
    path = tmp_path / "tuned.csv"
    path.write_text(TABLE)
    op = _ops(DISPATCH, table=str(path), bucket="n", relax={"mode": "s"}).f

    def pick(capturing=False, table=str(path), **v):
        return op.name_of(op.pick(dict(op._blank, **v), table=table, capturing=capturing))

    assert pick(n=4, dt="a") == "k1 b2"  # the fastest; mode None matches the default "s"
    assert pick(n=3, dt="a") == "k1 b2"  # point 4
    assert pick(n=4, dt="c") == "k2" and pick(n=4, dt="c", capturing=True) == "k3"
    with pytest.raises(ValueError, match=r"can run under CUDA-graph capture: k2: not graph"):
        pick(n=4, dt="b", capturing=True)  # only k2 is tuned there
    assert pick(n=5, dt="a") == "k3" and pick(n=100, dt="a") == "k3"  # point 16, the largest
    assert pick(n=4, dt="a", mode="t") == "k3"
    assert pick(n=4, dt="a", mode="u") == "k1 b2"  # no rows: relax mode -> s
    assert pick(n=4, dt="b") == "k2"
    with pytest.raises(ValueError, match=r"no tuned config for \{'n': 4, 'dt': 'd'\} in .*"
                                         r"tuned.csv: tune it, or pass config="):
        pick(n=4, dt="d")  # no rows, none after relax
    with pytest.raises(ValueError, match=r"\(no table\)"):
        pick(n=4, dt="a", table=None)
    with pytest.raises(ValueError, match="no tuned config"):
        pick(n=4, dt="a", table=str(tmp_path / "none.csv"))
    vocab = dict(op._blank, n=4, dt="a")
    assert op.pick(vocab) is op.pick(dict(vocab))  # memoized
    assert [op.name_of(c) for c in op.tuned(dict(vocab, n=3))] == ["k1 b2", "k2", "k3"]
    assert op.tuned(dict(vocab, dt="d")) == [] and op.tuned(vocab, table=None) == []
    assert [(n, [op.name_of(c) for _, c in cs]) for n, cs in op.points(vocab)] == [
        (4, ["k1 b2", "k2", "k3"]), (16, ["k3", "k1 b1"])]  # zz: not an implementation
    with pytest.raises(ValueError, match="needs n"):
        op.pick(dict(op._blank, dt="a"))
    with pytest.raises(TypeError, match="lacks"):
        op.pick(dict(n=4))
    # a call: dims and hints -> pick -> run
    x3, x4 = [0.0] * 3, [0.0] * 4
    assert op(x4) == ("k1", 2) and op(x4, mode="t") == ("k3",) and op(x3) == ("k1", 2)
    assert op(x3, config="k3") == ("k3",) and op(x3, config=dict(kind="k1", b=1)) == ("k1", 1)
    with pytest.raises(ValueError, match=r"\(no table\)"):
        op(x4, table=None)
    with pytest.raises(TypeError, match=r"unknown argument\(s\) \['bogus'\]"):
        op(x4, bogus=1)
    with pytest.raises(TypeError, match="len"):  # dims runs with config= too: it checks
        op(None, config="k3")
    with pytest.raises(TypeError, match="needs g_dims"):
        _ops("def g_ref(x): pass\ndef g_k(x): pass").g([1])
    # the table file changes: it is read again (its mtime is read at most once a second)
    path.write_text(TABLE.replace("k1 b2,1.0", "k1 b2,9.0"))
    os.utime(path, (1, 1))
    assert pick(n=4, dt="a") == "k1 b2"
    kit._seen[str(path)] = (0.0, kit._seen[str(path)][1])  # a second later
    assert pick(n=4, dt="a") == "k2"
    assert op(x4) == ("k2",)  # the memo of calls too
    # a table without the mode column: mode is no key
    old = tmp_path / "old.csv"
    old.write_text("dt,n,config,us\na,4,k3,1.0\n")
    assert pick(n=4, dt="a", mode="t", table=str(old)) == "k3"
    bad = tmp_path / "bad.csv"
    bad.write_text("dt,n,config,us\na,4,k1 bx,1.0\n")
    with pytest.raises(ValueError, match=r"bad.csv:2: .*'k1 bx'"):
        pick(n=4, dt="a", table=str(bad))
    nocols = tmp_path / "nocols.csv"
    nocols.write_text("dt,n\na,4\n")
    with pytest.raises(ValueError, match="config and us"):
        pick(n=4, dt="a", table=str(nocols))


def test_table(tmp_path):
    from pyhip.kit import table

    path = str(tmp_path / "t.csv")
    fields = ("dt", "n", "config", "us", "err")
    rows = [dict(dt="b", n="16", config="k1", us="2.0", err="0"),
            dict(dt="a", n="16", config="k1", us="3.0", err="0"),
            dict(dt="a", n="4", config="k2", us="2.0", err="0"),
            dict(dt="a", n="4", config="k1", us="1.0", err="0")]
    table.merge(path, rows, fields)
    got_fields, got = table.read(path)
    assert got_fields == fields
    # sorted by identity without config (numbers as numbers: 4 < 16), then us
    assert [(r["dt"], r["n"], r["config"]) for r in got] == [
        ("a", "4", "k1"), ("a", "4", "k2"), ("a", "16", "k1"), ("b", "16", "k1")]
    table.merge(path, [dict(dt="a", n="4", config="k2", us="0.5", err="1")])  # replaces
    _, got = table.read(path)
    assert [(r["config"], r["us"], r["err"]) for r in got[:2]] == [("k2", "0.5", "1"),
                                                                   ("k1", "1.0", "0")]
    assert len(got) == 4 and table.identity(fields) == ("dt", "n", "config")
    with pytest.raises(ValueError, match="config column"):
        table.identity(("dt", "us"))
    new = str(tmp_path / "new.csv")
    table.merge(new, [dict(dt="a", n="4", config="k1", us="0.4", err="0"),
                      dict(dt="c", n="4", config="k3", us="1.0", err="0")], fields)
    lines = []
    assert table.compare(path, new, out=lines.append) == 1
    assert lines[0].startswith("dt=a n=4  old k2") and "*" in lines[0]
    assert "(no old row)" in lines[1]


def test_tune(tmp_path, monkeypatch):
    import sys
    import types

    from pyhip.kit import table
    from pyhip.kit import tune as kt

    src = TOY.replace("ops = Registry()", "ops = Registry(table=TABLE, bucket='n', points=(4, 6), "
                                          "unkeyed=('seed',))")
    mod = types.ModuleType("kit_toy_tune")
    mod.TABLE = str(tmp_path / "tuned.csv")
    exec(compile(src, "toy.py", "exec"), mod.__dict__)
    monkeypatch.setitem(sys.modules, "kit_toy_tune", mod)
    op = kt.resolve("kit_toy_tune:axpy")
    assert op is mod.ops.axpy
    with pytest.raises(SystemExit, match="no op 'nope'"):
        kt.resolve("kit_toy_tune:nope")
    with pytest.raises(SystemExit, match="expected <module>:<op>"):
        kt.resolve("axpy")
    cases = kt.cases(op)  # the n axis (the bucket) is dropped
    assert list(cases) == ["a1", "a2"] and cases["a2"] == dict(n=None, alpha=2.0, seed=0)
    assert kt.case(op, "a1", ["alpha=3", "seed=5"]) == dict(n=None, alpha=3, seed=5)
    with pytest.raises(SystemExit, match="no tune case"):
        kt.case(op, "a3")
    assert kt.fields(op, ["err"]) == ["alpha", "n", "config", "us", "err"]
    assert kt.gpu_list("0-2,5") == [0, 1, 2, 5]
    monkeypatch.setattr(kt, "time_cold", lambda fn, iters: (fn(), 1.5)[1])
    logs = []
    kt.tune(op, cases["a1"], [4, 6], mod.TABLE, rounds=2, iters=1, log=logs.append, cid="a1")
    fields, rows = table.read(mod.TABLE)
    assert fields == ("alpha", "n", "config", "us", "err")
    got = [(r["n"], r["config"]) for r in rows]
    assert got == [("4", c) for c in ("loop", "tiled b2", "tiled b4", "queue fast", "queue slow")] \
        + [("6", c) for c in ("loop", "tiled b2", "queue fast", "queue slow")]  # b4 can't run 6
    assert rows[0]["us"] == "1.5" and rows[0]["alpha"] == "1.0" and float(rows[0]["err"]) == 0
    assert "(1 n/a: AssertionError)" in logs[-1]
    assert [op.name_of(c) for c in op.tuned(dict(cases["a1"], n=5))] == [
        "loop", "tiled b2", "queue fast", "queue slow"]  # the dispatch reads what tune wrote


def test_marks_return_the_function_in_written_order():
    a, b, c = param("p", (1,)), no_capture("r"), meta(k=1)

    def f():
        pass

    assert a(b(c(f))) is f
    assert f.kit_marks == [a, b, c]

    def g():
        pass

    assert stack(a, stack(b, c))(g) is g
    assert g.kit_marks == [a, b, c]
    with pytest.raises(TypeError):
        stack(a, len)
    with pytest.raises(TypeError):
        a(b)


def test_cases(toy):
    cases = toy["ops"]["axpy"].cases()
    assert [cid for cid, _ in cases] == ["n4-a1", "n4-a2", "n6-a1", "n6-a2"]
    assert cases[1][1] == dict(n=4, alpha=2.0, seed=0)
    assert toy["ops"]["axpy_dyn"].cases() == []


def test_param_forms():
    p = param("tokens, valid", {"v0": (512, 0), "v37": (512, 37)})
    assert p.points() == [("v0", dict(tokens=512, valid=0)), ("v37", dict(tokens=512, valid=37))]
    assert p.token(dict(tokens=512, valid=37)) == "v37"
    q = param("split", lambda dim: [s for s in (1, 2, 4) if dim % s == 0], id="ks{}")
    assert q.points(dict(dim=6)) == [("ks1", dict(split=1)), ("ks2", dict(split=2))]
    for bad in (lambda: param("a, b", (1, 2)), lambda: param("a", {"x": 1}, id="{}"),
                lambda: param("a", (1,), id="a"), lambda: param("a a", (1,)),
                lambda: param("a, b", {"x": (1, 2, 3)}), lambda: param("a", "xy")):
        with pytest.raises(TypeError):
            bad()


def test_configs_and_capture(toy):
    op = toy["ops"]["axpy"]
    cfgs = op.configs(dict(n=6, alpha=1.0, seed=0))
    assert [op.name_of(c) for c in cfgs] == ["loop", "tiled b2", "tiled b4", "queue fast",
                                             "queue slow"]
    assert [op.capturable(c) for c in cfgs] == [True, True, True, False, False]
    assert op.capturable(dict(kind="tiled", block=4, extra=1))
    with pytest.raises(AssertionError, match="n=6 % block 4"):  # limits: the impl raises
        op.run(dict(kind="tiled", block=4), [0.0] * 6, [0.0] * 6)
    with pytest.raises(KeyError, match="no implementation 'nope'"):
        op.capturable(dict(kind="nope"))
    with pytest.raises(TypeError, match="unknown knob"):
        op.capturable(dict(kind="queue", mode="fast", extra=1))


def test_names(toy):
    op = toy["ops"]["axpy"]
    for cfg in op.configs(dict(n=4, alpha=1.0, seed=0)):
        assert op.parse(op.name_of(cfg)) == cfg
    cfg = dict(kind="tiled", block=4, arb=128, tag="x1", f=0.5)
    assert op.name_of(cfg) == "tiled b4:arb=128,tag=x1,f=0.5"
    assert op.parse("tiled b4:arb=128,tag=x1,f=0.5") == cfg
    for bad in ("tiled", "tiled bx", "tiled b4 x", "loop:a=1", "tiled b4:a", "nope b4", ""):
        with pytest.raises(ValueError):
            op.parse(bad)


def test_meta_and_run(toy):
    op = toy["ops"]["axpy"]
    assert op.meta(dict(kind="tiled", block=4), "rows") == 32
    assert op.meta(dict(kind="tiled"), "graph_safe") is True
    assert op.meta(dict(kind="queue", mode="slow"), "rows") is None
    with pytest.raises(KeyError, match="no meta 'nope'"):
        op.meta(dict(kind="loop"), "nope")
    x, y = [1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0]
    assert op.run(dict(kind="tiled", block=4), x, y, alpha=2.0) == [3.0, 5.0, 7.0, 9.0]
    assert op.run(dict(kind="tiled", block=2, probe=1), x, y) == [2.0, 3.0, 4.0, 5.0]


def test_convention_errors():
    ref = "def f_ref(x, *, a=1): pass\n"
    _raises(ref + "def f_k(x, *, a=2, b=1): pass", "must start with exactly")
    _raises(ref + "def f_k(x, *, a=1, b=1): pass", "each knob needs exactly one @param")
    _raises(ref + "@param('b', (1,))\n@param('c', (1,))\ndef f_k(x, *, a=1, b=1): pass",
            "each knob needs exactly one @param")
    _raises(ref + "@param('b', (1,))\ndef f_k(x, b, *, a=1): pass", "must start with exactly")
    _raises(ref + "@param('b', (1,))\ndef f_k(x, *, a=1, b): pass", "keyword-only knob")
    _raises(ref + "def f_test(n=1): pass\n@param('n', (1,))\ndef f_k(x, *, a=1, n=1): pass",
            r"clash with the vocabulary")
    _raises(ref + "@param('b', (1,))\n@meta(m=lambda q: q)\ndef f_k(x, *, a=1, b=1): pass",
            r"meta 'm' uses unknown")
    _raises(ref + "@param('b', lambda q: (1,))\ndef f_k(x, *, a=1, b=1): pass",
            "outside the vocabulary")
    _raises(ref + "def f_test(n, m=1): pass", r"\['n'\] need a default or a @param")
    _raises(ref + "@param('q', (1,))\ndef f_test(n=1): pass", "must be distinct parameters")
    _raises(ref + "@param('n', lambda: (1,))\ndef f_test(n): pass", "a sequence or a dict")
    _raises(ref + "@no_capture('r')\ndef f_test(n=1): pass", "only @param marks")
    _raises(ref + "def f_test(*a): pass", "named parameters only")
    _raises(ref + "def f_check(out): pass", "does not take the arguments")
    _raises("def f_ref(x, *, out=None): pass\ndef f_check(out, expected, **_): pass",
            "multiple values for argument 'out'")
    _raises(ref + "def f_dims(y): pass", "does not take the arguments")
    _raises("def f_ref(x, *a): pass", "named parameters only")
    _raises("@meta(k=1)\ndef f_ref(x): pass", "only implementations")
    _raises(ref + "@param('b', ('x y', 'z'), id='{}')\ndef f_k(x, *, a=1, b='z'): pass",
            "cannot parse")
    _raises(ref + "@param('b', ('1', '2'), id='{}')\ndef f_k(x, *, a=1, b=0): pass",
            "reads back")
    ok = _ops(ref + "def f_check(out, expected, **inputs): pass\ndef f_dims(x, *, a=1): pass\n")
    assert list(ok["f"].impls) == [] and ok.f.check is not None and ok.f.dims is not None


def test_default_check():
    torch = pytest.importorskip("torch")
    assert_close(torch.ones(3, dtype=torch.bfloat16), torch.ones(3))
    with pytest.raises(AssertionError):
        assert_close(torch.ones(3), torch.zeros(3))
    # the interface's own arguments may be called out / expected
    assert_close(torch.ones(3), torch.ones(3), out=None, expected=None)


TUNED_TOY = ("n,config,us\n" + "".join(f"4,{c},1.0\n" for c in ("loop", "tiled b2", "tiled b4",
                                                                "queue fast", "queue slow"))
             + "".join(f"6,{c},1.0\n" for c in ("loop", "tiled b2", "queue fast")))


def _toy_project(pytester, src=TOY):
    """The toy with a tuned table (tiled b4 does not run n=6: not tuned there)."""
    pytester.makepyfile(toy=src.replace(
        "ops = Registry()",
        "import os\nops = Registry(table=os.path.join(os.path.dirname(__file__), 'tuned.csv'), "
        "bucket='n')"))
    pytester.makefile(".csv", tuned=TUNED_TOY)
    pytester.makepyfile(test_toy="from pyhip.kit.testing import op_tests\n"
                                 "from toy import ops\n"
                                 "test_ops = op_tests(ops)\n")


def test_generated_tests(pytester):
    _toy_project(pytester)
    res = pytester.runpytest("-q", "-p", "no:cacheprovider")
    # n4 cases: dims + 5 tuned configs; n6 cases: dims + 3
    res.assert_outcomes(passed=2 * 6 + 2 * 4)
    ids = pytester.runpytest("--collect-only", "-q").stdout.lines
    assert "test_toy.py::test_ops[axpy-n4-a1-tiled b4]" in ids
    assert "test_toy.py::test_ops[axpy-n6-a2-dims]" in ids
    assert "test_toy.py::test_ops[axpy-n6-a2-tiled b4]" not in ids


def test_generated_tests_without_table(pytester):
    pytester.makepyfile(toy=TOY)
    pytester.makepyfile(test_toy="from pyhip.kit.testing import op_tests\n"
                                 "from toy import ops\n"
                                 "test_ops = op_tests(ops)\n")
    res = pytester.runpytest("-q", "-p", "no:cacheprovider")
    # every grid point must run: tiled b4 fails on the n6 cases
    res.assert_outcomes(passed=4 * 6 - 2, failed=2)
    res.stdout.fnmatch_lines(["*n=6 % block 4*"])


def test_generated_tests_report_failures(pytester):
    _toy_project(pytester, TOY.replace("out += [alpha * a + b", "out += [alpha * a - b"))
    res = pytester.runpytest("-q", "-p", "no:cacheprovider", "-k", "tiled")
    res.assert_outcomes(failed=6)


def test_toy_runs_without_pytest_helpers(toy):
    op = toy["ops"]["axpy"]
    for _, case in op.cases():
        inputs = op.test(**case)
        assert op.dims(**inputs) == {k: case[k] for k in ("n", "alpha")}
        expected = op.ref(**inputs)
        for cfg in op.configs(case):
            if cfg.get("block", 1) > 2 and case["n"] == 6:
                with pytest.raises(AssertionError):
                    op.run(cfg, **inputs)
                continue
            assert op.check(op.run(cfg, **inputs), expected, **inputs)["err"] < 1e-12
