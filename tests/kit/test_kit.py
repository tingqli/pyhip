"""CPU tests of pyhip.kit (no GPU): registration, grids, config names, dispatch, the
checks a registration runs on the conventions, and the pytest tests ``kit.testing``
generates."""

import os
import textwrap

import pytest

from pyhip import kit
from pyhip.kit import Registry, dim, knob, meta, no_capture, stack
from pyhip.kit.testing import assert_close

pytest_plugins = ["pytester"]

TOY = '''
import random
from pyhip.kit import Registry, dim, knob, meta, no_capture, stack

ops = Registry()
op_register = ops.register


@op_register
def axpy_ref(x, y, *, alpha=1.0):
    return [alpha * a + b for a, b in zip(x, y)]


@op_register
@dim("n", (4, 6), id="n{}")
@dim("alpha", {"a1": 1.0, "a2": 2.0})
def axpy_problem(n, alpha, *, seed=0):
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


tiles = stack(knob("block", (2, 4), id="b{}"))


@op_register
@tiles
@meta(rows=lambda block: block * 8, graph_safe=True)
def axpy_tiled(x, y, *, alpha=1.0, block=2, **over):
    assert len(x) % block == 0, f"n={len(x)} % block {block}"  # a limit: the solver raises
    out = []
    for i in range(0, len(x), block):
        out += [alpha * a + b for a, b in zip(x[i:i + block], y[i:i + block])]
    return out


@op_register
@knob("mode", ("fast", "slow"), id="{}")
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
    exec("\n".join(out), {"op_register": reg.register, "dim": dim, "knob": knob,
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
    assert (op.ref, op.problem, op.check, op.dims) == (toy["axpy_ref"], toy["axpy_problem"],
                                                        toy["axpy_check"], toy["axpy_dims"])
    assert list(op.solvers) == ["loop", "tiled", "queue"]  # axpy_dyn_*: the longer prefix
    assert op.dim_names == ("n", "alpha")  # seed: the inputs' random seed, not a dim
    dyn = ops["axpy_dyn"]
    assert list(dyn.solvers) == ["loop"] and dyn.problem is None and dyn.dim_names == ()
    assert (toy["axpy_tiled"].kit_op, toy["axpy_dyn_loop"].kit_op) == ("axpy", "axpy_dyn")
    assert repr(op) == "Op('axpy', solvers=['loop', 'tiled', 'queue'])"
    assert repr(ops) == "Registry(['axpy', 'axpy_dyn'])"
    with pytest.raises(AttributeError):
        op.name = "x"
    with pytest.raises(AttributeError, match="no op 'nope'; registered: "):
        ops.nope  # noqa: B018
    with pytest.raises(AttributeError):
        ops._nope  # noqa: B018
    other = _toy()["ops"]  # a package imported twice: two independent registries
    assert other["axpy"] is not op and list(other["axpy"].solvers) == list(op.solvers)


def test_registration_rules():
    ref = "def f_ref(x, *, a=1): pass\n"
    _raises("def f_k(x): pass\n" + ref, r"register its <op>_ref first")
    _raises(ref + "def f_k(x, *, a=1): pass\ndef f_problem(n=1): pass",
            "register it before the solvers")
    _raises(ref + "def f_dyn_problem(n=1): pass", "register f_dyn_ref first")
    _raises(ref + "def f_dyn_k(x, *, a=1): pass\ndef f_dyn_ref(y): pass",
            "f_dyn_k was registered before it")
    _raises(ref + ref, "op 'f' is already registered")
    _raises(ref + "def f_dims(x, *, a=1): pass\n" * 2, "f_dims is already registered")
    _raises(ref + "def f_k(x, *, a=1): pass\n" * 2, "f_k is already registered")
    _raises("def items_ref(x): pass", r"taken by Registry.items")
    _raises("def _f_ref(x): pass", "private")
    reg = Registry()
    for bad in (int, len, knob("a", (1,)), lambda x: x):
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
        stack(knob("b", (1,)))(g_ref)

    def g_bad(x, *, b=1):  # a knob without its @knob
        pass

    @knob("b", ("x y", "z"), id="{}")
    def g_odd(x, *, b="z"):  # a value that cannot appear in a config name
        pass

    for f, match in ((g_bad, "each knob needs exactly one @knob"), (g_odd, "cannot parse")):
        with pytest.raises(TypeError, match=match):
            reg.register(f)
        assert "kit_op" not in f.__dict__ and list(reg.g.solvers) == ["k"]  # left no trace
    _raises("def f_ref(x, *, config=None): pass", r"\['config'\] are reserved")
    assert repr(Registry("lib")) == "Registry('lib', [])"


DISPATCH = '''
def f_ref(x, *, dt="a"):
    return ("ref",)

@dim("n", (4,))
def f_problem(n, dt="a", mode="s", *, seed=0):
    return dict(x=[0.0] * n, dt=dt)

def f_dims(x, *, dt="a"):
    return dict(n=len(x), dt=dt)

@knob("b", (1, 2), id="b{}")
def f_k1(x, *, dt="a", b=1):
    return ("k1", b)

@no_capture("not graph-safe")
def f_k2(x, *, dt="a"):
    return ("k2",)

def f_k3(x, *, dt="a"):
    return ("k3",)
'''

TABLE = """dt,mode,~n,config,us,err
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
    op = _ops(DISPATCH, table=str(path)).f

    def pick(capturing=False, table=str(path), **v):
        return op.name_of(op.pick(dict(op._blank, **v), table=table, capturing=capturing))

    assert pick(n=4, dt="a") == "k1 b2"  # the fastest; mode None matches the default "s"
    assert pick(n=3, dt="a") == "k1 b2"  # point 4
    assert pick(n=4, dt="c") == "k2" and pick(n=4, dt="c", capturing=True) == "k3"
    with pytest.raises(ValueError, match=r"can run under CUDA-graph capture: k2: not graph"):
        pick(n=4, dt="b", capturing=True)  # only k2 is tuned there
    assert pick(n=5, dt="a") == "k3" and pick(n=100, dt="a") == "k3"  # point 16, the largest
    assert pick(n=4, dt="a", mode="t") == "k3"
    assert pick(n=4, dt="b") == "k2"
    with pytest.raises(ValueError, match=r"no tuned config for \{'n': 4, 'dt': 'd'\} in .*"
                                         r"tuned.csv: tune it, or pass config="):
        pick(n=4, dt="d")  # no rows of this key
    with pytest.raises(ValueError, match="no tuned config"):
        pick(n=4, dt="a", mode="u")  # every key dim matches exactly
    with pytest.raises(ValueError, match=r"\(no table\)"):
        pick(n=4, dt="a", table=None)
    with pytest.raises(ValueError, match="no tuned config"):
        pick(n=4, dt="a", table=str(tmp_path / "none.csv"))
    prob = dict(op._blank, n=4, dt="a")
    assert op.pick(prob) is op.pick(dict(prob))  # memoized
    assert [op.name_of(c) for c in op.tuned(dict(prob, n=3))] == ["k1 b2", "k2", "k3"]
    assert op.tuned(dict(prob, dt="d")) == [] and op.tuned(prob, table=None) == []
    assert [(n, [op.name_of(c) for _, c in cs]) for n, cs in op.points(prob)] == [
        (4, ["k1 b2", "k2", "k3"]), (16, ["k3", "k1 b1"])]  # zz: not a solver
    with pytest.raises(ValueError, match="needs n"):
        op.pick(dict(op._blank, dt="a"))
    with pytest.raises(TypeError, match="lacks"):
        op.pick(dict(n=4))
    # a call: dims and hints -> pick -> run
    x3, x4 = [0.0] * 3, [0.0] * 4
    assert op(x4) == ("k1", 2) and op(x4, mode="t") == ("k3",) and op(x3) == ("k1", 2)
    assert op(x3, config="k3") == ("k3",) and op(x3, config=dict(solver="k1", b=1)) == ("k1", 1)
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
    # a table without the mode column: mode is no key; n not marked ~: an exact key
    old = tmp_path / "old.csv"
    old.write_text("dt,n,config,us\na,4,k3,1.0\n")
    assert pick(n=4, dt="a", mode="t", table=str(old)) == "k3"
    with pytest.raises(ValueError, match="no tuned config"):
        pick(n=3, dt="a", table=str(old))
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
    assert table.bucket(fields) is None and table.bucket(("dt", "~n", "config")) == "n"
    with pytest.raises(ValueError, match="at most one bucket"):
        table.bucket(("~a", "~n", "config"))
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

    src = TOY.replace("ops = Registry()", "ops = Registry(table=TABLE)") + (
        "\nfrom pyhip.kit.tune import Tuning\n"
        "TUNING = Tuning(ops.axpy, problems=ops.axpy.problems('alpha'), bucket='n', "
        "points=(4, 6))\n")
    mod = types.ModuleType("kit_toy_tune")
    mod.TABLE = str(tmp_path / "tuned.csv")
    exec(compile(src, "toy.py", "exec"), mod.__dict__)
    monkeypatch.setitem(sys.modules, "kit_toy_tune", mod)
    t = kt.resolve("kit_toy_tune:TUNING")
    op = t.op
    assert t is mod.TUNING and op is mod.ops.axpy and t.table == mod.TABLE
    with pytest.raises(SystemExit, match="not a pyhip.kit.tune.Tuning"):
        kt.resolve("kit_toy_tune:ops")
    with pytest.raises(SystemExit, match="expected <module>:<Tuning name>"):
        kt.resolve("axpy")
    assert list(t.problems) == ["a1", "a2"] and t.problems["a2"] == dict(n=None, alpha=2.0)
    assert t.problem("a1", ["alpha=3"]) == dict(n=None, alpha=3)
    with pytest.raises(SystemExit, match="name in \\['n', 'alpha'\\]"):
        t.problem("a1", ["seed=5"])  # not a dim
    with pytest.raises(SystemExit, match="no tune problem"):
        t.problem("a3")
    assert t.fields(["err"]) == ["alpha", "~n", "config", "us", "err"]
    for kw, match in ((dict(bucket="x"), "not a dim"), (dict(keys=("n",)), "keys"),
                      (dict(rotate=("z",)), "rotate"),
                      (dict(problems=[("p", {}), ("p", {})]), "two problems")):
        with pytest.raises(TypeError, match=match):
            kt.Tuning(op, **{**dict(problems=[], bucket="n", points=()), **kw})
    assert kt.gpu_list("0-2,5") == [0, 1, 2, 5]
    monkeypatch.setattr(kt, "time_cold", lambda fn, iters: (fn(), 1.5)[1])
    logs = []
    kt.tune(t, t.problems["a1"], [4, 6], mod.TABLE, rounds=2, iters=1, log=logs.append, pid="a1")
    fields, rows = table.read(mod.TABLE)
    assert fields == ("alpha", "~n", "config", "us", "err")
    got = [(r["~n"], r["config"]) for r in rows]
    assert got == [("4", c) for c in ("loop", "tiled b2", "tiled b4", "queue fast", "queue slow")] \
        + [("6", c) for c in ("loop", "tiled b2", "queue fast", "queue slow")]  # b4 can't run 6
    assert rows[0]["us"] == "1.5" and rows[0]["alpha"] == "1.0" and float(rows[0]["err"]) == 0
    assert "(1 n/a: AssertionError)" in logs[-1]
    assert [op.name_of(c) for c in op.tuned(dict(t.problems["a1"], n=5))] == [
        "loop", "tiled b2", "queue fast", "queue slow"]  # the dispatch reads what tune wrote


def test_bench(tmp_path, monkeypatch):
    import json
    import types

    import torch

    from pyhip.kit import bench as kb
    from pyhip.kit import tune as kt

    src = TOY.replace("ops = Registry()", "ops = Registry(table=TABLE)")
    mod = types.ModuleType("kit_toy_bench")
    mod.TABLE = str(tmp_path / "tuned.csv")
    with open(mod.TABLE, "w") as f:
        f.write("alpha,~n,config,us,err\n1.0,4,loop,2.0,0\n1.0,4,tiled b4,1.0,0\n")
    exec(compile(src, "toy.py", "exec"), mod.__dict__)
    op = mod.ops.axpy
    t = kt.Tuning(op, problems=op.problems("alpha"), bucket="n", points=(4, 6))
    prob = dict(n=None, alpha=1.0)

    # variants: [label=]base[:k=v,...] and py:<file>:<func>
    v = kb.parse_variant(op, "t=tiled b2:block=4,mode=x/1")
    assert (v.label, v.base, v.overrides) == ("t", "tiled b2", {"block": 4, "mode": ("x", 1)})
    assert kb.parse_variant(op, "b4", alias=lambda b: "tiled " + b).base == "tiled b4"
    with pytest.raises(SystemExit, match="unknown knob"):
        kb.parse_variant(op, "loop:block=4")
    with pytest.raises(SystemExit, match="cannot parse config name"):
        kb.parse_variant(op, "nope")
    (tmp_path / "mine.py").write_text("def f(x, y, *, alpha=1.0):\n    return list(y)\n")
    py = kb.parse_variant(op, f"m=py:{tmp_path / 'mine.py'}:f")
    r = kb.resolve(op, py, dict(prob, n=4))
    assert (r.config, r.cfg, r.cached) == (py.spec, None, None)
    assert r.run(dict(x=[1.0], y=[2.0], alpha=3.0)) == [2.0]

    # resolve: the overrides become config entries; @table: the fastest tuned config
    r = kb.resolve(op, kb.parse_variant(op, "tiled b2:block=4"), dict(prob, n=4))
    assert r.config == "tiled b4" and op.meta(r.cfg, "rows") == 32
    assert r.run(dict(x=[1.0] * 4, y=[1.0] * 4, alpha=2.0)) == [3.0] * 4
    table = kb.parse_variant(op, "@table")
    assert kb.resolve(op, table, dict(prob, n=4)).config == "tiled b4"
    assert kb.resolve(op, table, dict(prob, n=6)).config == "tiled b4"  # the nearest point
    assert "no tuned config" in kb.resolve(op, table, dict(prob, n=4, alpha=3.0))

    # protocol check: unknown cached loads need coldrot with >= MIN_COPIES copies
    kb.check_protocols([r], ["coldrot"], kb.MIN_COPIES, False)
    with pytest.raises(SystemExit, match="cached loads"):
        kb.check_protocols([r], ["cold"], kb.MIN_COPIES, False)
    kb.check_protocols([r], ["cold"], kb.MIN_COPIES, True)  # warns

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    meta = dict(time="t", git="g", flydsl="f", torch="t", gpu="gpu", arch="a", cus=1,
                visible="0", power_cap_w=None, cmd="cmd")

    def records(name):
        with open(tmp_path / "out" / f"{name}.jsonl") as f:
            return [json.loads(line) for line in f][1:]

    # check: FAIL for a config the solver can't run (the first variant that runs is the base),
    # n/a (not a failure) for @table without a tuned config
    bad = kb.check(t, prob, [4, 6], ["@table", "loop", "tiled b4"], diff_tol=1e-9,
                   rec=kb.Recorder(str(tmp_path / "out"), "check", meta))
    assert bad == 2
    got = [(r["T"], r["variant"], r["result"][:19]) for r in records("check")]
    assert got == [(4, "@table", "OK"), (4, "loop", "OK"), (4, "tiled b4", "OK"),
                   (6, "@table", "FAIL AssertionError"), (6, "loop", "OK"),
                   (6, "tiled b4", "FAIL AssertionError")]
    assert kb.check(t, dict(prob, alpha=3.0), [4], ["loop", "@table"],
                    rec=kb.Recorder(None, None, meta)) == 0

    # ab: timing stubbed; the base timed twice; the verdict beyond the noise band
    us = {"loop": 15.0, "tiled b4": 10.0, "tiled b4#2": 10.0, "tiled b2": 10.01}

    def interleaved(fns, protocols, timer, rounds, iters):
        for fn in fns.values():
            fn({})
        return {(p, lab): [us[lab]] * rounds for p in protocols for lab in fns}

    monkeypatch.setattr(kb, "Timer", lambda Ws: None)
    monkeypatch.setattr(kb, "interleaved", interleaved)
    kb.ab(t, prob, [4], ["tiled b4", "loop", "tiled b2"], rounds=2, diff_tol=1e-9,
          rec=kb.Recorder(str(tmp_path / "out"), "ab", meta))
    got = {r["variant"]: (r["verdict"], r["err_ref"], r["err_base"]) for r in records("ab")}
    assert got == {"tiled b4": ("base", 0.0, 0.0), "tiled b4#2": ("noise floor", 0.0, 0.0),
                   "loop": ("slower", 0.0, 0.0), "tiled b2": ("~", 0.0, 0.0)}
    with pytest.raises(SystemExit, match="the base at n 4: axpy: no tuned config"):
        kb.ab(t, dict(prob, alpha=3.0), [4], ["@table", "loop"],
              rec=kb.Recorder(None, None, meta))


def test_marks_return_the_function_in_written_order():
    a, b, c = knob("p", (1,)), no_capture("r"), meta(k=1)

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


def test_problems(toy):
    op = toy["ops"]["axpy"]
    probs = op.problems("n", "alpha")  # the @dim values of axpy_problem, by axis name
    assert [pid for pid, _ in probs] == ["n4-a1", "n4-a2", "n6-a1", "n6-a2"]
    assert probs[1][1] == dict(n=4, alpha=2.0)
    assert op.problems(("alpha", ["a2"]), dim("n", (8,), id="n{}")) == [("a2-n8", dict(n=8, alpha=2.0))]
    assert op.problems(("alpha", "a1")) == [("a1", dict(n=None, alpha=1.0))]
    with pytest.raises(KeyError, match="no @dim 'm'"):
        op.problems("m")
    with pytest.raises(KeyError, match=r"no value \['a3'\]"):
        op.problems(("alpha", ["a3"]))
    for axes in (("n", dim("n", (1,))), (dim("m", (1,)),)):
        with pytest.raises(TypeError, match="at most once"):
            op.problems(*axes)
    assert toy["ops"]["axpy_dyn"].problems() == [("", {})]
    named = _ops("def f_ref(x): pass\n@dim('m, n', {'sq': (2, 2)}, name='shape')\n"
                 "def f_problem(m, n, k=1): pass").f
    assert named.problems("shape") == [("sq", dict(m=2, n=2, k=1))]
    _raises("def f_ref(x): pass\n@dim('m, n', {'sq': (2, 2)})\ndef f_problem(m, n): pass",
            "needs name=")
    _raises("def f_ref(x): pass\n@dim('m', (1,), name='a')\n@dim('n', (1,), name='a')\n"
            "def f_problem(m, n): pass", "two @dim axes named 'a'")


def test_axis_forms():
    p = dim("tokens, valid", {"v0": (512, 0), "v37": (512, 37)})
    assert p.points() == [("v0", dict(tokens=512, valid=0)), ("v37", dict(tokens=512, valid=37))]
    assert p.token(dict(tokens=512, valid=37)) == "v37"
    q = knob("split", lambda n: [s for s in (1, 2, 4) if n % s == 0], id="ks{}")
    assert q.points(dict(n=6)) == [("ks1", dict(split=1)), ("ks2", dict(split=2))]
    for bad in (lambda: knob("a, b", (1, 2)), lambda: knob("a", {"x": 1}, id="{}"),
                lambda: knob("a", (1,), id="a"), lambda: knob("a a", (1,)),
                lambda: dim("a, b", {"x": (1, 2, 3)}), lambda: dim("a", "xy"),
                lambda: dim("a", lambda: (1,))):
        with pytest.raises(TypeError):
            bad()


def test_configs_and_capture(toy):
    op = toy["ops"]["axpy"]
    cfgs = op.configs(dict(n=6, alpha=1.0))
    assert [op.name_of(c) for c in cfgs] == ["loop", "tiled b2", "tiled b4", "queue fast",
                                             "queue slow"]
    assert [op.capturable(c) for c in cfgs] == [True, True, True, False, False]
    assert op.capturable(dict(solver="tiled", block=4, extra=1))
    with pytest.raises(AssertionError, match="n=6 % block 4"):  # limits: the solver raises
        op.run(dict(solver="tiled", block=4), [0.0] * 6, [0.0] * 6)
    with pytest.raises(KeyError, match="no solver 'nope'"):
        op.capturable(dict(solver="nope"))
    with pytest.raises(TypeError, match="unknown knob"):
        op.capturable(dict(solver="queue", mode="fast", extra=1))


def test_names(toy):
    op = toy["ops"]["axpy"]
    for cfg in op.configs(dict(n=4, alpha=1.0)):
        assert op.parse(op.name_of(cfg)) == cfg
    cfg = dict(solver="tiled", block=4, arb=128, tag="x1", f=0.5)
    assert op.name_of(cfg) == "tiled b4:arb=128,tag=x1,f=0.5"
    assert op.parse("tiled b4:arb=128,tag=x1,f=0.5") == cfg
    for bad in ("tiled", "tiled bx", "tiled b4 x", "loop:a=1", "tiled b4:a", "nope b4", ""):
        with pytest.raises(ValueError):
            op.parse(bad)


def test_meta_and_run(toy):
    op = toy["ops"]["axpy"]
    assert op.meta(dict(solver="tiled", block=4), "rows") == 32
    assert op.meta(dict(solver="tiled"), "graph_safe") is True
    assert op.meta(dict(solver="queue", mode="slow"), "rows") is None
    with pytest.raises(KeyError, match="no meta 'nope'"):
        op.meta(dict(solver="loop"), "nope")
    x, y = [1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0]
    assert op.run(dict(solver="tiled", block=4), x, y, alpha=2.0) == [3.0, 5.0, 7.0, 9.0]
    assert op.run(dict(solver="tiled", block=2, probe=1), x, y) == [2.0, 3.0, 4.0, 5.0]


def test_convention_errors():
    ref = "def f_ref(x, *, a=1): pass\n"
    _raises(ref + "def f_k(x, *, a=2, b=1): pass", "must start with exactly")
    _raises(ref + "def f_k(x, *, a=1, b=1): pass", "each knob needs exactly one @knob")
    _raises(ref + "@knob('b', (1,))\n@knob('c', (1,))\ndef f_k(x, *, a=1, b=1): pass",
            "each knob needs exactly one @knob")
    _raises(ref + "@knob('b', (1,))\ndef f_k(x, b, *, a=1): pass", "must start with exactly")
    _raises(ref + "@knob('b', (1,))\ndef f_k(x, *, a=1, b): pass", "keyword-only knob")
    _raises(ref + "def f_problem(n=1): pass\n@knob('n', (1,))\ndef f_k(x, *, a=1, n=1): pass",
            r"clash with the dims")
    _raises(ref + "@knob('b', (1,))\n@meta(m=lambda q: q)\ndef f_k(x, *, a=1, b=1): pass",
            r"meta 'm' uses unknown")
    _raises(ref + "@knob('b', lambda q: (1,))\ndef f_k(x, *, a=1, b=1): pass",
            "outside the dims")
    _raises(ref + "@dim('b', (1,))\ndef f_k(x, *, a=1, b=1): pass", "@dim describes the problem")
    _raises(ref + "@dim('q', (1,))\ndef f_problem(n=1): pass", "must be distinct dims")
    _raises(ref + "@dim('n', lambda: (1,))\ndef f_problem(n): pass", "a sequence or a dict")
    _raises(ref + "@no_capture('r')\ndef f_problem(n=1): pass", "only @dim marks")
    _raises(ref + "@knob('n', (1,))\ndef f_problem(n=1): pass", "only @dim marks")
    _raises(ref + "def f_problem(n=1, seed=0): pass", "seed .* must be keyword-only")
    _raises(ref + "def f_problem(*a): pass", "named parameters only")
    _raises(ref + "def f_check(out): pass", "does not take the arguments")
    _raises("def f_ref(x, *, out=None): pass\ndef f_check(out, expected, **_): pass",
            "multiple values for argument 'out'")
    _raises(ref + "def f_dims(y): pass", "does not take the arguments")
    _raises("def f_ref(x, *a): pass", "named parameters only")
    _raises("@meta(k=1)\ndef f_ref(x): pass", "only solvers")
    _raises(ref + "@knob('b', ('x y', 'z'), id='{}')\ndef f_k(x, *, a=1, b='z'): pass",
            "cannot parse")
    _raises(ref + "@knob('b', ('1', '2'), id='{}')\ndef f_k(x, *, a=1, b=0): pass",
            "reads back")
    ok = _ops(ref + "def f_check(out, expected, **inputs): pass\ndef f_dims(x, *, a=1): pass\n")
    assert list(ok["f"].solvers) == [] and ok.f.check is not None and ok.f.dims is not None


def test_default_check():
    torch = pytest.importorskip("torch")
    assert_close(torch.ones(3, dtype=torch.bfloat16), torch.ones(3))
    with pytest.raises(AssertionError):
        assert_close(torch.ones(3), torch.zeros(3))
    # the interface's own arguments may be called out / expected
    assert_close(torch.ones(3), torch.ones(3), out=None, expected=None)


TUNED_TOY = ("~n,config,us\n" + "".join(f"4,{c},1.0\n" for c in ("loop", "tiled b2", "tiled b4",
                                                                "queue fast", "queue slow"))
             + "".join(f"6,{c},1.0\n" for c in ("loop", "tiled b2", "queue fast")))


def _toy_project(pytester, src=TOY):
    """The toy with a tuned table (tiled b4 does not run n=6: not tuned there)."""
    pytester.makepyfile(toy=src.replace(
        "ops = Registry()",
        "import os\nops = Registry(table=os.path.join(os.path.dirname(__file__), 'tuned.csv'))"))
    pytester.makefile(".csv", tuned=TUNED_TOY)
    pytester.makepyfile(test_toy=TEST_TOY)


TEST_TOY = ("from pyhip.kit.testing import op_tests\nfrom toy import ops\n"
            "test_ops = op_tests(ops.axpy, ops.axpy.problems('n', 'alpha'))\n")


def test_generated_tests(pytester):
    _toy_project(pytester)
    res = pytester.runpytest("-q", "-p", "no:cacheprovider")
    # n4 problems: dims + 5 tuned configs; n6 problems: dims + 3
    res.assert_outcomes(passed=2 * 6 + 2 * 4)
    ids = pytester.runpytest("--collect-only", "-q").stdout.lines
    assert "test_toy.py::test_ops[axpy-n4-a1-tiled b4]" in ids
    assert "test_toy.py::test_ops[axpy-n6-a2-dims]" in ids
    assert "test_toy.py::test_ops[axpy-n6-a2-tiled b4]" not in ids


def test_generated_tests_without_table(pytester):
    pytester.makepyfile(toy=TOY)
    pytester.makepyfile(test_toy=TEST_TOY)
    res = pytester.runpytest("-q", "-p", "no:cacheprovider")
    # every config must run: tiled b4 fails on the n6 problems
    res.assert_outcomes(passed=4 * 6 - 2, failed=2)
    res.stdout.fnmatch_lines(["*n=6 % block 4*"])


def test_generated_tests_report_failures(pytester):
    _toy_project(pytester, TOY.replace("out += [alpha * a + b", "out += [alpha * a - b"))
    res = pytester.runpytest("-q", "-p", "no:cacheprovider", "-k", "tiled")
    res.assert_outcomes(failed=6)


def test_toy_runs_without_pytest_helpers(toy):
    from pyhip.kit.testing import op_tests

    op = toy["ops"]["axpy"]
    with pytest.raises(TypeError, match=r"dims \['n'\] unset"):
        op_tests(op, op.problems("alpha"))
    for _, prob in op.problems("n", "alpha"):
        inputs = op.problem(**prob)
        assert op.dims(**inputs) == {k: prob[k] for k in ("n", "alpha")}
        expected = op.ref(**inputs)
        for cfg in op.configs(prob):
            if cfg.get("block", 1) > 2 and prob["n"] == 6:
                with pytest.raises(AssertionError):
                    op.run(cfg, **inputs)
                continue
            assert op.check(op.run(cfg, **inputs), expected, **inputs)["err"] < 1e-12
