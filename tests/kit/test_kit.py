"""CPU tests of pyhip.kit (no GPU): discovery, grids, limits, config names, the checks
``collect`` runs on the conventions, and the pytest tests ``kit.testing`` generates."""

import textwrap

import pytest

from pyhip.kit import collect, meta, param, requires, stack
from pyhip.kit.testing import assert_close

pytest_plugins = ["pytester"]

TOY = '''
import random
from pyhip.kit import collect, meta, param, requires, stack


def axpy_ref(x, y, *, alpha=1.0):
    return [alpha * a + b for a, b in zip(x, y)]


@param("n", (4, 6), id="n{}")
@param("alpha", {"a1": 1.0, "a2": 2.0})
def axpy_test(n, alpha, seed=0):
    rng = random.Random(seed + n)
    return dict(x=[rng.random() for _ in range(n)], y=[rng.random() for _ in range(n)],
                alpha=alpha)


def axpy_dims(x, y, *, alpha=1.0):
    return dict(n=len(x), alpha=alpha)


def axpy_check(out, expected, **inputs):
    assert len(out) == len(expected), (len(out), len(expected))
    err = max((abs(a - b) for a, b in zip(out, expected)), default=0.0)
    assert err < 1e-12, err
    return {"err": err}


def axpy_loop(x, y, *, alpha=1.0):
    return [alpha * a + b for a, b in zip(x, y)]


tiles = stack(param("block", (2, 4), id="b{}"),
              requires(lambda n, block: n % block == 0, "n={n} % block {block}"))


@tiles
@meta(rows=lambda block: block * 8, graph_safe=True)
def axpy_tiled(x, y, *, alpha=1.0, block=2, **over):
    out = []
    for i in range(0, len(x), block):
        out += [alpha * a + b for a, b in zip(x[i:i + block], y[i:i + block])]
    return out


@param("mode", ("fast", "slow"), id="{}")
@requires(lambda capturing: not capturing, "not graph-safe")
@meta(rows=None, graph_safe=False)
def axpy_queue(x, y, *, alpha=1.0, mode="fast"):
    return [alpha * a + b for a, b in zip(x, y)]


def axpy_dyn_ref(x, y, valid, *, alpha=1.0):
    return [alpha * a + b if i < valid else 0.0 for i, (a, b) in enumerate(zip(x, y))]


def axpy_dyn_loop(x, y, valid, *, alpha=1.0):
    return axpy_dyn_ref(x, y, valid, alpha=alpha)


OPS = collect(globals())
'''


@pytest.fixture(scope="module")
def toy():
    ns = {"__name__": "toy"}
    exec(compile(TOY, "toy.py", "exec"), ns)
    return ns


def _ops(src):
    ns = {}
    exec(textwrap.dedent(src), {"collect": collect, "param": param, "requires": requires,
                                "meta": meta, "stack": stack}, ns)
    return collect(ns)


def _raises(src, match):
    with pytest.raises(TypeError, match=match):
        _ops(src)


def test_discovery(toy):
    ops = toy["OPS"]
    assert list(ops) == ["axpy", "axpy_dyn"]
    op = ops["axpy"]
    assert (op.ref, op.test, op.check, op.dims) == (toy["axpy_ref"], toy["axpy_test"],
                                                     toy["axpy_check"], toy["axpy_dims"])
    assert list(op.impls) == ["loop", "tiled", "queue"]  # axpy_dyn_* go to the longer prefix
    assert op.vocab == ("n", "alpha", "seed")
    dyn = ops["axpy_dyn"]
    assert list(dyn.impls) == ["loop"] and dyn.test is None and dyn.vocab == ()
    assert repr(op) == "Op('axpy', impls=['loop', 'tiled', 'queue'])"
    with pytest.raises(AttributeError):
        op.name = "x"


def test_marks_return_the_function_in_written_order():
    a, b, c = param("p", (1,)), requires(lambda p: True, "r"), meta(k=1)

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
    cases = toy["OPS"]["axpy"].cases()
    assert [cid for cid, _ in cases] == ["n4-a1", "n4-a2", "n6-a1", "n6-a2"]
    assert cases[1][1] == dict(n=4, alpha=2.0, seed=0)
    assert toy["OPS"]["axpy_dyn"].cases() == []


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


def test_configs_and_limits(toy):
    op = toy["OPS"]["axpy"]
    vocab = dict(n=6, alpha=1.0, seed=0)
    got = [(op.name_of(c), why) for c, why in op.configs(vocab)]
    assert got == [("loop", None), ("tiled b2", None), ("tiled b4", "n=6 % block 4"),
                   ("queue fast", None), ("queue slow", None)]
    assert [op.name_of(c) for c in op.candidates(vocab, capturing=True)] == ["loop", "tiled b2"]
    assert op.why_not(dict(kind="queue", mode="fast"), vocab, capturing=True) == "not graph-safe"
    assert op.why_not(dict(kind="tiled", block=4), dict(vocab, n=None)) is None  # unknown: skip
    assert op.why_not(dict(kind="tiled"), vocab) is None  # knob default (block=2)
    assert op.why_not(dict(kind="tiled", block=4, extra=1), vocab) == "n=6 % block 4"
    with pytest.raises(TypeError, match="missing from the vocabulary"):
        op.why_not(dict(kind="tiled", block=4), dict(alpha=1.0))
    with pytest.raises(KeyError, match="no implementation 'nope'"):
        op.why_not(dict(kind="nope"), vocab)
    with pytest.raises(TypeError, match="unknown knob"):
        op.why_not(dict(kind="queue", mode="fast", extra=1), vocab)


def test_names(toy):
    op = toy["OPS"]["axpy"]
    for cfg, _ in op.configs(dict(n=4, alpha=1.0, seed=0)):
        assert op.parse(op.name_of(cfg)) == cfg
    cfg = dict(kind="tiled", block=4, arb=128, tag="x1", f=0.5)
    assert op.name_of(cfg) == "tiled b4:arb=128,tag=x1,f=0.5"
    assert op.parse("tiled b4:arb=128,tag=x1,f=0.5") == cfg
    for bad in ("tiled", "tiled bx", "tiled b4 x", "loop:a=1", "tiled b4:a", "nope b4", ""):
        with pytest.raises(ValueError):
            op.parse(bad)


def test_meta_and_run(toy):
    op = toy["OPS"]["axpy"]
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
    _raises(ref + "@param('n', (1,))\ndef f_test(n): pass\n"
            "@param('b', (1,))\n@requires(lambda m: m, 'r')\ndef f_k(x, *, a=1, b=1): pass",
            r"unknown name\(s\) \['m'\]")
    _raises(ref + "@param('n', (1,))\ndef f_test(n): pass\n"
            "@param('b', (1,))\n@requires(lambda n: n, '{q}')\ndef f_k(x, *, a=1, b=1): pass",
            r"unknown name\(s\) \['q'\]")
    _raises(ref + "@param('b', (1,))\n@meta(m=lambda q: q)\ndef f_k(x, *, a=1, b=1): pass",
            r"meta 'm' uses unknown")
    _raises(ref + "@param('b', lambda q: (1,))\ndef f_k(x, *, a=1, b=1): pass",
            "outside the vocabulary")
    _raises(ref + "def f_test(n, m=1): pass", r"\['n'\] need a default or a @param")
    _raises(ref + "@param('q', (1,))\ndef f_test(n=1): pass", "must be distinct parameters")
    _raises(ref + "@param('n', lambda: (1,))\ndef f_test(n): pass", "a sequence or a dict")
    _raises(ref + "@requires(lambda n: n, 'r')\ndef f_test(n=1): pass", "only @param marks")
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
    ok = _ops(ref + "def f_check(out, expected, **inputs): pass\ndef f_dims(x, *, a=1): pass\n"
              "_hidden = lambda x: x\nf_alias = int\n")
    assert list(ok["f"].impls) == []


def test_default_check():
    torch = pytest.importorskip("torch")
    assert_close(torch.ones(3, dtype=torch.bfloat16), torch.ones(3))
    with pytest.raises(AssertionError):
        assert_close(torch.ones(3), torch.zeros(3))
    # the interface's own arguments may be called out / expected
    assert_close(torch.ones(3), torch.ones(3), out=None, expected=None)


def test_generated_tests(pytester):
    pytester.makepyfile(toy=TOY)
    pytester.makepyfile(test_toy="from pyhip.kit.testing import op_tests\n"
                                 "from toy import OPS\n"
                                 "test_ops = op_tests(OPS)\n")
    res = pytester.runpytest("-q", "-rs", "-p", "no:cacheprovider")
    # 4 cases x (dims + loop + tiled b2/b4 + queue fast/slow); n6 cases skip tiled b4
    res.assert_outcomes(passed=4 * 6 - 2, skipped=2)
    res.stdout.fnmatch_lines(["*n=6 % block 4*"])
    ids = pytester.runpytest("--collect-only", "-q").stdout.lines
    assert "test_toy.py::test_ops[axpy-n4-a1-tiled b4]" in ids
    assert "test_toy.py::test_ops[axpy-n6-a2-dims]" in ids


def test_generated_tests_report_failures(pytester):
    pytester.makepyfile(toy=TOY.replace("out += [alpha * a + b", "out += [alpha * a - b"))
    pytester.makepyfile(test_toy="from pyhip.kit.testing import op_tests\n"
                                 "from toy import OPS\n"
                                 "test_ops = op_tests(OPS)\n")
    res = pytester.runpytest("-q", "-p", "no:cacheprovider", "-k", "tiled")
    res.assert_outcomes(failed=6, skipped=2)


def test_toy_runs_without_pytest_helpers(toy):
    op = toy["OPS"]["axpy"]
    for _, case in op.cases():
        inputs = op.test(**case)
        assert op.dims(**inputs) == {k: case[k] for k in ("n", "alpha")}
        expected = op.ref(**inputs)
        for cfg in op.candidates(case):
            assert op.check(op.run(cfg, **inputs), expected, **inputs)["err"] < 1e-12
