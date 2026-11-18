"""pytest adapter of ``pyhip.kit``: the problems to test are passed in.

    from pyhip.kit import dim
    from pyhip.kit.testing import op_tests

    test_axpy = op_tests(ops.axpy, ops.axpy.problems("dtype", dim("n", (1, 1000), id="n{}")))

generates, for every problem (``op.problems(...)``: [(id, problem)]):

* ``<op>-<problem>-dims``: ``<op>_dims(**inputs)`` agrees with the problem (when the op
  has ``<op>_dims``);
* ``<op>-<problem>-<config name>`` for every config that should run the problem: ``run``
  it on the problem's inputs (``<op>_test``) and ``<op>_check`` it against ``<op>_ref``
  (default: ``torch.testing.assert_close``). With a tuned table (``table=``, default the
  registry's) these are the tuned configs of the problem's key and point (``op.tuned``:
  what the dispatch can pick; no rows, no items); with ``table=None``, every grid point
  (all must run).

Select with ``-k``, e.g. ``-k "tp8 and tiled"``. Items run problem by problem; the inputs
and the reference result of the current problem are computed once and kept until the
next.
"""

import inspect

import pytest

__all__ = ["op_tests", "assert_close"]

_DEFAULT = object()  # table=: the registry's


def assert_close(out, expected, /, **_):
    """The default ``<op>_check``: ``torch.testing.assert_close`` after casting
    ``expected`` to ``out``'s dtype (positional: an interface may have an ``out``)."""
    import torch

    if torch.is_tensor(out) and torch.is_tensor(expected):
        expected = expected.to(out.dtype)
    torch.testing.assert_close(out, expected)


def op_tests(op, problems, *, table=_DEFAULT):
    """A pytest test function over every (problem, config) of ``op``; ``problems``:
    [(id, problem)] (``op.problems(...)``), every dim without an ``<op>_test`` default
    set."""
    if op.test is None:
        raise TypeError(f"{op.name}: op_tests needs {op.name}_test (problem -> inputs)")
    path = op.registry.table if table is _DEFAULT else table
    items = []
    for pid, prob in problems:
        missing = [n for n in op.dim_names if prob.get(n) is None and n not in op._test_defaults]
        if missing:
            raise TypeError(f"{op.name} problem {pid!r}: dims {missing} unset")
        pre = f"{op.name}-{pid}" if pid else op.name
        if op.dims is not None:
            items.append(pytest.param(op, pid, prob, None, id=f"{pre}-dims"))
        cfgs = op.tuned(prob, table=path) if path is not None else op.configs(prob)
        for cfg in cfgs:
            items.append(pytest.param(op, pid, prob, cfg, id=f"{pre}-{op.name_of(cfg)}"))
    current = {}

    def prepared(op, pid, prob):
        key = (op.name, pid)
        if key not in current:
            current.clear()  # one problem at a time: inputs can be large
            inputs = op.test(**prob)
            try:
                inspect.signature(op.ref).bind(**inputs)
            except TypeError as e:
                raise TypeError(f"{op.name}_test{prob} returned arguments {op.name}_ref does "
                                f"not take: {e}") from None
            current[key] = inputs, op.ref(**inputs)
        return current[key]

    @pytest.mark.parametrize("op, pid, prob, cfg", items)
    def test_op(op, pid, prob, cfg):
        inputs, expected = prepared(op, pid, prob)
        if cfg is None:
            got = op.dims(**inputs)
            bad = {k: (v, prob.get(k, "<not a dim>")) for k, v in got.items()
                   if k not in prob or prob[k] != v}
            assert not bad, (f"{op.name}_dims disagrees with the problem, {{name: (dims, "
                             f"problem)}}: {bad}")
            return
        out = op.run(cfg, **inputs)
        (op.check or assert_close)(out, expected, **inputs)

    return test_op
