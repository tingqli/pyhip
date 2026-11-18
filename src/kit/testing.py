"""pytest adapter of ``pyhip.kit``.

    from pyhip.kit.testing import op_tests
    test_ops = op_tests(ops)          # in a test module; ops: a kit.Registry

generates, for every op with an ``<op>_test`` and every case of its grid:

* ``<op>-<case>-dims``: ``<op>_dims(**inputs)`` agrees with the case (when the op has
  ``<op>_dims``);
* ``<op>-<case>-<config name>`` for every config that should run the case: ``run`` it
  on the case's inputs and ``<op>_check`` it against ``<op>_ref`` (default:
  ``torch.testing.assert_close``). With a tuned table (``Registry(table=)``) these are
  the tuned configs of the case's key and point (``op.tuned``: what the dispatch can
  pick; no rows, no items); without one, every grid point (all must run).

Select with ``-k``, e.g. ``-k "tp8 and tiled"``. Items run case by case; the inputs and
the reference result of the current case are computed once and kept until the next.
"""

import inspect

import pytest

__all__ = ["op_tests", "assert_close"]


def assert_close(out, expected, /, **_):
    """The default ``<op>_check``: ``torch.testing.assert_close`` after casting
    ``expected`` to ``out``'s dtype (positional: an interface may have an ``out``)."""
    import torch

    if torch.is_tensor(out) and torch.is_tensor(expected):
        expected = expected.to(out.dtype)
    torch.testing.assert_close(out, expected)


def op_tests(ops):
    """A pytest test function over every (op, case, config) of ``ops`` (a ``Registry``,
    or any mapping {name: Op})."""
    items = []
    for op in ops.values():
        if op.test is None:
            continue
        for cid, case in op.cases():
            pre = f"{op.name}-{cid}" if cid else op.name
            if op.dims is not None:
                items.append(pytest.param(op, cid, case, None, id=f"{pre}-dims"))
            tuned = op.registry.table is not None
            for cfg in op.tuned(case) if tuned else op.configs(case):
                items.append(pytest.param(op, cid, case, cfg, id=f"{pre}-{op.name_of(cfg)}"))
    current = {}

    def prepared(op, cid, case):
        key = (op.name, cid)
        if key not in current:
            current.clear()  # one case at a time: inputs can be large
            inputs = op.test(**case)
            try:
                inspect.signature(op.ref).bind(**inputs)
            except TypeError as e:
                raise TypeError(f"{op.name}_test{case} returned arguments {op.name}_ref does "
                                f"not take: {e}") from None
            current[key] = inputs, op.ref(**inputs)
        return current[key]

    @pytest.mark.parametrize("op, cid, case, cfg", items)
    def test_ops(op, cid, case, cfg):
        inputs, expected = prepared(op, cid, case)
        if cfg is None:
            got = op.dims(**inputs)
            bad = {k: (v, case.get(k, "<not a dim>")) for k, v in got.items()
                   if k not in case or case[k] != v}
            assert not bad, f"{op.name}_dims disagrees with the case, {{name: (dims, case)}}: {bad}"
            return
        out = op.run(cfg, **inputs)
        (op.check or assert_close)(out, expected, **inputs)

    return test_ops
