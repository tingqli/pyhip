# SPDX-License-Identifier: MIT
"""Tensor-native 128-bit store with an explicit CDNA cache policy."""

import flydsl.expr as fx


def store_with_policy(value, destination, policy):
    """Keep destination layout/bounds intact; no manual address reconstruction."""
    fragment = fx.make_rmem_tensor(4, fx.Int32)
    fragment.store(value)
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(cache_modifier=policy), fx.Int32)
    fx.copy(atom, fragment, destination)