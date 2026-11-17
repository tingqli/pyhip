# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The public interface (host side only, no kernel runs):

    pytest a8w4/tests/test_a8w4_api.py -q          # from the directory holding a8w4/

* names and signatures of the entry points and of the references;
* the ops registry (``ops.moe`` / ``ops.moe_dyn``, their implementations, the table);
* dispatch: ``pick`` returns the fastest tuned config, a known kind, for the
  MiMo-V2.6-Flash TP8 / EP8 shapes, never tp1kq (``@no_capture``) under capture;
  ``moe_dyn`` has EP kernels only; an untuned key raises ValueError;
* every config string of the shipped tuned table parses and prints back unchanged.
"""

import csv
import inspect
import os

import pytest

import pyhip.contrib.flydsl.a8w4 as A
from pyhip.contrib.flydsl.a8w4 import ref
from pyhip.contrib.flydsl.a8w4.moe import shape_vocab

TP8 = (4096, 256, 256, 256, 8)
EP8 = (4096, 2048, 32, 256, 8)
KINDS = ("tp2k", "tp1k", "tp1kq", "ep3k", "ep1k")
MOE, MOE_DYN = A.ops.moe, A.ops.moe_dyn

SIGNATURES = {
    "prepare_weights": "(w1, w2, w1_scale, w2_scale)",
    "shuffle_mxfp4": "(w1q, w1s, w2q, w2s)",
    "a8w4_moe": "(x, W, topk_weights, topk_ids, *, expert_offset=0, num_experts=None, "
                "act='silu', situ=None, **dispatch)",
    "a8w4_moe_dyn": "(x, W, topk_weights, topk_ids, m_valid, *, x_scale=None, "
                    "expert_offset=0, num_experts=None, act='silu', situ=None, "
                    "est_tokens=None, out=None, **dispatch)",
}
REF_SIGNATURES = {
    "ref_moe": "(x, topk_weights, topk_ids, raw, expert_offset=0, act='silu', situ=None, "
               "quant_x=True)",
    "moe_ref": "(x, W, topk_weights, topk_ids, *, expert_offset=0, num_experts=None, "
               "act='silu', situ=None)",
}


@pytest.mark.parametrize("name", list(SIGNATURES))
def test_public_signature(name):
    assert str(inspect.signature(getattr(A, name))) == SIGNATURES[name]


@pytest.mark.parametrize("name", list(REF_SIGNATURES))
def test_ref_signature(name):
    assert str(inspect.signature(getattr(ref, name))) == REF_SIGNATURES[name]


def test_registry():
    assert list(A.ops) == ["moe", "moe_dyn"]
    assert list(MOE.impls) == list(KINDS)
    assert list(MOE_DYN.impls) == ["ep3k"]
    assert A.ops.table == A.DEFAULT_CSV and os.path.isfile(A.DEFAULT_CSV)
    assert A.ops.bucket == "num_tokens"
    assert [MOE.capturable(dict(kind=k)) for k in KINDS] == [True, True, False, True, True]


@pytest.mark.parametrize("shape", [TP8, EP8])
@pytest.mark.parametrize("tokens", [1, 77, 4096])
@pytest.mark.parametrize("act", ["silu", "situv2"])
def test_pick_kind(shape, tokens, act):
    vocab = shape_vocab(shape, tokens, act)
    cfg = MOE.pick(vocab)
    assert cfg["kind"] in KINDS and cfg == MOE.tuned(vocab)[0]
    assert MOE.pick(vocab, capturing=True)["kind"] != "tp1kq"  # tp1kq is @no_capture


def test_dyn_pick():
    assert MOE_DYN.pick(shape_vocab(EP8, 1, "silu", op=MOE_DYN))["kind"] == "ep3k"
    with pytest.raises(ValueError):  # the MORI path needs the EP kernels: none tuned
        MOE_DYN.pick(shape_vocab(TP8, 1, "silu", op=MOE_DYN))


def test_untuned():
    for vocab in (shape_vocab((4000, 256, 256, 256, 8), 16, "silu"),  # an untuned shape
                  shape_vocab(TP8, 16, "silu", "hot")):  # no rows of this routing
        with pytest.raises(ValueError, match="no tuned config"):
            MOE.pick(vocab)
    with pytest.raises(ValueError, match=r"\(no table\)"):
        MOE.pick(shape_vocab(TP8, 16, "silu"), table=None)


def test_table_config_roundtrip():
    with open(A.DEFAULT_CSV) as fh:
        names = {r["config"] for r in csv.DictReader(fh)}
    assert names
    for n in sorted(names):
        assert MOE.name_of(MOE.parse(n)) == n
