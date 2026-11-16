# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Contracts a refactor must keep (host side only, no kernel runs):

    pytest a8w4/tests/test_a8w4_api.py -q          # from the directory holding a8w4/

* the frozen public API (what sglang calls): names and signatures, dict configs
  accepted by ``supported``, ``lookup_config`` results indexable by "kind",
  ``dyn_config`` raising on a shape without EP kernels, ``ref.ref_moe``'s positional
  argument order;
* every config string of the shipped tuned table parses and prints back unchanged.
"""

import csv
import inspect
import re

import pytest

import pyhip.contrib.flydsl.a8w4 as A
from pyhip.contrib.flydsl.a8w4 import ref

TP8 = (4096, 256, 256, 256, 8)
EP8 = (4096, 2048, 32, 256, 8)
KINDS = ("tp2k", "tp1k", "tp1kq", "ep3k", "ep1k")

# CSV: the default tuned table (an absolute path in the signature)
FROZEN = {
    "prepare_weights": "(w1, w2, w1_scale, w2_scale)",
    "shuffle_mxfp4": "(w1q, w1s, w2q, w2s)",
    "a8w4_moe": "(x, W, topk_weights, topk_ids, *, expert_offset=0, num_experts=None, "
                "act='silu', situ=None, config=None, routing='uniform', csv_path=CSV)",
    "a8w4_moe_dyn": "(x, W, topk_weights, topk_ids, m_valid, *, x_scale=None, "
                    "expert_offset=0, num_experts=None, act='silu', situ=None, "
                    "est_tokens=None, out=None, config=None, routing='uniform', "
                    "csv_path=CSV)",
    "dyn_config": "(shape, tokens, *, act='silu', routing='uniform', csv_path=CSV)",
    "supported": "(cfg, shape, *, act='silu', capturing=False, tokens=None)",
    "lookup_config": "(shape, tokens, *, act='silu', capturing=False, routing='uniform', "
                     "csv_path=CSV)",
}
FROZEN_REF = {
    "ref_moe": "(x, topk_weights, topk_ids, raw, expert_offset=0, act='silu', situ=None, "
               "quant_x=True)",
}


def _sig(fn):
    return re.sub(r"'[^']*/configs/a8w4_moe_tuned\.csv'", "CSV", str(inspect.signature(fn)))


@pytest.mark.parametrize("name", list(FROZEN))
def test_public_signature(name):
    assert _sig(getattr(A, name)) == FROZEN[name]


@pytest.mark.parametrize("name", list(FROZEN_REF))
def test_ref_signature(name):
    assert _sig(getattr(ref, name)) == FROZEN_REF[name]


def test_supported_takes_dict_config():
    # sglang's load-time shape check
    cfg = dict(kind="tp2k", block_m=64, g1_bcm=1, g2="base")
    assert A.supported(cfg, TP8)
    assert not A.supported(cfg, (4000, 256, 256, 256, 8))


@pytest.mark.parametrize("shape", [TP8, EP8])
@pytest.mark.parametrize("tokens", [1, 77, 4096])
@pytest.mark.parametrize("act", ["silu", "situv2"])
def test_lookup_config_kind(shape, tokens, act):
    assert A.lookup_config(shape, tokens, act=act)["kind"] in KINDS


def test_dyn_config():
    assert A.dyn_config(EP8, 1)["kind"] == "ep3k"
    with pytest.raises(ValueError):
        A.dyn_config(TP8, 1)  # sglang: the MORI path needs the EP kernels


def test_table_config_roundtrip():
    with open(A.DEFAULT_CSV) as fh:
        names = {r["config"] for r in csv.DictReader(fh)}
    assert names
    for n in sorted(names):
        assert A.config_name(A.parse_config(n)) == n
