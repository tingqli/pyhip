import os
import random
os.environ['PYHIP_JIT_LOG'] = '1'
import pyhip
import torch
import math

from pyhip.asmjit import Addr2D, float_to_ieee754_bits_little
torch.cuda.set_device(2)
torch.set_default_device('cuda')
torch.manual_seed(0)
torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )

from functools import cache

def test_aiter(hidden_states,
               w1,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
               w2,  # [expert(local_expert:EP), dim, inter_dim]
               topk_weight,
               topk_ids,
               ):
    from aiter.fused_moe import fused_moe
    return fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
    )

def replace_kernel(stage):
    from aiter import get_2stage_cfgs

    if stage == 'stage1':
        pass
    elif stage == 'stage2':
        pass
    else:
        assert 0, f'unkown {stage=}'

B = 1
HIDDEN_SIZE = 2048
TP = 4
INTER_SIZE = 768
E = 128
TOPK = 8
INTER_SIZE_TP = INTER_SIZE // TP
BUF_COPY = 32
hidden_states = torch.randn([BUF_COPY, B, HIDDEN_SIZE], dtype=torch.bfloat16)
w1 = torch.randn([BUF_COPY, E, INTER_SIZE_TP * 2, HIDDEN_SIZE], dtype=torch.bfloat16)
w2 = torch.randn([BUF_COPY, E, HIDDEN_SIZE, INTER_SIZE_TP], dtype=torch.bfloat16)
topk_weight = torch.randn([BUF_COPY, B, TOPK], dtype=torch.float32)
topk_ids = torch.ones([BUF_COPY, B, TOPK], dtype=torch.int32)
topk_ids_base = torch.randperm(E, dtype=torch.int32)

if 1:
    import aiter
    test_aiter(hidden_states=hidden_states[0], w1=w1[0], w2=w2[0], topk_weight=topk_weight[0], topk_ids=topk_ids[0])

    i = 0
    for _ in range(10):
        idx_start = random.randint(0, E - TOPK)
        topk_ids[i,:,] = topk_ids_base[idx_start : idx_start + TOPK]
        with pyhip.cudaPerf(B * HIDDEN_SIZE * INTER_SIZE_TP * 2, B * (HIDDEN_SIZE * INTER_SIZE_TP * 3 * TOPK), name="moe"):
            test_aiter(hidden_states=hidden_states[i], w1=w1[i], w2=w2[i], topk_weight=topk_weight[i], topk_ids=topk_ids[i])
        i = (i + 1) % BUF_COPY
