import torch
from pyhip.contrib.gluon.moe_gemm_4wave import moe_2stage_gateup, moe_2stage_down

#####################################################################
import pytest
from typing import Optional
from pyhip import cudaPerf, jit, JIT, torchPerf

def div_up(a, b):
    return (a + b - 1) // b

# https://github.com/huggingface/transformers/blob/1fed6166c00b800330fcda8494f78cbcad8e4e3b/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L235-L263
def get_torch_ref(hidden_states, w1, w2, topk_weight, topk_ids):
    batch_size, hidden_dim = hidden_states.shape
    E, N1, K1 = w1.shape
    INTER_SIZE = N1 // 2
    final_hidden_states = torch.zeros(
        (batch_size, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )

    expert_mask = torch.nn.functional.one_hot(topk_ids.to(dtype=torch.long), num_classes=E).permute(2, 1, 0)

    for expert_idx in range(E):
        def expert_forward(n, x):
            gate_proj = w1[n, 0 : INTER_SIZE].t()
            up_proj = w1[n, INTER_SIZE :,].t()
            down_proj = w2[n].t()
            return (torch.nn.functional.silu(x @ gate_proj) * (x @ up_proj)) @ down_proj
        idx, top_x = torch.where(expert_mask[expert_idx])

        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_forward(expert_idx, current_state) * topk_weight[top_x, idx, None]

        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    return final_hidden_states


def _run_batch(kernel_type, B=1, weight_type=torch.bfloat16, TILE_M=128, TILE_N=64, run_count=10, HIDDEN_SIZE=2048, INTER_SIZE=1024, TOPK=8, E=128, TP=8):
    from aiter.ops.shuffle import shuffle_weight
    INTER_SIZE_TP = INTER_SIZE // TP
    BUF_COPY = 10
    hidden_states = (torch.randn([BUF_COPY, B, HIDDEN_SIZE], dtype=torch.bfloat16) + 1)*0.001
    if weight_type == torch.bfloat16:
        w_ = torch.randn([E, INTER_SIZE_TP * 2, HIDDEN_SIZE], dtype=weight_type)*0.1
        w1_ref = w_
        w_shuffled = shuffle_weight(w_).reshape(E, INTER_SIZE_TP * 2 // 16, -1)
        w1 = [w_shuffled.clone() for _ in range(BUF_COPY)]
        w1_org = [x.reshape(E, INTER_SIZE_TP * 2, -1) for x in w1]
        w_ = torch.randn([E, HIDDEN_SIZE, INTER_SIZE_TP], dtype=weight_type)
        w2_ref = w_
        w_shuffled = shuffle_weight(w_).reshape(E, HIDDEN_SIZE // 16, -1)
        w2 = [w_shuffled.clone() for _ in range(BUF_COPY)]
        w2_org = [x.reshape(E, HIDDEN_SIZE, -1) for x in w2]
        w1_scale = [None] * BUF_COPY
        w2_scale = [None] * BUF_COPY
    else:
        assert 0, f'Only bf16 weight is supported in this test, current weight_type={weight_type}'

    topk_weight = torch.randn([BUF_COPY, B, TOPK], dtype=torch.float32)
    topk_ids = torch.ones([BUF_COPY, B, TOPK], dtype=torch.int32)
    rep_e = div_up(B * TOPK, E)
    topk_ids_1d = torch.ones([rep_e, E], dtype=torch.int32)
    topk_ids_1d[:, ] = torch.randperm(E, dtype=torch.int32)
    topk_ids[:, ] = topk_ids_1d.reshape(-1)[ : B * TOPK].reshape(B, TOPK)
    access_expert = torch.unique(topk_ids[0])
    access_expert = access_expert.shape[0]

    flops = 2 * B * TOPK * (HIDDEN_SIZE * INTER_SIZE_TP * 2 + HIDDEN_SIZE * INTER_SIZE_TP)
    mem_size = B * HIDDEN_SIZE * 2 + (HIDDEN_SIZE * INTER_SIZE_TP * 2 + HIDDEN_SIZE * INTER_SIZE_TP) * access_expert * 2

    from aiter.fused_moe import moe_sorting

    def run(hidden_states, w1, w2, topk_weight, topk_ids, w1_scale, w2_scale):
        B = hidden_states.shape[0]
        N1, K1 = INTER_SIZE_TP * 2, HIDDEN_SIZE
        N2, K2 = HIDDEN_SIZE, INTER_SIZE_TP
        gemm1_out = torch.empty([B, TOPK, N1 // 2], dtype=hidden_states.dtype, device=hidden_states.device)
        stage2_out = torch.empty((B, TOPK, N2), dtype=torch.bfloat16, device=hidden_states.device)
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, cur_out = moe_sorting(
            topk_ids,
            topk_weight,
            E,
            K1,
            hidden_states.dtype,
            TILE_M,
            None,
            None,
            0,
        )
        BLOCK_TILE_SIZE_M = TILE_M
        BLOCK_TILE_SIZE_N = TILE_N
        grid = sorted_expert_ids.shape[0]
        if B * TOPK <= E:
            grid = B * TOPK
        moe_2stage_gateup[(N1 // BLOCK_TILE_SIZE_N, grid)](
                            hidden_states, w1, gemm1_out, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, w1_scale[0] if w1_scale is not None else None, B,
                            w1.dtype, TOPK, K1, N1, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, 4)

        moe_2stage_down[(N2 // BLOCK_TILE_SIZE_N, grid)](
                            gemm1_out, w2, stage2_out, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, w2_scale[0] if w2_scale is not None else None, B,
                            w1.dtype, TOPK, K2, N2, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, 4)
        return stage2_out.sum(dim=1)

    tflops_res = []
    latencies = []
    bw = []
    if kernel_type == 'aiter':
        from aiter.fused_moe import fused_moe
        from aiter import QuantType
        i = 0
        for _ in range(run_count):
            with cudaPerf(flops, mem_size, name=f"{kernel_type}[{B=},{str(weight_type).split('.')[1]}]") as p:
                fused_moe(hidden_states[i], w1_org[i], w2_org[i], topk_weight[i], topk_ids[i], quant_type=QuantType.No)
            i = (i + 1) % BUF_COPY
            tflops_res.append(p.tflops())
            latencies.append(p.dt())
            bw.append(p.bw())
    else:
        w1_qt_aiter = w1[0]
        w2_qt_aiter = w2[0]
        ref_out = get_torch_ref(hidden_states=hidden_states[0], w1=w1_ref, w2=w2_ref, topk_weight=topk_weight[0], topk_ids=topk_ids[0])
        cur_out = run(hidden_states=hidden_states[0], w1=w1_qt_aiter, w2=w2_qt_aiter, topk_weight=topk_weight[0], topk_ids=topk_ids[0], w1_scale=w1_scale[0], w2_scale=w2_scale[0])

        i = 0
        for _ in range(run_count):
            with cudaPerf(flops, mem_size, name=f"{kernel_type}[{B=},{str(weight_type).split('.')[1]}]") as p:
                run(hidden_states=hidden_states[i], w1=w1[i], w2=w2[i], topk_weight=topk_weight[i], topk_ids=topk_ids[i], w1_scale=w1_scale[i], w2_scale=w2_scale[i])
            i = (i + 1) % BUF_COPY
            tflops_res.append(p.tflops())
            latencies.append(p.dt())
            bw.append(p.bw())

        if not torch.allclose(ref_out, cur_out, rtol=0.1, atol=0.03):
            print(ref_out)
            print(cur_out)
            idx = torch.where(torch.abs(ref_out - cur_out) > 0.03)
            if len(idx[0]):
                print(f'idx = {idx}\nref={ref_out[idx]}\ncur={cur_out[idx]}\n{len(idx[0])}')
            assert 0, f"{kernel_type=}, {B=}, {weight_type=}, {TILE_M=}, {TILE_N=}, {run_count=}"
        else:
            print(f"{kernel_type}[{B=} {weight_type=}] acc OK")
    if run_count > 0:
        return {'flops': sum(tflops_res[1:])/len(tflops_res[1:]),
                'latency': sum(latencies[1:])/len(latencies[1:]) * 1e6,
                'bw': sum(bw[1:]) / len(bw[1:])}

def init_env():
    torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
    torch.set_default_device('cuda')
    torch.manual_seed(0)

def test_acc(TILE_M=128, TILE_N=64, HIDDEN_SIZE=4096, INTER_SIZE=2048, TP=8):
    init_env()
    batch = list(range(2, 64))
    batch += list(range(128, 256))
    batch += [i * 256 for i in range(1, 4)]
    batch += [i * 2048 for i in range(1, 5)]
    batch += list(range(2048 * 3, 2048 * 3 + 256))
    entry_common('mxn_splitk_2s', batch=batch, prec=[torch.bfloat16], TILE_M=TILE_M, TILE_N=TILE_N, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP, run_count=0)

def entry_common(kernel_type, batch, prec=[torch.bfloat16], TILE_M=128, TILE_N=64, HIDDEN_SIZE=2048, INTER_SIZE=1024, TOPK=10, E=512, TP=8, run_count=10):
    perf = {}
    perf[kernel_type] = {}
    for weight_type in prec:
        if weight_type is None: continue
        perf_prec = {}
        for i in batch:
            perf_prec[f'{i}'] = _run_batch(kernel_type, B=i, weight_type=weight_type, TILE_M=TILE_M, TILE_N=TILE_N, run_count=run_count, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TOPK=TOPK, E=E, TP=TP)
        perf[kernel_type][str(weight_type)] = perf_prec
    return perf

def show_perf(perf):
    print('\nsummary:')
    for kernel, vals in perf.items():
        for prec, vals_ in vals.items():
            for b, data in vals_.items():
                print(f'{kernel}[{prec:<4} B={b:<4}]: {data["latency"]:5.0f} us, {data["bw"]:6.1f} GB/s, {data["flops"]:4.1f} tflops')


@pytest.mark.parametrize("batch", [[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]])
def test_perf(batch, TILE_M=128, TILE_N=64, HIDDEN_SIZE=4096, INTER_SIZE=2048, TP=8, E=32, TOPK=8):
    init_env()
    perf = {}
    perf.update(entry_common('aiter', batch, prec=[torch.bfloat16,], HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP, TILE_M=TILE_M, TILE_N=TILE_N, E=E, TOPK=TOPK))
    perf.update(entry_common('mxn_splitk_2s', batch=batch, prec=[torch.bfloat16], TILE_M=TILE_M, TILE_N=TILE_N, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP, E=E, TOPK=TOPK))
    show_perf(perf)

if __name__ == '__main__':
    TILE_M = 256
    TILE_N = 256
    HIDDEN_SIZE = 4096
    INTER_SIZE = 1536
    TP = 1
    init_env()
    batch = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    batch = [4, 8, 16, 32, 64, 128, 256, 512]
    batch = [4096, 8192, 16384, 32768]
    batch = [8192, 19149]
    test_perf(batch, TILE_M=TILE_M, TILE_N=TILE_N, HIDDEN_SIZE=HIDDEN_SIZE, INTER_SIZE=INTER_SIZE, TP=TP, E=400, TOPK=20)
