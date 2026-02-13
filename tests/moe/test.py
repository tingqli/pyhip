import pyhip
import torch

from pyhip.kernels import moe_gemm_mxfp4, moe_gemm_mxfp4_gateup_4wave, moe_gemm_mxfp4_gateup_8wave, moe_gemm_final_reduce_bf16

import aiter
from aiter.utility import fp4_utils
from aiter.ops.shuffle import shuffle_weight
import os

torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
torch.set_default_device('cuda')
torch.manual_seed(0)
#torch.cuda.set_device(4)

EIDX = int(os.getenv("EIDX","0"))

def mxfp4_dequant(quant, scale):
    _src = fp4_utils.mxfp4_to_f32(quant.view(torch.float4_e2m1fn_x2)).to(dtype=torch.bfloat16)
    rows, cols = _src.shape
    #return _src
    #print(rows, cols, _src.shape)
    _scale = fp4_utils.e8m0_to_f32(scale).to(dtype=torch.bfloat16)
    _scale = _scale.reshape(rows//32, cols//256, 4, 1, 16, 4).repeat(1,1,1,32,1,1).view(rows//32, cols//256, 128, 16, 4).permute(0,1,4,3,2)
    for r in range(0, rows, 32):
        for c in range(0, cols, 256):
            sss = _scale[r//32, c//256, :, :, :]
            _src[r:r+16, c:c+128] *= sss[0]
            _src[r+16:r+32, c:c+128] *= sss[1]
            _src[r:r+16, c+128:c+256] *= sss[2]
            _src[r+16:r+32, c+128:c+256] *= sss[3]

    return _src

def div_up(x, y):
    return (x + y - 1) // y

def generate_weights(shape, BUF_COPY, weight_type = torch.float4_e2m1fn_x2):
    import aiter
    from aiter.utility import fp4_utils
    from aiter.ops.shuffle import shuffle_weight

    assert weight_type == torch.float4_e2m1fn_x2
    w_ = torch.randn(shape, dtype=torch.bfloat16)
    w1_qt, w1_sc = aiter.get_torch_quant(aiter.QuantType.per_1x32)(w_, quant_dtype=weight_type)
    w1_sc = fp4_utils.e8m0_shuffle(w1_sc)
    w1_qt = shuffle_weight(w1_qt)
    multi_w1_qt = [w1_qt.clone() for _ in range(BUF_COPY)]
    multi_w1_sc = [w1_sc.clone() for _ in range(BUF_COPY)]
    return multi_w1_qt, multi_w1_sc

def generate_moe_inputs(B, HIDDEN_SIZE, INTER_SIZE_TP, EXPERTS, TOPK):
    BUF_COPY=32
    weight_type = torch.float4_e2m1fn_x2
    act_dtype = torch.bfloat16

    h = torch.randn([B, HIDDEN_SIZE], dtype=act_dtype)
    hidden_states = [h.clone() for _ in range(BUF_COPY)]
    
    w1q, w1s = generate_weights([EXPERTS, INTER_SIZE_TP * 2, HIDDEN_SIZE], BUF_COPY, weight_type)
    w2q, w2s = generate_weights([EXPERTS, HIDDEN_SIZE, INTER_SIZE_TP], BUF_COPY, weight_type)

    t = torch.randn([B, TOPK], dtype=torch.float32)
    topk_weight = [t.clone() for _ in range(BUF_COPY)]

    i = torch.ones([B, TOPK], dtype=torch.int32)

    tokens_per_expert = div_up(B * TOPK, EXPERTS)
    topk_ids_1d = torch.ones([tokens_per_expert, EXPERTS], dtype=torch.int32)
    topk_ids_1d[:, ] = torch.randperm(EXPERTS, dtype=torch.int32)
    topk_ids = torch.ones([BUF_COPY, B, TOPK], dtype=torch.int32)
    topk_ids[:, ] = topk_ids_1d.reshape(-1)[ : B * TOPK].reshape(B, TOPK)
    access_expert = torch.unique(topk_ids[0])
    
    # prepare stage is over, now yield 
    for i in range(100000):
        k = i % BUF_COPY
        yield hidden_states[k], topk_weight[k], topk_ids[k], w1q[k], w1s[k], w2q[k], w2s[k]


def de_shuffle_weight(weight, mfma_MN = 16):
    M, K = weight.shape
    K_bytes = K * weight.itemsize
    sizeof_DW4 = 16
    mfma_K_lanes = 64 // mfma_MN
    mfma_K_L = sizeof_DW4//weight.itemsize
    mfma_K = mfma_K_lanes * mfma_K_L 

    assert M % mfma_MN == 0
    mfma_K_bytes = mfma_K_lanes * sizeof_DW4
    assert K_bytes % mfma_K_bytes == 0
    #x = x.reshape(M//mfma_MN, mfma_MN, K//mfma_K, mfma_K_lanes, mfma_K_L)
    #x = x.permute(0,2,3,1,4)

    assert K % mfma_K == 0
    weight = weight.reshape(M//mfma_MN, K//mfma_K, mfma_K_lanes, mfma_MN, mfma_K_L)
    weight = weight.permute(0,3,1,2,4)
    weight = weight.reshape(M, K).contiguous()
    return weight

def moe_gemm_ref(BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, gate_up,
                 sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
                 weight, w_scale, input, i_scale, output, debug_e_idx = -1, EIDX = None):
    M = input.shape[0]

    if "_x2" in str(weight.dtype):
        print(f"{BLOCK_TILE_SIZE_M=}")
        print(f"{BLOCK_TILE_SIZE_N=}")
        print(f"{input.shape=} {input.dtype=}")         # input.shape=[24064, 2048] input.dtype=torch.float4_e2m1fn_x2
        if i_scale is not None:
            print(f"  {i_scale.shape} {i_scale.dtype}")   # [208896, 128] torch.float8_e8m0fnu
        print(f"{output.shape=}")                       # output.shape = [24064, 8, 1536]
        print(f"{weight.shape=} {weight.dtype=}")       # weight.shape=[128, 3072, 2048] weight.dtype=torch.float4_e2m1fn_x2
        if w_scale is not None:
            print(f"  {w_scale.shape} {w_scale.dtype}")   # [393216 (128*3072), 128] torch.float8_e8m0fnu
        print(f"{sorted_ids.shape=}")                   # sorted_ids.shape=torch.Size([208888])
        print(f"{sorted_expert_ids.shape=}")            # sorted_expert_ids.shape=torch.Size([1632])
        print(f"{sorted_weights.shape=}")               # sorted_weights.shape=torch.Size([208888])
        print(f"{num_valid_ids=}")

    NUM_EXPERTS, OC, IC = weight.shape
    if "_x2" in str(weight.dtype): IC *= 2
    NUM_BLOCKS = sorted_expert_ids.shape[0]

    if i_scale is not None:
        input = input.view(torch.int8)
        print(f"===== i_scale {i_scale.shape} {i_scale.dtype}")
    if w_scale is not None:
        weight = weight.view(torch.int8)
        w_scale = w_scale.view(NUM_EXPERTS, OC, -1)
        print(f"===== w_scale {w_scale.shape} {w_scale.dtype}")

    num_sorted_ids = sorted_ids.shape[0]
    
    #EIDX = 1
    if EIDX is None:
        EIDX = sorted_expert_ids.shape[0]
    
    #for e_idx in range(sorted_expert_ids.shape[0]):
    for e_idx in range(EIDX):
        #for e_idx in range(1):
        #e_idx = EIDX
        s_e_id = sorted_expert_ids[e_idx]
        max_id = num_valid_ids[0]
        if e_idx * BLOCK_TILE_SIZE_M >= max_id: continue
        i0 = e_idx*BLOCK_TILE_SIZE_M
        i1 = (e_idx+1)*BLOCK_TILE_SIZE_M

        ids = sorted_ids[i0:i1].clone()
        valid_mask = (ids & 0xFFFFFF) < torch.tensor(M)
        ids[(ids & 0xFFFFFF) >= torch.tensor(M)] = 0
        tok_ids = ids & 0xFFFFFF
        top_k = ids >> 24
        tok_w = sorted_weights[i0:i1]
        expert_w = weight[s_e_id, ...]
        if 0:
            print("====================== tok_ids")
            print(tok_ids)
            print("====================== top_k")
            print(top_k)
            print("====================== s_e_id")
            print(s_e_id)
            print("====================== tok_w")
            print(tok_w)

            if e_idx == -11:
                print(tok_ids)

        if gate_up:
            src = input[tok_ids,...]
        else:
            src = input[tok_ids, top_k, ...]

        if 0:
            print("??????????? ", src.shape, src.dtype)
            print(src.view(torch.int32)[:16, :16])
            

        if i_scale is not None:
            src = mxfp4_dequant(src, i_scale[i0:i1,...])

        w = de_shuffle_weight(expert_w)

        if 0:
            print(w.shape, w.dtype, w.view(torch.int32).shape)
            wi32_gate = w.view(torch.int32)
            wi32_up = w[OC//2:,:].view(torch.int32)
            print(wi32_gate[:16, :16])
            print(wi32_up[:16, :16])
            print(wi32_gate[16:16*2, :16])
            print(wi32_up[16:16*2, :16])
            
        #assert 0

        if debug_e_idx == e_idx:
            print(w.view(torch.int32)[:16, :16])

        if w_scale is not None:
            w = mxfp4_dequant(w, w_scale[s_e_id,...])

        act = src @ w.t()

        if 0:
            m0,n0 = 0,0
            print(act[m0:m0+16, n0:n0+32])
            n0 += OC//2
            print(act[m0:m0+16, n0:n0+32])
        
        if gate_up:
            act_gate = act[:, :(OC//2)]
            act_up = act[:,(OC//2):]
            #print("=============================")
            #print(act_gate)
            #print(act_up)
            act = torch.nn.functional.silu(act_gate) * act_up

        if debug_e_idx == e_idx:
            print(f"======== ===========")
            print(tok_ids)
            print(f"======== {debug_e_idx=}  {act.shape} {act.dtype}")
            m0 = 0*(BLOCK_TILE_SIZE_M//2)
            n0 = (0)*(BLOCK_TILE_SIZE_N//2)
            print(act[m0:m0+16, n0:n0+64])
            print(act[m0, n0:n0+64].tolist())

        if gate_up:
            # output[tok_ids[valid_mask], top_k[valid_mask], :] = act[valid_mask, ...]
            #n0 = BLOCK_TILE_SIZE_N//2 * 0
            #n1 = n0 + BLOCK_TILE_SIZE_N//2
            #print(act.shape, n0, n1)
            #assert 0

            output[tok_ids[valid_mask], top_k[valid_mask], :] = act[valid_mask, :]
        else:
            output[tok_ids[valid_mask], ...] += act[valid_mask, ...] * tok_w[valid_mask, None]

def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:    # Which means that all elements in x and y are 0
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    diff = (1 - sim).item()
    assert diff == diff, "diff is nan!"
    return diff



def test_gateup(B, HIDDEN_SIZE, INTER_SIZE_TP, EXPERTS, TOPK, TILE_M=128, TILE_N=128, test_acc=True):
    from aiter.utility.fp4_utils import moe_mxfp4_sort
    from aiter.fused_moe import moe_sorting

    quant_func = aiter.get_hip_quant(aiter.QuantType.per_1x32)

    #EXPERTS, TOPK = 128, 8
    #B, HIDDEN_SIZE, INTER_SIZE_TP = 256, 4096, 1536
    gen_input = generate_moe_inputs(B, HIDDEN_SIZE, INTER_SIZE_TP, EXPERTS, TOPK)

    hidden_states, topk_weight, topk_ids, w1, w1_scale, w2q, w2s = next(gen_input)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, cur_out = \
        moe_sorting(topk_ids, topk_weight, EXPERTS, HIDDEN_SIZE, hidden_states.dtype, TILE_M, None, None, 0)
    # quantize input
    hidden_states_q, hidden_states_scale = \
        quant_func(hidden_states, scale=None, quant_dtype=torch.float4_e2m1fn_x2, num_rows=None,)
    hidden_states_scale = \
        moe_mxfp4_sort(hidden_states_scale, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids, token_num=B, block_size=TILE_M)

    gemm1_out = torch.zeros([B, TOPK, INTER_SIZE_TP], dtype=torch.bfloat16)

    if test_acc:
        gemm1_out[...] = 0
        the_out = gemm1_out.clone()
        moe_gemm_ref(TILE_M, TILE_N, True, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
                     w1, w1_scale,
                     hidden_states_q, hidden_states_scale,
                     gemm1_out)

        gateup_OC = w1.shape[1]
        assert gateup_OC % TILE_N == 0
        num_oc_blocks = gateup_OC // TILE_N
        num_e_blocks = sorted_expert_ids.shape[0]
        moe_gemm_mxfp4_gateup = moe_gemm_mxfp4_gateup_4wave
        num_threads_per_wg = 256

        moe_gemm_mxfp4_gateup([num_oc_blocks * num_e_blocks],[num_threads_per_wg],
            TILE_M, TILE_N,
            w1.shape[0], w1.shape[1], w1.shape[2], 
            True, TOPK, # gate_up,
            sorted_ids.data_ptr(),
            sorted_weights.data_ptr(),
            sorted_expert_ids.data_ptr(),
            num_valid_ids.data_ptr(),
            w1.data_ptr(), w1_scale.data_ptr(),
            hidden_states_q.data_ptr(), hidden_states_scale.data_ptr(),
            the_out.data_ptr(), B)
        
        diff = calc_diff(gemm1_out, the_out)
        print(f"{diff=:.2f}")
        if diff > 0.01:
            for i in range(B):
                for t in range(TOPK):
                    diff = calc_diff(gemm1_out[i,t,:], the_out[i,t,:])
                    if diff != diff or diff > 0.02:
                        print(f"=============== {i},{t}   {diff:.3f}")
                        print(f"     gemm1_out ")
                        print(gemm1_out[i,t,:].view(-1,64))
                        print("      the_out ")
                        print(the_out[i,t,:].view(-1,64))
                        assert 0
        print("accuracy passed!")
    else:
        for i in range(16*2):
            hidden_states, topk_weight, topk_ids, w1, w1_scale, w2q, w2s = next(gen_input)
            sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, cur_out = \
                moe_sorting(topk_ids, topk_weight, EXPERTS, HIDDEN_SIZE, hidden_states.dtype, TILE_M, None, None, 0)
            
            #print(sorted_expert_ids.tolist())
            #assert 0
            # quantize input
            hidden_states_q, hidden_states_scale = \
                quant_func(hidden_states, scale=None, quant_dtype=torch.float4_e2m1fn_x2, num_rows=None,)
            hidden_states_scale = \
                moe_mxfp4_sort(hidden_states_scale, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids, 
                               token_num=B, block_size=TILE_M)
            the_out = torch.empty([B, TOPK, INTER_SIZE_TP], dtype=torch.bfloat16)

            gateup_OC, gateup_IC = w1.shape[1:]
            assert gateup_OC % TILE_N == 0
            num_oc_blocks = gateup_OC // TILE_N
            num_e_blocks = sorted_expert_ids.shape[0]

            flops = num_oc_blocks * num_e_blocks * (TILE_N * TILE_M * HIDDEN_SIZE * 2)
            mem_size = 0
            moe_gemm_mxfp4_gateup = moe_gemm_mxfp4_gateup_4wave
            num_threads_per_wg = 256
            with pyhip.cudaPerf(flops, mem_size, name=f"moe_gemm_mxfp4_gateup_{TILE_M}_{TILE_N} [{num_oc_blocks},{num_e_blocks}]") as p:
                moe_gemm_mxfp4_gateup([num_oc_blocks * num_e_blocks],[num_threads_per_wg],
                                TILE_M, TILE_N,
                                w1.shape[0], w1.shape[1], w1.shape[2], 
                                True, TOPK, # gate_up,
                                sorted_ids.data_ptr(),
                                sorted_weights.data_ptr(),
                                sorted_expert_ids.data_ptr(),
                                num_valid_ids.data_ptr(),
                                w1.data_ptr(), w1_scale.data_ptr(),
                                hidden_states_q.data_ptr(), hidden_states_scale.data_ptr(),
                                the_out.data_ptr(), B)
        print(f"{gateup_IC=} {num_valid_ids=}")


def test_down():

    E = 128
    TOPK = 8
    HIDDEN_SIZE = 4096
    INTER_SIZE_TP = 768
    TILE_M = 128
    TILE_N = 128
    # torch.save((sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, w2, w2_scale, gemm1_out_q, gemm1_out_scale, cur_out), 'tensors_tuple.pt')
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, w2, w2_scale, gemm1_out_q, gemm1_out_scale, cur_out = torch.load('tensors_tuple.pt')

    B = gemm1_out_q.shape[0] // TOPK
    
    gate_up = False 
    if 0:
        print(gemm1_out_q.shape)
        print(gemm1_out_scale.shape, gemm1_out_scale.dtype)
        print(w2_scale.shape, w2_scale.dtype)
        
        #assert 0
        #assert 0
        # sorted_ids[TILE_M + 32] = sorted_ids[TILE_M + 33]
        #swap = sorted_ids[TILE_M:TILE_M*2].clone()
        sorted_ids[TILE_M:TILE_M*2] = sorted_ids[:TILE_M]
        sorted_weights[TILE_M:TILE_M+TILE_M] = sorted_weights[:TILE_M]
        cnnt0, cnnt1 = 32, 64
        #cnnt0, cnnt1 = 0, 32
        print(">>>>>>>>>>>>>")
        print(gemm1_out_scale[cnnt0:cnnt1])
        print(">>>>>>>>>>>>>")
        print(gemm1_out_scale[TILE_M+cnnt0:TILE_M+cnnt1])

        gemm1_out_scale[31] = 0.3

        for k in range(gemm1_out_scale.shape[0]//TILE_M):
            if not torch.all(gemm1_out_scale[k*TILE_M+cnnt0:k*TILE_M+cnnt1] == 0).item():
                print(k, gemm1_out_scale[k*TILE_M+cnnt0:k*TILE_M+cnnt1])

        #assert 0
        gemm1_out_scale[TILE_M+cnnt0:TILE_M+cnnt1] = gemm1_out_scale[cnnt0:cnnt1]
        sorted_expert_ids[1] = sorted_expert_ids[0]

        #sorted_ids[:TILE_M] = swap[...]
        
        #sorted_ids[:TILE_M] = sorted_ids[TILE_M:TILE_M*2]

    the_out = cur_out.clone()
    moe_gemm_ref(TILE_M, TILE_N, False, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
                    w2, w2_scale, 
                    #gemm1_out, None,
                    gemm1_out_q.view(B, TOPK, -1), gemm1_out_scale,
                    cur_out)

    print(">>>>>>>>>>>>>>>>>>>>>>")
    for iii in range(0,sorted_ids.shape[0],TILE_M):
        print((sorted_ids[iii:iii+TILE_M] & 0xFFFFFF).tolist())
    print(">>>>>>>>>>>>>>>>>>>>>>")

    down_OC = w2.shape[1]
    assert down_OC % TILE_N == 0
    num_oc_blocks = down_OC // TILE_N
    num_e_blocks = sorted_expert_ids.shape[0]
    print(f"{num_oc_blocks=} {num_e_blocks=}")
    moe_gemm_mxfp4([num_oc_blocks, num_e_blocks],[256],
        TILE_M, TILE_N,
        w2.shape[0], w2.shape[1], w2.shape[2], 
        False, TOPK, # gate_up,
        sorted_ids.data_ptr(),
        sorted_weights.data_ptr(),
        sorted_expert_ids.data_ptr(),
        num_valid_ids.data_ptr(),
        w2.data_ptr(), w2_scale.data_ptr(),
        gemm1_out_q.data_ptr(), gemm1_out_scale.data_ptr(),
        the_out.data_ptr(), B)
    print("=============== cur_out")
    print(cur_out[:16,:128])
    print("=============== the_out")
    print(the_out[:16,:128])
    print(f"=== {calc_diff(cur_out, the_out):.3f}")
    for ttt in range(B):
        diff = calc_diff(cur_out[ttt], the_out[ttt])
        if diff > 0.01:
            print(f"=============== {ttt} diff : {diff:.4f}")
            for oc_b in range(num_oc_blocks):
                co = cur_out[ttt, oc_b*TILE_N:oc_b*TILE_N + TILE_N]
                to = the_out[ttt, oc_b*TILE_N:oc_b*TILE_N + TILE_N]
                #print(f"        {co.view(-1,32)}")
                #print(f"        {to.view(-1,32)}")
                d = calc_diff(co, to)
                print(f"{d:.2f},", end="")
            print()
            assert 0


def test_moe_gemm_final_reduce_bf16():
    TOPK = 8
    OC = 4096
    num_tokens_total = 24000
    input = torch.randn(num_tokens_total, TOPK, OC, dtype=torch.bfloat16)
    output = torch.empty(num_tokens_total, OC, dtype=torch.bfloat16)
    num_CU = torch.cuda.get_device_properties().multi_processor_count
    num_WG = num_CU * 2
    
    num_tokens_wg = num_tokens_total // num_WG
    num_extra_tokens = num_tokens_total % num_WG
    '''
    num_big_wg = num_extra_tokens
    if wg_id < num_big_wg:
        tok0 = wg_id * (1 + num_tokens_wg) # need to do 1 more 
        tok1 = tok0 + (1 + num_tokens_wg)
    else:
        tok_base = num_big_wg * (1 + num_tokens_wg)
        tok0 = tok_base + (wg_id - num_big_wg) * num_tokens_wg
        tok1 = tok0 + num_tokens_wg
    '''

    print(num_WG, num_tokens_wg, num_extra_tokens, num_tokens_total)
    moe_gemm_final_reduce_bf16([num_WG], [64], TOPK, OC,
                               input.data_ptr(),
                               output.data_ptr(),
                               num_tokens_wg, num_extra_tokens, num_tokens_total)
    
    ref = torch.zeros(num_tokens_total, OC, dtype=torch.float)
    for i in range(num_tokens_total):
        for t in range(TOPK):
            ref[i] += input[i, t]

    ref = ref.to(torch.bfloat16)
    for i in range(num_tokens_total):
        if not torch.allclose(ref[i], output[i]):
            print(i)
            print(ref[i])
            print(output[i])
            assert 0
    
    for _ in range(10):
        with pyhip.cudaPerf(name="moe_gemm_final_reduce_bf16"):
            moe_gemm_final_reduce_bf16([num_WG], [64], TOPK, OC,
                                    input.data_ptr(),
                                    output.data_ptr(),
                                    num_tokens_wg, num_extra_tokens, num_tokens_total)
    assert 0

if __name__ == "__main__":

    #test_moe_gemm_final_reduce_bf16()
    #test_down()
    if 1:
        #test_gateup(256*8, 4096, 1536, 128, 8, TILE_M=128, TILE_N=128, test_acc=True)
        #test_gateup(256*8, 4096, 1536, 128, 8, TILE_M=256, TILE_N=128, test_acc=True)
        #test_gateup(256*8, 4096, 1536, 128, 8, TILE_M=128, TILE_N=256, test_acc=True)
        test_gateup(256*8, 4096, 1536, 128, 8, TILE_M=256, TILE_N=256, test_acc=True)

    B = 2048*32
    #test_gateup(B, 4096, 1536, 128, 8, TILE_M=128, TILE_N=128, use_8wave=use_8wave, test_acc=False)
    #test_gateup(B, 4096, 1536, 128, 8, TILE_M=256, TILE_N=128, use_8wave=use_8wave, test_acc=False)
    #test_gateup(B, 4096, 1536, 128, 8, TILE_M=128, TILE_N=256, use_8wave=use_8wave, test_acc=False)
    test_gateup(B, 4096, 1536, 128, 8, TILE_M=256, TILE_N=256, test_acc=False)
