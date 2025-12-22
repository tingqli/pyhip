import os
import random
from typing import Optional
os.environ['PYHIP_JIT_LOG'] = '1'
import pyhip
import torch
import math
from torch import Tensor

from pyhip.asmjit import Addr2D, float_to_ieee754_bits_little
torch.cuda.set_device(2)
torch.set_default_device('cuda')
torch.manual_seed(0)
torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )

from functools import cache
#####################################################################
# kernel define
@cache
def get_lane_id(J):
    vgpr_lane_id = J.gpr(J.threadIdx.x[0] & 63)
    return vgpr_lane_id

@cache
def get_lane_id_div(J, divisor):
    assert isinstance(divisor, int)
    return J.gpr(get_lane_id(J) // divisor)

@cache
def get_lane_id_mod(J, divisor):
    assert isinstance(divisor, int)
    return J.gpr(get_lane_id(J) % divisor)

def div_up(x, y):
    return (x + y - 1) // y

"""
def shuffle_weight(x: torch.Tensor, layout=(16, 16), use_int4=False) -> torch.Tensor:
    # Hardcode BLOCK_K and BLOCK_N
    IN, IK = layout
    BK = IK * 2
    K = 16 // x.element_size() if not use_int4 else 32
    BN = IN
    assert x.shape[-2] % BN == 0, f"{x.shape[-2]} % {BN} == {x.shape[-2] % BN }"
    assert x.shape[-1] % BK == 0, f"{x.shape[-1]} % {BK} == {x.shape[-1] % BK }"

    x_ = x
    x_ = x_.view(-1, x.shape[-2] // BN, BN, x.shape[-1] // BK, BK // K, K)
    x_ = x_.permute(0, 1, 3, 4, 2, 5)
    x_ = x_.contiguous()
    x_ = x_.view(*x.shape)
    return x_

    [-1, N//16, 16n, K//32, 4k, 8k]
    [-1, N//16, K//32, 4k, 16n, 8k]
"""
@cache
def get_kernel(
        K,
        N,
        with_silu
        ):
    BLOCK_SIZE_M = 16  # nBM * 16
    BLOCK_SIZE_N = 32  # nBN * 32
    BLOCK_SIZE_K = 32
    assert K % BLOCK_SIZE_K == 0
    assert BLOCK_SIZE_M % 16 == 0
    assert BLOCK_SIZE_N % 32 == 0

    nBM = BLOCK_SIZE_M // 16
    nBN = BLOCK_SIZE_N // 32
    use_split_K = with_silu #K > 4*32
    num_split_K = 4 if use_split_K else 1

    @pyhip.jit(kernel_suffix=f'{K=}-{N=}-SILU={with_silu}')
    def moe_gemm_batch1(J:pyhip.JIT,
                        p_input:"void*",
                        p_weight:"void*",
                        p_output:"void*",
                        p_topk_ids:"void*",
                        p_topk_weight:"float*", M:"int"):
        sizeof_bf16 = 2
        sizeof_f32 = 4
        stride_AB = K*sizeof_bf16

        Creg = J.gpr(2, 4, "vf32")
        Areg = J.gpr(2, 2, 2, "vbf16x2") # 8-bf16 == DWORDx4
        Breg = J.gpr(2, 2, 2, 2, "vbf16x2") # 8-bf16 == DWORDx4

        Creg[:] = 0

        # expert index(not id)
        e_idx = J.blockIdx.y
        s_e_id = J.gpr(1, 'su32')
        J.s_load_dword(s_e_id, p_topk_ids, e_idx[0] * 4)

        # hide following initialization into s_waitcnt
        pp_reg_id = 0
        s_weight = J.gpr(1, 'su32')

        if True:
            # one WG per CU,  4 waves split on K, 
            lane_mod_16 = get_lane_id_mod(J, 16)
            lane_div_16 = get_lane_id_div(J, 16)
            warp_id = J.gpr("su32")
            J.v_readfirstlane_b32(warp_id, J.threadIdx.x[0] // 64)
            assert K % (num_split_K*32) == 0
            p_weight[:] = p_weight[:] + (16 if with_silu else 32)*J.blockIdx.x*stride_AB
            p_output[:] = p_output[:] + (16 if with_silu else 32)*J.blockIdx.x*sizeof_bf16

            p_weight1 = J.gpr(2,"su32")
            if with_silu:
                J.s_add_u32(p_weight1[0], p_weight[0], (N//2)*K*sizeof_bf16)
                J.s_addc_u32(p_weight1[1], p_weight[1], 0)
            else:
                J.s_add_u32(p_weight1[0], p_weight[0], 16*K*sizeof_bf16)
                J.s_addc_u32(p_weight1[1], p_weight[1], 0)
            voffset_a = J.gpr((J.threadIdx.x % 16)*stride_AB + (J.threadIdx.x//16) * 16)
            voffset_b = J.gpr(J.threadIdx.x * 16)

            soffset_kb = J.gpr("su32")
            soffset_ka = J.gpr("su32")
            soffset_kb[0] = 0
            soffset_ka[0] = 0

            if with_silu:
                # [E, M, N//2]
                p_output[:] = p_output[:] + e_idx * (N // 2 * sizeof_bf16)
            else:
                # [M, N]
                p_input[:] = p_input[:] + e_idx * (K * sizeof_bf16)
                J.s_load_dword(s_weight, p_topk_weight, e_idx[0] * 4)

            buff_a = J.Buffer(p_input, M*K*sizeof_bf16)
            buff_a.load_dwordx4(Areg[pp_reg_id], voffset_a, soffset_ka)

        J.s_waitcnt(mod=f"lgkmcnt(0)")
        p_weight[:] = p_weight[:] + s_e_id * (N * K * sizeof_bf16)
        p_weight1[:] = p_weight1[:] + s_e_id * (N * K * sizeof_bf16)

        buff_b0 = J.Buffer(p_weight, 16*K*sizeof_bf16)
        buff_b1 = J.Buffer(p_weight1, 16*K*sizeof_bf16)

        # ping pong register buffer id
        buff_b0.load_dwordx4(Breg[pp_reg_id, 0], voffset_b, soffset_kb)
        buff_b1.load_dwordx4(Breg[pp_reg_id, 1], voffset_b, soffset_kb)
        soffset_kb[0] = soffset_kb[0] + 16*(32*num_split_K)*sizeof_bf16
        soffset_ka[0] = soffset_ka[0] + 32*num_split_K*sizeof_bf16
        pp_reg_id = pp_reg_id ^ 1

        k_max = div_up(K, num_split_K*32)
        for k in range(k_max):
            # [16,32] * [2,16,32] => [2,16,16]
            if k + 1 < k_max:
                buff_a.load_dwordx4(Areg[pp_reg_id], voffset_a, soffset_ka)
                buff_b0.load_dwordx4(Breg[pp_reg_id, 0], voffset_b, soffset_kb)
                buff_b1.load_dwordx4(Breg[pp_reg_id, 1], voffset_b, soffset_kb)
                pp_reg_id = pp_reg_id ^ 1
                soffset_kb[0] = soffset_kb[0] + 16*(32*num_split_K)*sizeof_bf16
                soffset_ka[0] = soffset_ka[0] + 32*num_split_K*sizeof_bf16
                J.s_waitcnt(mod=f"vmcnt(3)")
            else:
                pp_reg_id = pp_reg_id ^ 1
                J.s_waitcnt(mod=f"vmcnt(0)")

            J.v_mfma_f32_16x16x16_bf16(Creg[0], Breg[pp_reg_id,0,0], Areg[pp_reg_id,0], Creg[0])
            J.v_mfma_f32_16x16x16_bf16(Creg[1], Breg[pp_reg_id,1,0], Areg[pp_reg_id,0], Creg[1])

            J.v_mfma_f32_16x16x16_bf16(Creg[0], Breg[pp_reg_id,0,1], Areg[pp_reg_id,1], Creg[0])
            J.v_mfma_f32_16x16x16_bf16(Creg[1], Breg[pp_reg_id,1,1], Areg[pp_reg_id,1], Creg[1])

        if use_split_K:
            assert with_silu
            # split K
            lds_buff = J.LDSTensor([4*16, 32], torch.float)

            vaddr = J.gpr("vu32")
            lds_buff.write("b128", Creg[0], lane_mod_16 + warp_id*16, lane_div_16*4)
            lds_buff.write("b128", Creg[1], lane_mod_16 + warp_id*16, lane_div_16*4 + 16)

            J.s_barrier()
            # each wave reduce 4 rows
            vrow = J.gpr(lane_div_16 + warp_id*4)
            gate_up = J.gpr(4, 2, "vf32")
            # interleave 
            lds_buff.read("b32", gate_up[0], vrow + 0*16, (lane_mod_16), offset1 = 16)
            lds_buff.read("b32", gate_up[1], vrow + 1*16, (lane_mod_16), offset1 = 16)
            lds_buff.read("b32", gate_up[2], vrow + 2*16, (lane_mod_16), offset1 = 16)
            lds_buff.read("b32", gate_up[3], vrow + 3*16, (lane_mod_16), offset1 = 16)

            J.s_waitcnt(mod=f"lgkmcnt(2)")
            J.v_pk_add_f32(gate_up[0], gate_up[0], gate_up[1])
            J.s_waitcnt(mod=f"lgkmcnt(0)")
            J.v_pk_add_f32(gate_up[2], gate_up[2], gate_up[3])
            J.v_pk_add_f32(gate_up[0], gate_up[0], gate_up[2])

            if with_silu:
                out = J.gpr(gate_up[0, 1] * J.silu(gate_up[0, 0]))
                vaddr = J.gpr(vrow * (N//2*sizeof_bf16) + lane_mod_16*(sizeof_bf16))
                with J.ExecMask(vrow < M[0]):
                    J.global_store_short_d16_hi(vaddr, out[0], p_output)
            else:
                # convert to bf16
                vaddr = J.gpr(vrow * (N*sizeof_bf16) + lane_mod_16*(2*sizeof_bf16))
                with J.ExecMask(vrow < M[0]):
                    out = J.gpr("vf32")
                    out[0] = (gate_up[0,0]>>16)|(gate_up[0,1]&0xFFFF0000)
                    J.global_store_dword(vaddr, out[0], p_output)
        else:
            # split N
            assert not with_silu
            vaddr = J.gpr(lane_mod_16 * (N*sizeof_bf16) + lane_div_16*(4*sizeof_bf16))
            creg_low = J.gpr(2, 2, "vbf16x2", align=4)
            J.s_waitcnt(mod=f"lgkmcnt(0)")
            for i in range(2):
                for j in range(4):
                    Creg[i, j] = Creg[i, j] * s_weight
            for i in range(2):
                creg_low[i,0] = (Creg[i,0]>>16)|(Creg[i,1]&0xFFFF0000)
                creg_low[i,1] = (Creg[i,2]>>16)|(Creg[i,3]&0xFFFF0000)
            with J.ExecMask(lane_mod_16 < M[0]):
                J.global_atomic_pk_add_bf16(vaddr, creg_low[0,0], p_output)
                J.global_atomic_pk_add_bf16(vaddr+4, creg_low[0,1], p_output)
                J.global_atomic_pk_add_bf16(vaddr+16*sizeof_bf16, creg_low[1,0], p_output)
                J.global_atomic_pk_add_bf16(vaddr+16*sizeof_bf16+4, creg_low[1,1], p_output)

    return moe_gemm_batch1, 64*num_split_K

#####################################################################

def test():
    def test_aiter(hidden_states,
                w1,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
                w2,  # [expert(local_expert:EP), dim, inter_dim]
                topk_weight,
                topk_ids,
                ):
        from aiter.fused_moe import fused_moe
        # from aiter.fused_moe_bf16_asm import asm_moe_tkw1
        return fused_moe(
            hidden_states,
            w1,
            w2,
            topk_weight,
            topk_ids,
        )

    B = 1
    HIDDEN_SIZE = 2048
    TP = 8
    INTER_SIZE = 768
    E = 128
    TOPK = 8
    INTER_SIZE_TP = INTER_SIZE // TP
    BUF_COPY = 32
    hidden_states = (torch.randn([BUF_COPY, B, HIDDEN_SIZE], dtype=torch.bfloat16) + 1)*0.001
    w_ = torch.randn([E, INTER_SIZE_TP * 2, HIDDEN_SIZE], dtype=torch.bfloat16)
    w1 = [w_.clone() for _ in range(BUF_COPY)]
    w_ = torch.randn([E, HIDDEN_SIZE, INTER_SIZE_TP], dtype=torch.bfloat16)
    w2 = [w_.clone() for _ in range(BUF_COPY)]
    topk_weight = torch.randn([BUF_COPY, B, TOPK], dtype=torch.float32)
    topk_ids = torch.ones([BUF_COPY, B, TOPK], dtype=torch.int32)
    topk_ids_base = torch.randperm(E, dtype=torch.int32)

    if 1:
        import aiter

        topk_ids[0,:,] = topk_ids_base[: TOPK]
        from aiter.ops.shuffle import shuffle_weight
        # aiter needs preshuffle weights
        w1_qt_aiter = shuffle_weight(w1[0], layout=(16, 16))
        w2_qt_aiter = shuffle_weight(w2[0], layout=(16, 16))
        ref_out = test_aiter(hidden_states=hidden_states[0], w1=w1_qt_aiter, w2=w2_qt_aiter, topk_weight=topk_weight[0], topk_ids=topk_ids[0])
        i = 0
        for _ in range(10):
            idx_start = random.randint(0, E - TOPK)
            topk_ids[i,:,] = topk_ids_base[idx_start : idx_start + TOPK]
            with pyhip.cudaPerf(B * HIDDEN_SIZE * INTER_SIZE_TP * 2, B * (HIDDEN_SIZE * INTER_SIZE_TP * 3 * TOPK), name="moe"):
                test_aiter(hidden_states=hidden_states[i], w1=w1[i], w2=w2[i], topk_weight=topk_weight[i], topk_ids=topk_ids[i])
            i = (i + 1) % BUF_COPY

    if 1:
        import aiter
        from aiter import ActivationType, QuantType, dtypes
        org_fused_moe = None
        def my_fused_moe(
            hidden_states,
            w1,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
            w2,  # [expert(local_expert:EP), dim, inter_dim]
            topk_weight,
            topk_ids,
            expert_mask: Optional[torch.tensor] = None,  # EP
            activation=ActivationType.Silu,
            quant_type=QuantType.No,
            doweight_stage1=False,
            # following for quant
            w1_scale: Optional[torch.tensor] = None,  # [expert(local_expert:EP), inter_dim, 1]
            w2_scale: Optional[torch.tensor] = None,  # [expert(local_expert:EP), model_dim, 1]
            a1_scale: Optional[torch.tensor] = None,  # [expert(local_expert:EP), 1, model_dim]
            a2_scale: Optional[torch.tensor] = None,  # [expert(local_expert:EP), 1, inter_dim]
            # following for tuning
            block_size_M=None,
            num_local_tokens: Optional[torch.tensor] = None,
            moe_sorting_dispatch_policy=0,
            dtype=None,
            # following for cktile support
            hidden_pad=0,
            intermediate_pad=0,
            bias1=None,
            bias2=None,
        ):
            # the following should be added in aiter.fused_moe.fused_moe
            if hidden_states.shape[0] == 1 and hidden_states.dtype == torch.bfloat16 and expert_mask is None and activation == ActivationType.Silu and quant_type == QuantType.No and w1.dtype == torch.bfloat16:
                N1, K1 = w1.shape[1], w1.shape[2]
                N2, K2 = w2.shape[1], w2.shape[2]
                assert N1 == 2 * K2
                gemm1, wg_size1 = get_kernel(K1, N1, True)
                gemm2, wg_size2 = get_kernel(K2, N2, False)
                gemm1_out = torch.empty([TOPK, 1, N1 // 2], dtype=hidden_states.dtype, device=hidden_states.device)
                gemm2_out = torch.zeros([1, N2], dtype=hidden_states.dtype, device=hidden_states.device)
                gemm1([N1//32, TOPK],[wg_size1], hidden_states.data_ptr(), w1.data_ptr(), gemm1_out.data_ptr(), topk_ids.data_ptr(), topk_weight.data_ptr(), 1)
                gemm2([N2//32, TOPK],[wg_size2], gemm1_out.data_ptr(), w2.data_ptr(), gemm2_out.data_ptr(), topk_ids.data_ptr(), topk_weight.data_ptr(), 1)
                return gemm2_out
            else:
                return org_fused_moe(hidden_states,
                    w1,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
                    w2,  # [expert(local_expert:EP), dim, inter_dim]
                    topk_weight,
                    topk_ids,
                    expert_mask,  # EP
                    activation,
                    quant_type,
                    doweight_stage1,
                    # following for quant
                    w1_scale,  # [expert(local_expert:EP), inter_dim, 1]
                    w2_scale,  # [expert(local_expert:EP), model_dim, 1]
                    a1_scale,  # [expert(local_expert:EP), 1, model_dim]
                    a2_scale,  # [expert(local_expert:EP), 1, inter_dim]
                    # following for tuning
                    block_size_M,
                    num_local_tokens,
                    moe_sorting_dispatch_policy,
                    dtype,
                    # following for cktile support
                    hidden_pad,
                    intermediate_pad,
                    bias1,
                    bias2)
        org_fused_moe = aiter.fused_moe.fused_moe
        aiter.fused_moe.fused_moe = my_fused_moe
        print(f'\nprofile new moe...')

        topk_ids[0,:,] = topk_ids_base[: TOPK]
        w1_qt_aiter = shuffle_weight(w1[0], layout=(16, 16))
        w2_qt_aiter = shuffle_weight(w2[0], layout=(16, 16))
        cur_out = test_aiter(hidden_states=hidden_states[0], w1=w1_qt_aiter, w2=w2_qt_aiter, topk_weight=topk_weight[0], topk_ids=topk_ids[0])
        i = 0
        for _ in range(10):
            idx_start = random.randint(0, E - TOPK)
            topk_ids[i,:,] = topk_ids_base[idx_start : idx_start + TOPK]
            with pyhip.cudaPerf(B * HIDDEN_SIZE * INTER_SIZE_TP * 2, B * (HIDDEN_SIZE * INTER_SIZE_TP * 3 * TOPK), name="moe"):
                test_aiter(hidden_states=hidden_states[i], w1=w1[i], w2=w2[i], topk_weight=topk_weight[i], topk_ids=topk_ids[i])
            i = (i + 1) % BUF_COPY

        print(ref_out.shape, cur_out.shape)
        if not torch.allclose(ref_out, cur_out, rtol=0.02, atol=0.02):
            idx = torch.where(torch.abs(ref_out - cur_out) > 0.02)
            if len(idx[0]):
                print(f'idx = {idx}\nref={ref_out[idx]}\ncur={cur_out[idx]}\n{len(idx[0])}')
            assert 0
        else:
            print("acc OK")

if __name__ == '__main__':
    test()