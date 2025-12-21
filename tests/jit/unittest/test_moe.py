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

    def moe_gemm_kernel(J:pyhip.JIT, p_input, p_weight, p_output, M, weight):
        # use 16x16
        sizeof_bf16 = 2
        sizeof_f32 = 4
        # one WG per CU,  4 waves split on K, 
        lane_id = get_lane_id(J)
        lane_mod_16 = get_lane_id_mod(J, 16)
        lane_div_16 = get_lane_id_div(J, 16)


        warp_id = J.gpr("su32")
        J.v_readfirstlane_b32(warp_id, J.threadIdx.x[0] // 64)
        stride_AB = K*sizeof_bf16

        if use_split_K:
            lds_base = J.alloc_lds(64*1024)
            p_weight[:] = p_weight[:] + 32*J.blockIdx.x*stride_AB
            p_output[:] = p_output[:] + (16 if with_silu else 32)*J.blockIdx.x*sizeof_bf16
        else:
            # split N
            p_weight[:] = p_weight[:] + 32*J.blockIdx.x*stride_AB
            p_output[:] = p_output[:] + (16 if with_silu else 32)*J.blockIdx.x*sizeof_bf16
        buff_a = J.Buffer(p_input, M*K*sizeof_bf16)
        buff_b = J.Buffer(p_weight, 32*K*sizeof_bf16)

        # v_mfma_f32_16x16x16_bf16       vdst:f32x4,  vsrc0:bf16x4, vsrc1:bf16x4, src2:f32x4   cbsz abid blgp

        Creg = J.gpr(2, 4, "vf32")
        Areg = J.gpr(2, 2, 2, "vbf16x2") # 8-bf16 == DWORDx4
        Breg = J.gpr(2, 2, 2, 2, "vbf16x2") # 8-bf16 == DWORDx4

        for m in range(2):
            for i in range(4):
                Creg[m,i] = 0

        # vdst, voffset, soffset, offset12=0):
        voffset_ab = J.gpr((J.threadIdx.x % 16)*stride_AB + (J.threadIdx.x//16) * 16)
        soffset_k = J.gpr("su32")
        soffset_k[0] = 0

        # ping pong register buffer id
        pp_reg_id = 0
        buff_a.load_dwordx4(Areg[pp_reg_id], voffset_ab, soffset_k)
        buff_b.load_dwordx4(Breg[pp_reg_id, 0], voffset_ab, soffset_k)
        buff_b.load_dwordx4(Breg[pp_reg_id, 1], voffset_ab, soffset_k + (16*stride_AB))
        soffset_k[0] = soffset_k[0] + 32*(4 if use_split_K else 1)*sizeof_bf16
        pp_reg_id = pp_reg_id ^ 1

        if use_split_K:
            # split K
            k_max = div_up(K, 4*32)
            for k in range(k_max):
                # [16,32] * [2,16,32] => [2,16,16]
                if k + 1 < k_max:
                    buff_a.load_dwordx4(Areg[pp_reg_id], voffset_ab, soffset_k)
                    buff_b.load_dwordx4(Breg[pp_reg_id, 0], voffset_ab, soffset_k)
                    buff_b.load_dwordx4(Breg[pp_reg_id, 1], voffset_ab, soffset_k + (16*stride_AB))
                    pp_reg_id = pp_reg_id ^ 1
                    soffset_k[0] = soffset_k[0] + 32*4*sizeof_bf16
                    J.s_waitcnt(mod=f"vmcnt(3)")
                else:
                    pp_reg_id = pp_reg_id ^ 1
                    J.s_waitcnt(mod=f"vmcnt(0)")

                J.v_mfma_f32_16x16x16_bf16(Creg[0], Breg[pp_reg_id,0,0], Areg[pp_reg_id,0], Creg[0])
                J.v_mfma_f32_16x16x16_bf16(Creg[1], Breg[pp_reg_id,1,0], Areg[pp_reg_id,0], Creg[1])

                J.v_mfma_f32_16x16x16_bf16(Creg[0], Breg[pp_reg_id,0,1], Areg[pp_reg_id,1], Creg[0])
                J.v_mfma_f32_16x16x16_bf16(Creg[1], Breg[pp_reg_id,1,1], Areg[pp_reg_id,1], Creg[1])

                # save 2 16x16 float to LDS

            vaddr = J.gpr("vu32")
            vaddr[0] = lds_base + warp_id * (16*32*sizeof_f32) + lane_mod_16 * (32*sizeof_f32) + lane_div_16*(4*sizeof_f32)
            J.ds_write_b128(vaddr, Creg[0])
            J.ds_write_b128(vaddr, Creg[1], mod=f"offset:{16*sizeof_f32}")

            J.s_barrier()
            # each wave reduce 4 rows
            vrow = J.gpr(lane_div_16 + warp_id*4)
            vaddr[0] = lds_base + vrow * (32*sizeof_f32) + lane_mod_16*(2*sizeof_f32)
            gate_up = J.gpr(4, 2, "vf32")
            J.ds_read_b64(gate_up[0], vaddr, mod=f"offset:{0*(16*32*sizeof_f32)}") # 4x16 dword2
            J.ds_read_b64(gate_up[1], vaddr, mod=f"offset:{1*(16*32*sizeof_f32)}")
            J.ds_read_b64(gate_up[2], vaddr, mod=f"offset:{2*(16*32*sizeof_f32)}")
            J.ds_read_b64(gate_up[3], vaddr, mod=f"offset:{3*(16*32*sizeof_f32)}")

            J.s_waitcnt(mod=f"lgkmcnt(2)")
            J.v_pk_add_f32(gate_up[0], gate_up[0], gate_up[1])
            J.s_waitcnt(mod=f"lgkmcnt(0)")
            J.v_pk_add_f32(gate_up[2], gate_up[2], gate_up[3])
            J.v_pk_add_f32(gate_up[0], gate_up[0], gate_up[2])
            # apply activation
            # gate_up[0]
            if with_silu:
                # silu: x/(1+exp(-x))
                out = J.gpr("vf32")
                out[0] = gate_up[0, 0] * (-math.log2(math.exp(1)))
                J.v_exp_f32(out[0], out[0])
                out[0] = out[0] + 1.0
                J.v_rcp_f32(out[0], out[0])
                out[0] = gate_up[0, 0] * out[0]
                out[0] = gate_up[0, 1] * out[0]
                vaddr[0] = vrow * (N//2*sizeof_bf16) + lane_mod_16*(sizeof_bf16)
                with J.ExecMask(vrow < M[0]):
                    J.global_store_short_d16_hi(vaddr, out[0], p_output)
            else:
                # convert to bf16
                vaddr[0] = vrow * (N*sizeof_bf16) + lane_mod_16*(2*sizeof_bf16)
                with J.ExecMask(vrow < M[0]):
                    out = J.gpr("vf32")
                    out[0] = (gate_up[0,0]>>16)|(gate_up[0,1]&0xFFFF0000)
                    J.global_store_dword(vaddr, out[0], p_output)
        else:
            # split N
            k_max = div_up(K, 32)
            for k in range(k_max):
                # [16,32] * [2,16,32] => [2,16,16]
                if k + 1 < k_max:
                    buff_a.load_dwordx4(Areg[pp_reg_id], voffset_ab, soffset_k)
                    buff_b.load_dwordx4(Breg[pp_reg_id, 0], voffset_ab, soffset_k)
                    buff_b.load_dwordx4(Breg[pp_reg_id, 1], voffset_ab, soffset_k + (16*stride_AB))
                    pp_reg_id = pp_reg_id ^ 1
                    soffset_k[0] = soffset_k[0] + 32*sizeof_bf16
                    J.s_waitcnt(mod=f"vmcnt(3)")
                else:
                    pp_reg_id = pp_reg_id ^ 1
                    J.s_waitcnt(mod=f"vmcnt(0)")

                J.v_mfma_f32_16x16x16_bf16(Creg[0], Breg[pp_reg_id,0,0], Areg[pp_reg_id,0], Creg[0])
                J.v_mfma_f32_16x16x16_bf16(Creg[1], Breg[pp_reg_id,1,0], Areg[pp_reg_id,0], Creg[1])

                J.v_mfma_f32_16x16x16_bf16(Creg[0], Breg[pp_reg_id,0,1], Areg[pp_reg_id,1], Creg[0])
                J.v_mfma_f32_16x16x16_bf16(Creg[1], Breg[pp_reg_id,1,1], Areg[pp_reg_id,1], Creg[1])
        
            assert not with_silu
            # convert to bf16?
            vaddr = J.gpr(lane_mod_16 * (N*sizeof_bf16) + lane_div_16*(4*sizeof_bf16))
            creg_low = J.gpr(2, 2, "vbf16x2", align=4)
            J.s_waitcnt(mod=f"lgkmcnt(0)")
            for i in range(2):
                for j in range(4):
                    Creg[i, j] = Creg[i, j] * weight
            for i in range(2):
                creg_low[i,0] = (Creg[i,0]>>16)|(Creg[i,1]&0xFFFF0000)
                creg_low[i,1] = (Creg[i,2]>>16)|(Creg[i,3]&0xFFFF0000)
            with J.ExecMask(lane_mod_16 < M[0]):
                J.global_atomic_pk_add_bf16(vaddr, creg_low[0,0], p_output)
                J.global_atomic_pk_add_bf16(vaddr+4, creg_low[0,1], p_output)
                J.global_atomic_pk_add_bf16(vaddr+16*sizeof_bf16, creg_low[1,0], p_output)
                J.global_atomic_pk_add_bf16(vaddr+16*sizeof_bf16+4, creg_low[1,1], p_output)

    @pyhip.jit(kernel_suffix=f'{K=}-{N=}-SILU={with_silu}')
    def moe_gemm_batch1(J:pyhip.JIT, p_input:"void*", p_weight:"void*", p_output:"void*", p_topk_ids:"void*", p_topk_weight:"float*", M:"int"):
        # expert index(not id)
        e_idx = J.blockIdx.y
        s_e_id = J.gpr(1, 'su32')
        J.s_load_dword(s_e_id, p_topk_ids, e_idx[0] * 4)
        J.s_waitcnt(mod=f"lgkmcnt(0)")
        p_weight[:] = p_weight[:] + s_e_id * (N * K * 2)
        # p_output[:] = p_output[:] + s_e_id * M * (N * 2)
        s_weight = None
        if with_silu:
            # [E, M, N//2]
            p_output[:] = p_output[:] + e_idx * (N // 2 * 2)
        else:
            # [M, N]
            p_input[:] = p_input[:] + e_idx * (K * 2)
            s_weight = J.gpr(1, 'su32')
            J.s_load_dword(s_weight, p_topk_weight, e_idx[0] * 4)
        return moe_gemm_kernel(J, p_input=p_input, p_weight=p_weight, p_output=p_output, M=M, weight=s_weight)

    return moe_gemm_batch1, (64*4) if use_split_K else (64)

@cache
def get_reordered_weight(w:Tensor, interlave=True):
    # [E, INTER_SIZE_TP * 2, HIDDEN_SIZE]
    E, N2, K = w.shape
    N = N2 // 2
    out = torch.empty_like(w)
    # restore layout from aiter.ops.shuffle.shuffle_weight
    IN, IK = 16, 16
    BK = IK * 2
    K = 16 // w.element_size()
    BN = IN
    assert w.shape[-2] % BN == 0, f"{w.shape[-2]} % {BN} == {w.shape[-2] % BN }"
    assert w.shape[-1] % BK == 0, f"{w.shape[-1]} % {BK} == {w.shape[-1] % BK }"
    x_ = w.view(-1, w.shape[-2] // BN, w.shape[-1] // BK, BK // K, BN, K)
    x_ = x_.permute(0, 1, 4, 2, 3, 5)
    x_ = x_.contiguous()
    x_ = x_.view(*w.shape)
    if interlave:
        out[:, 0::2, :] = x_[:, :N, :]
        out[:, 1::2, :] = x_[:, N:, :]
    else:
        out[:] = x_[:]
    return out
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

    # cache weight
    for i in range(BUF_COPY):
        get_reordered_weight(w1[i], True)
        get_reordered_weight(w2[i], False)

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
                TOPK = topk_ids.shape[1]
                assert N1 == 2 * K2
                gemm1, wg_size1 = get_kernel(K1, N1, True)
                gemm2, wg_size2 = get_kernel(K2, N2, False)
                gemm1_out = torch.empty([TOPK, 1, N1 // 2], dtype=hidden_states.dtype)
                gemm2_out = torch.zeros([1, N2], dtype=hidden_states.dtype)
                w1_reordered = get_reordered_weight(w1, True)
                w2_reordered = get_reordered_weight(w2, False)
                gemm1([N1//32, TOPK],[wg_size1], hidden_states.data_ptr(), w1_reordered.data_ptr(), gemm1_out.data_ptr(), topk_ids.data_ptr(), topk_weight.data_ptr(), 1)
                gemm2([N2//32, TOPK],[wg_size2], gemm1_out.data_ptr(), w2_reordered.data_ptr(), gemm2_out.data_ptr(), topk_ids.data_ptr(), topk_weight.data_ptr(), 1)
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

        if not torch.allclose(ref_out, cur_out, rtol=0.02, atol=0.02):
            idx = torch.where(torch.abs(ref_out - cur_out) > 0.02)
            if len(idx[0]):
                print(f'idx = {idx}\nref={ref_out[idx]}\ncur={cur_out[idx]}')
            assert 0
        else:
            print("acc OK")

if __name__ == '__main__':
    test()