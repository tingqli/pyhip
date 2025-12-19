import pyhip

import math
import torch
torch.set_printoptions(linewidth=300)
torch.cuda.set_device(5)
torch.set_default_device('cuda')
torch.manual_seed(0)

from functools import cache

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

    @pyhip.jit()
    def kernel(J, p_input:"void*", p_weight:"void*", p_output:"void*", M:"int"):
        
        # use 16x16
        sizeof_bf16 = 2
        sizeof_f32 = 4
        # one WG per CU,  4 waves split on K, 
        lane_id = get_lane_id(J)
        lane_mod_16 = get_lane_id_mod(J, 16)
        lane_div_16 = get_lane_id_div(J, 16)

        lds_base = J.alloc_lds(64*1024)

        warp_id = J.gpr("su32")
        J.v_readfirstlane_b32(warp_id, J.threadIdx.x[0] // 64)
        stride_AB = K*sizeof_bf16

        p_weight[:] = p_weight[:] + 32*J.blockIdx.x*stride_AB
        p_output[:] = p_output[:] + (16 if with_silu else 32)*J.blockIdx.x*sizeof_f32
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
        soffset_k[0] = soffset_k[0] + 32*4*sizeof_bf16
        pp_reg_id = pp_reg_id ^ 1

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
            vaddr[0] = vrow * (N//2*sizeof_f32) + lane_mod_16*(sizeof_f32)
            with J.ExecMask(vrow < M[0]):
                J.global_store_dword(vaddr, out[0], p_output)
        else:
            # convert to bf16?
            vaddr[0] = vrow * (N*sizeof_f32) + lane_mod_16*(2*sizeof_f32)
            with J.ExecMask(vrow < M[0]):
                J.global_store_dwordx2(vaddr, gate_up[0], p_output)

    return kernel

with_silu = 1

M,K,N = 15,2048,192*8
#M,K,N = 1,96,2048
gateup_kernel = get_kernel(K, N, with_silu)

input = (torch.randn(M, K, dtype=torch.bfloat16) + 1)*0.01
w_gate_up = torch.randn(N, K, dtype=torch.bfloat16)
output_res = torch.zeros(M, N//2 if with_silu else N, dtype=torch.float)

input[1:,:] = input[0,:]

layers = []
layer_bytes = input.numel()*2 + w_gate_up.numel()*2
total_bytes = 0
while total_bytes < 2e9:
    layers.append([input.clone(), w_gate_up.clone(), output_res.clone()])
    total_bytes += layer_bytes

for i in range(10):
    di = i % len(layers)
    input, weight, _ = layers[di]
    with pyhip.cudaPerf(M*N*K*2, name=f"torch-linear-{di}"):
        if with_silu:
            gate_up = torch.nn.functional.linear(input, weight)
            gate = gate_up[:,0::2]
            up = gate_up[:,1::2]
            output_ref = torch.nn.functional.silu(gate) * up
        else:
            output_ref = torch.nn.functional.linear(input, weight)
    # flat_store_dword(v[2:3], v0)

assert N % 32 == 0
for i in range(10,20):
    di = i % len(layers)
    input, weight, res = layers[di]
    with pyhip.cudaPerf(M*N*K*2, name=f"gateup_kernel-{di}"):
        gateup_kernel([N//32],[256], input.data_ptr(), weight.data_ptr(), res.data_ptr(), M)

ref_out = output_ref.to(torch.float)
if not torch.allclose(ref_out, res, rtol=0.02, atol=0.02):
    print(ref_out[:,0:4])
    print(res[:,0:4])
    idx = torch.where(torch.abs(ref_out - res) > 0.02)
    if len(idx[0]):
        print(f'idx = {idx}\nref={ref_out[idx]}\ncur={res[idx]}')
    assert 0
else:
    print("acc OK")
# jit = pyhip.JIT()
# kernel(jit)
# kernel = jit.build("test(int* p, int K)")


#torch.cuda.synchronize()
