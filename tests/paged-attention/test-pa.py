import torch
import pyhip

torch.cuda.set_device(4)
torch.set_default_device('cuda')

B = 1
HQ = 32
HK = 4
S = 128
KV_LEN = 45694
#KV_LEN = 512
DT = torch.bfloat16
BLOCK_SIZE = 1
BLOCK_NUM = B * KV_LEN + 1000
FAKE_Q = 0
FAKE_K_IDX = 0
OUTPUT_QK = 0
BUF_COPY = 1
BUF_COPY = 32
KV_PART_SIZE = 256 * 4
USE_REDUCE_JIT = False

######################################################################
if USE_REDUCE_JIT:
    @pyhip.jit()
    def pa_reduce_jit(J, kv_indptr:"uint*",
                        out_seg:"__bf16*",
                        max_out:"float*",
                        sum_out:"float*",
                        out:"__bf16*",
                        max_part:"int",
                        checks:"float*"):
        # asm volatile(";xxx  %0  %1  %2"::"s"(blockIdx.x),"s"(blockIdx.y),"s"(blockIdx.z));
        # 上面的hip代码诱导编译器告诉我们blockIdx存放位置是s2/3/4
        b = J.blockIdx.x
        hq = J.blockIdx.y
        lane_id = J.gpr(J.threadIdx.x[0] % 64)
        warp_id = J.gpr(J.threadIdx.x[0] // 64)
        s_warp_id = J.gpr("su32")
        J.v_readfirstlane_b32(s_warp_id, warp_id)

        # 每个WG处理一个batch的一个head，
        offset1 = J.gpr(b * HQ * max_part + hq * max_part)
        offset4 = J.gpr(offset1 * 4)

        # 支持2xsgpr和sgpr的运算？
        # 支持指令参数表达式？可以节省代码函数和手工分配临时变量寄存器？
        J.s_add_u32(max_out[0], max_out[0], offset4)
        J.s_addc_u32(max_out[1], max_out[1], 0)

        kv_inds = J.gpr("si32x2",align=2)
        J.s_load_dwordx2(kv_inds, kv_indptr, 0)
        J.s_waitcnt(mod=f"lgkmcnt(0)") # 这类的wait应该可以自动生成，在第一次使用load指令结果的地方？
        kv_len = J.gpr(kv_inds[1] - kv_inds[0])
        part_num = J.gpr((kv_len[0] + (KV_PART_SIZE - 1))//KV_PART_SIZE)

        # 每个wave都独立的把最大值求出来，这一步其实可以WG内的wave协作分工来求
        # 最后来一次跨wave的reduce
        real_max = J.gpr("vf32")
        real_max[0] = torch.finfo(torch.float).min
        vi = J.auto_gpr(lane_id[0])
        with J.While() as loop:
            with J.ExecMask(vi < part_num):
                vdst = J.gpr("vf32")
                J.global_load_dword(vdst, vi[0] << 2, max_out)
                J.s_waitcnt(mod=f"vmcnt(0)")
                J.v_max_f32(real_max, real_max, vdst)
            J.s_cbranch_scc0(mod=loop["end"])# 没有非零的exec-mask，退出loop
            vi[0] = vi + 64

        # per-wave cross-lane reduce
        real_max = J.reduce("v_max_f32", real_max)

        cur_val = J.gpr("vf32x2")
        cur_val[0] = 0
        cur_val[1] = 0
        warp_sum = J.gpr("vf32")
        warp_sum[0] = 0

        # N个wave协作加权part_num这么多个token，但是part_num可能不能被N整除，
        # 因此循环次数每个wave的都不一样，但是一个wave内部的所有threads/lanes循环次数一样
        # 因此无需ExecMask

        J.s_add_u32(out_seg[0], out_seg[0], offset1*(S*2))
        J.s_addc_u32(out_seg[1], out_seg[1], 0)
        
        J.s_add_u32(sum_out[0], sum_out[0], offset4)
        J.s_addc_u32(sum_out[1], sum_out[1], 0)

        J.s_add_u32(out[0], out[0], (b * HQ * S + hq * S)*2)
        J.s_addc_u32(out[1], out[1], 0)

        J.s_waitcnt(mod=f"lgkmcnt(0)")    
        si = J.gpr("si32")
        J.v_readfirstlane_b32(si, warp_id)
        with J.While(si < part_num) as loop:
            cur_val_low = J.gpr("vf32x2") # bf16x2 => f32x2
            vaddr = J.gpr("vi32")

            soff = J.gpr(si*(2*S))
            vaddr[0] = lane_id*4 + soff

            cur_max = J.gpr("sf32")
            cur_sum = J.gpr("sf32")
            soff = J.gpr(si << 2)
            J.s_load_dword(cur_max, max_out, soff)
            J.s_load_dword(cur_sum, sum_out, soff)
            assert S % 64 == 0
            assert S == 128
            J.global_load_dword(cur_val_low[0], vaddr, out_seg)
            J.s_waitcnt(mod=f"vmcnt(0) lgkmcnt(0)")

            # bf16x2 => f32x2
            cur_val_low[1] = cur_val_low[0] & 0xffff0000
            cur_val_low[0] = cur_val_low[0] << 16
            
            # SALU do not support float
            import math
            alpha = math.log2(math.exp(1))

            exp_d_max = J.gpr("vf32")
            J.v_exp_f32_e32(exp_d_max, (cur_max[0] - real_max[0])*alpha)
            exp_d_max[0] = cur_sum[0] * exp_d_max

            cur_val[0] = cur_val[0] + cur_val_low[0] * exp_d_max[0]
            cur_val[1] = cur_val[1] + cur_val_low[1] * exp_d_max[0]

            warp_sum[0] = warp_sum[0] + exp_d_max[0]

            si[0] = si + 4

        sum_lds = J.alloc_lds(4*4)   # __shared__ float sum_lds[4];
        out_lds = J.alloc_lds(4*4*S) # __shared__ float out_lds[4 * S];

        vaddr = J.gpr("vi32")
        vaddr[0] = sum_lds + (s_warp_id<<2)
        J.ds_write_b32(vaddr, warp_sum)
        vaddr[0] = (S*4)
        vaddr[0] = out_lds + (s_warp_id*vaddr[0] + lane_id*(2*4))
        J.ds_write_b64(vaddr, cur_val)
        J.s_waitcnt(mod=f"lgkmcnt(0)")

        J.s_barrier()

        J.Jump("exit_final", s_warp_id[0] != 0)

        other_sum = J.gpr("vf32x3")
        vaddr[0] = sum_lds
        J.ds_read_b32(other_sum[0], vaddr, mod=f"offset:4")
        J.ds_read_b32(other_sum[1], vaddr, mod=f"offset:8")
        J.ds_read_b32(other_sum[2], vaddr, mod=f"offset:12")
        J.s_waitcnt(mod=f"lgkmcnt(0)")
        warp_sum[0] = warp_sum[0] + other_sum[0] + other_sum[1] + other_sum[2]

        other_val = J.gpr("vf32x2")
        for i in range(1,4):
            vaddr[0] = out_lds + (i*S*4 + lane_id*(2*4))
            J.ds_read_b64(other_val, vaddr)
            J.s_waitcnt(mod=f"lgkmcnt(0)")
            J.v_pk_add_f32(cur_val, cur_val, other_val)

        inv_sum_scale = J.gpr("vf32x1")

        J.v_rcp_f32(inv_sum_scale, warp_sum[0] + 1e-6)
        J.v_mul_f32(cur_val[0], cur_val[0], inv_sum_scale)
        J.v_mul_f32(cur_val[1], cur_val[1], inv_sum_scale)

        bf16x2 = J.gpr((cur_val[1] & 0xffff0000) | (cur_val[0] >> 16))

        J.global_store_dword(lane_id<<2, bf16x2, out)
        J.s_waitcnt(mod=f"vmcnt(0)")

        if 0:
            J.Jump("skip", (b[0] == 0) & (hq[0] == 0) & (s_warp_id[0] == 0), reverse=True)
            checks = J.gpr("su32x2",align=2)
            vaddr = J.gpr(lane_id[0]<<2)
            J.global_store_dword(vaddr, bf16x2, checks)
            J.s_waitcnt(mod=f"vmcnt(0)")
            J.Label("skip")
            J.s_nop(0)

        J.Label("exit_final")
######################################################################

print(f'kvcache = {B * (HK * KV_LEN * S * 2 * 2) // 1024 // 1024:,} MB')
workspace_buffer = torch.empty(
                (512 * HQ * 256 * S)                                            #(max_bs * self.num_head * self.max_num_partitions * self.head_dim)
                * 4                                                             #* nbyes_per_qo_elem
                + 2 * (512 * HQ * 256) * 4,                                     # + 2 * (max_bs * self.num_head * self.max_num_partitions) * 4,
                dtype=torch.uint8,
            )
# [B, H, S]
if FAKE_Q:
    query = torch.ones(B, HQ, S, dtype=DT)
else:
    query = torch.randint(-2, 3, [B, HQ, S], dtype=DT)
key_caches = []
value_caches = []
kv_indptrs = []
kv_page_indices_ = []
kv_last_page_lens_ = []
# [N, BLOCK_SIZE, HK, S]
for _ in range(BUF_COPY):
    key_cache = torch.randint(-2, 3, [BLOCK_NUM, BLOCK_SIZE, HK, S], dtype=DT)
    value_cache = torch.randint(-2, 3, [BLOCK_NUM, BLOCK_SIZE, HK, S], dtype=DT)
    batch_start = [0] * (B + 1)
    for b in range(B):
        batch_start[b + 1] = (b + 1) * KV_LEN
    kv_indptr = torch.tensor(batch_start, dtype=torch.int32)
    kv_page_indices = torch.linspace(1, KV_LEN * B, KV_LEN * B, dtype=torch.int32)
    kv_last_page_lens = torch.ones([KV_LEN], dtype=torch.int32)
    key_caches.append(key_cache)
    value_caches.append(value_cache)
    kv_indptrs.append(kv_indptr)
    kv_page_indices_.append(kv_page_indices)
    kv_last_page_lens_.append(kv_last_page_lens)
scale = 1 / (S**0.5)

def test_aiter(query,
               key_cache,
               value_cache,
               scale,
               kv_indptr,
               kv_page_indices,
               kv_last_page_lens,
               block_size=BLOCK_SIZE,
               max_num_partitions=256,
               alibi_slopes=None,
               kv_cache_dtype='auto',
               kv_cache_layout='NHD',
               logits_soft_cap=0.0,
               k_scale=torch.tensor([1.0], dtype=torch.float32),
               v_scale=torch.tensor([1.0], dtype=torch.float32),
               fp8_out_scale=None,
               partition_size=256,
               mtp=1,):
    out = torch.empty([B, HQ, S], dtype=DT)
    torch.ops.aiter.paged_attention_ragged(
        out,
        workspace_buffer,
        query,
        key_cache,
        value_cache,
        scale,
        kv_indptr,
        kv_page_indices,
        kv_last_page_lens,
        block_size,
        max_num_partitions,
        alibi_slopes,
        kv_cache_dtype,
        kv_cache_layout,
        logits_soft_cap,
        k_scale,
        v_scale,
        fp8_out_scale,
        partition_size,
        mtp,
    )
    return out

if 0:
    import aiter
    out_ref = None
    # if 0:
    #     torch.save({
    #         "q": query,
    #         "k": key_cache[1:45694+1],
    #         "v": value_cache[1:45694+1],
    #         "kv_page_indices": kv_page_indices,
    #         "out": out
    #     }, "/mywork/users/luocheng/sglang/mytest/pa.pt")
    if 0:
        import os.path
        data = torch.load(os.path.dirname(__file__) + '/pa.pt')
        query = data['q'].to(device=query.device)
        key_caches[-1][1:KV_LEN+1] = data['k']
        value_caches[-1][1:KV_LEN+1] = data['v']
        kv_page_indices_[-1][:] = data['kv_page_indices']
        out_ref = data['out'].to(device=query.device)
    out = test_aiter(query=query, key_cache=key_caches[-1], value_cache=value_caches[-1], scale=scale, 
                    kv_indptr=kv_indptrs[-1], kv_page_indices=kv_page_indices_[-1], kv_last_page_lens=kv_last_page_lens_[-1])

    if out_ref is not None:
        assert torch.allclose(out, out_ref), "aiter acc is wrong"
        print('aiter acc ok')
    i = 0
    for _ in range(10):
        with pyhip.cudaPerf(B * HQ // HK * KV_LEN * S * 2 * 2, B * (HK * KV_LEN * S * 2 * 2), name="aiter"):
            test_aiter(query=query, key_cache=key_caches[i], value_cache=value_caches[i], scale=scale, 
                    kv_indptr=kv_indptrs[i], kv_page_indices=kv_page_indices_[i], kv_last_page_lens=kv_last_page_lens_[i])
        i = (i + 1) % BUF_COPY

####################################################################
# pa hip
def div_up(x, y):
    return (x + y - 1) // y
# -g -ggdb -O1
hip = pyhip.module("pa.cpp", f"-D{HQ=} -D{HK=} -D{S=} -D{BLOCK_SIZE=} -DSCALE={scale} -D{KV_PART_SIZE=} -D{FAKE_Q=} -D{FAKE_K_IDX=} -D{OUTPUT_QK=}")
pa = hip.pa
pa_reduce = hip.pa_reduce
my_out_seg = torch.ones([B, HQ, div_up(KV_LEN, KV_PART_SIZE), S], dtype=DT) * 3
my_out = torch.empty([B, HQ, S], dtype=DT)
my_max = torch.empty([B, HQ, div_up(KV_LEN, KV_PART_SIZE), 1], dtype=torch.float32) * 5
my_sum = torch.empty([B, HQ, div_up(KV_LEN, KV_PART_SIZE), 1], dtype=torch.float32) * 6
qk_out = torch.ones([B, HK, HQ // HK, div_up(KV_LEN, KV_PART_SIZE) * KV_PART_SIZE], dtype=torch.float32)
pa([B, HK, div_up(KV_LEN, KV_PART_SIZE)], [256], query.data_ptr(), key_caches[-1].data_ptr(), value_caches[-1].data_ptr(), kv_indptrs[-1].data_ptr(), kv_page_indices_[-1].data_ptr(), my_out_seg.data_ptr(), qk_out.data_ptr(), my_max.data_ptr(), my_sum.data_ptr())
if not USE_REDUCE_JIT:
    pa_reduce([B, HQ], [256], kv_indptrs[-1].data_ptr(), my_out_seg.data_ptr(), my_max.data_ptr(), my_sum.data_ptr(), my_out.data_ptr(), div_up(KV_LEN, KV_PART_SIZE))
else:
    checks0 = torch.empty([64,], dtype=torch.float32)
    checks1 = torch.empty([64,], dtype=torch.float32)
    pa_reduce_jit([B, HQ], [256], 
                kv_indptrs[-1].data_ptr(), my_out_seg.data_ptr(), my_max.data_ptr(), my_sum.data_ptr(), my_out.data_ptr(), div_up(KV_LEN, KV_PART_SIZE),
                checks1.data_ptr())
#assert torch.allclose(out, my_out), "pa acc is wrong"
print('pa acc ok')
# check q*k
if 1:
    ref_qk = query.reshape(B, HK, HQ // HK, -1) @ key_caches[-1][1:KV_LEN*B +1].reshape(B, -1, HK, S).permute(0, 2, 3, 1)
    ref_qk = ref_qk.to(torch.float32)
    ref_qk = ref_qk * scale
    if OUTPUT_QK:
        qk_out = qk_out[..., :KV_LEN]
        idx = torch.where(torch.abs(ref_qk - qk_out) > 1)
        if len(idx[0]):
            print(f'idx = {idx}\nref_qk={ref_qk[idx]}\ncur={qk_out[idx]}')
        assert torch.allclose(ref_qk.to(torch.float32), qk_out, rtol=0.01, atol=0.01), "pa qk is wrong"
    s = torch.softmax(ref_qk, dim=-1).to(value_cache.dtype)
    ref_out = s @ value_caches[-1][1:KV_LEN*B+1].reshape(B, -1, HK, S).permute(0, 2, 1, 3)
    ref_out = ref_out.reshape(B, HQ, S)
    cur_out = my_out # my_out_seg[:,:,0,:]
    idx = torch.where(torch.abs(ref_out - cur_out) > 0.05)
    if len(idx[0]):
        print(f'idx = {idx}\nref_out={ref_out[idx]}\ncur={cur_out[idx]}')
    assert torch.allclose(ref_out, cur_out, rtol=0.01, atol=0.01), "pa out is wrong"

i = 0
for _ in range(10):
    with pyhip.cudaPerf(B * HQ // HK * KV_LEN * S * 2 * 2, B * (HK * KV_LEN * S * 2 * 2), name="pa"):
        pa([B, HK, div_up(KV_LEN, KV_PART_SIZE)], [256], query.data_ptr(), key_caches[i].data_ptr(), value_caches[i].data_ptr(), kv_indptrs[i].data_ptr(), kv_page_indices_[i].data_ptr(), my_out_seg.data_ptr(), 0, my_max.data_ptr(), my_sum.data_ptr())
        if not USE_REDUCE_JIT:
            pa_reduce([B, HQ], [256], kv_indptrs[i].data_ptr(), my_out_seg.data_ptr(), my_max.data_ptr(), my_sum.data_ptr(), my_out.data_ptr(), div_up(KV_LEN, KV_PART_SIZE))
        else:
            pa_reduce_jit([B, HQ], [256], kv_indptrs[i].data_ptr(), my_out_seg.data_ptr(), my_max.data_ptr(), my_sum.data_ptr(), my_out.data_ptr(), div_up(KV_LEN, KV_PART_SIZE))
    i = (i + 1) % BUF_COPY
print('done')