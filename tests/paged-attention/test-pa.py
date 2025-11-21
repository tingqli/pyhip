import torch
import aiter
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
BUF_COPY = 1
BUF_COPY = 32
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
KV_PART_SIZE = 256
# -g -ggdb -O1
hip = pyhip.module("pa.cpp", f"-D{HQ=} -D{HK=} -D{S=} -D{BLOCK_SIZE=} -DSCALE={scale} -D{KV_PART_SIZE=} -D{FAKE_Q=} -D{FAKE_K_IDX=}")
pa = hip.pa
pa_reduce = hip.pa_reduce
my_out_seg = torch.ones([B, HQ, div_up(KV_LEN, KV_PART_SIZE), S], dtype=DT) * 3
my_out = torch.empty([B, HQ, S], dtype=DT)
my_max = torch.empty([B, HQ, div_up(KV_LEN, KV_PART_SIZE), 1], dtype=torch.float32) * 5
my_sum = torch.empty([B, HQ, div_up(KV_LEN, KV_PART_SIZE), 1], dtype=torch.float32) * 6
qk_out = torch.ones([B, HK, HQ // HK, div_up(KV_LEN, KV_PART_SIZE) * KV_PART_SIZE], dtype=torch.float32)
pa([B, HK, div_up(KV_LEN, KV_PART_SIZE)], [256], query.data_ptr(), key_caches[-1].data_ptr(), value_caches[-1].data_ptr(), kv_indptrs[-1].data_ptr(), kv_page_indices_[-1].data_ptr(), my_out_seg.data_ptr(), qk_out.data_ptr(), my_max.data_ptr(), my_sum.data_ptr())
pa_reduce([B, HQ], [256], kv_indptrs[-1].data_ptr(), my_out_seg.data_ptr(), my_max.data_ptr(), my_sum.data_ptr(), my_out.data_ptr(), div_up(KV_LEN, KV_PART_SIZE))
#assert torch.allclose(out, my_out), "pa acc is wrong"
print('pa acc ok')
# check q*k
if 1:
    ref_qk = query.reshape(B, HK, HQ // HK, -1) @ key_caches[-1][1:KV_LEN*B +1].reshape(B, -1, HK, S).permute(0, 2, 3, 1)
    ref_qk = ref_qk.to(torch.float32)
    qk_out = qk_out[..., :KV_LEN]
    ref_qk = ref_qk * scale
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
        pa_reduce([B, HQ], [256], kv_indptrs[i].data_ptr(), my_out_seg.data_ptr(), my_max.data_ptr(), my_sum.data_ptr(), my_out.data_ptr(), div_up(KV_LEN, KV_PART_SIZE))
    i = (i + 1) % BUF_COPY
print('done')