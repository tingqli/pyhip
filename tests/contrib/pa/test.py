import pyhip
from pyhip import div_up
from pyhip.contrib.pa import *
import torch

def test():
    B = 1
    HQ = 32
    HK = 4
    S = 128

    # KV_LEN 如何切分为CU个数的整数倍,切分不均匀是很大的原因，
    #
    # B*HK=4, KV_LEN需要是20的整数倍才能保证均匀分给80个CU 45694/20=2284.7
    # 因此每个block需要完成2284~2285个token，但是每个batch的kv-len数目不同
    # 每个block可以遍历 kv_indptr 来确定自己需要负责的kv-len区间，因为 kv_indptr
    # 相对较小，遍历很快，
    KV_LEN = 40*1024
    KV_LEN = 45694

    #KV_LEN = 512
    DT = torch.bfloat16
    BLOCK_SIZE16 = 16
    # should be [32, 64]
    BLOCK_SIZE = 32
    BUF_COPY = 1
    BUF_COPY = 32

    KV_MIN_PART_SIZE = 256
    # should be [256, 512, 1024]
    KV_PART_SIZE = 256 * 4

    query = torch.randint(-2, 3, [B, HQ, S], dtype=DT)
    ITEMSIZE = 16 // query.itemsize
    seq_lens = {}
    key_caches = {}
    value_caches = {}
    block_tables = {}
    max_num_blocks_per_seq = {}
    block_num = {}
    # [N, BLOCK_SIZE, HK, S]
    for block_size in (BLOCK_SIZE16, BLOCK_SIZE):
        max_num_blocks_per_seq[block_size] = (KV_LEN + block_size - 1) // block_size
        block_num = B * max_num_blocks_per_seq[block_size]
        key_caches[block_size] = []
        value_caches[block_size] = []
        block_tables[block_size] = []
        seq_lens[block_size] = []
        for _ in range(BUF_COPY):
            if block_size == BLOCK_SIZE16:
                key_cache_shape = (block_num, HK, S // ITEMSIZE, block_size, ITEMSIZE)
            else:
                # [..., block_size // mfma M group, mfma M group, K // vec_size, mfma M, vec_size]
                # the highest 2 dimensions are for one 16x16x(4*8)mfma, `S // ITEMSIZE` is used for reduced dimension
                #   in order to match value_cache reduced dimension (4*8), the result of key*query should be also (4*8) which means each thread will have contious 8 elments.
                #   so the 5th `16` dimension and 3rd `2` come from:
                #      actual tokens 5th   3rd
                #      token 0- 3    0- 3  group0
                #      token 4- 7    0- 3  group1
                #      token 8-11    4- 7  group0
                #      token12-15    4- 7  group1
                #      token16-19    8-11  gourp0
                #      token20-23    8-11  gourp1
                #      token24-27   12-15  gourp0
                #      token28-31   12-15  gourp1
                key_cache_shape = (block_num, HK, block_size // (16*2), 2, S // ITEMSIZE, 16, ITEMSIZE)
            key_cache = torch.randint(-2, 3, key_cache_shape, dtype=DT)
            if block_size == BLOCK_SIZE16:
                value_cache_shape = (block_num, HK, block_size // ITEMSIZE, S, ITEMSIZE)
            else:
                # the highest 3 dimensions are for one 16x16x(4*8)mfma, `block_size // (4*ITEMSIZE)` is used for reduced dimension
                #   `S // 16` is for N dimension
                # [..., block_size // mfma K, M // 16, mfma K col, mfma M, vec_size]
                value_cache_shape = (block_num, HK, block_size // (4*ITEMSIZE), S // 16, 4, 16, ITEMSIZE)
            value_cache = torch.randint(-2, 3, value_cache_shape, dtype=DT)
            seq_len = torch.full(size=(B,), fill_value=KV_LEN, dtype=torch.int)
            block_table = torch.linspace(0, block_num - 1, block_num, dtype=torch.int32).reshape(B, max_num_blocks_per_seq[block_size])
            seq_lens[block_size].append(seq_len)
            key_caches[block_size].append(key_cache)
            value_caches[block_size].append(value_cache)
            block_tables[block_size].append(block_table)

    scale = 1 / (S**0.5)

    num_parts = KV_PART_SIZE//KV_MIN_PART_SIZE

    # pa hip
    max_num_parts = div_up(KV_LEN, KV_PART_SIZE)
    # -g -ggdb -O1
    my_out_seg = torch.ones([B, HQ, max_num_parts, S], dtype=DT) * 3
    my_out = torch.empty([B, HQ, S], dtype=DT)
    my_max = torch.empty([B, HQ, max_num_parts, 1], dtype=torch.float32) * 5
    my_sum = torch.empty([B, HQ, max_num_parts, 1], dtype=torch.float32) * 6

    def get_ref(block_size):
        # [block_num, HK, block_size // (16*2), 2, S // ITEMSIZE, 16, ITEMSIZE]
        #  ->[block_num, HK, block_size // (16*2), 2, 16, S // ITEMSIZE, ITEMSIZE]
        #  ->[block_num, HK, block_size // (16*2), 32, S]
        key_cache = key_caches[block_size][0].permute(0, 1, 2, 3, 5, 4, 6).reshape(-1, HK, block_size // 32, 32, S)
        #  ->[block_num, HK, block_size // (16*2), 8, 4, S]
        key_cache = key_cache.reshape(-1, HK, block_size // (16*2), 8, 4, S)
        # interleave 0, 1, ... 4, 5... to 0, 4, 1, 5 ...
        key_cache = key_cache[:,:,:,(0,4,1,5,2,6,3,7), :, :]
        #  ->[block_num, HK, block_size, S]
        key_cache = key_cache.reshape(-1, HK, block_size, S)
        #  ->[B, max_num_blocks_per_seq, HK, block_size, S] -> [B, HK, max_num_blocks_per_seq, block_size, S]
        key_cache = key_cache.reshape(B, max_num_blocks_per_seq[block_size], HK, block_size, S).permute(0, 2, 1, 3, 4)
        #  ->[B, HK, KV_LEN_pad, S]
        key_cache = key_cache.reshape(B, HK, -1, S)
        key_cache = key_cache[..., :KV_LEN, :].transpose(2, 3)
        ref_qk = query.reshape(B, HK, HQ // HK, -1) @ key_cache
        ref_qk = ref_qk.to(torch.float32)
        ref_qk = ref_qk * scale
        s = torch.softmax(ref_qk, dim=-1).to(key_cache.dtype)
        # [block_num, HK, block_size // (4*ITEMSIZE), S // 16, 4, 16, ITEMSIZE]
        #  ->[block_num, HK, block_size // (4*ITEMSIZE), 4, ITEMSIZE, S // 16, 16]
        #  ->[block_num, HK, block_size, S]
        value_cache = value_caches[block_size][0].permute(0, 1, 2, 4, 6, 3, 5).reshape(-1, HK, block_size, S)
        #  ->[B, max_num_blocks_per_seq, HK, block_size, S] -> [B, HK, max_num_blocks_per_seq, block_size, S]
        value_cache = value_cache.reshape(B, max_num_blocks_per_seq[block_size], HK, block_size, S).permute(0, 2, 1, 3, 4)
        #  ->[B, HK, KV_LEN_pad, S]
        value_cache = value_cache.reshape(B, HK, -1, S)
        value_cache = value_cache[..., :KV_LEN, :]
        ref_out = s @ value_cache
        ref_out = ref_out.reshape(B, HQ, S)
        return ref_out.to(query.dtype)

    def run_aiter(query,
                key_cache,
                value_cache,
                block_tables,
                seq_lens,
                max_num_blocks_per_seq):
        import aiter
        return aiter.pa_fwd_asm(
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            max_num_blocks_per_seq,
        )

    def test_aiter_perf():
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
        # if 0:
        #     import os.path
        #     data = torch.load(os.path.dirname(__file__) + '/pa.pt')
        #     query = data['q'].to(device=query.device)
        #     key_caches[-1][1:KV_LEN+1] = data['k']
        #     value_caches[-1][1:KV_LEN+1] = data['v']
        #     out_ref = data['out'].to(device=query.device)
        out = run_aiter(query=query, key_cache=key_caches[BLOCK_SIZE16][-1], value_cache=value_caches[BLOCK_SIZE16][-1],
                        block_tables=block_tables[BLOCK_SIZE16][-1], seq_lens=seq_lens[BLOCK_SIZE16][-1], max_num_blocks_per_seq=max_num_blocks_per_seq[BLOCK_SIZE16])

        if out_ref is not None:
            assert torch.allclose(out, out_ref), "aiter acc is wrong"
            print('aiter acc ok')
        i = 0
        for _ in range(10):
            with pyhip.cudaPerf(B * HQ // HK * KV_LEN * S * 2 * 2, B * (HK * KV_LEN * S * 2 * 2), name="aiter"):
                run_aiter(query=query, key_cache=key_caches[BLOCK_SIZE16][i], value_cache=value_caches[BLOCK_SIZE16][i],
                        block_tables=block_tables[BLOCK_SIZE16][i], seq_lens=seq_lens[BLOCK_SIZE16][i], max_num_blocks_per_seq=max_num_blocks_per_seq[BLOCK_SIZE16])
            i = (i + 1) % BUF_COPY

    def test_acc():
        print("======================= verify correctness ==============================")
        pa_jit([B, HK, max_num_parts], [256],
            HQ, HK, S, BLOCK_SIZE, KV_PART_SIZE, scale, num_parts,
                query.data_ptr(),           # [B, HQ, S]
                key_caches[BLOCK_SIZE][0].data_ptr(),   # [BLOCK_NUM, HK, S // 8, BLOCK_SIZE, 8]
                value_caches[BLOCK_SIZE][0].data_ptr(), # [BLOCK_NUM, HK, S, BLOCK_SIZE // 8, 8]
                block_tables[BLOCK_SIZE][0].data_ptr(),
                seq_lens[BLOCK_SIZE][0].data_ptr(),
                my_out_seg.data_ptr(),
                my_max.data_ptr(), 
                my_sum.data_ptr(),
                max_num_parts,
                max_num_blocks_per_seq[BLOCK_SIZE])
        pa_reduce_jit([B, HQ], [256], KV_PART_SIZE, HQ, S,
                    seq_lens[BLOCK_SIZE][0].data_ptr(), my_out_seg.data_ptr(), my_max.data_ptr(), my_sum.data_ptr(), my_out.data_ptr(), max_num_parts,
                    0)

        ref_out = get_ref(BLOCK_SIZE)
        idx = torch.where(torch.abs(ref_out - my_out) > 0.05)
        if len(idx[0]):
            print(f'idx = {idx}\nref_out={ref_out[idx]}\ncur={my_out[idx]}')

        assert torch.allclose(ref_out, my_out, rtol=0.02, atol=0.02), "out is wrong"
        print('acc ok')

    def test_perf():
        print("======================= test performance ==============================")

        i = 0
        for round in range(10):
            with pyhip.cudaPerf(B * HQ // HK * KV_LEN * S * 2 * 2, B * (HK * KV_LEN * S * 2 * 2), name="pa_jit"):
                pa_jit([B, HK, max_num_parts], [256],
                        HQ, HK, S, BLOCK_SIZE, KV_PART_SIZE, scale, num_parts,
                        query.data_ptr(),                       # [B, HQ, S]
                        key_caches[BLOCK_SIZE][i].data_ptr(),   # [BLOCK_NUM, HK, S // 8, BLOCK_SIZE, 8]
                        value_caches[BLOCK_SIZE][i].data_ptr(), # [BLOCK_NUM, HK, S, BLOCK_SIZE // 8, 8]
                        block_tables[BLOCK_SIZE][i].data_ptr(),
                        seq_lens[BLOCK_SIZE][i].data_ptr(),
                        my_out_seg.data_ptr(),
                        my_max.data_ptr(), 
                        my_sum.data_ptr(),
                        max_num_parts,
                        max_num_blocks_per_seq[BLOCK_SIZE])
                pa_reduce_jit([B, HQ], [256], KV_PART_SIZE, HQ, S,
                            seq_lens[BLOCK_SIZE][i].data_ptr(), my_out_seg.data_ptr(), my_max.data_ptr(), my_sum.data_ptr(), my_out.data_ptr(), max_num_parts,
                            0)

            i = (i + 1) % BUF_COPY
        print(f"[{B}, {HK}, {div_up(KV_LEN, KV_PART_SIZE)}]")

    test_aiter_perf()
    test_acc()
    test_perf()

if __name__ == '__main__':
    torch.set_default_device('cuda')
    torch.manual_seed(0)
    torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
    
    test()