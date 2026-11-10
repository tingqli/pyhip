"""xGMI P2P microbenchmarks on 8x MI300X (all ranks at once, MORI symmetric heap peers).

    torchrun --nproc_per_node=8 tests/contrib/moe/ep8/p2p_microbench.py

put:    dispatch-like FP8 rows (4096 B) or combine-like BF16 rows (8192 B) stored into the peers, row r -> peer (me+r)%8.
atomic: route-level combine with global_atomic_pk_add_bf16 into the peers' [4096, 4096] BF16 output.
"""

import os
import statistics

import torch
import torch.distributed as dist

import mori
import mori.shmem as ms
import pyhip

HERE = os.path.dirname(os.path.abspath(__file__))


@pyhip.module(os.path.join(HERE, "p2p_microbench.cpp"))
def p2p_put(src, peer_base, rows, row_u4, npes, me, sc): ...


@pyhip.module(os.path.join(HERE, "p2p_microbench.cpp"))
def p2p_atomic_bf16(src, peer_base, rows, row_u32, npes, me, dst_rows): ...


def peer_table(t, rank, world):
    return torch.tensor([ms.shmem_ptr_p2p(t.data_ptr(), rank, p) for p in range(world)], dtype=torch.int64, device="cuda")


def timed(fn, iters=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    out = []
    for _ in range(iters):
        dist.barrier()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) * 1e3)
    return statistics.median(out)


def main():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("cpu:gloo,cuda:nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
    import torch._C._distributed_c10d as c10d

    c10d._register_process_group("default", dist.group.WORLD)
    ms.shmem_torch_process_group_init("default")

    cases = [("dispatch put 4KB rows", 21632, 4096), ("combine put 8KB rows", 32640, 8192)]
    recv = mori.shmem.mori_shmem_create_tensor((32640 * 8192,), torch.uint8)
    table = peer_table(recv, rank, world)
    results = []
    for name, rows, row_bytes in cases:
        src = torch.randint(0, 255, (rows * row_bytes,), dtype=torch.uint8, device="cuda")
        for sc in (0, 1):
            for wgs in (16, 32, 64, 128, 304):
                us = timed(lambda: p2p_put([wgs, 1, 1], [256, 1, 1], src.data_ptr(), table.data_ptr(), rows, row_bytes // 16, world, rank, sc))
                results.append((f"{name} nt={sc}", wgs, us, rows * row_bytes / us / 1e3))
    out = mori.shmem.mori_shmem_create_tensor((4096 * 4096,), torch.bfloat16)
    out.zero_()
    otable = peer_table(out, rank, world)
    rows = 32640
    src = (torch.randn((rows, 4096), device="cuda") * 0.01).to(torch.bfloat16)
    for wgs in (64, 128, 304, 1216):
        us = timed(lambda: p2p_atomic_bf16([wgs, 1, 1], [256, 1, 1], src.data_ptr(), otable.data_ptr(), rows, 2048, world, rank, 4096), iters=5)
        results.append(("combine atomic pk_add_bf16 8KB rows", wgs, us, rows * 8192 / us / 1e3))
    # local reference: same atomics into this rank's own buffer only
    local = torch.tensor([out.data_ptr()] * world, dtype=torch.int64, device="cuda")
    for wgs in (304, 1216):
        us = timed(lambda: p2p_atomic_bf16([wgs, 1, 1], [256, 1, 1], src.data_ptr(), local.data_ptr(), rows, 2048, world, rank, 4096), iters=5)
        results.append(("local atomic pk_add_bf16 8KB rows", wgs, us, rows * 8192 / us / 1e3))
    gathered = [None] * world
    dist.all_gather_object(gathered, results)
    if rank == 0:
        print(f"{'case':42s} {'WGs':>5s} {'max us':>9s} {'min GB/s/GPU':>12s}")
        for i, (name, wgs, _, _) in enumerate(results):
            us = max(g[i][2] for g in gathered)
            gbs = min(g[i][3] for g in gathered)
            print(f"{name:42s} {wgs:5d} {us:9.1f} {gbs:12.1f}")
    dist.barrier()
    ms.shmem_finalize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
