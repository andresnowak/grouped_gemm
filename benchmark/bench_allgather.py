"""Benchmark all_gather: compare NCCL (GPU tensors) vs Gloo (CPU tensors).

Usage:
    torchrun --nproc_per_node=2 grouped_gemm/bench_allgather.py
    torchrun --nproc_per_node=4 grouped_gemm/bench_allgather.py --dtype float32
"""

import torch
import torch.distributed as dist
import argparse
import time

WARMUP = 10
ITERS = 50

# Tensor sizes to benchmark (number of elements)
DEFAULT_SIZES = [
    1024,              # 2 KB bf16
    64 * 1024,         # 128 KB
    256 * 1024,        # 512 KB
    1024 * 1024,       # 2 MB
    4 * 1024 * 1024,   # 8 MB
    16 * 1024 * 1024,  # 32 MB
    64 * 1024 * 1024,  # 128 MB
    64 * 2048 * 2048 * 3,  # 128 MB
]


def bench_allgather_gpu(tensor, world_size, iters):
    """NCCL: all_gather on GPU tensors, timed with CUDA events."""
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]

    for _ in range(WARMUP):
        dist.all_gather(gathered, tensor)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        dist.all_gather(gathered, tensor)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0


def bench_allgather_cpu(tensor, world_size, iters):
    """Gloo: all_gather on CPU tensors, timed with perf_counter."""
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]

    for _ in range(WARMUP):
        dist.all_gather(gathered, tensor)

    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_gather(gathered, tensor)
    return time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark all_gather: NCCL (GPU) vs Gloo (CPU)")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--sizes", type=int, nargs="*", default=None,
                        help="Override default tensor sizes (in number of elements)")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    sizes = args.sizes if args.sizes else DEFAULT_SIZES
    iters = args.iters
    elem_bytes = torch.tensor([], dtype=dtype).element_size()

    # --- NCCL (GPU) ---
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    nccl_results = {}
    if rank == 0:
        print(f"\n{'=' * 80}")
        print(f"  NCCL all_gather (GPU tensors)")
        print(f"  world_size={world_size}  dtype={dtype}  iters={iters}")
        print(f"{'=' * 80}")

    for numel in sizes:
        nbytes = numel * elem_bytes
        tensor = torch.randn(numel, dtype=dtype, device="cuda")
        elapsed = bench_allgather_gpu(tensor, world_size, iters)
        avg_us = elapsed / iters * 1e6
        alg_bw = (nbytes * iters) / elapsed / 1e9
        nccl_results[numel] = {"avg_us": avg_us, "alg_bw": alg_bw, "nbytes": nbytes}

        if rank == 0:
            kb = nbytes / 1024
            tag = f"{kb:.1f} KB" if kb < 1024 else f"{kb / 1024:.1f} MB"
            print(f"  {tag:>12s} ({numel:>12,d} elems)  "
                  f"latency: {avg_us:>10.2f} us  algBW: {alg_bw:>8.2f} GB/s")

    dist.destroy_process_group()

    # --- Gloo (CPU) ---
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    gloo_results = {}
    if rank == 0:
        print(f"\n{'=' * 80}")
        print(f"  Gloo all_gather (CPU tensors)")
        print(f"  world_size={world_size}  dtype={dtype}  iters={iters}")
        print(f"{'=' * 80}")

    for numel in sizes:
        nbytes = numel * elem_bytes
        tensor = torch.randn(numel, dtype=dtype, device="cpu")
        elapsed = bench_allgather_cpu(tensor, world_size, iters)
        avg_us = elapsed / iters * 1e6
        alg_bw = (nbytes * iters) / elapsed / 1e9
        gloo_results[numel] = {"avg_us": avg_us, "alg_bw": alg_bw, "nbytes": nbytes}

        if rank == 0:
            kb = nbytes / 1024
            tag = f"{kb:.1f} KB" if kb < 1024 else f"{kb / 1024:.1f} MB"
            print(f"  {tag:>12s} ({numel:>12,d} elems)  "
                  f"latency: {avg_us:>10.2f} us  algBW: {alg_bw:>8.2f} GB/s")

    dist.destroy_process_group()

    # --- Comparison ---
    if rank == 0:
        print(f"\n{'=' * 100}")
        print(f"  Comparison: NCCL (GPU) vs Gloo (CPU)")
        print(f"{'=' * 100}")
        print(f"  {'Size':>12s}  {'Elements':>12s}  "
              f"{'NCCL (us)':>12s}  {'NCCL BW':>12s}  "
              f"{'Gloo (us)':>12s}  {'Gloo BW':>12s}  "
              f"{'NCCL faster':>12s}")
        print("  " + "-" * 96)

        for numel in sizes:
            r0 = nccl_results[numel]
            r1 = gloo_results[numel]
            kb = r0["nbytes"] / 1024
            tag = f"{kb:.1f} KB" if kb < 1024 else f"{kb / 1024:.1f} MB"
            speedup = r1["avg_us"] / r0["avg_us"]
            print(f"  {tag:>12s}  {numel:>12,d}  "
                  f"{r0['avg_us']:>10.2f}    {r0['alg_bw']:>7.2f} GB/s  "
                  f"{r1['avg_us']:>10.2f}    {r1['alg_bw']:>7.2f} GB/s  "
                  f"{speedup:>10.2f}x")


if __name__ == "__main__":
    main()
