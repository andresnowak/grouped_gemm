"""Benchmark tensor accumulation (+=) on CPU vs GPU."""

import torch
import argparse
import time

WARMUP = 10
ITERS = 100

# Typical shapes from the grouped_gemm workload
DEFAULT_SHAPES = [
    (4096, 4096),        # 64 MB bf16
    (8192, 4096),        # 128 MB bf16
    (4096, 8192),        # 128 MB bf16
    (16384, 4096),       # 256 MB bf16
    (4096, 14336),       # ~220 MB bf16 (SwiGLU-style)
]


def bench_cpu_accum(dst, src, iters):
    for _ in range(WARMUP):
        dst.add_(src)
    t0 = time.perf_counter()
    for _ in range(iters):
        dst.add_(src)
    t1 = time.perf_counter()
    return t1 - t0


def bench_gpu_accum(dst, src, iters):
    for _ in range(WARMUP):
        dst.add_(src)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        dst.add_(src)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--num-experts", type=int, default=64,
                        help="Accumulate across this many independent tensors")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    iters = args.iters

    print(f"=== Tensor Accumulation Benchmark: CPU vs GPU ===")
    print(f"dtype={dtype}, iters={iters}, num_experts={args.num_experts}")
    print()
    print(f"{'Shape':>20s}  {'Elements':>12s}  {'Size (MB)':>10s}  "
          f"{'CPU (us)':>10s}  {'CPU BW (GB/s)':>14s}  "
          f"{'GPU (us)':>10s}  {'GPU BW (GB/s)':>14s}  {'Speedup':>8s}")
    print("-" * 120)

    for shape in DEFAULT_SHAPES:
        numel = 1
        for s in shape:
            numel *= s
        elem_bytes = torch.tensor([], dtype=dtype).element_size()
        nbytes = numel * elem_bytes
        size_mb = nbytes / (1024 ** 2)

        # --- CPU benchmark ---
        dst_cpu = torch.randn(shape, dtype=torch.float32, device="cpu")
        src_cpu = torch.randn(shape, dtype=dtype, device="cpu")
        cpu_elapsed = bench_cpu_accum(dst_cpu, src_cpu, iters)
        cpu_us = cpu_elapsed / iters * 1e6
        cpu_bw = (nbytes * iters) / cpu_elapsed / 1e9  # read src + read/write dst ~3x nbytes

        # --- GPU benchmark ---
        dst_gpu = torch.randn(shape, dtype=torch.float32, device="cuda")
        src_gpu = torch.randn(shape, dtype=dtype, device="cuda")
        gpu_elapsed = bench_gpu_accum(dst_gpu, src_gpu, iters)
        gpu_us = gpu_elapsed / iters * 1e6
        gpu_bw = (nbytes * iters) / gpu_elapsed / 1e9

        speedup = cpu_elapsed / gpu_elapsed

        print(f"{str(shape):>20s}  {numel:>12,d}  {size_mb:>10.1f}  "
              f"{cpu_us:>10.2f}  {cpu_bw:>14.2f}  "
              f"{gpu_us:>10.2f}  {gpu_bw:>14.2f}  {speedup:>7.2f}x")

    # --- Multi-tensor accumulation (simulates grad accumulation across experts) ---
    if args.num_experts > 1:
        print()
        print(f"--- Multi-tensor accumulation ({args.num_experts} experts) ---")
        shape = DEFAULT_SHAPES[0]
        numel = shape[0] * shape[1]
        elem_bytes = torch.tensor([], dtype=dtype).element_size()
        nbytes = numel * elem_bytes

        # CPU: accumulate N sources into one destination
        dst_cpu = torch.randn(shape, dtype=torch.float32, device="cpu")
        srcs_cpu = [torch.randn(shape, dtype=dtype, device="cpu") for _ in range(args.num_experts)]

        for _ in range(WARMUP):
            for s in srcs_cpu:
                dst_cpu.add_(s)
        t0 = time.perf_counter()
        for _ in range(iters):
            for s in srcs_cpu:
                dst_cpu.add_(s)
        cpu_elapsed = time.perf_counter() - t0
        cpu_us = cpu_elapsed / iters * 1e6

        # GPU
        dst_gpu = torch.randn(shape, dtype=torch.float32, device="cuda")
        srcs_gpu = [torch.randn(shape, dtype=dtype, device="cuda") for _ in range(args.num_experts)]

        for _ in range(WARMUP):
            for s in srcs_gpu:
                dst_gpu.add_(s)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            for s in srcs_gpu:
                dst_gpu.add_(s)
        end.record()
        torch.cuda.synchronize()
        gpu_elapsed = start.elapsed_time(end) / 1000.0
        gpu_us = gpu_elapsed / iters * 1e6

        total_bytes_per_iter = nbytes * args.num_experts
        cpu_bw = (total_bytes_per_iter * iters) / cpu_elapsed / 1e9
        gpu_bw = (total_bytes_per_iter * iters) / gpu_elapsed / 1e9

        print(f"  {args.num_experts}x {shape}  CPU: {cpu_us:.2f} us/iter ({cpu_bw:.2f} GB/s)  "
              f"GPU: {gpu_us:.2f} us/iter ({gpu_bw:.2f} GB/s)  "
              f"speedup: {cpu_elapsed / gpu_elapsed:.2f}x")


if __name__ == "__main__":
    main()
