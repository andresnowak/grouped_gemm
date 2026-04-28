import argparse
import torch
import torch.cuda.nvtx as nvtx
import grouped_gemm

NUM_LOCAL_EXPERTS = 64
CHUNK_SIZE = 16
HIDDEN_SIZE = 4096
MOE_FFN_HIDDEN_SIZE = 2048
TOKEN_PER_EXPERT = 1024

WARMUP = 5
ITERS = 50


def bench_groupgemm_with_h2d(skip_prefetch_timing=False):
    """Compare grouped GEMM with weights on GPU vs double-buffered H2D from CPU."""

    def _gg_no_h2d(a, b_gpu, batch_sizes, c, label):
        nvtx.range_push(f"{label}/no_h2d/warmup")
        for _ in range(WARMUP):
            grouped_gemm.grouped_gemm.backend.gmmfwd(a, b_gpu, batch_sizes, [], c=c)
        torch.cuda.synchronize()
        nvtx.range_pop()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        nvtx.range_push(f"{label}/no_h2d/bench")
        start.record()
        for i in range(ITERS):
            nvtx.range_push(f"{label}/no_h2d/iter_{i}")
            grouped_gemm.grouped_gemm.backend.gmmfwd(a, b_gpu, batch_sizes, [], c=c)
            nvtx.range_pop()
        end.record()
        torch.cuda.synchronize()
        nvtx.range_pop()
        return start.elapsed_time(end) / ITERS

    def _gg_h2d(a, b_cpu, batch_sizes, c, label, skip_prefetch=False):
        num_stages = len(b_cpu) // CHUNK_SIZE
        tokens_per_stage = CHUNK_SIZE * batch_sizes[0].item()
        bs_chunk = batch_sizes[:CHUNK_SIZE]

        h2d_stream = torch.cuda.Stream()
        compute_stream = torch.cuda.Stream()
        # compute_stream = torch.cuda.current_stream()

        slots = [
            [torch.empty_like(b_cpu[i], device="cuda:0") for i in range(CHUNK_SIZE)],
            [torch.empty_like(b_cpu[i], device="cuda:0") for i in range(CHUNK_SIZE)],
        ]
        ready = [torch.cuda.Event() for _ in range(num_stages)]
        gemm_done = [torch.cuda.Event() for _ in range(num_stages)]

        def _h2d_stage(s, slot):
            grouped_gemm.grouped_gemm.backend.batched_h2d_async(
                b_cpu[s * CHUNK_SIZE:(s + 1) * CHUNK_SIZE], slot, h2d_stream.cuda_stream
            )

        def _measure_single_h2d_group_load_ms():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            nvtx.range_push(f"{label}/h2d/group_load_bench")
            start.record(h2d_stream)
            for i in range(ITERS):
                _h2d_stage(0, slots[i % 2])
            end.record(h2d_stream)
            torch.cuda.synchronize()
            nvtx.range_pop()
            return start.elapsed_time(end) / ITERS

        def _prefetch(iter_label):
            nvtx.range_push(f"{iter_label}/stage_0/h2d")
            _h2d_stage(0, slots[0])
            ready[0].record(h2d_stream)
            nvtx.range_pop()
            if num_stages > 1:
                nvtx.range_push(f"{iter_label}/stage_1/h2d")
                _h2d_stage(1, slots[1])
                ready[1].record(h2d_stream)
                nvtx.range_pop()

        def _run(iter_label, include_prefetch=True):
            if include_prefetch:
                _prefetch(iter_label)

            for s in range(num_stages):
                si = s % 2
                nvtx.range_push(f"{iter_label}/stage_{s}/gemm")
                compute_stream.wait_event(ready[s])
                off = s * tokens_per_stage
                with torch.cuda.stream(compute_stream):
                    grouped_gemm.grouped_gemm.backend.gmmfwd(
                        a[off:off + tokens_per_stage], slots[si], bs_chunk, [],
                        c=c[off:off + tokens_per_stage],
                    )
                    gemm_done[s].record(compute_stream)
                nvtx.range_pop()

                if s + 2 < num_stages:
                    h2d_stream.wait_event(gemm_done[s])
                    nvtx.range_push(f"{iter_label}/stage_{s+2}/h2d")
                    _h2d_stage(s + 2, slots[si])
                    ready[s + 2].record(h2d_stream)
                    nvtx.range_pop()

        nvtx.range_push(f"{label}/h2d/warmup")
        for i in range(WARMUP):
            _run(f"{label}/h2d/warmup_iter_{i}")
        torch.cuda.synchronize()
        nvtx.range_pop()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        nvtx.range_push(f"{label}/h2d/bench")
        start.record(compute_stream)
        for i in range(ITERS):
            _run(f"{label}/h2d/iter_{i}")
        end.record(compute_stream)
        torch.cuda.synchronize()
        nvtx.range_pop()
        avg_ms = start.elapsed_time(end) / ITERS
        if not skip_prefetch_timing:
            return avg_ms

        h2d_group_load_ms = _measure_single_h2d_group_load_ms()
        return max(avg_ms - h2d_group_load_ms, 0.0)

    torch.manual_seed(0)
    num_experts = NUM_LOCAL_EXPERTS
    tokens_per_expert = TOKEN_PER_EXPERT
    hidden = HIDDEN_SIZE
    total_tokens = num_experts * tokens_per_expert
    batch_sizes = torch.full((num_experts,), tokens_per_expert, dtype=torch.int64)

    # FC1
    nvtx.range_push("setup/fc1")
    a_fc1 = torch.randn(total_tokens, hidden, device="cuda:0", dtype=torch.bfloat16)
    b_fc1_cpu = [torch.randn(hidden, MOE_FFN_HIDDEN_SIZE * 2, dtype=torch.bfloat16, pin_memory=True) for _ in range(num_experts)]
    b_fc1_gpu = [t.to("cuda:0") for t in b_fc1_cpu]
    c_fc1 = torch.empty(total_tokens, MOE_FFN_HIDDEN_SIZE * 2, device="cuda:0", dtype=torch.bfloat16)
    fc1_flops = num_experts * 2.0 * tokens_per_expert * hidden * MOE_FFN_HIDDEN_SIZE * 2
    nvtx.range_pop()

    fc1_no_h2d_ms = _gg_no_h2d(a_fc1, b_fc1_gpu, batch_sizes, c_fc1, "fc1")
    fc1_h2d_ms = _gg_h2d(a_fc1, b_fc1_cpu, batch_sizes, c_fc1, "fc1")
    fc1_no_h2d_tflops = fc1_flops / (fc1_no_h2d_ms * 1e-3) / 1e12
    fc1_h2d_tflops = fc1_flops / (fc1_h2d_ms * 1e-3) / 1e12

    # FC2
    nvtx.range_push("setup/fc2")
    a_fc2 = torch.randn(total_tokens, MOE_FFN_HIDDEN_SIZE, device="cuda:0", dtype=torch.bfloat16)
    b_fc2_cpu = [torch.randn(MOE_FFN_HIDDEN_SIZE, hidden, dtype=torch.bfloat16, pin_memory=True) for _ in range(num_experts)]
    b_fc2_gpu = [t.to("cuda:0") for t in b_fc2_cpu]
    c_fc2 = torch.empty(total_tokens, hidden, device="cuda:0", dtype=torch.bfloat16)
    fc2_flops = num_experts * 2.0 * tokens_per_expert * MOE_FFN_HIDDEN_SIZE * hidden
    nvtx.range_pop()

    fc2_no_h2d_ms = _gg_no_h2d(a_fc2, b_fc2_gpu, batch_sizes, c_fc2, "fc2")
    fc2_h2d_ms = _gg_h2d(a_fc2, b_fc2_cpu, batch_sizes, c_fc2, "fc2")
    fc2_no_h2d_tflops = fc2_flops / (fc2_no_h2d_ms * 1e-3) / 1e12
    fc2_h2d_tflops = fc2_flops / (fc2_h2d_ms * 1e-3) / 1e12

    h2d_label = "+H2D ov skip-prefetch-timing" if skip_prefetch_timing else "+H2D ov"

    print(f"[GroupedGEMM+H2D] {num_experts} experts ({NUM_LOCAL_EXPERTS//CHUNK_SIZE} stages x {CHUNK_SIZE}), {tokens_per_expert} tokens/expert")
    print(f"  FC1  no-H2D: {fc1_no_h2d_ms:.3f} ms | TFLOPS: {fc1_no_h2d_tflops:.2f} | MFU: {fc1_no_h2d_tflops / 989 * 100:.2f}%")
    print(f"  FC1 {h2d_label}: {fc1_h2d_ms:.3f} ms | TFLOPS: {fc1_h2d_tflops:.2f} | MFU: {fc1_h2d_tflops / 989 * 100:.2f}%")
    print(f"  FC2  no-H2D: {fc2_no_h2d_ms:.3f} ms | TFLOPS: {fc2_no_h2d_tflops:.2f} | MFU: {fc2_no_h2d_tflops / 989 * 100:.2f}%")
    print(f"  FC2 {h2d_label}: {fc2_h2d_ms:.3f} ms | TFLOPS: {fc2_h2d_tflops:.2f} | MFU: {fc2_h2d_tflops / 989 * 100:.2f}%")
    return fc1_no_h2d_ms, fc2_no_h2d_ms, fc1_h2d_ms, fc2_h2d_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Disable warmup and run 1 iteration (for nsys profiling)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help=f"Experts per H2D stage (default: {CHUNK_SIZE})")
    parser.add_argument(
        "--skip-prefetch-timing",
        action="store_true",
        help="Subtract the measured time of one H2D group load from the averaged overlapped timing",
    )
    args = parser.parse_args()

    if args.profile:
        WARMUP = 0
        ITERS = 1
    CHUNK_SIZE = args.chunk_size

    fc1_no_h2d_ms, fc2_no_h2d_ms, fc1_ov_ms, fc2_ov_ms = bench_groupgemm_with_h2d(
        skip_prefetch_timing=args.skip_prefetch_timing
    )

    print("\n=== Overlap Summary ===")
    for name, no_h2d, ov in [("FC1", fc1_no_h2d_ms, fc1_ov_ms), ("FC2", fc2_no_h2d_ms, fc2_ov_ms)]:
        efficiency = no_h2d / ov * 100
        print(f"  {name}: GEMM(no-H2D)={no_h2d:.3f}ms | Overlapped={ov:.3f}ms | Overlap efficiency={efficiency:.1f}%")
