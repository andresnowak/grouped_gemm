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


def bench_groupgemm_with_h2d(
    num_local_experts=NUM_LOCAL_EXPERTS,
    chunk_size=CHUNK_SIZE,
    hidden_size=HIDDEN_SIZE,
    moe_ffn_hidden_size=MOE_FFN_HIDDEN_SIZE,
    total_tokens=NUM_LOCAL_EXPERTS * TOKEN_PER_EXPERT,
    variable_tokens_per_expert=False,
    variable_tokens_max_factor=2.0,
    warmup=WARMUP,
    iters=ITERS,
    skip_prefetch_timing=False,
):
    """Compare grouped GEMM with weights on GPU vs double-buffered H2D from CPU."""

    def _make_batch_sizes():
        if not variable_tokens_per_expert:
            base_tokens_per_expert = total_tokens // num_local_experts
            batch_sizes = torch.full((num_local_experts,), base_tokens_per_expert, dtype=torch.int64)
            remainder = total_tokens - base_tokens_per_expert * num_local_experts
            if remainder > 0:
                batch_sizes[:remainder] += 1
            return batch_sizes

        # Sample bounded per-expert weights around uniform, then discretize while
        # preserving the global token budget exactly.

        weights = 1.0 + (variable_tokens_max_factor - 1.0) * torch.rand(num_local_experts)
        raw_counts = weights / weights.sum() * total_tokens # sums to exactly total_tokens
        batch_sizes = torch.floor(raw_counts).to(torch.int64)
        remainder = total_tokens - int(batch_sizes.sum().item())
        if remainder > 0:
            topk = torch.topk(raw_counts - batch_sizes.to(raw_counts.dtype), k=remainder).indices
            batch_sizes[topk] += 1 # add 1 to k counts so as to preserve the total token budget
        return batch_sizes

    def _gg_no_h2d(a, b_gpu, batch_sizes, c, label):
        nvtx.range_push(f"{label}/no_h2d/warmup")
        for _ in range(warmup):
            grouped_gemm.grouped_gemm.backend.gmmfwd(a, b_gpu, batch_sizes, [], c=c)
        torch.cuda.synchronize()
        nvtx.range_pop()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        nvtx.range_push(f"{label}/no_h2d/bench")
        start.record()
        for i in range(iters):
            nvtx.range_push(f"{label}/no_h2d/iter_{i}")
            grouped_gemm.grouped_gemm.backend.gmmfwd(a, b_gpu, batch_sizes, [], c=c)
            nvtx.range_pop()
        end.record()
        torch.cuda.synchronize()
        nvtx.range_pop()
        return start.elapsed_time(end) / iters

    def _gg_h2d(a, b_cpu, batch_sizes, c, label, skip_prefetch=False):
        num_stages = len(b_cpu) // chunk_size
        batch_size_chunks = [batch_sizes[s * chunk_size:(s + 1) * chunk_size] for s in range(num_stages)]
        stage_token_counts = [int(bs_chunk.sum().item()) for bs_chunk in batch_size_chunks]

        h2d_stream = torch.cuda.Stream()
        compute_stream = torch.cuda.Stream()
        # compute_stream = torch.cuda.current_stream()

        slots = [
            [torch.empty_like(b_cpu[i], device="cuda:0") for i in range(chunk_size)],
            [torch.empty_like(b_cpu[i], device="cuda:0") for i in range(chunk_size)],
        ]
        ready = [torch.cuda.Event() for _ in range(num_stages)]
        gemm_done = [torch.cuda.Event() for _ in range(num_stages)]

        def _h2d_stage(s, slot):
            grouped_gemm.grouped_gemm.backend.batched_h2d_async(
                b_cpu[s * chunk_size:(s + 1) * chunk_size], slot, h2d_stream.cuda_stream
            )

        def _measure_single_h2d_group_load_ms():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            nvtx.range_push(f"{label}/h2d/group_load_bench")
            start.record(h2d_stream)
            for i in range(iters):
                _h2d_stage(0, slots[i % 2])
            end.record(h2d_stream)
            torch.cuda.synchronize()
            nvtx.range_pop()
            return start.elapsed_time(end) / iters

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

            off = 0
            for s in range(num_stages):
                si = s % 2
                bs_chunk = batch_size_chunks[s]
                tokens_in_stage = stage_token_counts[s]
                nvtx.range_push(f"{iter_label}/stage_{s}/gemm")
                compute_stream.wait_event(ready[s])
                with torch.cuda.stream(compute_stream):
                    grouped_gemm.grouped_gemm.backend.gmmfwd(
                        a[off:off + tokens_in_stage], slots[si], bs_chunk, [],
                        c=c[off:off + tokens_in_stage],
                    )
                    gemm_done[s].record(compute_stream)
                nvtx.range_pop()
                off += tokens_in_stage

                if s + 2 < num_stages:
                    h2d_stream.wait_event(gemm_done[s])
                    nvtx.range_push(f"{iter_label}/stage_{s+2}/h2d")
                    _h2d_stage(s + 2, slots[si])
                    ready[s + 2].record(h2d_stream)
                    nvtx.range_pop()

        nvtx.range_push(f"{label}/h2d/warmup")
        for i in range(warmup):
            _run(f"{label}/h2d/warmup_iter_{i}")
        torch.cuda.synchronize()
        nvtx.range_pop()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        nvtx.range_push(f"{label}/h2d/bench")
        start.record(compute_stream)
        for i in range(iters):
            _run(f"{label}/h2d/iter_{i}")
        end.record(compute_stream)
        torch.cuda.synchronize()
        nvtx.range_pop()
        avg_ms = start.elapsed_time(end) / iters
        if not skip_prefetch_timing:
            return avg_ms

        h2d_group_load_ms = _measure_single_h2d_group_load_ms()
        return max(avg_ms - h2d_group_load_ms, 0.0)

    torch.manual_seed(0)
    num_experts = num_local_experts
    hidden = hidden_size
    batch_sizes = _make_batch_sizes()
    realized_total_tokens = int(batch_sizes.sum().item())

    # FC1
    nvtx.range_push("setup/fc1")
    a_fc1 = torch.randn(realized_total_tokens, hidden, device="cuda:0", dtype=torch.bfloat16)
    b_fc1_cpu = [torch.randn(hidden, moe_ffn_hidden_size * 2, dtype=torch.bfloat16, pin_memory=True) for _ in range(num_experts)]
    b_fc1_gpu = [t.to("cuda:0") for t in b_fc1_cpu]
    c_fc1 = torch.empty(realized_total_tokens, moe_ffn_hidden_size * 2, device="cuda:0", dtype=torch.bfloat16)
    fc1_flops = 2.0 * realized_total_tokens * hidden * moe_ffn_hidden_size * 2
    nvtx.range_pop()

    fc1_no_h2d_ms = _gg_no_h2d(a_fc1, b_fc1_gpu, batch_sizes, c_fc1, "fc1")
    fc1_h2d_ms = _gg_h2d(a_fc1, b_fc1_cpu, batch_sizes, c_fc1, "fc1")
    fc1_no_h2d_tflops = fc1_flops / (fc1_no_h2d_ms * 1e-3) / 1e12
    fc1_h2d_tflops = fc1_flops / (fc1_h2d_ms * 1e-3) / 1e12

    # FC2
    nvtx.range_push("setup/fc2")
    a_fc2 = torch.randn(realized_total_tokens, moe_ffn_hidden_size, device="cuda:0", dtype=torch.bfloat16)
    b_fc2_cpu = [torch.randn(moe_ffn_hidden_size, hidden, dtype=torch.bfloat16, pin_memory=True) for _ in range(num_experts)]
    b_fc2_gpu = [t.to("cuda:0") for t in b_fc2_cpu]
    c_fc2 = torch.empty(realized_total_tokens, hidden, device="cuda:0", dtype=torch.bfloat16)
    fc2_flops = 2.0 * realized_total_tokens * moe_ffn_hidden_size * hidden
    nvtx.range_pop()

    fc2_no_h2d_ms = _gg_no_h2d(a_fc2, b_fc2_gpu, batch_sizes, c_fc2, "fc2")
    fc2_h2d_ms = _gg_h2d(a_fc2, b_fc2_cpu, batch_sizes, c_fc2, "fc2")
    fc2_no_h2d_tflops = fc2_flops / (fc2_no_h2d_ms * 1e-3) / 1e12
    fc2_h2d_tflops = fc2_flops / (fc2_h2d_ms * 1e-3) / 1e12

    h2d_label = "+H2D ov skip-prefetch-timing" if skip_prefetch_timing else "+H2D ov"
    token_mode = "variable" if variable_tokens_per_expert else "uniform"
    batch_min = int(batch_sizes.min().item())
    batch_max = int(batch_sizes.max().item())

    print(
        f"[GroupedGEMM+H2D] {num_experts} experts ({num_experts // chunk_size} stages x {chunk_size}), "
        f"total_tokens={realized_total_tokens}, avg={realized_total_tokens / num_experts:.1f} tokens/expert, "
        f"mode={token_mode}, min={batch_min}, max={batch_max}"
    )
    print(f"  FC1  no-H2D: {fc1_no_h2d_ms:.3f} ms | TFLOPS: {fc1_no_h2d_tflops:.2f} | MFU: {fc1_no_h2d_tflops / 989 * 100:.2f}%")
    print(f"  FC1 {h2d_label}: {fc1_h2d_ms:.3f} ms | TFLOPS: {fc1_h2d_tflops:.2f} | MFU: {fc1_h2d_tflops / 989 * 100:.2f}%")
    print(f"  FC2  no-H2D: {fc2_no_h2d_ms:.3f} ms | TFLOPS: {fc2_no_h2d_tflops:.2f} | MFU: {fc2_no_h2d_tflops / 989 * 100:.2f}%")
    print(f"  FC2 {h2d_label}: {fc2_h2d_ms:.3f} ms | TFLOPS: {fc2_h2d_tflops:.2f} | MFU: {fc2_h2d_tflops / 989 * 100:.2f}%")
    return fc1_no_h2d_ms, fc2_no_h2d_ms, fc1_h2d_ms, fc2_h2d_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Disable warmup and run 1 iteration (for nsys profiling)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help=f"Experts per H2D stage (default: {CHUNK_SIZE})")
    parser.add_argument("--num-local-experts", type=int, default=NUM_LOCAL_EXPERTS, help=f"Number of local experts (default: {NUM_LOCAL_EXPERTS})")
    parser.add_argument(
        "--total-tokens",
        type=int,
        default=NUM_LOCAL_EXPERTS * TOKEN_PER_EXPERT,
        help=f"Total token budget across all experts (default: {NUM_LOCAL_EXPERTS * TOKEN_PER_EXPERT})",
    )
    parser.add_argument(
        "--variable-tokens-per-expert",
        action="store_true",
        help="Use a variable per-expert token distribution while keeping the same total token budget",
    )
    parser.add_argument(
        "--variable-tokens-max-factor",
        type=float,
        default=2.0,
        help="Maximum multiplier used to skew per-expert token counts around the average (default: 2.0)",
    )
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE, help=f"Model hidden size (default: {HIDDEN_SIZE})")
    parser.add_argument(
        "--moe-ffn-hidden-size",
        type=int,
        default=MOE_FFN_HIDDEN_SIZE,
        help=f"MoE FFN hidden size per projection branch (default: {MOE_FFN_HIDDEN_SIZE})",
    )
    parser.add_argument("--warmup", type=int, default=WARMUP, help=f"Warmup iterations (default: {WARMUP})")
    parser.add_argument("--iters", type=int, default=ITERS, help=f"Benchmark iterations (default: {ITERS})")
    parser.add_argument(
        "--skip-prefetch-timing",
        action="store_true",
        help="Subtract the measured time of one H2D group load from the averaged overlapped timing",
    )
    args = parser.parse_args()

    if args.num_local_experts % args.chunk_size != 0:
        raise ValueError("--num-local-experts must be divisible by --chunk-size")
    if args.total_tokens < args.num_local_experts:
        raise ValueError("--total-tokens must be >= --num-local-experts")
    if args.variable_tokens_max_factor < 1.0:
        raise ValueError("--variable-tokens-max-factor must be >= 1.0")

    warmup = 0 if args.profile else args.warmup
    iters = 1 if args.profile else args.iters

    fc1_no_h2d_ms, fc2_no_h2d_ms, fc1_ov_ms, fc2_ov_ms = bench_groupgemm_with_h2d(
        num_local_experts=args.num_local_experts,
        chunk_size=args.chunk_size,
        hidden_size=args.hidden_size,
        moe_ffn_hidden_size=args.moe_ffn_hidden_size,
        total_tokens=args.total_tokens,
        variable_tokens_per_expert=args.variable_tokens_per_expert,
        variable_tokens_max_factor=args.variable_tokens_max_factor,
        warmup=warmup,
        iters=iters,
        skip_prefetch_timing=args.skip_prefetch_timing
    )

    print("\n=== Overlap Summary ===")
    for name, no_h2d, ov in [("FC1", fc1_no_h2d_ms, fc1_ov_ms), ("FC2", fc2_no_h2d_ms, fc2_ov_ms)]:
        efficiency = no_h2d / ov * 100
        print(f"  {name}: GEMM(no-H2D)={no_h2d:.3f}ms | Overlapped={ov:.3f}ms | Overlap efficiency={efficiency:.1f}%")
