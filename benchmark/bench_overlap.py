import torch
import time
import grouped_gemm

EDP = 8
NUM_LOCAL_LAYERS = 4
NUM_LOCAL_EXPERTS = 64

# Overlapping
CHUNK_SIZE = 16
HIDDEN_SIZE = 4096
MOE_FFN_HIDDEN_SIZE = 2048
TOKEN_PER_EXPERT = 1024

WARMUP = 5
ITERS = 50


def bench_h2d():

    def _h2d(srcs, dsts):
        stream = torch.cuda.Stream()
        # warmup
        for _ in range(WARMUP):
            grouped_gemm.grouped_gemm.backend.batched_h2d_async(srcs, dsts, stream.cuda_stream)
        stream.synchronize()

        # benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        for _ in range(ITERS):
            grouped_gemm.grouped_gemm.backend.batched_h2d_async(srcs, dsts, stream.cuda_stream)
        end.record(stream)
        torch.cuda.synchronize()

        return start.elapsed_time(end) / ITERS

    torch.manual_seed(0)
    num_tensors = CHUNK_SIZE

    # FC1
    srcs, dsts = [], []
    for _ in range(num_tensors):
        srcs.append(torch.randn((HIDDEN_SIZE, MOE_FFN_HIDDEN_SIZE*2), dtype=torch.bfloat16, pin_memory=True, device="cpu"))
        dsts.append(torch.empty((HIDDEN_SIZE, MOE_FFN_HIDDEN_SIZE*2), device="cuda:0", dtype=torch.bfloat16))

    total_bytes = sum(s.numel() * s.element_size() for s in srcs)
    fc1_elapsed_ms = _h2d(srcs, dsts)
    fc1_bw = total_bytes / (fc1_elapsed_ms * 1e-3) / 1e9  # GB/s

    # FC2
    srcs, dsts = [], []
    for _ in range(num_tensors):
        srcs.append(torch.randn((MOE_FFN_HIDDEN_SIZE, HIDDEN_SIZE), dtype=torch.bfloat16, pin_memory=True, device="cpu"))
        dsts.append(torch.empty((MOE_FFN_HIDDEN_SIZE, HIDDEN_SIZE), device="cuda:0", dtype=torch.bfloat16))
    total_bytes = sum(s.numel() * s.element_size() for s in srcs)
    fc2_elapsed_ms = _h2d(srcs, dsts)
    fc2_bw = total_bytes / (fc2_elapsed_ms * 1e-3) / 1e9  # GB/s

    print(f"[H2D] {num_tensors} experts, {total_bytes / 1e9:.2f} GB total")
    print(f"  Shape per tensor: ({HIDDEN_SIZE}, {MOE_FFN_HIDDEN_SIZE*2}), dtype: bfloat16")
    print(f"  FC1 Latency: {fc1_elapsed_ms:.3f} ms | Bandwidth: {fc1_bw:.2f} GB/s | MBU: {fc1_bw / 450 * 100:.2f}%")
    print(f"  FC2 Latency: {fc2_elapsed_ms:.3f} ms | Bandwidth: {fc2_bw:.2f} GB/s | MBU: {fc2_bw / 450 * 100:.2f}%")
    return fc1_elapsed_ms, fc2_elapsed_ms


def bench_groupgemmfwd():

    def _gg(a, b, batch_sizes):
        # warmup
        for _ in range(WARMUP):
            _ = grouped_gemm.grouped_gemm.backend.gmm(a, b, batch_sizes)
        torch.cuda.synchronize()

        # benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERS):
            _ = grouped_gemm.grouped_gemm.backend.gmm(a, b, batch_sizes)
        end.record()
        torch.cuda.synchronize()

        return start.elapsed_time(end) / ITERS
    
    torch.manual_seed(0)
    num_experts = CHUNK_SIZE
    tokens_per_expert = TOKEN_PER_EXPERT
    hidden = HIDDEN_SIZE
    total_tokens = num_experts * tokens_per_expert

    batch_sizes = torch.full((num_experts,), tokens_per_expert, dtype=torch.int64)

    # FC1 weight tensor: (num_experts, hidden, ffn_hidden)
    a = torch.randn(total_tokens, hidden, device="cuda:0", dtype=torch.bfloat16)
    b = torch.randn(num_experts, hidden, MOE_FFN_HIDDEN_SIZE*2, device="cuda:0", dtype=torch.bfloat16)

    fc1_elapsed_ms = _gg(a, b, batch_sizes)
    fc1_flops = num_experts * 2.0 * tokens_per_expert * hidden * MOE_FFN_HIDDEN_SIZE*2
    fc1_tflops = fc1_flops / (fc1_elapsed_ms * 1e-3) / 1e12

    # FC2 weight tensor: (num_experts, ffn_hidden, hidden)
    a = torch.randn(total_tokens, MOE_FFN_HIDDEN_SIZE, device="cuda:0", dtype=torch.bfloat16)
    b = torch.randn(num_experts, MOE_FFN_HIDDEN_SIZE, hidden, device="cuda:0", dtype=torch.bfloat16)

    fc2_elapsed_ms = _gg(a, b, batch_sizes)
    fc2_flops = num_experts * 2.0 * tokens_per_expert * MOE_FFN_HIDDEN_SIZE * hidden
    fc2_tflops = fc2_flops / (fc2_elapsed_ms * 1e-3) / 1e12

    print(f"[GroupedGEMM FWD] {num_experts} experts, {tokens_per_expert} tokens/expert, "
          f"({hidden}, {MOE_FFN_HIDDEN_SIZE*2})")
    print(f"  Shape of A: ({total_tokens}, {hidden}), Shape of B: ({num_experts}, {hidden}, {MOE_FFN_HIDDEN_SIZE*2}), ")
    print(f"  FC1 Latency: {fc1_elapsed_ms:.3f} ms | TFLOPS: {fc1_tflops:.2f} | MFU: {fc1_tflops / 989 * 100:.2f}%")
    print(f"  FC2 Latency: {fc2_elapsed_ms:.3f} ms | TFLOPS: {fc2_tflops:.2f} | MFU: {fc2_tflops / 989 * 100:.2f}%")

    return fc1_elapsed_ms, fc2_elapsed_ms

def moe_size():
    total_moe_weight_bytes = (HIDDEN_SIZE * MOE_FFN_HIDDEN_SIZE*2 + MOE_FFN_HIDDEN_SIZE * HIDDEN_SIZE) * NUM_LOCAL_LAYERS * NUM_LOCAL_EXPERTS
    total_moe_main_grad_bytes = total_moe_weight_bytes*2
    total_main_param_bytes = total_moe_weight_bytes*2 / EDP
    total_adam_momentum_bytes = total_moe_weight_bytes*2 / EDP
    total_adam_variance_bytes = total_moe_weight_bytes*2 / EDP
    total_optimizer_state_bytes = total_adam_momentum_bytes + total_adam_variance_bytes + total_main_param_bytes
    total = total_moe_weight_bytes + total_moe_main_grad_bytes + total_optimizer_state_bytes
    print(f"Total size of MoE FFN weights: {total_moe_weight_bytes / 1e9:.2f} GB")
    print(f"Total size of MoE FFN main gradients: {total_moe_main_grad_bytes / 1e9:.2f} GB")
    print(f"Total size of MoE optimizer state after FSDP: {total_optimizer_state_bytes / 1e9:.2f} GB")
    print(f"Total size: {total / 1e9:.2f} GB")



if __name__ == "__main__":
    # print("=== MoE FFN Size Estimation ===")
    # moe_size()
    fc1_h2d_latency, fc2_h2d_latency = bench_h2d()
    fc1_gg_latency, fc2_gg_latency = bench_groupgemmfwd()
    fc1_gap_ms = fc1_h2d_latency - fc1_gg_latency
    fc2_gap_ms = fc2_h2d_latency - fc2_gg_latency
    fc1_overlap_efficiency = fc1_gg_latency / fc1_h2d_latency
    fc2_overlap_efficiency = fc2_gg_latency / fc2_h2d_latency
    print(f"\nExposed H2D Latency: FC1={fc1_gap_ms:.3f} ms | FC2={fc2_gap_ms:.3f} ms")
    print(f"Overlap efficiency: FC1={fc1_overlap_efficiency*100:.2f}% | FC2={fc2_overlap_efficiency*100:.2f}%")

