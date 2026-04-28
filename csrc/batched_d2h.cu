#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>


namespace grouped_gemm {


void BatchedD2HAsync(
    std::vector<torch::Tensor> gpu_tensors,
    std::vector<torch::Tensor> cpu_tensors,
    int64_t d2h_stream
) {
    int N = gpu_tensors.size();

    std::vector<void*> srcs(N), dsts(N);
    std::vector<size_t> sizes(N);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(d2h_stream);
    int dev = gpu_tensors[0].get_device();

    for (int i = 0; i < N; i++) {
        TORCH_CHECK(cpu_tensors[i].is_pinned(),
                    "CPU tensor ", i, " must be pinned memory");
        TORCH_CHECK(cpu_tensors[i].is_contiguous(),
                    "CPU tensor ", i, " must be contiguous");
        TORCH_CHECK(gpu_tensors[i].get_device() == dev,
                    "all gpu tensors must be on same device");

        srcs[i]  = gpu_tensors[i].data_ptr();
        dsts[i]  = cpu_tensors[i].data_ptr();
        sizes[i] = gpu_tensors[i].nbytes();
    }

    // Single attribute profile: all D2H
    cudaMemcpyAttributes attr = {};
    attr.srcAccessOrder  = cudaMemcpySrcAccessOrderStream; // we guarantee that we read on the order of the loads in the stream
    attr.srcLocHint.type = cudaMemLocationTypeDevice;
    attr.dstLocHint.id   = dev;
    std::vector<size_t> attrsIdxs = {0};

    cudaError_t err = cudaMemcpyBatchAsync(
        dsts.data(), srcs.data(), sizes.data(), (size_t) N,
        &attr, attrsIdxs.data(), 1,
        stream
    );

    TORCH_CHECK(err == cudaSuccess, "cudaMemcpyBatchAsync D2H failed: ", cudaGetErrorString(err));
}
}
