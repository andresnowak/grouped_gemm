#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>


namespace grouped_gemm {


void BatchedH2DAsync(
    std::vector<torch::Tensor> cpu_tensors,
    std::vector<torch::Tensor> gpu_tensors,
    int64_t h2d_stream
) {
    int N = cpu_tensors.size();

    std::vector<void*> srcs(N), dsts(N);
    std::vector<size_t> sizes(N);

    cudaStream_t h2d_sxtream_cast = reinterpret_cast<cudaStream_t>(h2d_stream);
    int dev = gpu_tensors[0].get_device();

    for (int i = 0; i < N; i++) {
        // CPU tensors MUST be pinned for async copy
        TORCH_CHECK(cpu_tensors[i].is_pinned(),
                    "CPU tensor ", i, " must be pinned memory");
        TORCH_CHECK(cpu_tensors[i].is_contiguous(),
                    "CPU tensor ", i, " must be contiguous");
        TORCH_CHECK(gpu_tensors[i].get_device() == dev, 
                    "all gpu tensors must be on same device");

        srcs[i]  = cpu_tensors[i].data_ptr();
        dsts[i]  = gpu_tensors[i].data_ptr();
        sizes[i] = cpu_tensors[i].nbytes();
    }

    // Single attribute profile: all H2D
    cudaMemcpyAttributes attr = {};
    attr.srcAccessOrder  = cudaMemcpySrcAccessOrderStream;
    attr.srcLocHint.type = cudaMemLocationTypeHost;
    attr.dstLocHint.type = cudaMemLocationTypeDevice;
    attr.dstLocHint.id   = dev;

    std::vector<size_t> attrsIdxs = {0};

    cudaError_t err = cudaMemcpyBatchAsync(
        dsts.data(), srcs.data(), sizes.data(), (size_t) N,
        &attr, attrsIdxs.data(), 1,
        h2d_sxtream_cast
    );

    TORCH_CHECK(err == cudaSuccess, "cudaMemcpyBatchAsync failed: ", cudaGetErrorString(err));
}
}