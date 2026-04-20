#include <torch/extension.h>
#include <vector>

namespace grouped_gemm {


void BatchedH2DAsync(
    std::vector<torch::Tensor> cpu_tensors,
    std::vector<torch::Tensor> gpu_tensors,
    int64_t h2d_stream
);

}