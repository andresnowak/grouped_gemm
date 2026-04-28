#include <torch/extension.h>
#include <vector>

namespace grouped_gemm {

void BatchedD2HAsync(
    std::vector<torch::Tensor> gpu_tensors,
    std::vector<torch::Tensor> cpu_tensors,
    int64_t d2h_stream
);

}
