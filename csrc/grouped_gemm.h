#include <torch/extension.h>
#include <vector>

namespace grouped_gemm {

void GroupedGemm(torch::Tensor a,
		 torch::Tensor b,
		 torch::Tensor c,
		 torch::Tensor batch_sizes,
		 bool trans_a, bool trans_b,
		 float alpha, float beta);

void GroupedGemmBwd(torch::Tensor a,
		 torch::Tensor b,
		 std::vector<torch::Tensor> c,
		 torch::Tensor batch_sizes,
		 std::vector<int64_t> compute_streams,
		 bool trans_a, bool trans_b,
		 float alpha, float beta);

void GroupedGemmFwd(
		torch::Tensor a,
		std::vector<torch::Tensor> b,
		torch::Tensor c,
		torch::Tensor batch_sizes,
		std::vector<int64_t> compute_streams,
		bool trans_a, bool trans_b,
		float alpha, float beta);

}  // namespace grouped_gemm
