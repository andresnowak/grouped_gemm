#include "grouped_gemm.h"
#include "permute.h"
#include "sinkhorn.h"
#include "batched_h2d.h"
#include "batched_d2h.h"

#include <torch/extension.h>

namespace grouped_gemm {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("batched_h2d_async", &BatchedH2DAsync, "Batched Host to Device Async Copy.");
  m.def("batched_d2h_async", &BatchedD2HAsync, "Batched Device to Host Async Copy.");
  m.def("gmm", &GroupedGemm, "Grouped GEMM.");
  m.def("gmmfwd", &GroupedGemmFwd, "Grouped GEMM Forward.");
  m.def("gmmbwd", &GroupedGemmBwd, "Grouped GEMM Backward.");
  m.def("sinkhorn", &sinkhorn, "Sinkhorn kernel");
  m.def("permute", &moe_permute_topK_op, "Token permutation kernel");
  m.def("unpermute", &moe_recover_topK_op, "Token un-permutation kernel");
  m.def("unpermute_bwd", &moe_recover_topK_bwd_op, "Token un-permutation backward kernel");
}

}  // namespace grouped_gemm
