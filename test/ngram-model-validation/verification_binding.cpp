#include <ATen/ATen.h>
#include <torch/library.h>

void verify_tree_greedy_cpu(at::Tensor, at::Tensor, at::Tensor,
                            const at::Tensor &, const at::Tensor &,
                            const at::Tensor &, const at::Tensor &,
                            const at::Tensor &);
void reconstruct_indices_from_tree_mask_cpu(const at::Tensor &,
                                            const at::Tensor &, at::Tensor,
                                            at::Tensor, at::Tensor, at::Tensor,
                                            int64_t, int64_t);

// Only registration is test-specific; both implementations are linked unchanged
// from the repository's CPU speculative-kernel translation unit.
TORCH_LIBRARY(ngram_validation, m) {
  m.def("verify(Tensor(a!) predicts, Tensor(b!) indices, Tensor(c!) count, "
        "Tensor candidates, "
        "Tensor retrieval, Tensor child, Tensor sibling, Tensor target) -> ()",
        &verify_tree_greedy_cpu);
  m.def("reconstruct(Tensor mask, Tensor lengths, Tensor(a!) positions, "
        "Tensor(b!) retrieval, "
        "Tensor(c!) child, Tensor(d!) sibling, int batch_size, int draft_size) "
        "-> ()",
        &reconstruct_indices_from_tree_mask_cpu);
}
