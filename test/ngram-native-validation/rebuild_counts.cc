#include "trie.h"

#include <iostream>
#include <stdexcept>
#include <vector>

using namespace sglang::ngram;
size_t validation_rebuild_count = 0;

int main() {
  Param p{};
  p.max_trie_depth = 18;
  p.min_bfs_breadth = p.max_bfs_breadth = 1;
  p.draft_token_num = 4;
  p.match_type = "BFS";
  Trie trie(10000, p);
  const std::vector<int32_t> seed{10, 20};
  trie.insert(seed.data(), seed.size());
  MatchState state;
  trie.buildRecency(seed.data(), seed.size(), 20, 3, p, state, 2);
  const std::vector<int32_t> unrelated{70, 80};
  trie.insert(unrelated.data(), unrelated.size());
  const std::vector<int32_t> extended{10, 20, 30};
  validation_rebuild_count = 0;
  trie.buildRecency(extended.data(), extended.size(), 30, 3, p, state, 3);
  if (validation_rebuild_count != 0)
    throw std::runtime_error("all-live growth forced rebuild");

  // The previous advance created misses under the latest topology. With no
  // intervening growth, re-querying them must not repeat a full root traversal.
  trie.buildRecency(extended.data(), extended.size(), 30, 3, p, state, 3);
  if (validation_rebuild_count != 0)
    throw std::runtime_error("advance failed to refresh miss provenance");
  trie.insert(seed.data(), seed.size());
  trie.buildRecency(extended.data(), extended.size(), 30, 3, p, state, 3);
  if (validation_rebuild_count != 0)
    throw std::runtime_error("frequency-only update forced rebuild");

  const std::vector<int32_t> learned{10, 20, 30, 40, 50, 60};
  trie.insert(learned.data(), learned.size());
  const auto result =
      trie.buildRecency(extended.data(), extended.size(), 30, 3, p, state, 3);
  if (validation_rebuild_count != 1 ||
      result.token != std::vector<int32_t>{30, 40, 50, 60})
    throw std::runtime_error(
        "relevant growth did not perform one correct rebuild");
  std::cout << "rebuild-count controls passed\n";
}
