#include "trie.h"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sglang::ngram;
using Clock = std::chrono::steady_clock;
using Tokens = std::vector<int32_t>;

int main(int argc, char** argv) {
  if (argc != 2) throw std::runtime_error("expected variant name");
  constexpr size_t iterations = 2000;
  for (size_t depth : {18, 64}) {
    for (const std::string workload : {"static-live", "static-miss", "growth-live", "growth-miss"}) {
      Param p{};
      p.max_trie_depth = depth;
      p.min_bfs_breadth = p.max_bfs_breadth = 1;
      p.draft_token_num = 4;
      p.match_type = "BFS";
      Trie trie(50000, p);
      const bool live = workload.ends_with("live");
      Tokens seed;
      for (size_t i = 0; i < (live ? 4 * depth : 7); ++i) seed.push_back(1 + i % 7);
      trie.insert(seed.data(), seed.size());
      Tokens request;
      request.reserve(depth + iterations + 1);
      for (size_t i = 0; i < depth; ++i) request.push_back(1 + i % 7);
      MatchState state;
      trie.buildRecency(request.data(), depth, request.back(), 3, p, state, depth);
      int64_t query_ns = 0;
      int64_t checksum = 0;
      const auto cycle_start = Clock::now();
      for (size_t i = 0; i < iterations; ++i) {
        if (workload.starts_with("growth")) {
          // Disjoint vocabulary changes topology but cannot change query results.
          const Tokens other{1000 + static_cast<int32_t>(3 * i),
                             1001 + static_cast<int32_t>(3 * i),
                             1002 + static_cast<int32_t>(3 * i)};
          trie.insert(other.data(), other.size());
        }
        request.push_back(1 + request.size() % 7);
        const auto* tail = request.data() + request.size() - depth;
        const auto start = Clock::now();
        const auto result = trie.buildRecency(tail, depth, request.back(), 3, p, state, request.size());
        query_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
        checksum += result.token.back();
        if (i % 64 == 0) {
          MatchState fresh;
          const auto expected = trie.buildRecency(tail, depth, request.back(), 3, p, fresh, request.size());
          if (result.token != expected.token || result.mask != expected.mask)
            throw std::runtime_error("benchmark workload changed semantics");
        }
      }
      const auto cycle_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - cycle_start).count();
      std::cout << argv[1] << ',' << workload << ',' << depth << ',' << iterations << ',' << query_ns << ','
                << cycle_ns << ',' << checksum << '\n';
    }
  }
}
