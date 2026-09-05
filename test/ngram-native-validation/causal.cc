#include "trie.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sglang::ngram;
using Tokens = std::vector<int32_t>;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

Result query(Trie& trie, const Param& p, const Tokens& tokens, MatchState& state) {
  return p.match_type == "BFS"
             ? trie.buildRecency(tokens.data(), tokens.size(), tokens.back(), 3, p, state, tokens.size())
             : trie.buildFrequency(tokens.data(), tokens.size(), tokens.back(), 3, p, state, tokens.size());
}

void insert(Trie& trie, const Tokens& tokens) {
  trie.insert(tokens.data(), tokens.size());
}

int main(int argc, char** argv) {
  require(argc == 2, "expected baseline or corrected variant");
  const bool baseline = std::string(argv[1]) == "baseline";
  size_t interventions = 0;
  for (const std::string mode : {"BFS", "PROB"}) {
    Param p{};
    p.max_trie_depth = 18;
    p.min_bfs_breadth = p.max_bfs_breadth = 1;
    p.draft_token_num = 4;
    p.match_type = mode;
    for (size_t appended = 0; appended <= 2; ++appended) {
      Trie trie(10000, p);
      insert(trie, {20, 99});
      const Tokens previous{10, 20};
      MatchState original;
      query(trie, p, previous, original);
      require(original.anchors[0].ptr != nullptr && original.anchors[1].ptr == nullptr, "partial miss fixture");
      const auto live = original.anchors[0];
      const Tokens learned{10, 20, 30, 40, 50, 60, 61};
      Tokens current(learned.begin(), learned.begin() + 2 + appended);
      Tokens distractor(current.begin() + 1, current.end());
      distractor.insert(distractor.end(), {70, 80, 90});
      insert(trie, learned);
      insert(trie, distractor);
      insert(trie, distractor);
      require(live.ptr->version == live.version, "live anchor was not evicted/reused");

      MatchState fresh_previous;
      query(trie, p, previous, fresh_previous);
      require(fresh_previous.trie_epoch == original.trie_epoch, "no reset occurred");
      require(fresh_previous.anchors[0].ptr == live.ptr, "live suffix identity stayed fixed");
      require(fresh_previous.anchors[1].ptr != nullptr, "old missing suffix now exists");

      // Change only the old negative observation, preserving live references,
      // request length and epochs. No corpus, ranking or threading change.
      auto repaired = original;
      repaired.anchors[1] = fresh_previous.anchors[1];
      auto cached = original;
      const auto actual = query(trie, p, current, cached);
      const auto intervention = query(trie, p, current, repaired);
      MatchState fresh;
      const auto expected = query(trie, p, current, fresh);
      Tokens golden(learned.begin() + 1 + appended, learned.begin() + 5 + appended);
      require(expected.token == golden, "independent longest-context golden");
      require(intervention.token == golden && intervention.mask == expected.mask, "null-only repair did not restore result");
      require((actual.token != golden) == baseline, "unexpected baseline/corrected behavior");
      ++interventions;
    }

    // Unrelated topology growth cannot create the missing 10,20 suffix.
    Trie trie(10000, p);
    insert(trie, {20, 30, 40, 50, 60});
    MatchState cached;
    query(trie, p, {10, 20}, cached);
    insert(trie, {700, 800, 900});
    const auto actual = query(trie, p, {10, 20, 30}, cached);
    MatchState fresh;
    const auto expected = query(trie, p, {10, 20, 30}, fresh);
    require(actual.token == expected.token && actual.mask == expected.mask, "unrelated-growth negative control");
  }
  std::cout << "null-only interventions=" << interventions << " negative-controls=2 passed\n";
}
