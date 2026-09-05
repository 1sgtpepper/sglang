// Behavioral checks against the real native implementation, not an algorithm
// replica.
#include "ngram.h"
#include <algorithm>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

using sglang::ngram::MatchState;
using sglang::ngram::Ngram;
using sglang::ngram::Param;
using sglang::ngram::Result;
using sglang::ngram::Trie;
using Tokens = std::vector<int32_t>;

struct Checks {
  size_t checked = 0, failed = 0;
  void check(bool pass, const std::string &label) {
    ++checked;
    if (!pass && ++failed <= 6)
      std::cout << "FAIL " << label << '\n';
  }
};

Param params(const std::string &mode, size_t depth = 18, size_t breadth = 1) {
  Param p{};
  p.max_trie_depth = depth;
  p.min_bfs_breadth = 1;
  p.max_bfs_breadth = breadth;
  p.draft_token_num = 4;
  p.match_type = mode;
  return p;
}

Tokens tail(const Tokens &full, size_t depth) {
  return Tokens(full.end() - std::min(full.size(), depth), full.end());
}

Result query(Trie &trie, const Param &p, MatchState &state,
             const Tokens &full) {
  auto t = tail(full, p.max_trie_depth);
  return p.match_type == "BFS"
             ? trie.buildRecency(t.data(), t.size(), t.back(), 3, p, state,
                                 full.size())
             : trie.buildFrequency(t.data(), t.size(), t.back(), 3, p, state,
                                   full.size());
}

bool valid_mask(const Result &r) {
  const size_t n = r.token.size();
  if (n != 4 || r.mask.size() != n * n)
    return false;
  for (size_t i = 0; i < n; ++i) {
    if (r.mask[i * n] != 1 || r.mask[i * n + i] != 1)
      return false;
    for (size_t j = i + 1; j < n; ++j)
      if (r.mask[i * n + j] != 0)
        return false;
    if (i == 0)
      continue;
    size_t parent = 0;
    for (size_t j = 1; j < i; ++j)
      if (r.mask[i * n + j])
        parent = j;
    for (size_t j = 0; j < i; ++j)
      if (r.mask[i * n + j] != r.mask[parent * n + j])
        return false;
  }
  return true;
}

// Full public native API, including its insertion thread and completion fence.
// Mix empty and partially matched cached contexts; append 0, 1, or 2 tokens.
void golden_api(Checks &checks) {
  for (const std::string mode : {"BFS", "PROB"}) {
    for (bool partial : {false, true}) {
      for (size_t appended = 0; appended <= 2; ++appended) {
        auto p = params(mode);
        Ngram corpus(10000, p);
        if (partial) {
          corpus.asyncInsert({{20, 99}});
          corpus.synchronize();
        }
        corpus.batchMatch({1}, {{10, 20}}, {2});
        const Tokens learned{10, 20, 30, 40, 50, 60, 61};
        Tokens full(learned.begin(), learned.begin() + 2 + appended);
        Tokens distractor(full.begin() + 1, full.end());
        distractor.insert(distractor.end(), {70, 80, 90});
        corpus.asyncInsert({learned, distractor, distractor});
        corpus.synchronize();
        auto cached = corpus.batchMatch({1}, {full}, {full.size()});
        auto fresh = corpus.batchMatch({2}, {full}, {full.size()});
        Tokens expected(learned.begin() + 1 + appended,
                        learned.begin() + 5 + appended);
        auto label = mode + " partial=" + std::to_string(partial) +
                     " appended=" + std::to_string(appended);
        checks.check(fresh.token == expected, label + " independent golden");
        checks.check(cached.token == expected, label + " cached golden");
        checks.check(cached.mask == fresh.mask && valid_mask(cached),
                     label + " masks");
        corpus.eraseMatchState({1});
        auto erased = corpus.batchMatch({1}, {full}, {full.size()});
        checks.check(erased.token == expected && erased.mask == fresh.mask,
                     label + " erased state");
      }
    }
  }
}

// Independent oracle: enumerate all stored substrings from insertion inputs.
// No trie traversal or copy of the matcher is used to decide expected
// existence.
void add_oracle(std::set<Tokens> &oracle, const Tokens &data, size_t depth) {
  for (size_t start = 0; start < data.size(); ++start)
    for (size_t n = 1; n <= depth && start + n <= data.size(); ++n)
      oracle.emplace(data.begin() + start, data.begin() + start + n);
}

void oracle_queries(Checks &checks) {
  for (const std::string mode : {"BFS", "PROB"}) {
    for (size_t depth : {4, 8, 18}) {
      for (size_t breadth : {1, 8}) {
        auto p = params(mode, depth, breadth);
        Trie trie(50000, p); // Far above the bounded corpus below: no eviction.
        std::set<Tokens> oracle;
        std::map<size_t, MatchState> states;
        std::map<size_t, Tokens> requests;
        std::mt19937 rng(1701 + static_cast<unsigned>(depth));
        for (size_t step = 0; step < 240; ++step) {
          if (step == 120) {
            trie.reset();
            oracle.clear();
          } // Keep cached states.
          size_t id = step % 3;
          auto &full = requests[id];
          size_t appended = full.empty() ? 2
                                         : (step % 3 == 0   ? 0
                                            : step % 3 == 1 ? 1
                                                            : 3);
          for (size_t i = 0; i < appended; ++i)
            full.push_back(1 + rng() % 7);
          auto actual = query(trie, p, states[id], full);
          MatchState empty;
          auto fresh = query(trie, p, empty, full);
          const auto t = tail(full, depth);
          const auto label = mode + " depth=" + std::to_string(depth) +
                             " step=" + std::to_string(step);
          checks.check(actual.token == fresh.token && actual.mask == fresh.mask,
                       label + " fresh equivalence");
          checks.check(valid_mask(actual) && actual.token.front() == t.back(),
                       label + " tree shape");
          checks.check(states[id].anchors.size() == t.size(),
                       label + " anchor count");
          for (size_t d = 1; d <= t.size(); ++d) {
            Tokens suffix(t.end() - d, t.end());
            const auto &ref = states[id].anchors.at(d - 1);
            checks.check((ref.ptr != nullptr) == oracle.contains(suffix),
                         label + " suffix existence");
            if (ref.ptr) {
              Tokens path(d);
              const auto *node = ref.ptr;
              for (size_t i = d; i > 0; --i) {
                path[i - 1] = node->token;
                node = node->parent;
              }
              checks.check(path == suffix && ref.ptr->version == ref.version,
                           label + " anchor identity");
            }
          }
          // The actual worker queries before inserting its current request
          // tail. Every fifth iteration does no insertion; repetitions update
          // frequency only.
          if (step % 5 != 0) {
            trie.insert(t.data(), t.size());
            add_oracle(oracle, t, depth);
            if (step % 7 == 0)
              trie.insert(t.data(), t.size());
          }
          if (step % 4 == 0) {
            Tokens other{1 + static_cast<int>(rng() % 7),
                         1 + static_cast<int>(rng() % 7),
                         1 + static_cast<int>(rng() % 7),
                         1 + static_cast<int>(rng() % 7)};
            trie.insert(other.data(), other.size());
            add_oracle(oracle, other, depth);
          }
        }
      }
    }
  }
}

// Capacity pressure cannot use the all-ever-inserted oracle. Compare the
// stateful path to a fresh traversal of the SAME live trie after eviction.
void eviction_queries(Checks &checks) {
  for (const std::string mode : {"BFS", "PROB"}) {
    auto p = params(mode, 6, 8);
    Trie trie(150, p);
    MatchState cached;
    Tokens full{5000, 5001, 5002};
    for (int step = 0; step < 120; ++step) {
      Tokens inserted;
      for (int j = 0; j < 20; ++j)
        inserted.push_back(5000 + step * 20 + j);
      trie.insert(inserted.data(), inserted.size());
      if (step % 2 == 0)
        full.push_back(inserted[2]);
      auto actual = query(trie, p, cached, full);
      MatchState empty;
      auto fresh = query(trie, p, empty, full);
      checks.check(actual.token == fresh.token && actual.mask == fresh.mask,
                   mode + " eviction equivalence");
      checks.check(valid_mask(actual), mode + " eviction mask");
    }
  }
}

int main() {
  Checks checks;
  golden_api(checks);
  oracle_queries(checks);
  eviction_queries(checks);
  std::cout << "checks=" << checks.checked << " failures=" << checks.failed
            << '\n';
  return checks.failed == 0 ? 0 : 1;
}
