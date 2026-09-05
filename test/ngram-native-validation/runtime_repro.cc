#include "ngram.h"
#include <iostream>
using namespace sglang::ngram;
int main() {
  bool ok = true;
  for (const std::string mode : {"BFS", "PROB"}) {
    Param p{};
    p.max_trie_depth = 18;
    p.min_bfs_breadth = p.max_bfs_breadth = 1;
    p.draft_token_num = 4;
    p.match_type = mode;
    Ngram corpus(10000, p);
    corpus.batchMatch({1}, {{10, 20}}, {2});
    corpus.asyncInsert({{10, 20, 30, 40, 50, 60}, {30, 70, 80, 90}, {30, 70, 80, 90}});
    corpus.synchronize();
    auto cached = corpus.batchMatch({1}, {{10, 20, 30}}, {3});
    auto fresh = corpus.batchMatch({2}, {{10, 20, 30}}, {3});
    std::cout << mode << " cached=";
    for (int token : cached.token) std::cout << token << ',';
    std::cout << " fresh=";
    for (int token : fresh.token) std::cout << token << ',';
    const bool pass = cached.token == std::vector<int32_t>{30, 40, 50, 60} &&
                      cached.token == fresh.token && cached.mask == fresh.mask;
    std::cout << " " << (pass ? "PASS" : "FAIL: newly inserted longest match is omitted") << '\n';
    ok &= pass;
  }
  return ok ? 0 : 1;
}
