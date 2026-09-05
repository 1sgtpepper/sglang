"""Proposed Python/FFI regression; syntax checked, NOT executed in this audit.

Run in an installed, pinned SGLang checkout with:
    python -m pytest -q /path/to/ci_regression.py
For upstream integration, place the test method in the existing
TestNgramCorpusIncremental class in test/registered/unit/spec/test_ngram_corpus.py.
"""

import unittest

import numpy as np

from sglang.srt.speculative.cpp_ngram.ngram_corpus import NgramCorpus


class TestCachedMissAfterInsert(unittest.TestCase):
    def test_new_suffix_after_insert(self):
        for mode in ("BFS", "PROB"):
            for partial in (False, True):
                for appended in range(3):
                    with self.subTest(mode=mode, partial=partial, appended=appended):
                        corpus = NgramCorpus(
                            max_trie_depth=18,
                            min_bfs_breadth=1,
                            max_bfs_breadth=1,
                            draft_token_num=4,
                            match_type=mode,
                            capacity=10000,
                        )
                        if partial:
                            corpus.batch_put([[20, 99]])
                            corpus.synchronize()
                        corpus.batch_get(["cached"], [[10, 20]], [2])
                        learned = [10, 20, 30, 40, 50, 60, 61]
                        query = learned[: 2 + appended]
                        distractor = query[1:] + [70, 80, 90]
                        corpus.batch_put([learned, distractor, distractor])
                        corpus.synchronize()
                        ids, mask = corpus.batch_get(["cached"], [query], [len(query)])
                        fresh_ids, fresh_mask = corpus.batch_get(
                            ["fresh"], [query], [len(query)]
                        )
                        expected = learned[1 + appended : 5 + appended]
                        np.testing.assert_array_equal(ids, expected)
                        np.testing.assert_array_equal(fresh_ids, expected)
                        np.testing.assert_array_equal(mask, fresh_mask)
                        np.testing.assert_array_equal(
                            mask.reshape(4, 4), np.tril(np.ones((4, 4), dtype=bool))
                        )


if __name__ == "__main__":
    unittest.main()
