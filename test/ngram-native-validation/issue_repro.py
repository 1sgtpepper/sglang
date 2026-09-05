import numpy as np

from sglang.srt.speculative.cpp_ngram.ngram_corpus import NgramCorpus

ok = True
for mode in ("BFS", "PROB"):
    corpus = NgramCorpus(
        max_trie_depth=18,
        min_bfs_breadth=1,
        max_bfs_breadth=1,
        draft_token_num=4,
        match_type=mode,
        capacity=10000,
    )
    initial, _ = corpus.batch_get(["cached"], [[10, 20]], [2])
    np.testing.assert_array_equal(initial, [20, 0, 0, 0])

    corpus.batch_put([[10, 20, 30, 40, 50, 60], [30, 70, 80, 90], [30, 70, 80, 90]])
    corpus.synchronize()
    ids, masks = corpus.batch_get(
        ["cached", "fresh"], [[10, 20, 30], [10, 20, 30]], [3, 3]
    )
    cached, fresh = ids.reshape(2, 4).tolist()
    expected = [30, 40, 50, 60]
    assert fresh == expected, (mode, "fresh control", fresh)
    chain_mask = np.tril(np.ones((4, 4), dtype=bool))
    np.testing.assert_array_equal(masks.reshape(2, 4, 4), [chain_mask, chain_mask])

    # Change only request state: the corpus and query stay the same.
    corpus.erase_match_state(["cached"])
    erased, erased_mask = corpus.batch_get(["cached"], [[10, 20, 30]], [3])
    np.testing.assert_array_equal(erased, expected)
    np.testing.assert_array_equal(erased_mask.reshape(4, 4), chain_mask)
    print(f"{mode}: cached={cached}, fresh={fresh}, after_erasure={erased.tolist()}")
    ok &= cached == expected

assert ok, "cached request omits newly inserted longer context"
