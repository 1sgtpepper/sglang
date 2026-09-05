"""Real NGRAM FFI -> production CPU tree reconstruction -> greedy verification.

Target predictions are controlled fixtures, not a serving/model benchmark.
"""

import json
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from sglang.srt.speculative.cpp_ngram.ngram_corpus import NgramCorpus


def verify(ids, mask, target, width):
    candidates = torch.tensor(ids, dtype=torch.int64).reshape(2, width)
    retrieval = torch.full((2, width), -1, dtype=torch.int64)
    child = torch.full_like(retrieval, -1)
    sibling = torch.full_like(retrieval, -1)
    positions = torch.empty(2 * width, dtype=torch.int64)
    torch.ops.ngram_validation.reconstruct(
        torch.tensor(mask, dtype=torch.bool),
        torch.tensor([3, 3], dtype=torch.int64),
        positions,
        retrieval,
        child,
        sibling,
        2,
        width,
    )
    targets = torch.tensor(
        [[target.get(int(token), 999) for token in row] for row in candidates],
        dtype=torch.int64,
    )
    predicts = torch.full((2 * width,), -1, dtype=torch.int32)
    accepted = torch.full((2, width), -1, dtype=torch.int32)
    counts = torch.empty(2, dtype=torch.int32)
    torch.ops.ngram_validation.verify(
        predicts, accepted, counts, candidates, retrieval, child, sibling, targets
    )
    emitted = []
    for row, count in enumerate(counts.tolist()):
        indices = accepted[row, : count + 1].to(torch.int64)
        tokens = predicts[indices].tolist()
        expected = []
        current = 30
        for _ in tokens:
            current = target[current]
            expected.append(current)
        assert tokens == expected, (tokens, expected)
        emitted.append(tokens)
    return counts.tolist(), emitted


def main():
    variant = sys.argv[1]
    assert variant in ("baseline", "patched")
    root = Path(__file__).resolve().parents[2]
    load(
        name="ngram_verification_validation",
        sources=[
            str(root / "python/sglang/kernels/aot/csrc/cpu/spec.cpp"),
            str(Path(__file__).with_name("verification_binding.cpp")),
        ],
        extra_cflags=["-O2"],
        is_python_module=False,
        verbose=True,
    )
    torch.set_num_threads(1)
    for mode in ("BFS", "PROB"):
        for external in (False, True):
            width = 6 if external else 4
            corpus = NgramCorpus(
                max_trie_depth=18,
                min_bfs_breadth=1,
                max_bfs_breadth=1,
                draft_token_num=width,
                match_type=mode,
                capacity=10000,
                external_sam_budget=2 if external else 0,
            )
            if external:
                count = corpus.load_external_corpus_named(
                    "unrelated", [[30, 100, 101, 102]]
                )
                corpus.commit_external_corpus_load("unrelated", count)
            corpus.batch_get(["cached"], [[10, 20]], [2])
            corpus.batch_put(
                [[10, 20, 30, 40, 50, 60], [30, 70, 80, 90], [30, 70, 80, 90]]
            )
            corpus.synchronize()
            ids, mask = corpus.batch_get(
                ["cached", "fresh"], [[10, 20, 30], [10, 20, 30]], [3, 3]
            )
            for target_name, target, expected in (
                (
                    "long-context",
                    {30: 40, 40: 50, 50: 60, 60: 61},
                    [0, 3] if variant == "baseline" else [3, 3],
                ),
                (
                    "short-context-control",
                    {30: 70, 70: 80, 80: 90, 90: 91},
                    [3, 0] if variant == "baseline" else [0, 0],
                ),
            ):
                counts, emitted = verify(ids, mask, target, width)
                assert counts == expected, (variant, mode, external, counts, expected)
                print(
                    json.dumps(
                        dict(
                            variant=variant,
                            mode=mode,
                            external=external,
                            target=target_name,
                            accepted_drafts=counts,
                            emitted=emitted,
                        )
                    ),
                    flush=True,
                )
    print("downstream verification: 8 cases passed", flush=True)


if __name__ == "__main__":
    main()
