"""Fork-only real-model verifier checks and corpus API timing, not serving."""

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np

from sglang.srt.speculative.cpp_ngram.ngram_corpus import NgramCorpus

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
PROMPTS = (
    "The capital of France is",
    "The first five positive integers are",
    "A Python function that adds two numbers:\n",
    "Water freezes at a temperature of",
    "Translate hello into French:",
    "The opposite of hot is",
    "Repeat this phrase: red blue green. red blue green.",
    "Complete the sentence: A triangle has",
)


def check_model(variant, output):
    import torch
    from torch.utils.cpp_extension import load
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    torch.set_num_threads(2)
    torch.manual_seed(0)
    load(
        name="ngram_verification_validation",
        sources=[
            str(Path.cwd() / "python/sglang/kernels/aot/csrc/cpu/spec.cpp"),
            str(Path(__file__).with_name("verification_binding.cpp")),
        ],
        extra_cflags=["-O2"],
        is_python_module=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        dtype=torch.float32,
        attn_implementation="eager",
        use_safetensors=True,
    ).eval()
    # The checkpoint's repetition penalty changes raw greedy predictions.
    generation = GenerationConfig(
        max_new_tokens=4,
        do_sample=False,
        use_cache=False,
        repetition_penalty=1.0,
        eos_token_id=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    rows = []
    with torch.inference_mode():
        for prompt_id, prompt in enumerate(PROMPTS):
            prefix = tokenizer.encode(prompt)
            reference = model.generate(
                torch.tensor([prefix]),
                attention_mask=torch.ones((1, len(prefix)), dtype=torch.int64),
                generation_config=generation,
            )[0, len(prefix) :].tolist()
            for mode in ("BFS", "PROB"):
                # A different first token guarantees rejection of the short branch.
                conflict = (reference[0] + 1) % model.config.vocab_size
                corpus = NgramCorpus(
                    max_trie_depth=18,
                    min_bfs_breadth=1,
                    max_bfs_breadth=1,
                    draft_token_num=4,
                    match_type=mode,
                    capacity=10000,
                )
                corpus.batch_get(["cached"], [prefix[-4:-1]], [len(prefix) - 1])
                corpus.batch_put(
                    [prefix[-4:] + reference]
                    + [[prefix[-1], conflict, conflict, conflict]] * 2
                )
                corpus.synchronize()
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU],
                    record_shapes=True,
                ) as profile:
                    with torch.profiler.record_function("ngram_match"):
                        ids, masks = corpus.batch_get(
                            ["cached", "fresh"], [prefix[-4:]] * 2, [len(prefix)] * 2
                        )
                    candidates = torch.tensor(ids, dtype=torch.int64).reshape(2, 4)
                    np.testing.assert_array_equal(
                        candidates[1].numpy(), [prefix[-1]] + reference[:3]
                    )
                    mask = torch.tensor(masks, dtype=torch.bool).reshape(2, 4, 4)
                    targets = torch.empty_like(candidates)
                    with torch.profiler.record_function("target_forward"):
                        for row, node in itertools.product(range(2), range(4)):
                            path = candidates[row][mask[row, node]].tolist()
                            assert path[0] == prefix[-1]
                            logits = model(torch.tensor([prefix[:-1] + path])).logits
                            targets[row, node] = logits[0, -1].argmax()
                    with torch.profiler.record_function("greedy_verify"):
                        retrieval = torch.full((2, 4), -1, dtype=torch.int64)
                        child = torch.full_like(retrieval, -1)
                        sibling = torch.full_like(retrieval, -1)
                        positions = torch.empty(8, dtype=torch.int64)
                        torch.ops.ngram_validation.reconstruct(
                            mask.flatten(),
                            torch.tensor([len(prefix)] * 2),
                            positions,
                            retrieval,
                            child,
                            sibling,
                            2,
                            4,
                        )
                        predicts = torch.full((8,), -1, dtype=torch.int32)
                        accepted = torch.full((2, 4), -1, dtype=torch.int32)
                        counts = torch.empty(2, dtype=torch.int32)
                        torch.ops.ngram_validation.verify(
                            predicts,
                            accepted,
                            counts,
                            candidates,
                            retrieval,
                            child,
                            sibling,
                            targets,
                        )
                emitted = []
                for row, count in enumerate(counts.tolist()):
                    tokens = predicts[accepted[row, : count + 1].long()].tolist()
                    assert tokens == reference[: len(tokens)], (tokens, reference)
                    emitted.append(tokens)
                expected = [0, 3] if variant == "baseline" else [3, 3]
                assert counts.tolist() == expected, (prompt_id, mode, counts)
                rows.append(
                    dict(
                        prompt_id=prompt_id,
                        mode=mode,
                        reference=reference,
                        drafts=candidates.tolist(),
                        accepted=counts.tolist(),
                        emitted=emitted,
                    )
                )
                if prompt_id == 0 and mode == "BFS":
                    profile.export_chrome_trace(str(output / f"{variant}-profile.json"))
                    (output / f"{variant}-profile.txt").write_text(
                        profile.key_averages().table(
                            sort_by="self_cpu_time_total", row_limit=25
                        )
                    )
                del corpus
    (output / f"{variant}-model.json").write_text(
        json.dumps(
            dict(
                model=MODEL,
                revision=REVISION,
                variant=variant,
                generation=generation.to_diff_dict(),
                prompts=PROMPTS,
                cases=rows,
            ),
            indent=2,
        )
    )
    print(f"{variant}: {len(rows)} real-model cases passed", flush=True)


def benchmark(variant, trial, output):
    rows = []
    for batch_size, live, growth in itertools.product(
        (1, 16, 48), (False, True), (False, True)
    ):
        corpus = NgramCorpus(
            max_trie_depth=18,
            min_bfs_breadth=1,
            max_bfs_breadth=1,
            draft_token_num=12,
            match_type="BFS",
            capacity=10000,
        )
        corpus.batch_put([list(range(1, 32))])
        corpus.synchronize()
        context = list(range(1, 19)) if live else [20001, 20002, 20003]
        keys = [str(i) for i in range(batch_size)]
        inputs = [context] * batch_size
        lengths = [len(context)] * batch_size
        for _ in range(50):
            ids, mask = corpus.batch_get(keys, inputs, lengths)
        expected_ids, expected_mask = ids.copy(), mask.copy()
        query_ns = 0
        update_ns = 0
        for iteration in range(300):
            if growth:
                start = time.perf_counter_ns()
                corpus.batch_put([[100000 + iteration, 200000 + iteration]])
                corpus.synchronize()
                update_ns += time.perf_counter_ns() - start
            start = time.perf_counter_ns()
            ids, mask = corpus.batch_get(keys, inputs, lengths)
            query_ns += time.perf_counter_ns() - start
            np.testing.assert_array_equal(ids, expected_ids)
            np.testing.assert_array_equal(mask, expected_mask)
        digest = hashlib.sha256(ids.tobytes() + mask.tobytes()).hexdigest()
        rows.append(
            dict(
                batch_size=batch_size,
                live=live,
                growth=growth,
                query_us=query_ns / 300 / 1000,
                update_us=update_ns / 300 / 1000,
                checksum=digest,
            )
        )
        del corpus
    (output / f"{variant}-timing-{trial}.json").write_text(json.dumps(rows, indent=2))
    print(f"{variant}: timing trial {trial} passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("model", "benchmark"))
    parser.add_argument("variant", choices=("baseline", "patched"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--trial", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "model":
        check_model(args.variant, args.output)
    else:
        benchmark(args.variant, args.trial, args.output)
