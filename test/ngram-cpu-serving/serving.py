"""Matched HTTP serving validation on an AMX CPU, using the real SGLang worker."""

import gzip
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import requests
from huggingface_hub import snapshot_download

REVISIONS = {
    "baseline": "bd16c22a04b0eb9bc2e775795bda6b11727a5d38",
    "patched": "17b250829c15b11563c0ff6098c97962ec6c2aa7",
}
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
URL = "http://127.0.0.1:30000"
PROMPTS = (
    "The capital of France is",
    "Once upon a time, there was a",
    "The three primary colors are",
    "def fibonacci(n):",
    "A Python function that adds two numbers:\n",
    "Water freezes at a temperature of",
    "Translate hello into French:",
    "Repeat this phrase: red blue green. red blue green.",
)


@contextmanager
def server(model_path, variant, mode, output, label):
    subprocess.run(
        ["git", "checkout", "--force", "--detach", REVISIONS[variant]], check=True
    )
    assert (
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        == REVISIONS[variant]
    )
    command = [
        "sglang",
        "serve",
        "--model-path",
        model_path,
        "--device",
        "cpu",
        "--attention-backend",
        "intel_amx",
        "--dtype",
        "bfloat16",
        "--host",
        "127.0.0.1",
        "--port",
        "30000",
        "--context-length",
        "1024",
        "--max-total-tokens",
        "4096",
        "--max-running-requests",
        "2",
        "--mem-fraction-static",
        "0.5",
        "--disable-overlap-schedule",
    ]
    if mode is not None:
        command += [
            "--speculative-algorithm",
            "NGRAM",
            "--speculative-num-draft-tokens",
            "8",
            "--speculative-ngram-match-type",
            mode,
            "--speculative-ngram-capacity",
            "100000",
        ]
    (output / f"{label}-command.json").write_text(json.dumps(command, indent=2))
    with (output / f"{label}-server.log").open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                assert process.poll() is None, f"server exited: {label}"
                try:
                    if requests.get(URL + "/health", timeout=2).status_code == 200:
                        break
                except requests.ConnectionError:
                    pass
                time.sleep(0.2)
            else:
                raise TimeoutError(f"server startup: {label}")
            response = requests.get(URL + "/server_info", timeout=30)
            response.raise_for_status()
            info = response.json()
            assert info["attention_backend"] == "intel_amx", info
            assert info["device"] == "cpu", info
            assert info["speculative_algorithm"] == ("NGRAM" if mode else None), info
            (output / f"{label}-server-info.json").write_text(
                json.dumps(info, indent=2)
            )
            yield
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)


def generate(prompt, tokens=32):
    response = requests.post(
        URL + "/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": tokens,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "logprob_start_len": -1,
        },
        timeout=120,
    )
    response.raise_for_status()
    result = response.json()
    assert result["meta_info"]["completion_tokens"] == tokens, result
    result["token_ids"] = [
        entry[1] for entry in result["meta_info"]["output_token_logprobs"]
    ]
    assert len(result["token_ids"]) == tokens, result
    return result


def benchmark(model_path, output, label, concurrency):
    destination = output / f"{label}-c{concurrency}.jsonl"
    command = [
        sys.executable,
        "-m",
        "sglang.benchmark.serving",
        "--backend",
        "sglang",
        "--base-url",
        URL,
        "--model",
        model_path,
        "--dataset-name",
        "random",
        "--num-prompts",
        "10",
        "--max-concurrency",
        str(concurrency),
        "--request-rate",
        "inf",
        "--random-input-len",
        "128",
        "--random-output-len",
        "32",
        "--random-range-ratio",
        "1",
        "--warmup-requests",
        "2",
        "--seed",
        "42",
        "--tokenize-prompt",
        "--output-file",
        str(destination),
    ]
    with (output / f"{label}-c{concurrency}.log").open("w") as log:
        subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180
        )
    results = [json.loads(line) for line in destination.read_text().splitlines()]
    assert len(results) == 1, results
    assert results[0]["completed"] == 10, results
    assert results[0]["total_output_tokens"] == 320, results
    return results[0]


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    model_path = snapshot_download(
        MODEL,
        revision=MODEL_REVISION,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "merges.txt",
            "vocab.json",
            "tokenizer*",
        ],
    )
    reference = []
    with server(model_path, "baseline", None, output, "reference"):
        for prompt in PROMPTS:
            reference.append(generate(prompt))
    (output / "reference.json").write_text(json.dumps(reference, indent=2))
    parity = []
    timings = []
    for trial in range(3):
        order = ("baseline", "patched") if trial % 2 == 0 else ("patched", "baseline")
        for mode in ("BFS", "PROB"):
            for variant in order:
                label = f"{variant}-{mode}-{trial}"
                with server(model_path, variant, mode, output, label):
                    if trial == 0:
                        for index, prompt in enumerate(PROMPTS):
                            actual = generate(prompt)
                            parity.append(
                                dict(
                                    variant=variant,
                                    mode=mode,
                                    prompt=index,
                                    result=actual,
                                )
                            )
                            (output / "parity.json").write_text(
                                json.dumps(parity, indent=2)
                            )
                            assert (
                                actual["token_ids"] == reference[index]["token_ids"]
                            ), (label, index, actual, reference[index])
                            assert actual["text"] == reference[index]["text"], (
                                label,
                                index,
                            )
                    for concurrency in (1, 2):
                        result = benchmark(model_path, output, label, concurrency)
                        timings.append(
                            dict(
                                variant=variant,
                                mode=mode,
                                trial=trial,
                                concurrency=concurrency,
                                result=result,
                            )
                        )
                        (output / "timings.json").write_text(
                            json.dumps(timings, indent=2)
                        )
                    if variant == "patched" and trial == 0 and mode == "BFS":
                        profile_dir = output / "profiles"
                        response = requests.post(
                            URL + "/start_profile",
                            json={
                                "activities": ["CPU"],
                                "output_dir": str(profile_dir),
                                "record_shapes": True,
                            },
                            timeout=30,
                        )
                        response.raise_for_status()
                        generate(PROMPTS[0], tokens=4)
                        requests.post(
                            URL + "/stop_profile", timeout=120
                        ).raise_for_status()
                        traces = list(profile_dir.rglob("*.json")) + list(
                            profile_dir.rglob("*.json.gz")
                        )
                        assert traces, "missing server profile"
                        for trace in traces:
                            opener = gzip.open if trace.suffix == ".gz" else open
                            with opener(trace, "rt") as file:
                                assert json.load(file)["traceEvents"], trace
                print(f"{label}: passed", flush=True)
    summary = []
    for mode in ("BFS", "PROB"):
        for concurrency in (1, 2):
            row = dict(mode=mode, concurrency=concurrency)
            for variant in REVISIONS:
                samples = [
                    entry["result"]["output_throughput"]
                    for entry in timings
                    if entry["mode"] == mode
                    and entry["concurrency"] == concurrency
                    and entry["variant"] == variant
                ]
                assert len(samples) == 3
                row[variant] = dict(
                    median=statistics.median(samples),
                    minimum=min(samples),
                    maximum=max(samples),
                )
            summary.append(row)
    (output / "summary.json").write_text(
        json.dumps(
            dict(
                model=MODEL,
                model_revision=MODEL_REVISION,
                source=REVISIONS,
                summary=summary,
            ),
            indent=2,
        )
    )
    print(
        "32 serving parity cases, 24 serving benchmarks and CPU profile passed",
        flush=True,
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
