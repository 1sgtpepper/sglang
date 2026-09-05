"""Compare bounded, source-pinned alternatives; run only on fork CI."""

import csv
import io
import json
import random
import statistics
import subprocess
import tempfile
from pathlib import Path

BASE = "bd16c22a04b0eb9bc2e775795bda6b11727a5d38"
ROOT = Path(__file__).resolve().parents[2]
TESTS = Path(__file__).resolve().parent
NATIVE = Path("python/sglang/kernels/jit/csrc/ngram_corpus")


def run(command, *, expected=0, cwd=ROOT):
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, timeout=180
    )
    print("$", " ".join(map(str, command)), flush=True)
    print(result.stdout + result.stderr, end="", flush=True)
    assert result.returncode == expected, (command, result.returncode, expected)
    return result.stdout


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


def main():
    variants = ("baseline", "growth", "any-null", "always-rebuild")
    output = ROOT / "comparison-results"
    output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ngram-comparison-") as directory:
        tmp = Path(directory)
        binaries = {}
        for variant in variants:
            tree = tmp / variant
            source = tree / NATIVE
            source.mkdir(parents=True)
            for path in sorted((ROOT / NATIVE).iterdir()):
                if path.name == "ngram_corpus_ffi.cpp" or path.suffix not in (
                    ".cpp",
                    ".h",
                ):
                    continue
                data = subprocess.check_output(
                    ["git", "show", f"{BASE}:{NATIVE / path.name}"], cwd=ROOT
                )
                (source / path.name).write_bytes(data)
            cpp = source / "trie.cpp"
            if variant == "growth":
                run(
                    [
                        "git",
                        "apply",
                        "--unidiff-zero",
                        str(TESTS / "cached-miss.patch"),
                    ],
                    cwd=tree,
                )
            elif variant == "any-null":
                cpp.write_text(
                    replace_once(
                        cpp.read_text(),
                        "if (ref.ptr && !resolve(state, ref))",
                        "if (!resolve(state, ref))",
                    )
                )
            elif variant == "always-rebuild":
                cpp.write_text(
                    replace_once(
                        cpp.read_text(),
                        "if (can_advance && advanceMatchState_",
                        "if (false && can_advance && advanceMatchState_",
                    )
                )
            objects = []
            for path in sorted(source.glob("*.cpp")):
                obj = tree / (path.stem + ".o")
                run(
                    [
                        "g++",
                        "-std=c++20",
                        "-O3",
                        "-pthread",
                        "-I",
                        str(source),
                        "-c",
                        str(path),
                        "-o",
                        str(obj),
                    ]
                )
                objects.append(str(obj))
            for name in ("regression", "causal", "benchmark"):
                binary = tree / name
                run(
                    [
                        "g++",
                        "-std=c++20",
                        "-O3",
                        "-pthread",
                        "-I",
                        str(source),
                        str(TESTS / f"{name}.cc"),
                        *objects,
                        "-o",
                        str(binary),
                    ]
                )
                if name == "regression":
                    result = run(
                        [str(binary)], expected=1 if variant == "baseline" else 0
                    )
                    expected = (
                        "checks=33500 failures=2790"
                        if variant == "baseline"
                        else "checks=35352 failures=0"
                    )
                    assert result.strip().splitlines()[-1] == expected
                elif name == "causal":
                    run([str(binary), variant])
                else:
                    binaries[variant] = binary
        # Rebuild counts run in a separate instrumented copy, after all timing
        # executables were linked against uninstrumented production objects.
        tree = tmp / "growth"
        source = tree / NATIVE
        cpp = source / "trie.cpp"
        production = cpp.read_text()
        entry = "void Trie::rebuildMatchState_(const int32_t* context, size_t len, MatchState& state, size_t total_len) const {"
        instrumented = (
            "#include <cstddef>\nextern size_t validation_rebuild_count;\n"
            + replace_once(
                production, entry, entry + "\n  ++::validation_rebuild_count;"
            )
        )
        for mutation in (False, True):
            text = instrumented
            if mutation:
                text = replace_once(
                    text,
                    "  state.processed_total_len = total_len;\n  state.growth_epoch = growth_epoch_;\n  return true;",
                    "  state.processed_total_len = total_len;\n  return true;",
                )
            diagnostic = tree / "diagnostic.cpp"
            diagnostic.write_text(text)
            binary = tree / "rebuild-counts"
            objects = [
                str(tree / (name + ".o"))
                for name in ("ngram", "result", "suffix_automaton")
            ]
            run(
                [
                    "g++",
                    "-std=c++20",
                    "-O3",
                    "-pthread",
                    "-I",
                    str(source),
                    str(TESTS / "rebuild_counts.cc"),
                    str(diagnostic),
                    *objects,
                    "-o",
                    str(binary),
                ]
            )
            if mutation:
                result = subprocess.run(
                    [str(binary)], capture_output=True, text=True, timeout=30
                )
                assert result.returncode != 0
                assert "advance failed to refresh miss provenance" in result.stderr
                print(
                    "Missing advance epoch update: rebuild-count control caught regression",
                    flush=True,
                )
            else:
                run([str(binary)])
        # Warm each executable once; retain seven interleaved measurements.
        for variant in variants:
            run([str(binaries[variant]), variant])
        rng = random.Random(1701)
        samples = []
        for repetition in range(7):
            order = list(variants)
            rng.shuffle(order)
            for variant in order:
                text = run([str(binaries[variant]), variant])
                for row in csv.reader(io.StringIO(text)):
                    name, workload, depth, iterations, query_ns, cycle_ns, checksum = (
                        row
                    )
                    samples.append(
                        dict(
                            variant=name,
                            workload=workload,
                            depth=int(depth),
                            repetition=repetition,
                            query_ns_per_call=int(query_ns) / int(iterations),
                            cycle_ns_per_call=int(cycle_ns) / int(iterations),
                            checksum=int(checksum),
                        )
                    )
        summary = []
        for depth in (18, 64):
            for workload in (
                "static-live",
                "static-miss",
                "growth-live",
                "growth-miss",
            ):
                for variant in variants:
                    group = [
                        s
                        for s in samples
                        if (s["depth"], s["workload"], s["variant"])
                        == (depth, workload, variant)
                    ]
                    summary.append(
                        dict(
                            depth=depth,
                            workload=workload,
                            variant=variant,
                            median_query_ns=statistics.median(
                                s["query_ns_per_call"] for s in group
                            ),
                            min_query_ns=min(s["query_ns_per_call"] for s in group),
                            max_query_ns=max(s["query_ns_per_call"] for s in group),
                        )
                    )
        report = dict(
            base=BASE,
            samples=samples,
            summary=summary,
            limits="CPU native microbenchmark, controlled disjoint growth; not serving throughput.",
        )
        (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
