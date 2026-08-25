# `server_args.py` — style guideline and cleanup plan

Reference date for every age computation below: **2026-08-25**.
Target file: `python/sglang/srt/server_args.py` (10,745 lines at time of writing).

---

# Section 1 — Style guideline for `server_args.py`

## 1.1 File layout

The file has exactly one legal top-to-bottom order. Anything that does not fit
one of these slots does not belong in this file (see 1.3).

```
1.  License header, module docstring
2.  Imports
3.  Module constants
      3a. Extension points: a `Choices` list + its adder alias on the next line
      3b. Scalar defaults and sentinels  (immutable, not choice lists)
      3c. Deprecated aliases             (dated, scheduled for removal)
4.  (nothing — the adders live in 3a, on the line under their list)
5.  @dataclasses.dataclass class ServerArgs
      5a. Class docstring ("Adding new arguments")
      5b. Fields, grouped by `# ---` section banners
      5c. __post_init__ / resolve_once / replace_resolved / _declare
      5d. _run_resolution_pipeline  (the dispatcher)
      5e. _handle_* / _apply_* / _validate_* handlers, in dispatcher order
      5f. add_cli_args / from_cli_args
      5g. Derived properties and check_* methods
6.  Module-level helpers that take a ServerArgs  (e.g. compute_world_size)
7.  Process-global publish/accessor shims, prepare_server_args
8.  PortArgs and its own constants
```

Two consequences worth stating explicitly, because both are violated today:

- **No function definitions inside the constants block.** A `def` between two
  constants means the block is no longer scannable as a list of extension
  points.
- **Constants belonging to `PortArgs` stay next to `PortArgs`** (slot 8). The
  "constants at the top" rule governs `ServerArgs` choice lists, not every
  module-level assignment.

## 1.2 The constants block

**A module-level choice list means exactly one thing: this is an extension
point.** Out-of-tree platforms and plugins extend the runtime by appending to
it before `ServerArgs` is constructed. A list that nobody extends is not a
constant — it is a field's `choices=` value that got hoisted for no reason, and
it belongs inline in the field definition.

That gives the rule for both directions:

- **No adder means no adder is needed.** Do not add an adder to make a list
  "consistent". If a list has no adder, inline it into its field and delete
  the module-level name.
- **Do not add new adders.** A new extension point is a deliberate public-API
  decision, not a side effect of declaring a field with `choices=`. New choice
  lists are inlined; promoting one to an extension point is its own change with
  its own justification.

**Declare the list and its extensibility in one place.** Extensibility is a
property of the list, expressed by the `Choices` type, with the legacy adder
name bound on the line directly below it:

```python
ATTENTION_BACKEND_CHOICES = Choices([
    "triton",
    ...
])
add_attention_backend_choices = ATTENTION_BACKEND_CHOICES.add
```

`Choices` is a two-line `list` subclass in `arg_groups/arg_utils.py` whose only
method is `add(choices)`. Because the adder sits under its list rather than in
a second block, the two cannot drift out of order, and no lint is needed to
keep them aligned. `Choices` is a real `list`, so `isinstance`, `in`, indexing,
object identity, and argparse's `choices=` all behave exactly as today, and
`add_x_choices(["y"])` keeps its current signature and semantics.

Remaining rules:

- **Sort:** group by domain under `# ---` banners (load/quantization,
  attention, MoE/GEMM, cache/policy, LoRA, CP/DSA, mamba/linear-attention,
  transport), alphabetical within each group.
- **Never write `choices=SOME_LIST + ["extra"]`.** The concatenation is
  evaluated at class-definition time and produces a plain `list`, so that field
  silently stops tracking later extensions while its siblings keep tracking
  them. If a field needs one extra value, give it its own inline list, or add
  the value to the shared list.
- **No conditionals, env reads, or platform probes at module scope.** Import-time
  branching makes a constant depend on import order and forces tests into
  `importlib.reload`, which is a smell in itself. A choice set that varies at
  runtime is computed inside `add_cli_args`, where it is evaluated once per
  parser and is trivially testable.
- **Non-list constants** (sentinels, scalar defaults, private tuples) live in
  their own group, never interleaved with the choice lists.
- **Deprecated aliases** (`NSA_CHOICES = DSA_CHOICES`) live in one clearly
  marked group at the bottom of the constants block, each carrying the date it
  was deprecated so the retirement clock in 1.6 can be applied.

## 1.3 What does not belong in this file

`server_args.py` owns: field declarations, CLI wiring, the resolution
dispatcher, generic handlers, and the extension-point lists. It does not own:

| Kind of code | Where it goes |
| --- | --- |
| Per-architecture defaults keyed on a `model_arch` string | the declarative registry in `arg_groups/overrides.py` |
| Imperative multi-step per-model or per-feature setup | `arg_groups/<name>_hook.py`, following the existing `mega_moe_hook` / `kimi_k3_hook` / `speculative_hook` shape |
| A pure utility with no argument-declaration role | the module that owns the concept |

The quick test: **if a module-level function's name or body contains a model,
vendor, or single-kernel name, it does not belong in `server_args.py`.**
`m3_fp8_attn_gemm_enabled` and `resolve_encoder_transfer_backend` both fail it.

The existing hook convention is: a module-level
`def handle_<feature>(server_args: ServerArgs) -> None` that reads fields off
the record and writes only through `declare_resolution(server_args, "<source>",
**fields)`, with `ServerArgs` imported under `TYPE_CHECKING` so the module stays
import-cycle-free.

## 1.4 `_run_resolution_pipeline`

The five dispatcher principles already in the docstring stay. Add three:

6. **Append, never prepend.** A new step goes at the *end* of the dispatcher —
   immediately before the `materialize_declarations(self)` call, which must
   remain last. The steps at the top of the pipeline are there because
   something below them depends on their output; a new step has no such claim,
   and inserting it at the top silently reorders every dependency beneath it.
   Moving a step *earlier* requires a written reason in the handler's docstring.
7. **Everything before the dummy-model boundary states why.** The
   `if self.model_path.lower() in ["none", "dummy"]: return` line is a
   contract, not an optimization. A handler placed above it carries a one-line
   docstring saying which dummy-path consumer needs it (a unit fixture, a
   direct handler call, an error that must fire for dummy models). Without that
   sentence, the handler belongs below the boundary.
8. **No inline imports and no inline calls in the dispatcher body.** A hook that
   needs a deferred import gets a thin `self._handle_*` wrapper that performs
   the import and the call. This is principle 1 ("keep this method as an ordered
   dispatcher") and principle 4 ("hide narrow integrations behind general
   handler names") applied together: the dispatcher should read as a list of
   phase names, with no vendor name and no `from ... import ...` in it.

## 1.5 `add_cli_args`

Subsection order, each appended to at its own end:

1. `add_cli_args_from_dataclass(parser, ServerArgs)` — the auto-derived bulk.
2. Fields whose choices are computed at parse time (plugin registries, env
   gates). This is the *only* place a dynamic choice set may be built.
3. The `--config` meta-argument.
4. Deprecated registrations, ordered by deprecation date, oldest first.

Everything else is an `A[T, Arg(...)]` field annotation. The class docstring
already says this; it is repeated here because the deprecated subsection is
where drive-by additions land.

## 1.6 Deprecation lifecycle

Every deprecated flag carries a `# Deprecated YYYY-MM-DD` comment on its
registration line. That date drives a three-stage clock:

| Age since deprecation | Behavior | Mechanism |
| --- | --- | --- |
| 0–2 months | Warn and forward to the replacement | `DeprecatedAliasStoreAction` / `DeprecatedStoreTrueAction` / `DeprecatedStoreConstAction` |
| 2–4 months | **Hard error** naming the replacement. No auto-redirection. | `DeprecatedAction(error_message=...)`, or an explicit `raise` in the handler for runtime-side deprecations |
| > 4 months | **Deleted** — registration, dataclass field, and handler branch | — |

The same clock governs runtime deprecations inside `_handle_deprecated_args`,
not just argparse registrations.

Retiring a flag is not done until its call sites are migrated: `test/`,
`docs/`, `scripts/ci/`, and `python/sglang/test/`. A flip that leaves call
sites behind converts a warning into a CI outage, so the migration commit lands
*before* the flip.

## 1.7 Append-here markers

Four places in the file attract drive-by additions and each gets a banner
comment telling the next author (human or agent) to append rather than prepend:

- the end of the extension-point constants block (stating that new choice
  lists are inlined into their field, not added here),
- `_run_resolution_pipeline`, immediately above `materialize_declarations`,
- the end of each `add_cli_args` subsection.

Rationale to state in the banner: later-added configuration is, on average,
less load-bearing than what is already there, so it belongs at the later stage
where it cannot perturb existing ordering.

## 1.8 Comments

`.claude/rules/comment-style.md` applies unchanged. In particular, the ordering
comments in the dispatcher ("must run before X") are exactly the cross-boundary
constraints that rule asks for — keep them, and add one whenever a handler's
position is load-bearing.

---

# Section 2 — Violations in the current file, and the fixes

Ordered by dependency: A and B are mechanical and land first, C changes
behavior in one narrow place, D is documentation, E is the largest and is
staged last.

## Group A — misplaced code (guidelines 1.1, 1.3)

### A1. `resolve_encoder_transfer_backend` sits inside the constants block

`server_args.py:352-359`, wedged between `ENCODER_TRANSFER_BACKEND_CHOICES`
(344) and `DSA_PREFILL_CP_SPLIT_CHOICES` (362). It also branches on
`"KimiK3ForConditionalGeneration"`, so it fails the model-name test in 1.3
twice over.

**Fix.** Move it to `arg_groups/encoder_transfer_hook.py` (new file, following
the hook convention). Callers: one in-tree caller,
`_handle_encoder_disaggregation` at `server_args.py:8340`, plus
`test/registered/unit/disaggregation/test_kimi_k3_encoder_mode.py:50`, whose
import line changes.
**Risk:** none — pure relocation of a pure function with two call sites.

### A2. `m3_fp8_attn_gemm_enabled` is MiniMax-M3-specific

`server_args.py:10462-10481`. A model-specific predicate with a 10-line
docstring about MSA fp8 kernels, living in the argument-declaration module.

**Fix.** Move to `arg_groups/minimax_m3_hook.py`. Importers to update:
`layers/attention/minimax_sparse_backend.py:23`,
`mem_cache/kv_cache_configurator.py:1522` (already a function-local import),
`test/registered/unit/test_model_overrides.py:2085`. The prose references in
`arg_groups/overrides.py:922,968` and `environ.py:1439` are comments only.
**Risk:** none — pure relocation; note `minimax_sparse_backend.py` imports it at
module scope, so check that the new module does not reintroduce a cycle (it
will not: the hook module imports `ServerArgs` only under `TYPE_CHECKING`).

### A3. Things that look misplaced but are correct — leave them

- `compute_world_size` (`10453-10459`) is a generic helper over a `ServerArgs`;
  slot 6 is its correct home.
- `ZMQ_TCP_PORT_DELTA` / `DP_ATTENTION_HANDSHAKE_PORT_DELTA` (`10568-10569`)
  belong to `PortArgs`; slot 8. Add the `# ---` banner so the section reads as
  deliberate. Neither is dead: `DP_ATTENTION_HANDSHAKE_PORT_DELTA` is imported
  by `managers/data_parallel_controller.py:63` and
  `python/sglang/test/cache_consistency_jitter.py:198`.

### A4. The rest of the model-specific logic — **not in this plan**

A1 and A2 are the two module-level offenders and are cheap (~30 lines moved
between them). Inside the class there is roughly 1,500 lines more, and moving
it is explicitly **out of scope**: it exceeds the size ceiling this plan works
under, and no single PR should carry it.

Recorded here only so the boundary is deliberate and guideline 1.3 has
something to point at. The largest concentrations are
`_handle_model_specific_adjustments` (`5595-6141`, 547 LOC, ~20 per-arch
branches), `_handle_linear_attn_backend` (`6637-6863`, 227 LOC, entirely
Mamba/GDN/KDA), the DSA/MLA arm at `5691-5862` (172 LOC), the Mamba cluster
(`6143-6212` and `6561-6635`, ~150 LOC), and `_get_default_attn_backend`
(`6224-6296`, 73 LOC, a hardware dispatch table with one Whisper special case).
The per-arch *field writes* have already largely migrated to the
`arg_groups/overrides.py` registry; what remains is asserts, env writes, and
hook dispatch.

Two items from that survey are small enough to be worth doing, and are folded
into A5 below rather than left here:

- four branches in `_handle_model_specific_adjustments` whose body is now a
  bare `pass` (StepFun `5991-5998`, MossVL `6043-6046`, Qwen3-MoE `6080-6092`,
  GLM4-MoE `6094-6098`) plus a comment-only block at `6108-6112` — ~30 lines,
  pure deletion;
- `_apply_inkling_prefill_cuda_graph_default` (`4728-4748`) and
  `_apply_muse_glimmer_prefill_cuda_graph_max_bs_default` (`4750-4761`), two
  single-model method names sitting in the dispatcher, which is what principle
  4 forbids — collapse behind one
  `self._apply_model_arch_cuda_graph_defaults()`, ~35 lines.

### A5. Small hygiene items found along the way

- **Four dead `pass` branches and one comment-only block** in
  `_handle_model_specific_adjustments`: `5991-5998` (StepFun), `6043-6046`
  (MossVL), `6080-6092` (Qwen3-MoE), `6094-6098` (GLM4-MoE), `6108-6112`
  (MiniMax-M2 / Qwen3-VL leftover comments). Their logic moved to the override
  registry; only the `elif` skeletons remain. ~30 lines, pure deletion.
- **Two single-model dispatcher entries.**
  `_apply_inkling_prefill_cuda_graph_default` (`4728-4748`) and
  `_apply_muse_glimmer_prefill_cuda_graph_max_bs_default` (`4750-4761`) name a
  model in the dispatcher body (`3851-3852`), which principle 4 forbids.
  Collapse both behind one `self._apply_model_arch_cuda_graph_defaults()`;
  the bodies stay as they are.
- **`DEFAULT_LORA_EVICTION_POLICY` (`367`) is dead.** No reference anywhere in
  the repo; the `lora_eviction_policy` field (`3006-3013`) hard-codes `"lru"`.
  Delete it rather than sorting it into slot 3b.
- **`LANGUAGE_MODEL_ONLY_ARCHITECTURES` (`8271`)** is a bare class attribute —
  not an `A[...]` dataclass field — wedged between two methods, holding
  `("MuseGlimmerForConditionalGeneration",)`. Read only by
  `_handle_language_model_only` (`8296`, `8299`). Move it with that handler
  when A4 is taken; at minimum it should not sit between two `def`s.
- **Two stale cross-references in `arg_groups/overrides.py`.**
  `overrides.py:834` says "Keep in sync with `MIMO_V2_MODEL_ARCHS`
  (server_args.py / configs/hf_config.py)" — the constant lives in
  `configs/model_config.py:46`, and there is no `configs/hf_config.py`.
  `overrides.py:1143` says "Keep in sync with `LLAMA4_MODEL_ARCHS`
  (server_args.py)" — that name does not exist anywhere in the repo. Fix both
  to point at what they actually mean.

## Group B — constants and adders (guideline 1.2)

### B1. Import-time env branch on `SAMPLING_BACKEND_CHOICES`

```python
SAMPLING_BACKEND_CHOICES = {"flashinfer", "pytorch", "ascend"}   # :115
if envs.SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE.get():              # :116-117
    SAMPLING_BACKEND_CHOICES.add("token_oracle")
```

**Fix.** Keep the base set static at module scope; move the `token_oracle` gate
into `add_cli_args`, in the dynamic-choices subsection, registering
`--sampling-backend` explicitly and marking the field `no_cli=True`.

**Call sites that must change:**
- `test/registered/unit/server_args/test_server_args.py:1935-1985`
  (`TestSamplingBackendTokenOracleEnvGate`) exists solely because the gate is
  evaluated at import time — it calls `importlib.reload(server_args_module)`
  around each subtest. Rewrite it to build a parser with the env var set,
  which is both simpler and no longer order-dependent.
- `test/registered/mock_model/test_self_unit_sampler_hookpoint.py:38` asserts
  membership in the module-level set; it exercises `register_sampler_backend`,
  so it keeps working once B2 lands.

### B2. Collapse the adder block into the lists (guideline 1.2)

The 16 adders at `414-475` occupy about 80 lines (a 3-line `def` plus blank
lines each) to express 16 one-line bodies, and they form a second block that
has to be kept in the same order as the first.

Worth knowing before touching them: **none of the 16 has an in-tree caller.**
The only in-tree reference is
`model_executor/model_runner.py:190-195`, which re-exports
`add_chunked_prefix_cache_attention_backend` under
`# noqa: F401  (re-export)` — an import, not a call. The whole block is an
out-of-tree plugin API. That is a reason to stop growing it, not a reason to
delete it: out-of-tree callers are exactly the ones we cannot grep for.

**Fix.** Add `Choices` to `arg_groups/arg_utils.py` next to `A` / `Arg` / `NS`:

```python
class Choices(list):
    """A CLI choice list out-of-tree code may extend before ServerArgs is built.

    `Arg(choices=...)` and argparse both hold this object, not a copy, so an
    append made at import time is visible to the parser.
    """

    def add(self, choices: Iterable[str]) -> None:
        if isinstance(choices, str):
            raise TypeError("add() takes an iterable of choices, not a string")
        self.extend(choices)
```

Then, for each of the 16 surviving extension points, wrap the literal in
`Choices([...])` and bind the legacy name on the following line. Net effect:
~80 lines to ~16, the second block disappears, and the ordering constraint
between the two blocks disappears with it — which also removes the need for the
alignment lint I proposed in the first draft of this plan.

The `isinstance(choices, str)` guard is not decoration: `list.extend("abc")`
appends three characters silently, and B1 converts `SAMPLING_BACKEND_CHOICES`
from a `set` (whose `.add` takes a single element) to a `Choices`, so exactly
that mistake is now reachable from `layers/sampler.py:537-541`. That call site
changes to `add_sampling_backend_choices([backend])`, matching every other
adder.

### B2b. Inline the fourteen lists that are not extension points

None of these has an adder, and per guideline 1.2 that means none of them needs
one. Nine are referenced by exactly one field and can be inlined outright:

| List | Line | Sole use |
| --- | --- | --- |
| `RETRACTION_POLICY_CHOICES` | 334 | field at 918 |
| `BF16_GEMM_BACKEND_CHOICES` | 331 | field at 1846 |
| `DSV4_PREFILL_BACKEND_CHOICES` | 382 | field at 1867 |
| `DSA_PAGED_MQA_LOGITS_BACKEND_CHOICES` | 390 | field at 1884 |
| `DSA_TOPK_BACKEND_CHOICES` | 388 | field at 1892 |
| `MAMBA_BACKEND_CHOICES` | 399 | field at 1915 |
| `MAMBA_RADIX_CACHE_STRATEGY_CHOICES` | 392 | field at 2649 |
| `LORA_BACKEND_CHOICES` | 342 | field at 3018 |
| `ENCODER_TRANSFER_BACKEND_CHOICES` | 344 | field at 3273, plus `= ENCODER_TRANSFER_BACKEND_CHOICES[0]` as the default at 3276 — write `"auto"` there instead |

Three are shared by two or more fields, so they keep a module-level name but
still get no adder. They move into a `# --- Shared choice lists (not extension
points) ---` group, kept separate from 3a so the "list ⟺ extension point"
invariant holds for the extension-point group:

- `MOE_A2A_BACKEND_CHOICES` (290) — fields at 2287 and 2429
- `DSA_CHOICES` (369) — fields at 1854 and 1875, two deprecated registrations
  at 9393/9403, and the `NSA_CHOICES` alias
- `DSA_PREFILL_CP_SPLIT_CHOICES` (362) — used only by two E2-tier deprecated
  registrations (9533, 9547) and the `NSA_PREFILL_CP_SPLIT_CHOICES` alias, so
  it disappears entirely once E2 lands. Leave it alone until then.

`PREFILL_CP_SPLIT_CHOICES` (365) is used only by the deprecated
`--prefill-cp-mode` registration at 9557; E2 deletes both. No action here.

### B2c. `linear_attn_verify_backend` silently opted out of its extension point

`server_args.py:2706` reads
`choices=LINEAR_ATTN_KERNEL_BACKEND_CHOICES + ["nv_cutedsl"]`. The `+` is
evaluated at class-definition time and yields a new plain `list`, so this field
is a frozen snapshot while its three siblings (`linear_attn_backend` 2682,
`linear_attn_decode_backend` 2690, `linear_attn_prefill_backend` 2698) hold the
live object. An out-of-tree platform calling
`add_linear_attn_kernel_backend_choices(["mine"])` gets `--linear-attn-backend
mine` accepted and `--linear-attn-verify-backend mine` rejected.

**Fix.** Add `"nv_cutedsl"` to `LINEAR_ATTN_KERNEL_BACKEND_CHOICES` itself
(the list already contains eight backends including `cutedsl`; verify with the
linear-attention owners whether `nv_cutedsl` is genuinely verify-only before
widening the other three fields), or give the field its own inline list and
accept that it is not an extension point. Either way the current form should
not survive. `SUPPORTED_LORA_TARGET_MODULES + [LORA_TARGET_ALL_MODULES]` at
2983 has the same shape but is harmless — `SUPPORTED_LORA_TARGET_MODULES` is
not an extension point and has no adder.

### B3. Non-list constants interleaved with the choice lists

`DEFAULT_UVICORN_ACCESS_LOG_EXCLUDE_PREFIXES` (113), `MIS_DELIMITER_TOKEN_ID`
(268, sitting between the grammar and MoE lists), `_LORA_SPEC_ALGORITHMS`
(340), `DEFAULT_LORA_EVICTION_POLICY` (367).

**Fix.** Collect into a "scalar defaults and sentinels" group (slot 3b) below
the choice lists. `MIS_DELIMITER_TOKEN_ID` is imported by
`managers/tokenizer_manager_score_mixin.py`,
`managers/scheduler_components/logprob_result_processor.py`, and
`test/registered/unit/managers/test_embed_overrides.py`; moving it within the
same module does not touch them.

### B4. Deprecated aliases inline

`NSA_PREFILL_CP_SPLIT_CHOICES = DSA_PREFILL_CP_SPLIT_CHOICES` (363) and
`NSA_CHOICES = DSA_CHOICES` (380).

**Fix.** Move both to a `# --- Deprecated aliases ---` group at the bottom of
the constants block, each with `# Deprecated 2026-05-20`.
`test/manual/test_dsa_alias_cli_registry_env.py` asserts the two names are the
*same object*; keep the assignment form, only the position changes.

### B5. Sort, then lock it in

Apply the 1.2 grouping and alphabetical-within-group ordering. The
"adders mirror the constants" lint from the first draft of this plan is no
longer needed — B2 makes the adder adjacent to its list, so they cannot drift.

What is still worth a test is the invariant B2/B2b establish, as a
decrease-only ratchet in the style of the existing
`test/registered/unit/test_*_ratchet.py`: **the number of module-level
`*_CHOICES` names in `server_args.py` may not increase.** That is a single
`dir(server_args_module)` count, it enforces "no new extension points, inline
instead" without needing to understand anything about the lists, and it fails
loudly on the exact drive-by addition guideline 1.2 is written to prevent.

## Group C — `_run_resolution_pipeline` audit (guideline 1.4)

The five calls before the dummy-model boundary (`3807-3813`, boundary at
`3818-3819`) were the reported violation. I checked each; **three of the five
must not move, and the two that can are less than half the story.** Verdicts,
with the evidence:

| Step | Verdict | Evidence |
| --- | --- | --- |
| `handle_mega_moe` (3807-3809) | **Do not move** | `test/registered/unit/server_args/test_server_args.py:68-81` launches `prepare_server_args(["--model-path", "dummy"])`, calls `resolve_once()`, and asserts `DG_USE_FP4_ACTS == "1"`. Moving the call below the boundary makes that assertion fail. It is also legitimately model-independent CLI-alias normalization, which principle 2 permits above the boundary. |
| `_handle_return_hidden_states_mode` (3810) | **Do not move** (corrected during execution) | The pre-execution analysis checked only for tests calling the handler by name. `test_server_args.py::test_return_hidden_states_mode_configuration` resolves `ServerArgs(model_path="dummy", ...)` and asserts the reconciled `enable_return_hidden_states` / `return_hidden_states_mode` pair, so the dummy path is a real consumer. |
| `_handle_media_url_security` (3811) | **Do not move** (corrected during execution) | The process-global half is indeed re-applied by workers, but the handler also writes the *normalized* `allowed_media_domains` back onto the record, and `test_server_args_migration.py::test_media_url_security_args` reads it after parsing `--model dummy`. |
| `_handle_hicache_ratio_default` (3812) | **Do not move** | Its own docstring (`7871-7880`) states the requirement: "Runs before the dummy-model boundary: direct HostKVCache consumers (unit fixtures, dummy-model launches) must never see a None ratio." `TestHiCacheArgs._make_args` (`test_server_args.py:1313-1320`) works around the boundary by calling the handler by hand, which is the same dependency observed from the other side. |
| `_validate_prefill_decode_interval` (3813) | **Do not move** | `test_server_args.py:83-91` asserts `ServerArgs(model_path="dummy", prefill_decode_interval=-1).resolve_once()` raises. It is an argument-validation error that should fire for dummy models — exactly the third category principle 2 admits above the boundary. |

**Fixes for Group C:**

- **C1.** ~~Move `_handle_return_hidden_states_mode` and
  `_handle_media_url_security` below the boundary.~~ **Dropped during
  execution**: both were tried, both broke dummy-path tests, both reverted.
  All five pre-boundary steps stay where they are; what the audit actually
  yields is C2-C4 below, which is the more durable half — every one of the
  five now documents why it is early, so the next reader does not have to
  re-derive it the hard way.
- **C2.** Wrap the mega-MoE hook: replace the inline
  `from sglang.srt.arg_groups.mega_moe_hook import handle_mega_moe` +
  `handle_mega_moe(self)` with `self._handle_moe_backend_aliases()`, a
  three-line method that performs the import and the call. This satisfies
  principle 1 (no inline imports in the dispatcher) and principle 4 (no vendor
  name in the dispatcher) without moving the step. **Do not** relocate it into
  the MoE section further down: it rewrites `moe_a2a_backend` from `"none"` to
  `"megamoe"`, and `_apply_deepep_adjustments` (4785),
  `_disable_tc_piecewise_cudagraph_if_incompatible` (4940),
  `_disable_breakable_cudagraph_if_incompatible` (5024) and
  `reserve_for_deepep_a2a_mb` (5472, 5488) all read that field earlier, via
  `_handle_cuda_graph_config` (3857) and `_handle_gpu_memory_settings` (3878).
- **C3.** Same wrapper treatment for the two remaining inline imports and the
  inline call in the dispatcher body: `declare_direct_writes(...)` at
  `3868-3872` becomes `self._handle_platform_defaults()`;
  `from ...speculative_hook import handle_speculative_decoding` at `3941-3943`
  becomes `self._handle_speculative_decoding()`. The
  `materialize_declarations` import at `3987` stays — it is the terminal step,
  not a phase.
- **C4.** Add the docstring sentence required by principle 7 to the three
  handlers that stay above the boundary. `_handle_hicache_ratio_default`
  already has it; `_handle_moe_backend_aliases`,
  `_validate_prefill_decode_interval`, and
  `_handle_hardware_runtime_validation` (3817, which has an inline comment —
  promote it to the handler) do not.

Verification for C1/C2: `test/registered/unit/server_args/` contains
`test_resolution_is_reproducible.py` and `test_resolution_declarations.py`,
which compare the full declaration stash across a resolution. A reorder that
changes any resolved field will show up there as a diff, which is the check
that makes the two moves safe to assert rather than hope.

## Group D — append-here banners (guideline 1.7)

Purely additive; no behavior change.

- **D1.** Banner at the end of the constants block and at the end of the adder
  block: new choice lists and their adders are appended here, in the matching
  domain group.
- **D2.** Banner in `_run_resolution_pipeline` immediately above
  `materialize_declarations(self)` (~3983-3989). The wording must say
  *"append immediately above this call"*, not "at the end of the function" —
  `materialize_declarations` applies the accumulated declarations onto the
  fields and must remain the last statement.
- **D3.** Banner at the end of each `add_cli_args` subsection.
- **D4.** Extend the class docstring's "Adding new arguments" section
  (`480-518`) with the append rule and a pointer to the deprecation clock in
  1.6.

## Group E — deprecated argument sweep (guideline 1.6)

Deprecation dates below are the commit date of the change that first attached a
deprecation marker to the flag, not the date the flag was introduced. Two
confounders were excluded by hand: `b28e990161a` (2026-06-22) relocated the
whole deprecated block, and `c64274c746f` (2026-03-02) introduced
`--disable-piecewise-cuda-graph` / `--enforce-piecewise-cuda-graph` as *live*
flags next to a different deprecated one.

Cutoffs: **remove** if deprecated on or before 2026-04-25; **hard error** if
between 2026-04-26 and 2026-06-25; **leave the redirect** if after 2026-06-25.

### E1 — Remove (> 4 months)

| Flag / behavior | Deprecated | Age | What removal touches |
| --- | --- | --- | --- |
| `--prefill-round-robin-balance` | 2025-12-31 | 7.9 mo | No dataclass field; registration only. Call sites: 10 `scripts/ci/slurm/recipes/**/*.yaml`, 2 NPU tests, `test_server_args_cli_metadata.py`, 6 docs pages |
| `tool_call_parser` `qwen25`→`qwen`, `glm45`→`glm` (`_handle_deprecated_args:4367-4376`) | 2025-10-07 | 10.6 mo | Also drop the `"qwen25"` / `"glm45"` keys from `FunctionCallParser.ToolCallParserEnum` (`function_call_parser.py:74,88`) — otherwise removing the rename silently routes them to their own detectors instead of erroring |
| `--stream-output` | 2026-03-14 | 5.4 mo | No field. Call sites: `test_server_args_migration.py`, 1 docs page |
| `--enable-flashinfer-allreduce-fusion` | 2026-03-17 | 5.3 mo | Is a real dataclass field. Remove the field, the manual `store_true` registration (`9563-9568`), and the `_handle_deprecated_args` branch (`4378-4394`, including the unconditional `enable_flashinfer_allreduce_fusion=False` declaration). Call sites: 4 tests, 8 docs/cookbook pages. `arg_groups/overrides.py:2148+` uses `flashinfer_allreduce_fusion_backend`, the replacement, and is unaffected |
| `--collect-tokens-histogram` | 2026-04-24 | 4.0 mo | No field. Call sites: 1 ascend test, 2 docs pages |

### E2 — Convert to a hard error, no redirect (2–4 months)

Nineteen flags. Ordered by migration cost, smallest first — this is the
recommended landing order, one commit per row or per tight group, each
preceded by its call-site migration.

| Flag | Deprecated | Age | Non-`server_args` call sites |
| --- | --- | --- | --- |
| `--prefill-cp-mode` | 2026-06-10 | 2.5 mo | 2 |
| `--nsa-decode-backend` | 2026-05-20 | 3.2 mo | 2 |
| `--piecewise-cuda-graph-tokens` | 2026-06-09 | 2.5 mo | 3 |
| `--speculative-dflash-draft-window-size` | 2026-05-16 | 3.3 mo | 3 |
| `--nsa-prefill-backend` | 2026-05-20 | 3.2 mo | 4 |
| `--piecewise-cuda-graph-compiler` | 2026-06-09 | 2.5 mo | 7 |
| `--piecewise-cuda-graph-max-tokens` | 2026-06-09 | 2.5 mo | 8 |
| `--nsa-prefill-cp-mode` | 2026-05-20 | 3.2 mo | 8 |
| `--enable-nsa-prefill-context-parallel` | 2026-05-20 | 3.2 mo | 9 |
| `--enable-prefill-context-parallel` | 2026-06-10 | 2.5 mo | 9 |
| `--enable-breakable-cuda-graph` | 2026-06-09 | 2.5 mo | 10 |
| `--enable-dsa-prefill-context-parallel` | 2026-05-20 | 3.2 mo | 11 |
| `--dsa-prefill-cp-mode` | 2026-05-20 | 3.2 mo | 11 |
| `--cuda-graph-max-bs` | 2026-06-09 | 2.5 mo | 16 |
| `--mamba-scheduler-strategy` | 2026-06-19 | 2.2 mo | 31 |
| `attention_backend="compressed"` → `dsv4` (`_handle_deprecated_args:4395-4410`) | 2026-05-07 | 3.6 mo | runtime-side; convert the rename loop into a `raise ValueError` naming `dsv4`, and keep `"compressed"` in `ATTENTION_BACKEND_CHOICES` so the error message is ours rather than argparse's generic one |
| `--enforce-piecewise-cuda-graph` | 2026-06-09 | 2.5 mo | ~10, incl. docs and AMD tests |
| `--disable-piecewise-cuda-graph` | 2026-06-09 | 2.5 mo | ~26 files across `test/registered/`, `python/sglang/test/`, 3 docs pages |
| `--cuda-graph-bs` | 2026-06-09 | 2.5 mo | ~110 |
| `--disable-cuda-graph` | 2026-06-09 | 2.5 mo | ~139 |

**This is the part of the plan that must not be done mechanically.** The last
four rows alone account for roughly 280 call sites in tests, docs, and CI
recipes. Flipping `--disable-cuda-graph` and `--cuda-graph-bs` to a hard error
in the same commit as the rest would take CI down. Sequence them as their own
PRs, at the end, after the cheap rows have validated the pattern.

Mechanism for every row: replace the redirecting action with
`action=DeprecatedAction, error_message="<flag> was removed. Use <replacement>."`.
`DeprecatedAction` already calls `parser.error(...)` when `error_message` is
set (`arg_groups/argparse_actions.py:32-43`), which is the required hard crash
with a message; no new action class is needed.

### E3 — Leave the redirect in place (< 2 months)

| Flag | Deprecated | Age | Note |
| --- | --- | --- | --- |
| `--grpc-mode` | 2026-07-08 | 1.6 mo | runtime warning in `_handle_deprecated_args:4412-4422` |
| `--enable-gdn-replayssm-spec` | 2026-08-04 | 0.7 mo | |
| `--disable-fast-image-processor` | 2026-08-12 | 0.4 mo | runtime warning in `_handle_deprecated_args:4355-4365` |
| `--enable-expert-distribution-metrics` | 2026-08-16 | 0.3 mo | already a hard error via `error_message` |

Add the `# Deprecated YYYY-MM-DD` comment to all four now, so the next sweep
does not have to reconstruct the dates from `git log -G` the way this one did.

### E4 — Out of scope, noted here so it is not lost

The `attention_backend="nsa"` alias (deprecated 2026-05-20) is not a
`server_args.py`-local concern: it is threaded through
`layers/attention/attention_registry.py:140`,
`models/deepseek_common/utils.py:64`,
`models/deepseek_common/attention_backend_handler.py:244`,
`models/dots3_common/modeling.py`, and `models/sarvam_moe.py`. Retiring it is
its own change across the model layer. This cleanup only dates the alias in the
constants block (B4).

## Group F — verification

Per-group, in the order the groups land:

- **A, B3, B4** are relocations. Reproduce them with the
  `mechanical-refactor-verify` skill so the move is machine-checked rather than
  eyeballed; a relocation that changes a byte other than the import lines is a
  bug.
- **B1, B2** — run `test/registered/unit/server_args/test_server_args.py`,
  `test/registered/mock_model/test_self_unit_sampler_hookpoint.py`,
  `test/manual/test_dsa_alias_cli_registry_env.py`.
- **C** — `test/registered/unit/server_args/test_resolution_is_reproducible.py`
  and `test_resolution_declarations.py` are the real gate: they compare the
  whole declaration stash, so any unintended reorder surfaces as a field diff
  rather than as a silent behavior change. Also run
  `test_server_args_migration.py` and `test_server_args_cli_metadata.py`.
- **E** — for each flag, before flipping: `grep -rE -- "--<flag>([\"' =,)]|$)"`
  across `test/`, `docs/`, `scripts/`, `python/sglang/test/`, migrate every hit
  to the replacement, land that, then flip.
- **B2/B2b** — beyond the tests above, confirm the `Choices` swap is
  behaviour-neutral: `isinstance(LOAD_FORMAT_CHOICES, list)`,
  `NSA_CHOICES is DSA_CHOICES`, and one round-trip through
  `prepare_server_args` per inlined field to confirm argparse still accepts and
  rejects the same values.
- **B5** adds the `*_CHOICES` count ratchet, which is what keeps Section 1 true
  after this cleanup rather than only at the moment it lands.

## Suggested commit sequence

Every step below is under ~400 lines touched; none is a large code movement.

| # | Step | Rough size |
| --- | --- | --- |
| 1 | A1, A2 — move the two model-specific utilities out (2 commits) | ~30 lines moved, 4 import sites |
| 2 | A5 — dead branches, the two single-model dispatcher entries, dead constant, stale comments | ~70 lines, mostly deletion |
| 3 | B2 — add `Choices`; collapse the 16 adders onto their lists | ~80 lines removed |
| 4 | B2b, B2c — inline the 9 single-use lists; fix `linear_attn_verify_backend` | ~90 lines |
| 5 | B3, B4, B5-sort — regroup and sort what remains of the constants block | ~250 lines reordered, no logic |
| 6 | B1 — de-dynamize `SAMPLING_BACKEND_CHOICES`; rewrite the reload-based test | ~40 lines |
| 7 | C2, C3, C4 — dispatcher wrappers and docstrings, no reordering | ~40 lines |
| 8 | C1 — the two safe moves below the boundary, gated on the reproducibility tests | ~10 lines |
| 9 | D1–D4 — banners and class docstring | ~40 lines added |
| 10 | B5-ratchet — the "no new `*_CHOICES`" count test | new test file |
| 11 | E1 — the five removals | ~60 lines plus call-site migration |
| 12 | E2 — hard-error flips, one PR per row from the top of the table down, `--disable-cuda-graph` and `--cuda-graph-bs` last | 19 PRs; the flip is small, the call-site migration is not |
| 13 | E3 — date comments on the four young deprecations | ~4 lines |

Steps 1–10 are self-contained and land quickly. Step 12 is the long tail and
should be tracked separately.

## What this plan deliberately does not do

The governing constraint: **no step in this plan is a code movement over ~1k
lines.** Three things fall outside it.

- **A4** — relocating the ~1,500 lines of in-class model-specific logic. Its
  largest single piece (`_handle_model_specific_adjustments`, 547 LOC) would
  already dominate any commit here, and each destination hook needs its own
  test surface. Separate effort, separate tracking.
- **E4** — retiring the `attention_backend="nsa"` alias, which lives mostly in
  the model layer.
- **Splitting the file.** At 10,745 lines it is five times over the ~2k
  guidance in `.claude/rules/general-code-style.md`, but the natural seam is
  the one A4 describes. Splitting before the model-specific logic is drained
  into `arg_groups/` just moves the problem into a second oversized file.
