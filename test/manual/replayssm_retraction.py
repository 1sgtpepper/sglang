"""Focused GPU recurrence/backup diagnostic against unchanged production code.

Run from the repository root with PYTHONPATH=python:
  python test/manual/replayssm_retraction.py

No model weights or transport service are required. Each case uses the actual
GDN kernels, request/pool allocation, and CPU-tensor backup/restore. The optional
pre-backup materialization is an experimental causal control, not a proposed fix.
"""

import json
from array import array

import torch

from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode import (
    commit_gdn_replayssm_circular,
    commit_gdn_replayssm_spec,
    gdn_replayssm_spec_decode,
)
from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.common import retraction_backup, retraction_restore
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.sampling.sampling_params import SamplingParams


def reference(state, values):
    """Sequential delta-rule in float64, independent of the chunk/ring code."""
    q, k, v, a, b, a_log, bias = [x.cpu().double() for x in values]
    h = state.cpu().double().clone()
    outputs, states = [], []
    for t in range(q.shape[0]):
        qt = q[t] / torch.sqrt(q[t].square().sum(-1, keepdim=True) + 1e-6)
        kt = k[t] / torch.sqrt(k[t].square().sum(-1, keepdim=True) + 1e-6)
        qt = qt.repeat_interleave(h.shape[0] // qt.shape[0], 0)
        kt = kt.repeat_interleave(h.shape[0] // kt.shape[0], 0)
        alpha = torch.exp(-a_log.exp() * torch.nn.functional.softplus(a[t] + bias))
        beta = b[t].sigmoid()
        h = h * alpha[:, None, None]
        prediction = torch.einsum("hvk,hk->hv", h, kt)
        delta = beta[:, None] * (v[t] - prediction)
        h = h + delta[:, :, None] * kt[:, None, :]
        outputs.append(torch.einsum("hvk,hk->hv", h, qt) * h.shape[-1] ** -0.5)
        states.append(h.clone())
    return torch.stack(outputs), states


def inputs(width, seed):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(width, 2, 128, generator=gen)
    k = torch.randn(width, 2, 128, generator=gen)
    v = torch.randn(width, 4, 128, generator=gen)
    # Modest decay retains history while avoiding an ill-conditioned oracle.
    a = torch.full((width, 4), -3.0)
    b = torch.full((width, 4), 0.5)
    return tuple(x.to(device="cuda", dtype=torch.bfloat16) for x in (q, k, v, a, b)) + (
        torch.zeros(4, device="cuda"),
        torch.zeros(4, device="cuda"),
    )


def make_case(dtype, width, ring_len, carries_mamba):
    shape = Mamba2StateShape.create(
        tp_world_size=1, intermediate_size=512, n_groups=2, num_heads=4,
        head_dim=128, state_size=128, conv_kernel=4,
    )
    pool = HybridReqToTokenPool(
        size=4, mamba_size=8, mamba_spec_state_size=4, max_context_len=128,
        device="cuda", enable_memory_saver=False,
        cache_params=Mamba2CacheParams(
            shape=shape, dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=dtype),
            layers=[0, 1],
        ),
        mamba_layer_ids=[0, 1], enable_mamba_extra_buffer=False,
        speculative_num_draft_tokens=width, speculative_eagle_topk=1,
        enable_linear_replayssm_spec=True, linear_replayssm_cache_len=ring_len,
    )
    kv = HybridLinearKVPool(
        size=128, dtype=torch.bfloat16, page_size=1, head_num=1, head_dim=64,
        full_attention_layer_ids=[2], device="cuda", mamba_pool=pool.mamba_pool,
    )
    allocator = TokenToKVPoolAllocator(
        size=128, dtype=torch.bfloat16, device="cuda",
        kvcache=kv if carries_mamba else kv.full_kv_pool, need_sort=False,
    )
    req = Req("retraction", "", array("q", [1, 2, 3]), SamplingParams(max_new_tokens=100))
    req.output_ids.append(4)
    pool.alloc([req])
    assert req.kv.req_pool_idx != int(req.kv.mamba_pool_idx)
    return req, pool, allocator


def verify(req, pool, values, width):
    mp = pool.mamba_pool
    state = mp.mamba_cache
    output = []
    for layer in range(2):
        out = torch.empty(width, 4, 128, device="cuda", dtype=torch.bfloat16)
        gdn_replayssm_spec_decode(
            q=values[0], k=values[1], v=values[2], a=values[3], b=values[4],
            A_log=values[5], dt_bias=values[6],
            checkpoint_state=state.temporal[layer],
            d_cache=state.replayssm_d[layer], k_cache=state.replayssm_k[layer],
            g_cache=state.replayssm_g[layer], rawv_cache=state.replayssm_rawv[layer],
            rawk_cache=state.replayssm_rawk[layer], beta_cache=None,
            out=out, query_start_loc=torch.tensor([0, width], device="cuda", dtype=torch.int32),
            ssm_state_indices=req.kv.mamba_pool_idx.reshape(1),
            replay_indices=torch.tensor([req.kv.req_pool_idx], device="cuda", dtype=torch.int64),
            write_pos=mp.replayssm_spec_write_pos, cache_base=mp.replayssm_cache_base,
            is_flush=mp.replayssm_is_flush,
            max_cache_len=mp.linear_replayssm_cache_len, max_spec_len=width,
            null_block_id=-1, launch_mode="verify",
        )
        output.append(out)
    return torch.stack(output)


def fold(req, pool, accepted):
    mp = pool.mamba_pool
    state = mp.mamba_cache
    commit_gdn_replayssm_circular(
        checkpoint_state=state.temporal,
        d_cache=state.replayssm_d, k_cache=state.replayssm_k, g_cache=state.replayssm_g,
        d_residual_cache=state.replayssm_rawv, k_residual_cache=state.replayssm_rawk,
        state_batch_indices=req.kv.mamba_pool_idx.reshape(1),
        replay_indices=torch.tensor([req.kv.req_pool_idx], device="cuda", dtype=torch.int64),
        write_pos=mp.replayssm_spec_write_pos, cache_base=mp.replayssm_cache_base,
        is_flush=mp.replayssm_is_flush,
        accept_lens=torch.tensor([accepted], device="cuda", dtype=torch.int32),
        null_block_id=-1,
    )


def commit(req, pool, width, accepted):
    mp = pool.mamba_pool
    commit_gdn_replayssm_spec(
        write_pos=mp.replayssm_spec_write_pos, cache_base=mp.replayssm_cache_base,
        is_flush=mp.replayssm_is_flush,
        num_accepted=torch.tensor([accepted], device="cuda", dtype=torch.int32),
        replay_indices=torch.tensor([req.kv.req_pool_idx], device="cuda", dtype=torch.int64),
        max_cache_len=mp.linear_replayssm_cache_len, max_spec_len=width,
        fold_every_commit=mp.mamba_cache.temporal.dtype != torch.float32, null_block_id=-1,
    )
    fold(req, pool, accepted)
    req.output_ids.extend(range(accepted))


def roundtrip(req, pool, allocator, materialize, move_row, reuse_state):
    mp = pool.mamba_pool
    old_req, old_state = req.kv.req_pool_idx, int(req.kv.mamba_pool_idx)
    if materialize:
        # Sole intervention: fold the accepted pending history before exporting.
        mp.replayssm_is_flush[old_req] = 1
        fold(req, pool, 0)
    n = req.seqlen - 1
    previous = req.kv.kv_allocated_len
    slots = torch.cat((
        pool.req_to_token[old_req, :previous].to(torch.int64),
        allocator.alloc(n - previous),
    ))
    pool.write((old_req, slice(0, n)), slots.to(torch.int32))
    req.kv.kv_allocated_len = req.kv.kv_committed_len = n
    kv = allocator.get_kvcache()
    kv = getattr(kv, "full_kv_pool", kv)
    kv.k_buffer[0][slots] = 0.25
    kv.v_buffer[0][slots] = 0.75
    mp.mamba_cache.conv[0][:, old_state] = 0.5
    assert retraction_backup(req, None, pool, allocator, "cpu_tensor")
    held_states = None
    if reuse_state:
        held_states = pool.mamba_allocator.alloc(pool.mamba_allocator.available_size())
    allocator.free(slots)
    pool.free_mamba_cache(req)
    pool.free(req)
    req.reset_for_retract()
    held_row = pool.alloc_rows(1) if move_row else []
    pool.alloc([req])
    if reuse_state:
        assert int(req.kv.mamba_pool_idx) == old_state
        pool.mamba_allocator.free(held_states)
    if move_row:
        assert req.kv.req_pool_idx != old_req
        pool.free_rows(held_row)
    slots = allocator.alloc(n)
    pool.write((req.kv.req_pool_idx, slice(0, n)), slots.to(torch.int32))
    req.kv.kv_allocated_len = req.kv.kv_committed_len = n
    # Ensure restore cannot accidentally benefit from the destination's old bytes.
    mp.mamba_cache.temporal[:, req.kv.mamba_pool_idx] = -0.125
    retraction_restore(req, None, pool, allocator, "cpu_tensor")
    assert req.kv.retraction_backup is None
    assert int(mp.replayssm_spec_write_pos[req.kv.req_pool_idx]) == 0
    torch.testing.assert_close(kv.k_buffer[0][slots], torch.full_like(kv.k_buffer[0][slots], 0.25))
    torch.testing.assert_close(kv.v_buffer[0][slots], torch.full_like(kv.v_buffer[0][slots], 0.75))
    torch.testing.assert_close(
        mp.mamba_cache.conv[0][:, req.kv.mamba_pool_idx],
        torch.full_like(mp.mamba_cache.conv[0][:, req.kv.mamba_pool_idx], 0.5),
    )
    return [old_req, old_state], [req.kv.req_pool_idx, int(req.kv.mamba_pool_idx)]


def run(dtype, width, accepted, materialize, rounds, ring_len=16,
        carries_mamba=True, move_row=False, reuse_state=False):
    req, pool, allocator = make_case(dtype, width, ring_len, carries_mamba)
    expected_state = torch.zeros(4, 128, 128, dtype=torch.float64)
    errors, identities, pending = [], [], []
    for iteration in range(rounds):
        data = inputs(width, 100 + iteration)
        verify(req, pool, data, width)
        _, states = reference(expected_state, data)
        if accepted:
            expected_state = states[accepted - 1]
        commit(req, pool, width, accepted)
        pending.append(int(pool.mamba_pool.replayssm_spec_write_pos[req.kv.req_pool_idx]))
        next_data = inputs(width, 200 + iteration)
        uninterrupted = verify(req, pool, next_data, width).cpu().double()
        expected, _ = reference(expected_state, next_data)
        oracle_error = float((uninterrupted - expected.unsqueeze(0)).abs().max())
        # This tolerance is checked against the independent recurrence before
        # the lifecycle comparison, so rounding cannot explain a large loss.
        assert oracle_error < 0.001, oracle_error
        identities.append(roundtrip(req, pool, allocator, materialize, move_row, reuse_state))
        restored = verify(req, pool, next_data, width).cpu().double()
        delta = float((restored - uninterrupted).abs().max())
        errors.append({"oracle_error": oracle_error, "restore_delta": delta})
        if not materialize and dtype == torch.float32 and pending[-1] > 0:
            # A lossy baseline cannot supply the intended recurrence to another
            # round. Repeated transitions are measured by the preserving cases.
            break
    result = dict(dtype=str(dtype), width=width, accepted=accepted, rounds=rounds,
                  ring_len=ring_len, carries_mamba=carries_mamba,
                  move_row=move_row, reuse_state=reuse_state,
                  materialize=materialize, pending=pending, identities=identities, errors=errors)
    result["preserved"] = all(e["restore_delta"] < 0.001 for e in errors)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    results = []
    for dtype, width, accepted in [
        (torch.float32, 6, 0),
        (torch.float32, 6, 1),
        (torch.float32, 6, 3),
        (torch.float32, 6, 4),
        (torch.float32, 6, 5),
        (torch.float32, 6, 6),
        (torch.float32, 8, 3),
        (torch.bfloat16, 6, 3),
    ]:
        for materialize in [False, True]:
            results.append(run(dtype, width, accepted, materialize, 1))
    for materialize in [False, True]:
        results.append(run(torch.float32, 8, 8, materialize, 1, ring_len=32))
        for carries_mamba, move_row, reuse_state in [
            (True, True, False), (True, False, True), (True, True, True),
            (False, False, False), (False, True, True),
        ]:
            results.append(run(
                torch.float32, 6, 3, materialize, 3,
                carries_mamba=carries_mamba, move_row=move_row, reuse_state=reuse_state,
            ))
    results.append(run(torch.bfloat16, 6, 3, False, 3, move_row=True, reuse_state=True))
    results.append(run(torch.float32, 8, 3, False, 3, move_row=True, reuse_state=True))
    print(json.dumps({"baseline_failures": sum(not r["preserved"] for r in results if not r["materialize"]),
                      "intervention_failures": sum(not r["preserved"] for r in results if r["materialize"])}))
    # A causal diagnostic: controls must preserve state; only pending FP32
    # history should fail. These are measurements, not an upstream regression.
    for result in results:
        expected_preserved = result["materialize"] or not result["pending"][0]
        assert result["preserved"] == expected_preserved, result
