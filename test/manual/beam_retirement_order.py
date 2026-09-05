"""Deterministic beam-retirement diagnostic using real CUDA stream operations.

A graph captures the scheduler-shared metadata copy and the production external
read-done event. A device semaphore holds the later production beam remap.
The production scheduler barrier and beam commit/cleanup then run on a separate
stream. This is a component integration test, not a model-serving test.
"""

import faulthandler
import json
import os
import threading
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch
from cuda.bindings import driver

from sglang.srt.beam_search.beam_group import BeamGroup
from sglang.srt.beam_search.coordinator import BeamCoordinator
from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def fixture():
    pool = ReqToTokenPool(size=4, max_context_len=16, device="cuda", enable_memory_saver=False)
    allocator = TokenToKVPoolAllocator(
        size=64, dtype=torch.bfloat16, device="cuda", kvcache=None, need_sort=False,
    )
    allocated = allocator.alloc(8)
    leader = object.__new__(Req)
    from sglang.srt.managers.schedule_batch import ReqKvInfo
    rows_cpu = pool.alloc_rows(2)
    leader.kv = ReqKvInfo(req_pool_idx=rows_cpu[0], kv_allocated_len=5, kv_committed_len=5)
    leader.output_ids = array("q", [10])
    leader.to_finish = FINISH_ABORT("cancelled")
    leader.finished_reason = None
    group = BeamGroup(beam_width=2, max_new_tokens=20, device="cuda")
    group.leader = leader
    group.prompt_len = 2
    group.member_rows_cpu = torch.tensor(rows_cpu[1:], dtype=torch.int64)
    group.member_rows = group.member_rows_cpu.to("cuda")
    group.all_rows = torch.tensor(rows_cpu, device="cuda", dtype=torch.int64)
    leader.beam_group = group
    pool.req_to_token[group.all_rows, :2] = allocated[:2].to(torch.int32)
    pool.req_to_token[group.all_rows, 2:5] = allocated[2:].reshape(2, 3).to(torch.int32)
    future = FutureMap(
        device=torch.device("cuda"), spec_algo=SpeculativeAlgorithm.NONE,
        req_to_token_pool=pool, needs_cpu_seq_lens=False,
    )
    coordinator = BeamCoordinator(
        model_config=None, spec_algorithm=SpeculativeAlgorithm.NONE,
        dllm_enabled=False, max_req_len=16, req_to_token_pool=pool,
        token_to_kv_pool_allocator=allocator, tree_cache=None, future_map=future,
    )
    coordinator._num_live_groups = 1
    return pool, allocator, leader, group, coordinator


def run(mode):
    print(json.dumps({"mode": mode, "phase": "fixture"}), flush=True)
    forward, schedule, release = torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()
    parent = torch.tensor([0, 0], device="cuda", dtype=torch.int64)
    tokens = torch.tensor([12, 13], device="cuda", dtype=torch.int64)
    gate = torch.zeros(1, device="cuda", dtype=torch.int32)
    # Warm every data-dependent operation before holding a stream. This also
    # keeps cudaMalloc/module initialization from providing incidental fences.
    for _ in range(3):
        p, a, leader, g, c = fixture()
        forward.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(forward):
            c._apply_survivors(g, tokens, parent, 2)
        schedule.wait_stream(forward)
        with torch.cuda.stream(schedule):
            c.commit_decode(SimpleNamespace(reqs=[leader], forward_iter=2))
        torch.cuda.synchronize()
    del p, a, leader, g, c
    pool, allocator, leader, group, coordinator = fixture()
    initial_free = allocator.get_all_free_pages().cpu().tolist()
    member = int(group.member_rows_cpu[0])
    graph_owner = SimpleNamespace(
        device_module=torch.cuda, in_graph_metadata_prep_done=None,
        model_runner=SimpleNamespace(shared_read_done_event=None),
    )
    metadata = torch.empty((2, 16), device="cuda", dtype=torch.int32)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=forward):
        metadata.copy_(pool.req_to_token[group.all_rows])
        DecodeCudaGraphRunner._record_in_graph_metadata_prep_done(graph_owner)
        metadata.add_(0)
    torch.cuda.synchronize()
    assert graph_owner.in_graph_metadata_prep_done is not None
    scheduler = SimpleNamespace(
        _war_barrier_enabled=True, schedule_stream=schedule, forward_stream=forward,
        model_worker=SimpleNamespace(last_shared_read_runner=graph_owner.model_runner),
    )
    finished = torch.cuda.Event()
    gate.fill_(0)
    torch.cuda.synchronize()
    pending = []
    held = False
    rescued = threading.Event()

    def rescue():
        rescued.set()
        with torch.cuda.stream(release):
            gate.fill_(1)

    timer = threading.Timer(15, rescue)
    try:
        with torch.cuda.stream(forward):
            graph.replay()
            DecodeCudaGraphRunner._publish_read_done(graph_owner, in_graph=True)
            status, = driver.cuStreamWaitValue32(
                driver.CUstream(forward.cuda_stream),
                driver.CUdeviceptr(gate.data_ptr()), 1, 1,
            )
            assert status == driver.CUresult.CUDA_SUCCESS, status
            held = True
            timer.start()
            print(json.dumps({"mode": mode, "phase": "gate installed"}), flush=True)
            coordinator._apply_survivors(
                group, tokens, None if mode == "final" else parent, 2,
            )
            pending = list(group.pending_orphans)
            print(json.dumps({"mode": mode, "phase": "remap submitted"}), flush=True)
            finished.record()
            if mode == "late":
                graph_owner.model_runner.shared_read_done_event = finished

        with patch.dict(os.environ, {"SGLANG_FORCE_COARSE_WAR_BARRIER": "1" if mode == "coarse" else "0"}):
            Scheduler._apply_war_barrier(scheduler)
        print(json.dumps({"mode": mode, "phase": "barrier submitted"}), flush=True)

        if mode in ("coarse", "late"):
            # The scheduler already waits for the remap. Open the deterministic
            # gate before invoking the data-dependent cleanup on that stream.
            with torch.cuda.stream(release):
                gate.fill_(1)
            held = False
        with torch.cuda.stream(schedule):
            if mode in ("early", "ordinary") and pending:
                # Legal stale bytes in outputs whose producer has not run.
                # All values stay within the pool; no out-of-range index is used.
                pending[0].old_mapping.zero_()
                pending[0].new_mapping.zero_()
            if mode == "ordinary":
                leader.to_finish = None
            coordinator.commit_decode(SimpleNamespace(reqs=[leader], forward_iter=1))
            print(json.dumps({"mode": mode, "phase": "commit returned"}), flush=True)
            schedule.synchronize()
            retired_before_completion = group.retired and not finished.query()
            if mode == "ordinary":
                assert not group.retired and len(group.pending_orphans) == 1
                assert member not in pool.free_slots
            else:
                assert group.retired
                new_rows = pool.alloc_rows(1)
                assert new_rows == [member]
                replacement = allocator.alloc(3)
                pool.req_to_token[member, 2:5] = replacement.to(torch.int32)
                expected_mapping = replacement.cpu().tolist()
        schedule.synchronize()
        if held:
            with torch.cuda.stream(release):
                gate.fill_(1)
            held = False
        forward.synchronize()
        if mode == "ordinary":
            leader.to_finish = FINISH_ABORT("cancelled")
            with torch.cuda.stream(schedule):
                coordinator.commit_decode(SimpleNamespace(reqs=[leader], forward_iter=2))
            schedule.synchronize()
            observed_mapping = expected_mapping = None
        else:
            observed_mapping = pool.req_to_token[member, 2:5].cpu().tolist()
        free_before_repeat = allocator.get_all_free_pages().cpu().tolist()
        coordinator.commit_decode(SimpleNamespace(reqs=[leader], forward_iter=2))
        free_after_repeat = allocator.get_all_free_pages().cpu().tolist()
        assert free_before_repeat == free_after_repeat
        assert len(free_after_repeat) == len(set(free_after_repeat))
        result = dict(
            mode=mode, retired_before_remap_completed=retired_before_completion,
            replacement_expected=expected_mapping, replacement_observed=observed_mapping,
            replacement_preserved=expected_mapping == observed_mapping,
            idempotent_retire=True,
            initial_free_count=len(initial_free), final_free_count=len(free_after_repeat),
            watchdog_released_gate=rescued.is_set(),
        )
        print(json.dumps(result), flush=True)
        return result
    finally:
        timer.cancel()
        if held:
            with torch.cuda.stream(release):
                gate.fill_(1)
        torch.cuda.synchronize()


if __name__ == "__main__":
    faulthandler.dump_traceback_later(10, repeat=False)
    results = [run(mode) for mode in ("early", "coarse", "late", "ordinary", "final")]
    faulthandler.cancel_dump_traceback_later()
    assert results[0]["retired_before_remap_completed"]
    assert not results[0]["replacement_preserved"]
    assert all(result["replacement_preserved"] for result in results[1:])
