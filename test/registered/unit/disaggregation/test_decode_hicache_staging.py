"""Decode admission stages L3 hits into the host tier before promising them."""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeRequest
from sglang.srt.disaggregation.decode_hicache_mixin import DecodePrefixMatch
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.separate_buffer_allocator_double import (
    bind_separate_buffer_capacity,
)
from sglang.test.test_utils import CustomTestCase, enter_override

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _queue(decode_req: DecodeRequest) -> DecodePreallocQueue:
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue.pp_size = 1
    queue.queue = [decode_req]
    queue.pending_reqs = []
    queue.retracted_queue = []
    queue.num_reserved_decode_tokens = 0
    queue._resolve_pending_reqs = MagicMock()
    queue._update_handshake_waiters = MagicMock()
    queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
    queue._uses_swa_reservation = MagicMock(return_value=False)
    queue._allocatable_token_budgets = MagicMock(return_value=0)
    queue._match_prefix_and_lock = MagicMock()
    queue._pre_alloc = MagicMock(side_effect=AssertionError("must not prealloc"))
    queue.transfer_queue = MagicMock(queue=[], enable_staging=False)
    queue.tree_cache = MagicMock()
    queue.req_to_token_pool = MagicMock(mamba_allocator=None)
    queue.req_to_token_pool.available_size.return_value = 1
    queue.req_to_metadata_buffer_idx_allocator = MagicMock()
    queue.req_to_metadata_buffer_idx_allocator.available_size.return_value = 1
    queue.token_to_kv_pool_allocator = MagicMock(page_size=1)
    bind_separate_buffer_capacity(queue.token_to_kv_pool_allocator)
    scheduler = MagicMock(
        enable_hisparse=False,
        enable_decode_hicache=True,
        enable_priority_scheduling=False,
        enable_lora=False,
        waiting_queue=[],
        last_batch=None,
    )
    scheduler.running_batch.reqs = []
    queue.scheduler = scheduler
    return queue


def _decode_req() -> DecodeRequest:
    req = MagicMock(
        rid="req-0",
        finished_reason=None,
        origin_input_ids=list(range(8)),
        output_ids=[99],
        pd_rebootstrap_in_progress=False,
    )
    req.sampling_params.max_new_tokens = 16
    return DecodeRequest(req=req, kv_receiver=MagicMock(), waiting_for_input=True)


class TestDecodeHiCacheStagingAdmission(CustomTestCase):
    def setUp(self):
        enter_override(
            self,
            get_context().override_server_args(
                disaggregation_decode_enable_radix_cache=True
            ),
        )

    def test_started_fetch_leaves_the_request_queued_unpinned(self):
        decode_req = _decode_req()
        queue = _queue(decode_req)
        queue._match_prefix_and_lock.return_value = None

        preallocated, failed = queue.pop_preallocated()

        self.assertEqual((preallocated, failed), ([], []))
        self.assertTrue(decode_req.hicache_staging)
        self.assertEqual(queue.queue, [decode_req])

    def test_in_flight_fetch_is_not_matched_again(self):
        decode_req = _decode_req()
        decode_req.hicache_staging = True
        queue = _queue(decode_req)
        queue.tree_cache.check_prefetch_progress.return_value = False

        queue.pop_preallocated()

        queue._match_prefix_and_lock.assert_not_called()
        self.assertTrue(decode_req.hicache_staging)

    def test_resolved_fetch_is_promised_from_the_host_tier(self):
        # Once the fetch lands, the request is matched again and its promise
        # (here rejected by the budget) carries the staged pages as L2.
        decode_req = _decode_req()
        decode_req.hicache_staging = True
        decode_req.hicache_storage_tried = True
        queue = _queue(decode_req)
        queue.tree_cache.check_prefetch_progress.return_value = True
        queue._match_prefix_and_lock.return_value = DecodePrefixMatch(
            prefix_indices=torch.arange(2),
            l2_host_hit_length=4,
            last_device_node=11,
            host_anchor=5,
            host_lock="pin",
        )

        queue.pop_preallocated()

        queue.tree_cache.pop_prefetch_loaded_tokens.assert_called_once()
        queue._match_prefix_and_lock.assert_called_once_with(decode_req)
        self.assertFalse(decode_req.hicache_staging)
        # The budget rejected it, so both pins are dropped before retrying.
        queue.tree_cache.dec_host_lock_ref.assert_called_once_with(5, "pin")

    def test_abort_while_staging_releases_the_fetch(self):
        decode_req = _decode_req()
        decode_req.hicache_staging = True
        decode_req.req.finished_reason = FINISH_ABORT("client gone")
        queue = _queue(decode_req)

        _, failed = queue.pop_preallocated()

        self.assertEqual(failed, [decode_req])
        queue.tree_cache.finish.assert_called_once()
        self.assertFalse(decode_req.hicache_staging)


if __name__ == "__main__":
    unittest.main()
