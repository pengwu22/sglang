"""Decode-side HiCache load-back restores base KV only."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCacheTransferMixin,
    DecodePrefixMatch,
    HiCacheRestoreResult,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _tree_cache(*, new_indices) -> Mock:
    return Mock(
        check_prefetch_progress=Mock(return_value=True),
        init_load_back=Mock(return_value=(new_indices, 99)),
        inc_lock_ref=Mock(return_value=Mock(to_dec_params=Mock())),
    )


def _decode_req(*, prefix_indices, l2: int, l3: int) -> SimpleNamespace:
    return SimpleNamespace(
        req=SimpleNamespace(
            rid="req-0",
            cache_request_handle=object(),
            origin_input_ids=list(range(8)),
            extra_key=None,
            cache_salt=None,
            last_node=None,
        ),
        prefix_match=DecodePrefixMatch(
            prefix_indices=prefix_indices,
            l2_host_hit_length=l2,
            l3_storage_hit_length=l3,
            last_device_node=11,
            last_host_node=None,
        ),
        hicache_restore_status=HiCacheRestoreResult.PENDING,
        hicache_restored_node=None,
        hicache_load_consumer_index=-1,
    )


def _rematch(device_indices, *, host_hit_length: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        best_match_node=5,
        last_device_node=11,
        host_hit_length=host_hit_length,
        device_indices=device_indices,
    )


class TestDecodeLoadBackIsKvOnly(CustomTestCase):
    @patch("sglang.srt.disaggregation.decode_hicache_mixin.match_prefix_for_req")
    def test_init_load_back_requests_kv_only(self, match_prefix):
        # L1 = 2 device tokens, L2 = 2 host tokens to restore.
        match_prefix.return_value = _rematch(torch.tensor([10, 11]), host_hit_length=2)
        tree_cache = _tree_cache(
            new_indices=torch.tensor([20, 21]),
        )
        dr = _decode_req(prefix_indices=torch.tensor([10, 11]), l2=2, l3=0)

        queued = DecodeHiCacheTransferMixin._try_hicache_queue_load_back(
            SimpleNamespace(tree_cache=tree_cache), dr
        )

        self.assertTrue(queued)
        params = tree_cache.init_load_back.call_args.args[0]
        # Component state (SWA window / Mamba) comes from the P/D transfer;
        # the decode restore must never write it.
        self.assertTrue(params.kv_only)
        self.assertEqual(params.best_match_node, 5)
        self.assertEqual(params.host_hit_length, 2)
        self.assertEqual(dr.hicache_restored_node, 99)
        self.assertEqual(dr.hicache_restored_kv_indices.tolist(), [20, 21])

    @patch("sglang.srt.disaggregation.decode_hicache_mixin.match_prefix_for_req")
    def test_resident_full_kv_needs_only_a_completion_fence(self, match_prefix):
        # FULL-only rematching sees resident KV even without live components.
        # There is no new DMA, but readiness still needs the stream fence.
        match_prefix.return_value = _rematch(torch.tensor([10, 11, 20, 21]))
        tree_cache = _tree_cache(
            new_indices=torch.tensor([20, 21]),
        )
        dr = _decode_req(prefix_indices=torch.tensor([10, 11]), l2=2, l3=0)

        queued = DecodeHiCacheTransferMixin._try_hicache_queue_load_back(
            SimpleNamespace(tree_cache=tree_cache), dr
        )

        self.assertTrue(queued)
        self.assertEqual(dr.hicache_restore_status, HiCacheRestoreResult.PENDING)
        self.assertEqual(dr.hicache_restored_kv_indices.tolist(), [20, 21])

    @patch("sglang.srt.disaggregation.decode_hicache_mixin.match_prefix_for_req")
    def test_full_kv_behind_tombstoned_component_state_is_restored(self, match_prefix):
        # A prefix shared across requests (GSM8K few-shot header on an SWA
        # model): its FULL KV is in the tree but the SWA state is tombstoned,
        # so the all-component rematch reports nothing at any tier while the
        # KV-only L3 promise covers it. The restore must locate the KV the
        # KV-only way instead of failing the coverage check with a 500.
        match_prefix.return_value = _rematch(
            torch.tensor([], dtype=torch.int64), host_hit_length=4
        )
        tree_cache = _tree_cache(
            new_indices=torch.tensor([20, 21, 22, 23]),
        )
        dr = _decode_req(prefix_indices=torch.tensor([], dtype=torch.int64), l2=0, l3=4)

        queued = DecodeHiCacheTransferMixin._try_hicache_queue_load_back(
            SimpleNamespace(tree_cache=tree_cache), dr
        )

        self.assertTrue(queued)
        self.assertEqual(dr.hicache_restore_status, HiCacheRestoreResult.PENDING)
        self.assertTrue(match_prefix.call_args.kwargs["kv_only"])
        self.assertEqual(match_prefix.call_args.args[2], [0, 1, 2, 3])
        params = tree_cache.init_load_back.call_args.args[0]
        self.assertEqual(params.best_match_node, 5)
        self.assertEqual(params.host_hit_length, 4)
        self.assertTrue(params.kv_only)
        self.assertEqual(dr.hicache_restored_kv_indices.tolist(), [20, 21, 22, 23])

    @patch("sglang.srt.disaggregation.decode_hicache_mixin.match_prefix_for_req")
    def test_device_match_covering_the_promise_needs_no_load_back(self, match_prefix):
        # The rematch already covers decode_prefix_len (and more): nothing to
        # restore, and the commit is bounded to [l1, decode_prefix_len).
        match_prefix.return_value = _rematch(torch.tensor([10, 11, 12, 13]))
        tree_cache = _tree_cache(
            new_indices=torch.tensor([], dtype=torch.int64),
        )
        dr = _decode_req(prefix_indices=torch.tensor([10, 11]), l2=0, l3=0)

        queued = DecodeHiCacheTransferMixin._try_hicache_queue_load_back(
            SimpleNamespace(tree_cache=tree_cache), dr
        )

        self.assertTrue(queued)
        self.assertEqual(dr.hicache_restore_status, HiCacheRestoreResult.PENDING)
        tree_cache.init_load_back.assert_not_called()
        self.assertEqual(dr.hicache_restored_node, 11)
        self.assertEqual(dr.hicache_restored_kv_indices.numel(), 0)


class TestDecodeLoadBackCompletion(CustomTestCase):
    @patch("sglang.srt.disaggregation.decode_hicache_mixin.match_prefix_for_req")
    def test_shared_prefix_in_one_batch_waits_for_the_same_dma(self, match_prefix):
        match_prefix.side_effect = [
            _rematch(torch.empty(0, dtype=torch.int64), host_hit_length=4),
            _rematch(torch.arange(4)),
        ]
        cache = _tree_cache(new_indices=torch.arange(4))
        counter = SimpleNamespace(producer_index=-1, num_counters=3)
        cache.cache_controller.layer_done_counter = counter
        done = True
        cache.is_load_back_event_done.side_effect = lambda index: index < 0 or done

        def start_loading():
            nonlocal done
            counter.producer_index = 0
            done = False
            return 0

        cache.ready_to_load_host_cache.side_effect = start_loading
        queue = DecodeHiCacheTransferMixin()
        queue.tree_cache = cache
        reqs = [
            _decode_req(prefix_indices=torch.empty(0, dtype=torch.int64), l2=4, l3=0)
            for _ in range(2)
        ]
        queue._process_hicache_local_restores(reqs)
        cache.init_load_back.assert_called_once()
        for dr in reqs:
            self.assertEqual(dr.hicache_restore_status, HiCacheRestoreResult.PENDING)
            self.assertEqual(dr.hicache_load_consumer_index, 0)
        done = True
        queue._process_hicache_local_restores(reqs)
        for dr in reqs:
            self.assertEqual(dr.hicache_restore_status, HiCacheRestoreResult.READY)

    @patch("sglang.srt.disaggregation.decode_hicache_mixin.match_prefix_for_req")
    def test_resident_alias_waits_for_previous_dma(self, match_prefix):
        # A prior request queued a restore, then a split gave this request a
        # different node id. Its indices exist but their bytes are not ready.
        match_prefix.return_value = _rematch(torch.arange(4))
        cache = _tree_cache(
            new_indices=torch.arange(4),
        )
        counter = SimpleNamespace(producer_index=0, num_counters=3)
        cache.cache_controller.layer_done_counter = counter
        done = {0: False, 1: True}
        cache.is_load_back_event_done.side_effect = lambda index: (
            index < 0 or done[index]
        )
        cache.ready_to_load_host_cache.return_value = -1
        queue = DecodeHiCacheTransferMixin()
        queue.tree_cache = cache
        dr = _decode_req(prefix_indices=torch.empty(0, dtype=torch.int64), l2=4, l3=0)

        queue._process_hicache_local_restores([dr])

        self.assertEqual(dr.hicache_restore_status, HiCacheRestoreResult.PENDING)
        self.assertEqual(dr.hicache_load_consumer_index, 0)
        done[0] = True
        queue._process_hicache_local_restores([dr])
        self.assertEqual(dr.hicache_restore_status, HiCacheRestoreResult.READY)

    def test_l1_hit_waits_for_previous_dma(self):
        cache = Mock()
        cache.cache_controller.layer_done_counter = SimpleNamespace(
            producer_index=0, num_counters=3
        )
        cache.is_load_back_event_done.side_effect = lambda index: index != 0
        queue = DecodeHiCacheTransferMixin()
        queue.tree_cache = cache
        dr = _decode_req(prefix_indices=torch.arange(4), l2=0, l3=0)

        queue._process_hicache_local_restores([dr])

        self.assertEqual(dr.hicache_restore_status, HiCacheRestoreResult.PENDING)
        # Newer loads must not move this request's fence and starve its L1 hit.
        cache.cache_controller.layer_done_counter.producer_index = 1
        cache.is_load_back_event_done.side_effect = lambda index: index != 1
        queue._process_hicache_local_restores([dr])
        self.assertEqual(dr.hicache_restore_status, HiCacheRestoreResult.READY)


if __name__ == "__main__":
    unittest.main()
