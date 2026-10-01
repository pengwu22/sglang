"""Decode-side HiCache restore: FULL-only rematch of the promise, KV-only load."""

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
    )


def _rematch(device_indices, *, host_hit_length: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        best_match_node=5,
        last_device_node=11,
        host_hit_length=host_hit_length,
        device_indices=device_indices,
    )


@patch("sglang.srt.disaggregation.decode_hicache_mixin.match_prefix_for_req")
class TestDecodeRestoreIsKvOnly(CustomTestCase):
    def test_rematch_covers_exactly_the_promise_full_kv_only(self, match_prefix):
        # L1 = 2 on device, L2 = 2 + L3 = 2 promised; the prompt is longer.
        match_prefix.return_value = _rematch(torch.tensor([10, 11]), host_hit_length=4)
        tree_cache = _tree_cache(new_indices=torch.tensor([20, 21, 22, 23]))
        dr = _decode_req(prefix_indices=torch.tensor([10, 11]), l2=2, l3=2)

        DecodeHiCacheTransferMixin._try_hicache_queue_load_back(
            SimpleNamespace(tree_cache=tree_cache), dr
        )

        (_, _, token_ids), kwargs = match_prefix.call_args
        # Bounded at decode_prefix_len: nothing past the promise is loaded.
        self.assertEqual(list(token_ids), [0, 1, 2, 3, 4, 5])
        # Same match as admission: tombstoned SWA / Mamba state must not end
        # it (a shared prefix whose window belongs to other requests' tails).
        self.assertTrue(kwargs["kv_only"])

    def test_host_hit_queues_a_kv_only_load_back(self, match_prefix):
        match_prefix.return_value = _rematch(torch.tensor([10, 11]), host_hit_length=2)
        tree_cache = _tree_cache(new_indices=torch.tensor([20, 21]))
        dr = _decode_req(prefix_indices=torch.tensor([10, 11]), l2=2, l3=0)

        queued = DecodeHiCacheTransferMixin._try_hicache_queue_load_back(
            SimpleNamespace(tree_cache=tree_cache), dr
        )

        self.assertTrue(queued)
        params = tree_cache.init_load_back.call_args.args[0]
        # Component state comes from the P/D transfer; the restore must never
        # write it into the slots that transfer lands in.
        self.assertTrue(params.kv_only)
        self.assertEqual(params.best_match_node, 5)
        self.assertEqual(params.host_hit_length, 2)
        self.assertEqual(dr.hicache_restored_node, 99)
        self.assertEqual(dr.hicache_restored_kv_indices.tolist(), [20, 21])
        self.assertEqual(dr.req.last_node, 11)

    def test_device_resident_promise_needs_no_dma(self, match_prefix):
        # Another request already loaded the promised range back to device.
        match_prefix.return_value = _rematch(torch.tensor([10, 11, 12, 13]))
        tree_cache = _tree_cache(new_indices=torch.tensor([], dtype=torch.int64))
        dr = _decode_req(prefix_indices=torch.tensor([10, 11]), l2=2, l3=0)

        queued = DecodeHiCacheTransferMixin._try_hicache_queue_load_back(
            SimpleNamespace(tree_cache=tree_cache), dr
        )

        self.assertFalse(queued)
        self.assertEqual(dr.hicache_restore_status, HiCacheRestoreResult.READY)
        self.assertEqual(dr.hicache_restored_kv_indices.tolist(), [12, 13])

    def test_short_coverage_fails_the_restore(self, match_prefix):
        match_prefix.return_value = _rematch(torch.tensor([10, 11]), host_hit_length=2)
        tree_cache = _tree_cache(new_indices=torch.tensor([], dtype=torch.int64))
        dr = _decode_req(prefix_indices=torch.tensor([10, 11]), l2=2, l3=0)

        queued = DecodeHiCacheTransferMixin._try_hicache_queue_load_back(
            SimpleNamespace(tree_cache=tree_cache), dr
        )

        self.assertFalse(queued)
        self.assertEqual(dr.hicache_restore_status, HiCacheRestoreResult.FAILED)
        tree_cache.inc_lock_ref.assert_not_called()


if __name__ == "__main__":
    unittest.main()
