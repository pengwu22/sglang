"""Unit tests for decode HiCache TreeCore interactions."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.disaggregation.decode_hicache_mixin import DecodeHiCachePreallocMixin
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _req():
    return SimpleNamespace(
        rid="req-0",
        cache_request_handle=CacheRequestHandle("req-0", 0),
        origin_input_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        extra_key="model",
        cache_salt="tenant-a",
    )


def _result():
    return SimpleNamespace(
        device_indices=torch.tensor([10, 11]),
        host_hit_length=2,
        last_device_node=11,
        last_host_node=22,
    )


class TestDecodeHiCacheTreeCore(CustomTestCase):
    def test_storage_probe_and_prefetch_use_node_handles(self):
        ongoing_prefetch = {}

        def register_prefetch(req_id, *_args, **_kwargs):
            ongoing_prefetch[req_id] = object()

        tree_cache = SimpleNamespace(
            hicache_storage_pass_prefix_keys=True,
            has_ongoing_prefetch=ongoing_prefetch.__contains__,
            is_backuped=Mock(return_value=True),
            is_root=Mock(return_value=False),
            get_last_hash_value=Mock(return_value="h2"),
            get_prefix_hash_values=Mock(return_value=["h0", "h1"]),
            query_storage_hit_length=Mock(return_value=2),
            prefetch_from_storage=Mock(side_effect=register_prefetch),
        )
        harness = SimpleNamespace(tree_cache=tree_cache)
        req = _req()

        staged = DecodeHiCachePreallocMixin._stage_from_storage(
            harness, req, _result(), reusable_len=8
        )

        self.assertTrue(staged)
        tree_cache.query_storage_hit_length.assert_called_once_with(
            22,
            [4, 5, 6, 7],
            "h2",
            ["h0", "h1"],
            extra_key="model",
            cache_salt="tenant-a",
        )
        tree_cache.prefetch_from_storage.assert_called_once_with(
            req.cache_request_handle,
            22,
            [4, 5],
            "h2",
            ["h0", "h1"],
            extra_key="model",
            cache_salt="tenant-a",
            kv_only=True,
        )

    def test_stale_prefetch_anchor_stages_nothing(self):
        tree_cache = SimpleNamespace(
            hicache_storage_pass_prefix_keys=True,
            is_backuped=Mock(return_value=True),
            is_root=Mock(return_value=False),
            get_last_hash_value=Mock(side_effect=KeyError(22)),
            get_prefix_hash_values=Mock(),
            query_storage_hit_length=Mock(),
            prefetch_from_storage=Mock(),
        )
        harness = SimpleNamespace(tree_cache=tree_cache)

        staged = DecodeHiCachePreallocMixin._stage_from_storage(
            harness, _req(), _result(), reusable_len=8
        )

        self.assertFalse(staged)
        tree_cache.get_prefix_hash_values.assert_not_called()
        tree_cache.prefetch_from_storage.assert_not_called()


if __name__ == "__main__":
    unittest.main()
