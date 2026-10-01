"""Unit tests for decode-side prefix-match shaping (admission and HiCache)."""

import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCachePreallocMixin,
    DecodePrefixMatch,
)
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PROMPT_LEN = 4096


def _req() -> SimpleNamespace:
    return SimpleNamespace(
        rid="req-0",
        cache_request_handle=CacheRequestHandle("req-0", 0),
        origin_input_ids=list(range(PROMPT_LEN)),
        output_ids=[PROMPT_LEN],
        extra_key=None,
        cache_salt=None,
    )


class TestDecodeAdmissionMatch(CustomTestCase):
    def _harness(self, *, uses_swa_tail: bool, swa_tail_len: int) -> SimpleNamespace:
        harness = SimpleNamespace(
            tree_cache=Mock(),
            _uses_swa_tail_prealloc=lambda: uses_swa_tail,
            _pre_alloc_fill_len=DecodePreallocQueue._pre_alloc_fill_len,
            _swa_tail_len=lambda seq_len: swa_tail_len,
            _build_decode_prefix_match=Mock(),
        )
        harness._reusable_prefix_len = types.MethodType(
            DecodePreallocQueue._reusable_prefix_len, harness
        )
        return harness

    @patch("sglang.srt.disaggregation.decode.match_prefix_for_req")
    def test_match_is_full_kv_only_and_stops_at_the_swa_tail(self, match_prefix):
        # Prefill transfers the SWA tail [fill - tail, fill) and the Mamba
        # state, so decode reuses FULL KV only, and only before the tail.
        harness = self._harness(uses_swa_tail=True, swa_tail_len=512)

        DecodePreallocQueue._match_prefix_and_lock(harness, _req())

        (_, _, token_ids), kwargs = match_prefix.call_args
        self.assertEqual(list(token_ids), list(range(PROMPT_LEN - 512)))
        self.assertTrue(kwargs["kv_only"])
        self.assertFalse(kwargs.get("cow_mamba", False))
        harness._build_decode_prefix_match.assert_called_once()
        self.assertEqual(
            harness._build_decode_prefix_match.call_args.args[2], PROMPT_LEN - 512
        )

    @patch("sglang.srt.disaggregation.decode.match_prefix_for_req")
    def test_whole_prompt_is_reusable_without_swa_tail(self, match_prefix):
        harness = self._harness(uses_swa_tail=False, swa_tail_len=0)

        DecodePreallocQueue._match_prefix_and_lock(harness, _req())

        (_, _, token_ids), kwargs = match_prefix.call_args
        self.assertEqual(len(token_ids), PROMPT_LEN)
        self.assertTrue(kwargs["kv_only"])


class TestDecodeHiCacheStorageQuery(CustomTestCase):
    def test_storage_query_stops_at_the_reusable_len(self):
        tree_cache = SimpleNamespace(
            hicache_storage_pass_prefix_keys=False,
            is_backuped=Mock(return_value=True),
            is_root=Mock(return_value=False),
            get_last_hash_value=Mock(return_value="hash"),
            query_storage_hit_length=Mock(return_value=1024),
        )
        harness = SimpleNamespace(
            scheduler=SimpleNamespace(enable_decode_hicache=True),
            tree_cache=tree_cache,
        )
        result = SimpleNamespace(
            device_indices=torch.arange(128),
            host_hit_length=256,
            last_host_node=22,
            last_device_node=11,
        )

        match = DecodeHiCachePreallocMixin._build_decode_prefix_match(
            harness, _req(), result, reusable_len=1536
        )

        suffix = tree_cache.query_storage_hit_length.call_args.args[1]
        self.assertEqual(list(suffix), list(range(384, 1536)))
        self.assertEqual(match.l2_host_hit_length, 256)
        self.assertEqual(match.l3_storage_hit_length, 1024)
        self.assertEqual(match.decode_prefix_len, 1408)
        self.assertEqual(match.last_host_node, 22)


class TestDecodeHiCachePrefetchDecline(CustomTestCase):
    def _prefetch(self, *, registers: bool) -> DecodePrefixMatch:
        ongoing_prefetch = {}

        def prefetch_from_storage(req_id, *_args, **_kwargs):
            if registers:
                ongoing_prefetch[req_id] = object()

        harness = SimpleNamespace(
            tree_cache=SimpleNamespace(
                hicache_storage_pass_prefix_keys=False,
                ongoing_prefetch=ongoing_prefetch,
                has_ongoing_prefetch=ongoing_prefetch.__contains__,
                get_last_hash_value=Mock(return_value="hash"),
                prefetch_from_storage=Mock(side_effect=prefetch_from_storage),
            ),
        )
        prefix_match = DecodePrefixMatch(
            prefix_indices=torch.arange(256),
            l2_host_hit_length=0,
            l3_storage_hit_length=512,
            last_device_node=11,
            last_host_node=22,
        )
        DecodeHiCachePreallocMixin._start_hicache_prefetch(
            harness, _req(), prefix_match
        )
        self.prefetch_call = harness.tree_cache.prefetch_from_storage.call_args
        return prefix_match

    def test_registered_prefetch_keeps_l3_promise(self):
        prefix_match = self._prefetch(registers=True)

        self.assertTrue(prefix_match.prefetch_registered)
        self.assertEqual(prefix_match.l3_storage_hit_length, 512)

    def test_prefetch_is_kv_only(self):
        # The hit query is KV-only; fetching component objects too would make
        # a hybrid prefetch all-or-nothing on state decode never reads.
        self._prefetch(registers=True)

        self.assertTrue(self.prefetch_call.kwargs["kv_only"])

    def test_declined_prefetch_degrades_to_l2_only(self):
        # A silently declined prefetch (rate limit, host buffer alloc failure)
        # would leave the promised L3 range unrestorable after the transfer
        # was already trimmed by decode_prefix_len.
        prefix_match = self._prefetch(registers=False)

        self.assertFalse(prefix_match.prefetch_registered)
        self.assertEqual(prefix_match.l3_storage_hit_length, 0)
        self.assertEqual(prefix_match.decode_prefix_len, 256)


if __name__ == "__main__":
    unittest.main()
