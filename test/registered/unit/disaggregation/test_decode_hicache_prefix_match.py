"""Unit tests for decode-side prefix-match shaping (admission and HiCache)."""

import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCachePreallocMixin,
    release_host_promise,
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


def _decode_req(req):
    return SimpleNamespace(req=req, hicache_storage_tried=False)


class TestDecodeAdmissionMatch(CustomTestCase):
    def _harness(
        self, *, uses_swa_tail: bool, swa_tail_len: int, hicache: bool = False
    ) -> SimpleNamespace:
        harness = SimpleNamespace(
            scheduler=SimpleNamespace(enable_decode_hicache=hicache),
            tree_cache=Mock(),
            _uses_swa_tail_prealloc=lambda: uses_swa_tail,
            _pre_alloc_fill_len=DecodePreallocQueue._pre_alloc_fill_len,
            _swa_tail_len=lambda seq_len: swa_tail_len,
            _build_decode_prefix_match=Mock(),
            _stage_from_storage=Mock(return_value=False),
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

        DecodePreallocQueue._match_prefix_and_lock(harness, _decode_req(_req()))

        (_, _, token_ids), kwargs = match_prefix.call_args
        self.assertEqual(list(token_ids), list(range(PROMPT_LEN - 512)))
        self.assertTrue(kwargs["kv_only"])
        self.assertFalse(kwargs.get("cow_mamba", False))
        harness._build_decode_prefix_match.assert_called_once()

    @patch("sglang.srt.disaggregation.decode.match_prefix_for_req")
    def test_whole_prompt_is_reusable_without_swa_tail(self, match_prefix):
        harness = self._harness(uses_swa_tail=False, swa_tail_len=0)

        DecodePreallocQueue._match_prefix_and_lock(harness, _decode_req(_req()))

        (_, _, token_ids), kwargs = match_prefix.call_args
        self.assertEqual(len(token_ids), PROMPT_LEN)
        self.assertTrue(kwargs["kv_only"])

    @patch("sglang.srt.disaggregation.decode.match_prefix_for_req")
    def test_l3_hit_is_staged_before_anything_is_promised(self, match_prefix):
        # A started fetch defers the promise: no lock, no prefix match, and
        # storage is not consulted again when the request is matched next.
        harness = self._harness(uses_swa_tail=False, swa_tail_len=0, hicache=True)
        harness._stage_from_storage.return_value = True
        decode_req = _decode_req(_req())

        self.assertIsNone(
            DecodePreallocQueue._match_prefix_and_lock(harness, decode_req)
        )
        harness.tree_cache.inc_lock_ref.assert_not_called()
        harness._build_decode_prefix_match.assert_not_called()
        self.assertEqual(harness._stage_from_storage.call_args.args[2], PROMPT_LEN)

        DecodePreallocQueue._match_prefix_and_lock(harness, decode_req)

        harness._stage_from_storage.assert_called_once()
        harness._build_decode_prefix_match.assert_called_once()


class TestDecodePromiseIsPinned(CustomTestCase):
    def test_host_part_of_the_promise_is_pinned_until_released(self):
        tree_cache = Mock()
        tree_cache.inc_host_lock_ref.return_value.to_dec_params.return_value = "pin"
        harness = SimpleNamespace(tree_cache=tree_cache)
        result = SimpleNamespace(
            device_indices=torch.arange(128),
            host_hit_length=256,
            last_device_node=11,
            best_match_node=33,
        )

        match = DecodeHiCachePreallocMixin._build_decode_prefix_match(harness, result)

        tree_cache.inc_host_lock_ref.assert_called_once_with(33)
        self.assertEqual(match.decode_prefix_len, 384)
        self.assertEqual(match.restore_token_count, 256)

        release_host_promise(tree_cache, match)
        release_host_promise(tree_cache, match)

        tree_cache.dec_host_lock_ref.assert_called_once_with(33, "pin")

    def test_device_only_promise_takes_no_host_pin(self):
        tree_cache = Mock()
        harness = SimpleNamespace(tree_cache=tree_cache)
        result = SimpleNamespace(
            device_indices=torch.arange(128),
            host_hit_length=0,
            last_device_node=11,
            best_match_node=11,
        )

        match = DecodeHiCachePreallocMixin._build_decode_prefix_match(harness, result)

        tree_cache.inc_host_lock_ref.assert_not_called()
        self.assertFalse(match.needs_local_restore)


class TestDecodeHiCacheStaging(CustomTestCase):
    def _tree_cache(self, *, hit: int, registers: bool) -> SimpleNamespace:
        ongoing_prefetch = {}

        def prefetch_from_storage(req_id, *_args, **_kwargs):
            if registers:
                ongoing_prefetch[req_id] = object()

        return SimpleNamespace(
            hicache_storage_pass_prefix_keys=False,
            has_ongoing_prefetch=ongoing_prefetch.__contains__,
            is_backuped=Mock(return_value=True),
            is_root=Mock(return_value=False),
            get_last_hash_value=Mock(return_value="hash"),
            query_storage_hit_length=Mock(return_value=hit),
            prefetch_from_storage=Mock(side_effect=prefetch_from_storage),
        )

    def _stage(self, tree_cache, *, reusable_len=1536) -> bool:
        result = SimpleNamespace(
            device_indices=torch.arange(128),
            host_hit_length=256,
            last_host_node=22,
            last_device_node=11,
        )
        return DecodeHiCachePreallocMixin._stage_from_storage(
            SimpleNamespace(tree_cache=tree_cache), _req(), result, reusable_len
        )

    def test_query_and_fetch_stop_at_the_reusable_len_kv_only(self):
        tree_cache = self._tree_cache(hit=1024, registers=True)

        self.assertTrue(self._stage(tree_cache))

        suffix = tree_cache.query_storage_hit_length.call_args.args[1]
        self.assertEqual(list(suffix), list(range(384, 1536)))
        fetch = tree_cache.prefetch_from_storage.call_args
        self.assertEqual(list(fetch.args[2]), list(range(384, 1408)))
        # The hit query counts base KV only; component fetches would make a
        # hybrid fetch all-or-nothing on state decode never reads.
        self.assertTrue(fetch.kwargs["kv_only"])

    def test_storage_miss_fetches_nothing(self):
        tree_cache = self._tree_cache(hit=0, registers=True)

        self.assertFalse(self._stage(tree_cache))

        tree_cache.prefetch_from_storage.assert_not_called()

    def test_declined_fetch_promises_without_waiting(self):
        # A silently declined fetch (rate limit, host buffer) must not leave
        # the request waiting on a prefetch that was never registered.
        self.assertFalse(self._stage(self._tree_cache(hit=1024, registers=False)))

    def test_poll_waits_for_the_fetch_then_consumes_it(self):
        tree_cache = SimpleNamespace(
            check_prefetch_progress=Mock(side_effect=[False, True]),
            pop_prefetch_loaded_tokens=Mock(),
        )
        harness = SimpleNamespace(tree_cache=tree_cache)
        decode_req = SimpleNamespace(req=_req(), hicache_staging=True)

        self.assertFalse(
            DecodeHiCachePreallocMixin._poll_hicache_staging(harness, decode_req)
        )
        self.assertTrue(
            DecodeHiCachePreallocMixin._poll_hicache_staging(harness, decode_req)
        )
        self.assertFalse(decode_req.hicache_staging)
        tree_cache.pop_prefetch_loaded_tokens.assert_called_once()


if __name__ == "__main__":
    unittest.main()
