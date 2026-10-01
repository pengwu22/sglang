"""Decode prealloc on SWA models: the allocator sees the committed prefix."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.disaggregation.decode import alloc_for_decode_prealloc
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE_SIZE = 64
SWA_TAIL_LEN = 192


class TestAllocForDecodePreallocSwa(CustomTestCase):
    def test_swa_branch_uses_total_prefix_len(self):
        # L1 = 1024 on device, L2 = 2048 restored by load_back, fill = 4096.
        allocator = SimpleNamespace(
            page_size=PAGE_SIZE,
            device="cpu",
            alloc_extend_swa_tail=Mock(return_value=torch.arange(1024)),
        )
        req = SimpleNamespace(kv=Mock())

        alloc_for_decode_prealloc(
            allocator,
            req=req,
            fill_len=4096,
            delta_len=1024,
            prefix_len=1024,
            total_prefix_len=3072,
            prefix_indices=torch.arange(1024),
            uses_swa_tail=True,
            swa_tail_len=SWA_TAIL_LEN,
        )

        kwargs = allocator.alloc_extend_swa_tail.call_args.kwargs
        # The allocator must see the same prefix the extend size was derived
        # from; otherwise it fills fill - l1 slots into a fill - total buffer.
        self.assertEqual(kwargs["prefix_lens_cpu"].item(), 3072)
        self.assertEqual(kwargs["seq_lens_cpu"].item(), 4096)
        self.assertEqual(kwargs["extend_num_tokens"], 1024)
        self.assertEqual(kwargs["swa_tail_len"], SWA_TAIL_LEN)
        req.kv.set_evicted_seqlen.assert_called_once_with(
            ComponentType.SWA, 4096 - SWA_TAIL_LEN
        )


if __name__ == "__main__":
    unittest.main()
