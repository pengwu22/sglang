"""Write-only PD decode cache: admission reuses nothing and never queries storage."""

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.pd_disaggregation_hook import handle_pd_disaggregation
from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_override

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PROMPT = array("q", range(8))


class _ResidentTree:
    """Holds every queried key on device; records matches and storage queries."""

    storage_prefetch_is_all_or_nothing = False
    hicache_storage_pass_prefix_keys = False

    def __init__(self):
        self.matches = []
        self.storage_queries = 0
        self.locked = None

    def swa_reprefill_tail_tokens(self):
        return 0

    def supports_mamba(self):
        return True

    def match_prefix(self, params):
        matched = len(params.key.token_ids)
        self.matches.append((list(params.key.token_ids), params.cow_mamba))
        node = "leaf" if matched else "root"
        return SimpleNamespace(
            device_indices=torch.arange(matched),
            last_device_node=node,
            last_host_node=node,
            best_match_node=node,
            host_hit_length=0,
            swa_host_hit_length=0,
            mamba_host_hit_length=0,
            swa_branching_seqlen=None,
            mamba_branching_seqlen=None,
            cache_protected_len=None,
        )

    def inc_lock_ref(self, node):
        self.locked = node
        return SimpleNamespace(to_dec_params=lambda: ("receipt", node))

    def is_backuped(self, node):
        return True

    def is_root(self, node):
        return node == "root"

    def get_last_hash_value(self, node):
        return "hash"

    def query_storage_hit_length(self, *args, extra_key, cache_salt):
        self.storage_queries += 1
        return 0


class TestDecodeCacheWriteOnlyAdmission(CustomTestCase):
    def _admit(self, *, write_only):
        enter_override(
            self,
            get_context().override_server_args(
                disaggregation_mode="decode",
                disaggregation_decode_enable_radix_cache=True,
                disaggregation_decode_cache_write_only=write_only,
            ),
        )
        tree = _ResidentTree()
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.tree_cache = tree
        queue.scheduler = SimpleNamespace(enable_decode_hicache=True)
        req = SimpleNamespace(
            origin_input_ids=_PROMPT,
            extra_key=None,
            cache_salt=None,
            kv=SimpleNamespace(cache_protected_len=0),
            _compute_max_prefix_len=lambda input_len: max(input_len - 1, 0),
        )
        prefix_match = queue._match_prefix_and_lock(req)
        self.assertEqual(req.lock_receipt, ("receipt", tree.locked))
        return tree, prefix_match.decode_prefix_len

    def test_write_only_admission_matches_only_the_root(self):
        tree, decode_prefix_len = self._admit(write_only=True)

        self.assertEqual(tree.matches, [([], False)])
        self.assertEqual(tree.locked, "root")
        self.assertEqual(decode_prefix_len, 0)
        self.assertEqual(tree.storage_queries, 0)

    def test_read_write_admission_reuses_the_resident_prompt(self):
        tree, decode_prefix_len = self._admit(write_only=False)

        self.assertEqual(tree.matches, [(list(_PROMPT), True)])
        self.assertEqual(tree.locked, "leaf")
        self.assertEqual(decode_prefix_len, len(_PROMPT))
        self.assertEqual(tree.storage_queries, 1)


class TestDecodeCacheWriteOnlyArgs(CustomTestCase):
    def _args(self, **overrides):
        return ServerArgs(
            model_path="dummy",
            disaggregation_mode="decode",
            disaggregation_decode_enable_radix_cache=True,
            disaggregation_decode_cache_write_only=True,
            disaggregation_transfer_backend="nixl",
            enable_hierarchical_cache=True,
            **overrides,
        )

    def test_requires_the_storage_tier(self):
        with self.assertRaisesRegex(
            ValueError, "--disaggregation-decode-cache-write-only requires"
        ):
            handle_pd_disaggregation(self._args())

    def test_keeps_the_decode_radix_tree(self):
        args = self._args(hicache_storage_backend="file")
        handle_pd_disaggregation(args)
        self.assertIs(resolution_result(args, "disable_radix_cache"), False)


if __name__ == "__main__":
    unittest.main()
