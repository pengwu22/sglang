"""KV-only match and load-back spec: FULL KV independent of component state.

The shared unified radix cache suite covers real SWA / Mamba components on
both TreeCore backends; this CPU test pins the contract with a component
whose state is always tombstoned.
"""

import unittest
from array import array

import torch

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.base import (
    BASE_COMPONENT_TYPE,
    EvictLayer,
    TreeComponent,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE_SIZE = 2


class _TombstonedSwaComponent(TreeComponent):
    """Every node's SWA state is tombstoned: the all-component match sees nothing.

    On the P/D decode tier this is the shared-prefix shape: the SWA tail is
    transferred fresh per request and belongs to each request's own end.
    """

    component_type = ComponentType.SWA

    def create_match_validator(self, match_device_only: bool = False):
        return lambda node: False

    def build_hicache_transfers(self, node, phase, **kwargs):
        # Like the real SWA component: a tombstoned node has nothing to build.
        raise AssertionError("tombstoned SWA state has no transfer")

    def redistribute_on_node_split(self, new_parent, child):
        return None

    def evict_component(
        self, node, device_frees, host_frees, target: EvictLayer = EvictLayer.DEVICE
    ) -> tuple[int, int]:
        return 0, 0

    def acquire_component_lock(self, node, result):
        return result

    def release_component_lock(self, node, params):
        return None

    def _evict_device_start(self, request_cnt) -> None:
        pass

    def _evict_device_next_node(self, tracker, device_frees, host_frees):
        return None

    def _evict_device_end(self) -> None:
        pass

    def _dec_session_coverage(self, session_id, leaf) -> None:
        pass

    def _advance_session_coverage(self, session_id, leaf, old_ancestor) -> None:
        pass

    def _recede_session_coverage(self, session_id, leaf, fallback) -> None:
        pass


def _cache() -> UnifiedRadixCache:
    return UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=PAGE_SIZE,
            tree_components=(ComponentType.FULL, ComponentType.SWA),
            component_registry_override={ComponentType.SWA: _TombstonedSwaComponent},
        )
    )


def _key(tokens) -> RadixKey:
    return RadixKey(array("q", tokens))


def _add_node(tree_core, parent, tokens, *, device: bool, host: bool):
    node = tree_core._new_node()
    node.parent = parent
    node.key = _key(tokens)
    node.hash_value = []
    cd = node.component_data[BASE_COMPONENT_TYPE]
    base = 100 * node.id
    if device:
        cd.value = torch.arange(base, base + len(tokens))
    if host:
        cd.host_value = torch.arange(base, base + len(tokens))
    parent.children[node.key.child_key(PAGE_SIZE)] = node
    return node


class TestMatchKvOnly(CustomTestCase):
    def setUp(self):
        self.cache = _cache()
        self.tree_core = self.cache.tree_core
        root = self.tree_core.root_node
        self.a = _add_node(self.tree_core, root, [1, 2, 3, 4], device=True, host=True)
        self.b = _add_node(
            self.tree_core, self.a, [5, 6, 7, 8], device=False, host=True
        )

    def _match(self, tokens, *, kv_only: bool):
        return self.cache.match_prefix(
            MatchPrefixParams(key=_key(tokens), kv_only=kv_only)
        )

    def test_finds_full_kv_behind_tombstoned_component_state(self):
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        self.tree_core.set_hicache_enabled()

        everything = self._match(tokens, kv_only=False)
        self.assertEqual(len(everything.device_indices), 0)
        self.assertEqual(everything.host_hit_length, 0)

        result = self._match(tokens, kv_only=True)

        self.assertEqual(
            result.device_indices.tolist(),
            self.a.component_data[BASE_COMPONENT_TYPE].value.tolist(),
        )
        self.assertEqual(result.last_device_node, self.a.id)
        self.assertEqual(result.host_hit_length, 4)
        self.assertEqual(result.best_match_node, self.b.id)

    def test_dead_node_ends_the_match(self):
        _add_node(self.tree_core, self.b, [9, 10], device=False, host=False)
        self.tree_core.set_hicache_enabled()

        result = self._match([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], kv_only=True)

        self.assertEqual(result.best_match_node, self.b.id)
        self.assertEqual(result.host_hit_length, 4)

    def test_kv_only_load_back_spec_builds_no_component_transfers(self):
        # Building a component transfer for a node whose state is tombstoned
        # asserts, so the KV-only restore must not ask for one.
        with self.assertRaises(AssertionError):
            self.tree_core.build_load_back_spec(self.b.id)

        kv_xfer, comp_xfers = self.tree_core.build_load_back_spec(
            self.b.id, kv_only=True
        )

        self.assertEqual(comp_xfers, {})
        self.assertEqual(
            kv_xfer.host_indices.tolist(),
            self.b.component_data[BASE_COMPONENT_TYPE].host_value.tolist(),
        )


if __name__ == "__main__":
    unittest.main()
