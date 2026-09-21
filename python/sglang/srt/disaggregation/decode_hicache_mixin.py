"""HiCache integration mixins for the decode side of PD disaggregation"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.managers.schedule_policy import match_prefix_for_req
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestOutcome,
    InitLoadBackParams,
)

if TYPE_CHECKING:
    from sglang.srt.disaggregation.decode import DecodeRequest
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


@dataclass
class DecodePrefixMatch:
    prefix_indices: torch.Tensor
    l2_host_hit_length: int
    l3_storage_hit_length: int
    last_device_node: Any
    last_host_node: Any = None
    prefetch_registered: bool = False

    @property
    def l1_prefix_len(self) -> int:
        return len(self.prefix_indices)

    @property
    def decode_prefix_len(self) -> int:
        return self.l1_prefix_len + self.l2_host_hit_length + self.l3_storage_hit_length

    @property
    def needs_local_restore(self) -> bool:
        return self.decode_prefix_len > self.l1_prefix_len

    @property
    def restore_token_count(self) -> int:
        """Number of tokens that need L2/L3 load_back to device."""
        return self.decode_prefix_len - self.l1_prefix_len


class HiCacheRestoreResult(Enum):
    """Outcome of one tick of the HiCache local-restore state machine."""

    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class DecodeHiCachePreallocMixin:
    """HiCache hooks for ``DecodePreallocQueue``: issue prefetch + reserve tokens."""

    def _build_decode_prefix_match(self, req: Req, result: Any) -> DecodePrefixMatch:
        """Convert a ``match_prefix_for_req`` result into ``DecodePrefixMatch``.

        Performs the optional L3 storage hit length query when decode-side
        HiCache is enabled and the last host node is backed up.
        """
        prefix_indices = result.device_indices
        l1_prefix_len = len(prefix_indices)
        l2_host_hit_length = result.host_hit_length

        l3_storage_hit_length = 0
        last_host_node = None
        # The hit query is KV-only, which matches what decode fetches
        # (kv_only prefetch below): component state comes from the transfer.
        if self.scheduler.enable_decode_hicache:
            last_host_node = result.last_host_node
            if self.tree_cache.is_backuped(last_host_node) or self.tree_cache.is_root(
                last_host_node
            ):
                matched_len = l1_prefix_len + l2_host_hit_length
                suffix_tokens = req.origin_input_ids[matched_len:]
                last_hash = self.tree_cache.get_last_hash_value(last_host_node)
                prefix_keys = (
                    self.tree_cache.get_prefix_hash_values(last_host_node)
                    if self.tree_cache.hicache_storage_pass_prefix_keys
                    else None
                )
                l3_storage_hit_length = self.tree_cache.query_storage_hit_length(
                    result.last_host_node,
                    suffix_tokens,
                    last_hash,
                    prefix_keys,
                    extra_key=req.extra_key,
                    cache_salt=req.cache_salt,
                )

        # Admission bounds the tree match at the sliding-window start.
        # Keep the L3 query within that same bound, page aligned.
        if (
            l2_host_hit_length + l3_storage_hit_length > 0
            and self._uses_swa_tail_prealloc()
        ):
            fill_len = self._pre_alloc_fill_len(req)
            swa_prefix_cap = max(0, fill_len - self._swa_tail_len(fill_len))
            if l1_prefix_len + l2_host_hit_length > swa_prefix_cap:
                l2_host_hit_length = 0
                l3_storage_hit_length = 0
            else:
                page_size = self.token_to_kv_pool_allocator.page_size
                allowed = swa_prefix_cap - l1_prefix_len - l2_host_hit_length
                l3_storage_hit_length = min(l3_storage_hit_length, allowed)
                l3_storage_hit_length -= l3_storage_hit_length % page_size

        return DecodePrefixMatch(
            prefix_indices=prefix_indices,
            l2_host_hit_length=l2_host_hit_length,
            l3_storage_hit_length=l3_storage_hit_length,
            last_device_node=result.last_device_node,
            last_host_node=(
                result.last_host_node if l3_storage_hit_length > 0 else None
            ),
        )

    def _start_hicache_prefetch(
        self, req: Req, prefix_match: Optional[DecodePrefixMatch]
    ) -> None:
        """Issue L3 storage prefetch after admission succeeds.

        On failure, degrades to L2-only restore by clearing l3 fields.
        """
        if (
            prefix_match is None
            or prefix_match.l3_storage_hit_length <= 0
            or prefix_match.last_host_node is None
        ):
            return
        matched_len = prefix_match.l1_prefix_len + prefix_match.l2_host_hit_length
        suffix = req.origin_input_ids[
            matched_len : matched_len + prefix_match.l3_storage_hit_length
        ]
        try:
            last_hash = self.tree_cache.get_last_hash_value(prefix_match.last_host_node)
            prefix_keys = (
                self.tree_cache.get_prefix_hash_values(prefix_match.last_host_node)
                if self.tree_cache.hicache_storage_pass_prefix_keys
                else None
            )
        except KeyError as e:
            logger.warning(
                "HiCache L3 prefetch failed for rid=%s: %s; falling back to L2-only LoadingBack",
                req.rid,
                e,
            )
            prefix_match.l3_storage_hit_length = 0
            prefix_match.prefetch_registered = False
            return
        self.tree_cache.prefetch_from_storage(
            req.cache_request_handle,
            prefix_match.last_host_node,
            suffix,
            last_hash,
            prefix_keys,
            extra_key=req.extra_key,
            cache_salt=req.cache_salt,
            # Base KV only, like the load-back: SWA / Mamba state comes
            # from the transfer, and hybrid component fetches are
            # all-or-nothing, which the KV-only hit query cannot promise.
            kv_only=True,
        )
        prefix_match.prefetch_registered = self.tree_cache.has_ongoing_prefetch(
            req.cache_request_handle
        )
        if not prefix_match.prefetch_registered:
            # A silently declined prefetch leaves the promised L3 range
            # unrestorable; degrade to L2-only.
            logger.warning(
                "HiCache L3 prefetch declined for rid=%s (len=%s); "
                "falling back to L2-only LoadingBack",
                req.rid,
                prefix_match.l3_storage_hit_length,
            )
            prefix_match.l3_storage_hit_length = 0

    def _hicache_pending_restore_tokens(self) -> int:
        """Total device tokens reserved for pending HiCache L2/L3 load_back."""
        if not self.scheduler.enable_decode_hicache:
            return 0
        return sum(
            dr.prefix_match.restore_token_count
            for dr in self.transfer_queue.queue
            if dr.prefix_match is not None
            and dr.hicache_restore_status == HiCacheRestoreResult.PENDING
            and dr.hicache_restored_node is None
        )


class HiCacheRestoreGatedKVReceiver:
    """Wraps a kv_receiver so KVPoll.Success is gated on HiCache restore READY."""

    def __init__(self, decode_req: DecodeRequest):
        self.decode_req = decode_req

    def poll(self) -> KVPoll:
        poll = self.decode_req.kv_receiver.poll()
        if (
            poll == KVPoll.Success
            and self.decode_req.hicache_restore_status == HiCacheRestoreResult.PENDING
        ):
            return KVPoll.Transferring
        return poll


class DecodeHiCacheTransferMixin:
    """HiCache hooks for ``DecodeTransferQueue``: drive restore state machine."""

    def _clean_hicache_prefetch_resources(self, decode_req: DecodeRequest) -> None:
        if (
            decode_req.prefix_match is not None
            and decode_req.prefix_match.prefetch_registered
        ):
            self.tree_cache.finish(
                decode_req.req.cache_request_handle, CacheRequestOutcome.ABORT
            )
        if decode_req.hicache_restored_node is not None:
            self.tree_cache.dec_lock_ref(
                decode_req.hicache_restored_node,
                decode_req.hicache_restore_lock_receipt,
            )
            decode_req.hicache_restored_node = None
            decode_req.hicache_restore_lock_receipt = None

    def _try_hicache_queue_load_back(self, dr: DecodeRequest) -> bool:
        """Prepare one restore; True when its indices need a completion fence.

        On success, ``dr.hicache_restored_node`` and ``hicache_restored_kv_indices``
        are populated, and an inc_lock_ref is held until commit/abort.
        Resident indices may belong to another request's unfinished DMA.
        The caller binds both resident and newly loaded indices to an event.
        Missing coverage flips to FAILED.
        """
        pm = dr.prefix_match

        # Wait for L3 -> L2 prefetch to drain (skip when no L3 hit).
        if pm.l3_storage_hit_length > 0:
            if not self.tree_cache.check_prefetch_progress(dr.req.cache_request_handle):
                return False
            self.tree_cache.pop_prefetch_loaded_tokens(dr.req.cache_request_handle)

        # Admission and restore use the same FULL-only match, bounded by the
        # promise. Component state belongs to the prefill transfer.
        rematch = match_prefix_for_req(
            self.tree_cache,
            dr.req,
            dr.req.origin_input_ids[: pm.decode_prefix_len],
            include_req=True,
            kv_only=True,
        )
        # A partial L3 fetch cannot fulfill the promise. Do not enqueue a DMA
        # that this failed request would leave unstarted and locked in the tree.
        if (
            rematch.host_hit_length > 0
            and len(rematch.device_indices) + rematch.host_hit_length
            >= pm.decode_prefix_len
        ):
            new_indices, restored_node = self.tree_cache.init_load_back(
                InitLoadBackParams(
                    best_match_node=rematch.best_match_node,
                    host_hit_length=rematch.host_hit_length,
                    req=dr.req,
                    kv_only=True,
                )
            )
        else:
            new_indices = rematch.device_indices[:0]
            restored_node = rematch.last_device_node
        # The rematch repointed req.last_node to feed init_load_back's device
        # boundary, but the prealloc lock and the receipt on the req still
        # belong to pm.last_device_node; restore the pairing so any release
        # before the commit hands over the restored lock hits the right node
        # (the receipt's anchor makes a mispaired release assert).
        dr.req.last_node = pm.last_device_node
        # Failback: total coverage < required prefix means device alloc likely failed.
        if len(rematch.device_indices) + len(new_indices) < pm.decode_prefix_len:
            logger.warning(
                "HiCache load_back failed for rid=%s: device_indices=%d, "
                "new_indices=%d, expected decode_prefix_len=%d (l1=%d, l2=%d, l3=%d)",
                dr.req.rid,
                len(rematch.device_indices),
                len(new_indices),
                pm.decode_prefix_len,
                pm.l1_prefix_len,
                pm.l2_host_hit_length,
                pm.l3_storage_hit_length,
            )
            dr.hicache_restore_status = HiCacheRestoreResult.FAILED
            return False

        # The commit covers exactly [l1, decode_prefix_len).
        dr.hicache_restored_kv_indices = torch.cat(
            [rematch.device_indices[pm.l1_prefix_len :], new_indices]
        )[: pm.decode_prefix_len - pm.l1_prefix_len]
        dr.hicache_restored_node = restored_node
        dr.hicache_restore_lock_receipt = self.tree_cache.inc_lock_ref(
            restored_node
        ).to_dec_params()

        return True

    def _process_hicache_local_restores(self, decode_reqs: List[DecodeRequest]) -> None:
        if not hasattr(self.tree_cache, "is_load_back_event_done"):
            return

        counter = self.tree_cache.cache_controller.layer_done_counter
        # Even an L1 hit can refer to indices published by an unfinished DMA.
        # All H->D operations share a stream, so the latest producer fences
        # every earlier load, including loads anchored on an ancestor or a node
        # that has since split. Node-id membership cannot establish readiness.
        active: List[DecodeRequest] = []
        for dr in decode_reqs:
            if dr.hicache_restore_status != HiCacheRestoreResult.PENDING:
                continue
            pm = dr.prefix_match
            if pm is None or not pm.needs_local_restore:
                if dr.hicache_load_consumer_index < 0:
                    dr.hicache_load_consumer_index = counter.producer_index
                if (
                    pm is None
                    or pm.l1_prefix_len == 0
                    or self.tree_cache.is_load_back_event_done(
                        dr.hicache_load_consumer_index
                    )
                ):
                    dr.hicache_restore_status = HiCacheRestoreResult.READY
                continue
            active.append(dr)

        # Phase A: advance in-flight DMAs to READY.
        for dr in active:
            if (
                dr.hicache_restored_node is not None
                and self.tree_cache.is_load_back_event_done(
                    dr.hicache_load_consumer_index
                )
            ):
                dr.hicache_restore_status = HiCacheRestoreResult.READY

        # Phase B: queue new load_back ops if the next slot is free.
        # The (producer_index + 1) check ensures we never overwrite a still-in-flight slot:
        # if a previous req holds that slot and isn't done, its event won't be signaled.
        if not self.tree_cache.is_load_back_event_done(
            (counter.producer_index + 1) % counter.num_counters
        ):
            return
        queued = [
            dr
            for dr in active
            if dr.hicache_restored_node is None
            and self._try_hicache_queue_load_back(dr)
        ]
        if not queued:
            return

        # Phase C: kick off merged DMA, bind consumer_index for Phase A polling next tick.
        consumer_index = self.tree_cache.ready_to_load_host_cache()
        if consumer_index < 0:
            consumer_index = counter.producer_index
        for dr in queued:
            dr.hicache_load_consumer_index = consumer_index
        if self.tree_cache.is_load_back_event_done(consumer_index):
            for dr in queued:
                dr.hicache_restore_status = HiCacheRestoreResult.READY

    def _commit_hicache_local_restore_to_req(self, decode_req: DecodeRequest) -> None:
        prefix_match = decode_req.prefix_match
        if prefix_match is None or not prefix_match.needs_local_restore:
            return

        req = decode_req.req
        restored_node = decode_req.hicache_restored_node
        restored_lock_receipt = decode_req.hicache_restore_lock_receipt
        assert restored_node is not None
        assert restored_lock_receipt is not None
        # Release preallocation before installing the restored lock receipt.
        self.tree_cache.dec_lock_ref(
            prefix_match.last_device_node,
            req.lock_receipt,
            skip_swa=req.swa_prefix_lock_released,
        )

        self.tree_cache.req_to_token_pool.write(
            (
                decode_req.req.kv.req_pool_idx,
                slice(prefix_match.l1_prefix_len, prefix_match.decode_prefix_len),
            ),
            decode_req.hicache_restored_kv_indices,
        )
        req.prefix_indices = torch.cat(
            [prefix_match.prefix_indices, decode_req.hicache_restored_kv_indices]
        )
        req.last_node = restored_node
        req.lock_receipt = restored_lock_receipt
        req.swa_prefix_lock_released = False
        # Prevent abort cleanup from releasing the transferred lock.
        decode_req.hicache_restored_node = None
        decode_req.hicache_restore_lock_receipt = None
