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
    DecLockRefParams,
    InitLoadBackParams,
)

if TYPE_CHECKING:
    from sglang.srt.disaggregation.decode import DecodeRequest
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


@dataclass
class DecodePrefixMatch:
    """The prefix decode promises prefill it already holds: ``[0, l1)`` on
    device and ``[l1, l1 + l2)`` on host, each pinned until the restore owns
    it. L3 hits are staged into the host tier before they are promised."""

    prefix_indices: torch.Tensor
    l2_host_hit_length: int
    last_device_node: Any
    host_anchor: Any = None
    host_lock: Optional[DecLockRefParams] = None

    @property
    def l1_prefix_len(self) -> int:
        return len(self.prefix_indices)

    @property
    def decode_prefix_len(self) -> int:
        return self.l1_prefix_len + self.l2_host_hit_length

    @property
    def needs_local_restore(self) -> bool:
        return self.decode_prefix_len > self.l1_prefix_len

    @property
    def restore_token_count(self) -> int:
        """Number of tokens that need L2 load_back to device."""
        return self.decode_prefix_len - self.l1_prefix_len


def release_host_promise(tree_cache, prefix_match: Optional[DecodePrefixMatch]) -> None:
    """Unpin the host part of a promise; the restore pins its own pages."""
    if prefix_match is not None and prefix_match.host_lock is not None:
        tree_cache.dec_host_lock_ref(prefix_match.host_anchor, prefix_match.host_lock)
        prefix_match.host_lock = None


class HiCacheRestoreResult(Enum):
    """Outcome of one tick of the HiCache local-restore state machine."""

    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class DecodeHiCachePreallocMixin:
    """HiCache hooks for ``DecodePreallocQueue``: stage L3 hits, pin promises,
    reserve restore tokens."""

    def _build_decode_prefix_match(self, result: Any) -> DecodePrefixMatch:
        """Turn a ``match_prefix_for_req`` result into the promise, pinning its
        host part: host eviction would otherwise leave it unrestorable once
        prefill has trimmed its transfer to ``decode_prefix_len``."""
        l2_host_hit_length = result.host_hit_length
        host_anchor = host_lock = None
        if l2_host_hit_length > 0:
            host_anchor = result.best_match_node
            host_lock = self.tree_cache.inc_host_lock_ref(host_anchor).to_dec_params()
        return DecodePrefixMatch(
            prefix_indices=result.device_indices,
            l2_host_hit_length=l2_host_hit_length,
            last_device_node=result.last_device_node,
            host_anchor=host_anchor,
            host_lock=host_lock,
        )

    def _stage_from_storage(self, req: Req, result: Any, reusable_len: int) -> bool:
        """Fetch the L3 part of the reusable prefix into the host tier; True
        when a fetch is in flight.

        Promising an L3 hit directly would break the request whenever the
        fetch comes back short (host capacity, storage errors) after prefill
        has trimmed its transfer. Staged first, a short or declined fetch only
        shrinks what is promised once the request is matched again.
        """
        anchor = result.last_host_node
        if not (self.tree_cache.is_backuped(anchor) or self.tree_cache.is_root(anchor)):
            return False
        matched_len = len(result.device_indices) + result.host_hit_length
        suffix = req.origin_input_ids[matched_len:reusable_len]
        if len(suffix) == 0:
            return False
        try:
            last_hash = self.tree_cache.get_last_hash_value(anchor)
            prefix_keys = (
                self.tree_cache.get_prefix_hash_values(anchor)
                if self.tree_cache.hicache_storage_pass_prefix_keys
                else None
            )
            hit = self.tree_cache.query_storage_hit_length(
                anchor,
                suffix,
                last_hash,
                prefix_keys,
                extra_key=req.extra_key,
                cache_salt=req.cache_salt,
            )
            if hit <= 0:
                return False
            self.tree_cache.prefetch_from_storage(
                req.cache_request_handle,
                anchor,
                suffix[:hit],
                last_hash,
                prefix_keys,
                extra_key=req.extra_key,
                cache_salt=req.cache_salt,
                # Base KV only: the transfer brings the SWA window and the
                # Mamba state, and the hit query counts base KV only.
                kv_only=True,
            )
        except Exception as e:
            logger.warning(
                "HiCache L3 prefetch failed for rid=%s: %s; promising L1/L2 only",
                req.rid,
                e,
            )
            return False
        return self.tree_cache.has_ongoing_prefetch(req.cache_request_handle)

    def _poll_hicache_staging(self, decode_req: DecodeRequest) -> bool:
        """True once the request's L3 fetch, if any, has resolved; its pages
        are then host-resident and the next match promises them as L2."""
        if not decode_req.hicache_staging:
            return True
        handle = decode_req.req.cache_request_handle
        if not self.tree_cache.check_prefetch_progress(handle):
            return False
        self.tree_cache.pop_prefetch_loaded_tokens(handle)
        decode_req.hicache_staging = False
        return True

    def _abort_hicache_staging(self, decode_req: DecodeRequest) -> None:
        if decode_req.hicache_staging:
            self.tree_cache.finish(
                decode_req.req.cache_request_handle, CacheRequestOutcome.ABORT
            )
            decode_req.hicache_staging = False

    def _hicache_pending_restore_tokens(self) -> int:
        """Total device tokens reserved for pending HiCache L2 load_back."""
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

    def _clean_hicache_restore_resources(self, decode_req: DecodeRequest) -> None:
        release_host_promise(self.tree_cache, decode_req.prefix_match)
        if decode_req.hicache_restored_node is not None:
            self.tree_cache.dec_lock_ref(
                decode_req.hicache_restored_node,
                decode_req.hicache_restore_lock_receipt,
            )
            decode_req.hicache_restored_node = None
            decode_req.hicache_restore_lock_receipt = None

    def _try_hicache_queue_load_back(self, dr: DecodeRequest) -> bool:
        """Queue one L2->L1 load_back op for ``dr``; True iff a DMA was queued.

        On success, ``dr.hicache_restored_node`` and ``hicache_restored_kv_indices``
        are populated, and an inc_lock_ref is held until commit/abort.
        Trivial cases (all-on-device / no needed coverage) auto-flip to READY.
        Failback paths flip to FAILED.
        """
        pm = dr.prefix_match

        # Re-match the promised range the way admission matched it: FULL KV
        # only. Restore it KV-only too: the SWA window and the Mamba state
        # come from the prefill transfer into the slots registered at
        # prealloc, which a restored checkpoint would race and overwrite.
        # req.last_node / prefix_indices now reflect the current device state.
        rematch = match_prefix_for_req(
            self.tree_cache,
            dr.req,
            dr.req.origin_input_ids[: pm.decode_prefix_len],
            include_req=True,
            kv_only=True,
        )
        new_indices, restored_node = self.tree_cache.init_load_back(
            InitLoadBackParams(
                best_match_node=rematch.best_match_node,
                host_hit_length=rematch.host_hit_length,
                req=dr.req,
                kv_only=True,
            )
        )
        # The load-back pins the host pages it reads; the promise's pin is done.
        release_host_promise(self.tree_cache, pm)
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
                "new_indices=%d, expected decode_prefix_len=%d (l1=%d, l2=%d)",
                dr.req.rid,
                len(rematch.device_indices),
                len(new_indices),
                pm.decode_prefix_len,
                pm.l1_prefix_len,
                pm.l2_host_hit_length,
            )
            dr.hicache_restore_status = HiCacheRestoreResult.FAILED
            return False

        dr.hicache_restored_kv_indices = torch.cat(
            [rematch.device_indices[pm.l1_prefix_len :], new_indices]
        )[: pm.restore_token_count]
        dr.hicache_restored_node = restored_node
        dr.hicache_restore_lock_receipt = self.tree_cache.inc_lock_ref(
            restored_node
        ).to_dec_params()

        if len(new_indices) == 0:
            # Whole prefix already on device; no DMA needed.
            dr.hicache_restore_status = HiCacheRestoreResult.READY
            return False
        return True

    def _process_hicache_local_restores(self, decode_reqs: List[DecodeRequest]) -> None:
        # Filter once: keep only PENDING reqs that still need restore work;
        # trivially-done reqs (no prefix_match / nothing to restore) flip to READY.
        active: List[DecodeRequest] = []
        for dr in decode_reqs:
            if dr.hicache_restore_status != HiCacheRestoreResult.PENDING:
                continue
            pm = dr.prefix_match
            if pm is None or not pm.needs_local_restore:
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

        # Phase B: queue new load_back ops if the next slot is free on every rank.
        if not self.tree_cache.has_free_load_back_slot():
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
            for dr in queued:
                dr.hicache_restore_status = HiCacheRestoreResult.READY
            return
        for dr in queued:
            dr.hicache_load_consumer_index = consumer_index

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
