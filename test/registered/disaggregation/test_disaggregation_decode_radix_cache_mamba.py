"""FULL+Mamba decode L3 restore parity at a non-checkpoint boundary."""

import tempfile
import time
import unittest

import requests
from prometheus_client.parser import text_string_to_metric_families
from test_disaggregation_decode_radix_cache import _has_mooncake

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    is_in_ci,
    popen_launch_server,
)

register_cuda_ci(est_time=240, stage="extra-b", runner_config="2-gpu-large")

COMMON_ARGS = [
    "--skip-tokenizer-init",
    "--random-seed",
    "1",
    "--page-size",
    "64",
    "--attention-backend",
    "triton",
    "--linear-attn-backend",
    "triton",
    "--mamba-track-interval",
    "256",
    "--max-mamba-cache-size",
    "32",
    "--max-running-requests",
    "8",
    "--max-total-tokens",
    "4096",
    "--context-length",
    "4096",
    "--cuda-graph-backend-decode",
    "disabled",
    "--cuda-graph-backend-prefill",
    "disabled",
]
HICACHE_ARGS = [
    "--enable-hierarchical-cache",
    "--hicache-ratio",
    "2",
    "--hicache-write-policy",
    "write_through",
    "--hicache-storage-backend",
    "file",
    "--hicache-storage-prefetch-policy",
    "wait_complete",
    "--hicache-io-backend",
    "kernel",
    "--hicache-mem-layout",
    "page_first",
    "--enable-metrics",
]


@unittest.skipUnless(is_in_ci() or _has_mooncake(), "Mooncake is required.")
class TestDisaggregationDecodeRadixHiCacheMamba(PDDisaggregationServerBase):
    model = "Qwen/Qwen3.5-0.8B"
    extra_prefill_args = COMMON_ARGS + HICACHE_ARGS
    extra_decode_args = (
        COMMON_ARGS + HICACHE_ARGS + ["--disaggregation-decode-enable-radix-cache"]
    )

    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory(prefix="sglang-hicache-mamba-")
        cls.addClassCleanup(directory.cleanup)
        cls.extra_prefill_env = cls.extra_decode_env = {
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": directory.name
        }
        super().setUpClass()
        cls.transfer_backend = ["--disaggregation-transfer-backend", "mooncake"]

    def _generate(self, ids):
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "input_ids": ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 8,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "top_logprobs_num": 5,
            },
            timeout=120,
        )
        response.raise_for_status()
        logprobs = response.json()["meta_info"]["output_token_logprobs"]
        self.assertEqual(len(logprobs), 8)
        return logprobs

    def _decode_counter(self, name, **labels):
        response = requests.get(self.decode_url + "/metrics", timeout=30)
        response.raise_for_status()
        return sum(
            sample.value
            for family in text_string_to_metric_families(response.text)
            for sample in family.samples
            if sample.name == name
            and all(sample.labels.get(key) == value for key, value in labels.items())
        )

    def test_partial_l3_prefix_matches_colocated_reference(self):
        # Decode must cross 512 to create its own checkpoint and L3 backup.
        full = [1] + [100 + i % 1000 for i in range(508)]
        # Prefill may also save its last page-aligned state at 448. Use 384
        # to end before both that state and the decode checkpoint at 512.
        partial = full[:384] + [2000]
        baseline = popen_launch_server(
            self.model,
            self.lb_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--trust-remote-code"] + COMMON_ARGS,
        )
        try:
            expected = [self._generate(ids) for ids in [full, partial]]
        finally:
            kill_process_tree(baseline.pid, wait_timeout=30)

        self.launch_all()
        first = self._generate(full)
        deadline = time.monotonic() + 30
        while self._decode_counter("sglang:backuped_tokens_total") < 512:
            self.assertLess(time.monotonic(), deadline, "decode did not back up to L3")
            time.sleep(0.1)
        before = self._decode_counter("sglang:load_back_tokens_total", pool="kv")
        for url in [self.prefill_url, self.decode_url]:
            response = requests.post(url + "/flush_cache?timeout=30", timeout=60)
            response.raise_for_status()
        restored = self._generate(partial)

        for want, got in zip(expected, [first, restored], strict=True):
            self.assertEqual([item[1] for item in want], [item[1] for item in got])
            for reference, actual in zip(want, got, strict=True):
                self.assertAlmostEqual(reference[0], actual[0], delta=0.05)
        self.assertGreater(
            self._decode_counter("sglang:load_back_tokens_total", pool="kv"), before
        )


if __name__ == "__main__":
    unittest.main()
