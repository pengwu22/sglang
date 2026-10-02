"""Write-only decode cache on hybrid models: decode backs up, prefill restores.

With ``--disaggregation-decode-cache-write-only`` the decode tier commits its
KV and writes it through to the storage backend shared with prefill, but never
reuses a prefix itself. A follow-up turn whose prompt extends past the prior
turn's prompt into its generated tokens is then restored by prefill, after
both tiers flush, beyond anything prefill wrote; its greedy logprobs must match
a colocated server's.
"""

import tempfile
import time
import unittest

import requests
from prometheus_client.parser import text_string_to_metric_families

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    is_in_ci,
    popen_launch_server,
)

register_cuda_ci(est_time=600, stage="extra-b", runner_config="8-gpu-h200")

PAGE_SIZE = 64
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
    "--enable-metrics",
]
WRITE_ONLY_ARGS = [
    "--disaggregation-decode-enable-radix-cache",
    "--disaggregation-decode-cache-write-only",
]
# 509 prompt tokens: prefill's own backup ends at page 448; decode's first turn
# generates past 512, so only decode-authored pages reach 512.
FIRST_TURN = [1] + [100 + i % 1000 for i in range(508)]
PREFILL_COVERAGE = len(FIRST_TURN) // PAGE_SIZE * PAGE_SIZE
DECODE_ENTRY_END = 512


def _has_nixl():
    try:
        import nixl._api  # noqa: F401
    except ImportError:
        return False
    return True


class DecodeCacheWriteOnlyTestMixin:
    model: str
    common_args: list[str]
    transfer_backend_name = "nixl"
    first_turn_new_tokens = 8
    logprob_delta = 0.05

    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory(prefix="sglang-hicache-write-only-")
        cls.addClassCleanup(directory.cleanup)
        cls.extra_prefill_env = cls.extra_decode_env = {
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": directory.name
        }
        cls.extra_prefill_args = cls.common_args + HICACHE_ARGS
        cls.extra_decode_args = cls.common_args + HICACHE_ARGS + WRITE_ONLY_ARGS
        super().setUpClass()
        cls.transfer_backend = [
            "--disaggregation-transfer-backend",
            cls.transfer_backend_name,
        ]

    def _generate(self, ids, max_new_tokens):
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "input_ids": ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
                "return_logprob": True,
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()

    def _counter(self, url, name):
        response = requests.get(url + "/metrics", timeout=30)
        response.raise_for_status()
        return sum(
            sample.value
            for family in text_string_to_metric_families(response.text)
            for sample in family.samples
            if sample.name == name
        )

    def _assert_logprobs_match(self, want, got):
        want = want["meta_info"]["output_token_logprobs"]
        got = got["meta_info"]["output_token_logprobs"]
        self.assertEqual([item[1] for item in want], [item[1] for item in got])
        for reference, actual in zip(want, got, strict=True):
            self.assertAlmostEqual(reference[0], actual[0], delta=self.logprob_delta)

    def test_prefill_restores_decode_generated_kv(self):
        baseline = popen_launch_server(
            self.model,
            self.lb_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--trust-remote-code"] + self.common_args,
        )
        try:
            first = self._generate(FIRST_TURN, self.first_turn_new_tokens)
            second_turn = FIRST_TURN + first["output_ids"] + [2000] * 8
            second = self._generate(second_turn, 8)
        finally:
            kill_process_tree(baseline.pid, wait_timeout=30)

        self.launch_all()
        self._assert_logprobs_match(
            first, self._generate(FIRST_TURN, self.first_turn_new_tokens)
        )
        deadline = time.monotonic() + 60
        while (
            self._counter(self.decode_url, "sglang:backuped_tokens_total")
            < DECODE_ENTRY_END - PREFILL_COVERAGE
        ):
            self.assertLess(time.monotonic(), deadline, "decode did not back up")
            time.sleep(0.2)
        for url in [self.prefill_url, self.decode_url]:
            requests.post(
                url + "/flush_cache?timeout=30", timeout=60
            ).raise_for_status()

        restored = self._generate(second_turn, 8)

        self.assertGreaterEqual(
            restored["meta_info"]["cached_tokens"], DECODE_ENTRY_END
        )
        self._assert_logprobs_match(second, restored)


@unittest.skipUnless(is_in_ci() or _has_nixl(), "NIXL is required.")
class TestDecodeCacheWriteOnlySWA(
    DecodeCacheWriteOnlyTestMixin, PDDisaggregationServerBase
):
    model = DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE
    common_args = [
        "--page-size",
        str(PAGE_SIZE),
        "--attention-backend",
        "triton",
        "--max-total-tokens",
        "65536",
        "--context-length",
        "4096",
    ]


@unittest.skipUnless(is_in_ci() or _has_nixl(), "NIXL is required.")
class TestDecodeCacheWriteOnlyMamba(
    DecodeCacheWriteOnlyTestMixin, PDDisaggregationServerBase
):
    model = "Qwen/Qwen3.5-0.8B"
    common_args = [
        "--skip-tokenizer-init",
        "--random-seed",
        "1",
        "--page-size",
        str(PAGE_SIZE),
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


if __name__ == "__main__":
    unittest.main()
