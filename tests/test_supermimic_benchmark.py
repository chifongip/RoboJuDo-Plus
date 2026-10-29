import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import test_supermimic

_spec = importlib.util.spec_from_file_location(
    "benchmark_supermimic", Path(__file__).resolve().parents[1] / "scripts/benchmark_supermimic.py"
)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


class TestSuperMimicBenchmark(unittest.TestCase):
    def test_statistics_and_error_gate(self):
        result = bench.summarize([1, 2, 3, 21])
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["over_20ms_fraction"], 0.25)
        self.assertEqual(result["p50_ms"], 2.5)
        self.assertTrue(bench.error_metrics([1, 2], [1, 2])["passed"])
        self.assertFalse(bench.error_metrics([np.nan], [1])["passed"])
        self.assertFalse(bench.error_metrics([1.01], [1])["passed"])

    def test_postprocessing_clips_before_scaling_and_adds_default(self):
        policy = SimpleNamespace(action_clip=0.5, action_scales=np.full(23, 2.0), default_pos=np.ones(23))
        np.testing.assert_array_equal(bench.postprocess(np.full(23, 2), policy), np.full(23, 2))
        with self.assertRaises(ValueError):
            bench.postprocess(np.full(23, np.inf), policy)

    def test_tape_and_cached_decoder_match_full_actor_including_resets_and_tail(self):
        fixture = test_supermimic.TestSuperMimic()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        motion_cfg, _ = fixture.make_checkpoint(num_joints=23)
        velocity_cfg, _ = fixture.make_checkpoint(num_joints=23, velocity=True)
        configs = {"motion": motion_cfg, "velocity": velocity_cfg}
        with tempfile.TemporaryDirectory() as root, patch.object(bench, "config", side_effect=configs.__getitem__):
            root = Path(root)
            for task in configs:
                directory = root / task
                directory.mkdir()
                bench.make_tape(task, directory, seed=7)
                with np.load(directory / "tape.npz") as source:
                    tape = {k: source[k] for k in source.files}
                self.assertEqual(tape["input"].shape[1], 3310 if task == "motion" else 753)
                self.assertEqual(np.count_nonzero(tape["reset"]), 8)
                self.assertEqual(int(tape["frame"].max()), 11)
                inputs = json.loads((directory / "inputs.json").read_text())
                self.assertEqual(inputs["tape_sha256"], bench.digest(directory / "tape.npz"))
                for route in ("A", "F") if task == "motion" else ("A",):
                    backend = bench.Backend(route, bench.load_policy(task), directory, tape)
                    for i in range(len(tape["input"])):
                        env = backend.strategy_input(tape, i)
                        action = backend.run(backend.observe(env, tape, i))
                        self.assertTrue(bench.error_metrics(action, tape["action"][i])["passed"])
                        np.testing.assert_allclose(
                            bench.postprocess(action, backend.policy), tape["target"][i], atol=1e-4, rtol=1e-4
                        )
                    if route == "F":
                        self.assertEqual(backend.cache.shape, (12, 32))
                        self.assertEqual(backend.setup["cache_bytes"], 12 * 32 * 4)

    def test_report_requires_three_correct_repeats_and_no_extra_deadline_misses(self):
        def result(route, repeat, p99, missed=0):
            return {
                "task": "motion",
                "route": route,
                "repeat": repeat,
                "label": bench.LABELS[route],
                "status": "passed",
                "measurements": {"paced/strategy": {"p50_ms": p99 / 2, "p99_ms": p99, "over_20ms_fraction": missed}},
            }

        results = [result(route, i, latency) for i in range(3) for route, latency in (("A", 10), ("F", 8))]
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            bench.report(root, results)
            self.assertIn("all three repeats: F", (root / "report.md").read_text())
            results[-1]["measurements"]["paced/strategy"]["over_20ms_fraction"] = 0.1
            bench.report(root, results)
            self.assertNotIn("all three repeats: F", (root / "report.md").read_text())
            results[-1]["status"] = "failed"
            bench.report(root, results)
            self.assertNotIn("all three repeats: F", (root / "report.md").read_text())
