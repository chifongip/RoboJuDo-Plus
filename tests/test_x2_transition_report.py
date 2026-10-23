"""Check generated report descriptions against recorded run settings."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts/report_x2_transition_methods.py"


class TransitionReportTests(unittest.TestCase):
    def generate(self, profiles, warmups, *, include_run_metadata=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "trace.csv").write_text("elapsed_s,arm_rms_error_rad,elbow_correction_nm\n55,0.1,0.2\n56,0.1,0.3\n")
            report = {
                "carry_posture_rad": {"elbow": 0.0},
                "duration_s": 105,
                "trials": [
                    {
                        "condition": "pd_only",
                        "profile": profile,
                        "repetition": repetition,
                        "warmup_s": warmup,
                        "status": "completed",
                        "trace": "trace.csv",
                        "phases": {"walk": {"peak_joint_error_rad": 0.1}},
                        "max_base_tilt_rad": 0.01,
                        "saturated_physics_fraction": 0.0,
                        "max_effort_slew_nm_s": 0.0,
                        "max_effort_bound_excess_nm": 0.0,
                    }
                    for profile in profiles
                    for repetition, warmup in enumerate(warmups, 1)
                ],
            }
            if include_run_metadata:
                report.update(profiles=profiles, conditions=["pd_only"], warmup_seconds=warmups)
            (root / "summary.json").write_text(json.dumps(report))
            subprocess.run([sys.executable, str(SCRIPT), directory], check=True, capture_output=True, text=True)
            return (root / "methods_comparison.md").read_text()

    def test_single_profile_and_default_warmup_are_described_accurately(self):
        text = self.generate(["constant"], [5.0])
        self.assertIn("Methods: PD only.", text)
        self.assertIn("Profiles: constant. Warmup settings (seconds): 5.", text)
        self.assertNotIn("4/5/6", text)
        self.assertNotIn("Profiles: unloaded", text)
        self.assertNotIn("three paired", text)
        self.assertNotIn("16 Nm/s release", text)

    def test_trial_metadata_describes_sway_and_multiple_warmups(self):
        text = self.generate(["unloaded", "sway"], [4.0, 5.0, 6.0], include_run_metadata=False)
        self.assertIn("Profiles: sway, unloaded. Warmup settings (seconds): 4/5/6.", text)
        self.assertNotIn("There is no swaying-load", text)
        self.assertIn("Completed **6/6**", text)


if __name__ == "__main__":
    unittest.main()
