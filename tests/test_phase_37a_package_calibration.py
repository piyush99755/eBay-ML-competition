"""
Unit tests for Phase 37A: Package Contents Precision Calibration Research.
"""

import json
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "data" / "cache" / "phase_37a_package_calibration_results.json"


class TestPhase37APackageCalibration(unittest.TestCase):
    """Verify Phase 36C reconstruction, candidate metrics, and qualitative trade-off analysis."""

    def test_results_file_existence(self):
        """Verify simulation results JSON exists."""
        self.assertTrue(RESULTS_PATH.exists(), "Results JSON missing!")

    def test_phase_36c_baseline_reconstruction(self):
        """Verify that Phase 36C baseline is reconstructed with 100% mathematical fidelity."""
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        base = data["phase_36c_baseline"]["overall"]
        self.assertEqual(base["tp"], 3045)
        self.assertEqual(base["fp"], 179)
        self.assertEqual(base["fn"], 728)
        self.assertEqual(base["tp"] + base["fn"], 3773)
        self.assertAlmostEqual(base["f_beta"], 0.938333, places=5)

        c1 = data["phase_36c_baseline"]["category_1"]
        c2 = data["phase_36c_baseline"]["category_2"]
        self.assertEqual(c1["tp"], 2028)
        self.assertEqual(c1["fp"], 129)
        self.assertEqual(c1["fn"], 502)
        self.assertEqual(c2["tp"], 1017)
        self.assertEqual(c2["fp"], 50)
        self.assertEqual(c2["fn"], 226)

    def test_candidate_invariants_and_tradeoff_ratios(self):
        """Verify candidate simulation invariants and prove destructive TP/FP trade-off ratio."""
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        candidates = data["candidates"]

        for key in ["th_70", "th_80"]:
            cand = candidates[key]
            self.assertEqual(cand["tp"] + cand["fn"], 3773, f"Invariant violated for {key}")
            self.assertEqual(cand["preds_removed"], cand["tp_lost"] + cand["fp_removed"], f"Removal mismatch for {key}")

            # Critical empirical proof: more TPs are lost than FPs removed
            self.assertGreater(cand["tp_lost"], cand["fp_removed"], f"TPs lost must exceed FPs removed for {key}")

            ratio = cand["error_breakdown_summary"]["ratio_tp_lost_per_fp_removed"]
            self.assertGreater(ratio, 1.5, f"Trade-off ratio for {key} should exceed 1.5 TPs lost per FP removed")

    def test_core_automotive_components_lost(self):
        """Verify that genuine physical components (discs, shoes, cylinders) are among lost TPs."""
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        tps_lost_80 = data["error_breakdowns"]["th_80"]["tps_lost"]
        lost_values = [item["val"] for item in tps_lost_80]

        # Key physical components that are valid ground truth
        critical_components = ["Scheiben", "Backen", "Radzylinder", "Bremsklötze", "Keilriemen"]
        for comp in critical_components:
            self.assertIn(comp, lost_values, f"Core component '{comp}' must be identified in lost TPs")


if __name__ == "__main__":
    unittest.main()
