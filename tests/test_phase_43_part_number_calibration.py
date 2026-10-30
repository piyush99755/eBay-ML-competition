"""
Unit tests for Phase 43: Category-Specific Part-Number Calibration.

Tests cover:
1. Champion file integrity and SHA-256 verification.
2. Exact local baseline reproduction (Phase 40).
3. Phase 42B / Phase 41 empirical claim verification (6 FPs, 3 FPs in [0.60, 0.80), 9 TPs in [0.60, 0.80)).
4. Category-specific threshold decoding.
5. Absolute preservation of non-target aspect predictions.
6. TP/FP/FN accounting and metric deltas for Candidate A and B.
7. Hierarchical metric calculation.
8. Decision rules and bootstrap interval bounds.
"""

import json
from pathlib import Path
import unittest

from src.evaluation.phase_43_part_number_calibration import (
    CHAMPION_TSV_PATH,
    EXPECTED_CHAMPION_SHA256,
    OUTPUT_RESULTS_PATH,
    compute_sha256,
    verify_champion,
    run_phase_43_evaluation,
)


class TestPhase43PartNumberCalibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load or run Phase 43 evaluation results."""
        if not OUTPUT_RESULTS_PATH.exists():
            cls.results = run_phase_43_evaluation(n_bootstrap=100, seed=42)
        else:
            with open(OUTPUT_RESULTS_PATH, "r", encoding="utf-8") as f:
                cls.results = json.load(f)

    def test_champion_file_integrity(self):
        """Verify Phase 40 champion exists and SHA-256 matches expected digest."""
        self.assertTrue(CHAMPION_TSV_PATH.exists(), f"Champion TSV missing: {CHAMPION_TSV_PATH}")
        actual_hash = compute_sha256(CHAMPION_TSV_PATH)
        self.assertEqual(
            actual_hash,
            EXPECTED_CHAMPION_SHA256,
            f"Champion SHA-256 mismatch! Expected {EXPECTED_CHAMPION_SHA256}, got {actual_hash}",
        )
        champ_info = verify_champion()
        self.assertEqual(champ_info["status"], "PASSED")
        self.assertEqual(champ_info["sha256"], EXPECTED_CHAMPION_SHA256)

    def test_baseline_reproduction(self):
        """Verify exact reproduction of Phase 40 local baseline."""
        b = self.results["baseline_reproduction"]
        self.assertAlmostEqual(b["overall_official_score"], 0.927771, places=5)
        self.assertAlmostEqual(b["category_1_score"], 0.918323, places=5)
        self.assertAlmostEqual(b["category_2_score"], 0.937219, places=5)
        self.assertEqual(b["micro_tp"], 3021)
        self.assertEqual(b["micro_fp"], 165)
        self.assertEqual(b["micro_fn"], 752)
        self.assertAlmostEqual(b["micro_f02"], 0.941539, places=5)

    def test_phase_42b_claim_verification(self):
        """Verify empirical status of Phase 42B's claim regarding Herstellernummer in Cat 2."""
        audit = self.results["phase_42b_audit"]
        # Claim: exactly 6 false positives in Category 2
        self.assertEqual(audit["cat2_fp_count"], 6)

        # Claim: 4 FPs with confidence in [0.60, 0.80)
        # Empirical finding: exactly 3 FPs have min_conf < 0.80, NOT 4!
        self.assertEqual(audit["cat2_fp_in_60_to_80_count"], 3)

        # Unreported finding: 9 True Positives have min_conf < 0.80!
        self.assertEqual(audit["cat2_tp_in_60_to_80_count"], 9)
        self.assertEqual(audit["cat2_preds_in_60_to_80_count"], 12)

    def test_preservation_of_non_target_predictions(self):
        """Confirm that non-target aspects are 100% untouched by threshold adjustments."""
        diff_audit = self.results["differences_audit"]
        self.assertEqual(diff_audit["non_target_aspect_changes"], 0)
        self.assertEqual(diff_audit["new_predictions"], 0)
        # All differences must be Herstellernummer in Category 2
        for d in diff_audit["differences"]:
            self.assertEqual(d["aspect"], "Herstellernummer")
            self.assertEqual(d["category"], 2)

    def test_candidate_metrics_and_accounting(self):
        """Verify metrics and exact TP/FP trade-off for Candidate A and Candidate B."""
        diff_audit = self.results["differences_audit"]
        self.assertEqual(diff_audit["total_prediction_changes"], 12)
        self.assertEqual(diff_audit["tp_removed"], 9)
        self.assertEqual(diff_audit["fp_removed"], 3)
        self.assertEqual(diff_audit["fn_introduced"], 9)

        ca = self.results["candidate_a"]
        cb = self.results["candidate_b"]

        # Candidate A deltas
        self.assertAlmostEqual(ca["overall_official_score"], 0.928656, places=5)
        self.assertAlmostEqual(ca["overall_delta"], 0.000885, places=5)
        self.assertAlmostEqual(ca["category_1_delta"], 0.0, places=6)
        self.assertAlmostEqual(ca["category_2_delta"], 0.001771, places=5)
        self.assertEqual(ca["micro_tp"], 3012)
        self.assertEqual(ca["micro_fp"], 162)
        self.assertEqual(ca["micro_fn"], 761)

        # Candidate B is identical to Candidate A because Cat 1 has 0 predictions in [0.60, 0.80)
        self.assertAlmostEqual(cb["overall_official_score"], ca["overall_official_score"], places=6)
        self.assertAlmostEqual(cb["overall_delta"], ca["overall_delta"], places=6)
        self.assertEqual(cb["micro_tp"], ca["micro_tp"])
        self.assertEqual(cb["micro_fp"], ca["micro_fp"])

    def test_hierarchical_metric_calculation(self):
        """Verify that overall delta is exactly the unweighted mean of category deltas."""
        ca = self.results["candidate_a"]
        expected_overall_delta = (ca["category_1_delta"] + ca["category_2_delta"]) / 2.0
        self.assertAlmostEqual(ca["overall_delta"], expected_overall_delta, places=6)

    def test_bootstrap_interval_and_decision_rules(self):
        """Verify bootstrap confidence interval and decision rule classification."""
        boot = self.results["bootstrap_analysis"]
        ci = boot["ci_95_overall"]
        # 95% CI must cross zero into negative territory
        self.assertLess(ci[0], 0.0, f"Expected lower bound < 0, got {ci[0]}")
        self.assertGreater(ci[1], 0.0, f"Expected upper bound > 0, got {ci[1]}")
        # Over 10% of resamples regress
        self.assertGreater(boot["fraction_le_0_overall"], 0.10)

        # Decision rule: delta < +0.0020 and CI crosses zero -> MODEST LOCAL CHALLENGER
        dec = self.results["decision"]
        self.assertIn(dec["verdict"], ["MODEST LOCAL CHALLENGER", "RETAIN PHASE 40 CHAMPION"])
        self.assertTrue(len(dec["reasons"]) >= 4)


if __name__ == "__main__":
    unittest.main()
