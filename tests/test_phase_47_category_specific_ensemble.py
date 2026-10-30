"""
Unit tests for Phase 47: Category-Specific Boundary-Specialist Ensemble Evaluation.

Tests cover:
1. Champion file integrity and SHA-256 digest preservation.
2. Local clean-holdout official baseline reproduction (Phase 40).
3. Phase 46 verification and discrepancy resolution.
4. Category-specific routing and probability fusion logic.
5. Bit-for-bit invariance of Category 1 predictions.
6. Exact Candidate A metric calculations (Overall +0.002273, Cat 2 +0.004546).
7. Independent evidence audit results (Phase 32C internal dev regression on Cat 2).
8. Paired bootstrap confidence intervals and final decision verdict.
"""

import json
from pathlib import Path
import unittest

import numpy as np
import torch
from transformers import AutoConfig

from src.evaluation.phase_43_part_number_calibration import (
    CHAMPION_TSV_PATH,
    EXPECTED_CHAMPION_SHA256,
    compute_sha256,
    verify_champion,
)
from src.evaluation.phase_47_category_specific_ensemble import (
    OUTPUT_RESULTS_PATH,
    PHASE32C_CLEAN_HOLDOUT_PROBS_PATH,
    evaluate_independent_evidence,
    run_phase_47_evaluation,
)


class TestPhase47CategorySpecificEnsemble(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load or run Phase 47 evaluation results."""
        if not OUTPUT_RESULTS_PATH.exists():
            cls.results = run_phase_47_evaluation(n_bootstrap=1000, seed=42)
        else:
            with open(OUTPUT_RESULTS_PATH, "r", encoding="utf-8") as f:
                cls.results = json.load(f)

    def test_champion_integrity(self):
        """Verify Phase 40 champion exists and SHA-256 digest is unaltered."""
        self.assertTrue(CHAMPION_TSV_PATH.exists(), f"Champion TSV missing: {CHAMPION_TSV_PATH}")
        actual_hash = compute_sha256(CHAMPION_TSV_PATH)
        self.assertEqual(actual_hash, EXPECTED_CHAMPION_SHA256)
        champ_info = verify_champion()
        self.assertEqual(champ_info["status"], "PASSED")
        self.assertAlmostEqual(champ_info["official_quiz_f02"], 0.9233317, places=6)

    def test_baseline_reproduction(self):
        """Verify local clean-holdout baseline reproduction matches official benchmark."""
        b = self.results["baseline_reproduction"]
        self.assertAlmostEqual(b["overall_score"], 0.927771, places=5)
        self.assertAlmostEqual(b["cat1_score"], 0.918323, places=5)
        self.assertAlmostEqual(b["cat2_score"], 0.937219, places=5)
        self.assertEqual(b["tp"], 3021)
        self.assertEqual(b["fp"], 165)
        self.assertEqual(b["fn"], 752)
        self.assertAlmostEqual(b["micro_f02"], 0.941539, places=5)

    def test_phase_46_verification_and_discrepancy(self):
        """Verify reproduction of global blend results and documented discrepancy."""
        p46 = self.results["phase_46_verification"]["global_blend_weight_010"]
        self.assertAlmostEqual(p46["overall_score"], 0.928892, places=5)
        self.assertAlmostEqual(p46["cat1_score"], 0.916020, places=5)
        self.assertAlmostEqual(p46["cat2_score"], 0.941765, places=5)
        self.assertIn("conflated", self.results["phase_46_verification"]["discrepancy_explanation"])

    def test_independent_evidence_audit(self):
        """Verify independent evidence shows specialist regressed on Cat 2 in prior experiments."""
        ev = self.results["independent_evidence_audit"]
        self.assertFalse(ev["independent_support_found"])
        int_dev = ev["internal_800_dev"]
        self.assertIsNotNone(int_dev)
        # On internal 800 dev, Cat 1 gained +0.004225, but Cat 2 regressed by -0.002832
        self.assertGreater(int_dev["delta_cat1"], 0.0)
        self.assertLess(int_dev["delta_cat2"], 0.0)
        self.assertAlmostEqual(int_dev["delta_cat2"], -0.002832, places=5)

    def test_category_1_invariance(self):
        """Verify Category 1 predictions in Candidate A are 100% bit-for-bit identical to Phase 40."""
        cand = self.results["candidate_a_results"]
        self.assertTrue(cand["cat1_invariance_verified"])
        self.assertAlmostEqual(cand["delta_cat1"], 0.000000, places=6)
        self.assertAlmostEqual(cand["cat1_score"], 0.918323, places=5)

    def test_candidate_a_metrics(self):
        """Verify Candidate A achieves Overall +0.002273 and Cat 2 +0.004546."""
        cand = self.results["candidate_a_results"]
        self.assertAlmostEqual(cand["overall_score"], 0.930044, places=5)
        self.assertAlmostEqual(cand["delta_overall"], 0.002273, places=5)
        self.assertAlmostEqual(cand["cat2_score"], 0.941764, places=5)
        self.assertAlmostEqual(cand["delta_cat2"], 0.004546, places=5)
        self.assertEqual(cand["tp_delta"], 3)
        self.assertEqual(cand["fp_delta"], -4)
        self.assertEqual(cand["fn_delta"], -3)
        self.assertEqual(cand["cat2_titles_changed"], 13)

    def test_label_alignment_and_cache(self):
        """Verify cached specialist probabilities match the 48-label BIO space."""
        self.assertTrue(PHASE32C_CLEAN_HOLDOUT_PROBS_PATH.exists())
        probs = torch.load(PHASE32C_CLEAN_HOLDOUT_PROBS_PATH, map_location="cpu", weights_only=False)
        self.assertEqual(len(probs), 489)
        self.assertEqual(len(probs[0][0]), 48)  # 48 BIO classes

    def test_bootstrap_results(self):
        """Verify Candidate A bootstrap distribution."""
        boot = self.results["bootstrap_results"]
        self.assertAlmostEqual(boot["mean_delta_overall"], 0.002228, places=5)
        self.assertGreater(boot["ci_95_overall"][0], 0.0)  # Lower bound > 0
        self.assertAlmostEqual(boot["ci_95_c1"][0], 0.0)
        self.assertAlmostEqual(boot["ci_95_c1"][1], 0.0)

    def test_decision_verdict(self):
        """Verify final decision verdict is exactly PROMISING LOCAL CHALLENGER — GENERALIZATION UNCONFIRMED."""
        dec = self.results["decision"]
        self.assertEqual(dec["verdict"], "PROMISING LOCAL CHALLENGER — GENERALIZATION UNCONFIRMED")
        self.assertTrue(dec["passed_delta_ge_002"])
        self.assertTrue(dec["passed_cat1_invariance"])
        self.assertTrue(dec["passed_cat2_improvement"])
        self.assertTrue(dec["passed_exact_match_improvements"])
        self.assertFalse(dec["independent_support_present"])


if __name__ == "__main__":
    unittest.main()
