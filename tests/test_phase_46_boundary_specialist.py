"""
Unit tests for Phase 46: Entity Boundary Specialist and Sixth-Model Ensemble.

Tests cover:
1. Champion file integrity and SHA-256 digest preservation.
2. Local clean-holdout official baseline reproduction (0.927771).
3. Trained model inventory (active ensemble vs unused historical checkpoints).
4. Boundary failure mode distribution (missing final tokens, merging, splitting, aspect confusion).
5. Unused checkpoint standalone performance and ensemble fusion behavior.
6. Bootstrap confidence interval assertions (failure to clear +0.0020 threshold).
7. Decision verdict assertion (RETAIN PHASE 40 — INSUFFICIENT EVIDENCE).
"""

import json
from pathlib import Path
import unittest

from src.evaluation.phase_43_part_number_calibration import (
    CHAMPION_TSV_PATH,
    EXPECTED_CHAMPION_SHA256,
    compute_sha256,
    verify_champion,
)
from src.evaluation.phase_46_boundary_specialist_evaluation import (
    OUTPUT_RESULTS_PATH,
    run_phase_46_evaluation,
)


class TestPhase46BoundarySpecialist(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load or run Phase 46 evaluation results."""
        if not OUTPUT_RESULTS_PATH.exists():
            cls.results = run_phase_46_evaluation(n_bootstrap=1000, seed=42)
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

    def test_model_inventory(self):
        """Verify model inventory reflects 4 active models and identifies unused checkpoints."""
        inv = self.results["model_inventory"]
        # Active Phase 40 models
        self.assertTrue(inv["phase33b_dapt_gbert_best"]["part_of_phase40"])
        self.assertEqual(inv["phase33b_dapt_gbert_best"]["phase40_weight"], 0.40)
        self.assertTrue(inv["xlmr_baseline_best"]["part_of_phase40"])
        self.assertTrue(inv["xlmr_seed_7_best"]["part_of_phase40"])
        self.assertTrue(inv["xlmr_seed_21_best"]["part_of_phase40"])

        # Unused models
        self.assertFalse(inv["gbert_baseline_best"]["part_of_phase40"])
        self.assertFalse(inv["gelectra_baseline_best"]["part_of_phase40"])
        self.assertFalse(inv["gbert_phase32c_challenger_best"]["part_of_phase40"])
        self.assertTrue(inv["gbert_baseline_best"]["weights_present"])

    def test_boundary_failure_modes(self):
        """Verify boundary failure modes audit counts across categories."""
        bf = self.results["boundary_failure_modes"]
        self.assertEqual(bf["1_missing_final_token"]["total"], 47)
        self.assertEqual(bf["1_missing_final_token"]["cat1"], 37)
        self.assertEqual(bf["1_missing_final_token"]["cat2"], 10)

        self.assertEqual(bf["2_extra_final_token"]["total"], 1)
        self.assertEqual(bf["3_missing_leading_token"]["total"], 6)
        self.assertEqual(bf["4_extra_leading_token"]["total"], 0)
        self.assertEqual(bf["5_incorrect_merging"]["total"], 42)
        self.assertEqual(bf["6_incorrect_splitting"]["total"], 15)
        self.assertEqual(bf["7_wrong_aspect_same_boundary"]["total"], 31)

    def test_unused_checkpoint_complementarity(self):
        """Verify standalone scores and error overlap for unused checkpoints."""
        comp = self.results["unused_checkpoint_complementarity"]
        # orig_gbert standalone
        self.assertAlmostEqual(comp["orig_gbert"]["standalone_score"], 0.904415, places=5)
        self.assertEqual(comp["orig_gbert"]["standalone_fp"], 263)
        self.assertEqual(comp["orig_gbert"]["fixed_errors_in_gold"], 130)
        self.assertEqual(comp["orig_gbert"]["damaged_correct_in_gold"], 114)

        # gelectra standalone
        self.assertAlmostEqual(comp["gelectra_baseline"]["standalone_score"], 0.872451, places=5)
        self.assertLess(comp["gelectra_baseline"]["fusion_weight_005_delta"], 0.0)

        # gbert_phase32c challenger
        self.assertAlmostEqual(comp["gbert_phase32c_challenger"]["standalone_score"], 0.898950, places=5)
        self.assertLess(comp["gbert_phase32c_challenger"]["fusion_weight_010_cat1_delta"], 0.0)

    def test_ensemble_fusion_results(self):
        """Verify that fusion with orig_gbert does not achieve the required +0.0020 improvement."""
        f = self.results["ensemble_fusion_experiments"]
        # Weight 0.10 achieves only +0.000239 delta, and Cat 1 regresses
        self.assertAlmostEqual(f["orig_gbert_weight_0.10"]["delta_overall"], 0.000239, places=5)
        self.assertLess(f["orig_gbert_weight_0.10"]["delta_overall"], 0.0020)
        self.assertLess(f["orig_gbert_weight_0.10"]["delta_cat1"], 0.0)

        # Weight 0.20 actively regresses overall
        self.assertLess(f["orig_gbert_weight_0.20"]["delta_overall"], 0.0)
        self.assertAlmostEqual(f["orig_gbert_weight_0.20"]["delta_overall"], -0.001608, places=5)

    def test_bootstrap_uncertainty(self):
        """Verify bootstrap confidence interval crosses zero and fails statistical gate."""
        boot = self.results["bootstrap_results"]["candidate_weight_010"]
        self.assertLess(boot["ci_95_overall"][0], 0.0)  # Lower bound is negative
        self.assertGreater(boot["fraction_le_0_overall"], 0.40)  # > 40% probability of <= 0
        self.assertLess(boot["mean_delta_cat1"], 0.0)  # Cat 1 mean delta is negative

    def test_decision_verdict(self):
        """Verify final decision unambiguously retains Phase 40 due to insufficient evidence."""
        dec = self.results["decision"]
        self.assertEqual(dec["verdict"], "RETAIN PHASE 40 — INSUFFICIENT EVIDENCE")
        self.assertFalse(dec["passed_delta_ge_002"])
        self.assertFalse(dec["passed_no_category_regression"])
        self.assertFalse(dec["passed_bootstrap_ci_positive"])


if __name__ == "__main__":
    unittest.main()
