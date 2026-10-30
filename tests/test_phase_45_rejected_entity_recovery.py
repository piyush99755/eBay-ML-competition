"""
Unit tests for Phase 45: Precision-Gated Recovery of Rejected Entities.

Tests cover:
1. Champion file integrity and SHA-256 digest preservation.
2. Local clean-holdout official baseline reproduction.
3. Reconstructed raw entities and audit of rejected predictions (569 total: 327 TP, 242 FP).
4. Predeclared 5-point gating rule evaluation (94 passing, 65 TP, 29 FP, 69.15% precision).
5. Score regression under F0.2 (overall delta -0.003968, both categories regressing).
6. 1,000 paired category-stratified bootstrap uncertainty distribution (strictly negative 95% CI).
7. Decision verdict and criteria verification (REJECT — RETAIN PHASE 40).
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
from src.evaluation.phase_45_rejected_entity_recovery import (
    OUTPUT_RESULTS_PATH,
    run_phase_45_evaluation,
)


class TestPhase45RejectedEntityRecovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load or run Phase 45 evaluation results."""
        if not OUTPUT_RESULTS_PATH.exists():
            cls.results = run_phase_45_evaluation(n_bootstrap=1000, seed=42)
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

    def test_rejected_entities_audit(self):
        """Verify raw decoding counts and exact TP/FP split of rejected predictions."""
        aud = self.results["rejected_entities_audit"]
        self.assertEqual(aud["total_raw_entities"], 3755)
        self.assertEqual(aud["total_accepted_entities"], 3186)
        self.assertEqual(aud["total_rejected_entities"], 569)
        self.assertEqual(aud["rejected_exact_tp"], 327)
        self.assertEqual(aud["rejected_exact_fp"], 242)
        self.assertAlmostEqual(aud["rejected_raw_precision"], 0.574692, places=5)
        self.assertEqual(aud["aspect_counts"]["Kompatibles_Fahrzeug_Modell"], 303)
        self.assertEqual(aud["aspect_counts"]["Produktart"], 82)
        self.assertEqual(aud["aspect_counts"]["Im_Lieferumfang_Enthalten"], 56)

    def test_predeclared_gate_results(self):
        """Verify candidates passing the predeclared 5-point gate and their precision."""
        gate = self.results["recovery_gate_results"]
        self.assertEqual(gate["candidates_passing_gate"], 94)
        self.assertEqual(gate["recovered_tp"], 65)
        self.assertEqual(gate["recovered_fp"], 29)
        self.assertAlmostEqual(gate["recovered_precision"], 0.691489, places=5)

        # Aspect breakdown checks
        asp = gate["aspect_breakdown"]
        self.assertEqual(asp["Produktart"]["total_recovered"], 23)
        self.assertEqual(asp["Produktart"]["tp"], 9)
        self.assertEqual(asp["Produktart"]["fp"], 14)
        self.assertAlmostEqual(asp["Produktart"]["precision"], 0.391304, places=5)

        self.assertEqual(asp["Kompatibles_Fahrzeug_Modell"]["total_recovered"], 52)
        self.assertEqual(asp["Kompatibles_Fahrzeug_Modell"]["tp"], 43)
        self.assertEqual(asp["Kompatibles_Fahrzeug_Modell"]["fp"], 9)
        self.assertAlmostEqual(asp["Kompatibles_Fahrzeug_Modell"]["precision"], 0.826923, places=5)

        self.assertEqual(asp["Im_Lieferumfang_Enthalten"]["total_recovered"], 19)
        self.assertEqual(asp["Im_Lieferumfang_Enthalten"]["tp"], 13)
        self.assertEqual(asp["Im_Lieferumfang_Enthalten"]["fp"], 6)

    def test_score_evaluation_regression(self):
        """Verify that recovering 65 TP and 29 FP degrades F0.2 due to 25x FP penalty."""
        eval_res = self.results["gated_score_evaluation"]
        self.assertAlmostEqual(eval_res["overall_score"], 0.923803, places=5)
        self.assertAlmostEqual(eval_res["overall_delta"], -0.003968, places=5)
        self.assertAlmostEqual(eval_res["category_1_delta"], -0.004095, places=5)
        self.assertAlmostEqual(eval_res["category_2_delta"], -0.003841, places=5)
        self.assertEqual(eval_res["tp"], 3086)
        self.assertEqual(eval_res["fp"], 194)
        self.assertEqual(eval_res["fn"], 687)
        self.assertAlmostEqual(eval_res["micro_delta"], -0.006093, places=5)

    def test_bootstrap_uncertainty(self):
        """Verify bootstrap confidence intervals are strictly negative with >= 98% regression."""
        boot = self.results["bootstrap_analysis"]
        self.assertEqual(boot["n_resamples"], 1000)
        self.assertLess(boot["ci_95_overall"][1], 0.0)  # Upper CI bound is negative!
        self.assertAlmostEqual(boot["ci_95_overall"][0], -0.006966, places=4)
        self.assertAlmostEqual(boot["ci_95_overall"][1], -0.000700, places=4)
        self.assertGreaterEqual(boot["fraction_le_0_overall"], 0.98)

    def test_decision_verdict(self):
        """Verify that decision unambiguously rejects recovery and retains Phase 40."""
        dec = self.results["decision"]
        self.assertEqual(dec["verdict"], "REJECT — RETAIN PHASE 40")
        self.assertFalse(dec["passed_delta_ge_002"])
        self.assertFalse(dec["passed_no_category_regression"])
        self.assertFalse(dec["passed_precision_breakeven"])
        self.assertFalse(dec["passed_bootstrap_ci_positive"])


if __name__ == "__main__":
    unittest.main()
