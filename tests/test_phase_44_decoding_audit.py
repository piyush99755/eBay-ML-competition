"""
Unit tests for Phase 44: Token-Level Decoding Opportunity & Feasibility Audit.

Tests cover:
1. Champion file integrity and SHA-256 verification.
2. Local baseline reproduction (Phase 40).
3. Production decoding pipeline invariants and invalid BIO transition frequency.
4. Quantitative error decomposition (Buckets A, B, C, D for FNs; boundary vs aspect confusion for FPs).
5. Theoretical headroom simulations and feasibility decisions.
"""

import json
from pathlib import Path
import unittest

from src.evaluation.phase_44_decoding_audit import (
    CHAMPION_TSV_PATH,
    EXPECTED_CHAMPION_SHA256,
    OUTPUT_RESULTS_PATH,
    compute_sha256,
    verify_champion,
    run_phase_44_audit,
)


class TestPhase44DecodingAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load or run Phase 44 audit results."""
        if not OUTPUT_RESULTS_PATH.exists():
            cls.results = run_phase_44_audit()
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

    def test_baseline_reproduction(self):
        """Verify local baseline reproduction matches Phase 40 benchmark."""
        b = self.results["baseline_reproduction"]
        self.assertAlmostEqual(b["overall_score"], 0.927771, places=5)
        self.assertAlmostEqual(b["cat1_score"], 0.918323, places=5)
        self.assertAlmostEqual(b["cat2_score"], 0.937219, places=5)
        self.assertEqual(b["tp"], 3021)
        self.assertEqual(b["fp"], 165)
        self.assertEqual(b["fn"], 752)
        self.assertAlmostEqual(b["f02"], 0.941539, places=5)

    def test_invalid_bio_transition_rate(self):
        """Verify that invalid BIO transitions in the fused model are negligible (< 0.5%)."""
        arch = self.results["decoding_architecture"]
        self.assertEqual(arch["invalid_transition_count"], 13)
        self.assertEqual(arch["total_tokens"], 5563)
        self.assertLess(arch["invalid_transition_rate"], 0.005)
        self.assertAlmostEqual(arch["invalid_transition_rate"], 0.002337, places=5)

    def test_error_decomposition_counts(self):
        """Verify decomposition of False Negatives (752) and False Positives (165)."""
        err = self.results["error_recoverability"]
        self.assertEqual(err["total_fn"], 752)
        self.assertEqual(err["total_fp"], 165)

        fn_b = err["fn_buckets"]
        self.assertEqual(fn_b["A"], 431)
        self.assertEqual(fn_b["B"], 154)
        self.assertEqual(fn_b["C"], 125)
        self.assertEqual(fn_b["D"], 42)
        self.assertEqual(sum(fn_b.values()), 752)

        fp_c = err["fp_categories"]
        self.assertEqual(fp_c["boundary_mismatch"], 120)
        self.assertEqual(fp_c["aspect_confusion"], 41)
        self.assertEqual(fp_c["pure_hallucination"], 4)
        self.assertEqual(sum(fp_c.values()), 165)

    def test_headroom_simulation_bounds(self):
        """Verify theoretical headroom upper bounds for each bucket."""
        headroom = self.results["error_recoverability"]["headroom_simulation"]
        # Bucket A max gain ~ +0.0257
        self.assertAlmostEqual(headroom["A"]["theoretical_max_delta"], 0.025708, places=4)
        # Bucket D (Aspect Confusion) max gain ~ +0.00455
        self.assertAlmostEqual(headroom["D"]["theoretical_max_delta"], 0.004551, places=4)

    def test_critical_feasibility_and_decision(self):
        """Verify critical feasibility questions and final decision verdict."""
        crit = self.results["critical_feasibility_check"]
        self.assertTrue(crit["1_probabilities_cached"])
        self.assertTrue(crit["2_alignments_reliable"])
        self.assertFalse(crit["4_parameters_learned_safely"])
        self.assertFalse(crit["8_sufficient_evidence_before_nov6"])

        dec = self.results["decision"]
        self.assertEqual(dec["verdict"], "RETAIN PHASE 40 — NO CONVINCING DECODING OPPORTUNITY")


if __name__ == "__main__":
    unittest.main()
