"""
Unit and regression tests for Phase 39C: Category-Specific Calibration & Package-Contents Re-Evaluation.

Tests:
1. Baseline integrity verification (champion SHA-256, checkpoints, cache files).
2. Prediction invariants for Candidate A (Cat 1 bit-for-bit unchanged, Cat 2 additive preservation).
3. Prediction invariants for Candidate B (strictly removes Im_Lieferumfang_Enthalten, 0 other aspects changed).
4. Reproduction of official competition scores and deltas on the 489 clean holdout.
5. Reproduction of global micro metrics and trade-offs.
6. Cross-partition consistency on 511 validation titles.
7. Challenger classification rule asserting 'STRONG LOCAL CHALLENGER — B'.
"""

import json
import os
import unittest

from src.evaluation.phase_39c_candidate_comparison import (
    CHAMPION_TSV_PATH,
    EXPECTED_CHAMPION_SHA256,
    RESULTS_CACHE_PATH,
    verify_phase_integrity,
    execute_phase_39c_pipeline,
)


class TestPhase39CCandidateComparison(unittest.TestCase):

    def test_verify_baseline_integrity(self):
        """Verify champion TSV hash, checkpoints, manifest, and cached probabilities."""
        res = verify_phase_integrity()
        self.assertEqual(res["status"], "PASSED")
        self.assertEqual(res["champion_sha256"], EXPECTED_CHAMPION_SHA256)
        self.assertEqual(res["checkpoints_verified"], 4)

    def test_cached_results_and_invariants(self):
        """Verify the stored pipeline results for prediction changes and score deltas."""
        self.assertTrue(os.path.exists(RESULTS_CACHE_PATH), f"Results cache missing: {RESULTS_CACHE_PATH}")
        with open(RESULTS_CACHE_PATH, "r", encoding="utf-8") as f:
            res = json.load(f)

        # Baseline reproduction
        base = res["clean_holdout"]["baseline"]
        self.assertAlmostEqual(base["official_score"], 0.924945, places=5)
        self.assertAlmostEqual(base["cat1_score"], 0.915723, places=5)
        self.assertAlmostEqual(base["cat2_score"], 0.934168, places=5)
        self.assertAlmostEqual(base["micro_f02"], 0.938333, places=5)
        self.assertEqual(base["tp"], 3045)
        self.assertEqual(base["fp"], 179)
        self.assertEqual(base["fn"], 728)

        # Candidate A metrics and invariants
        cand_a = res["clean_holdout"]["candidate_a"]
        self.assertAlmostEqual(cand_a["delta_cat1"], 0.000000, places=6)
        self.assertAlmostEqual(cand_a["delta_cat2"], +0.002088, places=5)
        self.assertAlmostEqual(cand_a["delta_official"], +0.001044, places=5)
        self.assertEqual(cand_a["delta_tp"], 35)
        self.assertEqual(cand_a["delta_fp"], 4)

        # Candidate B metrics and invariants
        cand_b = res["clean_holdout"]["candidate_b"]
        self.assertAlmostEqual(cand_b["delta_official"], +0.002825, places=5)
        self.assertAlmostEqual(cand_b["delta_cat1"], +0.002600, places=5)
        self.assertAlmostEqual(cand_b["delta_cat2"], +0.003051, places=5)
        self.assertAlmostEqual(cand_b["delta_micro_f02"], +0.003206, places=5)
        self.assertEqual(cand_b["delta_tp"], -24)
        self.assertEqual(cand_b["delta_fp"], -14)

        # Stability checks
        stab = res["stability"]
        self.assertEqual(stab["candidate_a"]["num_affected_titles"], 23)
        self.assertEqual(stab["candidate_a"]["num_added_entities"], 39)
        self.assertEqual(stab["candidate_b"]["num_affected_titles"], 33)
        self.assertEqual(stab["candidate_b"]["num_removed_entities"], 38)
        self.assertEqual(stab["candidate_b"]["removed_category_1"], 28)
        self.assertEqual(stab["candidate_b"]["removed_category_2"], 10)

        # Bootstrap verification
        boot = res["bootstrap"]
        self.assertGreater(boot["candidate_b"]["ci_95"][0], 0.0)
        self.assertLess(boot["candidate_a"]["ci_95"][0], 0.0)
        self.assertLessEqual(boot["candidate_b"]["p_val_le_zero"], 0.01)

        # Cross-partition diagnostic on 511 contaminated titles
        cross = res["contaminated_validation_diagnostic"]
        self.assertGreater(cross["candidate_b"]["delta_official"], +0.0020)
        self.assertLess(cross["candidate_a"]["delta_official"], 0.0)

        # Final decision classification
        self.assertEqual(res["decision"], "STRONG LOCAL CHALLENGER — B")

    def test_live_pipeline_execution_without_bootstrap(self):
        """Execute the pipeline live with run_bootstrap=False to verify determinism and invariants."""
        live_res = execute_phase_39c_pipeline(run_bootstrap=False, save_cache=False)
        self.assertAlmostEqual(
            live_res["clean_holdout"]["candidate_b"]["delta_official"],
            +0.002825,
            places=5,
        )
        self.assertAlmostEqual(
            live_res["clean_holdout"]["candidate_a"]["delta_cat1"],
            0.0,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
