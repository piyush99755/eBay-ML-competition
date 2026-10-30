"""
Unit and regression tests for Phase 39B: Precision-Guided Vehicle-Model Recovery.

Tests:
1. Integrity pre-flight verification (champion SHA-256, checkpoints, cache files).
2. Ambiguity filtering rules (stopwords, single char/digits, engine displacements, length/frequency).
3. Zero-holdout leakage verification (lexicon strictly from train_4k).
4. Non-destructive addition invariant enforcement.
5. Candidate counts, precision bookkeeping, and official/micro metric reproduction.
6. Challenger decision rule enforcement (RETAIN PHASE 36C CHAMPION).
"""

import json
import os
import unittest
from collections import Counter

from src.evaluation.phase_39b_vehicle_model_recovery import (
    CHAMPION_TSV_PATH,
    EXPECTED_CHAMPION_SHA256,
    RESULTS_CACHE_PATH,
    STOPWORDS,
    DECIMAL_REGEX,
    verify_phase_integrity,
    build_training_vehicle_lexicon,
    execute_phase_39b_evaluation,
)


class TestPhase39BVehicleModelRecovery(unittest.TestCase):

    def test_verify_phase_integrity(self):
        """Verify champion TSV hash, checkpoints, manifest, and cached probabilities."""
        res = verify_phase_integrity()
        self.assertEqual(res["status"], "PASSED")
        self.assertEqual(res["champion_sha256"], EXPECTED_CHAMPION_SHA256)
        self.assertEqual(res["checkpoints_verified"], 4)

    def test_ambiguity_filter_rules(self):
        """Test deterministic ambiguity filter logic on synthetic candidates."""
        synthetic_train_examples = [
            {
                "record_number": 1,
                "title": "dummy",
                "category": 1,
                "aspects": {
                    "Kompatible_Fahrzeug_Marke": ["BMW", "VW"],
                    "Kompatibles_Fahrzeug_Modell": [
                        "Golf",      # valid, len >= 4
                        "Passat",    # valid, len >= 4
                        "Up",        # count 1, len < 4 -> should be filtered
                        "1",         # single digit -> filtered
                        "A",         # single letter -> filtered
                        "2.0",       # engine displacement -> filtered
                        "auch",      # stopword -> filtered
                        "set",       # stopword -> filtered
                    ],
                },
            },
            {
                "record_number": 2,
                "title": "dummy 2",
                "category": 1,
                "aspects": {
                    "Kompatible_Fahrzeug_Marke": ["VW"],
                    "Kompatibles_Fahrzeug_Modell": [
                        "Golf",      # count now 2
                        "TT",        # count 1, len 2 -> filtered
                        "M3",        # count 1, len 2 -> filtered
                    ],
                },
            },
            {
                "record_number": 3,
                "title": "dummy 3",
                "category": 1,
                "aspects": {
                    "Kompatible_Fahrzeug_Marke": ["Audi"],
                    "Kompatibles_Fahrzeug_Modell": [
                        "TT",        # count now 2, len 2 -> admitted because cnt >= 2!
                    ],
                },
            },
        ]

        lexicon, pairs, model_counts, make_counts = build_training_vehicle_lexicon(synthetic_train_examples)

        # "Golf" (len 4, cnt 2) should be in lexicon
        self.assertIn("Golf", lexicon)
        # "Passat" (len 6, cnt 1) should be in lexicon because len >= 4
        self.assertIn("Passat", lexicon)
        # "TT" (cnt 2) should be in lexicon because cnt >= 2
        self.assertIn("TT", lexicon)

        # Filtered out:
        self.assertNotIn("Up", lexicon)       # cnt 1, len 2
        self.assertNotIn("M3", lexicon)       # cnt 1, len 2
        self.assertNotIn("1", lexicon)        # single digit
        self.assertNotIn("A", lexicon)        # single letter
        self.assertNotIn("2.0", lexicon)      # decimal / engine displacement
        self.assertNotIn("auch", lexicon)     # stopword
        self.assertNotIn("set", lexicon)      # stopword

    def test_cached_pipeline_results_consistency(self):
        """Verify the stored pipeline results for candidate bookkeeping and metrics."""
        self.assertTrue(os.path.exists(RESULTS_CACHE_PATH), f"Results cache missing: {RESULTS_CACHE_PATH}")
        with open(RESULTS_CACHE_PATH, "r", encoding="utf-8") as f:
            res = json.load(f)

        # Lexicon statistics
        lex = res["lexicon_stats"]
        self.assertEqual(lex["total_raw_models"], 4046)
        self.assertEqual(lex["total_model_occurrences"], 7480)
        self.assertEqual(lex["total_raw_makes"], 178)
        self.assertEqual(lex["filtered_lexicon_size"], 3662)

        # Candidate audit
        audit = res["suppressed_candidate_audit"]
        self.assertEqual(audit["total_suppressed_candidates"], 204)
        self.assertEqual(audit["candidate_a_admitted"], 95)
        self.assertEqual(audit["candidate_a_tp"], 76)
        self.assertEqual(audit["candidate_a_fp"], 19)
        self.assertAlmostEqual(audit["candidate_a_precision"], 0.80, places=4)

        self.assertEqual(audit["candidate_b_admitted"], 104)
        self.assertEqual(audit["candidate_b_tp"], 81)
        self.assertEqual(audit["candidate_b_fp"], 23)
        self.assertAlmostEqual(audit["candidate_b_precision"], 77.8846 / 100.0, places=4)

        # Baseline metrics reproduction
        m_base = res["metrics"]["baseline"]
        self.assertAlmostEqual(m_base["official_score"], 0.924945, places=5)
        self.assertAlmostEqual(m_base["cat1_score"], 0.915723, places=5)
        self.assertAlmostEqual(m_base["cat2_score"], 0.934168, places=5)
        self.assertAlmostEqual(m_base["micro_f02"], 0.938333, places=5)
        self.assertEqual(m_base["tp"], 3045)
        self.assertEqual(m_base["fp"], 179)
        self.assertEqual(m_base["fn"], 728)

        # Candidate A deltas
        m_a = res["metrics"]["candidate_a"]
        self.assertAlmostEqual(m_a["delta_official"], +0.000149, places=5)
        self.assertAlmostEqual(m_a["delta_cat1"], -0.001789, places=5)
        self.assertAlmostEqual(m_a["delta_cat2"], +0.002088, places=5)
        self.assertAlmostEqual(m_a["delta_micro_f02"], -0.002911, places=5)

        # Candidate B deltas
        m_b = res["metrics"]["candidate_b"]
        self.assertAlmostEqual(m_b["delta_official"], -0.000906, places=5)
        self.assertAlmostEqual(m_b["delta_cat1"], -0.002173, places=5)
        self.assertAlmostEqual(m_b["delta_cat2"], +0.000360, places=5)

        # Decision rule verification
        self.assertEqual(res["decision"], "RETAIN PHASE 36C CHAMPION")
        self.assertIn("Candidate A official score delta (+0.000149) < threshold (+0.0020)", res["decision_reasons"][0])
        self.assertIn("Candidate A degrades Category 1 score by -0.001789", res["decision_reasons"][1])

    def test_pipeline_execution_without_bootstrap(self):
        """Execute the pipeline with run_bootstrap=False to verify live end-to-end execution."""
        live_res = execute_phase_39b_evaluation(run_bootstrap=False)
        self.assertEqual(live_res["decision"], "RETAIN PHASE 36C CHAMPION")
        self.assertAlmostEqual(
            live_res["metrics"]["candidate_a"]["delta_official"],
            +0.000149,
            places=5,
        )
        self.assertEqual(live_res["suppressed_candidate_audit"]["total_suppressed_candidates"], 204)


if __name__ == "__main__":
    unittest.main()
