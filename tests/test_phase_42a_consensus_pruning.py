"""
Unit and regression tests for Phase 42A: Model Consensus Audit & Controlled Precision Pruning.
"""

import hashlib
import json
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "data" / "cache" / "phase_42a_consensus_pruning_results.json"
CHAMPION_TSV = PROJECT_ROOT / "submissions" / "submission_quiz_phase40_lieferumfang080.tsv"
EXPECTED_CHAMPION_SHA256 = "BF125FF2F4D7CBA79636739CF4C7FF66C7F42A4462531245E3104045961461E8"
OFFICIAL_QUIZ_SCORE_PHASE40 = 0.9233317116970503


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


class TestPhase42AConsensusPruning(unittest.TestCase):
    """Test suite for Phase 42A Consensus Audit & Pruning."""

    def test_01_champion_verification(self):
        """Verify Phase 40 champion file existence and cryptographic digest."""
        self.assertTrue(CHAMPION_TSV.exists(), f"Champion TSV missing: {CHAMPION_TSV}")
        actual_hash = compute_sha256(CHAMPION_TSV)
        self.assertEqual(actual_hash, EXPECTED_CHAMPION_SHA256, "Champion hash mismatch!")

    def test_02_baseline_reproduction(self):
        """Verify reproduction of local Phase 40 baseline on 489 clean holdout titles."""
        self.assertTrue(RESULTS_PATH.exists(), f"Results cache missing: {RESULTS_PATH}")
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        base = data["baseline_reproduction"]
        self.assertAlmostEqual(base["official_score"], 0.927771, delta=1e-4)
        self.assertAlmostEqual(base["cat1_score"], 0.918323, delta=1e-4)
        self.assertAlmostEqual(base["cat2_score"], 0.937219, delta=1e-4)
        self.assertEqual(base["tp"], 3021)
        self.assertEqual(base["fp"], 165)
        self.assertEqual(base["fn"], 752)

    def test_03_inconsistency_resolution(self):
        """Verify resolution of Phase 41 6-FP vs 11-removed inconsistency."""
        self.assertTrue(RESULTS_PATH.exists())
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        audit = data["inconsistency_audit"]
        self.assertEqual(audit["total_phase40_predictions"], 3186)
        self.assertEqual(audit["exact_support_1_count"], 12)
        self.assertEqual(audit["exact_support_1_tp"], 6)
        self.assertEqual(audit["exact_support_1_fp"], 6)
        self.assertEqual(audit["fp_exact_support_1_count"], 6)

    def test_04_candidate_metrics_and_invariants(self):
        """Verify Candidate A and Candidate B counts, subset properties, and metrics."""
        self.assertTrue(RESULTS_PATH.exists())
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Candidate A
        cand_a = data["candidate_a"]
        self.assertEqual(cand_a["total_removed"], 12)
        self.assertEqual(cand_a["tp_removed"], 6)
        self.assertEqual(cand_a["fp_removed"], 6)
        self.assertEqual(cand_a["affected_titles"], 12)
        self.assertAlmostEqual(cand_a["official_score"], 0.929526, delta=1e-4)
        self.assertAlmostEqual(cand_a["delta_official"], 0.001755, delta=1e-4)
        self.assertLess(cand_a["delta_official"], 0.0020, "Candidate A should not exceed +0.0020 gain threshold")

        # Candidate B
        cand_b = data["candidate_b"]
        self.assertEqual(cand_b["total_removed"], 9)
        self.assertEqual(cand_b["tp_removed"], 4)
        self.assertEqual(cand_b["fp_removed"], 5)
        self.assertEqual(cand_b["affected_titles"], 9)
        self.assertAlmostEqual(cand_b["official_score"], 0.929354, delta=1e-4)
        self.assertAlmostEqual(cand_b["delta_official"], 0.001584, delta=1e-4)
        self.assertLess(cand_b["delta_official"], 0.0020, "Candidate B should not exceed +0.0020 gain threshold")

        # Invariant: Candidate A and B removed True Positives become False Negatives
        self.assertEqual(cand_a["fn"], data["baseline_reproduction"]["fn"] + cand_a["tp_removed"])
        self.assertEqual(cand_b["fn"], data["baseline_reproduction"]["fn"] + cand_b["tp_removed"])

    def test_05_decision_retains_champion(self):
        """Verify decision logic correctly retains Phase 40 champion due to TP loss and insufficient gain."""
        self.assertTrue(RESULTS_PATH.exists())
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        verdict = data["decision"]["verdict"]
        self.assertEqual(verdict, "RETAIN PHASE 40 CHAMPION")


if __name__ == "__main__":
    unittest.main()
