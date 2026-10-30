"""
Unit and regression tests for Phase 42B: Training-Derived Vehicle-Model Boundary Repair.
"""

import hashlib
import json
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "data" / "cache" / "phase_42b_boundary_repair_results.json"
CHAMPION_TSV = PROJECT_ROOT / "submissions" / "submission_quiz_phase40_lieferumfang080.tsv"
EXPECTED_CHAMPION_SHA256 = "BF125FF2F4D7CBA79636739CF4C7FF66C7F42A4462531245E3104045961461E8"
OFFICIAL_QUIZ_SCORE_PHASE40 = 0.9233317116970503


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


class TestPhase42BVehicleBoundaryRepair(unittest.TestCase):
    """Test suite for Phase 42B Vehicle Boundary Repair."""

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
        self.assertEqual(base["vm_tp"], 473)
        self.assertEqual(base["vm_fp"], 74)
        self.assertEqual(base["vm_fn"], 451)

    def test_03_error_taxonomy_sums(self):
        """Verify recomputed vehicle-model error classifications sum to exactly 74 FPs."""
        self.assertTrue(RESULTS_PATH.exists())
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        tax = data["error_taxonomy"]
        self.assertEqual(tax["total_vm_fp"], 74)
        total_tax_counts = sum(tax["counts"].values())
        self.assertEqual(total_tax_counts, 74)
        self.assertEqual(tax["counts"].get("5. Two entities incorrectly merged", 0), 33)
        self.assertEqual(tax["counts"].get("2. Missing trailing token", 0), 25)

    def test_04_training_kb_integrity(self):
        """Verify training knowledge base is populated and non-empty."""
        self.assertTrue(RESULTS_PATH.exists())
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        kb = data["knowledge_base"]
        self.assertGreater(kb["total_vm_entities"], 3000)
        self.assertGreater(kb["unique_vm_strings"], 2000)

    def test_05_candidate_evaluations_and_invariants(self):
        """Verify candidate repair rules, accounting, and invariants."""
        self.assertTrue(RESULTS_PATH.exists())
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Candidate A: Safe Trailing Expansion
        cand_a = data["candidate_a"]
        self.assertEqual(cand_a["modifications_count"], 0)
        self.assertAlmostEqual(cand_a["delta_official"], 0.0, places=5)

        # Candidate B: Safe Trailing Trimming
        cand_b = data["candidate_b"]
        self.assertEqual(cand_b["modifications_count"], 17)
        self.assertLess(cand_b["delta_official"], 0.0, "Trimming should regress score")
        self.assertAlmostEqual(cand_b["delta_official"], -0.004576, delta=1e-4)

        # Multiset accounting: 14 of 17 trims convert TP to FP!
        acct = cand_b["accounting"]
        self.assertEqual(acct["converts_tp_to_fp"], 14)
        self.assertEqual(acct["converts_fp_to_tp"], 2)

        # Candidate C: Merge/Split
        cand_c = data["candidate_c"]
        self.assertEqual(cand_c["status"], "NOT FEASIBLE")

    def test_06_decision_retains_champion(self):
        """Verify decision logic correctly retains Phase 40 champion."""
        self.assertTrue(RESULTS_PATH.exists())
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        verdict = data["decision"]["verdict"]
        self.assertEqual(verdict, "RETAIN PHASE 40 CHAMPION")


if __name__ == "__main__":
    unittest.main()
