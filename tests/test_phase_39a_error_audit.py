"""
Unit and regression tests for Phase 39A: Complete Error Opportunity and Diagnostic Audit.
Verifies:
1. Results JSON existence and schema.
2. Exact 100% mathematical reconciliation of all 179 FPs and 728 FNs.
3. Invariant: exact TP == 3045, total gold == 3773.
4. Metric comparison: Reconstructed Official F0.2 (0.924945) vs Global Micro F0.2 (0.938333).
5. Category 1 and Category 2 official scores and 50/50 macro average.
6. Top 1 score loss aspect is Kompatibles_Fahrzeug_Modell in both Category 1 and Category 2.
"""

import json
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "data" / "cache" / "phase_39a_error_audit_results.json"


class TestPhase39AErrorAudit(unittest.TestCase):

    def test_results_file_existence(self):
        """Verify results JSON exists."""
        self.assertTrue(RESULTS_PATH.exists(), f"Results JSON missing at {RESULTS_PATH}")

    def test_mathematical_reconciliation_invariants(self):
        """Verify that every FP and FN is accounted for with zero double-counting."""
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        tax = data["error_taxonomy"]
        summary = tax["taxonomy_summary"]

        exact_tp = tax["exact_true_positives"]
        self.assertEqual(exact_tp, 3045)

        # FP reconciliation
        b_fp = summary["boundary_mismatch"]
        a_fp = summary["aspect_mismatch"]
        c_fp = summary["combined_boundary_aspect_mismatch"]
        s_fp = summary["spurious_entity"]

        self.assertEqual(b_fp, 109)
        self.assertEqual(a_fp, 35)
        self.assertEqual(c_fp, 4)
        self.assertEqual(s_fp, 31)
        self.assertEqual(b_fp + a_fp + c_fp + s_fp, 179)
        self.assertEqual(tax["total_false_positives"], 179)

        # FN reconciliation
        m_fn = summary["missing_entity"]
        self.assertEqual(m_fn, 580)
        self.assertEqual(b_fp + a_fp + c_fp + m_fn, 728)
        self.assertEqual(tax["total_false_negatives"], 728)

        # Total gold invariant
        self.assertEqual(exact_tp + tax["total_false_negatives"], 3773)

    def test_official_metric_vs_global_micro_scores(self):
        """Verify that official hierarchical score and global micro score match audit targets."""
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        off = data["official_score_results"]
        micro = data["global_micro_results"]

        # Official score: 0.924945
        self.assertAlmostEqual(off["official_score"], 0.924945, places=5)
        self.assertAlmostEqual(off["categories"]["1"]["score"], 0.915723, places=5)
        self.assertAlmostEqual(off["categories"]["2"]["score"], 0.934168, places=5)
        # Macro average verification
        expected_macro = (off["categories"]["1"]["score"] + off["categories"]["2"]["score"]) / 2.0
        self.assertAlmostEqual(off["official_score"], expected_macro, places=6)

        # Global micro score: 0.938333
        self.assertAlmostEqual(micro["global_f_beta"], 0.938333, places=5)
        self.assertAlmostEqual(micro["global_precision"], 0.944479, places=5)
        self.assertAlmostEqual(micro["global_recall"], 0.807050, places=5)

        # Difference
        diff = micro["global_f_beta"] - off["official_score"]
        self.assertAlmostEqual(diff, 0.013388, places=5)

    def test_top_score_loss_opportunities(self):
        """Verify that Kompatibles_Fahrzeug_Modell is the undisputed #1 opportunity in both categories."""
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        opps = data["ranked_score_loss_opportunities"]
        self.assertGreater(len(opps), 10)

        rank1 = opps[0]
        self.assertEqual(rank1["category"], 1)
        self.assertEqual(rank1["aspect"], "Kompatibles_Fahrzeug_Modell")
        self.assertAlmostEqual(rank1["max_score_loss_final"], 0.02126, places=4)

        rank2 = opps[1]
        self.assertEqual(rank2["category"], 2)
        self.assertEqual(rank2["aspect"], "Kompatibles_Fahrzeug_Modell")
        self.assertAlmostEqual(rank2["max_score_loss_final"], 0.01530, places=4)

        # Combined vehicle model score loss is over 0.036
        combined_vm_loss = rank1["max_score_loss_final"] + rank2["max_score_loss_final"]
        self.assertGreater(combined_vm_loss, 0.036)


if __name__ == "__main__":
    unittest.main()
