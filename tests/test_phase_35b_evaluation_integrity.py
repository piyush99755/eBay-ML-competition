"""
Unit tests for Phase 35B: Evaluation Integrity, Holdout Verification, and Metric Mathematics.
"""

import hashlib
import json
import unittest
from pathlib import Path

from src.evaluation.entity_metrics import (
    count_matches,
    precision_recall_fbeta,
    BETA,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOLDOUT_MANIFEST_PATH = PROJECT_ROOT / "data" / "cache" / "phase_35b_clean_holdout_manifest.json"
AUDIT_RESULTS_PATH = PROJECT_ROOT / "data" / "cache" / "phase_35b_audit_results.json"


class TestEvaluationIntegrity(unittest.TestCase):
    """Verify entity evaluation mathematics and argument ordering."""

    def test_count_matches_argument_order(self):
        """
        Verify that count_matches(predicted, gold) correctly computes TP, FP, FN,
        and that count_matches(gold, predicted) inverts FP and FN.
        """
        gold = [("Produktart", "Bremsscheiben"), ("Hersteller", "Bosch")]
        pred = [("Produktart", "Bremsscheiben"), ("Produktart", "Bremsbelag")]

        # Correct call: (pred, gold)
        tp_c, fp_c, fn_c = count_matches(pred, gold)
        self.assertEqual(tp_c, 1)  # Bremsscheiben matches
        self.assertEqual(fp_c, 1)  # Bremsbelag is false positive
        self.assertEqual(fn_c, 1)  # Bosch is false negative
        self.assertEqual(tp_c + fn_c, len(gold))  # Must equal total gold entities

        # Inverted call: (gold, pred)
        tp_i, fp_i, fn_i = count_matches(gold, pred)
        self.assertEqual(tp_i, 1)  # TP is symmetric
        self.assertEqual(fp_i, 1)  # Was FN_true
        self.assertEqual(fn_i, 1)  # Was FP_true

        # Unequal counts to prove inversion:
        gold_multi = [("A", "1"), ("A", "2"), ("A", "3")]
        pred_multi = [("A", "1")]

        tp_c, fp_c, fn_c = count_matches(pred_multi, gold_multi)
        self.assertEqual(tp_c, 1)
        self.assertEqual(fp_c, 0)
        self.assertEqual(fn_c, 2)
        self.assertEqual(tp_c + fn_c, 3)

        tp_i, fp_i, fn_i = count_matches(gold_multi, pred_multi)
        self.assertEqual(tp_i, 1)
        self.assertEqual(fp_i, 2)  # Swapped: reports 2 FPs when there are 0 FPs!
        self.assertEqual(fn_i, 0)  # Swapped: reports 0 FNs when there are 2 FNs!

    def test_precision_recall_fbeta_inversion_proof(self):
        """
        Mathematical proof that swapping P and R under beta=0.2 computes F_5.
        """
        p = 0.90
        r = 0.80
        beta = 0.2

        # Standard F_beta formula: (1 + beta^2) * (P * R) / (beta^2 * P + R)
        f_02_direct = (1.0 + beta**2) * (p * r) / (beta**2 * p + r)

        # Swapped P and R with beta=0.2
        f_swapped = (1.0 + beta**2) * (r * p) / (beta**2 * r + p)

        # F_5 with original P and R: beta=5.0
        beta_inv = 1.0 / beta  # 5.0
        f_5_direct = (1.0 + beta_inv**2) * (p * r) / (beta_inv**2 * p + r)

        self.assertAlmostEqual(f_swapped, f_5_direct, places=7)
        self.assertNotEqual(f_02_direct, f_swapped)

    def test_holdout_manifest_existence_and_integrity(self):
        """Verify the 489 clean holdout manifest."""
        self.assertTrue(HOLDOUT_MANIFEST_PATH.exists(), "Manifest file missing!")
        with open(HOLDOUT_MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(manifest["record_count"], 489)
        self.assertEqual(len(manifest["record_ids"]), 489)
        self.assertEqual(manifest["total_gold_entities"], 3773)

        # Verify checksum of record IDs
        ids_str = ",".join(str(i) for i in sorted(manifest["record_ids"]))
        computed_hash = hashlib.sha256(ids_str.encode("utf-8")).hexdigest()
        self.assertEqual(manifest["checksums"]["record_ids_sha256"], computed_hash)

        # Verify zero leakage recorded in manifest
        self.assertTrue(manifest["verification"]["record_id_leakage"])
        self.assertTrue(manifest["verification"]["exact_title_match_leakage"])
        self.assertTrue(manifest["verification"]["normalized_title_match_leakage"])

    def test_audit_results_tp_plus_fn_invariant(self):
        """
        Verify that across all 5 evaluated systems, TP + FN == 3,773 (total gold entities).
        """
        self.assertTrue(AUDIT_RESULTS_PATH.exists(), "Audit results file missing!")
        with open(AUDIT_RESULTS_PATH, "r", encoding="utf-8") as f:
            res = json.load(f)

        for sys_name, d in res["verified_metrics_holdout"].items():
            ov = d["overall"]
            tp = ov["tp"]
            fn = ov["fn"]
            self.assertEqual(
                tp + fn, 3773,
                f"Violation for {sys_name}: TP={tp} + FN={fn} = {tp+fn} != 3773"
            )

    def test_model_ranking_preserved(self):
        """
        Verify that under corrected mathematics:
        DAPT 4-Model > Five-Model > Original 4-Model.
        """
        with open(AUDIT_RESULTS_PATH, "r", encoding="utf-8") as f:
            res = json.load(f)

        dapt_f = res["verified_metrics_holdout"]["dapt_champ_4m"]["overall"]["f_beta"]
        five_f = res["verified_metrics_holdout"]["five_model_5m"]["overall"]["f_beta"]
        orig_f = res["verified_metrics_holdout"]["orig_champ_4m"]["overall"]["f_beta"]

        self.assertGreater(dapt_f, five_f)
        self.assertGreater(five_f, orig_f)


if __name__ == "__main__":
    unittest.main()
