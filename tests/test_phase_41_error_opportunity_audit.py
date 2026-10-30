"""
Unit and regression tests for Phase 41: Champion Promotion & Official-Metric Error Opportunity Audit.

Tests:
1. Champion Promotion & Fallback Verification:
   - Phase 40 Quiz TSV exists and matches expected SHA-256 (BF125FF2F4D7CBA79636739CF4C7FF66C7F42A4462531245E3104045961461E8).
   - Phase 36C fallback TSV exists and matches expected SHA-256 (9F76C2418C60708BC92486F5F39165021000EE9151DCCC7DD8F75CE8BBA9CC87).
   - Registry integrity: data/cache/champion_registry.json correctly documents Phase 40 as active champion.
   - All 4 ensemble model checkpoints exist.
2. Baseline Reproduction Invariants:
   - Clean holdout size: 489 titles.
   - Overall Official Score: 0.927771 +/- 1e-4.
   - Category 1 Score: 0.918323 +/- 1e-4.
   - Category 2 Score: 0.937219 +/- 1e-4.
   - Global Micro Metrics: TP = 3021, FP = 165, FN = 752.
3. Opportunity Table Invariants:
   - Exactly 29 official aspects covered.
   - Sum of aspect opportunities equals (1.0 - official_score) within 1e-6 (0.072229).
   - Kompatibles_Fahrzeug_Modell is the top deficit aspect (> 0.035, > 50% of total deficit).
   - Category 1 + Category 2 averaging matches: overall_opportunity == 0.5 * (cat1_deficit + cat2_deficit).
4. Error Taxonomy & Audit Invariants:
   - False Positives total: exactly 165.
   - FP error types sum to 165 (boundary_mismatch + aspect_mismatch + spurious == 165).
   - Spurious FPs <= 5 (verified 4).
   - False Negatives total: exactly 752.
   - FN types sum to 752 (missing_entity + boundary_mismatch + aspect_mismatch == 752).
"""

import hashlib
import json
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "data" / "cache" / "champion_registry.json"
RESULTS_PATH = PROJECT_ROOT / "data" / "cache" / "phase_41_audit_results.json"

PHASE40_TSV = PROJECT_ROOT / "submissions" / "submission_quiz_phase40_lieferumfang080.tsv"
EXPECTED_PHASE40_SHA256 = "BF125FF2F4D7CBA79636739CF4C7FF66C7F42A4462531245E3104045961461E8"
OFFICIAL_QUIZ_SCORE_PHASE40 = 0.9233317116970503

PHASE36C_TSV = PROJECT_ROOT / "submissions" / "submission_quiz_phase36c_produktart080.tsv"
EXPECTED_PHASE36C_SHA256 = "9F76C2418C60708BC92486F5F39165021000EE9151DCCC7DD8F75CE8BBA9CC87"
OFFICIAL_QUIZ_SCORE_PHASE36C = 0.9206543463689059

CHECKPOINT_PATHS = [
    PROJECT_ROOT / "checkpoints" / "phase33b_dapt_gbert_best",
    PROJECT_ROOT / "checkpoints" / "xlmr_baseline_best",
    PROJECT_ROOT / "checkpoints" / "xlmr_seed_7_best",
    PROJECT_ROOT / "checkpoints" / "xlmr_seed_21_best",
]


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


class TestPhase41ErrorOpportunityAudit(unittest.TestCase):
    """Regression and integrity test suite for Phase 41."""

    def test_01_champion_and_fallback_integrity(self):
        """Verify Phase 40 active champion and Phase 36C fallback files and SHA-256 hashes."""
        self.assertTrue(PHASE40_TSV.exists(), f"Phase 40 submission TSV missing: {PHASE40_TSV}")
        self.assertEqual(
            compute_sha256(PHASE40_TSV),
            EXPECTED_PHASE40_SHA256,
            "Phase 40 TSV SHA-256 mismatch!",
        )

        self.assertTrue(PHASE36C_TSV.exists(), f"Phase 36C submission TSV missing: {PHASE36C_TSV}")
        self.assertEqual(
            compute_sha256(PHASE36C_TSV),
            EXPECTED_PHASE36C_SHA256,
            "Phase 36C TSV SHA-256 mismatch!",
        )

        for cp in CHECKPOINT_PATHS:
            self.assertTrue(cp.exists() and cp.is_dir(), f"Model checkpoint directory missing: {cp}")

    def test_02_champion_registry_integrity(self):
        """Verify that champion_registry.json correctly marks Phase 40 as active champion."""
        self.assertTrue(REGISTRY_PATH.exists(), f"Registry file missing: {REGISTRY_PATH}")
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            reg = json.load(f)

        self.assertEqual(reg["active_champion"]["phase"], "Phase 40")
        self.assertEqual(reg["active_champion"]["sha256"], EXPECTED_PHASE40_SHA256)
        self.assertAlmostEqual(
            reg["active_champion"]["official_quiz_f02"],
            OFFICIAL_QUIZ_SCORE_PHASE40,
            places=6,
        )

        self.assertIn("historical_fallbacks", reg)
        p36c = next((f for f in reg["historical_fallbacks"] if f["phase"] == "Phase 36C"), None)
        self.assertIsNotNone(p36c, "Phase 36C fallback not found in registry")
        self.assertEqual(p36c["sha256"], EXPECTED_PHASE36C_SHA256)
        self.assertAlmostEqual(
            p36c["official_quiz_f02"],
            OFFICIAL_QUIZ_SCORE_PHASE36C,
            places=6,
        )

    def test_03_baseline_reproduction(self):
        """Verify reproduction of the Phase 40 local baseline metrics on the clean holdout."""
        self.assertTrue(RESULTS_PATH.exists(), f"Audit results cache missing: {RESULTS_PATH}")
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        base = data["baseline_reproduction"]
        self.assertAlmostEqual(base["official_score"], 0.927771, delta=1e-4)
        self.assertAlmostEqual(base["cat1_score"], 0.918323, delta=1e-4)
        self.assertAlmostEqual(base["cat2_score"], 0.937219, delta=1e-4)
        self.assertEqual(base["tp"], 3021)
        self.assertEqual(base["fp"], 165)
        self.assertEqual(base["fn"], 752)

    def test_04_opportunity_table_invariants(self):
        """Verify mathematical consistency and invariants of the 29-aspect opportunity table."""
        self.assertTrue(RESULTS_PATH.exists())
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        opp_table = data["opportunity_table"]
        self.assertEqual(len(opp_table), 29, "Opportunity table must contain all 29 official aspects.")

        total_opp = sum(item["overall_opportunity"] for item in opp_table)
        expected_deficit = 1.0 - data["baseline_reproduction"]["official_score"]
        self.assertAlmostEqual(total_opp, expected_deficit, places=5)

        # Check individual aspect calculation invariant: 0.5 * (cat1 + cat2)
        for item in opp_table:
            c1_def = item["cat1"]["weighted_deficit"]
            c2_def = item["cat2"]["weighted_deficit"]
            expected_item_opp = 0.5 * (c1_def + c2_def)
            self.assertAlmostEqual(item["overall_opportunity"], expected_item_opp, places=6)

        # Check top aspect is Kompatibles_Fahrzeug_Modell
        top_aspect = opp_table[0]
        self.assertEqual(top_aspect["aspect"], "Kompatibles_Fahrzeug_Modell")
        self.assertGreater(top_aspect["overall_opportunity"], 0.035)

    def test_05_false_positive_and_negative_audit(self):
        """Verify error taxonomy sums and properties."""
        self.assertTrue(RESULTS_PATH.exists())
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        fp_audit = data["false_positive_audit"]
        self.assertEqual(fp_audit["total_fp"], 165)

        by_type = fp_audit["by_error_type"]
        self.assertEqual(
            by_type.get("boundary_mismatch", 0)
            + by_type.get("aspect_mismatch", 0)
            + by_type.get("spurious", 0),
            165,
        )
        self.assertLessEqual(by_type.get("spurious", 0), 5)
        self.assertGreater(fp_audit["confidence_stats"]["mean"], 0.90)

        fn_audit = data["false_negative_audit"]
        self.assertEqual(fn_audit["total_fn"], 752)
        fn_by_type = fn_audit["by_fn_type"]
        self.assertEqual(
            fn_by_type.get("missing_entity", 0)
            + fn_by_type.get("boundary_mismatch", 0)
            + fn_by_type.get("aspect_mismatch", 0),
            752,
        )


if __name__ == "__main__":
    unittest.main()
