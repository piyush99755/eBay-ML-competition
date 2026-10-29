"""
Unit tests for Phase 36B: Aspect-Specific Confidence Calibration Research Utilities and Results Integrity.
"""

import json
import unittest
from pathlib import Path
import numpy as np

from src.evaluation.phase_36b_aspect_calibration import (
    summarize_distribution,
    analyze_aspect_confidence_separation,
    TARGET_ASPECTS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_DATA_PATH = PROJECT_ROOT / "data" / "cache" / "phase_36b_calibration_results.json"


class TestPhase36BAspectCalibration(unittest.TestCase):
    """Verify distribution summarization, evaluation invariants, and calibration results."""

    def test_summarize_distribution_empty(self):
        """Verify summary on empty input."""
        res = summarize_distribution([])
        self.assertEqual(res["n"], 0)
        self.assertEqual(res["mean"], 0.0)
        self.assertEqual(res["median"], 0.0)

    def test_summarize_distribution_values(self):
        """Verify summary on known values."""
        vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        res = summarize_distribution(vals)
        self.assertEqual(res["n"], 10)
        self.assertAlmostEqual(res["mean"], 0.55, places=5)
        self.assertAlmostEqual(res["median"], 0.55, places=5)
        self.assertAlmostEqual(res["min"], 0.1, places=5)
        self.assertAlmostEqual(res["max"], 1.0, places=5)

    def test_synthetic_aspect_separation(self):
        """Verify separation logic on controlled synthetic examples."""
        gold = [
            [{"aspect": "Produktart", "value": "Bremsscheiben", "start": 0, "end": 1}],
            [{"aspect": "Produktart", "value": "Bremsbelag", "start": 0, "end": 1}],
        ]
        # Example 0: correct match with high conf 0.95
        # Example 1: incorrect prediction "Satz" with lower conf 0.65
        preds = [
            [{"aspect": "Produktart", "value": "Bremsscheiben", "min_conf": 0.95}],
            [{"aspect": "Produktart", "value": "Satz", "min_conf": 0.65}],
        ]
        exs = [{"category": 1}, {"category": 1}]

        res = analyze_aspect_confidence_separation(gold, preds, exs, "Produktart")
        self.assertEqual(res["gold_count"], 2)
        self.assertEqual(res["pred_count"], 2)
        self.assertEqual(res["tp"], 1)
        self.assertEqual(res["fp"], 1)
        self.assertEqual(res["fn"], 1)
        self.assertAlmostEqual(res["precision"], 0.5, places=5)
        self.assertAlmostEqual(res["recall"], 0.5, places=5)
        self.assertEqual(res["tp_distribution"]["n"], 1)
        self.assertAlmostEqual(res["tp_distribution"]["mean"], 0.95, places=5)
        self.assertEqual(res["fp_distribution"]["n"], 1)
        self.assertAlmostEqual(res["fp_distribution"]["mean"], 0.65, places=5)
        self.assertEqual(res["mann_whitney_auc"], 1.0)

    def test_calibration_results_file_integrity(self):
        """Verify results JSON exists, matches baseline invariant, and confirms threshold behaviors."""
        self.assertTrue(CALIBRATION_DATA_PATH.exists(), "Results JSON missing!")
        with open(CALIBRATION_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        base = data["baseline_champion"]["overall"]
        self.assertEqual(base["tp"], 3072)
        self.assertEqual(base["fp"], 199)
        self.assertEqual(base["fn"], 701)
        self.assertEqual(base["tp"] + base["fn"], 3773)
        self.assertAlmostEqual(base["f_beta"], 0.933651, places=5)

        sims = data["threshold_simulations"]

        # Invariant check: for every simulation, TP + FN == 3773
        for k, sim in sims.items():
            self.assertEqual(sim["tp"] + sim["fn"], 3773, f"Invariant violated for {k}")
            self.assertEqual(sim["preds_removed"], sim["tp_lost"] + sim["fp_removed"], f"Removal mismatch for {k}")

        # Kompatible_Fahrzeug_Marke at 0.70 and 0.80 regresses Category 2
        self.assertLess(sims["Kompatible_Fahrzeug_Marke_th70"]["delta_cat2_f02"], 0.0)
        self.assertLess(sims["Kompatible_Fahrzeug_Marke_th80"]["delta_cat2_f02"], 0.0)

        # Produktart at 0.80 has highest overall F0.2 and strictly positive 95% CI
        pa_80 = sims["Produktart_th80"]
        self.assertGreater(pa_80["delta_f02"], 0.004)
        self.assertGreater(pa_80["ci_lower_95"], 0.0)
        self.assertGreater(pa_80["delta_cat1_f02"], 0.0)
        self.assertGreater(pa_80["delta_cat2_f02"], 0.0)

        # Produktart at 0.80 removes proportionally more FPs than TPs
        sep = data["separation_analysis"]["Produktart"]
        fp_removed_prop = pa_80["fp_removed"] / sep["fp"]
        tp_lost_prop = pa_80["tp_lost"] / sep["tp"]
        self.assertGreater(fp_removed_prop, tp_lost_prop)


if __name__ == "__main__":
    unittest.main()
