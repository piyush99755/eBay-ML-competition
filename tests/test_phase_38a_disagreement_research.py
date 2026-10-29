"""
Unit and regression tests for Phase 38A: Ensemble Disagreement & Prediction Reliability Research.
"""

import hashlib
import json
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "data" / "cache" / "phase_38a_disagreement_results.json"
CHAMPION_TSV_PATH = PROJECT_ROOT / "submissions" / "submission_quiz_phase36c_produktart080.tsv"
EXPECTED_CHAMPION_SHA256 = "9F76C2418C60708BC92486F5F39165021000EE9151DCCC7DD8F75CE8BBA9CC87".lower()


class TestPhase38ADisagreementResearch(unittest.TestCase):
    """Verify Phase 36C baseline reconstruction, agreement tiers, candidate invariants, and champion hash."""

    def test_results_file_existence(self):
        """Verify results JSON exists."""
        self.assertTrue(RESULTS_PATH.exists(), f"Results JSON missing at {RESULTS_PATH}")

    def test_champion_tsv_integrity(self):
        """Verify the Phase 36C champion submission TSV is present and unmodified."""
        self.assertTrue(CHAMPION_TSV_PATH.exists(), f"Champion TSV missing at {CHAMPION_TSV_PATH}")
        hasher = hashlib.sha256()
        with open(CHAMPION_TSV_PATH, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        actual_hash = hasher.hexdigest().lower()
        self.assertEqual(
            actual_hash,
            EXPECTED_CHAMPION_SHA256,
            f"Champion TSV SHA256 mismatch: {actual_hash} vs {EXPECTED_CHAMPION_SHA256}",
        )

    def test_phase_36c_baseline_reconstruction(self):
        """Verify Phase 36C champion baseline reconstructed with 100% mathematical fidelity."""
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        base = data["baseline_phase36c"]["overall"]
        self.assertEqual(base["tp"], 3045)
        self.assertEqual(base["fp"], 179)
        self.assertEqual(base["fn"], 728)
        self.assertEqual(base["tp"] + base["fn"], 3773)
        self.assertAlmostEqual(base["precision"], 0.944479, places=5)
        self.assertAlmostEqual(base["recall"], 0.807050, places=5)
        self.assertAlmostEqual(base["f_beta"], 0.938333, places=5)

        c1 = data["baseline_phase36c"]["category_1"]
        c2 = data["baseline_phase36c"]["category_2"]
        self.assertEqual(c1["tp"], 2028)
        self.assertEqual(c1["fp"], 129)
        self.assertEqual(c1["fn"], 502)
        self.assertAlmostEqual(c1["f_beta"], 0.933983, places=5)

        self.assertEqual(c2["tp"], 1017)
        self.assertEqual(c2["fp"], 50)
        self.assertEqual(c2["fn"], 226)
        self.assertAlmostEqual(c2["f_beta"], 0.947131, places=5)

    def test_raw_agreement_distribution_integrity(self):
        """Verify agreement tier distribution under Raw Argmax BIO mode."""
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw = data["raw_agreement_distribution"]
        self.assertEqual(raw["total_predictions"], 3224)

        tiers = raw["tiers"]
        # Tier 4 (all 4 models)
        t4 = tiers["4"]
        self.assertEqual(t4["total"], 3104)
        self.assertEqual(t4["tp"], 2964)
        self.assertEqual(t4["fp"], 140)
        self.assertGreater(t4["precision"], 0.954)
        self.assertGreater(t4["pct_of_total"], 96.0)

        # Tier 3
        t3 = tiers["3"]
        self.assertEqual(t3["total"], 84)
        self.assertEqual(t3["tp"], 62)
        self.assertEqual(t3["fp"], 22)

        # Tier 2
        t2 = tiers["2"]
        self.assertEqual(t2["total"], 32)
        self.assertEqual(t2["tp"], 17)
        self.assertEqual(t2["fp"], 15)

        # Tier 1
        t1 = tiers["1"]
        self.assertEqual(t1["total"], 4)
        self.assertEqual(t1["tp"], 2)
        self.assertEqual(t1["fp"], 2)

        # Tier 0
        t0 = tiers["0"]
        self.assertEqual(t0["total"], 0)

        # Sum of all tiers must equal 3224
        tot_sum = sum(tiers[str(k)]["total"] for k in range(5))
        self.assertEqual(tot_sum, 3224)

    def test_thresholded_agreement_distribution_integrity(self):
        """Verify agreement tier distribution under Thresholded BIO mode."""
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        thresh = data["thresholded_agreement_distribution"]
        self.assertEqual(thresh["total_predictions"], 3224)

        tiers = thresh["tiers"]
        self.assertEqual(tiers["4"]["total"], 2912)
        self.assertEqual(tiers["4"]["tp"], 2808)
        self.assertEqual(tiers["4"]["fp"], 104)
        self.assertGreater(tiers["4"]["precision"], 0.96)

        self.assertEqual(tiers["3"]["total"], 218)
        self.assertEqual(tiers["2"]["total"], 81)
        self.assertEqual(tiers["1"]["total"], 12)
        self.assertEqual(tiers["0"]["total"], 1)

    def test_disagreement_mechanisms_breakdown(self):
        """Verify breakdown of disagreement mechanisms for partial-agreement predictions."""
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        mech = data["disagreement_mechanisms"]
        b_count = mech["pure_boundary_disagreements"]
        a_count = mech["aspect_classification_mismatches"]
        o_count = mech["omissions_or_mixed"]

        self.assertEqual(b_count, 84)
        self.assertEqual(a_count, 30)
        self.assertEqual(o_count, 6)
        # 84 + 30 + 6 = 120 (which is exactly 3224 total - 3104 4-model agreement)
        self.assertEqual(b_count + a_count + o_count, 120)

    def test_candidate_simulation_invariants(self):
        """Verify ground truth invariants and removal math for all candidates."""
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        candidates = data["candidates"]

        # Raw Candidate A
        raw_a = candidates["raw_candidate_a"]
        self.assertEqual(raw_a["total_removed"], 4)
        self.assertEqual(raw_a["tp_lost"], 2)
        self.assertEqual(raw_a["fp_removed"], 2)
        self.assertEqual(raw_a["metrics"]["overall"]["tp"] + raw_a["metrics"]["overall"]["fn"], 3773)
        # p-value >= 0.05 (statistically insignificant)
        self.assertGreaterEqual(raw_a["bootstrap"]["p_value"], 0.05)
        # CI spans zero
        self.assertLess(raw_a["bootstrap"]["ci_lower_95"], 0.0)

        # Raw Candidate B
        raw_b = candidates["raw_candidate_b"]
        self.assertEqual(raw_b["total_removed"], 1)
        self.assertEqual(raw_b["tp_lost"], 0)
        self.assertEqual(raw_b["fp_removed"], 1)
        self.assertEqual(raw_b["metrics"]["overall"]["tp"] + raw_b["metrics"]["overall"]["fn"], 3773)

        # Thresholded Candidate A
        thresh_a = candidates["thresholded_candidate_a"]
        self.assertEqual(thresh_a["total_removed"], 13)
        self.assertEqual(thresh_a["tp_lost"], 6)
        self.assertEqual(thresh_a["fp_removed"], 7)
        self.assertEqual(thresh_a["metrics"]["overall"]["tp"] + thresh_a["metrics"]["overall"]["fn"], 3773)

        # Thresholded Candidate B
        thresh_b = candidates["thresholded_candidate_b"]
        self.assertEqual(thresh_b["total_removed"], 10)
        self.assertEqual(thresh_b["tp_lost"], 4)
        self.assertEqual(thresh_b["fp_removed"], 6)
        self.assertEqual(thresh_b["metrics"]["overall"]["tp"] + thresh_b["metrics"]["overall"]["fn"], 3773)

    def test_recommendation_is_retain_phase_36c(self):
        """Verify explicit final recommendation."""
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["recommendation"], "RETAIN PHASE 36C CHAMPION")


if __name__ == "__main__":
    unittest.main()
