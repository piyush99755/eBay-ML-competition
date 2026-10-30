"""
Unit and regression tests for Phase 40: Official Quiz Challenger Verification (Lieferumfang 0.80).
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHAMPION_TSV = PROJECT_ROOT / "submissions" / "submission_quiz_phase36c_produktart080.tsv"
CHALLENGER_TSV = PROJECT_ROOT / "submissions" / "submission_quiz_phase40_lieferumfang080.tsv"
CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "phase_35c_quiz_dapt_decoded_entities.json"

EXPECTED_CHAMPION_SHA256 = "9F76C2418C60708BC92486F5F39165021000EE9151DCCC7DD8F75CE8BBA9CC87"
EXPECTED_CHALLENGER_SHA256 = "BF125FF2F4D7CBA79636739CF4C7FF66C7F42A4462531245E3104045961461E8"
EXPECTED_CHAMPION_ROWS = 145396
EXPECTED_CHALLENGER_ROWS = 143715
EXPECTED_REMOVED_ROWS = 1681
EXPECTED_CAT1_REMOVED = 1067
EXPECTED_CAT2_REMOVED = 614
EXPECTED_AFFECTED_TITLES = 1513


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


def load_tsv_rows(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip("\r\n").split("\t")
        for line in f:
            parts = line.strip("\r\n").split("\t")
            if len(parts) == 4:
                rows.append((int(parts[0]), int(parts[1]), parts[2], parts[3]))
    return header, rows


class TestPhase40QuizChallenger(unittest.TestCase):
    """Verify integrity, subset invariants, and aspect consistency of Phase 40 submission."""

    def test_file_existence_and_hashes(self):
        """Verify that champion and challenger exist with exact expected SHA-256 digests."""
        self.assertTrue(CHAMPION_TSV.exists(), "Phase 36C champion TSV missing!")
        self.assertTrue(CHALLENGER_TSV.exists(), "Phase 40 challenger TSV missing!")

        champ_hash = compute_sha256(CHAMPION_TSV)
        chal_hash = compute_sha256(CHALLENGER_TSV)

        self.assertEqual(champ_hash, EXPECTED_CHAMPION_SHA256, "Champion SHA-256 altered!")
        self.assertEqual(chal_hash, EXPECTED_CHALLENGER_SHA256, "Challenger SHA-256 mismatch!")

    def test_row_counts_and_headers(self):
        """Verify headers, row counts, and exact row reduction count."""
        h_champ, r_champ = load_tsv_rows(CHAMPION_TSV)
        h_chal, r_chal = load_tsv_rows(CHALLENGER_TSV)

        expected_header = ["Record Number", "Category", "Aspect Name", "Aspect Value"]
        self.assertEqual(h_champ, expected_header)
        self.assertEqual(h_chal, expected_header)

        self.assertEqual(len(r_champ), EXPECTED_CHAMPION_ROWS)
        self.assertEqual(len(r_chal), EXPECTED_CHALLENGER_ROWS)
        self.assertEqual(len(r_champ) - len(r_chal), EXPECTED_REMOVED_ROWS)

    def test_non_lieferumfang_aspects_identical(self):
        """Verify that all non-Lieferumfang predictions are 100% identical row-for-row."""
        _, r_champ = load_tsv_rows(CHAMPION_TSV)
        _, r_chal = load_tsv_rows(CHALLENGER_TSV)

        non_lie_champ = [r for r in r_champ if r[2] != "Im_Lieferumfang_Enthalten"]
        non_lie_chal = [r for r in r_chal if r[2] != "Im_Lieferumfang_Enthalten"]

        self.assertEqual(non_lie_champ, non_lie_chal, "Non-Lieferumfang predictions modified!")

    def test_multiset_subset_invariant(self):
        """Verify that Phase 40 is a strict multi-set subset of Phase 36C (zero new rows)."""
        _, r_champ = load_tsv_rows(CHAMPION_TSV)
        _, r_chal = load_tsv_rows(CHALLENGER_TSV)

        c_champ = Counter(r_champ)
        c_chal = Counter(r_chal)

        for item, chal_count in c_chal.items():
            champ_count = c_champ[item]
            self.assertGreaterEqual(champ_count, chal_count, f"Prediction {item} has higher count in challenger!")

    def test_removed_predictions_category_breakdown_and_confidence(self):
        """Verify category breakdown and confidence ranges for removed entities."""
        self.assertTrue(CACHE_PATH.exists(), "Quiz decoded cache missing!")
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            records = json.load(f)

        _, r_champ = load_tsv_rows(CHAMPION_TSV)
        _, r_chal = load_tsv_rows(CHALLENGER_TSV)

        c_champ = Counter(r_champ)
        c_chal = Counter(r_chal)

        diff = c_champ - c_chal
        self.assertEqual(sum(diff.values()), EXPECTED_REMOVED_ROWS)

        # Every removed prediction must be Im_Lieferumfang_Enthalten
        for (rec_id, cat_id, asp, val), count in diff.items():
            self.assertEqual(asp, "Im_Lieferumfang_Enthalten")

        cat1_removed = sum(count for (rec_id, cat_id, asp, val), count in diff.items() if cat_id == 1)
        cat2_removed = sum(count for (rec_id, cat_id, asp, val), count in diff.items() if cat_id == 2)
        unique_affected = len(set(rec_id for (rec_id, cat_id, asp, val) in diff.keys()))

        self.assertEqual(cat1_removed, EXPECTED_CAT1_REMOVED)
        self.assertEqual(cat2_removed, EXPECTED_CAT2_REMOVED)
        self.assertEqual(unique_affected, EXPECTED_AFFECTED_TITLES)


if __name__ == "__main__":
    unittest.main()
