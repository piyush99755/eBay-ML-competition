"""
Unit Tests for Independent Final Submission Validator
eBay 2026 University Machine Learning Competition

Tests:
1. Valid submission passes validation.
2. Missing or incorrect header fails.
3. Row with incorrect field count (e.g. 3 or 5 fields) fails.
4. Invalid Category ID (e.g. 0, 3, 'Cat1') fails.
5. 'O' (Obscure) tag submitted as aspect name fails.
6. Unknown aspect name fails.
7. Category-aspect mismatch (e.g. Cat 1 aspect under Cat 2, or Cat 2 aspect under Cat 1) fails.
8. Empty or whitespace-only aspect value fails.
9. Accidental CSV quote wrapping around aspect value fails.
10. German Unicode characters (ä, ö, ü, ß, Ä, Ö, Ü) preserved and accepted.
11. Multi-aspect records and duplicate entity mentions in title handled properly.
12. Zero-aspect titles: titles without entities produce 0 rows without failing validation.
13. Test reference cross-check: record ID not in test reference dataset fails.
14. Test reference category mismatch: record Category in submission != test reference fails.
15. Empty file (0 bytes) fails gracefully.
"""

import tempfile
import unittest
from pathlib import Path

from src.submission.validate_final_submission import (
    OFFICIAL_HEADER_LINE,
    validate_submission,
)


class TestFinalSubmissionValidator(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_tsv(self, filename: str, content: str) -> Path:
        filepath = self.temp_path / filename
        with open(filepath, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        return filepath

    def test_valid_submission(self):
        content = (
            f"{OFFICIAL_HEADER_LINE}\n"
            "1001\t1\tHersteller\tBOSCH\n"
            "1001\t1\tProduktart\tBremsscheiben\n"
            "1001\t1\tEinbauposition\tVorderachse links\n"
            "1002\t2\tHersteller\tCastrol\n"
            "1002\t2\tSAE_Viskosität\t5W-30\n"
            "1002\t2\tMenge\t5 L\n"
        )
        sub_file = self._create_tsv("valid.tsv", content)
        is_valid, report = validate_submission(sub_file)
        self.assertTrue(is_valid, f"Expected valid, got errors: {report['errors']}")
        self.assertEqual(report["data_rows"], 6)
        self.assertEqual(report["unique_records_with_entities"], 2)
        self.assertEqual(len(report["errors"]), 0)

    def test_missing_or_corrupt_header(self):
        # 1. Missing header
        content = "1001\t1\tHersteller\tBOSCH\n"
        sub_file = self._create_tsv("no_header.tsv", content)
        is_valid, report = validate_submission(sub_file)
        self.assertFalse(is_valid)
        self.assertTrue(any("header" in e.lower() for e in report["errors"]))

        # 2. Corrupt header
        content = "Record_ID\tCategory\tAspect\tValue\n1001\t1\tHersteller\tBOSCH\n"
        sub_file2 = self._create_tsv("corrupt_header.tsv", content)
        is_valid2, report2 = validate_submission(sub_file2)
        self.assertFalse(is_valid2)
        self.assertTrue(any("header" in e.lower() for e in report2["errors"]))

    def test_wrong_field_count(self):
        # 3 fields instead of 4
        content = (
            f"{OFFICIAL_HEADER_LINE}\n"
            "1001\t1\tHersteller\n"
        )
        sub_file = self._create_tsv("too_few_fields.tsv", content)
        is_valid, report = validate_submission(sub_file)
        self.assertFalse(is_valid)
        self.assertTrue(any("expected exactly 4" in e.lower() for e in report["errors"]))

        # 5 fields instead of 4
        content2 = (
            f"{OFFICIAL_HEADER_LINE}\n"
            "1001\t1\tHersteller\tBOSCH\tExtraField\n"
        )
        sub_file2 = self._create_tsv("too_many_fields.tsv", content2)
        is_valid2, report2 = validate_submission(sub_file2)
        self.assertFalse(is_valid2)
        self.assertTrue(any("expected exactly 4" in e.lower() for e in report2["errors"]))

    def test_invalid_category_id(self):
        content = (
            f"{OFFICIAL_HEADER_LINE}\n"
            "1001\t3\tHersteller\tBOSCH\n"
        )
        sub_file = self._create_tsv("invalid_cat.tsv", content)
        is_valid, report = validate_submission(sub_file)
        self.assertFalse(is_valid)
        self.assertTrue(any("category must be 1 or 2" in e.lower() for e in report["errors"]))

    def test_o_tag_rejection(self):
        content = (
            f"{OFFICIAL_HEADER_LINE}\n"
            "1001\t1\tO\tBOSCH\n"
        )
        sub_file = self._create_tsv("o_tag.tsv", content)
        is_valid, report = validate_submission(sub_file)
        self.assertFalse(is_valid)
        self.assertTrue(any("'o' (obscure) tag" in e.lower() for e in report["errors"]))

    def test_unknown_aspect_name(self):
        content = (
            f"{OFFICIAL_HEADER_LINE}\n"
            "1001\t1\tFakeAspectName\tBOSCH\n"
        )
        sub_file = self._create_tsv("unknown_aspect.tsv", content)
        is_valid, report = validate_submission(sub_file)
        self.assertFalse(is_valid)
        self.assertTrue(any("unknown aspect name" in e.lower() for e in report["errors"]))

    def test_category_aspect_mismatch(self):
        # Category 1 exclusive aspect (Bremsscheiben-Aussendurchmesser) used under Category 2
        content1 = (
            f"{OFFICIAL_HEADER_LINE}\n"
            "1001\t2\tBremsscheiben-Aussendurchmesser\t300 mm\n"
        )
        sub_file1 = self._create_tsv("cat2_with_cat1_aspect.tsv", content1)
        is_valid1, report1 = validate_submission(sub_file1)
        self.assertFalse(is_valid1)
        self.assertTrue(any("not allowed in category 2" in e.lower() for e in report1["errors"]))

        # Category 2 exclusive aspect (SAE_Viskosität) used under Category 1
        content2 = (
            f"{OFFICIAL_HEADER_LINE}\n"
            "1002\t1\tSAE_Viskosität\t5W-30\n"
        )
        sub_file2 = self._create_tsv("cat1_with_cat2_aspect.tsv", content2)
        is_valid2, report2 = validate_submission(sub_file2)
        self.assertFalse(is_valid2)
        self.assertTrue(any("not allowed in category 1" in e.lower() for e in report2["errors"]))

    def test_empty_aspect_value(self):
        content = (
            f"{OFFICIAL_HEADER_LINE}\n"
            "1001\t1\tHersteller\t   \n"
        )
        sub_file = self._create_tsv("empty_value.tsv", content)
        is_valid, report = validate_submission(sub_file)
        self.assertFalse(is_valid)
        self.assertTrue(any("empty or whitespace-only" in e.lower() for e in report["errors"]))

    def test_csv_quote_wrapping(self):
        content = (
            f"{OFFICIAL_HEADER_LINE}\n"
            '1001\t1\tHersteller\t"BOSCH"\n'
        )
        sub_file = self._create_tsv("csv_quotes.tsv", content)
        is_valid, report = validate_submission(sub_file)
        self.assertFalse(is_valid)
        self.assertTrue(any("quote wrapping" in e.lower() for e in report["errors"]))

    def test_german_unicode_characters(self):
        content = (
            f"{OFFICIAL_HEADER_LINE}\n"
            "1001\t1\tHersteller\tMEYLE-ORIGINAL: True\n"
            "1001\t1\tProduktart\tÖlfiltereinsatz\n"
            "1001\t1\tMaßeinheit\tStück\n"
            "1001\t1\tIm_Lieferumfang_Enthalten\tDichtungsring für Ölwanne\n"
            "1001\t1\tKompatibles_Fahrzeug_Modell\tKäfer Cabriolet\n"
        )
        sub_file = self._create_tsv("german_unicode.tsv", content)
        is_valid, report = validate_submission(sub_file)
        self.assertTrue(is_valid, f"German Unicode failed: {report['errors']}")
        self.assertEqual(report["data_rows"], 5)

    def test_multiset_duplicate_entities_handling(self):
        content = (
            f"{OFFICIAL_HEADER_LINE}\n"
            "1001\t1\tMaßeinheit\tmm\n"
            "1001\t1\tMaßeinheit\tmm\n"  # repeated in title
        )
        sub_file = self._create_tsv("duplicate_mention.tsv", content)

        # By default, duplicate mentions are accepted as warnings for Counter multiset scoring
        is_valid, report = validate_submission(sub_file, disallow_duplicates=False)
        self.assertTrue(is_valid)
        self.assertEqual(report["duplicate_rows_count"], 1)

        # When disallow_duplicates is enabled, it triggers a fatal error
        is_valid_strict, report_strict = validate_submission(sub_file, disallow_duplicates=True)
        self.assertFalse(is_valid_strict)
        self.assertTrue(any("multiple occurrences" in e.lower() for e in report_strict["errors"]))

    def test_zero_aspect_titles_omission(self):
        # Candidate dataset has 3 records: 1001, 1002, 1003.
        # Record 1002 produces 0 entities, so only 1001 and 1003 appear in submission.
        test_content = (
            "Record Number\tCategory\tTitle\n"
            "1001\t1\tBOSCH Bremsscheiben\n"
            "1002\t1\tUnrelated item no aspect\n"
            "1003\t2\tCastrol Motoröl 5W-30\n"
        )
        test_file = self._create_tsv("test_dataset.tsv", test_content)

        sub_content = (
            f"{OFFICIAL_HEADER_LINE}\n"
            "1001\t1\tHersteller\tBOSCH\n"
            "1003\t2\tHersteller\tCastrol\n"
        )
        sub_file = self._create_tsv("sub_with_zero_entity.tsv", sub_content)

        is_valid, report = validate_submission(sub_file, test_path=test_file)
        self.assertTrue(is_valid, f"Validation failed: {report['errors']}")
        self.assertEqual(report["test_reference"]["total_test_records"], 3)
        self.assertEqual(report["test_reference"]["zero_entity_records"], 1)

    def test_test_reference_missing_record(self):
        # Submission contains record 9999 which is not in candidate test dataset
        test_content = (
            "Record Number\tCategory\tTitle\n"
            "1001\t1\tBOSCH Bremsscheiben\n"
        )
        test_file = self._create_tsv("test_small.tsv", test_content)

        sub_content = (
            f"{OFFICIAL_HEADER_LINE}\n"
            "1001\t1\tHersteller\tBOSCH\n"
            "9999\t1\tHersteller\tUnknown\n"
        )
        sub_file = self._create_tsv("sub_extra_record.tsv", sub_content)

        is_valid, report = validate_submission(sub_file, test_path=test_file)
        self.assertFalse(is_valid)
        self.assertTrue(any("does not exist in the candidate" in e.lower() for e in report["errors"]))

    def test_test_reference_category_mismatch(self):
        # Record 1001 is Category 1 in test file, but submission claims Category 2
        test_content = (
            "Record Number\tCategory\tTitle\n"
            "1001\t1\tBOSCH Bremsscheiben\n"
        )
        test_file = self._create_tsv("test_cat_ref.tsv", test_content)

        sub_content = (
            f"{OFFICIAL_HEADER_LINE}\n"
            "1001\t2\tHersteller\tBOSCH\n"
        )
        sub_file = self._create_tsv("sub_cat_mismatch.tsv", sub_content)

        is_valid, report = validate_submission(sub_file, test_path=test_file)
        self.assertFalse(is_valid)
        self.assertTrue(any("category mismatch" in e.lower() for e in report["errors"]))

    def test_empty_file_handling(self):
        sub_file = self._create_tsv("empty.tsv", "")
        is_valid, report = validate_submission(sub_file)
        self.assertFalse(is_valid)
        self.assertIn("fatal_error", report)


if __name__ == "__main__":
    unittest.main()
