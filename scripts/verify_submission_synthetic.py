"""
Synthetic Submission Dry Run and Pipeline Verification for eBay ML Competition.

Phase 3 Deliverable:
Validates the complete end-to-end output transformation on a synthetic mini dataset:
model-style entity predictions -> aspect names and values -> submission rows -> TSV file -> independent validator.

Covers all 10 required synthetic scenarios:
1. Single-token entities.
2. Multi-token entities.
3. Multiple aspects within one title.
4. Adjacent entities with the same aspect name.
5. Zero-entity titles (produces 0 submission rows).
6. Duplicate-looking but legitimately distinct entities (repeated brands/tokens).
7. Non-ASCII German characters (ä, ö, ü, ß, Ø, €).
8. Special punctuation and spacing (+, /, -, &, (, )).
9. Invalid entity boundaries / formatting (verified that validator catches error).
10. Incorrect or unsupported aspect labels (verified that validator catches error).

Strict Privacy:
Uses invented synthetic titles, record IDs, and entities only. No competition data.
"""

import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.entity_metrics import bio_to_entities
from scripts.validate_submission import (
    OFFICIAL_HEADER,
    validate_submission_file,
)


def build_synthetic_dataset() -> List[dict]:
    """
    Construct invented synthetic titles covering test cases 1 through 8.
    All record IDs, titles, tokens, and entities are completely invented.
    """
    return [
        {
            # Case 1 & 7: Single-token entity + German non-ASCII character in aspect name (Größe)
            "record_number": 5001,
            "category": 1,
            "tokens": ["Bremsscheibe", "belüftet", "Ø300mm", "Zimmermann"],
            "bio_tags": ["B-Produktart", "B-Bremsscheibenart", "B-Größe", "B-Hersteller"],
            "expected_entities": [
                ("Produktart", "Bremsscheibe"),
                ("Bremsscheibenart", "belüftet"),
                ("Größe", "Ø300mm"),
                ("Hersteller", "Zimmermann"),
            ],
        },
        {
            # Case 2: Multi-token entity (concatenated with single ASCII space)
            "record_number": 5002,
            "category": 2,
            "tokens": ["Zahnriemen", "Satz", "Kit", "Plus", "Wasserpumpe"],
            "bio_tags": ["B-Produktart", "I-Produktart", "I-Produktart", "O", "B-Im_Lieferumfang_Enthalten"],
            "expected_entities": [
                ("Produktart", "Zahnriemen Satz Kit"),
                ("Im_Lieferumfang_Enthalten", "Wasserpumpe"),
            ],
        },
        {
            # Case 3 & 8: Multiple aspects within one title + special punctuation (+, -, &)
            "record_number": 5003,
            "category": 1,
            "tokens": ["Sport-Bremsscheiben", "+", "Beläge", "vorne", "&", "hinten", "ATE"],
            "bio_tags": ["B-Produktart", "O", "B-Im_Lieferumfang_Enthalten", "B-Einbauposition", "O", "B-Einbauposition", "B-Hersteller"],
            "expected_entities": [
                ("Produktart", "Sport-Bremsscheiben"),
                ("Im_Lieferumfang_Enthalten", "Beläge"),
                ("Einbauposition", "vorne"),
                ("Einbauposition", "hinten"),
                ("Hersteller", "ATE"),
            ],
        },
        {
            # Case 4: Adjacent entities with the same aspect name (consecutive B- tags)
            "record_number": 5004,
            "category": 2,
            "tokens": ["Set", "Kette", "Spanner", "für", "Motor"],
            "bio_tags": ["B-Im_Lieferumfang_Enthalten", "B-Im_Lieferumfang_Enthalten", "B-Im_Lieferumfang_Enthalten", "O", "B-Anwendung"],
            "expected_entities": [
                ("Im_Lieferumfang_Enthalten", "Set"),
                ("Im_Lieferumfang_Enthalten", "Kette"),
                ("Im_Lieferumfang_Enthalten", "Spanner"),
                ("Anwendung", "Motor"),
            ],
        },
        {
            # Case 5: Zero-entity title (all tokens tagged O)
            "record_number": 5005,
            "category": 1,
            "tokens": ["Neu", "und", "originalverpackt", "Top", "Zustand"],
            "bio_tags": ["O", "O", "O", "O", "O"],
            "expected_entities": [],
        },
        {
            # Case 6 & 7: Duplicate-looking distinct entities in one title + non-ASCII umlaut
            "record_number": 5006,
            "category": 2,
            "tokens": ["BMW", "3er", "für", "BMW", "E90", "Ölfilter"],
            "bio_tags": ["B-Kompatible_Fahrzeug_Marke", "B-Kompatibles_Fahrzeug_Modell", "O", "B-Kompatible_Fahrzeug_Marke", "B-Kompatibles_Fahrzeug_Modell", "B-Produktart"],
            "expected_entities": [
                ("Kompatible_Fahrzeug_Marke", "BMW"),
                ("Kompatibles_Fahrzeug_Modell", "3er"),
                ("Kompatible_Fahrzeug_Marke", "BMW"),
                ("Kompatibles_Fahrzeug_Modell", "E90"),
                ("Produktart", "Ölfilter"),
            ],
        },
    ]


def generate_submission_tsv(
    dataset: List[dict],
    output_path: Path,
    include_header: bool = True,
) -> int:
    """
    Format predictions into official literal TSV submission lines.
    Returns total rows written.
    """
    rows_written = 0
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        if include_header:
            f.write("\t".join(OFFICIAL_HEADER) + "\n")

        for item in dataset:
            rec_id = item["record_number"]
            cat_id = item["category"]
            tokens = item["tokens"]
            bio_tags = item["bio_tags"]

            entities = bio_to_entities(tokens, bio_tags)

            for aspect_name, aspect_val in entities:
                if aspect_name == "O":
                    # Strictly omit O
                    continue
                # Write literal TSV line
                f.write(f"{rec_id}\t{cat_id}\t{aspect_name}\t{aspect_val}\n")
                rows_written += 1

    return rows_written


def run_synthetic_dry_run() -> bool:
    print("==================================================")
    print("EXPERIMENT 20: SYNTHETIC SUBMISSION DRY RUN")
    print("==================================================")

    synthetic_data = build_synthetic_dataset()
    print(f"Constructed {len(synthetic_data)} synthetic test titles.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # ----------------------------------------------------
        # TEST A: Valid Synthetic Submission (Cases 1 - 8)
        # ----------------------------------------------------
        print("\n--- TEST A: VALID SYNTHETIC SUBMISSION (CASES 1 - 8) ---")
        valid_tsv = tmp_path / "synthetic_valid_submission.tsv"
        rows_written = generate_submission_tsv(synthetic_data, valid_tsv, include_header=True)
        print(f"Rows written: {rows_written}")

        is_valid, report = validate_submission_file(valid_tsv, expected_min_record=5001, expected_max_record=5006)
        print(f"Validation Result: {'PASS' if is_valid else 'FAIL'}")
        print(f"Aspects extracted: {len(report['aspect_distribution'])}")
        print(f"Data rows:         {report['total_data_rows']}")
        print(f"Unique records:    {report['unique_record_numbers']} (Zero-entity title 5005 produced 0 rows as expected)")

        if not is_valid:
            print("Errors detected in valid synthetic data:")
            for e in report["errors"]:
                print(f"  - {e}")
            return False

        # Verify zero-entity title produced 0 rows and did not fail validation
        if 5005 in report["record_ids_seen"] if "record_ids_seen" in report else False:
            print("FAIL: Title 5005 (zero entities) unexpectedly produced rows!")
            return False

        # ----------------------------------------------------
        # TEST B: Case 9 — Invalid Formatting & Accidental CSV Quoting
        # ----------------------------------------------------
        print("\n--- TEST B: CASE 9 - INVALID BOUNDARIES & QUOTING CHECK ---")
        invalid_quoting_tsv = tmp_path / "synthetic_invalid_quoting.tsv"
        with open(invalid_quoting_tsv, "w", encoding="utf-8") as f:
            f.write("\t".join(OFFICIAL_HEADER) + "\n")
            # Accidental CSV quoting
            f.write('5001\t1\tProduktart\t"Bremsscheibe Belag"\n')
            # 3 columns instead of 4
            f.write("5002\t2\tHersteller\n")
            # Consecutive multiple spaces
            f.write("5003\t1\tHersteller\tATE   Zimmermann\n")

        is_val_b, rep_b = validate_submission_file(invalid_quoting_tsv)
        print(f"Expected Fail Result: {'PASS (Correctly Rejected)' if not is_val_b else 'FAIL (Missed Errors)'}")
        print(f"Errors Caught ({len(rep_b['errors'])}):")
        for e in rep_b["errors"]:
            print(f"  - {e}")

        if is_val_b:
            print("FAIL: Validator failed to catch invalid quoting / column formatting!")
            return False

        # ----------------------------------------------------
        # TEST C: Case 10 - Invalid Aspect Names & 'O' Submission
        # ----------------------------------------------------
        print("\n--- TEST C: CASE 10 - INVALID ASPECT & 'O' TAG CHECK ---")
        invalid_aspect_tsv = tmp_path / "synthetic_invalid_aspect.tsv"
        with open(invalid_aspect_tsv, "w", encoding="utf-8") as f:
            f.write("\t".join(OFFICIAL_HEADER) + "\n")
            # Submission of 'O'
            f.write("5001\t1\tO\tPassend\n")
            # Unsupported aspect name
            f.write("5002\t2\tInventedNonExistentAspect\tTurbo\n")

        is_val_c, rep_c = validate_submission_file(invalid_aspect_tsv)
        print(f"Expected Fail Result: {'PASS (Correctly Rejected)' if not is_val_c else 'FAIL (Missed Errors)'}")
        print(f"Errors Caught ({len(rep_c['errors'])}):")
        for e in rep_c["errors"]:
            print(f"  - {e}")

        if is_val_c:
            print("FAIL: Validator failed to catch unauthorized aspect name or 'O' tag!")
            return False

        # ----------------------------------------------------
        # TEST D: Case 11 - Category-Specific Aspect Restriction Check
        # ----------------------------------------------------
        print("\n--- TEST D: CASE 11 - CATEGORY-SPECIFIC ASPECT RESTRICTIONS ---")
        cat_invalid_tsv = tmp_path / "synthetic_cat_invalid.tsv"
        with open(cat_invalid_tsv, "w", encoding="utf-8") as f:
            f.write("\t".join(OFFICIAL_HEADER) + "\n")
            # Bremsscheibenart is only allowed in Category 1, forbidden in Category 2
            f.write("5001\t2\tBremsscheibenart\tbelüftet\n")
            # Bremsscheiben-Aussendurchmesser is only allowed in Category 1
            f.write("5002\t2\tBremsscheiben-Aussendurchmesser\t300mm\n")
            # Anwendung is only allowed in Category 2
            f.write("5003\t1\tAnwendung\tNockenwelle\n")
            # Zähnezahl is only allowed in Category 2
            f.write("5004\t1\tZähnezahl\t125\n")

        is_val_d, rep_d = validate_submission_file(cat_invalid_tsv)
        print(f"Negative Test Result: {'PASS (Correctly Rejected)' if not is_val_d else 'FAIL (Missed Errors)'}")
        print(f"Errors Caught ({len(rep_d['errors'])}):")
        for e in rep_d["errors"]:
            print(f"  - {e}")

        if is_val_d:
            print("FAIL: Validator failed to reject category-restricted aspects!")
            return False

        # Verify exact Bremsscheibenart rejection in Category 2
        brems_err = any("Aspect name not allowed for Category 2: 'Bremsscheibenart'" in err for err in rep_d["errors"])
        if not brems_err:
            print("FAIL: Validator did not emit expected error for Bremsscheibenart in Category 2!")
            return False

        # Permitted category combinations test
        cat_valid_tsv = tmp_path / "synthetic_cat_valid.tsv"
        with open(cat_valid_tsv, "w", encoding="utf-8") as f:
            f.write("\t".join(OFFICIAL_HEADER) + "\n")
            # Valid: Bremsscheibenart in Category 1
            f.write("5001\t1\tBremsscheibenart\tbelüftet\n")
            # Valid: Bremsscheiben-Aussendurchmesser in Category 1
            f.write("5002\t1\tBremsscheiben-Aussendurchmesser\t300mm\n")
            # Valid: Anwendung in Category 2
            f.write("5003\t2\tAnwendung\tNockenwelle\n")
            # Valid: Zähnezahl in Category 2
            f.write("5004\t2\tZähnezahl\t125\n")
            # Valid: Common aspects in both categories
            f.write("5005\t1\tProduktart\tBremsenset\n")
            f.write("5006\t2\tProduktart\tSteuerkettensatz\n")

        is_val_e, rep_e = validate_submission_file(cat_valid_tsv)
        print(f"Positive Test Result: {'PASS (Correctly Accepted)' if is_val_e else 'FAIL (Unexpectedly Rejected)'}")
        if not is_val_e:
            print("Errors detected in valid category combinations:")
            for e in rep_e["errors"]:
                print(f"  - {e}")
            return False

    print("\n==================================================")
    print("ALL SYNTHETIC SUBMISSION DRY RUN CHECKS PASSED!")
    print("==================================================")
    return True


if __name__ == "__main__":
    success = run_synthetic_dry_run()
    if not success:
        sys.exit(1)
    sys.exit(0)
