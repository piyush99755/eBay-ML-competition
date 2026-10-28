"""
Independent Final Submission Validation Tool
eBay 2026 University Machine Learning Competition

Validates final competition submission files against official specifications:
1. File format: Tab-Separated Values (TSV) or gzipped TSV (.tsv.gz), UTF-8 encoded.
2. Official header: Exactly 'Record Number\\tCategory\\tAspect Name\\tAspect Value'.
3. Exactly 4 tab-separated columns per row.
4. Record Number: Positive integer.
5. Category: Integer 1 (Car & Truck Parts) or 2 (Motorcycle Parts).
6. Aspect Name: One of the 29 official aspect names, strictly adhering to category restrictions:
   - Category 1: 23 permitted aspects (14 common + 9 Cat-1 exclusive).
   - Category 2: 20 permitted aspects (14 common + 6 Cat-2 exclusive).
   - No 'O' (Obscure) tags.
7. Aspect Value: Non-empty, single-space separated tokens, no quotes, no leading/trailing whitespace.
8. Duplicate entities: Tracked and audited. Note that the competition evaluation metric
   (Annexure.pdf & entity_metrics.py) evaluates multiset Counter matching where repeated
   identical entities in a single title are legitimate and scored. Supported as informational/warning
   unless --disallow-duplicates is explicitly passed.
9. Optional candidate/test dataset integrity verification:
   - Validates all submission record IDs exist in candidate dataset.
   - Validates submission Category matches candidate dataset Category for each record ID.
   - Reports zero-entity record counts (permissible per competition specification).
10. Comprehensive summary statistics and SHA-256 cryptographic checksum.
"""

import argparse
import gzip
import hashlib
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Set utf-8 encoding for standard outputs if available (avoids Windows cmd cp1252 issues)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Official 29 Semantic Aspects
OFFICIAL_ASPECTS_COMMON: Set[str] = {
    "Anzahl_Der_Einheiten",
    "Besonderheiten",
    "Einbauposition",
    "Größe",
    "Hersteller",
    "Herstellernummer",
    "Im_Lieferumfang_Enthalten",
    "Kompatible_Fahrzeug_Marke",
    "Kompatibles_Fahrzeug_Jahr",
    "Kompatibles_Fahrzeug_Modell",
    "Maßeinheit",
    "Modell",
    "Oe/Oem_Referenznummer(N)",
    "Produktart",
}

OFFICIAL_ASPECTS_CAT1_EXCLUSIVE: Set[str] = {
    "Bremsscheiben-Aussendurchmesser",
    "Bremsscheibenart",
    "Farbe",
    "Herstellungsland_Und_-Region",
    "Material",
    "Oberflächenbeschaffenheit",
    "Produktlinie",
    "Stärke",
    "Technologie",
}

OFFICIAL_ASPECTS_CAT2_EXCLUSIVE: Set[str] = {
    "Anwendung",
    "Breite",
    "Länge",
    "Menge",
    "SAE_Viskosität",
    "Zähnezahl",
}

OFFICIAL_ASPECTS_CAT1: Set[str] = OFFICIAL_ASPECTS_COMMON | OFFICIAL_ASPECTS_CAT1_EXCLUSIVE  # 23
OFFICIAL_ASPECTS_CAT2: Set[str] = OFFICIAL_ASPECTS_COMMON | OFFICIAL_ASPECTS_CAT2_EXCLUSIVE  # 20
OFFICIAL_ASPECTS_ALL: Set[str] = OFFICIAL_ASPECTS_CAT1 | OFFICIAL_ASPECTS_CAT2               # 29

OFFICIAL_HEADER = ["Record Number", "Category", "Aspect Name", "Aspect Value"]
OFFICIAL_HEADER_LINE = "\t".join(OFFICIAL_HEADER)


def compute_sha256(filepath: Path) -> str:
    """Compute uppercase SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


def load_test_reference(test_path: Path) -> Tuple[Dict[int, int], int]:
    """
    Load test reference file to get mapping of {record_id: category_id}.
    Supports TSV and TSV.GZ.
    """
    test_records: Dict[int, int] = {}
    total_lines = 0
    is_gz = str(test_path).endswith(".gz")
    open_fn = gzip.open if is_gz else open

    with open_fn(test_path, "rt", encoding="utf-8", errors="replace") as f:
        for line_idx, line in enumerate(f, start=1):
            stripped = line.rstrip("\r\n")
            if not stripped:
                continue
            parts = stripped.split("\t")
            if line_idx == 1:
                # Check if header
                try:
                    int(parts[0])
                except ValueError:
                    continue  # Skip header
            if len(parts) >= 2:
                try:
                    rec_id = int(parts[0])
                    cat_id = int(parts[1])
                    test_records[rec_id] = cat_id
                    total_lines += 1
                except ValueError:
                    pass
    return test_records, total_lines


def validate_submission(
    submission_path: Path,
    test_path: Optional[Path] = None,
    max_errors: int = 50,
    disallow_duplicates: bool = False,
) -> Tuple[bool, dict]:
    """
    Perform rigorous, independent validation on a submission file.
    
    Returns:
        (is_valid: bool, report: dict)
    """
    submission_path = Path(submission_path)
    if not submission_path.exists():
        return False, {"fatal_error": f"Submission file does not exist: {submission_path}"}

    file_size_bytes = submission_path.stat().st_size
    if file_size_bytes == 0:
        return False, {"fatal_error": f"Submission file is completely empty (0 bytes): {submission_path}"}

    is_gz = submission_path.suffix.lower() == ".gz" or str(submission_path).endswith(".tsv.gz")
    open_fn = gzip.open if is_gz else open

    errors: List[str] = []
    warnings: List[str] = []

    line_count = 0
    data_row_count = 0
    header_found = False

    seen_rows: Set[Tuple[int, int, str, str]] = set()
    duplicate_rows_count = 0
    records_with_entities: Set[int] = set()
    category_counts: Dict[int, int] = Counter()
    aspect_counts: Dict[str, int] = Counter()
    aspects_per_record: Dict[int, int] = Counter()
    category_mismatches = 0

    # Optional test reference
    test_records: Optional[Dict[int, int]] = None
    if test_path is not None:
        if not test_path.exists():
            return False, {"fatal_error": f"Test reference file does not exist: {test_path}"}
        test_records, _ = load_test_reference(test_path)

    try:
        with open_fn(submission_path, "rt", encoding="utf-8", errors="strict") as f:
            for line_idx, raw_line in enumerate(f, start=1):
                line_count += 1
                stripped = raw_line.rstrip("\r\n")

                if not stripped:
                    errors.append(f"Line {line_idx}: Blank or whitespace-only line detected.")
                    if len(errors) >= max_errors:
                        break
                    continue

                parts = stripped.split("\t")

                # Line 1: Header verification
                if line_idx == 1:
                    if parts == OFFICIAL_HEADER:
                        header_found = True
                        continue
                    else:
                        errors.append(
                            f"Line 1: Invalid header format: {parts}. Expected exactly {OFFICIAL_HEADER}."
                        )
                        # Check if line 1 was actually a data row
                        try:
                            int(parts[0])
                            errors.append("Line 1 appears to be a data row; missing mandatory header row.")
                        except ValueError:
                            pass
                        if len(errors) >= max_errors:
                            break
                        continue

                data_row_count += 1

                # Rule 1: Exactly 4 fields
                if len(parts) != 4:
                    errors.append(
                        f"Line {line_idx}: Expected exactly 4 tab-separated fields, got {len(parts)}: {parts}"
                    )
                    if len(errors) >= max_errors:
                        break
                    continue

                rec_str, cat_str, aspect_name, aspect_val = parts

                # Rule 2: Record Number must be positive integer
                try:
                    rec_num = int(rec_str)
                    if rec_num < 1:
                        errors.append(f"Line {line_idx}: Record Number must be >= 1, got {rec_num}.")
                except ValueError:
                    errors.append(f"Line {line_idx}: Record Number is not a valid integer: '{rec_str}'.")
                    rec_num = None

                # Rule 3: Category must be 1 or 2
                try:
                    cat_id = int(cat_str)
                    if cat_id not in (1, 2):
                        errors.append(f"Line {line_idx}: Category must be 1 or 2, got {cat_id}.")
                        cat_id = None
                except ValueError:
                    errors.append(f"Line {line_idx}: Category is not a valid integer: '{cat_str}'.")
                    cat_id = None

                # Rule 4: Aspect Name validation
                if aspect_name == "O":
                    errors.append(
                        f"Line {line_idx}: Prohibited 'O' (Obscure) tag submitted as aspect name."
                    )
                elif aspect_name not in OFFICIAL_ASPECTS_ALL:
                    errors.append(
                        f"Line {line_idx}: Unknown aspect name: '{aspect_name}'. Must be one of the 29 official aspects."
                    )
                elif cat_id == 1 and aspect_name not in OFFICIAL_ASPECTS_CAT1:
                    errors.append(
                        f"Line {line_idx}: Aspect '{aspect_name}' is not allowed in Category 1."
                    )
                elif cat_id == 2 and aspect_name not in OFFICIAL_ASPECTS_CAT2:
                    errors.append(
                        f"Line {line_idx}: Aspect '{aspect_name}' is not allowed in Category 2."
                    )

                # Rule 5: Aspect Value validation
                if not aspect_val or not aspect_val.strip():
                    errors.append(f"Line {line_idx}: Empty or whitespace-only Aspect Value.")
                elif aspect_val.startswith('"') and aspect_val.endswith('"') and len(aspect_val) > 1:
                    errors.append(
                        f"Line {line_idx}: Accidental CSV quote wrapping detected: '{aspect_val}'."
                    )
                elif aspect_val.startswith("'") and aspect_val.endswith("'") and len(aspect_val) > 1:
                    warnings.append(
                        f"Line {line_idx}: Single-quote wrapping detected on aspect value: '{aspect_val}'."
                    )
                elif "  " in aspect_val:
                    warnings.append(
                        f"Line {line_idx}: Multiple consecutive spaces in aspect value: '{aspect_val}'."
                    )
                elif aspect_val != aspect_val.strip():
                    errors.append(
                        f"Line {line_idx}: Leading or trailing whitespace in aspect value: '{aspect_val}'."
                    )

                # Rule 6: Duplicate row check
                if rec_num is not None and cat_id is not None:
                    row_key = (rec_num, cat_id, aspect_name, aspect_val)
                    if row_key in seen_rows:
                        duplicate_rows_count += 1
                        msg = f"Line {line_idx}: Multiple occurrences of ({aspect_name}: '{aspect_val}') for Record {rec_num}."
                        if disallow_duplicates:
                            errors.append(msg)
                        else:
                            if duplicate_rows_count <= 5:
                                warnings.append(msg)
                    seen_rows.add(row_key)

                    records_with_entities.add(rec_num)
                    aspects_per_record[rec_num] += 1
                    category_counts[cat_id] += 1
                    aspect_counts[aspect_name] += 1

                # Rule 7: Test file cross-reference
                if test_records is not None and rec_num is not None:
                    if rec_num not in test_records:
                        errors.append(
                            f"Line {line_idx}: Record {rec_num} does not exist in the candidate/test reference dataset."
                        )
                    else:
                        expected_cat = test_records[rec_num]
                        if cat_id is not None and cat_id != expected_cat:
                            errors.append(
                                f"Line {line_idx}: Category mismatch for Record {rec_num}: submission has {cat_id}, test dataset specifies {expected_cat}."
                            )
                            category_mismatches += 1

                if len(errors) >= max_errors:
                    errors.append(f"Reached maximum error threshold ({max_errors}). Terminating scan.")
                    break

    except UnicodeDecodeError as ude:
        errors.append(f"UTF-8 Decoding Error: {ude}")

    # Compute SHA-256
    file_sha256 = compute_sha256(submission_path)

    # Statistical distribution of aspects per record
    aspect_counts_list = list(aspects_per_record.values()) if aspects_per_record else [0]
    min_aspects = min(aspect_counts_list) if aspect_counts_list else 0
    max_aspects = max(aspect_counts_list) if aspect_counts_list else 0
    mean_aspects = sum(aspect_counts_list) / len(aspect_counts_list) if aspect_counts_list else 0.0

    # Cross-reference summary
    total_test_records = len(test_records) if test_records is not None else None
    zero_entity_records = (total_test_records - len(records_with_entities)) if total_test_records is not None else None

    is_valid = len(errors) == 0 and header_found and data_row_count > 0

    report = {
        "file_path": str(submission_path),
        "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
        "sha256": file_sha256,
        "is_valid": is_valid,
        "header_found": header_found,
        "total_lines": line_count,
        "data_rows": data_row_count,
        "unique_records_with_entities": len(records_with_entities),
        "duplicate_rows_count": duplicate_rows_count,
        "category_counts": dict(category_counts),
        "aspect_counts": dict(aspect_counts),
        "aspects_per_record": {
            "min": min_aspects,
            "max": max_aspects,
            "mean": round(mean_aspects, 3),
        },
        "errors": errors,
        "warnings": warnings,
        "test_reference": {
            "total_test_records": total_test_records,
            "zero_entity_records": zero_entity_records,
            "category_mismatches": category_mismatches,
        } if test_records is not None else None,
    }

    return is_valid, report


def print_validation_report(report: dict, verbose_aspects: bool = True) -> None:
    """Print beautifully formatted validation report."""
    print("=" * 72)
    print("FINAL SUBMISSION VALIDATION REPORT")
    print("=" * 72)
    print(f"File:         {report['file_path']}")
    print(f"Size:         {report['file_size_mb']} MB")
    print(f"SHA-256:      {report['sha256']}")
    print(f"Status:       {'PASS (VALID)' if report['is_valid'] else 'FAIL (INVALID)'}")
    print("-" * 72)
    print(f"Header:       {'Correct (' + OFFICIAL_HEADER_LINE + ')' if report['header_found'] else 'INVALID / MISSING'}")
    print(f"Total Lines:  {report['total_lines']:,}")
    print(f"Data Rows:    {report['data_rows']:,}")
    print(f"Unique Titles:{report['unique_records_with_entities']:,} with predicted entities")
    print(f"Duplicates:   {report['duplicate_rows_count']:,} recurring entity rows (Counter multiset evaluation)")

    if report.get("test_reference"):
        t_ref = report["test_reference"]
        tot = t_ref["total_test_records"]
        zero_ent = t_ref["zero_entity_records"]
        print(f"Test Titles:  {tot:,} total ({zero_ent:,} zero-entity titles, {(zero_ent/tot)*100:.2f}%)")
        print(f"Cat Mismatches: {t_ref['category_mismatches']}")

    print("-" * 72)
    print("CATEGORY DISTRIBUTION:")
    cat_counts = report["category_counts"]
    total_rows = report["data_rows"] if report["data_rows"] > 0 else 1
    for cat_id in sorted(cat_counts.keys()):
        cnt = cat_counts[cat_id]
        pct = (cnt / total_rows) * 100
        cat_name = "Car & Truck Parts" if cat_id == 1 else "Motorcycle Parts"
        print(f"  Category {cat_id} ({cat_name}): {cnt:,} rows ({pct:.1f}%)")

    print("-" * 72)
    apr = report["aspects_per_record"]
    print(f"ENTITIES PER RECORD: Min = {apr['min']}, Max = {apr['max']}, Mean = {apr['mean']}")

    if verbose_aspects and report["aspect_counts"]:
        print("-" * 72)
        print("ASPECT FREQUENCY BREAKDOWN (Top 15):")
        sorted_aspects = sorted(report["aspect_counts"].items(), key=lambda x: x[1], reverse=True)
        for asp, cnt in sorted_aspects[:15]:
            pct = (cnt / total_rows) * 100
            print(f"  {asp:<35} {cnt:>7,} ({pct:5.2f}%)")
        if len(sorted_aspects) > 15:
            print(f"  ... and {len(sorted_aspects) - 15} additional aspect types.")

    if report["warnings"]:
        print("-" * 72)
        print(f"NOTICES / WARNINGS ({len(report['warnings'])}):")
        for w in report["warnings"][:10]:
            print(f"  [WARN] {w}")
        if len(report["warnings"]) > 10:
            print(f"  ... and {len(report['warnings']) - 10} more notices.")

    if report["errors"]:
        print("-" * 72)
        print(f"ERRORS DETECTED ({len(report['errors'])}):")
        for e in report["errors"][:20]:
            print(f"  [ERROR] {e}")
        if len(report["errors"]) > 20:
            print(f"  ... and {len(report['errors']) - 20} more errors.")

    print("=" * 72)
    if report["is_valid"]:
        print(">>> RESULT: SUBMISSION COMPLIES 100% WITH OFFICIAL FORMAT SPECIFICATIONS <<<")
    else:
        print(">>> RESULT: SUBMISSION VALIDATION FAILED! DO NOT SUBMIT TO EVALAI <<<")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Independent Final Submission Validator for eBay ML Competition"
    )
    parser.add_argument(
        "--submission-file",
        type=Path,
        required=True,
        help="Path to submission TSV or TSV.GZ file to validate",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=None,
        help="Optional path to candidate/test reference dataset TSV or TSV.GZ to verify record IDs and categories",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=50,
        help="Maximum error messages to collect before aborting scan (default: 50)",
    )
    parser.add_argument(
        "--disallow-duplicates",
        action="store_true",
        help="Treat repeated identical entity rows within the same title as fatal errors",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed aspect frequency table",
    )

    args = parser.parse_args()

    is_valid, report = validate_submission(
        submission_path=args.submission_file,
        test_path=args.test_file,
        max_errors=args.max_errors,
        disallow_duplicates=args.disallow_duplicates,
    )

    print_validation_report(report, verbose_aspects=not args.quiet)

    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
