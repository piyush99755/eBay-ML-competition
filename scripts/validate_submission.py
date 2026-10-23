"""
Submission Validation Tool for eBay 2026 Machine Learning Competition.

Validates candidate submission files against official Annexure.pdf specifications:
1. File format: Tab-separated (TSV), UTF-8 encoded, optional gzip (.tsv.gz).
2. Schema: Exactly 4 fields per line:
   - Record Number (integer)
   - Category (integer: 1 or 2)
   - Aspect Name (string: one of the 29 official aspect names)
   - Aspect Value (string: reconstructed tokens separated by single ASCII space)
3. No CSV-style quoting (literal text, csv.QUOTE_NONE).
4. No 'O' (Obscure) tags submitted as aspect names.
5. Flexible header support (detects official header: 'Record Number\tCategory\tAspect Name\tAspect Value').
6. Allows variable rows per record (titles may produce zero, one, or multiple aspect rows).
7. Validates candidate range compliance (e.g. Quiz: 5001..30000).
"""

import argparse
import csv
import gzip
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Exact 29 official semantic aspect names from Annexure.pdf (excluding 'O')
OFFICIAL_ASPECTS: Set[str] = {
    "Anwendung",
    "Anzahl_Der_Einheiten",
    "Besonderheiten",
    "Breite",
    "Bremsscheiben-Aussendurchmesser",
    "Bremsscheibenart",
    "Einbauposition",
    "Farbe",
    "Größe",
    "Hersteller",
    "Herstellernummer",
    "Herstellungsland_Und_-Region",
    "Im_Lieferumfang_Enthalten",
    "Kompatible_Fahrzeug_Marke",
    "Kompatibles_Fahrzeug_Jahr",
    "Kompatibles_Fahrzeug_Modell",
    "Länge",
    "Material",
    "Maßeinheit",
    "Menge",
    "Modell",
    "Oberflächenbeschaffenheit",
    "Oe/Oem_Referenznummer(N)",
    "Produktart",
    "Produktlinie",
    "SAE_Viskosität",
    "Stärke",
    "Technologie",
    "Zähnezahl",
}

# Category 1 allowed aspects (23 aspects: 14 common + 9 Cat-1 exclusive per Annexure.pdf pp. 11-14)
OFFICIAL_ASPECTS_CAT1: Set[str] = {
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
    # Category 1 only:
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

# Category 2 allowed aspects (20 aspects: 14 common + 6 Cat-2 exclusive per Annexure.pdf pp. 11-14)
OFFICIAL_ASPECTS_CAT2: Set[str] = {
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
    # Category 2 only:
    "Anwendung",
    "Breite",
    "Länge",
    "Menge",
    "SAE_Viskosität",
    "Zähnezahl",
}

OFFICIAL_HEADER = ["Record Number", "Category", "Aspect Name", "Aspect Value"]
ALTERNATIVE_HEADER = ["Record Number", "Category Id", "Aspect Name", "Aspect Value"]


def validate_submission_file(
    file_path: Path,
    expected_min_record: Optional[int] = None,
    expected_max_record: Optional[int] = None,
    max_error_reports: int = 20,
) -> Tuple[bool, dict]:
    """
    Validate a submission TSV or TSV.GZ file.

    Returns:
    - is_valid: True if file complies 100% with all competition rules, False otherwise.
    - report: Summary dictionary of statistics and any detected validation errors.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return False, {"error": f"File does not exist: {file_path}"}

    is_gz = file_path.suffix.lower() == ".gz" or str(file_path).endswith(".tsv.gz")

    errors: List[str] = []
    line_count = 0
    data_rows = 0
    has_header = False

    aspect_counts: Dict[str, int] = defaultdict(int)
    category_counts: Dict[int, int] = defaultdict(int)
    record_ids_seen: Set[int] = set()

    try:
        open_fn = gzip.open if is_gz else open
        with open_fn(file_path, "rt", encoding="utf-8", errors="replace") as f:
            for line_idx, raw_line in enumerate(f, start=1):
                line_count += 1
                stripped = raw_line.rstrip("\r\n")

                if not stripped:
                    errors.append(f"Line {line_idx}: Empty line detected.")
                    if len(errors) >= max_error_reports:
                        break
                    continue

                parts = stripped.split("\t")

                # Check for header row on line 1
                if line_idx == 1:
                    if parts == OFFICIAL_HEADER or parts == ALTERNATIVE_HEADER:
                        has_header = True
                        continue
                    # Check if line 1 looks like a header with non-integer record number
                    try:
                        int(parts[0])
                    except ValueError:
                        has_header = True
                        if parts != OFFICIAL_HEADER and parts != ALTERNATIVE_HEADER:
                            errors.append(
                                f"Line 1: Invalid header format: {parts}. Expected {OFFICIAL_HEADER}."
                            )
                        continue

                data_rows += 1

                # Rule 1: Exactly 4 fields
                if len(parts) != 4:
                    errors.append(
                        f"Line {line_idx}: Expected 4 tab-separated fields, got {len(parts)}."
                    )
                    if len(errors) >= max_error_reports:
                        break
                    continue

                rec_str, cat_str, aspect_name, aspect_val = parts

                # Rule 2: Record Number must be an integer
                try:
                    rec_num = int(rec_str)
                    if rec_num < 1:
                        errors.append(f"Line {line_idx}: Record Number must be >= 1, got {rec_num}.")
                    record_ids_seen.add(rec_num)

                    if expected_min_record is not None and rec_num < expected_min_record:
                        errors.append(
                            f"Line {line_idx}: Record Number {rec_num} < expected min {expected_min_record}."
                        )
                    if expected_max_record is not None and rec_num > expected_max_record:
                        errors.append(
                            f"Line {line_idx}: Record Number {rec_num} > expected max {expected_max_record}."
                        )
                except ValueError:
                    errors.append(f"Line {line_idx}: Record Number is not an integer: '{rec_str}'.")

                # Rule 3: Category Id must be 1 or 2
                try:
                    cat_id = int(cat_str)
                    if cat_id not in (1, 2):
                        errors.append(f"Line {line_idx}: Category must be 1 or 2, got {cat_id}.")
                    category_counts[cat_id] += 1
                except ValueError:
                    errors.append(f"Line {line_idx}: Category is not an integer: '{cat_str}'.")

                # Rule 4: Aspect Name must be in official 29 aspects and allowed for this category
                if aspect_name == "O":
                    errors.append(
                        f"Line {line_idx}: 'O' (Obscure) tag submitted as aspect name. O must be omitted."
                    )
                elif aspect_name not in OFFICIAL_ASPECTS:
                    errors.append(
                        f"Line {line_idx}: Unknown or unauthorized aspect name: '{aspect_name}'."
                    )
                elif cat_id == 1 and aspect_name not in OFFICIAL_ASPECTS_CAT1:
                    errors.append(
                        f"Line {line_idx}: Aspect name not allowed for Category 1: '{aspect_name}'."
                    )
                elif cat_id == 2 and aspect_name not in OFFICIAL_ASPECTS_CAT2:
                    errors.append(
                        f"Line {line_idx}: Aspect name not allowed for Category 2: '{aspect_name}'."
                    )
                else:
                    aspect_counts[aspect_name] += 1

                # Rule 5: Aspect Value must be non-empty and literal text
                if not aspect_val.strip():
                    errors.append(f"Line {line_idx}: Empty or whitespace-only Aspect Value.")
                elif aspect_val.startswith('"') and aspect_val.endswith('"') and len(aspect_val) > 1:
                    # Accidental CSV quote escaping check
                    errors.append(
                        f"Line {line_idx}: Aspect Value appears to have accidental CSV quoting: '{aspect_val}'."
                    )
                elif "  " in aspect_val:
                    # Notice on consecutive spaces
                    errors.append(
                        f"Line {line_idx}: Aspect Value contains multiple consecutive spaces (must use single space 0x20)."
                    )

                if len(errors) >= max_error_reports:
                    errors.append(f"Maximum error threshold ({max_error_reports}) reached; stopping scan.")
                    break

    except Exception as e:
        return False, {"error": f"Failed reading file: {e}"}

    is_valid = len(errors) == 0

    report = {
        "is_valid": is_valid,
        "file_path": str(file_path),
        "is_gzip": is_gz,
        "has_header": has_header,
        "total_lines_read": line_count,
        "total_data_rows": data_rows,
        "unique_record_numbers": len(record_ids_seen),
        "min_record_number": min(record_ids_seen) if record_ids_seen else None,
        "max_record_number": max(record_ids_seen) if record_ids_seen else None,
        "category_distribution": dict(category_counts),
        "aspect_distribution": dict(sorted(aspect_counts.items())),
        "total_errors": len(errors),
        "errors": errors,
    }

    return is_valid, report


def print_validation_report(report: dict) -> None:
    print("\n==================================================")
    print("SUBMISSION VALIDATION REPORT")
    print("==================================================")
    print(f"File Path:                {report.get('file_path')}")
    print(f"File Valid:               {'YES (COMPLIANT)' if report.get('is_valid') else 'NO (INVALID)'}")
    print(f"Gzip Compressed:          {report.get('is_gzip')}")
    print(f"Header Present:           {report.get('has_header')}")
    print(f"Total Lines Read:         {report.get('total_lines_read')}")
    print(f"Data Aspect Rows:         {report.get('total_data_rows')}")
    print(f"Unique Record Numbers:    {report.get('unique_record_numbers')}")
    print(f"Record Range:             [{report.get('min_record_number')}, {report.get('max_record_number')}]")
    print(f"Category Distribution:    {report.get('category_distribution')}")
    print(f"Aspect Types Extracted:   {len(report.get('aspect_distribution', {}))} of 29 official aspects")

    if report.get("errors"):
        print(f"\n[!] Detected {len(report['errors'])} Validation Error(s):")
        for err in report["errors"]:
            print(f"  - {err}")
    else:
        print("\nAll schema, encoding, and Annexure.pdf formatting checks passed!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate eBay competition submission file.")
    parser.add_argument("file_path", type=Path, help="Path to submission .tsv or .tsv.gz file.")
    parser.add_argument("--min-record", type=int, default=None, help="Expected minimum record number.")
    parser.add_argument("--max-record", type=int, default=None, help="Expected maximum record number.")
    args = parser.parse_args()

    is_valid, report = validate_submission_file(
        args.file_path,
        expected_min_record=args.min_record,
        expected_max_record=args.max_record,
    )
    print_validation_report(report)
    if not is_valid:
        exit(1)


if __name__ == "__main__":
    main()
