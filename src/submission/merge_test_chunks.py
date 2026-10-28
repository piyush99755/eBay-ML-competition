"""
Safe Chunk Merging & Recovery Utility
eBay 2026 University Machine Learning Competition

Merges multiple partial submission TSV chunk files into a single, canonical,
sequentially ordered submission file.

Key Safeguards:
1. Validates each chunk has the exact official header:
   'Record Number\\tCategory\\tAspect Name\\tAspect Value'.
2. Verifies disjoint record ID sets across chunks (detects duplicate or overlapping chunk processing).
3. Deterministically sorts merged records in ascending Record Number order.
4. Computes SHA-256 cryptographic checksum of the merged output file.
5. Automatically invokes the independent validation suite (validate_submission)
   to ensure 100% compliance with official competition submission specifications.
"""

import argparse
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Set utf-8 encoding for standard outputs if available
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.submission.validate_final_submission import (
    OFFICIAL_HEADER,
    OFFICIAL_HEADER_LINE,
    compute_sha256,
    print_validation_report,
    validate_submission,
)


def merge_chunks(
    chunk_paths: List[Path],
    output_path: Path,
    test_reference_path: Path = None,
    sort_output: bool = True,
    strict_disjoint: bool = True,
) -> Tuple[bool, dict]:
    """
    Merge a list of chunk TSV files into output_path.

    Returns:
        (success: bool, stats: dict)
    """
    if not chunk_paths:
        raise ValueError("No chunk files provided to merge!")

    print("=" * 72)
    print("SUBMISSION CHUNK MERGER & INTEGRITY RECOVERY")
    print("=" * 72)
    print(f"Chunks to merge: {len(chunk_paths)}")
    print(f"Target output:   {output_path}")

    # Track record IDs per chunk to detect accidental overlap
    records_per_chunk: Dict[str, Set[int]] = {}
    all_rows: List[Tuple[int, int, str, str]] = []
    chunk_summaries: List[dict] = []

    for chunk_idx, c_path in enumerate(chunk_paths, start=1):
        c_path = Path(c_path)
        if not c_path.exists():
            raise FileNotFoundError(f"Chunk file does not exist: {c_path}")

        print(f"\n[{chunk_idx}/{len(chunk_paths)}] Reading chunk: {c_path.name}")
        chunk_rec_ids: Set[int] = set()
        chunk_rows = 0

        with open(c_path, "r", encoding="utf-8", errors="strict") as f:
            header_line = f.readline().rstrip("\r\n")
            if header_line.split("\t") != OFFICIAL_HEADER:
                raise ValueError(
                    f"Chunk {c_path.name} has invalid header:\n  Found:    '{header_line}'\n  Expected: '{OFFICIAL_HEADER_LINE}'"
                )

            for line_no, line in enumerate(f, start=2):
                stripped = line.rstrip("\r\n")
                if not stripped:
                    continue
                parts = stripped.split("\t")
                if len(parts) != 4:
                    raise ValueError(
                        f"Chunk {c_path.name} line {line_no} malformed: expected 4 fields, got {len(parts)}."
                    )
                rec_id = int(parts[0])
                cat_id = int(parts[1])
                asp_name = parts[2]
                asp_val = parts[3]

                chunk_rec_ids.add(rec_id)
                all_rows.append((rec_id, cat_id, asp_name, asp_val))
                chunk_rows += 1

        records_per_chunk[c_path.name] = chunk_rec_ids
        print(f"  Rows: {chunk_rows:,} | Unique records with entities: {len(chunk_rec_ids):,}")
        if chunk_rec_ids:
            print(f"  Record range: [{min(chunk_rec_ids)}, {max(chunk_rec_ids)}]")

        chunk_summaries.append({
            "chunk": c_path.name,
            "rows": chunk_rows,
            "unique_records": len(chunk_rec_ids),
            "min_rec": min(chunk_rec_ids) if chunk_rec_ids else None,
            "max_rec": max(chunk_rec_ids) if chunk_rec_ids else None,
        })

    # Disjointness check
    print("-" * 72)
    print("VERIFYING CHUNK INDEPENDENCE & RECORD DISJOINTNESS...")
    overlap_detected = False
    chunk_names = list(records_per_chunk.keys())
    for i in range(len(chunk_names)):
        for j in range(i + 1, len(chunk_names)):
            c1, c2 = chunk_names[i], chunk_names[j]
            intersection = records_per_chunk[c1].intersection(records_per_chunk[c2])
            if intersection:
                overlap_detected = True
                msg = f"OVERLAP DETECTED: Chunk '{c1}' and '{c2}' share {len(intersection)} records (e.g. {sorted(intersection)[:5]}...)"
                print(f"  [ERROR] {msg}")
                if strict_disjoint:
                    raise ValueError(msg)

    if not overlap_detected:
        print("  Disjoint check passed: All chunks contain mutually exclusive record IDs.")

    # Sorting
    if sort_output:
        print("-" * 72)
        print("Sorting merged rows by Record Number (ascending)...")
        # Primary sort: Record Number; Secondary: Aspect Name; Tertiary: Aspect Value
        all_rows.sort(key=lambda x: (x[0], x[2], x[3]))

    # Write merged TSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(all_rows):,} merged rows to {output_path}...")
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(OFFICIAL_HEADER_LINE + "\n")
        for rec_id, cat_id, asp_name, asp_val in all_rows:
            f.write(f"{rec_id}\t{cat_id}\t{asp_name}\t{asp_val}\n")

    merged_sha256 = compute_sha256(output_path)
    file_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Merged file written successfully ({file_mb:.2f} MB, SHA-256: {merged_sha256}).")

    # Run independent validation
    print("-" * 72)
    print("RUNNING INDEPENDENT VALIDATOR ON MERGED OUTPUT...")
    is_valid, report = validate_submission(
        submission_path=output_path,
        test_path=test_reference_path,
    )
    print_validation_report(report)

    return is_valid, {
        "output_path": str(output_path),
        "sha256": merged_sha256,
        "total_chunks": len(chunk_paths),
        "total_merged_rows": len(all_rows),
        "is_valid": is_valid,
        "chunk_summaries": chunk_summaries,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Safe Chunk Merging & Integrity Tool for eBay ML Final Test"
    )
    parser.add_argument(
        "--chunk-files",
        type=Path,
        nargs="+",
        help="Explicit list of chunk TSV files to merge in order",
    )
    parser.add_argument(
        "--chunk-dir",
        type=Path,
        default=None,
        help="Directory containing chunk files",
    )
    parser.add_argument(
        "--chunk-pattern",
        type=str,
        default="*.tsv",
        help="Glob pattern to find chunk files within --chunk-dir (default: '*.tsv')",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Destination path for merged TSV submission",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=None,
        help="Optional test reference file to cross-validate record IDs and categories",
    )
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Allow overlapping record IDs across chunks (default: disallow and fail)",
    )
    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="Do not sort rows by record number (maintain chunk order)",
    )

    args = parser.parse_args()

    # Collect chunk paths
    chunk_paths: List[Path] = []
    if args.chunk_files:
        chunk_paths.extend(args.chunk_files)
    elif args.chunk_dir:
        pattern = str(args.chunk_dir / args.chunk_pattern)
        matched = sorted(glob.glob(pattern))
        chunk_paths.extend([Path(p) for p in matched])
    else:
        print("Error: Specify either --chunk-files or --chunk-dir")
        sys.exit(1)

    if not chunk_paths:
        print(f"Error: No chunk files found matching specifications!")
        sys.exit(1)

    is_valid, _ = merge_chunks(
        chunk_paths=chunk_paths,
        output_path=args.output_file,
        test_reference_path=args.test_file,
        sort_output=not args.no_sort,
        strict_disjoint=not args.allow_overlap,
    )

    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
