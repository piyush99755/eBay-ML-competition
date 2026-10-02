"""
This script is designed to produce privacy-safe aggregate dataset information for local inspection without exposing restricted eBay competition records.
"""

import os
from pathlib import Path
import pandas as pd


def inspect_file(filepath: Path) -> None:
    """Inspects a single file in data/raw/ and prints safe aggregate statistics."""
    print("\n" + "=" * 80)
    print(f"FILE: {filepath.name}")
    print(f"PATH: {filepath}")
    print(f"EXTENSION: {filepath.suffix}")
    
    # Calculate file size in Megabytes (MB)
    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"SIZE: {size_mb:.2f} MB")
    print("=" * 80)

    # Check extension for special handling
    ext_lower = "".join(filepath.suffixes).lower()

    # Rule 8: PDF files - metadata only, no text parsing
    if ext_lower.endswith(".pdf"):
        print("[PDF Document] Metadata only (filename and size reported). Document content omitted for privacy.")
        return

    # Skip non-tabular files
    if not (ext_lower.endswith(".tsv") or ext_lower.endswith(".tsv.gz") or ext_lower.endswith(".csv") or ext_lower.endswith(".csv.gz")):
        print(f"[Non-Tabular File] Skipping table inspection for extension '{ext_lower}'.")
        return

    # Rule 3 & 7: Read tabular files directly with pandas
    try:
        sep = "\t" if "tsv" in ext_lower else ","
        df = pd.read_csv(
            filepath,
            sep=sep,
            compression="infer",
            low_memory=False,
            keep_default_na=False,
            na_values=None,
        )
    except Exception as e:
        print(f"[ERROR] Could not load tabular file '{filepath.name}': {e}")
        return

    # Section 1: Schema and Shape
    print("\n--- SCHEMA & SHAPE ---")
    print(f"Total Rows: {len(df):,}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Duplicate Rows Count: {df.duplicated().sum():,}")

    # Section 2: Columns, Data Types, and Missing Values
    print("\n--- COLUMNS, DATA TYPES & MISSING VALUES ---")
    missing_counts = df.isnull().sum()
    for col in df.columns:
        missing_cnt = missing_counts[col]
        missing_pct = (missing_cnt / len(df)) * 100 if len(df) > 0 else 0.0
        print(f"  • {col}: dtype={df[col].dtype}, missing={missing_cnt:,} ({missing_pct:.2f}%)")

    # Section 3: Column-Specific Privacy-Safe Aggregates
    for col in df.columns:
        col_lower = col.lower()

        # Aspect name column (Schema labels are safe to print)
        if "aspect" in col_lower and any(term in col_lower for term in ["name", "key", "label"]):
            print(f"\n--- ASPECT NAMES DISTRIBUTION ({col}) ---")
            print(f"Unique aspect names count: {df[col].nunique():,}")
            print("Aspect name frequency counts:")
            freq = df[col].value_counts().dropna()
            for name, count in freq.items():
                print(f"  - {name}: {count:,}")
            continue

        # Aspect value column (Skip printing individual values to protect privacy)
        if "aspect" in col_lower and any(term in col_lower for term in ["value", "val"]):
            print(f"\n--- ASPECT VALUES AGGREGATE ({col}) ---")
            print(f"Unique aspect values count: {df[col].nunique():,}")
            print("(Individual record aspect values omitted for privacy)")
            continue

        # Title or raw text columns (Aggregate length & word count stats only)
        if any(term in col_lower for term in ["title", "text", "description", "content", "raw"]):
            print(f"\n--- TEXT COLUMN AGGREGATES ({col}) ---")
            non_null_text = df[col].dropna().astype(str)
            print(f"Non-null count: {len(non_null_text):,}")
            if len(non_null_text) > 0:
                char_lens = non_null_text.str.len()
                word_counts = non_null_text.str.split().str.len()
                print(f"Character Length -> Min: {char_lens.min()}, Max: {char_lens.max()}, Mean: {char_lens.mean():.2f}, Median: {char_lens.median():.2f}")
                print(f"Word Count       -> Min: {word_counts.min()}, Max: {word_counts.max()}, Mean: {word_counts.mean():.2f}, Median: {word_counts.median():.2f}")
            print("(Raw item text content omitted for privacy)")
            continue

        # ID or Category columns (Distribution counts for non-text categories/IDs)
        if any(term in col_lower for term in ["id", "category", "cat", "class", "label"]):
            print(f"\n--- IDENTIFIER / CATEGORY COLUMN ({col}) ---")
            unique_cnt = df[col].nunique()
            print(f"Unique count: {unique_cnt:,}")
            # Print value counts if values are numeric or short alphanumeric IDs
            if pd.api.types.is_numeric_dtype(df[col]) or (unique_cnt <= 100 and df[col].dropna().astype(str).str.isalnum().all()):
                print("Category/ID value distribution:")
                freq = df[col].value_counts().dropna()
                for val, count in freq.head(50).items():
                    print(f"  - {val}: {count:,}")
                if len(freq) > 50:
                    print(f"  ... ({len(freq) - 50} additional unique IDs)")
            continue

    # Training file specific inspection for Tagged_Titles_Train.tsv.gz
    if filepath.name == "Tagged_Titles_Train.tsv.gz":
        if "Record Number" in df.columns:
            print("\n--- RECORD-LEVEL AGGREGATES ---")
            unique_records = df["Record Number"].nunique()
            print(f"Unique Record Numbers: {unique_records:,}")

            rows_per_record = df.groupby("Record Number").size()
            print("Rows/Tokens per record statistics:")
            print(f"  - Min: {rows_per_record.min()}")
            print(f"  - Max: {rows_per_record.max()}")
            print(f"  - Mean: {rows_per_record.mean():.2f}")
            print(f"  - Median: {rows_per_record.median():.2f}")

            if "Category" in df.columns:
                records_per_cat = (
                    df[["Record Number", "Category"]]
                    .drop_duplicates()
                    .groupby("Category")
                    .size()
                )
                print("\nUnique Records per Category:")
                for cat, count in records_per_cat.items():
                    print(f"  - Category {cat}: {count:,} records")

        if "Tag" in df.columns:
            print("\n--- TAG AGGREGATES ---")
            unique_tags = df["Tag"].nunique()
            print(f"Unique Non-Null Tags: {unique_tags:,}")
            print("Tag frequency distribution:")
            tag_counts = df["Tag"].value_counts(dropna=False)
            for tag, count in tag_counts.items():
                print(f"  - {tag}: {count:,}")



def main() -> None:
    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        print(f"[WARNING] Directory '{raw_dir}' does not exist.")
        return

    print("=" * 80)
    print("        PRIVACY-SAFE LOCAL DATASET INSPECTOR - eBAY ML COMPETITION")
    print("=" * 80)

    # Discover all files recursively under data/raw/
    files = sorted([p for p in raw_dir.rglob("*") if p.is_file()])

    if not files:
        print(f"No files found under '{raw_dir}'.")
        print("Please place the local competition files into data/raw/ to inspect them.")
        return

    print(f"Found {len(files)} file(s) under '{raw_dir}':\n")
    for f in files:
        rel_path = f.relative_to(raw_dir)
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  • {rel_path} ({size_mb:.2f} MB)")

    # Process each discovered file
    for f in files:
        inspect_file(f)

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE - Safe aggregate statistics generated without data leakage.")
    print("=" * 80)


if __name__ == "__main__":
    main()
