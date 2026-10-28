"""
Phase 33B: Unlabeled Dataset Sampler with Leakage Controls
"""

import gzip
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_examples import build_examples, load_training_data
from src.data.make_internal_split import normalize_title_for_grouping

DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
RAW_FILE = DATA_DIR / "raw" / "Listing_Titles.tsv.gz"
MANIFEST_FILE = CACHE_DIR / "dapt_50k_sample_manifest.json"


def build_exclusion_set() -> Tuple[Set[int], Set[str], dict]:
    """
    Build complete set of forbidden record IDs and forbidden normalized title strings.
    """
    # 1. Forbidden Record IDs:
    # All 1..5000 (labeled train/dev/val) and 5001..30000 (Quiz evaluation)
    forbidden_max_record_id = 30000

    # 2. Normalized titles from labeled dataset (all 5000 records)
    df = load_training_data()
    labeled_examples = build_examples(df)

    forbidden_normalized_titles = set()
    for ex in labeled_examples:
        norm = normalize_title_for_grouping(ex["tokens"])
        forbidden_normalized_titles.add(norm)

    # 3. Check internal dev split (800 titles) specifically
    dev_split_path = CACHE_DIR / "internal_split_3200_800.json"
    internal_dev_count = 0
    if dev_split_path.exists():
        with open(dev_split_path, "r", encoding="utf-8") as f:
            dev_manifest = json.load(f)
            internal_dev_count = len(dev_manifest.get("internal_dev_indices", []))

    stats = {
        "forbidden_max_record_id": forbidden_max_record_id,
        "total_labeled_titles_in_exclusion": len(labeled_examples),
        "unique_normalized_forbidden_titles": len(forbidden_normalized_titles),
        "internal_dev_count": internal_dev_count,
    }
    return forbidden_max_record_id, forbidden_normalized_titles, stats


def sample_dapt_corpus(
    sample_size_per_category: int = 25000,
    random_seed: int = 42,
) -> Tuple[List[dict], dict]:
    """
    Deterministically sample unlabeled titles from Listing_Titles.tsv.gz with strict leakage controls.
    """
    forbidden_max_rec, forbidden_titles, excl_stats = build_exclusion_set()

    rng = random.Random(random_seed)

    cat1_samples: List[dict] = []
    cat2_samples: List[dict] = []

    total_scanned = 0
    skipped_record_id = 0
    skipped_normalized_match = 0
    skipped_too_short = 0

    print(f"Streaming from {RAW_FILE}...")
    print(f"Exclusion criteria: Record ID <= {forbidden_max_rec} OR normalized title in forbidden set ({len(forbidden_titles)} unique titles)")

    with gzip.open(RAW_FILE, "rt", encoding="utf-8", errors="replace") as f:
        _ = f.readline()  # header: Record Number\tCategory\tTitle
        for line in f:
            total_scanned += 1
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 3:
                continue

            try:
                rec_id = int(parts[0])
                cat_id = int(parts[1])
            except ValueError:
                continue

            # Check 1: Record ID exclusion (strictly > 30000)
            if rec_id <= forbidden_max_rec:
                skipped_record_id += 1
                continue

            title = "\t".join(parts[2:]).strip()
            tokens = title.split()
            if len(tokens) < 2:  # skip single-word or empty titles
                skipped_too_short += 1
                continue

            # Check 2: Normalized title string collision
            norm_title = normalize_title_for_grouping(tokens)
            if norm_title in forbidden_titles:
                skipped_normalized_match += 1
                continue

            item = {
                "record_id": rec_id,
                "category": cat_id,
                "tokens": tokens,
            }

            # Reservoir / Category allocation with probability sampling
            # Since data is plentiful (~1.97M records), we sample deterministically
            if cat_id == 1:
                if len(cat1_samples) < sample_size_per_category:
                    cat1_samples.append(item)
                else:
                    # Reservoir replacement with probability
                    j = rng.randint(0, total_scanned)
                    if j < sample_size_per_category:
                        cat1_samples[j] = item
            elif cat_id == 2:
                if len(cat2_samples) < sample_size_per_category:
                    cat2_samples.append(item)
                else:
                    j = rng.randint(0, total_scanned)
                    if j < sample_size_per_category:
                        cat2_samples[j] = item

            # Stop once both reservoirs have seen sufficient candidate flow (e.g. 150k candidates)
            if len(cat1_samples) >= sample_size_per_category and len(cat2_samples) >= sample_size_per_category and total_scanned >= 150000:
                print(f"Sample quota reached at line {total_scanned:,}.")
                break

    # Combine and shuffle deterministically
    combined = cat1_samples + cat2_samples
    rng.shuffle(combined)

    sampling_stats = {
        "random_seed": random_seed,
        "sample_size_total": len(combined),
        "sample_size_cat1": len(cat1_samples),
        "sample_size_cat2": len(cat2_samples),
        "total_lines_scanned": total_scanned,
        "skipped_due_to_record_id_le_30000": skipped_record_id,
        "skipped_due_to_normalized_eval_title_match": skipped_normalized_match,
        "skipped_too_short": skipped_too_short,
        "min_sampled_record_id": min(x["record_id"] for x in combined),
        "max_sampled_record_id": max(x["record_id"] for x in combined),
        "mean_token_length": round(sum(len(x["tokens"]) for x in combined) / len(combined), 2),
    }

    return combined, sampling_stats


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    samples, stats = sample_dapt_corpus(sample_size_per_category=25000, random_seed=42)

    print("\n--- SAMPLING SUMMARY ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Save manifest (record_id, category, token_count) WITHOUT exposing raw title text
    manifest_records = [
        {"record_id": x["record_id"], "category": x["category"], "token_count": len(x["tokens"])}
        for x in samples
    ]

    manifest_data = {
        "stats": stats,
        "manifest": manifest_records,
    }

    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    # Compute manifest SHA256
    h = hashlib.sha256()
    with open(MANIFEST_FILE, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    manifest_hash = h.hexdigest().upper()

    print(f"\nManifest saved to: {MANIFEST_FILE}")
    print(f"Manifest SHA-256:  {manifest_hash}")


if __name__ == "__main__":
    main()
