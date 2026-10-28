"""
Phase 32C: Deterministic 3,200 / 800 Internal Train/Dev Partitioning.

Partitions the 4,000 Split-A training titles into:
- 3,200 Internal Train titles (50% Category 1, 50% Category 2)
- 800 Internal Dev titles (50% Category 1, 50% Category 2)

Features:
- StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
- Groups near-duplicate / identical titles together to prevent train/dev leakage.
- Completely disjoint from the 1,000 Split-A validation holdout.
- Saves an index manifest to data/cache/internal_split_3200_800.json.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_examples import build_examples, load_training_data
from src.data.make_splits import split_examples

CACHE_DIR = PROJECT_ROOT / "data" / "cache"


def normalize_title_for_grouping(tokens: List[str]) -> str:
    """Normalize title for grouping identical / near-duplicate listing titles."""
    text = " ".join(tokens).lower()
    return re.sub(r"\s+", " ", text).strip()


def get_internal_split(
    train_examples: List[dict],
    random_state: int = 42,
) -> Tuple[List[dict], List[dict], dict]:
    """
    Produce deterministic 3,200 internal train / 800 internal dev split.
    Returns:
        (internal_train_examples, internal_dev_examples, manifest)
    """
    categories = [int(ex["category"]) for ex in train_examples]
    groups = [normalize_title_for_grouping(ex["tokens"]) for ex in train_examples]

    sgkf = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    # Use first split for our 80/20 partition
    for train_indices, dev_indices in sgkf.split(train_examples, categories, groups):
        train_part = [train_examples[i] for i in train_indices]
        dev_part = [train_examples[i] for i in dev_indices]

        manifest = {
            "random_state": random_state,
            "train_indices": [int(i) for i in train_indices],
            "dev_indices": [int(i) for i in dev_indices],
            "train_size": len(train_indices),
            "dev_size": len(dev_indices),
            "train_categories": dict(Counter([categories[i] for i in train_indices])),
            "dev_categories": dict(Counter([categories[i] for i in dev_indices])),
        }
        break

    return train_part, dev_part, manifest


def audit_internal_split(
    train_part: List[dict],
    dev_part: List[dict],
    holdout_examples: List[dict],
) -> dict:
    """Audit leakage, group consistency, and category balance."""
    train_rec_ids = {ex["record_number"] for ex in train_part}
    dev_rec_ids = {ex["record_number"] for ex in dev_part}
    holdout_rec_ids = {ex["record_number"] for ex in holdout_examples}

    # 1. Disjointness check
    internal_overlap = train_rec_ids.intersection(dev_rec_ids)
    train_holdout_overlap = train_rec_ids.intersection(holdout_rec_ids)
    dev_holdout_overlap = dev_rec_ids.intersection(holdout_rec_ids)

    assert len(internal_overlap) == 0, f"Internal overlap detected: {len(internal_overlap)} IDs!"
    assert len(train_holdout_overlap) == 0, f"Train-holdout overlap detected: {len(train_holdout_overlap)} IDs!"
    assert len(dev_holdout_overlap) == 0, f"Dev-holdout overlap detected: {len(dev_holdout_overlap)} IDs!"

    # 2. Group check (near-duplicates)
    train_groups = {normalize_title_for_grouping(ex["tokens"]) for ex in train_part}
    dev_groups = {normalize_title_for_grouping(ex["tokens"]) for ex in dev_part}
    group_overlap = train_groups.intersection(dev_groups)

    assert len(group_overlap) == 0, f"Near-duplicate title overlap across train/dev: {len(group_overlap)} titles!"

    # 3. Category distribution
    train_cats = Counter([ex["category"] for ex in train_part])
    dev_cats = Counter([ex["category"] for ex in dev_part])

    return {
        "train_size": len(train_part),
        "dev_size": len(dev_part),
        "holdout_size": len(holdout_examples),
        "internal_overlap": len(internal_overlap),
        "holdout_overlap": len(train_holdout_overlap) + len(dev_holdout_overlap),
        "group_overlap": len(group_overlap),
        "train_category_counts": {int(k): v for k, v in train_cats.items()},
        "dev_category_counts": {int(k): v for k, v in dev_cats.items()},
        "is_leakage_free": True,
    }


def main():
    print("=" * 70)
    print("PHASE 32C: INTERNAL TRAIN / DEV PARTITIONING AUDIT")
    print("=" * 70)

    df = load_training_data()
    all_examples = build_examples(df)
    train_examples, val_examples = split_examples(all_examples, random_state=42)

    train_part, dev_part, manifest = get_internal_split(train_examples, random_state=42)
    audit = audit_internal_split(train_part, dev_part, val_examples)

    print(f"Internal Train Size:    {audit['train_size']} titles")
    print(f"  - Category 1:         {audit['train_category_counts'][1]} ({audit['train_category_counts'][1]/audit['train_size']*100:.1f}%)")
    print(f"  - Category 2:         {audit['train_category_counts'][2]} ({audit['train_category_counts'][2]/audit['train_size']*100:.1f}%)")
    print(f"Internal Dev Size:      {audit['dev_size']} titles")
    print(f"  - Category 1:         {audit['dev_category_counts'][1]} ({audit['dev_category_counts'][1]/audit['dev_size']*100:.1f}%)")
    print(f"  - Category 2:         {audit['dev_category_counts'][2]} ({audit['dev_category_counts'][2]/audit['dev_size']*100:.1f}%)")
    print(f"Untouched Holdout:      {audit['holdout_size']} titles")
    print(f"Train/Dev Overlap:      {audit['internal_overlap']} records")
    print(f"Holdout Overlap:        {audit['holdout_overlap']} records")
    print(f"Group/Duplicate Overlap:{audit['group_overlap']} titles")
    print(f"Leakage-Free Verified:  {audit['is_leakage_free']}")

    # Save manifest locally
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = CACHE_DIR / "internal_split_3200_800.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
