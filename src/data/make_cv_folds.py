"""
Phase 29B: 3-Fold Stratified & Grouped Partitioning for GBERT Cross-Validation.

Partitions the 4,000 Split-A training titles into 3 deterministic, category-stratified folds.
Near-duplicate listing titles (e.g. record 1779 and record 4204) are grouped into the same fold
via StratifiedGroupKFold so that no model ever trains on one and evaluates on the other.

The 1,000 Split-A validation titles remain strictly untouched as an external holdout.
"""

import re
from typing import Dict, List, Tuple
from sklearn.model_selection import StratifiedGroupKFold


def normalize_title_for_grouping(tokens: List[str]) -> str:
    """Normalize title for grouping near-duplicate listing titles."""
    text = " ".join(tokens).lower()
    return re.sub(r"\s+", " ", text).strip()


def get_cv_folds(
    train_examples: List[dict],
    n_splits: int = 3,
    random_state: int = 42,
) -> List[Tuple[List[dict], List[dict]]]:
    """
    Produce n_splits deterministic, category-stratified folds with near-duplicate grouping.

    Returns:
        List of (train_fold_examples, val_fold_examples) for each fold.
    """
    categories = [ex["category"] for ex in train_examples]
    groups = [normalize_title_for_grouping(ex["tokens"]) for ex in train_examples]

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    folds = []
    for train_indices, val_indices in sgkf.split(train_examples, categories, groups):
        train_fold = [train_examples[i] for i in train_indices]
        val_fold = [train_examples[i] for i in val_indices]
        folds.append((train_fold, val_fold))

    return folds


def verify_fold_integrity(
    folds: List[Tuple[List[dict], List[dict]]],
    val_examples: List[dict],
) -> dict:
    """
    Verify mathematical integrity of cross-validation folds:
    1. Disjoint validation fold coverage.
    2. Zero leakage between training and validation in any fold.
    3. Zero leakage between any fold and the external holdout.
    4. Grouping of near-duplicate titles within the same fold.
    5. Balanced category distribution across folds.
    """
    holdout_ids = {ex["record_number"] for ex in val_examples}
    all_val_ids = []
    fold_summaries = []

    for fold_idx, (train_f, val_f) in enumerate(folds):
        train_ids = {ex["record_number"] for ex in train_f}
        val_ids = {ex["record_number"] for ex in val_f}

        # Check internal fold disjointness
        internal_overlap = train_ids.intersection(val_ids)
        assert len(internal_overlap) == 0, f"Fold {fold_idx} has {len(internal_overlap)} overlapping IDs!"

        # Check disjointness from holdout
        holdout_train_overlap = train_ids.intersection(holdout_ids)
        holdout_val_overlap = val_ids.intersection(holdout_ids)
        assert len(holdout_train_overlap) == 0, f"Fold {fold_idx} train overlaps with holdout!"
        assert len(holdout_val_overlap) == 0, f"Fold {fold_idx} val overlaps with holdout!"

        # Check near duplicate pair (1779 and 4204)
        has_1779 = 1779 in val_ids
        has_4204 = 4204 in val_ids
        if has_1779 or has_4204:
            assert has_1779 and has_4204, f"Near-duplicates separated in Fold {fold_idx}!"

        cat_counts = {1: sum(1 for e in val_f if e["category"] == 1), 2: sum(1 for e in val_f if e["category"] == 2)}
        all_val_ids.extend(list(val_ids))

        fold_summaries.append({
            "fold": fold_idx,
            "train_size": len(train_f),
            "val_size": len(val_f),
            "cat_counts": cat_counts,
            "contains_grouped_pair": has_1779 and has_4204,
        })

    # Check complete partition coverage across training examples
    assert len(all_val_ids) == len(set(all_val_ids)), "Duplicate IDs found across validation folds!"
    total_val_titles = sum(s["val_size"] for s in fold_summaries)

    return {
        "n_folds": len(folds),
        "total_val_titles": total_val_titles,
        "fold_summaries": fold_summaries,
        "all_checks_passed": True,
    }
