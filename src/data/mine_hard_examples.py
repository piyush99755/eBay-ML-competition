"""
Phase 48: Training-Only Hard Example Mining.

Uses supervised training annotations (4,000 Split-A training titles) and
genuinely out-of-fold (OOF) 3-fold GBERT predictions from Phase 29B
(data/cache/oof_probs_gbert_fold*.npy) to identify and categorize difficult titles.

Strict Data Provenance:
- 1,000-title validation set (and 489 clean holdout) is STRICTLY EXCLUDED.
- Quiz and Test data are STRICTLY EXCLUDED.
- Zero data leakage: each example was evaluated ONLY by a model that never trained on it.
"""

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_examples import build_examples, load_training_data
from src.data.make_splits import split_examples
from src.data.make_cv_folds import get_cv_folds
from src.data.label_mapping import create_label_mappings
from src.evaluation.entity_metrics import (
    bio_to_entities_with_confidence,
    filter_entities_by_category,
    count_matches,
)

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
HARD_EXAMPLES_OUT_PATH = CACHE_DIR / "phase_48_hard_examples_manifest.json"

THRESHOLDS = {
    "Kompatibles_Fahrzeug_Modell": 0.80,
    "Produktart": 0.80,
    "Im_Lieferumfang_Enthalten": 0.80,
}
DEFAULT_THRESH = 0.60


def bio_to_spanned_entities(
    tokens: List[str],
    bio_tags: List[str],
    probs: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Convert BIO sequence to entity dictionaries with explicit token spans [start, end)."""
    entities = []
    current_asp = None
    current_start = -1
    current_tokens = []
    current_probs = []

    for idx, (tok, tag) in enumerate(zip(tokens, bio_tags)):
        p = probs[idx] if probs is not None else 1.0

        if tag == "O":
            if current_asp is not None:
                entities.append({
                    "aspect": current_asp,
                    "value": " ".join(current_tokens),
                    "start": current_start,
                    "end": idx,
                    "min_conf": float(np.min(current_probs)),
                    "mean_conf": float(np.mean(current_probs)),
                })
                current_asp = None
                current_tokens = []
                current_probs = []
            continue

        if tag.startswith("B-"):
            if current_asp is not None:
                entities.append({
                    "aspect": current_asp,
                    "value": " ".join(current_tokens),
                    "start": current_start,
                    "end": idx,
                    "min_conf": float(np.min(current_probs)),
                    "mean_conf": float(np.mean(current_probs)),
                })
            current_asp = tag[2:]
            current_start = idx
            current_tokens = [tok]
            current_probs = [p]
            continue

        if tag.startswith("I-"):
            asp = tag[2:]
            if current_asp == asp:
                current_tokens.append(tok)
                current_probs.append(p)
            else:
                if current_asp is not None:
                    entities.append({
                        "aspect": current_asp,
                        "value": " ".join(current_tokens),
                        "start": current_start,
                        "end": idx,
                        "min_conf": float(np.min(current_probs)),
                        "mean_conf": float(np.mean(current_probs)),
                    })
                current_asp = asp
                current_start = idx
                current_tokens = [tok]
                current_probs = [p]
            continue

    if current_asp is not None:
        entities.append({
            "aspect": current_asp,
            "value": " ".join(current_tokens),
            "start": current_start,
            "end": len(tokens),
            "min_conf": float(np.min(current_probs)),
            "mean_conf": float(np.mean(current_probs)),
        })

    return entities


def detect_invalid_bio_transitions(bio_tags: List[str]) -> int:
    """Count invalid BIO transitions in the predicted sequence."""
    invalid_count = 0
    prev_tag = "O"
    for tag in bio_tags:
        if tag.startswith("I-"):
            asp = tag[2:]
            if prev_tag == "O":
                invalid_count += 1
            elif prev_tag.startswith("B-") or prev_tag.startswith("I-"):
                prev_asp = prev_tag[2:]
                if prev_asp != asp:
                    invalid_count += 1
            else:
                invalid_count += 1
        prev_tag = tag
    return invalid_count


def audit_hard_examples() -> Dict[str, Any]:
    print("=" * 80)
    print("PHASE 48: TRAINING-ONLY HARD EXAMPLE MINING AUDIT")
    print("=" * 80)

    # 1. Load Data
    df = load_training_data()
    all_examples = build_examples(df)
    train_4k, val_1k = split_examples(all_examples, random_state=42)
    label2id, id2label = create_label_mappings(all_examples)
    folds = get_cv_folds(train_4k, n_splits=3, random_state=42)

    print(f"Total labeled dataset:       {len(all_examples)} titles")
    print(f"Training partition (Split A): {len(train_4k)} titles")
    print(f"Validation partition:        {len(val_1k)} titles (EXCLUDED FROM MINING)")

    # 2. Assemble OOF predictions
    oof_examples: List[Tuple[dict, np.ndarray, int]] = []
    for fold_idx in range(3):
        _, val_f = folds[fold_idx]
        oof_path = CACHE_DIR / f"oof_probs_gbert_fold{fold_idx}.npy"
        assert oof_path.exists(), f"Missing OOF cache: {oof_path}"
        flat_probs = np.load(oof_path)
        splits = np.cumsum([len(e["tokens"]) for e in val_f])[:-1]
        fold_probs = np.split(flat_probs, splits)
        assert len(fold_probs) == len(val_f)
        for ex, p in zip(val_f, fold_probs):
            oof_examples.append((ex, p, fold_idx))

    assert len(oof_examples) == 4000, f"Expected 4,000 OOF examples, got {len(oof_examples)}"
    print(f"Successfully aligned all {len(oof_examples)} OOF predictions across 3 folds.")

    # 3. Categorize errors per title
    title_analyses = []
    error_type_counts = Counter()
    aspect_error_counts = defaultdict(lambda: {"fp": 0, "fn": 0, "boundary": 0, "aspect_confusion": 0})
    category_error_counts = {1: {"total": 0, "perfect": 0, "imperfect": 0}, 2: {"total": 0, "perfect": 0, "imperfect": 0}}

    for ex, p, fold_idx in oof_examples:
        tokens = ex["tokens"]
        cat_id = ex["category"]
        category_error_counts[cat_id]["total"] += 1

        pred_ids = np.argmax(p, axis=-1)
        pred_bio = [id2label[pid] for pid in pred_ids]
        selected_probs = [float(p[t_idx, pred_ids[t_idx]]) for t_idx in range(len(tokens))]

        # Gold entities
        gold_spanned = bio_to_spanned_entities(tokens, ex["bio_tags"])
        gold_valid = [e for e in gold_spanned if filter_entities_by_category([(e["aspect"], e["value"])], cat_id)]

        # Raw predicted entities
        raw_pred_spanned = bio_to_spanned_entities(tokens, pred_bio, selected_probs)
        raw_valid = [e for e in raw_pred_spanned if filter_entities_by_category([(e["aspect"], e["value"])], cat_id)]

        # Calibrated predicted entities (Phase 40 thresholds)
        cal_valid = []
        for e in raw_valid:
            thresh = THRESHOLDS.get(e["aspect"], DEFAULT_THRESH)
            if e["min_conf"] >= thresh:
                cal_valid.append(e)

        # Match counting on calibrated predictions
        gold_tuples = [(e["aspect"], e["value"]) for e in gold_valid]
        cal_tuples = [(e["aspect"], e["value"]) for e in cal_valid]
        tp, fp, fn = count_matches(cal_tuples, gold_tuples)

        # Fine-grained error diagnosis
        invalid_bio_count = detect_invalid_bio_transitions(pred_bio)
        has_invalid_bio = invalid_bio_count > 0

        # Boundary, confusion, and span tracking
        matched_gold_indices = set()
        matched_cal_indices = set()

        # Step A: Find exact matches
        for g_idx, g in enumerate(gold_valid):
            for c_idx, c in enumerate(cal_valid):
                if c_idx not in matched_cal_indices and g_idx not in matched_gold_indices:
                    if g["start"] == c["start"] and g["end"] == c["end"] and g["aspect"] == c["aspect"]:
                        matched_gold_indices.add(g_idx)
                        matched_cal_indices.add(c_idx)

        # Step B: Check wrong aspect classification (exact span, different aspect)
        aspect_confusions = []
        for g_idx, g in enumerate(gold_valid):
            if g_idx in matched_gold_indices:
                continue
            for c_idx, c in enumerate(cal_valid):
                if c_idx not in matched_cal_indices:
                    if g["start"] == c["start"] and g["end"] == c["end"] and g["aspect"] != c["aspect"]:
                        aspect_confusions.append((g["aspect"], c["aspect"], g["value"]))
                        matched_gold_indices.add(g_idx)
                        matched_cal_indices.add(c_idx)
                        aspect_error_counts[g["aspect"]]["aspect_confusion"] += 1
                        break

        # Step C: Check boundary errors (same aspect, overlapping span, unequal boundaries)
        boundary_errors = []
        for g_idx, g in enumerate(gold_valid):
            if g_idx in matched_gold_indices:
                continue
            for c_idx, c in enumerate(cal_valid):
                if c_idx not in matched_cal_indices:
                    if g["aspect"] == c["aspect"]:
                        # Check overlap: max(start) < min(end)
                        if max(g["start"], c["start"]) < min(g["end"], c["end"]):
                            boundary_errors.append({
                                "aspect": g["aspect"],
                                "gold_val": g["value"],
                                "pred_val": c["value"],
                                "gold_span": [g["start"], g["end"]],
                                "pred_span": [c["start"], c["end"]],
                            })
                            matched_gold_indices.add(g_idx)
                            matched_cal_indices.add(c_idx)
                            aspect_error_counts[g["aspect"]]["boundary"] += 1
                            break

        # Step D: Unmatched gold entities (FNs)
        unmatched_gold = [gold_valid[i] for i in range(len(gold_valid)) if i not in matched_gold_indices]
        for g in unmatched_gold:
            aspect_error_counts[g["aspect"]]["fn"] += 1

        # Step E: Unmatched predicted entities (FPs)
        unmatched_cal = [cal_valid[i] for i in range(len(cal_valid)) if i not in matched_cal_indices]
        for c in unmatched_cal:
            aspect_error_counts[c["aspect"]]["fp"] += 1

        # Specific high-priority error flags
        missed_vehicle_model = any(g["aspect"] == "Kompatibles_Fahrzeug_Modell" for g in unmatched_gold)
        has_vehicle_boundary = any(b["aspect"] == "Kompatibles_Fahrzeug_Modell" for b in boundary_errors)
        part_number_errors = any(
            g["aspect"] in ("Herstellernummer", "Oe/Oem_Referenznummer(N)") for g in unmatched_gold
        ) or any(
            c["aspect"] in ("Herstellernummer", "Oe/Oem_Referenznummer(N)") for c in unmatched_cal
        )

        # Flag title difficulty
        is_perfect = (fp == 0 and fn == 0)
        if is_perfect:
            category_error_counts[cat_id]["perfect"] += 1
        else:
            category_error_counts[cat_id]["imperfect"] += 1

        if has_invalid_bio:
            error_type_counts["invalid_bio_transitions"] += 1
        if len(boundary_errors) > 0:
            error_type_counts["boundary_errors"] += 1
        if len(aspect_confusions) > 0:
            error_type_counts["aspect_confusions"] += 1
        if missed_vehicle_model:
            error_type_counts["missed_vehicle_model"] += 1
        if has_vehicle_boundary:
            error_type_counts["vehicle_boundary_errors"] += 1
        if part_number_errors:
            error_type_counts["part_number_errors"] += 1
        if len(unmatched_cal) > 0:
            error_type_counts["false_positive_hallucinations"] += 1
        if len(unmatched_gold) > 0:
            error_type_counts["missed_entities_fn"] += 1

        # Difficulty Score: weighted by F0.2 penalty (FP cost ~ 5x FN cost in typical training loss weighting)
        # Precision penalty is heavy: FP cost = 5, FN cost = 1, boundary error = 3, aspect confusion = 4
        difficulty_score = (
            5 * fp +
            1 * fn +
            3 * len(boundary_errors) +
            4 * len(aspect_confusions) +
            (4 if missed_vehicle_model else 0) +
            (3 if part_number_errors else 0)
        )

        title_analyses.append({
            "record_number": ex["record_number"],
            "category": cat_id,
            "fold_idx": fold_idx,
            "tokens": tokens,
            "is_perfect": is_perfect,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "difficulty_score": difficulty_score,
            "boundary_errors": boundary_errors,
            "aspect_confusions": aspect_confusions,
            "unmatched_gold": [(g["aspect"], g["value"]) for g in unmatched_gold],
            "unmatched_cal": [(c["aspect"], c["value"]) for c in unmatched_cal],
            "invalid_bio_count": invalid_bio_count,
        })

    # Summary Statistics
    total_titles = len(title_analyses)
    total_imperfect = sum(1 for t in title_analyses if not t["is_perfect"])
    total_perfect = total_titles - total_imperfect

    print(f"\nOverall Training Data Partition (4,000 titles):")
    print(f"  Perfect Titles (0 FP, 0 FN):   {total_perfect} ({total_perfect / total_titles * 100:.1f}%)")
    print(f"  Imperfect Titles (>=1 Error):  {total_imperfect} ({total_imperfect / total_titles * 100:.1f}%)")

    print("\nCategory Distribution of Errors:")
    for cat_id in [1, 2]:
        cat_stat = category_error_counts[cat_id]
        print(f"  Category {cat_id}: Total={cat_stat['total']}, Perfect={cat_stat['perfect']} ({cat_stat['perfect']/cat_stat['total']*100:.1f}%), Imperfect={cat_stat['imperfect']} ({cat_stat['imperfect']/cat_stat['total']*100:.1f}%)")

    print("\nError Category Counts (titles exhibiting error):")
    for err_name, count in sorted(error_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {err_name:<32} {count:>5} titles ({count / total_titles * 100:.1f}%)")

    print("\nTop 10 Aspects by Error Frequency:")
    sorted_aspects = sorted(
        aspect_error_counts.items(),
        key=lambda x: -(x[1]["fp"] + x[1]["fn"] + x[1]["boundary"] + x[1]["aspect_confusion"])
    )
    for asp, d in sorted_aspects[:10]:
        total_err = d["fp"] + d["fn"] + d["boundary"] + d["aspect_confusion"]
        print(f"  {asp:<32} Total Errors={total_err:>4} (FP={d['fp']}, FN={d['fn']}, Boundary={d['boundary']}, AspectConfusion={d['aspect_confusion']})")

    # Define Hard Examples Subset
    # Predeclare sampling design:
    # A title is deemed "HARD" if difficulty_score >= 6 (representing compound errors, boundary failures, or multiple FPs)
    hard_titles = [t for t in title_analyses if t["difficulty_score"] >= 6]
    hard_record_ids = set(t["record_number"] for t in hard_titles)

    print(f"\nHard Example Selection:")
    print(f"  Difficulty Score Threshold: >= 6")
    print(f"  Total Hard Titles Selected: {len(hard_titles)} ({len(hard_titles) / total_titles * 100:.1f}% of training data)")
    hard_cat1 = sum(1 for t in hard_titles if t["category"] == 1)
    hard_cat2 = sum(1 for t in hard_titles if t["category"] == 2)
    print(f"  Category Distribution of Hard Titles: Category 1: {hard_cat1} ({hard_cat1 / len(hard_titles) * 100:.1f}%), Category 2: {hard_cat2} ({hard_cat2 / len(hard_titles) * 100:.1f}%)")

    # Sample Difficult Patterns
    print("\nSample Difficult Patterns (5 examples):")
    for sample in hard_titles[:5]:
        print(f"  [Record {sample['record_number']}, Cat {sample['category']}, Score {sample['difficulty_score']}]:")
        print(f"    Title: {' '.join(sample['tokens'])}")
        if sample["boundary_errors"]:
            print(f"    Boundary Errors:   {sample['boundary_errors']}")
        if sample["aspect_confusions"]:
            print(f"    Aspect Confusions: {sample['aspect_confusions']}")
        if sample["unmatched_gold"]:
            print(f"    Missed (FN):       {sample['unmatched_gold']}")
        if sample["unmatched_cal"]:
            print(f"    False Pos (FP):    {sample['unmatched_cal']}")

    # Save manifest
    manifest = {
        "phase": "48",
        "description": "Training-only hard example mining using 3-fold OOF GBERT predictions",
        "total_train_titles": total_titles,
        "perfect_titles": total_perfect,
        "imperfect_titles": total_imperfect,
        "hard_titles_count": len(hard_titles),
        "hard_record_ids": sorted(list(hard_record_ids)),
        "category_distribution": {1: hard_cat1, 2: hard_cat2},
        "error_type_distribution": dict(error_type_counts),
        "aspect_error_distribution": {k: dict(v) for k, v in aspect_error_counts.items()},
        "sample_difficult_titles": hard_titles[:10],
    }

    # Clean helper for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        elif isinstance(obj, tuple):
            return [convert_to_serializable(i) for i in obj]
        return obj

    manifest = convert_to_serializable(manifest)

    HARD_EXAMPLES_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HARD_EXAMPLES_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nManifest saved to {HARD_EXAMPLES_OUT_PATH}")

    return manifest


if __name__ == "__main__":
    audit_hard_examples()
