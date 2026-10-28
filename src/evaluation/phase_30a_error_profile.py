"""
Phase 30A: Error-Driven Precision Research & Scorer Diagnostic Profile.

Performs:
1. Token-level probability fusion of frozen champion (GBERT 0.40 + 3x XLM-R 0.20).
2. Category-aware post-filtering & entity extraction with min confidence.
3. Detailed per-aspect & per-category error taxonomy:
   - TP, FP, FN, Precision, Recall, F0.2
   - Error breakdown: Boundary mismatch vs Aspect misclassification vs Spurious insertion
   - Confidence distributions for TP vs FP
   - FP clustering in confidence buckets [0.60, 0.70), [0.70, 0.80), [0.80, 0.90), [0.90, 1.00]
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_examples import build_examples, load_training_data
from src.data.make_splits import split_examples
from src.evaluation.entity_metrics import (
    bio_to_entities,
    count_matches,
    filter_entities_by_category,
    precision_recall_fbeta,
    BETA,
)
from src.evaluation.competition_metrics import calculate_competition_metrics
from src.evaluation.experiment_26a_confidence_calibration import (
    bio_to_entities_with_confidence,
    collect_active_probabilities,
)

CHAMPION_CHECKPOINTS = {
    "gbert": Path("checkpoints/gbert_baseline_best"),
    "xlmr42": Path("checkpoints/xlmr_baseline_best"),
    "xlmr7": Path("checkpoints/xlmr_seed_7_best"),
    "xlmr21": Path("checkpoints/xlmr_seed_21_best"),
}

CACHE_DIR = PROJECT_ROOT / "data" / "cache"

def bio_to_entities_with_spans_and_conf(
    tokens: List[str],
    bio_tags: List[str],
    selected_probs: List[float],
) -> List[Tuple[str, str, int, int, float, float]]:
    """
    Extract entities with (aspect, value, start_idx, end_idx, mean_conf, min_conf).
    start_idx and end_idx are inclusive token indices.
    """
    entities = []
    current_aspect = None
    current_tokens = []
    current_probs = []
    start_idx = None

    def close_current_entity(curr_end_idx: int):
        nonlocal current_aspect, current_tokens, current_probs, start_idx
        if current_aspect is not None and current_tokens:
            val = " ".join(current_tokens)
            mean_c = float(np.mean(current_probs))
            min_c = float(np.min(current_probs))
            entities.append((current_aspect, val, start_idx, curr_end_idx, mean_c, min_c))
        current_aspect = None
        current_tokens = []
        current_probs = []
        start_idx = None

    for idx, (token, tag, prob) in enumerate(zip(tokens, bio_tags, selected_probs)):
        if tag == "O":
            if current_aspect is not None:
                close_current_entity(idx - 1)
            continue

        if tag.startswith("B-"):
            if current_aspect is not None:
                close_current_entity(idx - 1)
            current_aspect = tag[2:]
            current_tokens = [token]
            current_probs = [prob]
            start_idx = idx
            continue

        if tag.startswith("I-"):
            aspect = tag[2:]
            if current_aspect == aspect:
                current_tokens.append(token)
                current_probs.append(prob)
            else:
                if current_aspect is not None:
                    close_current_entity(idx - 1)
                current_aspect = aspect
                current_tokens = [token]
                current_probs = [prob]
                start_idx = idx
            continue

    if current_aspect is not None:
        close_current_entity(len(tokens) - 1)

    return entities


def bio_to_true_spans(
    tokens: List[str],
    bio_tags: List[str],
) -> List[Tuple[str, str, int, int]]:
    """Extract true entities with (aspect, value, start_idx, end_idx)."""
    entities = []
    current_aspect = None
    current_tokens = []
    start_idx = None

    def close_entity(curr_end):
        nonlocal current_aspect, current_tokens, start_idx
        if current_aspect is not None and current_tokens:
            entities.append((current_aspect, " ".join(current_tokens), start_idx, curr_end))
        current_aspect = None
        current_tokens = []
        start_idx = None

    for idx, (token, tag) in enumerate(zip(tokens, bio_tags)):
        if tag == "O":
            if current_aspect is not None:
                close_entity(idx - 1)
            continue
        if tag.startswith("B-"):
            if current_aspect is not None:
                close_entity(idx - 1)
            current_aspect = tag[2:]
            current_tokens = [token]
            start_idx = idx
            continue
        if tag.startswith("I-"):
            aspect = tag[2:]
            if current_aspect == aspect:
                current_tokens.append(token)
            else:
                if current_aspect is not None:
                    close_entity(idx - 1)
                current_aspect = aspect
                current_tokens = [token]
                start_idx = idx
            continue
    if current_aspect is not None:
        close_entity(len(tokens) - 1)
    return entities


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Label Mapping from Reference Champion
    ref_cfg = AutoConfig.from_pretrained(CHAMPION_CHECKPOINTS["gbert"])
    id2label = {int(k): v for k, v in ref_cfg.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    # 2. Load Split A Validation Set
    df = load_training_data()
    all_examples = build_examples(df)
    _, val_examples = split_examples(all_examples, random_state=42)
    print(f"Loaded {len(val_examples)} Split-A validation titles.")

    # 3. Check for cached fused probabilities or compute them
    fused_cache_file = CACHE_DIR / "split_a_champion_fused_probs.npy"
    if fused_cache_file.exists():
        print(f"Loading cached fused probabilities from {fused_cache_file}...")
        p_fused_flat = np.load(fused_cache_file, allow_pickle=True)
        # Reconstruct list of arrays
        p_fused = list(p_fused_flat)
    else:
        print("Extracting champion model probabilities...")
        probs = {}
        for name, path in CHAMPION_CHECKPOINTS.items():
            print(f"  Extracting {name}...")
            probs[name] = collect_active_probabilities(path, val_examples, label2id, device, batch_size=8)
        p_fused = [
            0.40 * probs["gbert"][i] + 0.20 * probs["xlmr42"][i] + 0.20 * probs["xlmr7"][i] + 0.20 * probs["xlmr21"][i]
            for i in range(len(val_examples))
        ]
        # Cache for Phase 30A research
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(fused_cache_file, np.array(p_fused, dtype=object))
        print(f"Cached fused probabilities to {fused_cache_file}.")

    # 4. Run Diagnostic Extraction across all titles
    # Store per-aspect metrics and error taxonomy
    # We will analyze at MIN >= 0.60 (Champion) and also track candidates at lower/higher confidences
    aspect_stats = defaultdict(lambda: {
        "true_count": 0,
        "pred_count": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "fp_boundary": 0,     # Overlaps true span with same aspect, but boundary differs
        "fp_wrong_aspect": 0, # Overlaps true span with different aspect
        "fp_spurious": 0,     # Completely disjoint from any true entity
        "tp_confidences": [],
        "fp_confidences": [],
        "fp_buckets": {"[0.60, 0.70)": 0, "[0.70, 0.80)": 0, "[0.80, 0.90)": 0, "[0.90, 1.00]": 0},
    })

    category_stats = defaultdict(lambda: {
        "true_count": 0,
        "pred_count": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tp_confidences": [],
        "fp_confidences": [],
        "fp_buckets": {"[0.60, 0.70)": 0, "[0.70, 0.80)": 0, "[0.80, 0.90)": 0, "[0.90, 1.00]": 0},
    })

    champion_records = []
    all_extracted_entities = []

    for ex_idx, ex in enumerate(val_examples):
        tokens = ex["tokens"]
        cat_id = ex["category"]
        true_bio = ex["bio_tags"]

        p = p_fused[ex_idx]
        pred_ids = np.argmax(p, axis=-1)
        pred_bio = [id2label[pid] for pid in pred_ids]
        selected_probs = [float(p[t_idx, pred_ids[t_idx]]) for t_idx in range(len(tokens))]

        true_spans = bio_to_true_spans(tokens, true_bio)
        # Filter true spans by category
        cat_valid_true = [s for s in true_spans if filter_entities_by_category([(s[0], s[1])], cat_id)]

        pred_spans_conf = bio_to_entities_with_spans_and_conf(tokens, pred_bio, selected_probs)
        cat_valid_pred = [s for s in pred_spans_conf if filter_entities_by_category([(s[0], s[1])], cat_id)]

        all_extracted_entities.append({
            "ex_idx": ex_idx,
            "category": cat_id,
            "true_spans": cat_valid_true,
            "pred_spans": cat_valid_pred,
        })

        # Track true entity counts
        for asp, val, s_idx, e_idx in cat_valid_true:
            aspect_stats[(cat_id, asp)]["true_count"] += 1
            category_stats[cat_id]["true_count"] += 1

        # Evaluate MIN 0.60 champion predictions
        champ_preds = [(asp, val, s_idx, e_idx, mean_c, min_c) for asp, val, s_idx, e_idx, mean_c, min_c in cat_valid_pred if min_c >= 0.60]

        # Detailed matching with error taxonomy
        # True entities matched flag
        true_matched = [False] * len(cat_valid_true)

        for asp, val, p_s, p_e, mean_c, min_c in champ_preds:
            aspect_stats[(cat_id, asp)]["pred_count"] += 1
            category_stats[cat_id]["pred_count"] += 1

            # Check exact match
            exact_match_idx = None
            for t_idx, (t_asp, t_val, t_s, t_e) in enumerate(cat_valid_true):
                if not true_matched[t_idx] and t_asp == asp and t_val == val:
                    exact_match_idx = t_idx
                    break

            if exact_match_idx is not None:
                # True positive!
                true_matched[exact_match_idx] = True
                aspect_stats[(cat_id, asp)]["tp"] += 1
                category_stats[cat_id]["tp"] += 1
                aspect_stats[(cat_id, asp)]["tp_confidences"].append(min_c)
                category_stats[cat_id]["tp_confidences"].append(min_c)
            else:
                # False positive!
                aspect_stats[(cat_id, asp)]["fp"] += 1
                category_stats[cat_id]["fp"] += 1
                aspect_stats[(cat_id, asp)]["fp_confidences"].append(min_c)
                category_stats[cat_id]["fp_confidences"].append(min_c)

                # Confidence bucket
                if min_c < 0.70:
                    b_key = "[0.60, 0.70)"
                elif min_c < 0.80:
                    b_key = "[0.70, 0.80)"
                elif min_c < 0.90:
                    b_key = "[0.80, 0.90)"
                else:
                    b_key = "[0.90, 1.00]"
                aspect_stats[(cat_id, asp)]["fp_buckets"][b_key] += 1
                category_stats[cat_id]["fp_buckets"][b_key] += 1

                # Classify error type: boundary vs wrong aspect vs spurious
                # Check overlap with any true entity span
                overlapping_true = []
                for t_idx, (t_asp, t_val, t_s, t_e) in enumerate(cat_valid_true):
                    # Check span overlap
                    if max(p_s, t_s) <= min(p_e, t_e):
                        overlapping_true.append((t_asp, t_val, t_s, t_e))

                if not overlapping_true:
                    aspect_stats[(cat_id, asp)]["fp_spurious"] += 1
                else:
                    same_asp_overlaps = [o for o in overlapping_true if o[0] == asp]
                    if same_asp_overlaps:
                        aspect_stats[(cat_id, asp)]["fp_boundary"] += 1
                    else:
                        aspect_stats[(cat_id, asp)]["fp_wrong_aspect"] += 1

        # Unmatched true entities are false negatives
        for t_idx, (t_asp, t_val, t_s, t_e) in enumerate(cat_valid_true):
            if not true_matched[t_idx]:
                aspect_stats[(cat_id, t_asp)]["fn"] += 1
                category_stats[cat_id]["fn"] += 1

        champion_records.append({
            "category": cat_id,
            "true_entities": [(s[0], s[1]) for s in cat_valid_true],
            "predicted_entities": [(s[0], s[1]) for s in champ_preds],
        })

    # Save extracted profile data for Step 3 policy evaluation
    profile_out = CACHE_DIR / "phase_30a_diagnostic_profile.json"
    clean_aspect_stats = {}
    for (cat_id, asp), stats in aspect_stats.items():
        key = f"Cat{int(cat_id)}__{asp}"
        tp_c = stats["tp_confidences"]
        fp_c = stats["fp_confidences"]
        p = float(stats["tp"] / max(1, stats["tp"] + stats["fp"]))
        r = float(stats["tp"] / max(1, stats["tp"] + stats["fn"]))
        f02 = float((1.04 * p * r) / (0.04 * p + r)) if (0.04 * p + r) > 0 else 0.0

        clean_aspect_stats[key] = {
            "category": int(cat_id),
            "aspect": asp,
            "true_count": int(stats["true_count"]),
            "pred_count": int(stats["pred_count"]),
            "tp": int(stats["tp"]),
            "fp": int(stats["fp"]),
            "fn": int(stats["fn"]),
            "precision": p,
            "recall": r,
            "f02": f02,
            "fp_boundary": int(stats["fp_boundary"]),
            "fp_wrong_aspect": int(stats["fp_wrong_aspect"]),
            "fp_spurious": int(stats["fp_spurious"]),
            "fp_buckets": {k: int(v) for k, v in stats["fp_buckets"].items()},
            "tp_mean_conf": float(np.mean(tp_c)) if tp_c else 0.0,
            "tp_median_conf": float(np.median(tp_c)) if tp_c else 0.0,
            "fp_mean_conf": float(np.mean(fp_c)) if fp_c else 0.0,
            "fp_median_conf": float(np.median(fp_c)) if fp_c else 0.0,
        }

    clean_category_stats = {}
    for cat_id, stats in category_stats.items():
        tp_c = stats["tp_confidences"]
        fp_c = stats["fp_confidences"]
        p = float(stats["tp"] / max(1, stats["tp"] + stats["fp"]))
        r = float(stats["tp"] / max(1, stats["tp"] + stats["fn"]))
        clean_category_stats[str(int(cat_id))] = {
            "true_count": int(stats["true_count"]),
            "pred_count": int(stats["pred_count"]),
            "tp": int(stats["tp"]),
            "fp": int(stats["fp"]),
            "fn": int(stats["fn"]),
            "precision": p,
            "recall": r,
            "fp_buckets": {k: int(v) for k, v in stats["fp_buckets"].items()},
            "tp_mean_conf": float(np.mean(tp_c)) if tp_c else 0.0,
            "fp_mean_conf": float(np.mean(fp_c)) if fp_c else 0.0,
        }

    with open(profile_out, "w", encoding="utf-8") as f:
        json.dump({
            "aspects": clean_aspect_stats,
            "categories": clean_category_stats,
        }, f, indent=2)

    print(f"\nDiagnostic error profile saved to {profile_out}.")

    # Print Category-level summary
    print("\n" + "=" * 80)
    print("CATEGORY SUMMARY (MIN 0.60 Champion)")
    print("=" * 80)
    for cat_id in [1, 2]:
        c_data = clean_category_stats[str(cat_id)]
        print(f"Category {cat_id}:")
        print(f"  True Entities: {c_data['true_count']} | Pred Entities: {c_data['pred_count']}")
        print(f"  TP: {c_data['tp']} | FP: {c_data['fp']} | FN: {c_data['fn']}")
        print(f"  Micro Precision: {c_data['precision']:.4f} | Micro Recall: {c_data['recall']:.4f}")
        print(f"  TP Mean Conf: {c_data['tp_mean_conf']:.4f} | FP Mean Conf: {c_data['fp_mean_conf']:.4f}")
        print(f"  FP Buckets: {c_data['fp_buckets']}")

    # Print Aspect-level breakdown sorted by true_count desc
    print("\n" + "=" * 110)
    print("ASPECT ERROR TAXONOMY & PRECISION PROFILE (MIN 0.60 Champion)")
    print("=" * 110)
    print(f"{'Aspect':<32} {'Cat':<4} {'True':<6} {'Pred':<6} {'TP':<5} {'FP':<5} {'FN':<5} {'Prec':<7} {'Rec':<7} {'F0.2':<8} {'FP(B/W/S)':<12} {'FP [0.6,0.7)'}")
    print("-" * 110)

    for key, d in sorted(clean_aspect_stats.items(), key=lambda x: (x[1]["category"], -x[1]["true_count"])):
        rel_flag = " *" if d["true_count"] < 30 else ""
        b_w_s = f"{d['fp_boundary']}/{d['fp_wrong_aspect']}/{d['fp_spurious']}"
        fp_60_70 = d["fp_buckets"]["[0.60, 0.70)"]
        print(f"{d['aspect']:<32} {d['category']:<4} {d['true_count']:<6} {d['pred_count']:<6} {d['tp']:<5} {d['fp']:<5} {d['fn']:<5} {d['precision']:<7.4f} {d['recall']:<7.4f} {d['f02']:<8.4f} {b_w_s:<12} {fp_60_70:<5}{rel_flag}")

    print("\n* Marked low-support aspects (< 30 true examples) as statistically unreliable.")

if __name__ == "__main__":
    main()
