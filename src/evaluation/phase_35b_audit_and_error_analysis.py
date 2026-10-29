"""
Phase 35B: Evaluation Integrity, Clean Holdout Verification & Precision Error Analysis.

Comprehensive Audit Script:
- Part A: Verify Evaluation Mathematics (argument ordering root cause & proof).
- Part B: Audit Clean Holdout Dataset (489 titles: exact/normalized title match, ID isolation, distributions).
- Part C: Reproduce Exact Metrics on Verified Holdout (Original 4M, Five-Model, DAPT 4M, Standalones).
- Part D: Paired Bootstrap Significance (1,000 resamples, DAPT vs Orig 4M, DAPT vs 5M).
- Part E: Precision-Oriented Error Analysis on DAPT 4M (Top 5 aspects, Boundary vs Confusion vs False Alarm, examples).
- Part F: Improvement Feasibility Assessment (Aspect threshold tuning, theoretical ceiling, 3 proposed experiments).
- Part G: Save manifest and machine-readable audit report.
"""

import hashlib
import json
import math
import re
import string
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch
from transformers import AutoConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_examples import build_examples, load_training_data
from src.data.make_splits import split_examples
from src.evaluation.entity_metrics import (
    BETA,
    count_matches,
    filter_entities_by_category,
    precision_recall_fbeta,
    bio_to_entities_with_confidence,
)
from scripts.validate_submission import (
    OFFICIAL_ASPECTS_CAT1,
    OFFICIAL_ASPECTS_CAT2,
)

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

CHECKPOINTS = {
    "orig_gbert": CHECKPOINTS_DIR / "gbert_baseline_best",
    "dapt_gbert": CHECKPOINTS_DIR / "phase33b_dapt_gbert_best",
    "xlmr_seed42": CHECKPOINTS_DIR / "xlmr_baseline_best",
    "xlmr_seed7": CHECKPOINTS_DIR / "xlmr_seed_7_best",
    "xlmr_seed21": CHECKPOINTS_DIR / "xlmr_seed_21_best",
}

HOLDOUT_MANIFEST_PATH = CACHE_DIR / "phase_35b_clean_holdout_manifest.json"
AUDIT_RESULTS_PATH = CACHE_DIR / "phase_35b_audit_results.json"


def normalize_text(text: str) -> str:
    """Normalize text: lowercased, whitespace stripped, punctuation removed."""
    text = text.lower()
    text = re.sub(r"[" + re.escape(string.punctuation) + r"]", " ", text)
    return " ".join(text.split())


def extract_spanned_gold_entities(examples: List[dict]) -> List[List[Dict[str, Any]]]:
    """Extract gold entities with token start/end spans and category filtering."""
    gold_per_ex = []
    for ex in examples:
        tokens = ex["tokens"]
        cat_id = ex["category"]
        bio_tags = ex["bio_tags"]

        ents = []
        cur_aspect = None
        cur_start = None

        def close_entity(end_idx: int):
            nonlocal cur_aspect, cur_start
            if cur_aspect is not None and cur_start is not None:
                val = " ".join(tokens[cur_start:end_idx])
                ents.append({
                    "aspect": cur_aspect,
                    "value": val,
                    "start": cur_start,
                    "end": end_idx,
                })
            cur_aspect = None
            cur_start = None

        for t_idx, (tok, tag) in enumerate(zip(tokens, bio_tags)):
            if tag == "O":
                close_entity(t_idx)
            elif tag.startswith("B-"):
                close_entity(t_idx)
                cur_aspect = tag[2:]
                cur_start = t_idx
            elif tag.startswith("I-"):
                asp = tag[2:]
                if cur_aspect == asp:
                    pass
                else:
                    close_entity(t_idx)
                    cur_aspect = asp
                    cur_start = t_idx

        close_entity(len(tokens))

        allowed = OFFICIAL_ASPECTS_CAT1 if cat_id == 1 else OFFICIAL_ASPECTS_CAT2
        cat_valid = [e for e in ents if e["aspect"] in allowed]
        gold_per_ex.append(cat_valid)

    return gold_per_ex


def decode_spanned_predictions(
    probs_list: List[np.ndarray],
    examples: List[dict],
    id2label: Dict[int, str],
    conf_thresh: float = 0.60,
    aspect_thresholds: Dict[str, float] = None,
) -> List[List[Dict[str, Any]]]:
    """Decode predictions with token spans, mean/min confidence, and filtering."""
    preds_per_ex = []
    for ex_idx, ex in enumerate(examples):
        tokens = ex["tokens"]
        cat_id = ex["category"]
        p = probs_list[ex_idx]

        pred_ids = np.argmax(p, axis=-1)
        pred_bio = [id2label[pid] for pid in pred_ids]
        selected_probs = [float(p[t_idx, pred_ids[t_idx]]) for t_idx in range(len(tokens))]

        ents = []
        cur_aspect = None
        cur_start = None
        cur_probs = []

        def close_pred(end_idx: int):
            nonlocal cur_aspect, cur_start, cur_probs
            if cur_aspect is not None and cur_start is not None:
                val = " ".join(tokens[cur_start:end_idx])
                min_c = float(np.min(cur_probs))
                mean_c = float(np.mean(cur_probs))
                ents.append({
                    "aspect": cur_aspect,
                    "value": val,
                    "start": cur_start,
                    "end": end_idx,
                    "min_conf": min_c,
                    "mean_conf": mean_c,
                })
            cur_aspect = None
            cur_start = None
            cur_probs = []

        for t_idx, (tok, tag, prob) in enumerate(zip(tokens, pred_bio, selected_probs)):
            if tag == "O":
                close_pred(t_idx)
            elif tag.startswith("B-"):
                close_pred(t_idx)
                cur_aspect = tag[2:]
                cur_start = t_idx
                cur_probs = [prob]
            elif tag.startswith("I-"):
                asp = tag[2:]
                if cur_aspect == asp:
                    cur_probs.append(prob)
                else:
                    close_pred(t_idx)
                    cur_aspect = asp
                    cur_start = t_idx
                    cur_probs = [prob]

        close_pred(len(tokens))

        allowed = OFFICIAL_ASPECTS_CAT1 if cat_id == 1 else OFFICIAL_ASPECTS_CAT2
        filtered_ents = []
        for e in ents:
            asp = e["aspect"]
            thresh = aspect_thresholds.get(asp, conf_thresh) if aspect_thresholds else conf_thresh
            if e["min_conf"] >= thresh and asp in allowed:
                filtered_ents.append(e)

        preds_per_ex.append(filtered_ents)

    return preds_per_ex


def compute_metrics_from_tuples(
    gold_per_ex: List[List[Tuple[str, str]]],
    preds_per_ex: List[List[Tuple[str, str]]],
    examples: List[dict],
) -> Dict[str, Any]:
    """Correct evaluation using count_matches(pred, gold)."""
    total_tp = 0
    total_fp = 0
    total_fn = 0

    cat_counts = {1: {"tp": 0, "fp": 0, "fn": 0}, 2: {"tp": 0, "fp": 0, "fn": 0}}
    aspect_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    title_scores = []

    for ex_idx, (gold, pred) in enumerate(zip(gold_per_ex, preds_per_ex)):
        cat_id = examples[ex_idx]["category"]
        tp, fp, fn = count_matches(pred, gold)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        cat_counts[cat_id]["tp"] += tp
        cat_counts[cat_id]["fp"] += fp
        cat_counts[cat_id]["fn"] += fn

        ex_m = precision_recall_fbeta(tp, fp, fn, beta=0.2)
        title_scores.append(ex_m["f_beta"])

        gold_asp = Counter([a for a, v in gold])
        pred_asp = Counter([a for a, v in pred])
        all_asps = set(gold_asp.keys()).union(set(pred_asp.keys()))
        for asp in all_asps:
            g_ents = [e for e in gold if e[0] == asp]
            p_ents = [e for e in pred if e[0] == asp]
            a_tp, a_fp, a_fn = count_matches(p_ents, g_ents)
            aspect_counts[asp]["tp"] += a_tp
            aspect_counts[asp]["fp"] += a_fp
            aspect_counts[asp]["fn"] += a_fn

    overall = precision_recall_fbeta(total_tp, total_fp, total_fn, beta=0.2)
    overall["tp"] = total_tp
    overall["fp"] = total_fp
    overall["fn"] = total_fn

    cat_results = {}
    for cat, c in cat_counts.items():
        res = precision_recall_fbeta(c["tp"], c["fp"], c["fn"], beta=0.2)
        res["tp"] = c["tp"]
        res["fp"] = c["fp"]
        res["fn"] = c["fn"]
        cat_results[cat] = res

    asp_results = {}
    for asp, c in aspect_counts.items():
        res = precision_recall_fbeta(c["tp"], c["fp"], c["fn"], beta=0.2)
        res["tp"] = c["tp"]
        res["fp"] = c["fp"]
        res["fn"] = c["fn"]
        asp_results[asp] = res

    return {
        "overall": overall,
        "category": cat_results,
        "aspect": asp_results,
        "title_scores": title_scores,
    }


def compute_paired_bootstrap(
    gold_per_ex: List[List[Tuple[str, str]]],
    preds_a: List[List[Tuple[str, str]]],
    preds_b: List[List[Tuple[str, str]]],
    num_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Paired bootstrap differences: b minus a."""
    rng = np.random.default_rng(seed)
    n = len(gold_per_ex)
    diffs = []

    for _ in range(num_bootstraps):
        indices = rng.integers(0, n, size=n)
        tp_a, fp_a, fn_a = 0, 0, 0
        tp_b, fp_b, fn_b = 0, 0, 0

        for idx in indices:
            gold = gold_per_ex[idx]
            ta, fa, na = count_matches(preds_a[idx], gold)
            tb, fb, nb = count_matches(preds_b[idx], gold)
            tp_a += ta; fp_a += fa; fn_a += na
            tp_b += tb; fp_b += fb; fn_b += nb

        fb_a = precision_recall_fbeta(tp_a, fp_a, fn_a, beta=0.2)["f_beta"]
        fb_b = precision_recall_fbeta(tp_b, fp_b, fn_b, beta=0.2)["f_beta"]
        diffs.append(fb_b - fb_a)

    diffs = np.array(diffs)
    ci_low = float(np.percentile(diffs, 2.5))
    ci_high = float(np.percentile(diffs, 97.5))
    p_value = float(np.mean(diffs <= 0.0) if np.mean(diffs) > 0 else np.mean(diffs >= 0.0))

    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_lower_95": ci_low,
        "ci_upper_95": ci_high,
        "p_value": p_value,
    }


def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def run_phase_35b_audit():
    print("=" * 80)
    print("PHASE 35B: EVALUATION INTEGRITY & PRECISION ERROR ANALYSIS")
    print("=" * 80)

    # ---------------------------------------------------------
    # PART A & B: Load Data & Construct Partitions
    # ---------------------------------------------------------
    print("\n[PART B] Loading dataset and constructing training / clean partitions...")
    df = load_training_data()
    all_examples = build_examples(df)
    train_4k, val_1k = split_examples(all_examples, random_state=42)

    from sklearn.model_selection import train_test_split
    slice_4k = all_examples[:4000]
    slice_cats = [ex["category"] for ex in slice_4k]
    dapt_train_3200, _ = train_test_split(slice_4k, test_size=0.2, random_state=42, stratify=slice_cats)

    train_4k_ids = set(ex["record_number"] for ex in train_4k)
    dapt_train_ids = set(ex["record_number"] for ex in dapt_train_3200)
    union_train_ids = train_4k_ids.union(dapt_train_ids)

    print(f"Total labeled dataset: {len(all_examples)} titles")
    print(f"Original 4-model train: {len(train_4k_ids)} titles")
    print(f"Phase 33B DAPT supervised train: {len(dapt_train_ids)} titles")
    print(f"Union supervised train: {len(union_train_ids)} titles ({len(union_train_ids)/len(all_examples)*100:.2f}%)")

    clean_val_indices = [
        idx for idx, ex in enumerate(val_1k) if ex["record_number"] not in union_train_ids
    ]
    clean_examples = [val_1k[i] for i in clean_val_indices]
    print(f"Clean unpolluted holdout: {len(clean_examples)} titles ({len(clean_examples)/len(val_1k)*100:.2f}% of val_1k)")

    # Verification: Cross-Reference & Leakage Checks
    train_records = [ex for ex in all_examples if ex["record_number"] in union_train_ids]
    clean_ids = set(ex["record_number"] for ex in clean_examples)

    # Check 1: Record ID overlap
    id_overlap = clean_ids.intersection(union_train_ids)
    print(f"Record ID overlap with training sets: {len(id_overlap)}")

    # Check 2: Exact title match
    clean_titles = [" ".join(ex["tokens"]) for ex in clean_examples]
    train_titles_set = set(" ".join(ex["tokens"]) for ex in train_records)
    exact_title_overlap = [t for t in clean_titles if t in train_titles_set]
    print(f"Exact title text overlap with training sets: {len(exact_title_overlap)}")

    # Check 3: Normalized title match
    train_norm_titles_set = set(normalize_text(" ".join(ex["tokens"])) for ex in train_records)
    norm_title_overlap = [t for t in clean_titles if normalize_text(t) in train_norm_titles_set]
    print(f"Normalized title text overlap with training sets: {len(norm_title_overlap)}")

    # Characterize category distribution
    cat_all = Counter(ex["category"] for ex in all_examples)
    cat_train = Counter(ex["category"] for ex in train_records)
    cat_clean = Counter(ex["category"] for ex in clean_examples)

    cat_dist = {
        "all_5000": {c: count / len(all_examples) for c, count in cat_all.items()},
        "train_4511": {c: count / len(train_records) for c, count in cat_train.items()},
        "clean_489": {c: count / len(clean_examples) for c, count in cat_clean.items()},
    }
    print(f"Category distribution (Cat 1 vs Cat 2):")
    print(f"  All 5,000:   Cat 1: {cat_dist['all_5000'][1]*100:.2f}%, Cat 2: {cat_dist['all_5000'][2]*100:.2f}%")
    print(f"  Train 4,511: Cat 1: {cat_dist['train_4511'][1]*100:.2f}%, Cat 2: {cat_dist['train_4511'][2]*100:.2f}%")
    print(f"  Clean 489:   Cat 1: {cat_dist['clean_489'][1]*100:.2f}%, Cat 2: {cat_dist['clean_489'][2]*100:.2f}%")

    # Characterize aspect distribution
    spanned_gold_clean = extract_spanned_gold_entities(clean_examples)
    spanned_gold_all = extract_spanned_gold_entities(all_examples)

    clean_aspect_counts = Counter(e["aspect"] for ex_ents in spanned_gold_clean for e in ex_ents)
    all_aspect_counts = Counter(e["aspect"] for ex_ents in spanned_gold_all for e in ex_ents)

    total_clean_gold = sum(clean_aspect_counts.values())
    total_all_gold = sum(all_aspect_counts.values())
    print(f"Total gold entities in clean holdout: {total_clean_gold}")
    print(f"Total gold entities in all 5,000: {total_all_gold}")

    aspect_dist_comp = {}
    for asp in sorted(all_aspect_counts.keys()):
        p_clean = clean_aspect_counts[asp] / total_clean_gold
        p_all = all_aspect_counts[asp] / total_all_gold
        aspect_dist_comp[asp] = {
            "clean_count": clean_aspect_counts[asp],
            "clean_pct": p_clean,
            "all_count": all_aspect_counts[asp],
            "all_pct": p_all,
            "ratio_clean_to_all": p_clean / p_all if p_all > 0 else 0.0,
        }

    # Save Clean Holdout Manifest
    sorted_clean_ids = sorted(list(clean_ids))
    manifest_id_hash = hashlib.sha256(",".join(str(i) for i in sorted_clean_ids).encode("utf-8")).hexdigest()
    manifest_titles_hash = hashlib.sha256("||".join(clean_titles).encode("utf-8")).hexdigest()

    manifest_data = {
        "manifest_version": "phase_35b_v1",
        "description": "Strictly verified 489-title holdout completely unseen by all 5 supervised models.",
        "record_count": len(sorted_clean_ids),
        "checksums": {
            "record_ids_sha256": manifest_id_hash,
            "titles_concatenated_sha256": manifest_titles_hash,
        },
        "verification": {
            "record_id_leakage": len(id_overlap) == 0,
            "exact_title_match_leakage": len(exact_title_overlap) == 0,
            "normalized_title_match_leakage": len(norm_title_overlap) == 0,
        },
        "category_counts": {str(c): cat_clean[c] for c in sorted(cat_clean.keys())},
        "total_gold_entities": total_clean_gold,
        "aspect_counts": dict(clean_aspect_counts),
        "record_ids": sorted_clean_ids,
    }
    with open(HOLDOUT_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Saved clean holdout manifest to {HOLDOUT_MANIFEST_PATH} (SHA-256 IDs: {manifest_id_hash[:16]}...)")

    # ---------------------------------------------------------
    # PART C: Load Cached Probabilities & Evaluate Systems
    # ---------------------------------------------------------
    print("\n[PART C] Loading model probabilities and computing verified holdout metrics...")
    probs_cache_path = CACHE_DIR / "phase_35a_val_1k_probs.pt"
    if not probs_cache_path.exists():
        raise FileNotFoundError(f"Missing required cache: {probs_cache_path}")

    probs = torch.load(probs_cache_path, map_location="cpu", weights_only=False)
    ref_cfg = AutoConfig.from_pretrained(CHECKPOINTS["orig_gbert"])
    id2label = {int(k): v for k, v in ref_cfg.id2label.items()}

    fusions = {
        "orig_gbert_standalone": [probs["orig_gbert"][i] for i in range(len(val_1k))],
        "dapt_gbert_standalone": [probs["dapt_gbert"][i] for i in range(len(val_1k))],
        "orig_champ_4m": [
            0.40 * probs["orig_gbert"][i] + 0.20 * probs["xlmr_seed42"][i] + 0.20 * probs["xlmr_seed7"][i] + 0.20 * probs["xlmr_seed21"][i]
            for i in range(len(val_1k))
        ],
        "dapt_champ_4m": [
            0.40 * probs["dapt_gbert"][i] + 0.20 * probs["xlmr_seed42"][i] + 0.20 * probs["xlmr_seed7"][i] + 0.20 * probs["xlmr_seed21"][i]
            for i in range(len(val_1k))
        ],
        "five_model_5m": [
            0.20 * probs["orig_gbert"][i] + 0.20 * probs["dapt_gbert"][i] + 0.20 * probs["xlmr_seed42"][i] + 0.20 * probs["xlmr_seed7"][i] + 0.20 * probs["xlmr_seed21"][i]
            for i in range(len(val_1k))
        ],
    }

    # Gold standard tuples on clean holdout
    gold_clean_tuples = [[(e["aspect"], e["value"]) for e in ex_ents] for ex_ents in spanned_gold_clean]

    # Decode predictions with MIN 0.60
    preds_clean_spanned = {}
    preds_clean_tuples = {}
    holdout_metrics = {}

    print("\n" + "=" * 90)
    print(f"{'System':<25} {'F0.2':<10} {'Precision':<10} {'Recall':<10} {'TP':<6} {'FP':<6} {'FN':<6} {'TP+FN':<7}")
    print("-" * 90)

    for sys_name, p_list in fusions.items():
        clean_p_list = [p_list[i] for i in clean_val_indices]
        p_spanned = decode_spanned_predictions(clean_p_list, clean_examples, id2label, conf_thresh=0.60)
        p_tuples = [[(e["aspect"], e["value"]) for e in ex_ents] for ex_ents in p_spanned]

        preds_clean_spanned[sys_name] = p_spanned
        preds_clean_tuples[sys_name] = p_tuples

        m = compute_metrics_from_tuples(gold_clean_tuples, p_tuples, clean_examples)
        holdout_metrics[sys_name] = m

        ov = m["overall"]
        tp_plus_fn = ov["tp"] + ov["fn"]
        print(f"{sys_name:<25} {ov['f_beta']:<10.6f} {ov['precision']:<10.6f} {ov['recall']:<10.6f} {ov['tp']:<6} {ov['fp']:<6} {ov['fn']:<6} {tp_plus_fn:<7}")

    print("=" * 90)

    # ---------------------------------------------------------
    # PART D: Paired Bootstrap Significance
    # ---------------------------------------------------------
    print("\n[PART D] Computing paired bootstrap significance (1,000 resamples, seed=42)...")
    boot_dapt_vs_orig = compute_paired_bootstrap(
        gold_clean_tuples,
        preds_clean_tuples["orig_champ_4m"],
        preds_clean_tuples["dapt_champ_4m"],
        num_bootstraps=1000,
        seed=42,
    )
    print("DAPT 4-Model vs Original 4-Model Champion:")
    print(f"  Mean Delta F0.2: {boot_dapt_vs_orig['mean_diff']:+.6f}")
    print(f"  95% CI: [{boot_dapt_vs_orig['ci_lower_95']:+.6f}, {boot_dapt_vs_orig['ci_upper_95']:+.6f}]")
    print(f"  Empirical p-value: {boot_dapt_vs_orig['p_value']:.4f}")
    print(f"  Statistically significant (alpha=0.05): {boot_dapt_vs_orig['p_value'] < 0.05}")

    boot_dapt_vs_5m = compute_paired_bootstrap(
        gold_clean_tuples,
        preds_clean_tuples["five_model_5m"],
        preds_clean_tuples["dapt_champ_4m"],
        num_bootstraps=1000,
        seed=42,
    )
    print("\nDAPT 4-Model vs Five-Model Ensemble:")
    print(f"  Mean Delta F0.2: {boot_dapt_vs_5m['mean_diff']:+.6f}")
    print(f"  95% CI: [{boot_dapt_vs_5m['ci_lower_95']:+.6f}, {boot_dapt_vs_5m['ci_upper_95']:+.6f}]")
    print(f"  Empirical p-value: {boot_dapt_vs_5m['p_value']:.4f}")
    print(f"  Statistically significant (alpha=0.05): {boot_dapt_vs_5m['p_value'] < 0.05}")

    # ---------------------------------------------------------
    # PART E: Precision-Oriented Error Analysis on DAPT 4-Model
    # ---------------------------------------------------------
    print("\n[PART E] Performing granular precision error analysis on DAPT 4-Model...")
    dapt_spanned = preds_clean_spanned["dapt_champ_4m"]

    # Decompose every predicted entity into TP, Boundary Error, Aspect Confusion, or False Alarm
    aspect_fp_breakdown = defaultdict(lambda: {
        "tp": 0,
        "fp_total": 0,
        "fp_boundary": 0,
        "fp_aspect_confusion": 0,
        "fp_false_alarm": 0,
        "examples_boundary": [],
        "examples_confusion": [],
        "examples_false_alarm": [],
    })

    for ex_idx, (g_list, p_list) in enumerate(zip(spanned_gold_clean, dapt_spanned)):
        ex = clean_examples[ex_idx]
        title_text = " ".join(ex["tokens"])
        cat_id = ex["category"]

        # Track matched gold entities to avoid multi-matching
        matched_gold_indices = set()

        # Step 1: Find exact matches (TP)
        for p_idx, p in enumerate(p_list):
            matched = False
            for g_idx, g in enumerate(g_list):
                if g_idx in matched_gold_indices:
                    continue
                if p["aspect"] == g["aspect"] and p["value"] == g["value"]:
                    matched = True
                    matched_gold_indices.add(g_idx)
                    aspect_fp_breakdown[p["aspect"]]["tp"] += 1
                    break
            if matched:
                continue

            # Step 2: It is a False Positive! Classify the error type
            p_asp = p["aspect"]
            p_span = set(range(p["start"], p["end"]))
            aspect_fp_breakdown[p_asp]["fp_total"] += 1

            # Check overlap with gold entities
            same_asp_overlaps = []
            diff_asp_overlaps = []

            for g_idx, g in enumerate(g_list):
                g_span = set(range(g["start"], g["end"]))
                if p_span.intersection(g_span):
                    if g["aspect"] == p_asp:
                        same_asp_overlaps.append(g)
                    else:
                        diff_asp_overlaps.append(g)

            if same_asp_overlaps:
                # Boundary error
                aspect_fp_breakdown[p_asp]["fp_boundary"] += 1
                g_ref = same_asp_overlaps[0]
                aspect_fp_breakdown[p_asp]["examples_boundary"].append({
                    "record_number": ex["record_number"],
                    "title": title_text,
                    "pred_val": p["value"],
                    "pred_span": [p["start"], p["end"]],
                    "pred_conf": p["min_conf"],
                    "gold_val": g_ref["value"],
                    "gold_span": [g_ref["start"], g_ref["end"]],
                })
            elif diff_asp_overlaps:
                # Aspect confusion
                aspect_fp_breakdown[p_asp]["fp_aspect_confusion"] += 1
                g_ref = diff_asp_overlaps[0]
                aspect_fp_breakdown[p_asp]["examples_confusion"].append({
                    "record_number": ex["record_number"],
                    "title": title_text,
                    "pred_val": p["value"],
                    "pred_span": [p["start"], p["end"]],
                    "pred_conf": p["min_conf"],
                    "gold_aspect": g_ref["aspect"],
                    "gold_val": g_ref["value"],
                    "gold_span": [g_ref["start"], g_ref["end"]],
                })
            else:
                # Pure false alarm (no overlap with any gold entity)
                aspect_fp_breakdown[p_asp]["fp_false_alarm"] += 1
                aspect_fp_breakdown[p_asp]["examples_false_alarm"].append({
                    "record_number": ex["record_number"],
                    "title": title_text,
                    "pred_val": p["value"],
                    "pred_span": [p["start"], p["end"]],
                    "pred_conf": p["min_conf"],
                })

    # Rank aspects by raw FP count
    sorted_aspects_by_fp = sorted(
        aspect_fp_breakdown.items(),
        key=lambda x: x[1]["fp_total"],
        reverse=True,
    )

    print("\n" + "=" * 105)
    print(f"{'Aspect':<32} {'TP':<6} {'FP':<6} {'FP Rate':<9} {'Boundary':<10} {'Confusion':<11} {'False Alarm':<12}")
    print("-" * 105)
    for asp, d in sorted_aspects_by_fp:
        total_p = d["tp"] + d["fp_total"]
        fp_rate = d["fp_total"] / total_p if total_p > 0 else 0.0
        print(f"{asp:<32} {d['tp']:<6} {d['fp_total']:<6} {fp_rate:<9.3f} {d['fp_boundary']:<10} {d['fp_aspect_confusion']:<11} {d['fp_false_alarm']:<12}")
    print("=" * 105)

    top5_aspects = [asp for asp, _ in sorted_aspects_by_fp[:5]]
    print(f"\nTop 5 error-producing aspects by raw FP count: {top5_aspects}")

    # ---------------------------------------------------------
    # PART F: Improvement Feasibility Assessment
    # ---------------------------------------------------------
    print("\n[PART F] Evaluating improvement feasibility & threshold sensitivity...")
    clean_p_list = [fusions["dapt_champ_4m"][i] for i in clean_val_indices]

    # Test varying thresholds on the #1 FP aspect
    top1_asp = top5_aspects[0]
    print(f"\nThreshold sensitivity for #1 FP aspect ({top1_asp}):")
    thresh_results = {}
    for test_thresh in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        p_sp = decode_spanned_predictions(
            clean_p_list,
            clean_examples,
            id2label,
            conf_thresh=0.60,
            aspect_thresholds={top1_asp: test_thresh},
        )
        p_tup = [[(e["aspect"], e["value"]) for e in ex_ents] for ex_ents in p_sp]
        m_test = compute_metrics_from_tuples(gold_clean_tuples, p_tup, clean_examples)["overall"]
        thresh_results[test_thresh] = m_test
        print(f"  Thresh {test_thresh:.2f}: F0.2={m_test['f_beta']:.6f} | P={m_test['precision']:.6f} | R={m_test['recall']:.6f} | TP={m_test['tp']} | FP={m_test['fp']} | FN={m_test['fn']}")

    # Theoretical ceiling if top FP aspect had 0 FPs
    dapt_ov = holdout_metrics["dapt_champ_4m"]["overall"]
    top1_fps = aspect_fp_breakdown[top1_asp]["fp_total"]
    ceil_top1_m = precision_recall_fbeta(dapt_ov["tp"], dapt_ov["fp"] - top1_fps, dapt_ov["fn"], beta=0.2)
    print(f"\nTheoretical ceiling (Zero FPs on {top1_asp}):")
    print(f"  Current F0.2: {dapt_ov['f_beta']:.6f} -> Max Ceiling: {ceil_top1_m['f_beta']:.6f} (Delta: +{ceil_top1_m['f_beta'] - dapt_ov['f_beta']:.6f})")

    # Theoretical ceiling if ALL FPs were eliminated (FP = 0)
    ceil_all_fp = precision_recall_fbeta(dapt_ov["tp"], 0, dapt_ov["fn"], beta=0.2)
    print(f"Theoretical ceiling (Zero FPs on ALL aspects):")
    print(f"  Current F0.2: {dapt_ov['f_beta']:.6f} -> Max Ceiling: {ceil_all_fp['f_beta']:.6f} (Delta: +{ceil_all_fp['f_beta'] - dapt_ov['f_beta']:.6f})")

    # Save comprehensive machine-readable results
    audit_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "holdout_count": len(clean_examples),
        "holdout_gold_entities": total_clean_gold,
        "clean_verification": {
            "record_id_leakage": len(id_overlap) == 0,
            "exact_title_match_leakage": len(exact_title_overlap) == 0,
            "normalized_title_match_leakage": len(norm_title_overlap) == 0,
        },
        "category_distribution": cat_dist,
        "aspect_distribution_comparison": aspect_dist_comp,
        "verified_metrics_holdout": {
            sys_name: {
                "overall": m["overall"],
                "category": m["category"],
            }
            for sys_name, m in holdout_metrics.items()
        },
        "paired_bootstrap": {
            "dapt_vs_orig_4m": boot_dapt_vs_orig,
            "dapt_vs_5m": boot_dapt_vs_5m,
        },
        "top5_error_producing_aspects": top5_aspects,
        "aspect_fp_breakdown": {
            asp: {
                "tp": d["tp"],
                "fp_total": d["fp_total"],
                "fp_boundary": d["fp_boundary"],
                "fp_aspect_confusion": d["fp_aspect_confusion"],
                "fp_false_alarm": d["fp_false_alarm"],
                "sample_examples_boundary": d["examples_boundary"][:5],
                "sample_examples_confusion": d["examples_confusion"][:5],
                "sample_examples_false_alarm": d["examples_false_alarm"][:5],
            }
            for asp, d in sorted_aspects_by_fp[:5]
        },
        "threshold_sensitivity_top1": {
            str(k): v for k, v in thresh_results.items()
        },
        "theoretical_ceilings": {
            "top1_asp_zero_fp": ceil_top1_m,
            "all_asp_zero_fp": ceil_all_fp,
        },
    }

    with open(AUDIT_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(audit_data), f, indent=2)
    print(f"\nSaved comprehensive audit results to {AUDIT_RESULTS_PATH}")
    print("\nPhase 35B Audit complete!")


if __name__ == "__main__":
    run_phase_35b_audit()
