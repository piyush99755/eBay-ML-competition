"""
Phase 35A: Five-Model Ensemble Local Evaluation & Disagreement Audit.

Compares:
1. Original 4-Model Champion (Exp 10 + MIN 0.60):
   GBERT 0.40 + XLM-R 42 0.20 + XLM-R 7 0.20 + XLM-R 21 0.20
2. DAPT 4-Model Challenger (Phase 34A + MIN 0.60):
   DAPT GBERT 0.40 + XLM-R 42 0.20 + XLM-R 7 0.20 + XLM-R 21 0.20
3. Proposed 5-Model Ensemble (Phase 35A + MIN 0.60):
   Original GBERT 0.20 + DAPT GBERT 0.20 + XLM-R 42 0.20 + XLM-R 7 0.20 + XLM-R 21 0.20

Evaluates on:
- Canonical Split-A val_1k (1,000 records) — explicitly labeled as CONTAMINATED DIAGNOSTIC.
- Clean Unseen Holdout (489 records) — strictly unseen by all 5 models during supervised training.

Performs:
- Full metrics: F0.2, Precision, Recall, TP, FP, FN, Category & Aspect breakdown.
- Paired bootstrap confidence intervals (1,000 resamples) for score differences.
- Disagreement & Venn overlap analysis (shared, unique to orig GBERT, unique to DAPT GBERT, added/removed).
- Vehicle-model and boundary concentration analysis.
"""

import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

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
from src.evaluation.experiment_26a_confidence_calibration import collect_active_probabilities

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

CHECKPOINTS = {
    "orig_gbert": CHECKPOINTS_DIR / "gbert_baseline_best",
    "dapt_gbert": CHECKPOINTS_DIR / "phase33b_dapt_gbert_best",
    "xlmr_seed42": CHECKPOINTS_DIR / "xlmr_baseline_best",
    "xlmr_seed7": CHECKPOINTS_DIR / "xlmr_seed_7_best",
    "xlmr_seed21": CHECKPOINTS_DIR / "xlmr_seed_21_best",
}

RESULTS_PATH = CACHE_DIR / "phase_35a_local_evaluation_results.json"


def extract_gold_entities(examples: List[dict]) -> List[List[Tuple[str, str]]]:
    gold_per_ex = []
    for ex in examples:
        tokens = ex["tokens"]
        cat_id = ex["category"]
        bio_tags = ex["bio_tags"]

        ents = []
        cur_aspect = None
        cur_toks = []
        for tok, tag in zip(tokens, bio_tags):
            if tag == "O":
                if cur_aspect and cur_toks:
                    ents.append((cur_aspect, " ".join(cur_toks)))
                cur_aspect = None
                cur_toks = []
            elif tag.startswith("B-"):
                if cur_aspect and cur_toks:
                    ents.append((cur_aspect, " ".join(cur_toks)))
                cur_aspect = tag[2:]
                cur_toks = [tok]
            elif tag.startswith("I-"):
                aspect = tag[2:]
                if cur_aspect == aspect:
                    cur_toks.append(tok)
                else:
                    if cur_aspect and cur_toks:
                        ents.append((cur_aspect, " ".join(cur_toks)))
                    cur_aspect = aspect
                    cur_toks = [tok]
        if cur_aspect and cur_toks:
            ents.append((cur_aspect, " ".join(cur_toks)))

        cat_valid = filter_entities_by_category(ents, cat_id)
        gold_per_ex.append(cat_valid)
    return gold_per_ex


def decode_predictions(
    probs_list: List[np.ndarray],
    examples: List[dict],
    id2label: Dict[int, str],
    conf_thresh: float = 0.60,
) -> List[List[Tuple[str, str]]]:
    preds_per_ex = []
    for ex_idx, ex in enumerate(examples):
        tokens = ex["tokens"]
        cat_id = ex["category"]
        p = probs_list[ex_idx]

        pred_ids = np.argmax(p, axis=-1)
        pred_bio = [id2label[pid] for pid in pred_ids]
        selected_probs = [float(p[t_idx, pred_ids[t_idx]]) for t_idx in range(len(tokens))]

        raw_ents = bio_to_entities_with_confidence(tokens, pred_bio, selected_probs)
        filtered_ents = []
        for asp, val, mean_c, min_c in raw_ents:
            if min_c >= conf_thresh:
                if filter_entities_by_category([(asp, val)], cat_id):
                    filtered_ents.append((asp, val))
        preds_per_ex.append(filtered_ents)
    return preds_per_ex


def compute_metrics(
    gold_per_ex: List[List[Tuple[str, str]]],
    preds_per_ex: List[List[Tuple[str, str]]],
    examples: List[dict],
) -> Dict[str, Any]:
    total_tp = 0
    total_fp = 0
    total_fn = 0

    cat_counts = {1: {"tp": 0, "fp": 0, "fn": 0}, 2: {"tp": 0, "fp": 0, "fn": 0}}
    aspect_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    title_scores = []

    for ex_idx, (gold, pred) in enumerate(zip(gold_per_ex, preds_per_ex)):
        cat_id = examples[ex_idx]["category"]
        tp, fp, fn = count_matches(gold, pred)
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
            a_tp, a_fp, a_fn = count_matches(g_ents, p_ents)
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
    rng = np.random.default_rng(seed)
    n = len(gold_per_ex)
    diffs = []

    for _ in range(num_bootstraps):
        indices = rng.integers(0, n, size=n)
        tp_a, fp_a, fn_a = 0, 0, 0
        tp_b, fp_b, fn_b = 0, 0, 0

        for idx in indices:
            gold = gold_per_ex[idx]
            ta, fa, na = count_matches(gold, preds_a[idx])
            tb, fb, nb = count_matches(gold, preds_b[idx])
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


def run_evaluation():
    print("=" * 80)
    print("PHASE 35A: LOCAL EVALUATION & FIVE-MODEL DISAGREEMENT AUDIT")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load canonical Split-A validation set (val_1k)
    df = load_training_data()
    all_examples = build_examples(df)
    train_4k, val_1k = split_examples(all_examples, random_state=42)
    print(f"Loaded {len(val_1k)} Split-A validation examples.")

    # Identify clean records (489 records never seen in training by any of the 5 models)
    from sklearn.model_selection import train_test_split
    slice_4k = all_examples[:4000]
    slice_cats = [ex["category"] for ex in slice_4k]
    dapt_train_3200, _ = train_test_split(slice_4k, test_size=0.2, random_state=42, stratify=slice_cats)
    dapt_train_ids = set(ex["record_number"] for ex in dapt_train_3200)
    train_4k_ids = set(ex["record_number"] for ex in train_4k)
    union_train_ids = train_4k_ids.union(dapt_train_ids)

    clean_val_indices = [
        idx for idx, ex in enumerate(val_1k) if ex["record_number"] not in union_train_ids
    ]
    print(f"Clean unpolluted validation records: {len(clean_val_indices)} / {len(val_1k)} ({len(clean_val_indices)/len(val_1k)*100:.1f}%)")

    # 2. Extract Gold Entities
    gold_val_all = extract_gold_entities(val_1k)
    gold_val_clean = [gold_val_all[i] for i in clean_val_indices]
    clean_examples = [val_1k[i] for i in clean_val_indices]

    # Reference Label Mapping
    ref_cfg = AutoConfig.from_pretrained(CHECKPOINTS["orig_gbert"])
    id2label = {int(k): v for k, v in ref_cfg.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    # 3. Collect Probabilities Sequentially on val_1k (with lightweight cache)
    probs_cache_path = CACHE_DIR / "phase_35a_val_1k_probs.pt"
    t_start = time.time()
    if probs_cache_path.exists():
        print(f"\nLoading cached probabilities for all 5 models from {probs_cache_path}...")
        probs = torch.load(probs_cache_path, map_location="cpu")
    else:
        probs = {}
        for name, ckpt_path in CHECKPOINTS.items():
            print(f"\nCollecting probabilities for {name} from {ckpt_path}...")
            t0 = time.time()
            p_list = collect_active_probabilities(ckpt_path, val_1k, label2id, device, batch_size=16)
            probs[name] = p_list
            print(f"Completed in {time.time() - t0:.1f}s.")
        print(f"Saving collected probabilities to {probs_cache_path}...")
        torch.save(probs, probs_cache_path)

    print(f"\nAll 5 models ready in {time.time() - t_start:.1f}s.")

    # 4. Construct Probability Fusions
    # System 1: Original Champion (Exp 10): 0.40 orig_gbert + 0.20 xlmr42 + 0.20 xlmr7 + 0.20 xlmr21
    # System 2: DAPT Challenger (Phase 34A): 0.40 dapt_gbert + 0.20 xlmr42 + 0.20 xlmr7 + 0.20 xlmr21
    # System 3: Five-Model Ensemble: 0.20 orig_gbert + 0.20 dapt_gbert + 0.20 xlmr42 + 0.20 xlmr7 + 0.20 xlmr21
    # System 4: Standalone Original GBERT
    # System 5: Standalone DAPT GBERT

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

    # 5. Decode Predictions with MIN 0.60
    preds_all = {}
    preds_clean = {}
    for sys_name, p_list in fusions.items():
        p_dec = decode_predictions(p_list, val_1k, id2label, conf_thresh=0.60)
        preds_all[sys_name] = p_dec
        preds_clean[sys_name] = [p_dec[i] for i in clean_val_indices]

    # 6. Compute Metrics
    print("\n" + "=" * 80)
    print("LOCAL VALIDATION RESULTS SUMMARY")
    print("=" * 80)

    results = {"val_1k_contaminated_diagnostic": {}, "clean_489_unpolluted": {}, "bootstrap": {}}

    print("\n--- A. Split-A val_1k (1,000 Records) [CONTAMINATED DIAGNOSTIC ONLY] ---")
    print(f"{'System':<25} | {'F0.2':<9} | {'Precision':<9} | {'Recall':<9} | {'TP':<6} | {'FP':<6} | {'FN':<6}")
    print("-" * 75)
    for sys_name, preds in preds_all.items():
        m = compute_metrics(gold_val_all, preds, val_1k)
        results["val_1k_contaminated_diagnostic"][sys_name] = {
            "overall": m["overall"],
            "category": m["category"],
            "aspect": m["aspect"],
        }
        o = m["overall"]
        print(f"{sys_name:<25} | {o['f_beta']:<9.6f} | {o['precision']:<9.6f} | {o['recall']:<9.6f} | {o['tp']:<6} | {o['fp']:<6} | {o['fn']:<6}")

    print("\n--- B. Clean Unpolluted Holdout (489 Records) [STRICTLY UNSEEN BY ALL 5] ---")
    print(f"{'System':<25} | {'F0.2':<9} | {'Precision':<9} | {'Recall':<9} | {'TP':<6} | {'FP':<6} | {'FN':<6}")
    print("-" * 75)
    for sys_name, preds in preds_clean.items():
        m = compute_metrics(gold_val_clean, preds, clean_examples)
        results["clean_489_unpolluted"][sys_name] = {
            "overall": m["overall"],
            "category": m["category"],
            "aspect": m["aspect"],
        }
        o = m["overall"]
        print(f"{sys_name:<25} | {o['f_beta']:<9.6f} | {o['precision']:<9.6f} | {o['recall']:<9.6f} | {o['tp']:<6} | {o['fp']:<6} | {o['fn']:<6}")

    # 7. Bootstrap Confidence Intervals on Clean Holdout
    print("\n--- C. Paired Bootstrap Differences on Clean Holdout (1,000 Resamples) ---")
    boot_5m_vs_orig = compute_paired_bootstrap(gold_val_clean, preds_clean["orig_champ_4m"], preds_clean["five_model_5m"])
    boot_5m_vs_dapt = compute_paired_bootstrap(gold_val_clean, preds_clean["dapt_champ_4m"], preds_clean["five_model_5m"])
    results["bootstrap"]["5m_vs_orig_clean"] = boot_5m_vs_orig
    results["bootstrap"]["5m_vs_dapt_clean"] = boot_5m_vs_dapt

    print(f"5-Model vs Original Champion: Delta F0.2 = {boot_5m_vs_orig['mean_diff']:+.6f} [95% CI: {boot_5m_vs_orig['ci_lower_95']:+.6f}, {boot_5m_vs_orig['ci_upper_95']:+.6f}], p = {boot_5m_vs_orig['p_value']:.4f}")
    print(f"5-Model vs DAPT Champion:     Delta F0.2 = {boot_5m_vs_dapt['mean_diff']:+.6f} [95% CI: {boot_5m_vs_dapt['ci_lower_95']:+.6f}, {boot_5m_vs_dapt['ci_upper_95']:+.6f}], p = {boot_5m_vs_dapt['p_value']:.4f}")

    # 8. Detailed Disagreement and Venn Analysis on val_1k
    print("\n" + "=" * 80)
    print("DETAILED DISAGREEMENT & VENN OVERLAP ANALYSIS")
    print("=" * 80)

    # Convert predictions to record-level sets
    # Format: (record_number, aspect, value)
    def to_flat_set(preds_list, examples_list):
        flat = set()
        for idx, p_list in enumerate(preds_list):
            rec_id = examples_list[idx]["record_number"]
            for a, v in p_list:
                flat.add((rec_id, a, v))
        return flat

    set_orig = to_flat_set(preds_all["orig_champ_4m"], val_1k)
    set_dapt = to_flat_set(preds_all["dapt_champ_4m"], val_1k)
    set_5m = to_flat_set(preds_all["five_model_5m"], val_1k)
    set_sg_orig = to_flat_set(preds_all["orig_gbert_standalone"], val_1k)
    set_sg_dapt = to_flat_set(preds_all["dapt_gbert_standalone"], val_1k)

    shared_all_3 = set_orig & set_dapt & set_5m
    unique_orig_champ = set_orig - set_dapt - set_5m
    unique_dapt_champ = set_dapt - set_orig - set_5m
    unique_5m = set_5m - set_orig - set_dapt

    added_by_5m_vs_orig = set_5m - set_orig
    removed_by_5m_vs_orig = set_orig - set_5m

    added_by_5m_vs_dapt = set_5m - set_dapt
    removed_by_5m_vs_dapt = set_dapt - set_5m

    unique_to_sg_orig = set_sg_orig - set_sg_dapt
    unique_to_sg_dapt = set_sg_dapt - set_sg_orig

    print(f"Total Predictions: Orig Champ = {len(set_orig):,}, DAPT Champ = {len(set_dapt):,}, 5-Model = {len(set_5m):,}")
    print(f"Shared by ALL 3 Systems:          {len(shared_all_3):,} ({(len(shared_all_3)/len(set_orig))*100:.1f}% of orig champ)")
    print(f"Unique to Orig Champion:          {len(unique_orig_champ):,}")
    print(f"Unique to DAPT Champion:          {len(unique_dapt_champ):,}")
    print(f"Unique to 5-Model Fusion:         {len(unique_5m):,}")
    print(f"Unique to Standalone Orig GBERT:  {len(unique_to_sg_orig):,}")
    print(f"Unique to Standalone DAPT GBERT:  {len(unique_to_sg_dapt):,}")
    print(f"\n5-Model vs Orig Champion: Added = {len(added_by_5m_vs_orig):,}, Removed = {len(removed_by_5m_vs_orig):,}, Net Delta = {len(set_5m) - len(set_orig):+,}")
    print(f"5-Model vs DAPT Champion: Added = {len(added_by_5m_vs_dapt):,}, Removed = {len(removed_by_5m_vs_dapt):,}, Net Delta = {len(set_5m) - len(set_dapt):+,}")

    # Disagreement Aspect Distribution
    disagree_aspects = Counter([a for r, a, v in (set_orig ^ set_dapt)])
    print("\nTop Aspects with Disagreements between Orig and DAPT Champions:")
    for asp, count in disagree_aspects.most_common(8):
        print(f"  {asp:<35}: {count:,} disagreements")

    # Vehicle-Model Focus
    vm_orig = set(r for r in set_orig if r[1] == "Kompatibles_Fahrzeug_Modell")
    vm_dapt = set(r for r in set_dapt if r[1] == "Kompatibles_Fahrzeug_Modell")
    vm_5m = set(r for r in set_5m if r[1] == "Kompatibles_Fahrzeug_Modell")
    print(f"\nVehicle-Model (Kompatibles_Fahrzeug_Modell) Predictions:")
    print(f"  Orig Champion: {len(vm_orig):,}")
    print(f"  DAPT Champion: {len(vm_dapt):,}")
    print(f"  5-Model:       {len(vm_5m):,}")
    print(f"  Shared:        {len(vm_orig & vm_dapt):,}")
    print(f"  Orig Only:     {len(vm_orig - vm_dapt):,}")
    print(f"  DAPT Only:     {len(vm_dapt - vm_orig):,}")

    disagreement_stats = {
        "total_preds": {"orig": len(set_orig), "dapt": len(set_dapt), "five_model": len(set_5m)},
        "shared_all_3": len(shared_all_3),
        "unique_orig_champ": len(unique_orig_champ),
        "unique_dapt_champ": len(unique_dapt_champ),
        "unique_5m": len(unique_5m),
        "unique_sg_orig": len(unique_to_sg_orig),
        "unique_sg_dapt": len(unique_to_sg_dapt),
        "vs_orig": {"added": len(added_by_5m_vs_orig), "removed": len(removed_by_5m_vs_orig), "net": len(set_5m) - len(set_orig)},
        "vs_dapt": {"added": len(added_by_5m_vs_dapt), "removed": len(removed_by_5m_vs_dapt), "net": len(set_5m) - len(set_dapt)},
        "disagree_aspects_top8": dict(disagree_aspects.most_common(8)),
        "vehicle_model": {
            "orig": len(vm_orig),
            "dapt": len(vm_dapt),
            "five_model": len(vm_5m),
            "shared": len(vm_orig & vm_dapt),
            "orig_only": len(vm_orig - vm_dapt),
            "dapt_only": len(vm_dapt - vm_orig),
        },
    }
    results["disagreement_stats"] = disagreement_stats

    # Save to Cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nLocal evaluation results saved to {RESULTS_PATH}")

    return results


if __name__ == "__main__":
    run_evaluation()
