"""
Phase 36B: Aspect-Specific Confidence Calibration Research.

Investigates targeted confidence filtering on non-vehicle aspects:
- Produktart
- Im_Lieferumfang_Enthalten
- Kompatible_Fahrzeug_Marke

Evaluates:
- Baseline Phase 35C Champion (Vehicle 0.80, all others 0.60) on 489-title clean holdout.
- Confidence distribution and statistical separation (KS test, Mann-Whitney AUC) of TPs vs FPs.
- Isolated threshold sensitivity (0.60, 0.70, 0.80) per aspect.
- Paired title-level bootstrap (1,000 resamples, seed 42) for Delta F0.2 with 95% CIs.
- Category 1 vs Category 2 impacts.
- Risk analysis and decision gate.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.stats as stats
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
    precision_recall_fbeta,
    filter_entities_by_category,
)
from src.evaluation.phase_35b_audit_and_error_analysis import (
    extract_spanned_gold_entities,
    decode_spanned_predictions,
    compute_metrics_from_tuples,
    compute_paired_bootstrap,
    make_json_serializable,
)

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
HOLDOUT_MANIFEST_PATH = CACHE_DIR / "phase_35b_clean_holdout_manifest.json"
VAL_PROBS_PATH = CACHE_DIR / "phase_35a_val_1k_probs.pt"
OUTPUT_REPORT_DATA = CACHE_DIR / "phase_36b_calibration_results.json"

TARGET_ASPECTS = [
    "Produktart",
    "Im_Lieferumfang_Enthalten",
    "Kompatible_Fahrzeug_Marke",
]
CANDIDATE_THRESHOLDS = [0.60, 0.70, 0.80]


def summarize_distribution(vals: List[float]) -> Dict[str, Any]:
    """Compute summary statistics for confidence scores."""
    if not vals:
        return {
            "n": 0, "mean": 0.0, "std": 0.0, "min": 0.0,
            "q25": 0.0, "median": 0.0, "q75": 0.0, "q90": 0.0, "max": 0.0,
        }
    arr = np.array(vals)
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "q75": float(np.percentile(arr, 75)),
        "q90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def analyze_aspect_confidence_separation(
    spanned_gold: List[List[Dict[str, Any]]],
    base_spanned: List[List[Dict[str, Any]]],
    clean_examples: List[dict],
    aspect: str,
) -> Dict[str, Any]:
    """Analyze TP vs FP confidence distributions and separation metrics for a single aspect."""
    tp_confs = []
    fp_confs = []
    tp_count = 0
    fp_count = 0
    fn_count = 0
    gold_count = 0
    pred_count = 0

    cat_stats = {
        1: {"gold": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0, "tp_confs": [], "fp_confs": []},
        2: {"gold": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0, "tp_confs": [], "fp_confs": []},
    }

    for ex_idx, (g_list, p_list) in enumerate(zip(spanned_gold, base_spanned)):
        ex = clean_examples[ex_idx]
        cat = ex["category"]

        g_asp = [g["value"] for g in g_list if g["aspect"] == aspect]
        p_asp = [p for p in p_list if p["aspect"] == aspect]

        gold_count += len(g_asp)
        pred_count += len(p_asp)
        cat_stats[cat]["gold"] += len(g_asp)
        cat_stats[cat]["pred"] += len(p_asp)

        g_counter = Counter(g_asp)
        for p in p_asp:
            val = p["value"]
            c = float(p["min_conf"])
            if g_counter[val] > 0:
                g_counter[val] -= 1
                tp_count += 1
                tp_confs.append(c)
                cat_stats[cat]["tp"] += 1
                cat_stats[cat]["tp_confs"].append(c)
            else:
                fp_count += 1
                fp_confs.append(c)
                cat_stats[cat]["fp"] += 1
                cat_stats[cat]["fp_confs"].append(c)

        fn_this = sum(g_counter.values())
        fn_count += fn_this
        cat_stats[cat]["fn"] += fn_this

    prec = tp_count / pred_count if pred_count > 0 else 0.0
    rec = tp_count / gold_count if gold_count > 0 else 0.0
    denom = 0.04 * prec + rec
    f02 = 1.04 * prec * rec / denom if denom > 0 else 0.0

    ks_stat, ks_pval = 0.0, 1.0
    auc, mw_pval = 0.5, 1.0
    if tp_confs and fp_confs:
        ks_res = stats.ks_2samp(tp_confs, fp_confs)
        ks_stat = float(ks_res.statistic)
        ks_pval = float(ks_res.pvalue)

        mw_res = stats.mannwhitneyu(tp_confs, fp_confs, alternative="two-sided")
        auc = float(mw_res.statistic / (len(tp_confs) * len(fp_confs)))
        mw_pval = float(mw_res.pvalue)

    cat_breakdown = {}
    for cat in [1, 2]:
        cs = cat_stats[cat]
        c_p = cs["tp"] / cs["pred"] if cs["pred"] > 0 else 0.0
        c_r = cs["tp"] / cs["gold"] if cs["gold"] > 0 else 0.0
        c_d = 0.04 * c_p + c_r
        c_f02 = 1.04 * c_p * c_r / c_d if c_d > 0 else 0.0
        cat_breakdown[f"category_{cat}"] = {
            "gold": cs["gold"],
            "pred": cs["pred"],
            "tp": cs["tp"],
            "fp": cs["fp"],
            "fn": cs["fn"],
            "precision": c_p,
            "recall": c_r,
            "f02": c_f02,
            "tp_dist": summarize_distribution(cs["tp_confs"]),
            "fp_dist": summarize_distribution(cs["fp_confs"]),
        }

    return {
        "aspect": aspect,
        "gold_count": gold_count,
        "pred_count": pred_count,
        "tp": tp_count,
        "fp": fp_count,
        "fn": fn_count,
        "precision": prec,
        "recall": rec,
        "f02": f02,
        "tp_distribution": summarize_distribution(tp_confs),
        "fp_distribution": summarize_distribution(fp_confs),
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_pval,
        "mann_whitney_auc": auc,
        "mann_whitney_pvalue": mw_pval,
        "categories": cat_breakdown,
    }


def simulate_isolated_threshold(
    gold_clean_tuples: List[List[Tuple[str, str]]],
    base_spanned: List[List[Dict[str, Any]]],
    base_tuples: List[List[Tuple[str, str]]],
    base_ov: Dict[str, Any],
    base_cat: Dict[int, Dict[str, Any]],
    clean_examples: List[dict],
    target_aspect: str,
    threshold: float,
    num_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Simulate isolated threshold modification for one aspect against baseline."""
    n_examples = len(clean_examples)
    cand_spanned = []
    preds_removed = 0

    for i in range(n_examples):
        cur_p = []
        for e in base_spanned[i]:
            if e["aspect"] == target_aspect:
                if e["min_conf"] >= threshold:
                    cur_p.append(e)
                else:
                    preds_removed += 1
            else:
                cur_p.append(e)
        cand_spanned.append(cur_p)

    cand_tuples = [[(e["aspect"], e["value"]) for e in ex] for ex in cand_spanned]
    cand_metrics = compute_metrics_from_tuples(gold_clean_tuples, cand_tuples, clean_examples)
    c_ov = cand_metrics["overall"]
    c_cat1 = cand_metrics["category"][1]
    c_cat2 = cand_metrics["category"][2]

    tp_lost = base_ov["tp"] - c_ov["tp"]
    fp_removed = base_ov["fp"] - c_ov["fp"]
    fn_introduced = c_ov["fn"] - base_ov["fn"]
    delta_f02 = c_ov["f_beta"] - base_ov["f_beta"]
    delta_prec = c_ov["precision"] - base_ov["precision"]
    delta_rec = c_ov["recall"] - base_ov["recall"]
    delta_cat1_f02 = c_cat1["f_beta"] - base_cat[1]["f_beta"]
    delta_cat2_f02 = c_cat2["f_beta"] - base_cat[2]["f_beta"]

    boot = compute_paired_bootstrap(
        gold_clean_tuples, base_tuples, cand_tuples,
        num_bootstraps=num_bootstraps, seed=seed,
    )

    return {
        "aspect": target_aspect,
        "threshold": threshold,
        "tp": c_ov["tp"],
        "fp": c_ov["fp"],
        "fn": c_ov["fn"],
        "precision": c_ov["precision"],
        "recall": c_ov["recall"],
        "f02": c_ov["f_beta"],
        "delta_precision": delta_prec,
        "delta_recall": delta_rec,
        "delta_f02": delta_f02,
        "tp_lost": tp_lost,
        "fp_removed": fp_removed,
        "fn_introduced": fn_introduced,
        "preds_removed": preds_removed,
        "cat1_f02": c_cat1["f_beta"],
        "delta_cat1_f02": delta_cat1_f02,
        "cat2_f02": c_cat2["f_beta"],
        "delta_cat2_f02": delta_cat2_f02,
        "bootstrap_mean_diff": boot["mean_diff"],
        "ci_lower_95": boot["ci_lower_95"],
        "ci_upper_95": boot["ci_upper_95"],
        "bootstrap_p_value": boot["p_value"],
    }


def run_phase_36b_calibration_research() -> Dict[str, Any]:
    """Execute complete Phase 36B calibration research pipeline."""
    print("=" * 80)
    print("PHASE 36B: ASPECT-SPECIFIC CONFIDENCE CALIBRATION RESEARCH")
    print("=" * 80)

    # 1. Load Data and Manifest
    print("\n[Step 1] Loading clean holdout manifest and cached probabilities...")
    df = load_training_data()
    all_examples = build_examples(df)
    train_4k, val_1k = split_examples(all_examples, random_state=42)

    with open(HOLDOUT_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    clean_ids = set(manifest["record_ids"])
    clean_val_indices = [idx for idx, ex in enumerate(val_1k) if ex["record_number"] in clean_ids]
    clean_examples = [val_1k[i] for i in clean_val_indices]
    print(f"Loaded {len(clean_examples)} clean holdout titles from {HOLDOUT_MANIFEST_PATH}")

    probs = torch.load(VAL_PROBS_PATH, map_location="cpu", weights_only=False)
    ref_cfg = AutoConfig.from_pretrained("checkpoints/gbert_baseline_best")
    id2label = {int(k): v for k, v in ref_cfg.id2label.items()}

    clean_p_fused = [
        0.40 * probs["dapt_gbert"][i] + 0.20 * probs["xlmr_seed42"][i] + 0.20 * probs["xlmr_seed7"][i] + 0.20 * probs["xlmr_seed21"][i]
        for i in clean_val_indices
    ]

    spanned_gold = extract_spanned_gold_entities(clean_examples)
    gold_clean_tuples = [[(e["aspect"], e["value"]) for e in ex_ents] for ex_ents in spanned_gold]

    # 2. Decode Baseline Predictions (Phase 35C Champion: Vehicle 0.80, Others 0.60)
    print("\n[Step 2] Decoding Phase 35C Champion baseline (Vehicle 0.80 / Others 0.60)...")
    base_spanned = decode_spanned_predictions(
        clean_p_fused, clean_examples, id2label, conf_thresh=0.60,
        aspect_thresholds={"Kompatibles_Fahrzeug_Modell": 0.80},
    )
    base_tuples = [[(e["aspect"], e["value"]) for e in ex_ents] for ex_ents in base_spanned]
    base_metrics = compute_metrics_from_tuples(gold_clean_tuples, base_tuples, clean_examples)
    base_ov = base_metrics["overall"]
    base_cat = base_metrics["category"]

    print("Phase 35C Champion Baseline on 489 Clean Titles:")
    print(f"  F0.2: {base_ov['f_beta']:.6f} | Precision: {base_ov['precision']:.6f} | Recall: {base_ov['recall']:.6f}")
    print(f"  TP: {base_ov['tp']} | FP: {base_ov['fp']} | FN: {base_ov['fn']} | Total Gold (TP+FN): {base_ov['tp'] + base_ov['fn']}")
    assert base_ov["tp"] == 3072 and base_ov["fp"] == 199 and base_ov["fn"] == 701, "Phase 35C baseline invariant violated!"

    # 3. Analyze Confidence Separation for Target Aspects
    print("\n[Step 3] Analyzing confidence distributions and separation for non-vehicle aspects...")
    separation_results = {}
    for asp in TARGET_ASPECTS:
        sep = analyze_aspect_confidence_separation(spanned_gold, base_spanned, clean_examples, asp)
        separation_results[asp] = sep
        print(f"\n--- Aspect: {asp} ---")
        print(f"  Gold: {sep['gold_count']}, Pred: {sep['pred_count']}, TP: {sep['tp']}, FP: {sep['fp']}, FN: {sep['fn']}")
        print(f"  Precision: {sep['precision']:.6f}, Recall: {sep['recall']:.6f}, F0.2: {sep['f02']:.6f}")
        tp_d = sep["tp_distribution"]
        fp_d = sep["fp_distribution"]
        print(f"  TP Conf: Mean={tp_d['mean']:.4f}, Median={tp_d['median']:.4f}, Q25={tp_d['q25']:.4f}, Q75={tp_d['q75']:.4f}")
        print(f"  FP Conf: Mean={fp_d['mean']:.4f}, Median={fp_d['median']:.4f}, Q25={fp_d['q25']:.4f}, Q75={fp_d['q75']:.4f}")
        print(f"  KS statistic: {sep['ks_statistic']:.4f} (p-value: {sep['ks_pvalue']:.4e})")
        print(f"  Mann-Whitney AUC (TP > FP): {sep['mann_whitney_auc']:.4f} (p-value: {sep['mann_whitney_pvalue']:.4e})")

    # 4. Isolated Threshold Sensitivity Analysis
    print("\n[Step 4] Running isolated threshold sensitivity analysis (0.70, 0.80)...")
    threshold_results = {}
    for asp in TARGET_ASPECTS:
        for th in [0.70, 0.80]:
            key = f"{asp}_th{int(th*100)}"
            sim = simulate_isolated_threshold(
                gold_clean_tuples, base_spanned, base_tuples, base_ov, base_cat,
                clean_examples, target_aspect=asp, threshold=th,
                num_bootstraps=1000, seed=42,
            )
            threshold_results[key] = sim
            print(f"\n--- Configuration: {asp} @ {th:.2f} ---")
            print(f"  TP: {sim['tp']} (lost {sim['tp_lost']}), FP: {sim['fp']} (removed {sim['fp_removed']}), FN: {sim['fn']} (+{sim['fn_introduced']})")
            print(f"  Precision: {sim['precision']:.6f} ({sim['delta_precision']:+.6f})")
            print(f"  Recall:    {sim['recall']:.6f} ({sim['delta_recall']:+.6f})")
            print(f"  F0.2:      {sim['f02']:.6f} (delta: {sim['delta_f02']:+.6f})")
            print(f"  Cat 1 F0.2: {sim['cat1_f02']:.6f} (delta: {sim['delta_cat1_f02']:+.6f})")
            print(f"  Cat 2 F0.2: {sim['cat2_f02']:.6f} (delta: {sim['delta_cat2_f02']:+.6f})")
            print(f"  Bootstrap 95% CI: [{sim['ci_lower_95']:+.6f}, {sim['ci_upper_95']:+.6f}], p-value: {sim['bootstrap_p_value']:.4f}")

    # 5. Save Results
    output_data = {
        "baseline_champion": {
            "name": "Phase 35C Champion",
            "thresholds": {"Kompatibles_Fahrzeug_Modell": 0.80, "others": 0.60},
            "overall": base_ov,
            "category_1": base_cat[1],
            "category_2": base_cat[2],
        },
        "separation_analysis": separation_results,
        "threshold_simulations": threshold_results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(OUTPUT_REPORT_DATA, "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(output_data), f, indent=2)
    print(f"\nSaved research results to {OUTPUT_REPORT_DATA}")

    return output_data


if __name__ == "__main__":
    run_phase_36b_calibration_research()
