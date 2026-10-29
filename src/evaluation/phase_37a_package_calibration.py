"""
Phase 37A: Package Contents Precision Calibration Research.

Investigates whether increasing the confidence threshold for Im_Lieferumfang_Enthalten
improves precision-weighted F0.2 beyond the current official Phase 36C champion:
- Frozen Phase 36C Champion: Vehicle 0.80, Produktart 0.80, All others 0.60
- Evaluates Im_Lieferumfang_Enthalten thresholds: 0.60 (baseline), 0.70, 0.80
- On the 489-title clean holdout using cached ensemble probabilities.
- Reconstructs Phase 36C local baseline independently.
- Computes overall, category-level, and aspect-level metrics.
- Performs 1,000 paired title-level bootstrap resamples (seed 42).
- Extracts confidence distributions and qualitative error analysis (TPs lost vs FPs removed).
"""

from collections import Counter, defaultdict
import json
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
OUTPUT_REPORT_DATA = CACHE_DIR / "phase_37a_package_calibration_results.json"

TARGET_ASPECT = "Im_Lieferumfang_Enthalten"
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


def extract_package_error_breakdown(
    clean_examples: List[dict],
    spanned_gold: List[List[Dict[str, Any]]],
    base_spanned: List[List[Dict[str, Any]]],
    threshold: float,
) -> Dict[str, Any]:
    """Extract individual TPs lost and FPs removed when raising threshold from 0.60 to threshold."""
    fps_removed = []
    tps_lost = []
    removed_confs = []

    for ex_idx, (g_list, p_list) in enumerate(zip(spanned_gold, base_spanned)):
        ex = clean_examples[ex_idx]
        cat = ex["category"]
        title = " ".join(ex["tokens"])
        rec_id = ex["record_number"]

        g_asp = [g["value"] for g in g_list if g["aspect"] == TARGET_ASPECT]
        p_asp = [p for p in p_list if p["aspect"] == TARGET_ASPECT]

        g_counter = Counter(g_asp)
        for p in p_asp:
            val = p["value"]
            conf = float(p["min_conf"])
            if conf < threshold:
                removed_confs.append(conf)
                if g_counter[val] > 0:
                    g_counter[val] -= 1
                    tps_lost.append({"rec_id": rec_id, "cat": cat, "val": val, "conf": conf, "title": title})
                else:
                    fps_removed.append({"rec_id": rec_id, "cat": cat, "val": val, "conf": conf, "title": title})
            else:
                if g_counter[val] > 0:
                    g_counter[val] -= 1

    return {
        "threshold": threshold,
        "fps_removed": fps_removed,
        "tps_lost": tps_lost,
        "fp_removed_count": len(fps_removed),
        "tp_lost_count": len(tps_lost),
        "total_removed": len(fps_removed) + len(tps_lost),
        "removed_confidence_dist": summarize_distribution(removed_confs),
    }


def run_phase_37a_package_calibration() -> Dict[str, Any]:
    print("=" * 80)
    print("PHASE 37A: PACKAGE CONTENTS PRECISION CALIBRATION RESEARCH")
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

    # 2. Reconstruct Phase 36C Champion Baseline: Vehicle 0.80, Produktart 0.80, Others 0.60
    print("\n[Step 2] Reconstructing Phase 36C Champion baseline (Vehicle 0.80 / Produktart 0.80 / Others 0.60)...")
    p36c_spanned = decode_spanned_predictions(
        clean_p_fused, clean_examples, id2label, conf_thresh=0.60,
        aspect_thresholds={"Kompatibles_Fahrzeug_Modell": 0.80, "Produktart": 0.80},
    )
    p36c_tuples = [[(e["aspect"], e["value"]) for e in ex_ents] for ex_ents in p36c_spanned]
    p36c_metrics = compute_metrics_from_tuples(gold_clean_tuples, p36c_tuples, clean_examples)

    ov = p36c_metrics["overall"]
    c1 = p36c_metrics["category"][1]
    c2 = p36c_metrics["category"][2]
    lie = p36c_metrics["aspect"][TARGET_ASPECT]

    print("Phase 36C Champion Baseline on 489 Clean Titles:")
    print(f"  Overall: TP={ov['tp']}, FP={ov['fp']}, FN={ov['fn']}, P={ov['precision']:.6f}, R={ov['recall']:.6f}, F0.2={ov['f_beta']:.6f}")
    print(f"  Category 1: TP={c1['tp']}, FP={c1['fp']}, FN={c1['fn']}, P={c1['precision']:.6f}, R={c1['recall']:.6f}, F0.2={c1['f_beta']:.6f}")
    print(f"  Category 2: TP={c2['tp']}, FP={c2['fp']}, FN={c2['fn']}, P={c2['precision']:.6f}, R={c2['recall']:.6f}, F0.2={c2['f_beta']:.6f}")
    print(f"  Im_Lieferumfang_Enthalten: TP={lie['tp']}, FP={lie['fp']}, FN={lie['fn']}, P={lie['precision']:.6f}, R={lie['recall']:.6f}, F0.2={lie['f_beta']:.6f}")

    assert ov["tp"] == 3045 and ov["fp"] == 179 and ov["fn"] == 728, "Phase 36C baseline overall mismatch!"
    assert c1["tp"] == 2028 and c1["fp"] == 129 and c1["fn"] == 502, "Phase 36C category 1 mismatch!"
    assert c2["tp"] == 1017 and c2["fp"] == 50 and c2["fn"] == 226, "Phase 36C category 2 mismatch!"
    assert lie["tp"] == 666 and lie["fp"] == 49 and lie["fn"] == 44, "Phase 36C Lieferumfang mismatch!"
    print("RECONSTRUCTION CONFIRMED: 100% exact match for Phase 36C champion baseline.")

    # 3. Simulate Isolated Threshold Configurations for Im_Lieferumfang_Enthalten
    print("\n[Step 3] Evaluating candidate thresholds for Im_Lieferumfang_Enthalten (0.70, 0.80)...")
    candidate_results = {}
    error_breakdowns = {}

    for th in [0.70, 0.80]:
        key = f"th_{int(th*100)}"
        cand_spanned = []
        preds_removed = 0

        for i in range(len(clean_examples)):
            cur_p = []
            for e in p36c_spanned[i]:
                if e["aspect"] == TARGET_ASPECT:
                    if e["min_conf"] >= th:
                        cur_p.append(e)
                    else:
                        preds_removed += 1
                else:
                    cur_p.append(e)
            cand_spanned.append(cur_p)

        cand_tuples = [[(e["aspect"], e["value"]) for e in ex] for ex in cand_spanned]
        cand_metrics = compute_metrics_from_tuples(gold_clean_tuples, cand_tuples, clean_examples)

        c_ov = cand_metrics["overall"]
        c_c1 = cand_metrics["category"][1]
        c_c2 = cand_metrics["category"][2]
        c_lie = cand_metrics["aspect"][TARGET_ASPECT]

        tp_lost = ov["tp"] - c_ov["tp"]
        fp_removed = ov["fp"] - c_ov["fp"]
        fn_introduced = c_ov["fn"] - ov["fn"]
        delta_f02 = c_ov["f_beta"] - ov["f_beta"]
        delta_prec = c_ov["precision"] - ov["precision"]
        delta_rec = c_ov["recall"] - ov["recall"]
        delta_c1_f02 = c_c1["f_beta"] - c1["f_beta"]
        delta_c2_f02 = c_c2["f_beta"] - c2["f_beta"]

        # Bootstrap: candidate minus Phase 36C baseline
        boot = compute_paired_bootstrap(
            gold_clean_tuples, p36c_tuples, cand_tuples,
            num_bootstraps=1000, seed=42,
        )

        eb = extract_package_error_breakdown(clean_examples, spanned_gold, p36c_spanned, threshold=th)
        error_breakdowns[key] = eb

        candidate_results[key] = {
            "aspect": TARGET_ASPECT,
            "threshold": th,
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
            "category_1": {
                "tp": c_c1["tp"],
                "fp": c_c1["fp"],
                "fn": c_c1["fn"],
                "precision": c_c1["precision"],
                "recall": c_c1["recall"],
                "f02": c_c1["f_beta"],
                "delta_f02": delta_c1_f02,
            },
            "category_2": {
                "tp": c_c2["tp"],
                "fp": c_c2["fp"],
                "fn": c_c2["fn"],
                "precision": c_c2["precision"],
                "recall": c_c2["recall"],
                "f02": c_c2["f_beta"],
                "delta_f02": delta_c2_f02,
            },
            "aspect_metrics": {
                "tp": c_lie["tp"],
                "fp": c_lie["fp"],
                "fn": c_lie["fn"],
                "precision": c_lie["precision"],
                "recall": c_lie["recall"],
                "f02": c_lie["f_beta"],
            },
            "bootstrap_mean_diff": boot["mean_diff"],
            "ci_lower_95": boot["ci_lower_95"],
            "ci_upper_95": boot["ci_upper_95"],
            "bootstrap_p_value": boot["p_value"],
            "error_breakdown_summary": {
                "fp_removed_count": eb["fp_removed_count"],
                "tp_lost_count": eb["tp_lost_count"],
                "ratio_tp_lost_per_fp_removed": (eb["tp_lost_count"] / eb["fp_removed_count"]) if eb["fp_removed_count"] > 0 else 0.0,
                "confidence_dist": eb["removed_confidence_dist"],
            },
        }

        print(f"\n--- Candidate: Im_Lieferumfang_Enthalten @ {th:.2f} ---")
        print(f"  TP: {c_ov['tp']} (lost {tp_lost}), FP: {c_ov['fp']} (removed {fp_removed}), FN: {c_ov['fn']} (+{fn_introduced})")
        print(f"  Predictions removed: {preds_removed} (TP lost: {tp_lost}, FP removed: {fp_removed})")
        print(f"  Precision: {c_ov['precision']:.6f} ({delta_prec:+.6f})")
        print(f"  Recall:    {c_ov['recall']:.6f} ({delta_rec:+.6f})")
        print(f"  F0.2:      {c_ov['f_beta']:.6f} (delta: {delta_f02:+.6f})")
        print(f"  Category 1 F0.2: {c_c1['f_beta']:.6f} (delta: {delta_c1_f02:+.6f})")
        print(f"  Category 2 F0.2: {c_c2['f_beta']:.6f} (delta: {delta_c2_f02:+.6f})")
        print(f"  Lieferumfang Precision: {c_lie['precision']:.6f}, Recall: {c_lie['recall']:.6f}, F0.2: {c_lie['f_beta']:.6f}")
        print(f"  Bootstrap 95% CI: [{boot['ci_lower_95']:+.6f}, {boot['ci_upper_95']:+.6f}], p-value: {boot['p_value']:.4f}")
        print(f"  Trade-off: {tp_lost} TPs lost to remove {fp_removed} FPs (Ratio: {tp_lost/fp_removed:.2f} TPs lost per FP removed)")

    # 4. Save Results
    output_data = {
        "phase_36c_baseline": {
            "overall": ov,
            "category_1": c1,
            "category_2": c2,
            "lieferumfang": lie,
        },
        "candidates": candidate_results,
        "error_breakdowns": error_breakdowns,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(OUTPUT_REPORT_DATA, "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(output_data), f, indent=2)
    print(f"\nSaved research results to {OUTPUT_REPORT_DATA}")

    return output_data


if __name__ == "__main__":
    run_phase_37a_package_calibration()
