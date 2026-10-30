"""
Phase 39A: Complete Error Opportunity and Diagnostic Audit.

Chartered to perform an exhaustive, reproducible audit of the Phase 36C champion on the 489-title
clean holdout dataset to identify the highest-impact research direction for improving official EvalAI Quiz F0.2:
1. Compares Global Micro F0.2 vs Reconstructed Official-Style Hierarchical F0.2.
2. Constructs a mutually exclusive, deterministic 1-to-1 Error Taxonomy reconciling all 179 FPs and 728 FNs.
3. Investigates Boundary Errors (shortened vs overextended, compound words, model vs gold boundaries).
4. Analyzes False Negatives (completely undetected vs suppressed by threshold vs complementary model predictions).
5. Analyzes False Positives (boundary vs wrong-aspect vs spurious, high confidence, consensus).
6. Ranks the Top 10 Aspect/Category Score-Loss Opportunities using transparent counterfactual calculations.
7. Saves complete audit results to data/cache/phase_39a_error_audit_results.json.
"""

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from transformers import AutoConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_examples import build_examples, load_training_data
from src.data.make_splits import split_examples
from src.evaluation.phase_35b_audit_and_error_analysis import (
    extract_spanned_gold_entities,
    decode_spanned_predictions,
    make_json_serializable,
)
from src.evaluation.phase_39a_official_metric import (
    calculate_official_competition_metrics,
    calculate_global_micro_metrics,
    compute_fbeta,
    count_entity_matches,
)

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
HOLDOUT_MANIFEST_PATH = CACHE_DIR / "phase_35b_clean_holdout_manifest.json"
VAL_PROBS_PATH = CACHE_DIR / "phase_35a_val_1k_probs.pt"
OUTPUT_REPORT_DATA = CACHE_DIR / "phase_39a_error_audit_results.json"

MODEL_NAMES = ["dapt_gbert", "xlmr_seed42", "xlmr_seed7", "xlmr_seed21"]
MODEL_WEIGHTS = [0.40, 0.20, 0.20, 0.20]

ASPECT_THRESHOLDS = {
    "Kompatibles_Fahrzeug_Modell": 0.80,
    "Produktart": 0.80,
}
DEFAULT_THRESHOLD = 0.60


def build_error_taxonomy(
    clean_examples: List[dict],
    spanned_gold: List[List[Dict[str, Any]]],
    ens_spanned: List[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Constructs a deterministic, mutually exclusive 1-to-1 error taxonomy.
    Every false positive and false negative is accounted for without double counting.
    """
    taxonomy_counts = Counter()
    fp_by_category = Counter()
    fn_by_category = Counter()
    
    boundary_items = []
    aspect_mismatch_items = []
    combined_items = []
    spurious_items = []
    missing_items = []
    
    exact_tps_count = 0
    
    for i in range(len(clean_examples)):
        ex = clean_examples[i]
        cat = ex["category"]
        rec_id = ex["record_number"]
        title = " ".join(ex["tokens"])
        
        g_ents = list(spanned_gold[i])
        p_ents = list(ens_spanned[i])
        
        # Step 1: Match Exact TPs deterministically
        unmatched_g = []
        unmatched_p = list(p_ents)
        
        for g in g_ents:
            match_idx = None
            for p_idx, p in enumerate(unmatched_p):
                if p["aspect"] == g["aspect"] and p["value"] == g["value"]:
                    match_idx = p_idx
                    break
            if match_idx is not None:
                exact_tps_count += 1
                unmatched_p.pop(match_idx)
            else:
                unmatched_g.append(g)
                
        # Step 2: Match overlapping spans among remaining FPs and FNs
        overlap_pairs = []
        for p_idx, p in enumerate(unmatched_p):
            for g_idx, g in enumerate(unmatched_g):
                overlap = max(0, min(p["end"], g["end"]) - max(p["start"], g["start"]))
                if overlap > 0:
                    same_asp = (p["aspect"] == g["aspect"])
                    overlap_pairs.append((overlap, same_asp, p_idx, g_idx))
                    
        # Priority: largest overlap first, then same aspect
        overlap_pairs.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        matched_p = set()
        matched_g = set()
        
        for overlap, same_asp, p_idx, g_idx in overlap_pairs:
            if p_idx in matched_p or g_idx in matched_g:
                continue
            matched_p.add(p_idx)
            matched_g.add(g_idx)
            
            p = unmatched_p[p_idx]
            g = unmatched_g[g_idx]
            
            same_bounds = (p["start"] == g["start"] and p["end"] == g["end"])
            
            info = {
                "rec_id": rec_id,
                "cat": cat,
                "title": title,
                "pred_aspect": p["aspect"],
                "gold_aspect": g["aspect"],
                "pred_val": p["value"],
                "gold_val": g["value"],
                "pred_span": [p["start"], p["end"]],
                "gold_span": [g["start"], g["end"]],
                "conf": p["min_conf"],
                "overlap_tokens": overlap,
            }
            
            if same_asp and not same_bounds:
                p_len = p["end"] - p["start"]
                g_len = g["end"] - g["start"]
                subtype = "shortened" if p_len < g_len else ("overextended" if p_len > g_len else "shifted")
                info["subtype"] = subtype
                boundary_items.append(info)
                err_type = "boundary_mismatch"
            elif not same_asp and same_bounds:
                aspect_mismatch_items.append(info)
                err_type = "aspect_mismatch"
            elif not same_asp and not same_bounds:
                combined_items.append(info)
                err_type = "combined_boundary_aspect_mismatch"
            else:
                err_type = "unknown_overlap"
                
            taxonomy_counts[err_type] += 1
            fp_by_category[err_type] += 1
            fn_by_category[err_type] += 1
            
        # Step 3: Remaining unmatched predictions are Spurious Entities (FP)
        for p_idx, p in enumerate(unmatched_p):
            if p_idx not in matched_p:
                taxonomy_counts["spurious_entity"] += 1
                fp_by_category["spurious_entity"] += 1
                spurious_items.append({
                    "rec_id": rec_id,
                    "cat": cat,
                    "title": title,
                    "aspect": p["aspect"],
                    "value": p["value"],
                    "span": [p["start"], p["end"]],
                    "conf": p["min_conf"],
                })
                
        # Step 4: Remaining unmatched gold are Missing Entities (FN)
        for g_idx, g in enumerate(unmatched_g):
            if g_idx not in matched_g:
                taxonomy_counts["missing_entity"] += 1
                fn_by_category["missing_entity"] += 1
                missing_items.append({
                    "rec_id": rec_id,
                    "cat": cat,
                    "title": title,
                    "aspect": g["aspect"],
                    "value": g["value"],
                    "span": [g["start"], g["end"]],
                })
                
    return {
        "exact_true_positives": exact_tps_count,
        "taxonomy_summary": dict(taxonomy_counts),
        "total_false_positives": sum(fp_by_category.values()),
        "total_false_negatives": sum(fn_by_category.values()),
        "boundary_items": boundary_items,
        "aspect_mismatch_items": aspect_mismatch_items,
        "combined_items": combined_items,
        "spurious_items": spurious_items,
        "missing_items": missing_items,
    }


def analyze_false_negatives_in_depth(
    clean_examples: List[dict],
    spanned_gold: List[List[Dict[str, Any]]],
    ens_spanned: List[List[Dict[str, Any]]],
    indiv_raw: Dict[str, List[List[Dict[str, Any]]]],
    indiv_thresh: Dict[str, List[List[Dict[str, Any]]]],
    missing_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Analyzes why entities were missed:
    - Completely undetected across all 4 models.
    - Predicted by at least 1 individual model (complementary information lost in fusion).
    - Predicted by ensemble raw argmax but suppressed by aspect threshold.
    """
    missing_by_aspect = Counter()
    missing_by_cat = Counter()
    raw_support_dist = Counter()
    thresh_support_dist = Counter()
    complementary_recoverable = []
    
    for item in missing_items:
        rec_id = item["rec_id"]
        cat = item["cat"]
        asp = item["aspect"]
        val = item["value"]
        s, e = item["span"]
        title = item["title"]
        
        missing_by_aspect[asp] += 1
        missing_by_cat[cat] += 1
        
        # Find index in clean_examples
        ex_idx = next(i for i, ex in enumerate(clean_examples) if ex["record_number"] == rec_id)
        
        # Check raw support across 4 models
        raw_sup = {
            m: any(me["aspect"] == asp and me["start"] == s and me["end"] == e for me in indiv_raw[m][ex_idx])
            for m in MODEL_NAMES
        }
        num_raw = sum(raw_sup.values())
        raw_support_dist[num_raw] += 1
        
        # Check thresholded support across 4 models
        th_sup = {
            m: any(me["aspect"] == asp and me["start"] == s and me["end"] == e for me in indiv_thresh[m][ex_idx])
            for m in MODEL_NAMES
        }
        num_th = sum(th_sup.values())
        thresh_support_dist[num_th] += 1
        
        if num_raw > 0:
            complementary_recoverable.append({
                "rec_id": rec_id,
                "cat": cat,
                "title": title,
                "aspect": asp,
                "value": val,
                "num_raw_models": num_raw,
                "raw_models": [m for m, sup in raw_sup.items() if sup],
                "num_thresh_models": num_th,
            })
            
    return {
        "missing_by_aspect": dict(missing_by_aspect.most_common()),
        "missing_by_category": dict(missing_by_cat),
        "raw_model_support_distribution": dict(raw_support_dist),
        "thresholded_model_support_distribution": dict(thresh_support_dist),
        "total_missing_entities": len(missing_items),
        "missing_with_zero_model_support": raw_support_dist[0],
        "missing_with_at_least_one_model_support": len(complementary_recoverable),
        "sample_complementary_recoverable": complementary_recoverable[:10],
    }


def analyze_false_positives_in_depth(
    clean_examples: List[dict],
    ens_spanned: List[List[Dict[str, Any]]],
    indiv_raw: Dict[str, List[List[Dict[str, Any]]]],
    spurious_items: List[Dict[str, Any]],
    boundary_items: List[Dict[str, Any]],
    aspect_mismatch_items: List[Dict[str, Any]],
    combined_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Analyzes all 179 false positives:
    - Breakdown across boundary (109), aspect mismatch (35), combined (4), and spurious (31).
    - High-confidence false positives (conf >= 0.90).
    - Unanimously supported false positives.
    """
    all_fps = []
    for b in boundary_items:
        all_fps.append({"type": "boundary_mismatch", "aspect": b["pred_aspect"], "conf": b["conf"], "rec_id": b["rec_id"], "span": b["pred_span"], "val": b["pred_val"]})
    for a in aspect_mismatch_items:
        all_fps.append({"type": "aspect_mismatch", "aspect": a["pred_aspect"], "conf": a["conf"], "rec_id": a["rec_id"], "span": a["pred_span"], "val": a["pred_val"]})
    for c in combined_items:
        all_fps.append({"type": "combined_boundary_aspect", "aspect": c["pred_aspect"], "conf": c["conf"], "rec_id": c["rec_id"], "span": c["pred_span"], "val": c["pred_val"]})
    for s in spurious_items:
        all_fps.append({"type": "spurious_entity", "aspect": s["aspect"], "conf": s["conf"], "rec_id": s["rec_id"], "span": s["span"], "val": s["value"]})
        
    fp_by_aspect = Counter(fp["aspect"] for fp in all_fps)
    fp_by_type = Counter(fp["type"] for fp in all_fps)
    
    confs = [fp["conf"] for fp in all_fps]
    high_conf_90 = sum(1 for c in confs if c >= 0.90)
    high_conf_95 = sum(1 for c in confs if c >= 0.95)
    
    # Check unanimous raw model support
    unanimous_count = 0
    for fp in all_fps:
        rec_id = fp["rec_id"]
        asp = fp["aspect"]
        s, e = fp["span"]
        ex_idx = next(i for i, ex in enumerate(clean_examples) if ex["record_number"] == rec_id)
        if all(any(me["aspect"] == asp and me["start"] == s and me["end"] == e for me in indiv_raw[m][ex_idx]) for m in MODEL_NAMES):
            unanimous_count += 1
            
    return {
        "total_fps": len(all_fps),
        "fp_by_type": dict(fp_by_type),
        "fp_by_aspect": dict(fp_by_aspect.most_common()),
        "confidence_stats": {
            "mean": float(np.mean(confs)),
            "median": float(np.median(confs)),
            "ge_90_count": high_conf_90,
            "ge_90_pct": float(high_conf_90 / len(all_fps) * 100.0),
            "ge_95_count": high_conf_95,
            "ge_95_pct": float(high_conf_95 / len(all_fps) * 100.0),
        },
        "unanimously_supported_fps": unanimous_count,
        "unanimously_supported_pct": float(unanimous_count / len(all_fps) * 100.0),
    }


def rank_score_loss_opportunities(
    base_official: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Ranks score-loss opportunities per aspect within each category.
    Includes transparent counterfactuals:
    - Max score loss: 0.5 * weight * (1.0 - F0.2)
    - Counterfactual +25% FP reduction
    - Counterfactual +25% FN recovery
    """
    opportunities = []
    
    for cat in [1, 2]:
        cat_res = base_official["categories"][cat]
        for asp, a_m in cat_res["aspects"].items():
            weight = a_m["weight"]
            f02 = a_m["f_beta"]
            tp = a_m["tp"]
            fp = a_m["fp"]
            fn = a_m["fn"]
            gold_cnt = a_m["gold_count"]
            pred_cnt = a_m["pred_count"]
            
            # Category contributes 50% to final official score
            max_gain_final = float(0.5 * weight * (1.0 - f02))
            
            # Counterfactual A: 25% FP reduction
            fp_25 = int(round(fp * 0.75))
            p_cf_fp = tp / (tp + fp_25) if (tp + fp_25) > 0 else 0.0
            r_cf_fp = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f02_cf_fp = compute_fbeta(p_cf_fp, r_cf_fp)
            gain_fp25 = float(0.5 * weight * (f02_cf_fp - f02))
            
            # Counterfactual B: 25% FN recovery
            fn_rec = int(round(fn * 0.25))
            tp_cf_fn = tp + fn_rec
            fn_cf_fn = fn - fn_rec
            p_cf_fn = tp_cf_fn / (tp_cf_fn + fp) if (tp_cf_fn + fp) > 0 else 0.0
            r_cf_fn = tp_cf_fn / (tp_cf_fn + fn_cf_fn) if (tp_cf_fn + fn_cf_fn) > 0 else 0.0
            f02_cf_fn = compute_fbeta(p_cf_fn, r_cf_fn)
            gain_fn25 = float(0.5 * weight * (f02_cf_fn - f02))
            
            # Counterfactual C: 50% Boundary correction (FP and FN both convert to TP)
            # We estimate boundary errors for this aspect
            # If 50% of boundary errors are fixed, tp += b_half, fp -= b_half, fn -= b_half
            opportunities.append({
                "category": cat,
                "aspect": asp,
                "gold_count": gold_cnt,
                "pred_count": pred_cnt,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": a_m["precision"],
                "recall": a_m["recall"],
                "f_beta": f02,
                "weight": weight,
                "weighted_f_beta": a_m["weighted_f_beta"],
                "max_score_loss_final": max_gain_final,
                "gain_from_25pct_fp_reduction": gain_fp25,
                "gain_from_25pct_fn_recovery": gain_fn25,
            })
            
    opportunities.sort(key=lambda x: x["max_score_loss_final"], reverse=True)
    return opportunities


def run_phase_39a_audit() -> Dict[str, Any]:
    print("=" * 80)
    print("PHASE 39A: OFFICIAL METRIC VERIFICATION & ERROR OPPORTUNITY AUDIT")
    print("=" * 80)
    start_time = time.time()
    
    # 1. Load Data and Reconstruct Baseline
    print("\n[Step 1] Loading clean holdout dataset and cached probabilities...")
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
    cfg = AutoConfig.from_pretrained(PROJECT_ROOT / "checkpoints" / "gbert_baseline_best")
    id2label = {int(k): v for k, v in cfg.id2label.items()}
    
    clean_p_fused = [
        sum(w * probs[m][i] for w, m in zip(MODEL_WEIGHTS, MODEL_NAMES))
        for i in clean_val_indices
    ]
    
    ens_spanned = decode_spanned_predictions(
        clean_p_fused,
        clean_examples,
        id2label,
        conf_thresh=DEFAULT_THRESHOLD,
        aspect_thresholds=ASPECT_THRESHOLDS,
    )
    spanned_gold = extract_spanned_gold_entities(clean_examples)
    gold_tuples = [[(e["aspect"], e["value"]) for e in ex] for ex in spanned_gold]
    ens_tuples = [[(e["aspect"], e["value"]) for e in ex] for ex in ens_spanned]
    
    records = [
        {
            "category": clean_examples[i]["category"],
            "predicted_entities": ens_tuples[i],
            "gold_entities": gold_tuples[i],
        }
        for i in range(len(clean_examples))
    ]
    
    # 2. Metric Calculations: Official-Style vs Global Micro
    print("\n[Step 2] Calculating Official Hierarchical F0.2 vs Global Micro F0.2...")
    official_res = calculate_official_competition_metrics(records)
    micro_res = calculate_global_micro_metrics(records)
    
    print("\n--- METRIC COMPARISON ---")
    print(f"Reconstructed Official-Style F0.2: {official_res['official_score']:.6f}")
    print(f"  Category 1 Score:                {official_res['categories'][1]['score']:.6f}")
    print(f"  Category 2 Score:                {official_res['categories'][2]['score']:.6f}")
    print(f"Global Micro F0.2:                 {micro_res['global_f_beta']:.6f}")
    print(f"  Global Micro Precision:          {micro_res['global_precision']:.6f}")
    print(f"  Global Micro Recall:             {micro_res['global_recall']:.6f}")
    print(f"  Total TP: {micro_res['tp']}, FP: {micro_res['fp']}, FN: {micro_res['fn']}")
    print(f"  Difference (Micro - Official):   {micro_res['global_f_beta'] - official_res['official_score']:+.6f}")
    
    # Verify invariants
    assert micro_res["tp"] == 3045, f"Expected 3045 TP, got {micro_res['tp']}"
    assert micro_res["fp"] == 179, f"Expected 179 FP, got {micro_res['fp']}"
    assert micro_res["fn"] == 728, f"Expected 728 FN, got {micro_res['fn']}"
    assert micro_res["tp"] + micro_res["fn"] == 3773, f"Ground-truth invariant violated: {micro_res['tp'] + micro_res['fn']}"
    
    # 3. Individual Model Predictions for Error Source Attribution
    print("\n[Step 3] Loading individual model predictions for error attribution...")
    indiv_raw = {
        m: decode_spanned_predictions([probs[m][i] for i in clean_val_indices], clean_examples, id2label, conf_thresh=0.0)
        for m in MODEL_NAMES
    }
    indiv_thresh = {
        m: decode_spanned_predictions(
            [probs[m][i] for i in clean_val_indices],
            clean_examples,
            id2label,
            conf_thresh=DEFAULT_THRESHOLD,
            aspect_thresholds=ASPECT_THRESHOLDS,
        )
        for m in MODEL_NAMES
    }
    
    # 4. Error Taxonomy Construction
    print("\n[Step 4] Constructing 100% Reconciled Error Taxonomy...")
    taxonomy = build_error_taxonomy(clean_examples, spanned_gold, ens_spanned)
    print("Taxonomy Summary:")
    for k, v in taxonomy["taxonomy_summary"].items():
        print(f"  {k:<35}: {v}")
    print(f"Total FP accounted for: {taxonomy['total_false_positives']} (Matches 179: {taxonomy['total_false_positives'] == 179})")
    print(f"Total FN accounted for: {taxonomy['total_false_negatives']} (Matches 728: {taxonomy['total_false_negatives'] == 728})")
    print(f"Exact True Positives:   {taxonomy['exact_true_positives']} (Matches 3045: {taxonomy['exact_true_positives'] == 3045})")
    
    # 5. In-Depth Boundary Analysis
    print("\n[Step 5] Analyzing Boundary Errors...")
    b_subtypes = Counter(b["subtype"] for b in taxonomy["boundary_items"])
    b_by_aspect = Counter(b["pred_aspect"] for b in taxonomy["boundary_items"])
    print(f"  Boundary Subtypes: {dict(b_subtypes)}")
    print(f"  Top Boundary Aspects: {b_by_aspect.most_common(6)}")
    
    # 6. In-Depth False Negative Analysis
    print("\n[Step 6] Analyzing False Negatives (728 FNs)...")
    fn_analysis = analyze_false_negatives_in_depth(
        clean_examples, spanned_gold, ens_spanned, indiv_raw, indiv_thresh, taxonomy["missing_items"]
    )
    print(f"  Missing Entities (Zero Overlap):       {fn_analysis['total_missing_entities']}")
    print(f"  Boundary-Related FNs:                  {len(taxonomy['boundary_items'])}")
    print(f"  Aspect Mismatch FNs:                   {len(taxonomy['aspect_mismatch_items'])}")
    print(f"  Combined Boundary/Aspect FNs:          {len(taxonomy['combined_items'])}")
    print(f"  Missing with Zero Individual Models:   {fn_analysis['missing_with_zero_model_support']}")
    print(f"  Missing with >=1 Model Raw Support:    {fn_analysis['missing_with_at_least_one_model_support']}")
    print(f"  Top Missing Aspects:                   {list(fn_analysis['missing_by_aspect'].items())[:6]}")
    
    # 7. In-Depth False Positive Analysis
    print("\n[Step 7] Analyzing False Positives (179 FPs)...")
    fp_analysis = analyze_false_positives_in_depth(
        clean_examples,
        ens_spanned,
        indiv_raw,
        taxonomy["spurious_items"],
        taxonomy["boundary_items"],
        taxonomy["aspect_mismatch_items"],
        taxonomy["combined_items"],
    )
    print(f"  FP by Type:                            {fp_analysis['fp_by_type']}")
    print(f"  FPs with Confidence >= 0.90:           {fp_analysis['confidence_stats']['ge_90_count']} ({fp_analysis['confidence_stats']['ge_90_pct']:.1f}%)")
    print(f"  Unanimously Supported FPs (4 Models):  {fp_analysis['unanimously_supported_fps']} ({fp_analysis['unanimously_supported_pct']:.1f}%)")
    print(f"  Top FP Aspects:                        {list(fp_analysis['fp_by_aspect'].items())[:6]}")
    
    # 8. Rank Score-Loss Opportunities (Task 5)
    print("\n[Step 8] Ranking Highest-Impact Score-Loss Opportunities...")
    opportunities = rank_score_loss_opportunities(official_res)
    print(f"{'Rank':<5} {'Cat':<4} {'Aspect':<32} {'Gold':<6} {'Pred':<6} {'TP':<5} {'FP':<5} {'FN':<5} {'Prec':<7} {'Rec':<7} {'F0.2':<8} {'Weight':<7} {'Max Gain':<9} {'+25%FP':<8} {'+25%FN':<8}")
    for idx, o in enumerate(opportunities[:12], start=1):
        print(f"{idx:<5} Cat{o['category']} {o['aspect']:<32} {o['gold_count']:<6} {o['pred_count']:<6} {o['tp']:<5} {o['fp']:<5} {o['fn']:<5} {o['precision']:<7.4f} {o['recall']:<7.4f} {o['f_beta']:<8.4f} {o['weight']:<7.4f} {o['max_score_loss_final']:<9.5f} {o['gain_from_25pct_fp_reduction']:<8.5f} {o['gain_from_25pct_fn_recovery']:<8.5f}")
        
    # Compile complete audit data
    audit_results = {
        "phase": "39A",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_sec": time.time() - start_time,
        "official_score_results": official_res,
        "global_micro_results": micro_res,
        "error_taxonomy": taxonomy,
        "boundary_analysis": {
            "subtypes": dict(b_subtypes),
            "by_aspect": dict(b_by_aspect),
            "sample_items": taxonomy["boundary_items"][:10],
        },
        "false_negative_analysis": fn_analysis,
        "false_positive_analysis": fp_analysis,
        "ranked_score_loss_opportunities": opportunities,
    }
    
    OUTPUT_REPORT_DATA.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_DATA, "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(audit_results), f, indent=2)
    print(f"\nSaved Phase 39A complete error audit results to {OUTPUT_REPORT_DATA}")
    
    return audit_results


if __name__ == "__main__":
    run_phase_39a_audit()
