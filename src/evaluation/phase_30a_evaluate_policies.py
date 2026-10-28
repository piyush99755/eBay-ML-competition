"""
Phase 30A: Evaluation of Predeclared Challenger Policies vs Frozen Champion.

Baseline:
- Frozen Exp 10 Champion with global MIN 0.60 confidence filter.

Policy A:
- Global MIN 0.60, with conservative MIN 0.70 adjustment strictly for
  high-support, low-precision aspect: Kompatibles_Fahrzeug_Modell.

Policy B:
- Category-aware adjustment: Category 1 at MIN 0.65, Category 2 at MIN 0.60.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

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
from src.evaluation.phase_30a_error_profile import (
    bio_to_entities_with_spans_and_conf,
    bio_to_true_spans,
)

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
BOOTSTRAP_ROUNDS = 1000
BOOTSTRAP_SEED = 42

def evaluate_policy_records(records: List[dict]) -> dict:
    comp_res = calculate_competition_metrics(records)
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for r in records:
        tp, fp, fn = count_matches(r["predicted_entities"], r["true_entities"])
        total_tp += tp
        total_fp += fp
        total_fn += fn

    pr = precision_recall_fbeta(total_tp, total_fp, total_fn, beta=0.2)
    return {
        "final_score": comp_res["final_score"],
        "cat1_score": comp_res["categories"].get(1, {}).get("score", 0.0),
        "cat2_score": comp_res["categories"].get(2, {}).get("score", 0.0),
        "precision": pr["precision"],
        "recall": pr["recall"],
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "total_preds": sum(len(r["predicted_entities"]) for r in records),
        "comp_results": comp_res,
    }


def run_paired_title_bootstrap(
    base_records: List[dict],
    cand_records: List[dict],
    num_rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    np.random.seed(seed)
    n = len(base_records)
    assert len(cand_records) == n

    deltas = []
    base_scores = []
    cand_scores = []

    for _ in range(num_rounds):
        idx = np.random.randint(0, n, size=n)
        b_sample = [base_records[i] for i in idx]
        c_sample = [cand_records[i] for i in idx]

        b_res = calculate_competition_metrics(b_sample)
        c_res = calculate_competition_metrics(c_sample)

        b_score = b_res["final_score"]
        c_score = c_res["final_score"]

        base_scores.append(b_score)
        cand_scores.append(c_score)
        deltas.append(c_score - b_score)

    deltas = np.array(deltas)
    mean_delta = float(np.mean(deltas))
    std_err = float(np.std(deltas))
    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))
    p_better = float(np.mean(deltas > 0.0))

    return {
        "mean_delta": mean_delta,
        "std_err": std_err,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_better": p_better,
    }


def main():
    print("=" * 75)
    print("PHASE 30A: PREDECLARED CHALLENGER POLICIES EVALUATION")
    print("=" * 75)

    # 1. Load Split A Validation Set
    df = load_training_data()
    all_examples = build_examples(df)
    _, val_examples = split_examples(all_examples, random_state=42)
    print(f"Loaded {len(val_examples)} Split-A validation titles.")

    # 2. Load cached fused probabilities
    fused_cache_file = CACHE_DIR / "split_a_champion_fused_probs.npy"
    if not fused_cache_file.exists():
        print(f"Error: {fused_cache_file} not found. Run phase_30a_error_profile.py first.")
        sys.exit(1)
    p_fused = list(np.load(fused_cache_file, allow_pickle=True))

    # 3. Label Mapping from Reference Champion
    from transformers import AutoConfig
    ref_cfg = AutoConfig.from_pretrained(Path("checkpoints/gbert_baseline_best"))
    id2label = {int(k): v for k, v in ref_cfg.id2label.items()}

    # 4. Extract candidate entities with confidence for all titles
    title_candidates = []
    for ex_idx, ex in enumerate(val_examples):
        tokens = ex["tokens"]
        cat_id = ex["category"]
        true_bio = ex["bio_tags"]

        p = p_fused[ex_idx]
        pred_ids = np.argmax(p, axis=-1)
        pred_bio = [id2label[pid] for pid in pred_ids]
        selected_probs = [float(p[t_idx, pred_ids[t_idx]]) for t_idx in range(len(tokens))]

        true_ents = bio_to_entities(tokens, true_bio)
        cat_valid_true = filter_entities_by_category(true_ents, cat_id)

        pred_spans_conf = bio_to_entities_with_spans_and_conf(tokens, pred_bio, selected_probs)
        cat_valid_pred = []
        for asp, val, s_idx, e_idx, mean_c, min_c in pred_spans_conf:
            if filter_entities_by_category([(asp, val)], cat_id):
                cat_valid_pred.append((asp, val, min_c))

        title_candidates.append({
            "category": cat_id,
            "true_entities": cat_valid_true,
            "pred_candidates": cat_valid_pred,
        })

    # 5. Define Policies
    # BASELINE: MIN 0.60 global
    base_records = []
    for t in title_candidates:
        cat_id = t["category"]
        true_ents = t["true_entities"]
        ents = [(asp, val) for asp, val, min_c in t["pred_candidates"] if min_c >= 0.60]
        base_records.append({
            "category": cat_id,
            "true_entities": true_ents,
            "predicted_entities": ents,
        })

    base_metrics = evaluate_policy_records(base_records)
    print("\n--- BASELINE: FROZEN CHAMPION (MIN 0.60 GLOBAL) ---")
    print(f"Official F0.2: {base_metrics['final_score']:.6f} (Expected: 0.899802)")
    print(f"Category 1:    {base_metrics['cat1_score']:.6f}")
    print(f"Category 2:    {base_metrics['cat2_score']:.6f}")
    print(f"Precision:     {base_metrics['precision']:.4f}")
    print(f"Recall:        {base_metrics['recall']:.4f}")
    print(f"TP / FP / FN:  {base_metrics['tp']} / {base_metrics['fp']} / {base_metrics['fn']}")
    print(f"Total Preds:   {base_metrics['total_preds']}")

    assert abs(base_metrics["final_score"] - 0.899802) < 1e-5, f"Baseline mismatch! {base_metrics['final_score']}"

    # POLICY A: Global MIN 0.60, except Kompatibles_Fahrzeug_Modell at MIN 0.70
    pol_a_records = []
    pol_a_removed = 0
    pol_a_tp_removed = 0
    pol_a_fp_removed = 0
    pol_a_aspect_counts = defaultdict(int)

    for t_idx, t in enumerate(title_candidates):
        cat_id = t["category"]
        true_ents = t["true_entities"]
        ents = []
        for asp, val, min_c in t["pred_candidates"]:
            # Aspect-specific threshold
            thresh = 0.70 if asp == "Kompatibles_Fahrzeug_Modell" else 0.60
            if min_c >= thresh:
                ents.append((asp, val))
                pol_a_aspect_counts[asp] += 1

        pol_a_records.append({
            "category": cat_id,
            "true_entities": true_ents,
            "predicted_entities": ents,
        })

    pol_a_metrics = evaluate_policy_records(pol_a_records)
    pol_a_tp_diff = base_metrics["tp"] - pol_a_metrics["tp"]
    pol_a_fp_diff = base_metrics["fp"] - pol_a_metrics["fp"]
    pol_a_removed = base_metrics["total_preds"] - pol_a_metrics["total_preds"]

    # POLICY B: Category 1 at MIN 0.65, Category 2 at MIN 0.60
    pol_b_records = []
    pol_b_aspect_counts = defaultdict(int)

    for t_idx, t in enumerate(title_candidates):
        cat_id = t["category"]
        true_ents = t["true_entities"]
        ents = []
        # Category-specific threshold
        thresh = 0.65 if cat_id == 1 else 0.60
        for asp, val, min_c in t["pred_candidates"]:
            if min_c >= thresh:
                ents.append((asp, val))
                pol_b_aspect_counts[asp] += 1

        pol_b_records.append({
            "category": cat_id,
            "true_entities": true_ents,
            "predicted_entities": ents,
        })

    pol_b_metrics = evaluate_policy_records(pol_b_records)
    pol_b_tp_diff = base_metrics["tp"] - pol_b_metrics["tp"]
    pol_b_fp_diff = base_metrics["fp"] - pol_b_metrics["fp"]
    pol_b_removed = base_metrics["total_preds"] - pol_b_metrics["total_preds"]

    # 6. Run Paired Bootstrap Resampling
    print("\nRunning paired bootstrap resampling (1,000 rounds)...")
    boot_a = run_paired_title_bootstrap(base_records, pol_a_records)
    boot_b = run_paired_title_bootstrap(base_records, pol_b_records)

    # 7. Print Comparative Results
    print("\n" + "=" * 95)
    print("COMPARATIVE EVALUATION SUMMARY (Exploratory Split-A)")
    print("=" * 95)
    print(f"{'Configuration':<35} {'F0.2':<9} {'Delta':<10} {'Cat 1':<9} {'Cat 2':<9} {'Prec':<7} {'Rec':<7} {'TP':<6} {'FP':<5} {'Pruned (FP/TP)'}")
    print("-" * 95)
    print(f"{'Frozen Champion (MIN 0.60)':<35} {base_metrics['final_score']:<9.6f} {'--':<10} {base_metrics['cat1_score']:<9.6f} {base_metrics['cat2_score']:<9.6f} {base_metrics['precision']:<7.4f} {base_metrics['recall']:<7.4f} {base_metrics['tp']:<6} {base_metrics['fp']:<5} {'0 (0 / 0)'}")
    print(f"{'Policy A (Modell @ 0.70)':<35} {pol_a_metrics['final_score']:<9.6f} {pol_a_metrics['final_score'] - base_metrics['final_score']:<+10.6f} {pol_a_metrics['cat1_score']:<9.6f} {pol_a_metrics['cat2_score']:<9.6f} {pol_a_metrics['precision']:<7.4f} {pol_a_metrics['recall']:<7.4f} {pol_a_metrics['tp']:<6} {pol_a_metrics['fp']:<5} {pol_a_removed} ({pol_a_fp_diff} / {pol_a_tp_diff})")
    print(f"{'Policy B (Cat 1 @ 0.65)':<35} {pol_b_metrics['final_score']:<9.6f} {pol_b_metrics['final_score'] - base_metrics['final_score']:<+10.6f} {pol_b_metrics['cat1_score']:<9.6f} {pol_b_metrics['cat2_score']:<9.6f} {pol_b_metrics['precision']:<7.4f} {pol_b_metrics['recall']:<7.4f} {pol_b_metrics['tp']:<6} {pol_b_metrics['fp']:<5} {pol_b_removed} ({pol_b_fp_diff} / {pol_b_tp_diff})")

    print("\n--- Paired Bootstrap Resampling (1,000 iterations vs Champion):")
    print(f"Policy A vs Champion: Mean Delta = {boot_a['mean_delta']:+.6f} | SE = {boot_a['std_err']:.6f} | 95% CI: [{boot_a['ci_lower']:+.6f}, {boot_a['ci_upper']:+.6f}] | P(>0) = {boot_a['p_better']*100:.1f}%")
    print(f"Policy B vs Champion: Mean Delta = {boot_b['mean_delta']:+.6f} | SE = {boot_b['std_err']:.6f} | 95% CI: [{boot_b['ci_lower']:+.6f}, {boot_b['ci_upper']:+.6f}] | P(>0) = {boot_b['p_better']*100:.1f}%")

    # 8. Title-level distribution of score changes
    # How many titles improved, stayed same, worsened?
    title_diffs_a = []
    title_diffs_b = []
    for b_rec, a_rec, b2_rec in zip(base_records, pol_a_records, pol_b_records):
        # We can score individual titles
        b_sub = calculate_competition_metrics([b_rec])["final_score"]
        a_sub = calculate_competition_metrics([a_rec])["final_score"]
        b2_sub = calculate_competition_metrics([b2_rec])["final_score"]
        title_diffs_a.append(a_sub - b_sub)
        title_diffs_b.append(b2_sub - b_sub)

    pos_a = sum(1 for d in title_diffs_a if d > 1e-6)
    neg_a = sum(1 for d in title_diffs_a if d < -1e-6)
    zero_a = sum(1 for d in title_diffs_a if abs(d) <= 1e-6)

    pos_b = sum(1 for d in title_diffs_b if d > 1e-6)
    neg_b = sum(1 for d in title_diffs_b if d < -1e-6)
    zero_b = sum(1 for d in title_diffs_b if abs(d) <= 1e-6)

    print("\n--- Title-Level Distribution of Impact (out of 1,000 titles):")
    print(f"Policy A: Improved in {pos_a} titles | Degraded in {neg_a} titles | Unchanged in {zero_a} titles")
    print(f"Policy B: Improved in {pos_b} titles | Degraded in {neg_b} titles | Unchanged in {zero_b} titles")

    # Aspect Extinction Audit
    rare_aspects = ["Oe/Oem_Referenznummer(N)", "Modell", "Stärke", "Material", "Produktlinie", "Besonderheiten"]
    print("\n--- Rare Aspect Extinction Audit:")
    for asp in rare_aspects:
        base_cnt = sum(1 for r in base_records for a, _ in r["predicted_entities"] if a == asp)
        pol_a_cnt = pol_a_aspect_counts[asp]
        pol_b_cnt = pol_b_aspect_counts[asp]
        print(f"  {asp:<30}: Base = {base_cnt:<3} | Pol A = {pol_a_cnt:<3} | Pol B = {pol_b_cnt:<3}")

    # Save policy results to cache
    policy_results = {
        "baseline": {
            "f02": base_metrics["final_score"],
            "cat1_f02": base_metrics["cat1_score"],
            "cat2_f02": base_metrics["cat2_score"],
            "precision": base_metrics["precision"],
            "recall": base_metrics["recall"],
            "tp": base_metrics["tp"],
            "fp": base_metrics["fp"],
            "fn": base_metrics["fn"],
            "total_preds": base_metrics["total_preds"],
        },
        "policy_a": {
            "f02": pol_a_metrics["final_score"],
            "delta": pol_a_metrics["final_score"] - base_metrics["final_score"],
            "cat1_f02": pol_a_metrics["cat1_score"],
            "cat2_f02": pol_a_metrics["cat2_score"],
            "precision": pol_a_metrics["precision"],
            "recall": pol_a_metrics["recall"],
            "tp": pol_a_metrics["tp"],
            "fp": pol_a_metrics["fp"],
            "fn": pol_a_metrics["fn"],
            "total_preds": pol_a_metrics["total_preds"],
            "pruned": pol_a_removed,
            "tp_pruned": pol_a_tp_diff,
            "fp_pruned": pol_a_fp_diff,
            "bootstrap": boot_a,
            "title_distribution": {"pos": pos_a, "neg": neg_a, "zero": zero_a},
        },
        "policy_b": {
            "f02": pol_b_metrics["final_score"],
            "delta": pol_b_metrics["final_score"] - base_metrics["final_score"],
            "cat1_f02": pol_b_metrics["cat1_score"],
            "cat2_f02": pol_b_metrics["cat2_score"],
            "precision": pol_b_metrics["precision"],
            "recall": pol_b_metrics["recall"],
            "tp": pol_b_metrics["tp"],
            "fp": pol_b_metrics["fp"],
            "fn": pol_b_metrics["fn"],
            "total_preds": pol_b_metrics["total_preds"],
            "pruned": pol_b_removed,
            "tp_pruned": pol_b_tp_diff,
            "fp_pruned": pol_b_fp_diff,
            "bootstrap": boot_b,
            "title_distribution": {"pos": pos_b, "neg": neg_b, "zero": zero_b},
        },
    }

    out_file = CACHE_DIR / "phase_30a_policy_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(policy_results, f, indent=2)
    print(f"\nPolicy results saved to {out_file}")

if __name__ == "__main__":
    main()
