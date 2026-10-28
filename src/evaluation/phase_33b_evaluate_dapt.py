"""
Phase 33B: Comprehensive Evaluation and Error Analysis for DAPT Challenger

Evaluates:
1. Standalone Matched Baseline GBERT (1.00x) vs DAPT GBERT Challenger on 800-title internal dev split.
2. Exact-match Vehicle-Model precision, recall, F0.2, and boundary error breakdown.
3. 4-Model Ensemble comparison (Baseline GBERT 0.40 + 3x XLM-R 0.20 vs DAPT GBERT 0.40 + 3x XLM-R 0.20) with MIN 0.60.
4. Paired title-level bootstrap confidence interval (1,000 resamples) for Delta F0.2.
5. Head-to-head title-level and entity-level recovery vs regression audit.
6. Optional exploratory check on 1,000-title Split-A validation holdout.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_examples import build_examples, load_training_data
from src.data.make_splits import split_examples
from src.data.make_internal_split import get_internal_split
from src.evaluation.entity_metrics import (
    bio_to_entities,
    count_matches,
    filter_entities_by_category,
    precision_recall_fbeta,
    BETA,
)
from src.evaluation.competition_metrics import calculate_competition_metrics
from src.evaluation.experiment_26a_confidence_calibration import collect_active_probabilities
from src.evaluation.phase_30a_error_profile import (
    bio_to_entities_with_spans_and_conf,
    bio_to_true_spans,
)
from src.evaluation.phase_32c_evaluate_internal import (
    audit_vehicle_model_details,
    compare_models_head_to_head,
)

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

BASELINE_GBERT_PATH = CHECKPOINTS_DIR / "gbert_phase32c_baseline_best"
DAPT_GBERT_PATH = CHECKPOINTS_DIR / "phase33b_dapt_gbert_best"

FROZEN_XLMR_PATHS = [
    CHECKPOINTS_DIR / "xlmr_baseline_best",
    CHECKPOINTS_DIR / "xlmr_seed_7_best",
    CHECKPOINTS_DIR / "xlmr_seed_21_best",
]

RESULTS_JSON = CACHE_DIR / "phase33b_evaluation_results.json"


def evaluate_model_on_split(
    checkpoint_path: Path,
    examples: List[dict],
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    device: torch.device,
    conf_thresh: float = 0.0,
) -> Tuple[dict, List[dict], List[np.ndarray]]:
    """Run model inference and compute competition metrics, spans, and raw probabilities."""
    probs_list = collect_active_probabilities(checkpoint_path, examples, label2id, device, batch_size=8)

    records = []
    title_eval_data = []

    for ex_idx, ex in enumerate(examples):
        tokens = ex["tokens"]
        cat_id = ex["category"]
        true_bio = ex["bio_tags"]
        p = probs_list[ex_idx]

        pred_ids = np.argmax(p, axis=-1)
        pred_bio = [id2label[pid] for pid in pred_ids]
        selected_probs = [float(p[t_idx, pred_ids[t_idx]]) for t_idx in range(len(tokens))]

        true_spans = bio_to_true_spans(tokens, true_bio)
        cat_valid_true = [s for s in true_spans if filter_entities_by_category([(s[0], s[1])], cat_id)]

        pred_spans_conf = bio_to_entities_with_spans_and_conf(tokens, pred_bio, selected_probs)
        cat_valid_pred = [s for s in pred_spans_conf if filter_entities_by_category([(s[0], s[1])], cat_id)]

        if conf_thresh > 0.0:
            filtered_preds = [s for s in cat_valid_pred if s[5] >= conf_thresh]
        else:
            filtered_preds = cat_valid_pred

        pred_entities = [(s[0], s[1]) for s in filtered_preds]
        true_entities = [(s[0], s[1]) for s in cat_valid_true]

        records.append({
            "category": cat_id,
            "true_entities": true_entities,
            "predicted_entities": pred_entities,
        })

        title_eval_data.append({
            "tokens": tokens,
            "category": cat_id,
            "true_spans": cat_valid_true,
            "pred_spans": filtered_preds,
            "pred_bio": pred_bio,
            "probs": p,
        })

    comp_metrics = calculate_competition_metrics(records)

    tot_tp, tot_fp, tot_fn = 0, 0, 0
    for r in records:
        tp, fp, fn = count_matches(r["predicted_entities"], r["true_entities"])
        tot_tp += tp
        tot_fp += fp
        tot_fn += fn

    pr = precision_recall_fbeta(tot_tp, tot_fp, tot_fn, beta=BETA)

    summary = {
        "final_f02": comp_metrics["final_score"],
        "cat1_f02": comp_metrics["categories"][1]["score"],
        "cat2_f02": comp_metrics["categories"][2]["score"],
        "precision": pr["precision"],
        "recall": pr["recall"],
        "tp": tot_tp,
        "fp": tot_fp,
        "fn": tot_fn,
        "total_preds": tot_tp + tot_fp,
    }

    return summary, title_eval_data, probs_list


def evaluate_ensemble_probabilities(
    gbert_probs: List[np.ndarray],
    xlmr_probs_list: List[List[np.ndarray]],
    examples: List[dict],
    id2label: Dict[int, str],
    min_confidence: float = 0.60,
) -> Tuple[dict, List[dict]]:
    """Fuse probabilities with champion weights (0.40 GBERT + 0.60 3-seed XLM-R) and evaluate."""
    records = []
    title_eval_data = []
    target_aspect = "Kompatibles_Fahrzeug_Modell"
    vm_tp, vm_fp, vm_fn = 0, 0, 0

    num_xlmr = len(xlmr_probs_list)
    assert num_xlmr == 3, f"Expected 3 XLM-R probability sets, got {num_xlmr}"

    for ex_idx, ex in enumerate(examples):
        tokens = ex["tokens"]
        cat_id = ex["category"]
        true_bio = ex["bio_tags"]

        p_gb = gbert_probs[ex_idx]
        p_xl = sum(xlmr_probs_list[m_idx][ex_idx] for m_idx in range(num_xlmr)) / num_xlmr

        p_fused = 0.40 * p_gb + 0.60 * p_xl
        pred_ids = np.argmax(p_fused, axis=-1)
        pred_bio = [id2label[pid] for pid in pred_ids]
        selected_probs = [float(p_fused[t_idx, pred_ids[t_idx]]) for t_idx in range(len(tokens))]

        true_spans = bio_to_true_spans(tokens, true_bio)
        cat_valid_true = [s for s in true_spans if filter_entities_by_category([(s[0], s[1])], cat_id)]

        pred_spans_conf = bio_to_entities_with_spans_and_conf(tokens, pred_bio, selected_probs)
        cat_valid_pred = [s for s in pred_spans_conf if filter_entities_by_category([(s[0], s[1])], cat_id)]

        if min_confidence > 0.0:
            filtered_preds = [s for s in cat_valid_pred if s[5] >= min_confidence]
        else:
            filtered_preds = cat_valid_pred

        pred_entities = [(s[0], s[1]) for s in filtered_preds]
        true_entities = [(s[0], s[1]) for s in cat_valid_true]

        records.append({
            "category": cat_id,
            "true_entities": true_entities,
            "predicted_entities": pred_entities,
        })

        true_vm = [s for s in cat_valid_true if s[0] == target_aspect]
        pred_vm = [s for s in filtered_preds if s[0] == target_aspect]
        cur_tp, cur_fp, cur_fn = count_matches(
            [(s[0], s[1]) for s in pred_vm],
            [(s[0], s[1]) for s in true_vm],
        )
        vm_tp += cur_tp
        vm_fp += cur_fp
        vm_fn += cur_fn

        title_eval_data.append({
            "tokens": tokens,
            "category": cat_id,
            "true_spans": cat_valid_true,
            "pred_spans": filtered_preds,
            "pred_bio": pred_bio,
            "probs": p_fused,
        })

    comp_metrics = calculate_competition_metrics(records)

    tot_tp, tot_fp, tot_fn = 0, 0, 0
    for r in records:
        tp, fp, fn = count_matches(r["predicted_entities"], r["true_entities"])
        tot_tp += tp
        tot_fp += fp
        tot_fn += fn

    pr = precision_recall_fbeta(tot_tp, tot_fp, tot_fn, beta=BETA)
    vm_pr = precision_recall_fbeta(vm_tp, vm_fp, vm_fn, beta=BETA)

    summary = {
        "final_f02": comp_metrics["final_score"],
        "cat1_f02": comp_metrics["categories"][1]["score"],
        "cat2_f02": comp_metrics["categories"][2]["score"],
        "precision": pr["precision"],
        "recall": pr["recall"],
        "tp": tot_tp,
        "fp": tot_fp,
        "fn": tot_fn,
        "total_preds": tot_tp + tot_fp,
        "vehicle_model": {
            "tp": vm_tp,
            "fp": vm_fp,
            "fn": vm_fn,
            "precision": vm_pr["precision"],
            "recall": vm_pr["recall"],
            "f02": vm_pr["f_beta"],
        },
    }

    return summary, title_eval_data


def compute_bootstrap_ci(
    titles_base: List[dict],
    titles_challenger: List[dict],
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict:
    """Compute paired title-level bootstrap confidence interval for Delta F0.2."""
    rng = random.Random(seed)
    n_titles = len(titles_base)
    assert n_titles == len(titles_challenger)

    base_records = [
        {
            "category": t["category"],
            "true_entities": [(s[0], s[1]) for s in t["true_spans"]],
            "predicted_entities": [(s[0], s[1]) for s in t["pred_spans"]],
        }
        for t in titles_base
    ]
    chal_records = [
        {
            "category": t["category"],
            "true_entities": [(s[0], s[1]) for s in t["true_spans"]],
            "predicted_entities": [(s[0], s[1]) for s in t["pred_spans"]],
        }
        for t in titles_challenger
    ]

    deltas = []

    for _ in range(n_resamples):
        sample_indices = [rng.randint(0, n_titles - 1) for _ in range(n_titles)]
        b_records = [base_records[i] for i in sample_indices]
        c_records = [chal_records[i] for i in sample_indices]

        f_b = calculate_competition_metrics(b_records)["final_score"]
        f_c = calculate_competition_metrics(c_records)["final_score"]
        deltas.append(f_c - f_b)

    deltas.sort()
    ci_lower = deltas[int(0.025 * n_resamples)]
    ci_upper = deltas[int(0.975 * n_resamples)]
    mean_delta = sum(deltas) / len(deltas)

    return {
        "mean_delta_f02": round(mean_delta, 6),
        "ci_lower_95": round(ci_lower, 6),
        "ci_upper_95": round(ci_upper, 6),
        "p_value_greater_than_zero": round(sum(d > 0 for d in deltas) / n_resamples, 4),
    }


def run_full_evaluation():
    print("=" * 80)
    print("PHASE 33B: STAGES D & E — DEEP INTERNAL EVALUATION & ERROR BREAKDOWN")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Device: {device}")

    # 1. Load Data Splits
    df = load_training_data()
    all_examples = build_examples(df)
    train_4k, val_1k = split_examples(all_examples, random_state=42)
    train_3200, dev_800, manifest = get_internal_split(train_4k, random_state=42)

    print(f"Internal Dev Dataset: {len(dev_800)} titles")
    print(f"Split-A Val Dataset:  {len(val_1k)} titles")

    ref_cfg = AutoConfig.from_pretrained("checkpoints/gbert_baseline_best")
    label2id = {str(k): int(v) for k, v in ref_cfg.label2id.items()}
    id2label = {int(k): str(v) for k, v in ref_cfg.id2label.items()}

    # 2. Standalone Evaluations
    print("\n[Stage D.1] Evaluating Standalone Matched Baseline GBERT on 800-title Dev...")
    base_summary, base_titles, base_probs = evaluate_model_on_split(
        checkpoint_path=BASELINE_GBERT_PATH,
        examples=dev_800,
        label2id=label2id,
        id2label=id2label,
        device=device,
        conf_thresh=0.0,
    )
    base_vm_audit = audit_vehicle_model_details(base_titles)

    print("\n[Stage D.2] Evaluating Standalone DAPT GBERT Challenger on 800-title Dev...")
    chal_summary, chal_titles, chal_probs = evaluate_model_on_split(
        checkpoint_path=DAPT_GBERT_PATH,
        examples=dev_800,
        label2id=label2id,
        id2label=id2label,
        device=device,
        conf_thresh=0.0,
    )
    chal_vm_audit = audit_vehicle_model_details(chal_titles)

    head_to_head = compare_models_head_to_head(base_titles, chal_titles)

    # Standalone Bootstrap CI
    print("\nComputing Standalone Paired Bootstrap CI (1,000 resamples)...")
    standalone_ci = compute_bootstrap_ci(base_titles, chal_titles, n_resamples=1000, seed=42)

    # 3. Stage E: Internal Ensemble Evaluation
    print("\n[Stage E] Collecting Probabilities from 3 Frozen XLM-R Models on 800-title Dev...")
    xlmr_probs_list = []
    for xlmr_path in FROZEN_XLMR_PATHS:
        print(f"  Collecting from {xlmr_path.name}...")
        p_xl = collect_active_probabilities(xlmr_path, dev_800, label2id, device, batch_size=8)
        xlmr_probs_list.append(p_xl)

    print("\nEvaluating Baseline 4-Model Ensemble (GBERT Baseline 0.40 + 3x XLM-R 0.20, MIN 0.60)...")
    ens_base_summary, ens_base_titles = evaluate_ensemble_probabilities(
        gbert_probs=base_probs,
        xlmr_probs_list=xlmr_probs_list,
        examples=dev_800,
        id2label=id2label,
        min_confidence=0.60,
    )
    ens_base_vm_audit = audit_vehicle_model_details(ens_base_titles)

    print("Evaluating Challenger 4-Model Ensemble (GBERT DAPT 0.40 + 3x XLM-R 0.20, MIN 0.60)...")
    ens_chal_summary, ens_chal_titles = evaluate_ensemble_probabilities(
        gbert_probs=chal_probs,
        xlmr_probs_list=xlmr_probs_list,
        examples=dev_800,
        id2label=id2label,
        min_confidence=0.60,
    )
    ens_chal_vm_audit = audit_vehicle_model_details(ens_chal_titles)

    ens_head_to_head = compare_models_head_to_head(ens_base_titles, ens_chal_titles)

    print("\nComputing Ensemble Paired Bootstrap CI (1,000 resamples)...")
    ensemble_ci = compute_bootstrap_ci(ens_base_titles, ens_chal_titles, n_resamples=1000, seed=42)

    # 4. Optional Exploratory Split-A Holdout Check
    print("\n[Stage F] Optional Exploratory Check on 1,000-title Split-A Holdout...")
    split_a_base_summary, split_a_base_titles, split_a_base_probs = evaluate_model_on_split(
        checkpoint_path=BASELINE_GBERT_PATH,
        examples=val_1k,
        label2id=label2id,
        id2label=id2label,
        device=device,
        conf_thresh=0.0,
    )
    split_a_chal_summary, split_a_chal_titles, split_a_chal_probs = evaluate_model_on_split(
        checkpoint_path=DAPT_GBERT_PATH,
        examples=val_1k,
        label2id=label2id,
        id2label=id2label,
        device=device,
        conf_thresh=0.0,
    )

    # Split-A Ensemble
    split_a_xlmr = [collect_active_probabilities(p, val_1k, label2id, device, batch_size=8) for p in FROZEN_XLMR_PATHS]
    split_a_ens_base, _ = evaluate_ensemble_probabilities(split_a_base_probs, split_a_xlmr, val_1k, id2label, min_confidence=0.60)
    split_a_ens_chal, _ = evaluate_ensemble_probabilities(split_a_chal_probs, split_a_xlmr, val_1k, id2label, min_confidence=0.60)

    # Summary Compilation
    eval_results = {
        "standalone_internal_dev": {
            "baseline": base_summary,
            "baseline_vm_audit": base_vm_audit,
            "challenger": chal_summary,
            "challenger_vm_audit": chal_vm_audit,
            "delta_f02": round(chal_summary["final_f02"] - base_summary["final_f02"], 6),
            "delta_precision": round(chal_summary["precision"] - base_summary["precision"], 6),
            "delta_recall": round(chal_summary["recall"] - base_summary["recall"], 6),
            "delta_tp": chal_summary["tp"] - base_summary["tp"],
            "delta_fp": chal_summary["fp"] - base_summary["fp"],
            "delta_fn": chal_summary["fn"] - base_summary["fn"],
            "head_to_head": head_to_head,
            "bootstrap_ci_95": standalone_ci,
        },
        "ensemble_internal_dev_min060": {
            "baseline_ensemble": ens_base_summary,
            "baseline_vm_audit": ens_base_vm_audit,
            "challenger_ensemble": ens_chal_summary,
            "challenger_vm_audit": ens_chal_vm_audit,
            "delta_f02": round(ens_chal_summary["final_f02"] - ens_base_summary["final_f02"], 6),
            "delta_precision": round(ens_chal_summary["precision"] - ens_base_summary["precision"], 6),
            "delta_recall": round(ens_chal_summary["recall"] - ens_base_summary["recall"], 6),
            "delta_tp": ens_chal_summary["tp"] - ens_base_summary["tp"],
            "delta_fp": ens_chal_summary["fp"] - ens_base_summary["fp"],
            "delta_fn": ens_chal_summary["fn"] - ens_base_summary["fn"],
            "head_to_head": ens_head_to_head,
            "bootstrap_ci_95": ensemble_ci,
        },
        "exploratory_split_a_holdout": {
            "standalone_baseline_f02": split_a_base_summary["final_f02"],
            "standalone_challenger_f02": split_a_chal_summary["final_f02"],
            "standalone_delta_f02": round(split_a_chal_summary["final_f02"] - split_a_base_summary["final_f02"], 6),
            "ensemble_baseline_f02": split_a_ens_base["final_f02"],
            "ensemble_challenger_f02": split_a_ens_chal["final_f02"],
            "ensemble_delta_f02": round(split_a_ens_chal["final_f02"] - split_a_ens_base["final_f02"], 6),
        },
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    print("\n" + "=" * 80)
    print("STAGE D & E EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Standalone Internal Dev F0.2:  Baseline = {base_summary['final_f02']:.6f} | Challenger = {chal_summary['final_f02']:.6f} | Delta = {chal_summary['final_f02'] - base_summary['final_f02']:+.6f}")
    print(f"Ensemble Internal Dev F0.2:    Baseline = {ens_base_summary['final_f02']:.6f} | Challenger = {ens_chal_summary['final_f02']:.6f} | Delta = {ens_chal_summary['final_f02'] - ens_base_summary['final_f02']:+.6f}")
    print(f"Exploratory Split-A Ens F0.2:  Baseline = {split_a_ens_base['final_f02']:.6f} | Challenger = {split_a_ens_chal['final_f02']:.6f} | Delta = {split_a_ens_chal['final_f02'] - split_a_ens_base['final_f02']:+.6f}")
    print(f"Results written to: {RESULTS_JSON}")
    print("=" * 80)

    return eval_results


if __name__ == "__main__":
    run_full_evaluation()
