"""
Phase 32C: Deep Internal Evaluation and Error Breakdown Script.

Evaluates matched Baseline (1.00x) vs Challenger (1.15x) on:
1. Internal 800-title Dev Set (Primary Decision Gate)
2. (Optional Exploratory) 1,000-title Split-A Holdout Set

Detailed Metrics:
- Micro F0.2, Precision, Recall, TP, FP, FN (Overall & Category 1/2)
- Vehicle-Model specific TP, FP, FN, Precision, Recall, F0.2
- Multi-token vs Single-token vehicle-model exact match accuracy
- Boundary error breakdown: Multi-entity overlap vs Single-span boundary (sub-span vs super-span)
- Head-to-head title-level and entity-level recovery vs regression audit
"""

import json
import sys
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

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


def evaluate_model_on_split(
    checkpoint_path: Path,
    examples: List[dict],
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    device: torch.device,
    conf_thresh: float = 0.0,  # 0.0 for raw argmax, or 0.60 for champion threshold
) -> Tuple[dict, List[dict]]:
    """Run model inference and compute competition metrics and detailed spans."""
    probs_list = collect_active_probabilities(checkpoint_path, examples, label2id, device, batch_size=8)

    records = []
    title_eval_data = []
    target_aspect = "Kompatibles_Fahrzeug_Modell"

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

    # Micro counts
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

    return summary, title_eval_data


def audit_vehicle_model_details(title_eval_data: List[dict]) -> dict:
    """Detailed audit of vehicle-model exact-match, boundary errors, and multi-token accuracy."""
    target_aspect = "Kompatibles_Fahrzeug_Modell"

    tp = 0
    fp = 0
    fn = 0

    single_token_true = 0
    single_token_correct = 0
    multi_token_true = 0
    multi_token_correct = 0

    multi_true_overlaps = 0
    single_boundary_errors = 0
    missing_tokens_subspan = 0
    extra_tokens_superspan = 0
    wrong_aspect_errors = 0
    spurious_errors = 0

    for item in title_eval_data:
        tokens = item["tokens"]
        cat_id = item["category"]
        true_spans = [s for s in item["true_spans"] if s[0] == target_aspect]
        pred_spans = [s for s in item["pred_spans"] if s[0] == target_aspect]

        matched_true = set()

        for p_ent in pred_spans:
            p_asp, p_val, p_s, p_e = p_ent[0], p_ent[1], p_ent[2], p_ent[3]
            match = None
            for t_idx, t_ent in enumerate(true_spans):
                if t_idx not in matched_true and t_ent[1] == p_val:
                    match = t_idx
                    matched_true.add(t_idx)
                    break

            if match is not None:
                tp += 1
                t_ent = true_spans[match]
                if t_ent[3] == t_ent[2]:
                    single_token_correct += 1
                else:
                    multi_token_correct += 1
            else:
                fp += 1
                # Categorize FP error type
                overlapping_true = []
                for t_idx, t_ent in enumerate(item["true_spans"]):
                    if max(p_s, t_ent[2]) <= min(p_e, t_ent[3]):
                        overlapping_true.append(t_ent)

                if not overlapping_true:
                    spurious_errors += 1
                else:
                    same_model_overlaps = [o for o in overlapping_true if o[0] == target_aspect]
                    if len(same_model_overlaps) > 1:
                        multi_true_overlaps += 1
                    elif len(same_model_overlaps) == 1:
                        single_boundary_errors += 1
                        t_s, t_e = same_model_overlaps[0][2], same_model_overlaps[0][3]
                        if p_s >= t_s and p_e <= t_e and (p_e - p_s < t_e - t_s):
                            missing_tokens_subspan += 1
                        elif p_s <= t_s and p_e >= t_e and (p_e - p_s > t_e - t_s):
                            extra_tokens_superspan += 1
                    else:
                        wrong_aspect_errors += 1

        for t_idx, t_ent in enumerate(true_spans):
            if t_ent[3] == t_ent[2]:
                single_token_true += 1
            else:
                multi_token_true += 1

            if t_idx not in matched_true:
                fn += 1

    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f02 = (1.04 * p * r) / (0.04 * p + r) if (0.04 * p + r) > 0 else 0.0

    return {
        "model_tp": tp,
        "model_fp": fp,
        "model_fn": fn,
        "model_precision": p,
        "model_recall": r,
        "model_f02": f02,
        "single_token_true": single_token_true,
        "single_token_correct": single_token_correct,
        "single_token_acc": (single_token_correct / max(1, single_token_true)) * 100,
        "multi_token_true": multi_token_true,
        "multi_token_correct": multi_token_correct,
        "multi_token_acc": (multi_token_correct / max(1, multi_token_true)) * 100,
        "multi_true_overlaps": multi_true_overlaps,
        "single_boundary_errors": single_boundary_errors,
        "missing_tokens_subspan": missing_tokens_subspan,
        "extra_tokens_superspan": extra_tokens_superspan,
        "wrong_aspect_errors": wrong_aspect_errors,
        "spurious_errors": spurious_errors,
    }


def compare_models_head_to_head(
    base_eval: List[dict],
    chal_eval: List[dict],
) -> dict:
    """Compare baseline and challenger title-by-title to audit recoveries vs regressions."""
    target_aspect = "Kompatibles_Fahrzeug_Modell"

    title_improved = 0
    title_regressed = 0
    title_unchanged = 0

    base_f02_list = []
    chal_f02_list = []

    model_recovered_spans = 0
    model_regressed_spans = 0

    for b_item, c_item in zip(base_eval, chal_eval):
        cat_id = b_item["category"]
        true_ents = [(s[0], s[1]) for s in b_item["true_spans"]]
        b_preds = [(s[0], s[1]) for s in b_item["pred_spans"]]
        c_preds = [(s[0], s[1]) for s in c_item["pred_spans"]]

        b_sc = calculate_competition_metrics([{"category": cat_id, "true_entities": true_ents, "predicted_entities": b_preds}])["final_score"]
        c_sc = calculate_competition_metrics([{"category": cat_id, "true_entities": true_ents, "predicted_entities": c_preds}])["final_score"]

        base_f02_list.append(b_sc)
        chal_f02_list.append(c_sc)

        if c_sc > b_sc + 1e-6:
            title_improved += 1
        elif c_sc < b_sc - 1e-6:
            title_regressed += 1
        else:
            title_unchanged += 1

        # Check vehicle-model recoveries vs regressions
        b_model_matches = set([e[1] for e in b_preds if e[0] == target_aspect]).intersection([e[1] for e in true_ents if e[0] == target_aspect])
        c_model_matches = set([e[1] for e in c_preds if e[0] == target_aspect]).intersection([e[1] for e in true_ents if e[0] == target_aspect])

        # Recovered: in challenger matches but not in baseline matches
        rec = c_model_matches.difference(b_model_matches)
        reg = b_model_matches.difference(c_model_matches)

        model_recovered_spans += len(rec)
        model_regressed_spans += len(reg)

    return {
        "title_improved": title_improved,
        "title_regressed": title_regressed,
        "title_unchanged": title_unchanged,
        "net_title_gain": title_improved - title_regressed,
        "model_recovered_spans": model_recovered_spans,
        "model_regressed_spans": model_regressed_spans,
        "net_model_span_gain": model_recovered_spans - model_regressed_spans,
    }


def main():
    print("=" * 80)
    print("PHASE 32C: DEEP INTERNAL EVALUATION AND ERROR BREAKDOWN")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Device: {device}")

    # Load labels
    ref_cfg = AutoConfig.from_pretrained(Path("checkpoints/gbert_baseline_best"))
    id2label = {int(k): v for k, v in ref_cfg.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    # Load internal 800-title dev split
    df = load_training_data()
    all_examples = build_examples(df)
    train_4k, val_1k = split_examples(all_examples, random_state=42)
    _, dev_800, manifest = get_internal_split(train_4k, random_state=42)

    base_ckpt = CHECKPOINTS_DIR / "gbert_phase32c_baseline_best"
    chal_ckpt = CHECKPOINTS_DIR / "gbert_phase32c_challenger_best"

    if not base_ckpt.exists() or not chal_ckpt.exists():
        print(f"Error: One or both checkpoints not found!")
        print(f"  Baseline exists:   {base_ckpt.exists()} ({base_ckpt})")
        print(f"  Challenger exists: {chal_ckpt.exists()} ({chal_ckpt})")
        sys.exit(1)

    print("\n--- EVALUATING ON INTERNAL 800-TITLE DEV SET (PRIMARY DECISION GATE) ---")
    base_summary, base_eval = evaluate_model_on_split(base_ckpt, dev_800, label2id, id2label, device)
    chal_summary, chal_eval = evaluate_model_on_split(chal_ckpt, dev_800, label2id, id2label, device)

    base_model_audit = audit_vehicle_model_details(base_eval)
    chal_model_audit = audit_vehicle_model_details(chal_eval)

    h2h_comparison = compare_models_head_to_head(base_eval, chal_eval)

    print(f"{'Overall Metric':<35} {'Baseline (1.00x)':<20} {'Challenger (1.15x)':<20} {'Delta':<12}")
    print("-" * 88)
    print(f"{'Dev F0.2 (Macro Official)':<35} {base_summary['final_f02']:.6f}{'':<12} {chal_summary['final_f02']:.6f}{'':<12} {chal_summary['final_f02'] - base_summary['final_f02']:+.6f}")
    print(f"{'  - Category 1 F0.2':<35} {base_summary['cat1_f02']:.6f}{'':<12} {chal_summary['cat1_f02']:.6f}{'':<12} {chal_summary['cat1_f02'] - base_summary['cat1_f02']:+.6f}")
    print(f"{'  - Category 2 F0.2':<35} {base_summary['cat2_f02']:.6f}{'':<12} {chal_summary['cat2_f02']:.6f}{'':<12} {chal_summary['cat2_f02'] - base_summary['cat2_f02']:+.6f}")
    print(f"{'Micro Precision':<35} {base_summary['precision']:.4f}{'':<14} {chal_summary['precision']:.4f}{'':<14} {chal_summary['precision'] - base_summary['precision']:+.4f}")
    print(f"{'Micro Recall':<35} {base_summary['recall']:.4f}{'':<14} {chal_summary['recall']:.4f}{'':<14} {chal_summary['recall'] - base_summary['recall']:+.4f}")
    print(f"{'Total True Positives (TP)':<35} {base_summary['tp']:<20} {chal_summary['tp']:<20} {chal_summary['tp'] - base_summary['tp']:+d}")
    print(f"{'Total False Positives (FP)':<35} {base_summary['fp']:<20} {chal_summary['fp']:<20} {chal_summary['fp'] - base_summary['fp']:+d}")
    print(f"{'Total False Negatives (FN)':<35} {base_summary['fn']:<20} {chal_summary['fn']:<20} {chal_summary['fn'] - base_summary['fn']:+d}")

    print("\n" + "=" * 88)
    print("VEHICLE-MODEL (Kompatibles_Fahrzeug_Modell) SPECIFIC AUDIT")
    print("=" * 88)
    print(f"{'Vehicle-Model Metric':<35} {'Baseline (1.00x)':<20} {'Challenger (1.15x)':<20} {'Delta':<12}")
    print("-" * 88)
    print(f"{'Vehicle-Model F0.2':<35} {base_model_audit['model_f02']:.6f}{'':<12} {chal_model_audit['model_f02']:.6f}{'':<12} {chal_model_audit['model_f02'] - base_model_audit['model_f02']:+.6f}")
    print(f"{'Vehicle-Model Precision':<35} {base_model_audit['model_precision']:.4f}{'':<14} {chal_model_audit['model_precision']:.4f}{'':<14} {chal_model_audit['model_precision'] - base_model_audit['model_precision']:+.4f}")
    print(f"{'Vehicle-Model Recall':<35} {base_model_audit['model_recall']:.4f}{'':<14} {chal_model_audit['model_recall']:.4f}{'':<14} {chal_model_audit['model_recall'] - base_model_audit['model_recall']:+.4f}")
    print(f"{'Vehicle-Model TP':<35} {base_model_audit['model_tp']:<20} {chal_model_audit['model_tp']:<20} {chal_model_audit['model_tp'] - base_model_audit['model_tp']:+d}")
    print(f"{'Vehicle-Model FP':<35} {base_model_audit['model_fp']:<20} {chal_model_audit['model_fp']:<20} {chal_model_audit['model_fp'] - base_model_audit['model_fp']:+d}")
    print(f"{'Vehicle-Model FN':<35} {base_model_audit['model_fn']:<20} {chal_model_audit['model_fn']:<20} {chal_model_audit['model_fn'] - base_model_audit['model_fn']:+d}")
    print(f"{'Single-Token Exact Match Acc':<35} {base_model_audit['single_token_acc']:.2f}%{'':<13} {chal_model_audit['single_token_acc']:.2f}%{'':<13} {chal_model_audit['single_token_acc'] - base_model_audit['single_token_acc']:+.2f}%")
    print(f"{'Multi-Token Exact Match Acc':<35} {base_model_audit['multi_token_acc']:.2f}%{'':<13} {chal_model_audit['multi_token_acc']:.2f}%{'':<13} {chal_model_audit['multi_token_acc'] - base_model_audit['multi_token_acc']:+.2f}%")

    print("\n" + "=" * 88)
    print("BOUNDARY ERROR TAXONOMY & HEAD-TO-HEAD COMPARISON")
    print("=" * 88)
    print(f"Multi-Ground-Truth Overlap Errors: Baseline={base_model_audit['multi_true_overlaps']} | Challenger={chal_model_audit['multi_true_overlaps']} (Delta: {chal_model_audit['multi_true_overlaps'] - base_model_audit['multi_true_overlaps']:+d})")
    print(f"Single-Span Boundary Errors:       Baseline={base_model_audit['single_boundary_errors']} | Challenger={chal_model_audit['single_boundary_errors']} (Delta: {chal_model_audit['single_boundary_errors'] - base_model_audit['single_boundary_errors']:+d})")
    print(f"  - Sub-span Truncations (Missing):Baseline={base_model_audit['missing_tokens_subspan']} | Challenger={chal_model_audit['missing_tokens_subspan']} (Delta: {chal_model_audit['missing_tokens_subspan'] - base_model_audit['missing_tokens_subspan']:+d})")
    print(f"  - Super-span Expansions (Extra): Baseline={base_model_audit['extra_tokens_superspan']} | Challenger={chal_model_audit['extra_tokens_superspan']} (Delta: {chal_model_audit['extra_tokens_superspan'] - base_model_audit['extra_tokens_superspan']:+d})")
    print(f"\nHead-to-Head Title Counts (800 Titles):")
    print(f"  - Titles Improved:   {h2h_comparison['title_improved']}")
    print(f"  - Titles Regressed:  {h2h_comparison['title_regressed']}")
    print(f"  - Titles Unchanged:  {h2h_comparison['title_unchanged']}")
    print(f"  - Net Title Gain:    {h2h_comparison['net_title_gain']:+d}")
    print(f"\nModel Entity Span Counts:")
    print(f"  - Spans Recovered (Wrong -> Correct): {h2h_comparison['model_recovered_spans']}")
    print(f"  - Spans Regressed (Correct -> Wrong): {h2h_comparison['model_regressed_spans']}")
    print(f"  - Net Span Gain:                      {h2h_comparison['net_model_span_gain']:+d}")

    # Check Decision Gate
    gate_passed = chal_summary["final_f02"] > base_summary["final_f02"] + 1e-5
    print("\n" + "=" * 88)
    if gate_passed:
        print(">>> DECISION GATE RESULT: CHALLENGER BEATS BASELINE ON INTERNAL DEV <<<")
    else:
        print(">>> DECISION GATE RESULT: CHALLENGER DOES NOT BEAT BASELINE ON INTERNAL DEV <<<")
    print("=" * 88)

    # Save results
    eval_results = {
        "baseline_summary": base_summary,
        "challenger_summary": chal_summary,
        "baseline_model_audit": base_model_audit,
        "challenger_model_audit": chal_model_audit,
        "head_to_head": h2h_comparison,
        "gate_passed": gate_passed,
    }

    out_file = CACHE_DIR / "phase_32c_internal_eval_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nEvaluation metrics saved to {out_file}")


if __name__ == "__main__":
    main()
