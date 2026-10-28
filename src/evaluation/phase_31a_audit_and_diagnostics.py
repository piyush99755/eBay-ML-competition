"""
Phase 31A: Step 1 Baseline Reproduction, Boundary Error Audit & Step 2 Probability Analysis.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoConfig

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
from src.evaluation.experiment_26a_confidence_calibration import collect_active_probabilities
from src.evaluation.phase_30a_error_profile import (
    bio_to_entities_with_spans_and_conf,
    bio_to_true_spans,
)

CHAMPION_CHECKPOINTS = {
    "gbert": Path("checkpoints/gbert_baseline_best"),
    "xlmr42": Path("checkpoints/xlmr_baseline_best"),
    "xlmr7": Path("checkpoints/xlmr_seed_7_best"),
    "xlmr21": Path("checkpoints/xlmr_seed_21_best"),
}

CACHE_DIR = PROJECT_ROOT / "data" / "cache"

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
        p_fused = list(np.load(fused_cache_file, allow_pickle=True))
    else:
        print("Extracting champion model probabilities sequentially...")
        probs = {}
        for name, path in CHAMPION_CHECKPOINTS.items():
            print(f"  Extracting {name}...")
            probs[name] = collect_active_probabilities(path, val_examples, label2id, device, batch_size=8)
        p_fused = [
            0.40 * probs["gbert"][i] + 0.20 * probs["xlmr42"][i] + 0.20 * probs["xlmr7"][i] + 0.20 * probs["xlmr21"][i]
            for i in range(len(val_examples))
        ]
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(fused_cache_file, np.array(p_fused, dtype=object))
        print(f"Cached fused probabilities to {fused_cache_file}.")

    # =========================================================================
    # STEP 1: BASELINE REPRODUCTION & DETAILED BOUNDARY ERROR AUDIT
    # =========================================================================
    champ_records = []
    title_data = []

    for ex_idx, ex in enumerate(val_examples):
        tokens = ex["tokens"]
        cat_id = ex["category"]
        true_bio = ex["bio_tags"]

        p = p_fused[ex_idx]
        pred_ids = np.argmax(p, axis=-1)
        pred_bio = [id2label[pid] for pid in pred_ids]
        selected_probs = [float(p[t_idx, pred_ids[t_idx]]) for t_idx in range(len(tokens))]

        true_spans = bio_to_true_spans(tokens, true_bio)
        cat_valid_true = [s for s in true_spans if filter_entities_by_category([(s[0], s[1])], cat_id)]

        pred_spans_conf = bio_to_entities_with_spans_and_conf(tokens, pred_bio, selected_probs)
        cat_valid_pred = [s for s in pred_spans_conf if filter_entities_by_category([(s[0], s[1])], cat_id)]

        # MIN 0.60 filter
        ents_060 = [s for s in cat_valid_pred if s[5] >= 0.60]

        champ_records.append({
            "category": cat_id,
            "true_entities": [(s[0], s[1]) for s in cat_valid_true],
            "predicted_entities": [(s[0], s[1]) for s in ents_060],
        })

        title_data.append({
            "tokens": tokens,
            "category": cat_id,
            "true_spans": cat_valid_true,
            "pred_spans": ents_060,
            "pred_bio": pred_bio,
            "probs": p,
        })

    # Official scoring
    comp_res = calculate_competition_metrics(champ_records)
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for r in champ_records:
        tp, fp, fn = count_matches(r["predicted_entities"], r["true_entities"])
        total_tp += tp
        total_fp += fp
        total_fn += fn

    pr = precision_recall_fbeta(total_tp, total_fp, total_fn, beta=0.2)
    final_f02 = comp_res["final_score"]
    total_preds = sum(len(r["predicted_entities"]) for r in champ_records)

    print("\n" + "=" * 70)
    print("STEP 1: BASELINE CHAMPION REPRODUCTION")
    print("=" * 70)
    print(f"Official F0.2: {final_f02:.6f} (Target: 0.899802)")
    print(f"Category 1:    {comp_res['categories'][1]['score']:.6f} (Target: 0.899921)")
    print(f"Category 2:    {comp_res['categories'][2]['score']:.6f} (Target: 0.899683)")
    print(f"Precision:     {pr['precision']:.4f} (Target: 0.9172)")
    print(f"Recall:        {pr['recall']:.4f} (Target: 0.8350)")
    print(f"TP / FP / FN:  {total_tp} / {total_fp} / {total_fn} (Target: 6340 / 572 / 1253)")
    print(f"Total Preds:   {total_preds} (Target: 6912)")

    assert abs(final_f02 - 0.899802) < 1e-5, f"Baseline F0.2 mismatch: {final_f02}"
    assert total_tp == 6340, f"TP mismatch: {total_tp}"
    assert total_fp == 572, f"FP mismatch: {total_fp}"
    assert total_fn == 1253, f"FN mismatch: {total_fn}"
    assert total_preds == 6912, f"Pred count mismatch: {total_preds}"
    print(">>> EXACT BASELINE REPRODUCTION VERIFIED 100% BIT-FOR-BIT <<<\n")

    # Detailed Boundary Error Audit for Kompatibles_Fahrzeug_Modell
    # We analyze all false positives of Kompatibles_Fahrzeug_Modell
    target_aspect = "Kompatibles_Fahrzeug_Modell"

    audit_stats = {
        "total_fp": 0,
        "same_aspect_boundary_errors": 0,
        "wrong_aspect_errors": 0,
        "spurious_hallucinations": 0,
        "multi_true_overlap": 0,
        "start_only_error": 0,
        "end_only_error": 0,
        "both_boundary_error": 0,
        "missing_tokens_subspan": 0, # Pred is strict sub-span of true
        "extra_tokens_superspan": 0,  # Pred is strict super-span of true
        "cross_shifted_tokens": 0,    # Partial overlap, shifted
        "true_single_token": 0,
        "true_multi_token": 0,
        "pred_single_token": 0,
        "pred_multi_token": 0,
        "one_to_many_true_reused": 0, # Cases where multiple pred entities overlap the same true entity
    }

    per_category_audit = {
        1: defaultdict(int),
        2: defaultdict(int),
    }

    # Track how many times each true entity is overlapped
    for title in title_data:
        cat_id = title["category"]
        true_spans = title["true_spans"]
        pred_spans = title["pred_spans"]

        # Find true entities of target aspect
        true_matched = [False] * len(true_spans)
        true_overlap_counts = [0] * len(true_spans)

        # Mark exact matches first
        for p in pred_spans:
            p_asp, p_val, p_s, p_e, _, _ = p
            for t_idx, t in enumerate(true_spans):
                t_asp, t_val, t_s, t_e = t
                if not true_matched[t_idx] and t_asp == p_asp and t_val == p_val:
                    true_matched[t_idx] = True
                    break

        # Now audit false positives of target aspect
        for p in pred_spans:
            p_asp, p_val, p_s, p_e, _, p_min_c = p
            if p_asp != target_aspect:
                continue

            # Is it a true positive?
            is_tp = False
            for t_idx, t in enumerate(true_spans):
                if true_matched[t_idx] and t[0] == p_asp and t[1] == p_val:
                    is_tp = True
                    break

            if is_tp:
                continue

            # It is a False Positive!
            audit_stats["total_fp"] += 1
            per_category_audit[cat_id]["total_fp"] += 1

            if p_e > p_s:
                audit_stats["pred_multi_token"] += 1
            else:
                audit_stats["pred_single_token"] += 1

            # Find all overlapping true spans
            overlapping_true = []
            for t_idx, t in enumerate(true_spans):
                t_asp, t_val, t_s, t_e = t
                if max(p_s, t_s) <= min(p_e, t_e):
                    overlapping_true.append((t_idx, t))
                    true_overlap_counts[t_idx] += 1

            if len(overlapping_true) == 0:
                audit_stats["spurious_hallucinations"] += 1
                per_category_audit[cat_id]["spurious_hallucinations"] += 1
            elif len(overlapping_true) > 1:
                audit_stats["multi_true_overlap"] += 1
                per_category_audit[cat_id]["multi_true_overlap"] += 1
            else:
                # Exactly one overlapping true entity
                t_idx, (t_asp, t_val, t_s, t_e) = overlapping_true[0]

                if t_e > t_s:
                    audit_stats["true_multi_token"] += 1
                else:
                    audit_stats["true_single_token"] += 1

                if t_asp != target_aspect:
                    audit_stats["wrong_aspect_errors"] += 1
                    per_category_audit[cat_id]["wrong_aspect_errors"] += 1
                else:
                    # Same aspect boundary error!
                    audit_stats["same_aspect_boundary_errors"] += 1
                    per_category_audit[cat_id]["same_aspect_boundary_errors"] += 1

                    # Start vs End boundary
                    if p_s != t_s and p_e == t_e:
                        audit_stats["start_only_error"] += 1
                        per_category_audit[cat_id]["start_only_error"] += 1
                    elif p_s == t_s and p_e != t_e:
                        audit_stats["end_only_error"] += 1
                        per_category_audit[cat_id]["end_only_error"] += 1
                    else:
                        audit_stats["both_boundary_error"] += 1
                        per_category_audit[cat_id]["both_boundary_error"] += 1

                    # Missing vs Extra vs Shifted
                    if p_s >= t_s and p_e <= t_e:
                        audit_stats["missing_tokens_subspan"] += 1
                        per_category_audit[cat_id]["missing_tokens_subspan"] += 1
                    elif p_s <= t_s and p_e >= t_e:
                        audit_stats["extra_tokens_superspan"] += 1
                        per_category_audit[cat_id]["extra_tokens_superspan"] += 1
                    else:
                        audit_stats["cross_shifted_tokens"] += 1
                        per_category_audit[cat_id]["cross_shifted_tokens"] += 1

        # Check for one-to-many: multiple pred entities claiming the same true entity
        for t_idx, count in enumerate(true_overlap_counts):
            if count > 1 and true_spans[t_idx][0] == target_aspect:
                audit_stats["one_to_many_true_reused"] += (count - 1)

    print("=" * 70)
    print("STEP 1: DETAILED VEHICLE MODEL (Kompatibles_Fahrzeug_Modell) BOUNDARY ERROR AUDIT")
    print("=" * 70)
    print(f"Total False Positives for Aspect: {audit_stats['total_fp']} (Cat 1: {per_category_audit[1]['total_fp']}, Cat 2: {per_category_audit[2]['total_fp']})")
    print(f"  - Same-Aspect Boundary Errors:   {audit_stats['same_aspect_boundary_errors']} (Cat 1: {per_category_audit[1]['same_aspect_boundary_errors']}, Cat 2: {per_category_audit[2]['same_aspect_boundary_errors']})")
    print(f"  - Wrong-Aspect Overlap:          {audit_stats['wrong_aspect_errors']} (Cat 1: {per_category_audit[1]['wrong_aspect_errors']}, Cat 2: {per_category_audit[2]['wrong_aspect_errors']})")
    print(f"  - Overlaps Multiple True Spans:  {audit_stats['multi_true_overlap']} (Cat 1: {per_category_audit[1]['multi_true_overlap']}, Cat 2: {per_category_audit[2]['multi_true_overlap']})")
    print(f"  - Spurious Hallucinations:       {audit_stats['spurious_hallucinations']} (Cat 1: {per_category_audit[1]['spurious_hallucinations']}, Cat 2: {per_category_audit[2]['spurious_hallucinations']})")
    print(f"\nBoundary Discrepancy Breakdown (out of {audit_stats['same_aspect_boundary_errors']} same-aspect errors):")
    print(f"  - Start-Only Error (s!=t_s, e==t_e): {audit_stats['start_only_error']:<4} ({audit_stats['start_only_error']/audit_stats['same_aspect_boundary_errors']*100:.1f}%)")
    print(f"  - End-Only Error (s==t_s, e!=t_e):   {audit_stats['end_only_error']:<4} ({audit_stats['end_only_error']/audit_stats['same_aspect_boundary_errors']*100:.1f}%)")
    print(f"  - Both-Boundary Error (s!=, e!=):    {audit_stats['both_boundary_error']:<4} ({audit_stats['both_boundary_error']/audit_stats['same_aspect_boundary_errors']*100:.1f}%)")
    print(f"\nToken Coverage Discrepancy Breakdown:")
    print(f"  - Missing Tokens (Strict Sub-Span):  {audit_stats['missing_tokens_subspan']:<4} ({audit_stats['missing_tokens_subspan']/audit_stats['same_aspect_boundary_errors']*100:.1f}%)")
    print(f"  - Extra Tokens (Strict Super-Span):  {audit_stats['extra_tokens_superspan']:<4} ({audit_stats['extra_tokens_superspan']/audit_stats['same_aspect_boundary_errors']*100:.1f}%)")
    print(f"  - Shifted / Cross Tokens:            {audit_stats['cross_shifted_tokens']:<4} ({audit_stats['cross_shifted_tokens']/audit_stats['same_aspect_boundary_errors']*100:.1f}%)")
    print(f"\nMulti-Token Entity Analysis:")
    print(f"  - True Entity is Multi-Token:        {audit_stats['true_multi_token']:<4} | Single-Token: {audit_stats['true_single_token']:<4}")
    print(f"  - Pred Entity is Multi-Token:        {audit_stats['pred_multi_token']:<4} | Single-Token: {audit_stats['pred_single_token']:<4}")
    print(f"  - One-to-Many Multiple Overlaps:     {audit_stats['one_to_many_true_reused']}")

    # =========================================================================
    # STEP 2: ENSEMBLE PROBABILITIES & BIO TRANSITION INCONSISTENCY AUDIT
    # =========================================================================
    b_id = label2id["B-Kompatibles_Fahrzeug_Modell"]
    i_id = label2id["I-Kompatibles_Fahrzeug_Modell"]
    o_id = label2id["O"]

    # Check BIO transition violations in raw argmax predictions across all titles
    illegal_transitions = 0
    total_transitions = 0
    illegal_details = defaultdict(int)

    # Probabilities around vehicle model boundaries
    boundary_prob_stats = {
        "true_missing_token_b_prob": [],
        "true_missing_token_i_prob": [],
        "true_missing_token_o_prob": [],
        "pred_extra_token_b_prob": [],
        "pred_extra_token_i_prob": [],
        "pred_extra_token_o_prob": [],
    }

    for title in title_data:
        p_seq = title["probs"]
        bio_seq = title["pred_bio"]
        T = len(bio_seq)

        for t in range(T):
            tag = bio_seq[t]
            if t == 0:
                if tag.startswith("I-"):
                    illegal_transitions += 1
                    illegal_details["Start_with_I"] += 1
            else:
                prev_tag = bio_seq[t - 1]
                total_transitions += 1
                if tag.startswith("I-"):
                    asp = tag[2:]
                    if prev_tag != f"B-{asp}" and prev_tag != f"I-{asp}":
                        illegal_transitions += 1
                        illegal_details[f"{prev_tag} -> {tag}"] += 1

    print("\n" + "=" * 70)
    print("STEP 2: RAW ENSEMBLE BIO TRANSITION INCONSISTENCY AUDIT")
    print("=" * 70)
    print(f"Total Token Transitions: {total_transitions}")
    print(f"Illegal BIO Transitions: {illegal_transitions} ({illegal_transitions/max(1, total_transitions)*100:.3f}%)")
    print(f"Illegal Transition Breakdown: {dict(illegal_details)}")

    # Save Step 1 & 2 audit summary to JSON cache
    audit_results = {
        "baseline_reproduction": {
            "f02": final_f02,
            "cat1_f02": comp_res["categories"][1]["score"],
            "cat2_f02": comp_res["categories"][2]["score"],
            "precision": pr["precision"],
            "recall": pr["recall"],
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "total_preds": total_preds,
        },
        "boundary_audit": audit_stats,
        "per_category_boundary": {1: dict(per_category_audit[1]), 2: dict(per_category_audit[2])},
        "transition_violations": {
            "total": illegal_transitions,
            "details": dict(illegal_details),
        },
    }

    out_json = CACHE_DIR / "phase_31a_step1_step2_audit.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)
    print(f"\nAudit results saved to {out_json}")

if __name__ == "__main__":
    main()
