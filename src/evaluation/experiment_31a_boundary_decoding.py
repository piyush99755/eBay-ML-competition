"""
Phase 31A: Boundary-Aware Decoding Research.

Challengers evaluated:
- Baseline: Frozen Exp 10 Champion (Argmax + Category Filter + MIN 0.60)
- Challenger A: BIO-Constrained Viterbi Decoding
- Challenger B: Local Vehicle-Model Boundary Refinement
  * B1: Conservative Joint Window Likelihood (predeclared margin gamma = 0.5)
  * B2: Length-Normalized Span Likelihood (predeclared margin gamma = 0.1)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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


# ---------------------------------------------------------------------------
# CHALLENGER A: BIO-CONSTRAINED VITERBI DECODER
# ---------------------------------------------------------------------------

def build_bio_transition_mask(id2label: Dict[int, str]) -> np.ndarray:
    """
    Construct a boolean matrix legal_trans[i, j] indicating whether
    transition from label i to label j is legally permitted in BIO.
    """
    num_labels = len(id2label)
    legal = np.zeros((num_labels, num_labels), dtype=bool)

    for i in range(num_labels):
        tag_i = id2label[i]
        for j in range(num_labels):
            tag_j = id2label[j]

            # 1. Any tag can transition to 'O'
            if tag_j == "O":
                legal[i, j] = True
                continue

            # 2. Any tag can transition to 'B-X' (starts a new entity)
            if tag_j.startswith("B-"):
                legal[i, j] = True
                continue

            # 3. 'I-X' can ONLY follow 'B-X' or 'I-X' with the exact same aspect
            if tag_j.startswith("I-"):
                asp_j = tag_j[2:]
                if tag_i == f"B-{asp_j}" or tag_i == f"I-{asp_j}":
                    legal[i, j] = True
                else:
                    legal[i, j] = False
                continue

    return legal


def viterbi_decode_sequence(
    probs: np.ndarray,
    id2label: Dict[int, str],
    legal_trans: np.ndarray,
) -> Tuple[List[str], List[float]]:
    """
    Run BIO-constrained Viterbi decoding over token probabilities.
    Returns:
        (best_bio_tags, selected_token_probs)
    """
    T, num_labels = probs.shape
    log_probs = np.log(np.maximum(probs, 1e-12))

    # dp[t, j] stores max log-prob of sequence ending with label j at step t
    dp = np.full((T, num_labels), -np.inf, dtype=np.float64)
    backpointers = np.zeros((T, num_labels), dtype=np.int32)

    # Step t = 0: Only 'O' and 'B-*' are legal starts
    for j in range(num_labels):
        tag = id2label[j]
        if tag == "O" or tag.startswith("B-"):
            dp[0, j] = log_probs[0, j]
        else:
            dp[0, j] = -np.inf  # Illegal to start with I-tag

    # Steps t = 1 ... T-1
    for t in range(1, T):
        emiss = log_probs[t]
        for j in range(num_labels):
            valid_prev = np.where(legal_trans[:, j])[0]
            if len(valid_prev) == 0:
                continue

            prev_scores = dp[t - 1, valid_prev]
            best_idx = np.argmax(prev_scores)
            best_prev = valid_prev[best_idx]
            best_score = prev_scores[best_idx]

            dp[t, j] = best_score + emiss[j]
            backpointers[t, j] = best_prev

    # Traceback
    best_last_label = int(np.argmax(dp[T - 1]))
    path = [best_last_label]

    for t in range(T - 1, 0, -1):
        prev_label = backpointers[t, path[-1]]
        path.append(int(prev_label))

    path.reverse()
    best_tags = [id2label[pid] for pid in path]
    selected_probs = [float(probs[t, path[t]]) for t in range(T)]

    return best_tags, selected_probs


# ---------------------------------------------------------------------------
# CHALLENGER B: LOCAL VEHICLE-MODEL BOUNDARY REFINEMENT
# ---------------------------------------------------------------------------

def refine_vehicle_model_boundaries(
    tokens: List[str],
    pred_entities: List[Tuple[str, str, int, int, float, float]],
    probs: np.ndarray,
    label2id: Dict[str, int],
    scoring_mode: str = "length_norm",  # "window_joint" or "length_norm"
    margin: float = 0.10,
    min_conf_thresh: float = 0.60,
) -> Tuple[List[Tuple[str, str, int, int, float, float]], int]:
    """
    For Kompatibles_Fahrzeug_Modell entities, evaluate local boundary adjustments:
    [s, e] -> [s-1, e], [s+1, e], [s, e-1], [s, e+1].
    """
    T = len(tokens)
    target_aspect = "Kompatibles_Fahrzeug_Modell"
    b_id = label2id[f"B-{target_aspect}"]
    i_id = label2id[f"I-{target_aspect}"]
    o_id = label2id["O"]

    log_p = np.log(np.maximum(probs, 1e-12))

    # Mark tokens occupied by other predicted entities
    occupied_mask = np.zeros(T, dtype=bool)
    for ent in pred_entities:
        asp, val, s, e, _, _ = ent
        if asp != target_aspect:
            occupied_mask[s:e + 1] = True

    refined_ents = []
    num_changed = 0

    for ent_idx, ent in enumerate(pred_entities):
        asp, val, s, e, mean_c, min_c = ent

        if asp != target_aspect:
            refined_ents.append(ent)
            continue

        other_vm_occupied = np.zeros(T, dtype=bool)
        for o_idx, o_ent in enumerate(pred_entities):
            if o_idx != ent_idx:
                other_vm_occupied[o_ent[2]:o_ent[3] + 1] = True

        blocked = occupied_mask | other_vm_occupied

        candidates = [(s, e, "ORIGINAL")]
        # Extend left
        if s > 0 and not blocked[s - 1]:
            candidates.append((s - 1, e, "EXTEND_LEFT"))
        # Extend right
        if e < T - 1 and not blocked[e + 1]:
            candidates.append((s, e + 1, "EXTEND_RIGHT"))
        # Trim left
        if s < e:
            candidates.append((s + 1, e, "TRIM_LEFT"))
        # Trim right
        if s < e:
            candidates.append((s, e - 1, "TRIM_RIGHT"))

        def score_candidate(c_s: int, c_e: int) -> Tuple[float, float]:
            c_probs = [probs[c_s, b_id]] + [probs[k, i_id] for k in range(c_s + 1, c_e + 1)]
            c_min_conf = float(min(c_probs))

            if scoring_mode == "length_norm":
                c_score = float(np.mean([log_p[c_s, b_id]] + [log_p[k, i_id] for k in range(c_s + 1, c_e + 1)]))
            else:
                # Window joint likelihood
                min_w = max(0, min(c[0] - 1 for c in candidates))
                max_w = min(T - 1, max(c[1] + 1 for c in candidates))
                c_score = 0.0
                for k in range(min_w, max_w + 1):
                    if k == c_s:
                        c_score += log_p[k, b_id]
                    elif c_s < k <= c_e:
                        c_score += log_p[k, i_id]
                    else:
                        c_score += log_p[k, o_id]

            return c_score, c_min_conf

        orig_score, orig_min_conf = score_candidate(s, e)

        best_span = (s, e)
        best_score = orig_score
        best_min_conf = orig_min_conf
        best_action = "ORIGINAL"

        for c_s, c_e, action in candidates:
            if action == "ORIGINAL":
                continue

            c_score, c_min_conf = score_candidate(c_s, c_e)
            delta = c_score - orig_score

            if delta > margin and c_min_conf >= min_conf_thresh:
                if c_score > best_score:
                    best_score = c_score
                    best_span = (c_s, c_e)
                    best_min_conf = c_min_conf
                    best_action = action

        if best_action != "ORIGINAL":
            num_changed += 1
            new_s, new_e = best_span
            new_val = " ".join(tokens[new_s:new_e + 1])
            new_probs = [probs[new_s, b_id]] + [probs[k, i_id] for k in range(new_s + 1, new_e + 1)]
            new_mean_conf = float(np.mean(new_probs))
            refined_ents.append((asp, new_val, new_s, new_e, new_mean_conf, best_min_conf))
        else:
            refined_ents.append(ent)

    return refined_ents, num_changed


# ---------------------------------------------------------------------------
# EVALUATION & RECOVERY AUDIT PIPELINE
# ---------------------------------------------------------------------------

def run_evaluation(
    title_data: List[dict],
    candidate_predictions_per_title: List[List[Tuple[str, str]]],
) -> dict:
    records = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for title, preds in zip(title_data, candidate_predictions_per_title):
        cat_id = title["category"]
        true_ents = [(s[0], s[1]) for s in title["true_spans"]]
        filtered_preds = filter_entities_by_category(preds, cat_id)

        tp, fp, fn = count_matches(filtered_preds, true_ents)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        records.append({
            "category": cat_id,
            "true_entities": true_ents,
            "predicted_entities": filtered_preds,
        })

    comp_res = calculate_competition_metrics(records)
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
        "records": records,
    }


def audit_exact_recovery(
    title_data: List[dict],
    base_ents_per_title: List[List[Tuple[str, str, int, int, float, float]]],
    cand_ents_per_title: List[List[Tuple[str, str, int, int, float, float]]],
    target_aspect: str = "Kompatibles_Fahrzeug_Modell",
) -> dict:
    total_changed = 0
    wrong_to_correct = 0
    correct_to_wrong = 0
    wrong_to_wrong = 0

    title_improved = 0
    title_regressed = 0
    title_unchanged = 0

    for title, base_ents, cand_ents in zip(title_data, base_ents_per_title, cand_ents_per_title):
        true_spans = title["true_spans"]
        true_set = set((t[0], t[1]) for t in true_spans)

        b_preds = set((e[0], e[1]) for e in base_ents if e[0] == target_aspect)
        c_preds = set((e[0], e[1]) for e in cand_ents if e[0] == target_aspect)

        if b_preds != c_preds:
            total_changed += 1

            removed_from_base = b_preds - c_preds
            added_to_cand = c_preds - b_preds

            for r in removed_from_base:
                if r in true_set:
                    correct_to_wrong += 1

            for a in added_to_cand:
                if a in true_set:
                    wrong_to_correct += 1
                else:
                    wrong_to_wrong += 1

            b_all = [(e[0], e[1]) for e in base_ents]
            c_all = [(e[0], e[1]) for e in cand_ents]
            b_sc = calculate_competition_metrics([{"category": title["category"], "true_entities": list(true_set), "predicted_entities": b_all}])["final_score"]
            c_sc = calculate_competition_metrics([{"category": title["category"], "true_entities": list(true_set), "predicted_entities": c_all}])["final_score"]

            if c_sc > b_sc + 1e-6:
                title_improved += 1
            elif c_sc < b_sc - 1e-6:
                title_regressed += 1
            else:
                title_unchanged += 1
        else:
            title_unchanged += 1

    return {
        "titles_with_changes": total_changed,
        "wrong_to_correct": wrong_to_correct,
        "correct_to_wrong": correct_to_wrong,
        "wrong_to_wrong": wrong_to_wrong,
        "title_improved": title_improved,
        "title_regressed": title_regressed,
        "title_unchanged": title_unchanged,
    }


def main():
    print("=" * 80)
    print("PHASE 31A: BOUNDARY-AWARE DECODING RESEARCH")
    print("=" * 80)

    # 1. Load Data and Fused Probabilities
    df = load_training_data()
    all_examples = build_examples(df)
    _, val_examples = split_examples(all_examples, random_state=42)
    print(f"Loaded {len(val_examples)} Split-A validation titles.")

    fused_cache_file = CACHE_DIR / "split_a_champion_fused_probs.npy"
    if not fused_cache_file.exists():
        print(f"Error: {fused_cache_file} not found. Run phase_31a_audit_and_diagnostics.py first.")
        sys.exit(1)
    p_fused = list(np.load(fused_cache_file, allow_pickle=True))

    from transformers import AutoConfig
    ref_cfg = AutoConfig.from_pretrained(Path("checkpoints/gbert_baseline_best"))
    id2label = {int(k): v for k, v in ref_cfg.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    legal_trans = build_bio_transition_mask(id2label)
    title_data = []

    base_ents_060_list = []
    viterbi_ents_060_list = []
    refined_b1_ents_060_list = []  # window_joint (margin 0.5)
    refined_b2_ents_060_list = []  # length_norm (margin 0.1)

    total_b1_changed = 0
    total_b2_changed = 0

    for ex_idx, ex in enumerate(val_examples):
        tokens = ex["tokens"]
        cat_id = ex["category"]
        true_bio = ex["bio_tags"]
        p = p_fused[ex_idx]

        true_spans = bio_to_true_spans(tokens, true_bio)
        cat_valid_true = [s for s in true_spans if filter_entities_by_category([(s[0], s[1])], cat_id)]

        # --- Baseline: Argmax ---
        pred_ids = np.argmax(p, axis=-1)
        pred_bio = [id2label[pid] for pid in pred_ids]
        selected_probs = [float(p[t_idx, pred_ids[t_idx]]) for t_idx in range(len(tokens))]

        pred_spans_conf = bio_to_entities_with_spans_and_conf(tokens, pred_bio, selected_probs)
        cat_valid_pred = [s for s in pred_spans_conf if filter_entities_by_category([(s[0], s[1])], cat_id)]
        base_ents_060 = [s for s in cat_valid_pred if s[5] >= 0.60]
        base_ents_060_list.append(base_ents_060)

        # --- Challenger A: BIO-Constrained Viterbi ---
        vit_bio, vit_selected_probs = viterbi_decode_sequence(p, id2label, legal_trans)
        vit_spans_conf = bio_to_entities_with_spans_and_conf(tokens, vit_bio, vit_selected_probs)
        cat_valid_vit = [s for s in vit_spans_conf if filter_entities_by_category([(s[0], s[1])], cat_id)]
        vit_ents_060 = [s for s in cat_valid_vit if s[5] >= 0.60]
        viterbi_ents_060_list.append(vit_ents_060)

        # --- Challenger B1: Window Joint Likelihood (Margin 0.5) ---
        b1_ents, n_b1 = refine_vehicle_model_boundaries(
            tokens=tokens,
            pred_entities=base_ents_060,
            probs=p,
            label2id=label2id,
            scoring_mode="window_joint",
            margin=0.50,
            min_conf_thresh=0.60,
        )
        refined_b1_ents_060_list.append(b1_ents)
        total_b1_changed += n_b1

        # --- Challenger B2: Length-Normalized Span Likelihood (Margin 0.1) ---
        b2_ents, n_b2 = refine_vehicle_model_boundaries(
            tokens=tokens,
            pred_entities=base_ents_060,
            probs=p,
            label2id=label2id,
            scoring_mode="length_norm",
            margin=0.10,
            min_conf_thresh=0.60,
        )
        refined_b2_ents_060_list.append(b2_ents)
        total_b2_changed += n_b2

        title_data.append({
            "tokens": tokens,
            "category": cat_id,
            "true_spans": cat_valid_true,
            "probs": p,
        })

    # 3. Evaluate All Configurations
    base_metrics = run_evaluation(title_data, [[(e[0], e[1]) for e in l] for l in base_ents_060_list])
    vit_metrics = run_evaluation(title_data, [[(e[0], e[1]) for e in l] for l in viterbi_ents_060_list])
    b1_metrics = run_evaluation(title_data, [[(e[0], e[1]) for e in l] for l in refined_b1_ents_060_list])
    b2_metrics = run_evaluation(title_data, [[(e[0], e[1]) for e in l] for l in refined_b2_ents_060_list])

    # 4. Exact Entity Recovery Audits
    vit_recovery = audit_exact_recovery(title_data, base_ents_060_list, viterbi_ents_060_list)
    b1_recovery = audit_exact_recovery(title_data, base_ents_060_list, refined_b1_ents_060_list)
    b2_recovery = audit_exact_recovery(title_data, base_ents_060_list, refined_b2_ents_060_list)

    # 5. Print Comparative Results
    print("\n" + "=" * 105)
    print("COMPARATIVE EVALUATION RESULTS (Split A)")
    print("=" * 105)
    print(f"{'Configuration':<35} {'Official F0.2':<15} {'Delta':<10} {'Cat 1':<10} {'Cat 2':<10} {'Prec':<8} {'Rec':<8} {'TP':<6} {'FP':<5} {'FN'}")
    print("-" * 105)
    print(f"{'Frozen Champion (Argmax + MIN 0.60)':<35} {base_metrics['final_score']:<15.6f} {'--':<10} {base_metrics['cat1_score']:<10.6f} {base_metrics['cat2_score']:<10.6f} {base_metrics['precision']:<8.4f} {base_metrics['recall']:<8.4f} {base_metrics['tp']:<6} {base_metrics['fp']:<5} {base_metrics['fn']}")
    print(f"{'Challenger A (Constrained Viterbi)':<35} {vit_metrics['final_score']:<15.6f} {vit_metrics['final_score'] - base_metrics['final_score']:<+10.6f} {vit_metrics['cat1_score']:<10.6f} {vit_metrics['cat2_score']:<10.6f} {vit_metrics['precision']:<8.4f} {vit_metrics['recall']:<8.4f} {vit_metrics['tp']:<6} {vit_metrics['fp']:<5} {vit_metrics['fn']}")
    print(f"{'Challenger B1 (Joint Window Margin)':<35} {b1_metrics['final_score']:<15.6f} {b1_metrics['final_score'] - base_metrics['final_score']:<+10.6f} {b1_metrics['cat1_score']:<10.6f} {b1_metrics['cat2_score']:<10.6f} {b1_metrics['precision']:<8.4f} {b1_metrics['recall']:<8.4f} {b1_metrics['tp']:<6} {b1_metrics['fp']:<5} {b1_metrics['fn']}")
    print(f"{'Challenger B2 (Length-Norm Margin 0.1)':<35} {b2_metrics['final_score']:<15.6f} {b2_metrics['final_score'] - base_metrics['final_score']:<+10.6f} {b2_metrics['cat1_score']:<10.6f} {b2_metrics['cat2_score']:<10.6f} {b2_metrics['precision']:<8.4f} {b2_metrics['recall']:<8.4f} {b2_metrics['tp']:<6} {b2_metrics['fp']:<5} {b2_metrics['fn']}")

    # Per-Aspect Vehicle Model Performance
    print("\n" + "=" * 80)
    print("VEHICLE-MODEL (Kompatibles_Fahrzeug_Modell) SPECIFIC METRICS")
    print("=" * 80)
    target_asp = "Kompatibles_Fahrzeug_Modell"
    for name, m in [("Champion", base_metrics), ("Challenger A (Viterbi)", vit_metrics), ("Challenger B1 (Joint Window)", b1_metrics), ("Challenger B2 (Length-Norm)", b2_metrics)]:
        c1_asp = m["comp_results"]["categories"][1]["aspects"][target_asp]
        c2_asp = m["comp_results"]["categories"][2]["aspects"][target_asp]
        tot_tp = c1_asp["tp"] + c2_asp["tp"]
        tot_fp = c1_asp["fp"] + c2_asp["fp"]
        tot_fn = c1_asp["fn"] + c2_asp["fn"]
        p = tot_tp / max(1, tot_tp + tot_fp)
        r = tot_tp / max(1, tot_tp + tot_fn)
        f02 = (1.04 * p * r) / (0.04 * p + r) if (0.04 * p + r) > 0 else 0.0
        print(f"{name:<30}: F0.2 = {f02:.4f} | Prec = {p:.4f} | Rec = {r:.4f} | TP = {tot_tp} | FP = {tot_fp} | FN = {tot_fn}")

    # Exact Recovery Analysis
    print("\n" + "=" * 80)
    print("EXACT ENTITY RECOVERY & REGRESSION AUDIT (Vehicle Model)")
    print("=" * 80)
    print(f"Challenger A (Viterbi):")
    print(f"  - Titles with Span Changes:        {vit_recovery['titles_with_changes']}")
    print(f"  - Wrong-to-Correct (FP -> TP):     {vit_recovery['wrong_to_correct']}")
    print(f"  - Correct-to-Wrong (TP -> Error):  {vit_recovery['correct_to_wrong']}")
    print(f"  - Incorrect-to-Incorrect (FP->FP): {vit_recovery['wrong_to_wrong']}")
    print(f"  - Title Impact (Out of 1,000):     Improved: {vit_recovery['title_improved']} | Regressed: {vit_recovery['title_regressed']} | Unchanged: {vit_recovery['title_unchanged']}")

    print(f"\nChallenger B1 (Joint Window Margin = 0.5):")
    print(f"  - Total Spans Refined:             {total_b1_changed}")
    print(f"  - Invariant Property:              Argmax is already joint likelihood maximizer under independent emissions.")

    print(f"\nChallenger B2 (Length-Norm Margin = 0.1):")
    print(f"  - Total Spans Refined:             {total_b2_changed}")
    print(f"  - Titles with Span Changes:        {b2_recovery['titles_with_changes']}")
    print(f"  - Wrong-to-Correct (FP -> TP):     {b2_recovery['wrong_to_correct']}")
    print(f"  - Correct-to-Wrong (TP -> Error):  {b2_recovery['correct_to_wrong']}")
    print(f"  - Incorrect-to-Incorrect (FP->FP): {b2_recovery['wrong_to_wrong']}")
    print(f"  - Title Impact (Out of 1,000):     Improved: {b2_recovery['title_improved']} | Regressed: {b2_recovery['title_regressed']} | Unchanged: {b2_recovery['title_unchanged']}")

    # Rare Aspect Extinction Audit
    rare_aspects = ["Oe/Oem_Referenznummer(N)", "Modell", "Stärke", "Material", "Produktlinie"]
    print("\n--- Rare Aspect Extinction Audit:")
    for asp in rare_aspects:
        cnt_base = sum(1 for l in base_ents_060_list for e in l if e[0] == asp)
        cnt_vit = sum(1 for l in viterbi_ents_060_list for e in l if e[0] == asp)
        cnt_b1 = sum(1 for l in refined_b1_ents_060_list for e in l if e[0] == asp)
        cnt_b2 = sum(1 for l in refined_b2_ents_060_list for e in l if e[0] == asp)
        print(f"  {asp:<30}: Champion = {cnt_base:<3} | Viterbi = {cnt_vit:<3} | B1 = {cnt_b1:<3} | B2 = {cnt_b2:<3}")

    # Save complete evaluation artifact
    results_summary = {
        "champion": {
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
        "challenger_a_viterbi": {
            "f02": vit_metrics["final_score"],
            "delta": vit_metrics["final_score"] - base_metrics["final_score"],
            "cat1_f02": vit_metrics["cat1_score"],
            "cat2_f02": vit_metrics["cat2_score"],
            "precision": vit_metrics["precision"],
            "recall": vit_metrics["recall"],
            "tp": vit_metrics["tp"],
            "fp": vit_metrics["fp"],
            "fn": vit_metrics["fn"],
            "total_preds": vit_metrics["total_preds"],
            "recovery": vit_recovery,
        },
        "challenger_b1_joint_window": {
            "f02": b1_metrics["final_score"],
            "delta": b1_metrics["final_score"] - base_metrics["final_score"],
            "spans_changed": total_b1_changed,
            "recovery": b1_recovery,
        },
        "challenger_b2_length_norm": {
            "f02": b2_metrics["final_score"],
            "delta": b2_metrics["final_score"] - base_metrics["final_score"],
            "cat1_f02": b2_metrics["cat1_score"],
            "cat2_f02": b2_metrics["cat2_score"],
            "precision": b2_metrics["precision"],
            "recall": b2_metrics["recall"],
            "tp": b2_metrics["tp"],
            "fp": b2_metrics["fp"],
            "fn": b2_metrics["fn"],
            "total_preds": b2_metrics["total_preds"],
            "spans_changed": total_b2_changed,
            "recovery": b2_recovery,
        },
    }

    out_file = CACHE_DIR / "phase_31a_challenger_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nEvaluation summary saved to {out_file}")

if __name__ == "__main__":
    main()
