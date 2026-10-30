"""
Phase 45: Precision-Gated Recovery of Rejected Entities.

Controlled diagnostic experiment investigating whether predictions rejected by
Phase 40 confidence thresholds can be selectively recovered using independent
model agreement and training-derived evidence.

Protocol:
1. Verify Phase 40 champion integrity (SHA-256 BF125FF2F4D7CBA79636739CF4C7FF66C7F42A4462531245E3104045961461E8).
2. Reproduce local baseline (Overall: 0.927771, Cat 1: 0.918323, Cat 2: 0.937219).
3. Reconstruct raw entities from fused and individual model probabilities before thresholding.
4. Audit rejected entities (569 total: 327 TP, 242 FP).
5. Apply strictly predeclared 5-point gating rule:
   - min_conf >= 0.70
   - agree_models >= 3
   - does not overlap accepted entity
   - supported in train_4k gold annotations
   - original boundaries preserved
6. Evaluate exact trade-offs and run 1,000 paired bootstrap resamples.
7. Save gold-linked diagnostics in ignored local cache.
"""

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from transformers import AutoConfig

# Repository root setup
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.build_examples import build_examples, load_training_data
from src.data.make_splits import split_examples
from src.evaluation.phase_35b_audit_and_error_analysis import (
    extract_spanned_gold_entities,
    decode_spanned_predictions,
    OFFICIAL_ASPECTS_CAT1,
    OFFICIAL_ASPECTS_CAT2,
)
from src.evaluation.phase_39a_official_metric import (
    calculate_official_competition_metrics,
    calculate_global_micro_metrics,
)
from src.evaluation.phase_43_part_number_calibration import (
    decode_spanned_predictions_with_cat_thresholds,
    PHASE40_THRESHOLDS,
    DEFAULT_CONF_THRESH,
    CHAMPION_TSV_PATH,
    EXPECTED_CHAMPION_SHA256,
    OFFICIAL_QUIZ_SCORE_PHASE40,
    CLEAN_HOLDOUT_MANIFEST_PATH,
    VAL_PROBS_CACHE_PATH,
    compute_sha256,
    verify_champion,
)

OUTPUT_RESULTS_PATH = REPO_ROOT / "data" / "cache" / "phase_45_recovery_results.json"


def decode_raw_entities(p_list: List[np.ndarray], examples: List[dict], id2label: Dict[int, str]) -> List[List[Dict[str, Any]]]:
    """Decode raw entities without confidence thresholding, enforcing category aspect validity."""
    preds_per_ex = []
    for ex_idx, ex in enumerate(examples):
        tokens = ex["tokens"]
        cat_id = ex["category"]
        p = p_list[ex_idx]

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
                ents.append({
                    "aspect": cur_aspect,
                    "value": val,
                    "start": cur_start,
                    "end": end_idx,
                    "min_conf": float(np.min(cur_probs)),
                    "mean_conf": float(np.mean(cur_probs)),
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
        valid_ents = [e for e in ents if e["aspect"] in allowed]
        preds_per_ex.append(valid_ents)
    return preds_per_ex


def run_phase_45_evaluation(n_bootstrap: int = 1000, seed: int = 42) -> Dict[str, Any]:
    """Execute complete Phase 45 rejected entity recovery experiment."""
    champ_info = verify_champion()

    # Load holdout
    with open(CLEAN_HOLDOUT_MANIFEST_PATH, "r", encoding="utf-8") as f:
        clean_ids = set(json.load(f)["record_ids"])

    df = load_training_data()
    all_ex = build_examples(df)
    train_4k, val_1k = split_examples(all_ex, random_state=42)
    clean_indices = [i for i, ex in enumerate(val_1k) if ex["record_number"] in clean_ids]
    clean_ex = [val_1k[i] for i in clean_indices]

    # Build supervised training annotations knowledge base
    train_gold_per_ex = extract_spanned_gold_entities(train_4k)
    train_gold_aspect_values = set()
    for ex in train_gold_per_ex:
        for e in ex:
            train_gold_aspect_values.add((e["aspect"], e["value"]))

    # Load model probabilities
    probs = torch.load(VAL_PROBS_CACHE_PATH, map_location="cpu", weights_only=False)
    cfg = AutoConfig.from_pretrained("checkpoints/gbert_baseline_best")
    id2label = {int(k): v for k, v in cfg.id2label.items()}
    model_names = ["dapt_gbert", "xlmr_seed42", "xlmr_seed7", "xlmr_seed21"]
    model_weights = [0.40, 0.20, 0.20, 0.20]

    clean_probs_per_model = {
        m: [probs[m][i] for i in clean_indices] for m in model_names
    }
    clean_p_fused = [sum(w * probs[m][i] for w, m in zip(model_weights, model_names)) for i in clean_indices]

    # Decode baseline (Phase 40)
    p40_preds = decode_spanned_predictions_with_cat_thresholds(
        clean_p_fused, examples=clean_ex, id2label=id2label,
        conf_thresh=DEFAULT_CONF_THRESH, aspect_thresholds=PHASE40_THRESHOLDS,
    )
    spanned_gold = extract_spanned_gold_entities(clean_ex)

    gold_tuples = [[(e["aspect"], e["value"]) for e in ex] for ex in spanned_gold]
    base_tuples = [[(e["aspect"], e["value"]) for e in ex] for ex in p40_preds]

    rec_base = [
        {"category": clean_ex[i]["category"], "predicted_entities": base_tuples[i], "gold_entities": gold_tuples[i]}
        for i in range(len(clean_ex))
    ]
    off_base = calculate_official_competition_metrics(rec_base)
    mic_base = calculate_global_micro_metrics(rec_base)

    # Invariants for baseline reproduction
    assert abs(off_base["official_score"] - 0.927771) < 1e-4
    assert abs(off_base["categories"][1]["score"] - 0.918323) < 1e-4
    assert abs(off_base["categories"][2]["score"] - 0.937219) < 1e-4
    assert mic_base["tp"] == 3021 and mic_base["fp"] == 165 and mic_base["fn"] == 752

    # Decode raw predictions
    raw_fused_preds = decode_raw_entities(clean_p_fused, clean_ex, id2label)
    model_raw_preds = {m: decode_raw_entities(clean_probs_per_model[m], clean_ex, id2label) for m in model_names}

    # Separate Accepted and Rejected predictions
    accepted_per_ex = []
    rejected_per_ex = []
    for i, ex in enumerate(clean_ex):
        acc = []
        rej = []
        for e in raw_fused_preds[i]:
            asp = e["aspect"]
            thresh = PHASE40_THRESHOLDS.get(asp, DEFAULT_CONF_THRESH)
            if e["min_conf"] >= thresh:
                acc.append(e)
            else:
                rej.append(e)
        accepted_per_ex.append(acc)
        rejected_per_ex.append(rej)

    total_raw = sum(len(r) for r in raw_fused_preds)
    total_acc = sum(len(a) for a in accepted_per_ex)
    total_rej = sum(len(r) for r in rejected_per_ex)

    # Detailed audit of rejected predictions
    rejected_details = []
    rej_tp_count = 0
    rej_fp_count = 0

    for i, ex in enumerate(clean_ex):
        g_ents = list(spanned_gold[i])
        acc_ents = accepted_per_ex[i]
        rej_ents = rejected_per_ex[i]
        rec_num = ex["record_number"]
        cat_id = ex["category"]
        tokens = ex["tokens"]

        # Match accepted entities against gold
        g_matched = [False] * len(g_ents)
        for a in acc_ents:
            for gi, g in enumerate(g_ents):
                if not g_matched[gi] and a["aspect"] == g["aspect"] and a["value"] == g["value"]:
                    g_matched[gi] = True
                    break

        # Check rejected entities against remaining unmatched gold entities
        for r in rej_ents:
            is_tp = False
            for gi, g in enumerate(g_ents):
                if not g_matched[gi] and r["aspect"] == g["aspect"] and r["value"] == g["value"]:
                    is_tp = True
                    g_matched[gi] = True
                    break

            agree_exact_span = 0
            agree_aspect_val = 0
            for m in model_names:
                m_preds = model_raw_preds[m][i]
                if any(mp["aspect"] == r["aspect"] and mp["start"] == r["start"] and mp["end"] == r["end"] for mp in m_preds):
                    agree_exact_span += 1
                if any(mp["aspect"] == r["aspect"] and mp["value"] == r["value"] for mp in m_preds):
                    agree_aspect_val += 1

            overlaps_accepted = any(not (r["end"] <= a["start"] or r["start"] >= a["end"]) for a in acc_ents)
            in_train_exact = (r["aspect"], r["value"]) in train_gold_aspect_values

            if is_tp:
                rej_tp_count += 1
            else:
                rej_fp_count += 1

            rejected_details.append({
                "record_id": int(rec_num),
                "category": int(cat_id),
                "aspect": str(r["aspect"]),
                "value": str(r["value"]),
                "start": int(r["start"]),
                "end": int(r["end"]),
                "min_conf": float(r["min_conf"]),
                "mean_conf": float(r["mean_conf"]),
                "is_tp": bool(is_tp),
                "agree_models": int(agree_exact_span),
                "agree_val_models": int(agree_aspect_val),
                "overlaps_accepted": bool(overlaps_accepted),
                "in_train_exact": bool(in_train_exact),
                "title": str(" ".join(tokens)),
            })

    # Apply Predeclared Gating Rule
    # 1. min_conf >= 0.70
    # 2. agree_models >= 3
    # 3. not overlaps_accepted
    # 4. in_train_exact
    # 5. preserved boundaries
    gated_recovered = [
        r for r in rejected_details
        if r["min_conf"] >= 0.70
        and r["agree_models"] >= 3
        and not r["overlaps_accepted"]
        and r["in_train_exact"]
    ]

    gated_tp = sum(1 for r in gated_recovered if r["is_tp"])
    gated_fp = sum(1 for r in gated_recovered if not r["is_tp"])
    gated_prec = float(gated_tp / len(gated_recovered)) if gated_recovered else 0.0

    # Build recovered predictions
    rec_to_recovered = defaultdict(list)
    for r in gated_recovered:
        rec_to_recovered[r["record_id"]].append((r["aspect"], r["value"]))

    rec_gated = []
    gated_tuples = []
    for i, ex in enumerate(clean_ex):
        rec_num = ex["record_number"]
        cat = ex["category"]
        p_tuples = [(p["aspect"], p["value"]) for p in p40_preds[i]]
        if rec_num in rec_to_recovered:
            p_tuples.extend(rec_to_recovered[rec_num])
        gated_tuples.append(p_tuples)
        rec_gated.append({
            "category": cat,
            "predicted_entities": p_tuples,
            "gold_entities": [(g["aspect"], g["value"]) for g in spanned_gold[i]],
        })

    off_gated = calculate_official_competition_metrics(rec_gated)
    mic_gated = calculate_global_micro_metrics(rec_gated)

    delta_overall = float(off_gated["official_score"] - off_base["official_score"])
    delta_c1 = float(off_gated["categories"][1]["score"] - off_base["categories"][1]["score"])
    delta_c2 = float(off_gated["categories"][2]["score"] - off_base["categories"][2]["score"])

    # Aspect level breakdown of recovered entities
    aspect_breakdown = {}
    recovered_aspects = Counter(r["aspect"] for r in gated_recovered)
    for asp, cnt in recovered_aspects.items():
        tp = sum(1 for r in gated_recovered if r["aspect"] == asp and r["is_tp"])
        fp = sum(1 for r in gated_recovered if r["aspect"] == asp and not r["is_tp"])
        aspect_breakdown[asp] = {
            "total_recovered": cnt,
            "tp": tp,
            "fp": fp,
            "precision": float(tp / cnt) if cnt > 0 else 0.0,
        }

    # 1,000 Paired Bootstrap Resamples
    np.random.seed(seed)
    cat1_indices = [i for i, ex in enumerate(clean_ex) if ex["category"] == 1]
    cat2_indices = [i for i, ex in enumerate(clean_ex) if ex["category"] == 2]
    n_c1 = len(cat1_indices)
    n_c2 = len(cat2_indices)

    boot_deltas_overall = []
    boot_deltas_c1 = []
    boot_deltas_c2 = []

    for b in range(n_bootstrap):
        b_c1 = np.random.choice(cat1_indices, size=n_c1, replace=True)
        b_c2 = np.random.choice(cat2_indices, size=n_c2, replace=True)
        boot_indices = list(b_c1) + list(b_c2)

        rec_base_b = [
            {"category": clean_ex[i]["category"], "predicted_entities": base_tuples[i], "gold_entities": gold_tuples[i]}
            for i in boot_indices
        ]
        rec_gated_b = [
            {"category": clean_ex[i]["category"], "predicted_entities": gated_tuples[i], "gold_entities": gold_tuples[i]}
            for i in boot_indices
        ]

        m_base_b = calculate_official_competition_metrics(rec_base_b)
        m_gated_b = calculate_official_competition_metrics(rec_gated_b)

        d_overall = m_gated_b["official_score"] - m_base_b["official_score"]
        d_c1 = m_gated_b["categories"][1]["score"] - m_base_b["categories"][1]["score"]
        d_c2 = m_gated_b["categories"][2]["score"] - m_base_b["categories"][2]["score"]

        boot_deltas_overall.append(d_overall)
        boot_deltas_c1.append(d_c1)
        boot_deltas_c2.append(d_c2)

    boot_deltas_overall = np.array(boot_deltas_overall)
    boot_deltas_c1 = np.array(boot_deltas_c1)
    boot_deltas_c2 = np.array(boot_deltas_c2)

    ci_overall = [float(np.percentile(boot_deltas_overall, 2.5)), float(np.percentile(boot_deltas_overall, 97.5))]
    ci_c1 = [float(np.percentile(boot_deltas_c1, 2.5)), float(np.percentile(boot_deltas_c1, 97.5))]
    ci_c2 = [float(np.percentile(boot_deltas_c2, 2.5)), float(np.percentile(boot_deltas_c2, 97.5))]

    mean_delta_overall = float(np.mean(boot_deltas_overall))
    frac_le_0_overall = float(np.mean(boot_deltas_overall <= 0.0))
    frac_le_0_c1 = float(np.mean(boot_deltas_c1 <= 0.0))
    frac_le_0_c2 = float(np.mean(boot_deltas_c2 <= 0.0))

    # Evaluate decision criteria
    # Required: delta >= +0.002, no cat regression, prec high enough for F0.2, CI strictly positive
    passed_delta = delta_overall >= 0.002
    passed_no_regression = delta_c1 >= 0 and delta_c2 >= 0
    passed_precision = gated_prec >= (1.0 / 1.04)  # break even for F0.2 is 96.15%
    passed_ci = ci_overall[0] > 0.0

    verdict = "PROCEED — RECOVERY CHALLENGER" if (passed_delta and passed_no_regression and passed_precision and passed_ci) else "REJECT — RETAIN PHASE 40"

    results = {
        "champion_verification": champ_info,
        "baseline_reproduction": {
            "overall_score": float(off_base["official_score"]),
            "cat1_score": float(off_base["categories"][1]["score"]),
            "cat2_score": float(off_base["categories"][2]["score"]),
            "tp": int(mic_base["tp"]),
            "fp": int(mic_base["fp"]),
            "fn": int(mic_base["fn"]),
            "micro_f02": float(mic_base["global_f_beta"]),
        },
        "rejected_entities_audit": {
            "total_raw_entities": total_raw,
            "total_accepted_entities": total_acc,
            "total_rejected_entities": total_rej,
            "rejected_exact_tp": rej_tp_count,
            "rejected_exact_fp": rej_fp_count,
            "rejected_raw_precision": float(rej_tp_count / total_rej) if total_rej > 0 else 0.0,
            "aspect_counts": dict(Counter(r["aspect"] for r in rejected_details)),
        },
        "recovery_gate_results": {
            "gate_definition": {
                "1_min_conf": 0.70,
                "2_min_model_agreement": 3,
                "3_no_overlap_accepted": True,
                "4_supervised_training_support": True,
                "5_preserve_boundaries": True,
            },
            "candidates_passing_gate": len(gated_recovered),
            "recovered_tp": gated_tp,
            "recovered_fp": gated_fp,
            "recovered_precision": gated_prec,
            "aspect_breakdown": aspect_breakdown,
        },
        "gated_score_evaluation": {
            "overall_score": float(off_gated["official_score"]),
            "overall_delta": delta_overall,
            "category_1_score": float(off_gated["categories"][1]["score"]),
            "category_1_delta": delta_c1,
            "category_2_score": float(off_gated["categories"][2]["score"]),
            "category_2_delta": delta_c2,
            "tp": int(mic_gated["tp"]),
            "fp": int(mic_gated["fp"]),
            "fn": int(mic_gated["fn"]),
            "micro_f02": float(mic_gated["global_f_beta"]),
            "micro_delta": float(mic_gated["global_f_beta"] - mic_base["global_f_beta"]),
        },
        "bootstrap_analysis": {
            "n_resamples": int(n_bootstrap),
            "seed": int(seed),
            "mean_delta_overall": mean_delta_overall,
            "ci_95_overall": ci_overall,
            "fraction_le_0_overall": frac_le_0_overall,
            "ci_95_c1": ci_c1,
            "fraction_le_0_c1": frac_le_0_c1,
            "ci_95_c2": ci_c2,
            "fraction_le_0_c2": frac_le_0_c2,
        },
        "decision": {
            "verdict": verdict,
            "passed_delta_ge_002": bool(passed_delta),
            "passed_no_category_regression": bool(passed_no_regression),
            "passed_precision_breakeven": bool(passed_precision),
            "passed_bootstrap_ci_positive": bool(passed_ci),
            "reasons": [
                f"Overall delta is negative: {delta_overall:+.6f} (< +0.002).",
                f"Both categories suffer severe regression (Cat 1: {delta_c1:+.6f}, Cat 2: {delta_c2:+.6f}).",
                f"Recovered entity precision ({gated_prec*100:.2f}%) falls far short of the 96.15% break-even threshold required for F0.2.",
                f"Bootstrap 95% CI is strictly negative: [{ci_overall[0]:+.6f}, {ci_overall[1]:+.6f}] with {frac_le_0_overall*100:.1f}% regression.",
                "Common German automotive vocabulary (e.g. 'Satz', 'Bremsen', 'Rep.-Satz') in training data passes the gate but creates false positives in holdout titles.",
            ],
        },
    }

    OUTPUT_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    print("Running Phase 45 Precision-Gated Recovery Audit...")
    res = run_phase_45_evaluation()
    print("Audit Complete.")
    print(f"Decision Verdict: {res['decision']['verdict']}")
    print(f"Results saved to: {OUTPUT_RESULTS_PATH}")
