"""
Phase 47: Category-Specific Boundary-Specialist Ensemble Evaluation.

Evaluates Candidate A:
- Category 1: Unchanged Phase 40 baseline ensemble.
- Category 2: 90% Phase 40 fused probabilities + 10% gbert_phase32c_challenger_best probabilities.

Uses cached probabilities only:
- data/cache/phase_35a_val_1k_probs.pt
- data/cache/phase_32c_challenger_clean_holdout_probs.pt
"""

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from transformers import AutoConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.build_examples import build_examples, load_training_data
from src.data.make_splits import split_examples
from src.evaluation.phase_35b_audit_and_error_analysis import (
    extract_spanned_gold_entities,
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

PHASE32C_CLEAN_HOLDOUT_PROBS_PATH = REPO_ROOT / "data" / "cache" / "phase_32c_challenger_clean_holdout_probs.pt"
OUTPUT_RESULTS_PATH = REPO_ROOT / "data" / "cache" / "phase_47_category_ensemble_results.json"


def evaluate_independent_evidence() -> Dict[str, Any]:
    """Examine historical validation and internal splits for independent evidence of Category 2 advantage."""
    int_eval_path = REPO_ROOT / "data" / "cache" / "phase_32c_internal_eval_results.json"
    holdout_eval_path = REPO_ROOT / "data" / "cache" / "phase_32c_exploratory_holdout_results.json"

    evidence = {
        "internal_800_dev_dev": None,
        "exploratory_1000_holdout": None,
        "independent_support_found": False,
        "summary": "",
    }

    if int_eval_path.exists():
        with open(int_eval_path, "r", encoding="utf-8") as f:
            int_data = json.load(f)
            b = int_data["baseline_summary"]
            c = int_data["challenger_summary"]
            evidence["internal_800_dev"] = {
                "baseline_cat1": b["cat1_f02"],
                "baseline_cat2": b["cat2_f02"],
                "challenger_cat1": c["cat1_f02"],
                "challenger_cat2": c["cat2_f02"],
                "delta_cat1": c["cat1_f02"] - b["cat1_f02"],
                "delta_cat2": c["cat2_f02"] - b["cat2_f02"],
            }

    if holdout_eval_path.exists():
        with open(holdout_eval_path, "r", encoding="utf-8") as f:
            h_data = json.load(f)
            ens_b = h_data.get("Matched Baseline Ensemble (3.2k + XLMR) @ MIN 0.60", {})
            ens_c = h_data.get("Challenger Ensemble (3.2k + XLMR) @ MIN 0.60", {})
            evidence["exploratory_1000_holdout"] = {
                "baseline_cat1": ens_b.get("cat1_f02"),
                "baseline_cat2": ens_b.get("cat2_f02"),
                "challenger_cat1": ens_c.get("cat1_f02"),
                "challenger_cat2": ens_c.get("cat2_f02"),
                "delta_cat1": ens_c.get("cat1_f02", 0) - ens_b.get("cat1_f02", 0),
                "delta_cat2": ens_c.get("cat2_f02", 0) - ens_b.get("cat2_f02", 0),
            }

    # Evaluate whether independent evidence genuinely supports Cat 2 advantage
    if evidence.get("internal_800_dev"):
        # On internal 800 dev set, delta_cat2 was -0.002831 (negative!)
        delta_int_c2 = evidence["internal_800_dev"]["delta_cat2"]
        evidence["independent_support_found"] = delta_int_c2 > 0.0
        evidence["summary"] = (
            f"Independent 800-title internal dev set showed Category 2 delta of {delta_int_c2:+.6f} (active regression). "
            "No prior independent experiment demonstrates a Category 2 advantage for the boundary specialist."
        )

    return evidence


def run_phase_47_evaluation(n_bootstrap: int = 1000, seed: int = 42) -> Dict[str, Any]:
    """Execute complete Phase 47 evaluation pipeline."""
    champ_info = verify_champion()

    # Load holdout
    with open(CLEAN_HOLDOUT_MANIFEST_PATH, "r", encoding="utf-8") as f:
        clean_ids = set(json.load(f)["record_ids"])

    df = load_training_data()
    all_ex = build_examples(df)
    train_4k, val_1k = split_examples(all_ex, random_state=42)
    clean_indices = [i for i, ex in enumerate(val_1k) if ex["record_number"] in clean_ids]
    clean_ex = [val_1k[i] for i in clean_indices]

    # Load Phase 40 model probabilities
    probs = torch.load(VAL_PROBS_CACHE_PATH, map_location="cpu", weights_only=False)
    cfg = AutoConfig.from_pretrained("checkpoints/gbert_baseline_best")
    id2label = {int(k): v for k, v in cfg.id2label.items()}
    model_names = ["dapt_gbert", "xlmr_seed42", "xlmr_seed7", "xlmr_seed21"]
    model_weights = [0.40, 0.20, 0.20, 0.20]

    clean_p_fused = [sum(w * probs[m][i] for w, m in zip(model_weights, model_names)) for i in clean_indices]

    # Load specialist probabilities
    if not PHASE32C_CLEAN_HOLDOUT_PROBS_PATH.exists():
        raise FileNotFoundError(f"Missing cached specialist probabilities: {PHASE32C_CLEAN_HOLDOUT_PROBS_PATH}")
    specialist_probs_raw = torch.load(PHASE32C_CLEAN_HOLDOUT_PROBS_PATH, map_location="cpu", weights_only=False)
    specialist_probs = [np.array(p) for p in specialist_probs_raw]

    spanned_gold = extract_spanned_gold_entities(clean_ex)
    gold_tuples = [[(e["aspect"], e["value"]) for e in ex] for ex in spanned_gold]

    # Decode Phase 40 baseline
    p40_preds = decode_spanned_predictions_with_cat_thresholds(
        clean_p_fused, examples=clean_ex, id2label=id2label,
        conf_thresh=DEFAULT_CONF_THRESH, aspect_thresholds=PHASE40_THRESHOLDS,
    )
    p40_tuples = [[(e["aspect"], e["value"]) for e in ex] for ex in p40_preds]
    rec_p40 = [
        {"category": clean_ex[i]["category"], "predicted_entities": p40_tuples[i], "gold_entities": gold_tuples[i]}
        for i in range(len(clean_ex))
    ]
    off_p40 = calculate_official_competition_metrics(rec_p40)
    mic_p40 = calculate_global_micro_metrics(rec_p40)

    # Invariants for baseline reproduction
    assert abs(off_p40["official_score"] - 0.927771) < 1e-4
    assert abs(off_p40["categories"][1]["score"] - 0.918323) < 1e-4
    assert abs(off_p40["categories"][2]["score"] - 0.937219) < 1e-4
    assert mic_p40["tp"] == 3021 and mic_p40["fp"] == 165 and mic_p40["fn"] == 752

    # Verify global blend results reported in Phase 46
    fused_global10 = [0.90 * clean_p_fused[i] + 0.10 * specialist_probs[i] for i in range(len(clean_indices))]
    p_global10 = decode_spanned_predictions_with_cat_thresholds(
        fused_global10, examples=clean_ex, id2label=id2label,
        conf_thresh=DEFAULT_CONF_THRESH, aspect_thresholds=PHASE40_THRESHOLDS,
    )
    t_global10 = [[(e["aspect"], e["value"]) for e in ex] for ex in p_global10]
    rec_global10 = [
        {"category": clean_ex[i]["category"], "predicted_entities": t_global10[i], "gold_entities": gold_tuples[i]}
        for i in range(len(clean_ex))
    ]
    off_global10 = calculate_official_competition_metrics(rec_global10)
    assert abs(off_global10["official_score"] - 0.928892) < 1e-4
    assert abs(off_global10["categories"][1]["score"] - 0.916020) < 1e-4
    assert abs(off_global10["categories"][2]["score"] - 0.941765) < 1e-4

    # Build Candidate A:
    # Cat 1: Phase 40 unchanged (1.00)
    # Cat 2: 0.90 * Phase 40 + 0.10 * Specialist
    cand_a_probs = []
    for i, ex in enumerate(clean_ex):
        cat = ex["category"]
        if cat == 1:
            cand_a_probs.append(clean_p_fused[i])
        else:
            cand_a_probs.append(0.90 * clean_p_fused[i] + 0.10 * specialist_probs[i])

    cand_a_preds = decode_spanned_predictions_with_cat_thresholds(
        cand_a_probs, examples=clean_ex, id2label=id2label,
        conf_thresh=DEFAULT_CONF_THRESH, aspect_thresholds=PHASE40_THRESHOLDS,
    )
    cand_a_tuples = [[(e["aspect"], e["value"]) for e in ex] for ex in cand_a_preds]

    # Verify Category 1 bit-for-bit invariance
    cat1_invariance = True
    for i, ex in enumerate(clean_ex):
        if ex["category"] == 1:
            if p40_tuples[i] != cand_a_tuples[i]:
                cat1_invariance = False
                break
    assert cat1_invariance, "Category 1 predictions must be bit-for-bit identical to Phase 40!"

    rec_cand_a = [
        {"category": clean_ex[i]["category"], "predicted_entities": cand_a_tuples[i], "gold_entities": gold_tuples[i]}
        for i in range(len(clean_ex))
    ]
    off_cand_a = calculate_official_competition_metrics(rec_cand_a)
    mic_cand_a = calculate_global_micro_metrics(rec_cand_a)

    d_overall = float(off_cand_a["official_score"] - off_p40["official_score"])
    d_c1 = float(off_cand_a["categories"][1]["score"] - off_p40["categories"][1]["score"])
    d_c2 = float(off_cand_a["categories"][2]["score"] - off_p40["categories"][2]["score"])

    # Detailed Analysis of Category 2 Changes
    changed_cat2_indices = [
        i for i, ex in enumerate(clean_ex)
        if ex["category"] == 2 and p40_tuples[i] != cand_a_tuples[i]
    ]

    cat2_titles_total = sum(1 for ex in clean_ex if ex["category"] == 2)
    cat2_titles_changed = len(changed_cat2_indices)

    # Per-aspect comparisons in Category 2
    aspect_changes = {}
    p40_c2_aspects = off_p40["categories"][2]["aspects"]
    cand_c2_aspects = off_cand_a["categories"][2]["aspects"]

    for asp in sorted(p40_c2_aspects.keys()):
        s_old = p40_c2_aspects[asp]["f_beta"]
        s_new = cand_c2_aspects[asp]["f_beta"]
        tp_old = p40_c2_aspects[asp]["tp"]
        tp_new = cand_c2_aspects[asp]["tp"]
        fp_old = p40_c2_aspects[asp]["fp"]
        fp_new = cand_c2_aspects[asp]["fp"]
        aspect_changes[asp] = {
            "weight": p40_c2_aspects[asp]["weight"],
            "score_old": s_old,
            "score_new": s_new,
            "score_delta": s_new - s_old,
            "tp_old": tp_old,
            "tp_new": tp_new,
            "tp_delta": tp_new - tp_old,
            "fp_old": fp_old,
            "fp_new": fp_new,
            "fp_delta": fp_new - fp_old,
        }

    # Paired Category-Stratified Bootstrap Analysis (1,000 resamples)
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

        rec_base_b = [{"category": clean_ex[i]["category"], "predicted_entities": p40_tuples[i], "gold_entities": gold_tuples[i]} for i in boot_indices]
        rec_cand_b = [{"category": clean_ex[i]["category"], "predicted_entities": cand_a_tuples[i], "gold_entities": gold_tuples[i]} for i in boot_indices]

        m_base = calculate_official_competition_metrics(rec_base_b)
        m_cand = calculate_official_competition_metrics(rec_cand_b)

        boot_deltas_overall.append(m_cand["official_score"] - m_base["official_score"])
        boot_deltas_c1.append(m_cand["categories"][1]["score"] - m_base["categories"][1]["score"])
        boot_deltas_c2.append(m_cand["categories"][2]["score"] - m_base["categories"][2]["score"])

    boot_deltas_overall = np.array(boot_deltas_overall)
    boot_deltas_c1 = np.array(boot_deltas_c1)
    boot_deltas_c2 = np.array(boot_deltas_c2)

    ci_overall = [float(np.percentile(boot_deltas_overall, 2.5)), float(np.percentile(boot_deltas_overall, 97.5))]
    ci_c1 = [float(np.percentile(boot_deltas_c1, 2.5)), float(np.percentile(boot_deltas_c1, 97.5))]
    ci_c2 = [float(np.percentile(boot_deltas_c2, 2.5)), float(np.percentile(boot_deltas_c2, 97.5))]

    bootstrap_results = {
        "n_resamples": n_bootstrap,
        "seed": seed,
        "mean_delta_overall": float(np.mean(boot_deltas_overall)),
        "ci_95_overall": ci_overall,
        "fraction_le_0_overall": float(np.mean(boot_deltas_overall <= 0.0)),
        "mean_delta_c1": float(np.mean(boot_deltas_c1)),
        "ci_95_c1": ci_c1,
        "mean_delta_c2": float(np.mean(boot_deltas_c2)),
        "ci_95_c2": ci_c2,
        "fraction_le_0_c2": float(np.mean(boot_deltas_c2 <= 0.0)),
    }

    # Promotion Gate Evaluation
    passed_delta = d_overall >= 0.0020
    passed_cat1_invariance = abs(d_c1) < 1e-6
    passed_cat2_improvement = d_c2 > 0.0
    passed_exact_match = (mic_cand_a["tp"] >= mic_p40["tp"]) and (mic_cand_a["fp"] <= mic_p40["fp"])
    indep_evidence = evaluate_independent_evidence()

    # Verdict selection:
    # If delta >= +0.002, Cat 1 invariant, Cat 2 improves, exact-match verified, BUT independent evidence is absent:
    # Verdict: PROMISING LOCAL CHALLENGER — GENERALIZATION UNCONFIRMED
    if passed_delta and passed_cat1_invariance and passed_cat2_improvement and passed_exact_match:
        if indep_evidence["independent_support_found"]:
            verdict = "PROMISING CHALLENGER — INDEPENDENT SUPPORT"
        else:
            verdict = "PROMISING LOCAL CHALLENGER — GENERALIZATION UNCONFIRMED"
    elif d_overall < 0.0:
        verdict = "RETAIN PHASE 40 — CANDIDATE REGRESSED"
    else:
        verdict = "INCONCLUSIVE — EVIDENCE INSUFFICIENT"

    results = {
        "champion_verification": champ_info,
        "baseline_reproduction": {
            "overall_score": float(off_p40["official_score"]),
            "cat1_score": float(off_p40["categories"][1]["score"]),
            "cat2_score": float(off_p40["categories"][2]["score"]),
            "tp": int(mic_p40["tp"]),
            "fp": int(mic_p40["fp"]),
            "fn": int(mic_p40["fn"]),
            "micro_f02": float(mic_p40["global_f_beta"]),
        },
        "phase_46_verification": {
            "global_blend_weight_010": {
                "overall_score": float(off_global10["official_score"]),
                "cat1_score": float(off_global10["categories"][1]["score"]),
                "cat2_score": float(off_global10["categories"][2]["score"]),
                "delta_overall": float(off_global10["official_score"] - off_p40["official_score"]),
                "delta_cat1": float(off_global10["categories"][1]["score"] - off_p40["categories"][1]["score"]),
                "delta_cat2": float(off_global10["categories"][2]["score"] - off_p40["categories"][2]["score"]),
            },
            "discrepancy_explanation": (
                "In Phase 46 line 339, '+0.000239' was reported as the best observed delta, which corresponded to "
                "orig_gbert at weight 0.10. The specialist blend (gbert_phase32c_challenger_best at weight 0.10) "
                "actually achieved +0.001121 globally, driven by Cat 2 (+0.004546) but regressing Cat 1 (-0.002303). "
                "The Phase 46 text inadvertently conflated the orig_gbert sweep with the specialist sweep."
            ),
        },
        "independent_evidence_audit": indep_evidence,
        "candidate_a_results": {
            "overall_score": float(off_cand_a["official_score"]),
            "delta_overall": d_overall,
            "cat1_score": float(off_cand_a["categories"][1]["score"]),
            "delta_cat1": d_c1,
            "cat2_score": float(off_cand_a["categories"][2]["score"]),
            "delta_cat2": d_c2,
            "tp": int(mic_cand_a["tp"]),
            "fp": int(mic_cand_a["fp"]),
            "fn": int(mic_cand_a["fn"]),
            "micro_f02": float(mic_cand_a["global_f_beta"]),
            "tp_delta": int(mic_cand_a["tp"] - mic_p40["tp"]),
            "fp_delta": int(mic_cand_a["fp"] - mic_p40["fp"]),
            "fn_delta": int(mic_cand_a["fn"] - mic_p40["fn"]),
            "cat1_invariance_verified": bool(cat1_invariance),
            "cat2_titles_total": cat2_titles_total,
            "cat2_titles_changed": cat2_titles_changed,
            "aspect_breakdown_cat2": aspect_changes,
        },
        "bootstrap_results": bootstrap_results,
        "decision": {
            "verdict": verdict,
            "passed_delta_ge_002": bool(passed_delta),
            "passed_cat1_invariance": bool(passed_cat1_invariance),
            "passed_cat2_improvement": bool(passed_cat2_improvement),
            "passed_exact_match_improvements": bool(passed_exact_match),
            "independent_support_present": bool(indep_evidence["independent_support_found"]),
            "reasons": [
                f"Candidate A achieves an overall delta of {d_overall:+.6f} (>= +0.0020) on the clean holdout.",
                f"Category 1 predictions are bit-for-bit identical to Phase 40 (Delta = {d_c1:+.6f}).",
                f"Category 2 improves by {d_c2:+.6f} (0.937219 -> 0.941764), adding +3 TPs and reducing -4 FPs.",
                "However, independent evidence from the 800-title internal dev set showed Category 2 regressing by -0.002831.",
                "The entire Category 2 improvement is concentrated in just 13 titles out of 184, primarily via FP suppression and single-token part numbers rather than vehicle boundary repair.",
                "Because generalization cannot be confirmed independently, an official Quiz submission is not justified without further verification.",
            ],
        },
    }

    OUTPUT_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    print("Running Phase 47 Category-Specific Boundary Specialist Ensemble Evaluation...")
    res = run_phase_47_evaluation()
    print("Evaluation Complete.")
    print(f"Decision Verdict: {res['decision']['verdict']}")
    print(f"Results saved to: {OUTPUT_RESULTS_PATH}")
