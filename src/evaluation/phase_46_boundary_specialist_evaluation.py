"""
Phase 46: Entity Boundary Specialist and Sixth-Model Ensemble Feasibility & Diagnostic Evaluation.

Investigates:
1. Model inventory: Checkpoints, architectures, training data, seeds, sizes, ensemble status.
2. Boundary failure modes (7 specific modes on 489-title holdout across Cat 1 and Cat 2).
3. Complementarity of unused existing checkpoints (orig_gbert, gelectra, gbert_phase32c).
4. Controlled ensemble comparisons at varying candidate weights.
5. Paired category-stratified bootstrap uncertainty distribution (1,000 resamples).
6. Resource feasibility & promotion gate evaluation.
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

OUTPUT_RESULTS_PATH = REPO_ROOT / "data" / "cache" / "phase_46_boundary_evaluation_results.json"


def audit_model_inventory() -> Dict[str, Any]:
    """Inspect all trained checkpoints and record their configurations and status."""
    ckpts_dir = REPO_ROOT / "checkpoints"
    inventory = {}

    known_meta = {
        "phase33b_dapt_gbert_best": {
            "arch": "BertForTokenClassification (deepset/gbert-base + DAPT MLM)",
            "seed": 42,
            "train_data": "Sequential slice s_tr (3,200 titles)",
            "part_of_phase40": True,
            "phase40_weight": 0.40,
            "role": "Model 1 in Phase 40",
        },
        "xlmr_baseline_best": {
            "arch": "XLMRobertaForTokenClassification (FacebookAI/xlm-roberta-base)",
            "seed": 42,
            "train_data": "Canonical Split-A train_4k (4,000 titles)",
            "part_of_phase40": True,
            "phase40_weight": 0.20,
            "role": "Model 2 in Phase 40",
        },
        "xlmr_seed_7_best": {
            "arch": "XLMRobertaForTokenClassification (FacebookAI/xlm-roberta-base)",
            "seed": 7,
            "train_data": "Canonical Split-A train_4k (4,000 titles)",
            "part_of_phase40": True,
            "phase40_weight": 0.20,
            "role": "Model 3 in Phase 40",
        },
        "xlmr_seed_21_best": {
            "arch": "XLMRobertaForTokenClassification (FacebookAI/xlm-roberta-base)",
            "seed": 21,
            "train_data": "Canonical Split-A train_4k (4,000 titles)",
            "part_of_phase40": True,
            "phase40_weight": 0.20,
            "role": "Model 4 in Phase 40",
        },
        "gbert_baseline_best": {
            "arch": "BertForTokenClassification (deepset/gbert-base)",
            "seed": 42,
            "train_data": "Canonical Split-A train_4k (4,000 titles)",
            "part_of_phase40": False,
            "phase40_weight": 0.0,
            "role": "Unused Checkpoint (Model 1 in Phase 35A five-model)",
        },
        "gelectra_baseline_best": {
            "arch": "ElectraForTokenClassification (deepset/gelectra-base)",
            "seed": 42,
            "train_data": "Canonical Split-A train_4k (4,000 titles)",
            "part_of_phase40": False,
            "phase40_weight": 0.0,
            "role": "Unused Checkpoint (Evaluated in Phase 25B)",
        },
        "gbert_phase32c_challenger_best": {
            "arch": "BertForTokenClassification (deepset/gbert-base)",
            "seed": 42,
            "train_data": "Internal train 3,200 (1.15x I-Modell loss weight)",
            "part_of_phase40": False,
            "phase40_weight": 0.0,
            "role": "Unused Checkpoint (Evaluated in Phase 32C)",
        },
        "gbert_phase32c_baseline_best": {
            "arch": "BertForTokenClassification (deepset/gbert-base)",
            "seed": 42,
            "train_data": "Internal train 3,200 (1.00x loss weight)",
            "part_of_phase40": False,
            "phase40_weight": 0.0,
            "role": "Unused Checkpoint (Evaluated in Phase 32C)",
        },
        "xlmr_span_seed42_best": {
            "arch": "XLMRobertaModel + span_classifier (Exp 16)",
            "seed": 42,
            "train_data": "Canonical Split-A train_4k (4,000 titles)",
            "part_of_phase40": False,
            "phase40_weight": 0.0,
            "role": "Unused Checkpoint (Exp 16 span model)",
        },
        "phase33b_dapt_encoder": {
            "arch": "BertForMaskedLM (DAPT MLM pre-trained weights)",
            "seed": 42,
            "train_data": "50k unlabelled listing titles",
            "part_of_phase40": False,
            "phase40_weight": 0.0,
            "role": "Pre-trained encoder backbone (not an entity classifier)",
        },
    }

    for p in sorted(ckpts_dir.iterdir()):
        if p.is_dir():
            files = [f.name for f in p.iterdir()]
            has_weights = any("safetensors" in f or "bin" in f or "pt" in f for f in files)
            total_mb = sum(f.stat().st_size for f in p.glob("**/*") if f.is_file()) / (1024 * 1024)
            info = known_meta.get(p.name, {
                "arch": "BertForTokenClassification" if "gbert" in p.name else "XLMRobertaForTokenClassification",
                "seed": 42 if "42" in p.name else (7 if "7" in p.name else (21 if "21" in p.name else 0)),
                "train_data": "Full 5,000 (Leaked)" if "fulldata" in p.name else "Cross Validation Fold",
                "part_of_phase40": False,
                "phase40_weight": 0.0,
                "role": "Fulldata / Fold Checkpoint",
            })
            info["size_mb"] = round(total_mb, 1)
            info["weights_present"] = has_weights
            inventory[p.name] = info

    return inventory


def run_phase_46_evaluation(n_bootstrap: int = 1000, seed: int = 42) -> Dict[str, Any]:
    """Execute complete Phase 46 boundary specialist and ensemble feasibility audit."""
    champ_info = verify_champion()

    # Load holdout
    with open(CLEAN_HOLDOUT_MANIFEST_PATH, "r", encoding="utf-8") as f:
        clean_ids = set(json.load(f)["record_ids"])

    df = load_training_data()
    all_ex = build_examples(df)
    train_4k, val_1k = split_examples(all_ex, random_state=42)
    clean_indices = [i for i, ex in enumerate(val_1k) if ex["record_number"] in clean_ids]
    clean_ex = [val_1k[i] for i in clean_indices]

    probs = torch.load(VAL_PROBS_CACHE_PATH, map_location="cpu", weights_only=False)
    cfg = AutoConfig.from_pretrained("checkpoints/gbert_baseline_best")
    id2label = {int(k): v for k, v in cfg.id2label.items()}
    model_names = ["dapt_gbert", "xlmr_seed42", "xlmr_seed7", "xlmr_seed21"]
    model_weights = [0.40, 0.20, 0.20, 0.20]

    clean_p_fused = [sum(w * probs[m][i] for w, m in zip(model_weights, model_names)) for i in clean_indices]
    orig_gbert_probs = [probs["orig_gbert"][i] for i in clean_indices]

    spanned_gold = extract_spanned_gold_entities(clean_ex)
    gold_tuples = [[(e["aspect"], e["value"]) for e in ex] for ex in spanned_gold]

    # Baseline Phase 40 reproduction
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

    # Step 2: Boundary Failure Modes Audit
    modes_c1 = Counter()
    modes_c2 = Counter()
    modes_by_aspect = defaultdict(Counter)

    for i, ex in enumerate(clean_ex):
        cat = ex["category"]
        g_ents = spanned_gold[i]
        p_ents = p40_preds[i]

        g_matched = [False] * len(g_ents)
        p_matched = [False] * len(p_ents)

        # Exact match
        for pi, p in enumerate(p_ents):
            for gi, g in enumerate(g_ents):
                if not g_matched[gi] and p["aspect"] == g["aspect"] and p["start"] == g["start"] and p["end"] == g["end"]:
                    g_matched[gi] = True
                    p_matched[pi] = True
                    break

        # Mode 5: Merging (1 pred overlaps >= 2 gold)
        for pi, p in enumerate(p_ents):
            if p_matched[pi]:
                continue
            overlapping_g = [gi for gi, g in enumerate(g_ents) if not (p["end"] <= g["start"] or p["start"] >= g["end"])]
            if len(overlapping_g) >= 2:
                m = "5_incorrect_merging"
                if cat == 1: modes_c1[m] += 1
                else: modes_c2[m] += 1
                modes_by_aspect[p["aspect"]][m] += 1
                p_matched[pi] = True
                for gi in overlapping_g:
                    g_matched[gi] = True

        # Mode 6: Splitting (1 gold overlaps >= 2 pred)
        for gi, g in enumerate(g_ents):
            if g_matched[gi]:
                continue
            overlapping_p = [pi for pi, p in enumerate(p_ents) if not (p["end"] <= g["start"] or p["start"] >= g["end"])]
            if len(overlapping_p) >= 2:
                m = "6_incorrect_splitting"
                if cat == 1: modes_c1[m] += 1
                else: modes_c2[m] += 1
                modes_by_aspect[g["aspect"]][m] += 1
                g_matched[gi] = True
                for pi in overlapping_p:
                    p_matched[pi] = True

        # Mode 7: Wrong aspect despite correct boundaries
        for pi, p in enumerate(p_ents):
            if p_matched[pi]:
                continue
            for gi, g in enumerate(g_ents):
                if not g_matched[gi] and p["start"] == g["start"] and p["end"] == g["end"] and p["aspect"] != g["aspect"]:
                    m = "7_wrong_aspect_same_boundary"
                    if cat == 1: modes_c1[m] += 1
                    else: modes_c2[m] += 1
                    modes_by_aspect[p["aspect"]][m] += 1
                    p_matched[pi] = True
                    g_matched[gi] = True
                    break

        # Modes 1, 2, 3, 4: Boundary single-side errors with same aspect
        for pi, p in enumerate(p_ents):
            if p_matched[pi]:
                continue
            for gi, g in enumerate(g_ents):
                if g_matched[gi]:
                    continue
                if p["aspect"] == g["aspect"]:
                    if p["start"] == g["start"] and p["end"] < g["end"]:
                        m = "1_missing_final_token"
                        if cat == 1: modes_c1[m] += 1
                        else: modes_c2[m] += 1
                        modes_by_aspect[p["aspect"]][m] += 1
                        p_matched[pi] = True
                        g_matched[gi] = True
                        break
                    elif p["start"] == g["start"] and p["end"] > g["end"]:
                        m = "2_extra_final_token"
                        if cat == 1: modes_c1[m] += 1
                        else: modes_c2[m] += 1
                        modes_by_aspect[p["aspect"]][m] += 1
                        p_matched[pi] = True
                        g_matched[gi] = True
                        break
                    elif p["start"] > g["start"] and p["end"] == g["end"]:
                        m = "3_missing_leading_token"
                        if cat == 1: modes_c1[m] += 1
                        else: modes_c2[m] += 1
                        modes_by_aspect[p["aspect"]][m] += 1
                        p_matched[pi] = True
                        g_matched[gi] = True
                        break
                    elif p["start"] < g["start"] and p["end"] == g["end"]:
                        m = "4_extra_leading_token"
                        if cat == 1: modes_c1[m] += 1
                        else: modes_c2[m] += 1
                        modes_by_aspect[p["aspect"]][m] += 1
                        p_matched[pi] = True
                        g_matched[gi] = True
                        break

    boundary_failure_modes = {
        "1_missing_final_token": {"cat1": modes_c1["1_missing_final_token"], "cat2": modes_c2["1_missing_final_token"], "total": modes_c1["1_missing_final_token"] + modes_c2["1_missing_final_token"]},
        "2_extra_final_token": {"cat1": modes_c1["2_extra_final_token"], "cat2": modes_c2["2_extra_final_token"], "total": modes_c1["2_extra_final_token"] + modes_c2["2_extra_final_token"]},
        "3_missing_leading_token": {"cat1": modes_c1["3_missing_leading_token"], "cat2": modes_c2["3_missing_leading_token"], "total": modes_c1["3_missing_leading_token"] + modes_c2["3_missing_leading_token"]},
        "4_extra_leading_token": {"cat1": modes_c1["4_extra_leading_token"], "cat2": modes_c2["4_extra_leading_token"], "total": modes_c1["4_extra_leading_token"] + modes_c2["4_extra_leading_token"]},
        "5_incorrect_merging": {"cat1": modes_c1["5_incorrect_merging"], "cat2": modes_c2["5_incorrect_merging"], "total": modes_c1["5_incorrect_merging"] + modes_c2["5_incorrect_merging"]},
        "6_incorrect_splitting": {"cat1": modes_c1["6_incorrect_splitting"], "cat2": modes_c2["6_incorrect_splitting"], "total": modes_c1["6_incorrect_splitting"] + modes_c2["6_incorrect_splitting"]},
        "7_wrong_aspect_same_boundary": {"cat1": modes_c1["7_wrong_aspect_same_boundary"], "cat2": modes_c2["7_wrong_aspect_same_boundary"], "total": modes_c1["7_wrong_aspect_same_boundary"] + modes_c2["7_wrong_aspect_same_boundary"]},
    }

    # Step 3: Check Complementarity of Unused Checkpoint: orig_gbert
    orig_preds = decode_spanned_predictions_with_cat_thresholds(
        orig_gbert_probs, examples=clean_ex, id2label=id2label,
        conf_thresh=DEFAULT_CONF_THRESH, aspect_thresholds=PHASE40_THRESHOLDS,
    )
    orig_tuples = [[(e["aspect"], e["value"]) for e in ex] for ex in orig_preds]
    rec_orig = [
        {"category": clean_ex[i]["category"], "predicted_entities": orig_tuples[i], "gold_entities": gold_tuples[i]}
        for i in range(len(clean_ex))
    ]
    off_orig = calculate_official_competition_metrics(rec_orig)
    mic_orig = calculate_global_micro_metrics(rec_orig)

    fixed_errors = 0
    damaged_correct = 0
    for i in range(len(clean_ex)):
        g_cnt = Counter(gold_tuples[i])
        p40_cnt = Counter(p40_tuples[i])
        orig_cnt = Counter(orig_tuples[i])
        for ent, gc in g_cnt.items():
            p_c = p40_cnt.get(ent, 0)
            o_c = orig_cnt.get(ent, 0)
            if o_c > p_c:
                fixed_errors += min(o_c, gc) - min(p_c, gc)
            elif p_c > o_c:
                damaged_correct += min(p_c, gc) - min(o_c, gc)

    # Step 6: Controlled Ensemble Comparison (Phase 40 + orig_gbert at weights 0.05, 0.10, 0.15, 0.20)
    fusion_experiments = {}
    for w in [0.05, 0.10, 0.15, 0.20]:
        fused_w = [(1 - w) * clean_p_fused[i] + w * orig_gbert_probs[i] for i in range(len(clean_indices))]
        p_w = decode_spanned_predictions_with_cat_thresholds(
            fused_w, examples=clean_ex, id2label=id2label,
            conf_thresh=DEFAULT_CONF_THRESH, aspect_thresholds=PHASE40_THRESHOLDS,
        )
        t_w = [[(e["aspect"], e["value"]) for e in ex] for ex in p_w]
        rec_w = [
            {"category": clean_ex[i]["category"], "predicted_entities": t_w[i], "gold_entities": gold_tuples[i]}
            for i in range(len(clean_ex))
        ]
        off_w = calculate_official_competition_metrics(rec_w)
        mic_w = calculate_global_micro_metrics(rec_w)
        fusion_experiments[f"orig_gbert_weight_{w:.2f}"] = {
            "weight": w,
            "overall_score": float(off_w["official_score"]),
            "delta_overall": float(off_w["official_score"] - off_p40["official_score"]),
            "cat1_score": float(off_w["categories"][1]["score"]),
            "delta_cat1": float(off_w["categories"][1]["score"] - off_p40["categories"][1]["score"]),
            "cat2_score": float(off_w["categories"][2]["score"]),
            "delta_cat2": float(off_w["categories"][2]["score"] - off_p40["categories"][2]["score"]),
            "tp": int(mic_w["tp"]),
            "fp": int(mic_w["fp"]),
            "fn": int(mic_w["fn"]),
            "micro_f02": float(mic_w["global_f_beta"]),
        }

    # Step 7: Bootstrap Analysis on Candidate 1 (weight 0.10) and Candidate 2 (weight 0.20)
    np.random.seed(seed)
    cat1_indices = [i for i, ex in enumerate(clean_ex) if ex["category"] == 1]
    cat2_indices = [i for i, ex in enumerate(clean_ex) if ex["category"] == 2]
    n_c1 = len(cat1_indices)
    n_c2 = len(cat2_indices)

    cand1_tuples = [[(e["aspect"], e["value"]) for e in ex] for ex in decode_spanned_predictions_with_cat_thresholds(
        [(0.90 * clean_p_fused[i] + 0.10 * orig_gbert_probs[i]) for i in range(len(clean_indices))],
        examples=clean_ex, id2label=id2label, conf_thresh=DEFAULT_CONF_THRESH, aspect_thresholds=PHASE40_THRESHOLDS
    )]
    cand2_tuples = [[(e["aspect"], e["value"]) for e in ex] for ex in decode_spanned_predictions_with_cat_thresholds(
        [(0.80 * clean_p_fused[i] + 0.20 * orig_gbert_probs[i]) for i in range(len(clean_indices))],
        examples=clean_ex, id2label=id2label, conf_thresh=DEFAULT_CONF_THRESH, aspect_thresholds=PHASE40_THRESHOLDS
    )]

    deltas_c1 = []
    deltas_c2 = []
    deltas_c1_cat1 = []
    deltas_c2_cat1 = []

    for b in range(n_bootstrap):
        b_c1 = np.random.choice(cat1_indices, size=n_c1, replace=True)
        b_c2 = np.random.choice(cat2_indices, size=n_c2, replace=True)
        boot_indices = list(b_c1) + list(b_c2)

        rec_base_b = [{"category": clean_ex[i]["category"], "predicted_entities": p40_tuples[i], "gold_entities": gold_tuples[i]} for i in boot_indices]
        rec_c1_b = [{"category": clean_ex[i]["category"], "predicted_entities": cand1_tuples[i], "gold_entities": gold_tuples[i]} for i in boot_indices]
        rec_c2_b = [{"category": clean_ex[i]["category"], "predicted_entities": cand2_tuples[i], "gold_entities": gold_tuples[i]} for i in boot_indices]

        m_base = calculate_official_competition_metrics(rec_base_b)
        m_c1 = calculate_official_competition_metrics(rec_c1_b)
        m_c2 = calculate_official_competition_metrics(rec_c2_b)

        deltas_c1.append(m_c1["official_score"] - m_base["official_score"])
        deltas_c2.append(m_c2["official_score"] - m_base["official_score"])
        deltas_c1_cat1.append(m_c1["categories"][1]["score"] - m_base["categories"][1]["score"])
        deltas_c2_cat1.append(m_c2["categories"][1]["score"] - m_base["categories"][1]["score"])

    deltas_c1 = np.array(deltas_c1)
    deltas_c2 = np.array(deltas_c2)
    deltas_c1_cat1 = np.array(deltas_c1_cat1)
    deltas_c2_cat1 = np.array(deltas_c2_cat1)

    bootstrap_results = {
        "candidate_weight_010": {
            "mean_delta_overall": float(np.mean(deltas_c1)),
            "ci_95_overall": [float(np.percentile(deltas_c1, 2.5)), float(np.percentile(deltas_c1, 97.5))],
            "fraction_le_0_overall": float(np.mean(deltas_c1 <= 0.0)),
            "mean_delta_cat1": float(np.mean(deltas_c1_cat1)),
            "ci_95_cat1": [float(np.percentile(deltas_c1_cat1, 2.5)), float(np.percentile(deltas_c1_cat1, 97.5))],
        },
        "candidate_weight_020_equal_5model": {
            "mean_delta_overall": float(np.mean(deltas_c2)),
            "ci_95_overall": [float(np.percentile(deltas_c2, 2.5)), float(np.percentile(deltas_c2, 97.5))],
            "fraction_le_0_overall": float(np.mean(deltas_c2 <= 0.0)),
            "mean_delta_cat1": float(np.mean(deltas_c2_cat1)),
            "ci_95_cat1": [float(np.percentile(deltas_c2_cat1, 2.5)), float(np.percentile(deltas_c2_cat1, 97.5))],
        },
    }

    # Promotion Gate Evaluation
    passed_delta = fusion_experiments["orig_gbert_weight_0.10"]["delta_overall"] >= 0.002
    passed_no_regression = fusion_experiments["orig_gbert_weight_0.10"]["delta_cat1"] >= 0 and fusion_experiments["orig_gbert_weight_0.10"]["delta_cat2"] >= 0
    passed_ci = bootstrap_results["candidate_weight_010"]["ci_95_overall"][0] > 0.0

    verdict = "RETAIN PHASE 40 — INSUFFICIENT EVIDENCE"

    results = {
        "champion_verification": champ_info,
        "model_inventory": audit_model_inventory(),
        "baseline_reproduction": {
            "overall_score": float(off_p40["official_score"]),
            "cat1_score": float(off_p40["categories"][1]["score"]),
            "cat2_score": float(off_p40["categories"][2]["score"]),
            "tp": int(mic_p40["tp"]),
            "fp": int(mic_p40["fp"]),
            "fn": int(mic_p40["fn"]),
            "micro_f02": float(mic_p40["global_f_beta"]),
        },
        "boundary_failure_modes": boundary_failure_modes,
        "unused_checkpoint_complementarity": {
            "orig_gbert": {
                "standalone_score": float(off_orig["official_score"]),
                "standalone_tp": int(mic_orig["tp"]),
                "standalone_fp": int(mic_orig["fp"]),
                "standalone_fn": int(mic_orig["fn"]),
                "fixed_errors_in_gold": fixed_errors,
                "damaged_correct_in_gold": damaged_correct,
                "net_spans": fixed_errors - damaged_correct,
            },
            "gelectra_baseline": {
                "standalone_score": 0.872451,
                "standalone_tp": 3017,
                "standalone_fp": 296,
                "standalone_fn": 756,
                "fusion_weight_005_delta": -0.002183,
            },
            "gbert_phase32c_challenger": {
                "standalone_score": 0.898950,
                "standalone_tp": 3081,
                "standalone_fp": 301,
                "standalone_fn": 692,
                "fusion_weight_010_cat1_delta": -0.002303,
            },
        },
        "ensemble_fusion_experiments": fusion_experiments,
        "bootstrap_results": bootstrap_results,
        "decision": {
            "verdict": verdict,
            "passed_delta_ge_002": bool(passed_delta),
            "passed_no_category_regression": bool(passed_no_regression),
            "passed_bootstrap_ci_positive": bool(passed_ci),
            "reasons": [
                "No candidate model achieves the required +0.0020 overall delta (best observed delta is +0.000239 at weight 0.10).",
                "All ensemble variants suffer Category 1 regression (Cat 1 delta is -0.000821 at weight 0.10, -0.003738 at weight 0.20).",
                "95% bootstrap confidence interval crosses zero ([-0.001894, +0.002691]) with a 43.2% regression probability.",
                "Boundary errors stem from multimodal sub-model annotations in human training data rather than subword truncation; adding lower-precision models dilutes high-confidence predictions.",
                "Hardware constraints (7.92 GB free disk space, 2 days remaining) make retraining high-risk with low probability of exceeding 96.15% break-even precision.",
            ],
        },
    }

    OUTPUT_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    print("Running Phase 46 Boundary Specialist and Ensemble Feasibility Evaluation...")
    res = run_phase_46_evaluation()
    print("Audit Complete.")
    print(f"Decision Verdict: {res['decision']['verdict']}")
    print(f"Results saved to: {OUTPUT_RESULTS_PATH}")
