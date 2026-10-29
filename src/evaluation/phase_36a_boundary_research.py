"""
Phase 36A: Vehicle-Model Boundary Correction Research.

Investigates conservative entity-boundary corrections for Kompatibles_Fahrzeug_Modell
on the 489-title clean holdout under the Phase 35C champion configuration (0.80 threshold).

Evaluates:
- Baseline (Phase 35C Champion: Vehicle 0.80, Others 0.60)
- Detailed error breakdown of vehicle-model errors into 8 mutually exclusive categories.
- Rule A: Parenthetical chassis suffix removal (e.g. "VITO (W447)" -> "VITO")
- Rule B: Generation suffix removal (e.g. "Vectra B" -> "Vectra", "Golf IV" -> "Golf")
- Rule C: Extraneous leading/trailing punctuation cleanup
- Damage analysis (false corrections vs regressions)
- Category 1 vs Category 2 impacts
- Generalization and consistency analysis across full dataset
"""

import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
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
    make_json_serializable,
)

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
HOLDOUT_MANIFEST_PATH = CACHE_DIR / "phase_35b_clean_holdout_manifest.json"
VAL_PROBS_PATH = CACHE_DIR / "phase_35a_val_1k_probs.pt"
OUTPUT_REPORT_DATA = CACHE_DIR / "phase_36a_boundary_research_results.json"


def clean_parenthetical_chassis(val: str) -> str:
    """Rule A: Remove trailing parenthetical chassis codes like (W447), (E91), (8P)."""
    # Matches trailing parenthetical with alphanumeric/hyphen/slash/dot
    cleaned = re.sub(r"\s*\([A-Za-z0-9\-/\.]+\)$", "", val).strip()
    return cleaned if cleaned else val


def clean_generation_suffix(val: str) -> str:
    """Rule B: Remove trailing standalone generation identifiers (single letter or roman numeral)."""
    # Pattern 1: Roman numerals I, II, III, IV, V, VI, VII, VIII
    p_roman = r"\s+(I{1,3}|IV|V|VI{1,3}|VII|VIII|IX|X)$"
    # Pattern 2: Single uppercase letter generation code (e.g. Vectra B, Corsa D, Astra H)
    p_letter = r"\s+([A-HJ-Z])$"

    # Only strip if preceded by at least one word
    if re.search(p_roman, val, re.IGNORECASE):
        cleaned = re.sub(p_roman, "", val, flags=re.IGNORECASE).strip()
        if cleaned:
            return cleaned
    elif re.search(p_letter, val):
        cleaned = re.sub(p_letter, "", val).strip()
        if cleaned:
            return cleaned
    return val


def clean_extraneous_punctuation(val: str) -> str:
    """Rule C: Remove extraneous leading/trailing punctuation without altering internal hyphens."""
    cleaned = val.strip(" \t\r\n-/,.:;'+*|~()[]{}")
    return cleaned if cleaned else val


def run_boundary_research():
    print("=" * 80)
    print("PHASE 36A: VEHICLE-MODEL BOUNDARY CORRECTION RESEARCH")
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

    # 2. Decode Baseline Predictions (Phase 35C Champion: Vehicle 0.80, Others 0.60)
    print("\n[Step 2] Decoding baseline Phase 35C Champion predictions (Vehicle 0.80 / Others 0.60)...")
    baseline_spanned = decode_spanned_predictions(
        clean_p_fused, clean_examples, id2label, conf_thresh=0.60,
        aspect_thresholds={"Kompatibles_Fahrzeug_Modell": 0.80}
    )
    baseline_tuples = [[(e["aspect"], e["value"]) for e in ex_ents] for ex_ents in baseline_spanned]
    base_metrics = compute_metrics_from_tuples(gold_clean_tuples, baseline_tuples, clean_examples)
    base_ov = base_metrics["overall"]

    print(f"Baseline Phase 35C Metrics on 489 Clean Titles:")
    print(f"  F0.2: {base_ov['f_beta']:.6f} | Precision: {base_ov['precision']:.6f} | Recall: {base_ov['recall']:.6f}")
    print(f"  TP: {base_ov['tp']} | FP: {base_ov['fp']} | FN: {base_ov['fn']} | TP+FN: {base_ov['tp'] + base_ov['fn']}")
    assert base_ov["tp"] + base_ov["fn"] == 3773, "TP+FN invariant violated!"

    # 3. Analyze Vehicle Model Errors at 0.80
    print("\n[Step 3] Analyzing Kompatibles_Fahrzeug_Modell error breakdown at 0.80...")
    vm_total_preds = 0
    vm_tps = 0
    vm_fps = []

    for ex_idx, (g_list, p_list) in enumerate(zip(spanned_gold, baseline_spanned)):
        ex = clean_examples[ex_idx]
        title = " ".join(ex["tokens"])
        matched_g = set()

        for p in p_list:
            if p["aspect"] != "Kompatibles_Fahrzeug_Modell":
                continue
            vm_total_preds += 1
            matched = False
            for g_idx, g in enumerate(g_list):
                if g_idx in matched_g:
                    continue
                if g["aspect"] == "Kompatibles_Fahrzeug_Modell" and g["value"] == p["value"]:
                    matched = True
                    matched_g.add(g_idx)
                    vm_tps += 1
                    break
            if not matched:
                vm_fps.append({
                    "record_number": ex["record_number"],
                    "title": title,
                    "pred": p,
                    "gold_entities": g_list,
                })

    print(f"Total vehicle model predictions: {vm_total_preds}")
    print(f"Vehicle model TPs:               {vm_tps}")
    print(f"Vehicle model FPs:               {len(vm_fps)} (FP Rate: {len(vm_fps)/vm_total_preds*100:.2f}%)")

    # Classify each of the 74 vehicle model FPs into the 8 requested categories
    classified_errors = defaultdict(list)

    for item in vm_fps:
        p_val = item["pred"]["value"]
        p_start = item["pred"]["start"]
        p_end = item["pred"]["end"]
        p_span = set(range(p_start, p_end))
        g_list = item["gold_entities"]

        same_asp_g = [g for g in g_list if g["aspect"] == "Kompatibles_Fahrzeug_Modell"]
        diff_asp_g = [g for g in g_list if g["aspect"] != "Kompatibles_Fahrzeug_Modell"]

        # Check overlapping gold entities
        overlapping_same = [g for g in same_asp_g if p_span.intersection(set(range(g["start"], g["end"])))]
        overlapping_diff = [g for g in diff_asp_g if p_span.intersection(set(range(g["start"], g["end"])))]

        if not overlapping_same and not overlapping_diff:
            # Genuine False Alarm (no gold entity anywhere on this span)
            classified_errors["genuine_false_alarm"].append((item, None))
        elif not overlapping_same and overlapping_diff:
            # Aspect Confusion (model tagged as vehicle model, but gold tagged as Marke/Produktart/etc.)
            classified_errors["incorrect_aspect_label"].append((item, overlapping_diff[0]))
        else:
            # Overlaps with gold vehicle model! Boundary error!
            g_ref = overlapping_same[0]
            g_val = g_ref["value"]

            # Sub-classify boundary error
            if re.search(r"\s*\([A-Za-z0-9\-/\.]+\)$", p_val) and not re.search(r"\s*\([A-Za-z0-9\-/\.]+\)$", g_val):
                classified_errors["parenthetical_chassis_code"].append((item, g_ref))
            elif (p_val.startswith(g_val) and re.search(r"\s+([A-HJ-Z]|[0-9]{1,2}|I{1,3}|IV|V|VI{1,3}|VII|VIII|IX|X)$", p_val)) or \
                 (g_val.startswith(p_val) and re.search(r"\s+([A-HJ-Z]|[0-9]{1,2}|I{1,3}|IV|V|VI{1,3}|VII|VIII|IX|X)$", g_val)):
                classified_errors["generation_suffix"].append((item, g_ref))
            elif re.search(r"^[\s\-\/\,\.\:\'\"\\\|\+]+|[\s\-\/\,\.\:\'\"\\\|\+]+$", p_val):
                classified_errors["leading_trailing_punctuation"].append((item, g_ref))
            elif len(overlapping_same) > 1 or re.search(r"\b(und|and|\/|\+)\b", p_val):
                classified_errors["multi_token_boundary_error"].append((item, g_ref))
            elif len(p_val) < len(g_val) and p_val in g_val:
                classified_errors["missing_model_tokens"].append((item, g_ref))
            elif len(p_val) > len(g_val) and g_val in p_val:
                classified_errors["extra_model_variants"].append((item, g_ref))
            else:
                classified_errors["other_boundary_error"].append((item, g_ref))

    print("\n--- Vehicle Model Error Breakdown (74 FPs) ---")
    for cat_name, items in sorted(classified_errors.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {cat_name:<35}: {len(items):>2} cases ({len(items)/len(vm_fps)*100:>5.1f}%)")

    # 4. Simulate Three Conservative Rules Independently
    print("\n" + "=" * 80)
    print("SIMULATING CONSERVATIVE BOUNDARY RULES (INDEPENDENT EVALUATION)")
    print("=" * 80)

    rules = {
        "Rule A (Parenthetical Chassis Suffix)": clean_parenthetical_chassis,
        "Rule B (Generation Suffix)": clean_generation_suffix,
        "Rule C (Punctuation Cleanup)": clean_extraneous_punctuation,
    }

    rule_eval_results = {}

    for rule_name, rule_fn in rules.items():
        print(f"\n--- Evaluating {rule_name} ---")

        # Apply rule ONLY to Kompatibles_Fahrzeug_Modell predictions
        sim_tuples = []
        predictions_changed = 0
        changed_details = []

        for ex_idx, ex_ents in enumerate(baseline_spanned):
            ex_gold_tuples = gold_clean_tuples[ex_idx]
            new_ex_tuples = []
            for e in ex_ents:
                asp = e["aspect"]
                val = e["value"]
                if asp == "Kompatibles_Fahrzeug_Modell":
                    new_val = rule_fn(val)
                    if new_val != val:
                        predictions_changed += 1
                        was_correct_before = ((asp, val) in ex_gold_tuples)
                        is_correct_after = ((asp, new_val) in ex_gold_tuples)
                        changed_details.append({
                            "record_number": clean_examples[ex_idx]["record_number"],
                            "title": " ".join(clean_examples[ex_idx]["tokens"]),
                            "original_val": val,
                            "cleaned_val": new_val,
                            "was_correct_before": was_correct_before,
                            "is_correct_after": is_correct_after,
                            "gold_vehicle_models": [g[1] for g in ex_gold_tuples if g[0] == asp],
                        })
                    new_ex_tuples.append((asp, new_val))
                else:
                    new_ex_tuples.append((asp, val))
            sim_tuples.append(new_ex_tuples)

        # Compute Metrics
        sim_metrics = compute_metrics_from_tuples(gold_clean_tuples, sim_tuples, clean_examples)
        sim_ov = sim_metrics["overall"]
        sim_cat = sim_metrics["category"]

        # Damage Analysis
        fixed_errors = [d for d in changed_details if not d["was_correct_before"] and d["is_correct_after"]]
        damaged_correct = [d for d in changed_details if d["was_correct_before"] and not d["is_correct_after"]]
        still_incorrect = [d for d in changed_details if not d["was_correct_before"] and not d["is_correct_after"]]
        still_correct = [d for d in changed_details if d["was_correct_before"] and d["is_correct_after"]]

        delta_f02 = sim_ov["f_beta"] - base_ov["f_beta"]
        delta_p = sim_ov["precision"] - base_ov["precision"]
        delta_r = sim_ov["recall"] - base_ov["recall"]
        delta_tp = sim_ov["tp"] - base_ov["tp"]
        delta_fp = sim_ov["fp"] - base_ov["fp"]
        delta_fn = sim_ov["fn"] - base_ov["fn"]

        print(f"Predictions changed:             {predictions_changed}")
        print(f"  Fixed (Incorrect -> Correct):  {len(fixed_errors)}")
        print(f"  Damaged (Correct -> Incorrect):{len(damaged_correct)}")
        print(f"  Still Incorrect:               {len(still_incorrect)}")
        print(f"Net TP change:                   {delta_tp:+d} ({base_ov['tp']} -> {sim_ov['tp']})")
        print(f"Net FP change:                   {delta_fp:+d} ({base_ov['fp']} -> {sim_ov['fp']})")
        print(f"Net FN change:                   {delta_fn:+d} ({base_ov['fn']} -> {sim_ov['fn']})")
        print(f"Precision change:                {delta_p:+.6f} ({base_ov['precision']:.6f} -> {sim_ov['precision']:.6f})")
        print(f"Recall change:                   {delta_r:+.6f} ({base_ov['recall']:.6f} -> {sim_ov['recall']:.6f})")
        print(f"F0.2 change:                     {delta_f02:+.6f} ({base_ov['f_beta']:.6f} -> {sim_ov['f_beta']:.6f})")

        print("Category Breakdown:")
        for cat_id in [1, 2]:
            b_c = base_metrics["category"][cat_id]
            s_c = sim_cat[cat_id]
            c_df02 = s_c["f_beta"] - b_c["f_beta"]
            print(f"  Cat {cat_id}: F0.2 {b_c['f_beta']:.6f} -> {s_c['f_beta']:.6f} ({c_df02:+.6f}) | TP: {s_c['tp']-b_c['tp']:+d} | FP: {s_c['fp']-b_c['fp']:+d}")

        rule_eval_results[rule_name] = {
            "predictions_changed": predictions_changed,
            "fixed_count": len(fixed_errors),
            "damaged_count": len(damaged_correct),
            "still_incorrect_count": len(still_incorrect),
            "delta_tp": delta_tp,
            "delta_fp": delta_fp,
            "delta_fn": delta_fn,
            "delta_f02": delta_f02,
            "delta_precision": delta_p,
            "delta_recall": delta_r,
            "overall_metrics": sim_ov,
            "category_metrics": sim_cat,
            "sample_fixed": fixed_errors[:5],
            "sample_damaged": damaged_correct[:5],
            "sample_still_incorrect": still_incorrect[:5],
        }

    # 5. Global Corpus Annotation Consistency Check for Chassis & Generation
    print("\n" + "=" * 80)
    print("GLOBAL CORPUS ANNOTATION CONSISTENCY ANALYSIS (ALL 5,000 TITLES)")
    print("=" * 80)

    all_spanned_gold = extract_spanned_gold_entities(all_examples)
    all_gold_vm_values = [
        e["value"] for ex_ents in all_spanned_gold for e in ex_ents if e["aspect"] == "Kompatibles_Fahrzeug_Modell"
    ]
    print(f"Total gold vehicle model entities in 5,000 dataset: {len(all_gold_vm_values):,}")

    gold_with_chassis = [v for v in all_gold_vm_values if re.search(r"\s*\([A-Za-z0-9\-/\.]+\)$", v)]
    gold_with_roman = [v for v in all_gold_vm_values if re.search(r"\s+(I{1,3}|IV|V|VI{1,3}|VII|VIII|IX|X)$", v, re.IGNORECASE)]
    gold_with_letter = [v for v in all_gold_vm_values if re.search(r"\s+([A-HJ-Z])$", v)]
    gold_with_ext_punct = [v for v in all_gold_vm_values if re.search(r"^[\s\-\/\,\.\:\'\"\\\|\+]+|[\s\-\/\,\.\:\'\"\\\|\+]+$", v)]

    print(f"Gold vehicle models ending with parenthetical chassis: {len(gold_with_chassis):>5} ({len(gold_with_chassis)/len(all_gold_vm_values)*100:.2f}%)")
    print(f"Gold vehicle models ending with roman numeral:          {len(gold_with_roman):>5} ({len(gold_with_roman)/len(all_gold_vm_values)*100:.2f}%)")
    print(f"Gold vehicle models ending with single capital letter:  {len(gold_with_letter):>5} ({len(gold_with_letter)/len(all_gold_vm_values)*100:.2f}%)")
    print(f"Gold vehicle models with leading/trailing punctuation:  {len(gold_with_ext_punct):>5} ({len(gold_with_ext_punct)/len(all_gold_vm_values)*100:.2f}%)")

    # Sample gold entities with chassis to show counterexamples
    print(f"\nSample gold vehicle models that explicitly INCLUDE parenthetical chassis in ground truth:")
    for v in gold_with_chassis[:8]:
        print(f"  \"{v}\"")

    print(f"\nSample gold vehicle models that explicitly INCLUDE single letter generation in ground truth:")
    for v in gold_with_letter[:8]:
        print(f"  \"{v}\"")

    corpus_consistency = {
        "total_gold_vm": len(all_gold_vm_values),
        "gold_with_chassis_count": len(gold_with_chassis),
        "gold_with_chassis_pct": len(gold_with_chassis) / len(all_gold_vm_values),
        "gold_with_roman_count": len(gold_with_roman),
        "gold_with_roman_pct": len(gold_with_roman) / len(all_gold_vm_values),
        "gold_with_letter_count": len(gold_with_letter),
        "gold_with_letter_pct": len(gold_with_letter) / len(all_gold_vm_values),
        "gold_with_ext_punct_count": len(gold_with_ext_punct),
        "gold_with_ext_punct_pct": len(gold_with_ext_punct) / len(all_gold_vm_values),
        "sample_gold_with_chassis": gold_with_chassis[:10],
        "sample_gold_with_letter": gold_with_letter[:10],
    }

    # 6. Save Comprehensive Research Results
    results_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "holdout_count": len(clean_examples),
        "baseline_champion_phase35c": {
            "overall": base_ov,
            "category": base_metrics["category"],
        },
        "vehicle_model_error_classification_080": {
            cat_name: {
                "count": len(items),
                "pct_of_fps": len(items) / len(vm_fps),
                "examples": [
                    {
                        "record_number": it[0]["record_number"],
                        "title": it[0]["title"],
                        "pred_val": it[0]["pred"]["value"],
                        "pred_conf": float(it[0]["pred"]["min_conf"]),
                        "gold_ref": it[1]["value"] if it[1] else None,
                    }
                    for it in items[:5]
                ],
            }
            for cat_name, items in classified_errors.items()
        },
        "simulated_rules": rule_eval_results,
        "corpus_annotation_consistency": corpus_consistency,
    }

    with open(OUTPUT_REPORT_DATA, "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(results_payload), f, indent=2)
    print(f"\nSaved comprehensive research data to {OUTPUT_REPORT_DATA}")
    print("\nPhase 36A research simulation complete!")


if __name__ == "__main__":
    run_boundary_research()
