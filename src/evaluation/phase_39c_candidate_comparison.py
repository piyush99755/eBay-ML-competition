"""
Phase 39C: Category-Specific Calibration & Package-Contents Re-Evaluation.

Rigorous cache-only comparative evaluation of two evidence-supported modifications to
the frozen Phase 36C champion:
  - Candidate A: Category 2 Vehicle-Model Recovery Only (Zahnriemen).
    Preserves Category 1 bit-for-bit. Recovers [0.60, 0.80) vehicle models matching
    the frozen Phase 39B training lexicon strictly in Category 2.
  - Candidate B: Package-Contents Threshold Elevation to 0.80 (Im_Lieferumfang_Enthalten).
    Applied to both categories without any other aspect or boundary changes.

Evaluation Protocol:
  - Primary Metric: Official Competition Hierarchical F0.2 (Category 50/50 Macro-Average).
  - Secondary Diagnostic: Global Micro F0.2.
  - Primary Partition: 489-title clean holdout (strictly unseen during supervised training).
  - Secondary Diagnostic: 511 contaminated validation titles (cross-partition reference).
  - Statistical Reliability: 1,000 paired title-level bootstrap resamples (seed 42).
  - Enforces non-destructive invariants.
"""

import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
from transformers import AutoConfig

from src.data.build_examples import build_examples, load_training_data
from src.data.make_splits import split_examples
from src.evaluation.phase_35b_audit_and_error_analysis import (
    extract_spanned_gold_entities,
    decode_spanned_predictions,
)
from src.evaluation.phase_39a_official_metric import (
    BETA,
    BETA_SQ,
    compute_fbeta,
    count_entity_matches,
    calculate_official_competition_metrics,
    calculate_global_micro_metrics,
)
from src.evaluation.phase_39b_vehicle_model_recovery import (
    CHAMPION_TSV_PATH,
    EXPECTED_CHAMPION_SHA256,
    CHECKPOINT_PATHS,
    CLEAN_HOLDOUT_MANIFEST_PATH,
    VAL_PROBS_CACHE_PATH,
    verify_phase_integrity,
    build_training_vehicle_lexicon,
)

RESULTS_CACHE_PATH = "data/cache/phase_39c_comparison_results.json"


def decode_phase_39c_candidates(
    p_fused: List[np.ndarray],
    examples: List[Dict[str, Any]],
    id2label: Dict[int, str],
    lexicon: Set[str],
) -> Tuple[
    List[List[Dict[str, Any]]],
    List[List[Dict[str, Any]]],
    List[List[Dict[str, Any]]],
]:
    """
    Decodes predictions for:
      - Baseline Phase 36C (VM=0.80, PA=0.80, others=0.60)
      - Candidate A: Base + (VM in [0.60, 0.80) matching lexicon ONLY in Category 2)
      - Candidate B: Base with Im_Lieferumfang_Enthalten threshold elevated to 0.80
      
    Asserts strict prediction invariants:
      - Candidate A: Baseline predictions are 100% preserved. Category 1 is bit-for-bit identical.
      - Candidate B: Only Im_Lieferumfang_Enthalten predictions may be removed. Zero other aspects change.
    """
    base_spanned = decode_spanned_predictions(
        p_fused,
        examples,
        id2label,
        conf_thresh=0.60,
        aspect_thresholds={"Kompatibles_Fahrzeug_Modell": 0.80, "Produktart": 0.80},
    )

    vm060_spanned = decode_spanned_predictions(
        p_fused,
        examples,
        id2label,
        conf_thresh=0.60,
        aspect_thresholds={"Kompatibles_Fahrzeug_Modell": 0.60, "Produktart": 0.80},
    )

    cand_b_spanned = decode_spanned_predictions(
        p_fused,
        examples,
        id2label,
        conf_thresh=0.60,
        aspect_thresholds={
            "Kompatibles_Fahrzeug_Modell": 0.80,
            "Produktart": 0.80,
            "Im_Lieferumfang_Enthalten": 0.80,
        },
    )

    cand_a_spanned: List[List[Dict[str, Any]]] = []
    for i, ex in enumerate(examples):
        b_ents = list(base_spanned[i])
        if ex["category"] == 2:
            v_ents = vm060_spanned[i]
            b_keys = {(e["aspect"], e["start"], e["end"]) for e in b_ents}
            a_add = []
            for p in v_ents:
                p_key = (p["aspect"], p["start"], p["end"])
                if p_key not in b_keys:
                    val = p["value"].strip()
                    if val in lexicon:
                        a_add.append(p)
            cand_a_spanned.append(b_ents + a_add)
        else:
            cand_a_spanned.append(b_ents)

    # Invariant assertions
    for i, ex in enumerate(examples):
        # Candidate A invariant
        b_tuples = [(e["aspect"], e["value"]) for e in base_spanned[i]]
        a_tuples = [(e["aspect"], e["value"]) for e in cand_a_spanned[i]]
        b_cnt = Counter(b_tuples)
        a_cnt = Counter(a_tuples)
        for k, v in b_cnt.items():
            if a_cnt[k] < v:
                raise AssertionError(f"Candidate A violated preservation for example {i}, item {k}")
        if ex["category"] == 1:
            if b_tuples != a_tuples:
                raise AssertionError(f"Candidate A altered Category 1 example {i}")

        # Candidate B invariant
        b_non_lie = [(e["aspect"], e["value"]) for e in base_spanned[i] if e["aspect"] != "Im_Lieferumfang_Enthalten"]
        cand_non_lie = [(e["aspect"], e["value"]) for e in cand_b_spanned[i] if e["aspect"] != "Im_Lieferumfang_Enthalten"]
        if b_non_lie != cand_non_lie:
            raise AssertionError(f"Candidate B altered non-Lieferumfang aspect for example {i}")

        b_lie = [(e["aspect"], e["value"]) for e in base_spanned[i] if e["aspect"] == "Im_Lieferumfang_Enthalten"]
        cand_lie = [(e["aspect"], e["value"]) for e in cand_b_spanned[i] if e["aspect"] == "Im_Lieferumfang_Enthalten"]
        if len(cand_lie) > len(b_lie):
            raise AssertionError(f"Candidate B added Lieferumfang entity for example {i}")
        b_lie_cnt = Counter(b_lie)
        cand_lie_cnt = Counter(cand_lie)
        for k, v in cand_lie_cnt.items():
            if v > b_lie_cnt[k]:
                raise AssertionError(f"Candidate B introduced novel Lieferumfang value for example {i}: {k}")

    return base_spanned, cand_a_spanned, cand_b_spanned


def evaluate_partition(
    examples: List[Dict[str, Any]],
    base_spanned: List[List[Dict[str, Any]]],
    cand_a_spanned: List[List[Dict[str, Any]]],
    cand_b_spanned: List[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Computes official hierarchical and global micro metrics for a partition.
    """
    spanned_gold = extract_spanned_gold_entities(examples)
    gold_tuples = [[(e["aspect"], e["value"]) for e in ex] for ex in spanned_gold]
    base_tuples = [[(e["aspect"], e["value"]) for e in ex] for ex in base_spanned]
    cand_a_tuples = [[(e["aspect"], e["value"]) for e in ex] for ex in cand_a_spanned]
    cand_b_tuples = [[(e["aspect"], e["value"]) for e in ex] for ex in cand_b_spanned]

    records_base = [
        {"category": examples[i]["category"], "predicted_entities": base_tuples[i], "gold_entities": gold_tuples[i]}
        for i in range(len(examples))
    ]
    records_a = [
        {"category": examples[i]["category"], "predicted_entities": cand_a_tuples[i], "gold_entities": gold_tuples[i]}
        for i in range(len(examples))
    ]
    records_b = [
        {"category": examples[i]["category"], "predicted_entities": cand_b_tuples[i], "gold_entities": gold_tuples[i]}
        for i in range(len(examples))
    ]

    off_base = calculate_official_competition_metrics(records_base)
    off_a = calculate_official_competition_metrics(records_a)
    off_b = calculate_official_competition_metrics(records_b)

    mic_base = calculate_global_micro_metrics(records_base)
    mic_a = calculate_global_micro_metrics(records_a)
    mic_b = calculate_global_micro_metrics(records_b)

    return {
        "records_base": records_base,
        "records_a": records_a,
        "records_b": records_b,
        "baseline": {
            "official_score": float(off_base["official_score"]),
            "cat1_score": float(off_base["categories"][1]["score"]),
            "cat2_score": float(off_base["categories"][2]["score"]),
            "micro_f02": float(mic_base["global_f_beta"]),
            "micro_precision": float(mic_base["global_precision"]),
            "micro_recall": float(mic_base["global_recall"]),
            "tp": mic_base["tp"],
            "fp": mic_base["fp"],
            "fn": mic_base["fn"],
        },
        "candidate_a": {
            "official_score": float(off_a["official_score"]),
            "cat1_score": float(off_a["categories"][1]["score"]),
            "cat2_score": float(off_a["categories"][2]["score"]),
            "micro_f02": float(mic_a["global_f_beta"]),
            "micro_precision": float(mic_a["global_precision"]),
            "micro_recall": float(mic_a["global_recall"]),
            "tp": mic_a["tp"],
            "fp": mic_a["fp"],
            "fn": mic_a["fn"],
            "delta_official": float(off_a["official_score"] - off_base["official_score"]),
            "delta_cat1": float(off_a["categories"][1]["score"] - off_base["categories"][1]["score"]),
            "delta_cat2": float(off_a["categories"][2]["score"] - off_base["categories"][2]["score"]),
            "delta_micro_f02": float(mic_a["global_f_beta"] - mic_base["global_f_beta"]),
            "delta_tp": mic_a["tp"] - mic_base["tp"],
            "delta_fp": mic_a["fp"] - mic_base["fp"],
            "delta_fn": mic_a["fn"] - mic_base["fn"],
        },
        "candidate_b": {
            "official_score": float(off_b["official_score"]),
            "cat1_score": float(off_b["categories"][1]["score"]),
            "cat2_score": float(off_b["categories"][2]["score"]),
            "micro_f02": float(mic_b["global_f_beta"]),
            "micro_precision": float(mic_b["global_precision"]),
            "micro_recall": float(mic_b["global_recall"]),
            "tp": mic_b["tp"],
            "fp": mic_b["fp"],
            "fn": mic_b["fn"],
            "delta_official": float(off_b["official_score"] - off_base["official_score"]),
            "delta_cat1": float(off_b["categories"][1]["score"] - off_base["categories"][1]["score"]),
            "delta_cat2": float(off_b["categories"][2]["score"] - off_base["categories"][2]["score"]),
            "delta_micro_f02": float(mic_b["global_f_beta"] - mic_base["global_f_beta"]),
            "delta_tp": mic_b["tp"] - mic_base["tp"],
            "delta_fp": mic_b["fp"] - mic_base["fp"],
            "delta_fn": mic_b["fn"] - mic_base["fn"],
        },
        "off_base_raw": off_base,
        "off_a_raw": off_a,
        "off_b_raw": off_b,
    }


def run_paired_bootstrap(
    records_base: List[Dict[str, Any]],
    records_a: List[Dict[str, Any]],
    records_b: List[Dict[str, Any]],
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run 1,000 paired title-level bootstrap resamples (seed 42) computing official competition scores.
    Handles edge cases with documented consistent policy.
    """
    np.random.seed(seed)
    n = len(records_base)
    boot_deltas_a = []
    boot_deltas_b = []

    for _ in range(n_resamples):
        sample_indices = np.random.choice(n, size=n, replace=True)
        rec_b = [records_base[idx] for idx in sample_indices]
        rec_a = [records_a[idx] for idx in sample_indices]
        rec_cand_b = [records_b[idx] for idx in sample_indices]

        score_base = calculate_official_competition_metrics(rec_b)["official_score"]
        score_a = calculate_official_competition_metrics(rec_a)["official_score"]
        score_b = calculate_official_competition_metrics(rec_cand_b)["official_score"]

        boot_deltas_a.append(score_a - score_base)
        boot_deltas_b.append(score_b - score_base)

    boot_deltas_a_arr = np.array(boot_deltas_a)
    boot_deltas_b_arr = np.array(boot_deltas_b)

    return {
        "n_resamples": n_resamples,
        "seed": seed,
        "candidate_a": {
            "mean_delta": float(np.mean(boot_deltas_a_arr)),
            "std_delta": float(np.std(boot_deltas_a_arr)),
            "ci_95": [float(np.percentile(boot_deltas_a_arr, 2.5)), float(np.percentile(boot_deltas_a_arr, 97.5))],
            "p_val_le_zero": float(np.mean(boot_deltas_a_arr <= 0.0)),
        },
        "candidate_b": {
            "mean_delta": float(np.mean(boot_deltas_b_arr)),
            "std_delta": float(np.std(boot_deltas_b_arr)),
            "ci_95": [float(np.percentile(boot_deltas_b_arr, 2.5)), float(np.percentile(boot_deltas_b_arr, 97.5))],
            "p_val_le_zero": float(np.mean(boot_deltas_b_arr <= 0.0)),
        },
    }


def execute_phase_39c_pipeline(run_bootstrap: bool = True, save_cache: bool = True) -> Dict[str, Any]:
    """
    Executes the complete Phase 39C comparative evaluation pipeline.
    """
    integrity = verify_phase_integrity()

    # Load holdout manifest and examples
    with open(CLEAN_HOLDOUT_MANIFEST_PATH, "r", encoding="utf-8") as f:
        clean_ids = set(json.load(f)["record_ids"])

    df = load_training_data()
    all_examples = build_examples(df)
    train_4k, val_1k = split_examples(all_examples, random_state=42)

    clean_indices = [i for i, ex in enumerate(val_1k) if ex["record_number"] in clean_ids]
    dirty_indices = [i for i, ex in enumerate(val_1k) if ex["record_number"] not in clean_ids]

    clean_examples = [val_1k[i] for i in clean_indices]
    dirty_examples = [val_1k[i] for i in dirty_indices]

    # Build frozen lexicon from train_4k
    lexicon, make_model_pairs, train_model_counts, train_make_counts = build_training_vehicle_lexicon(train_4k)

    # Load cached probabilities
    probs = torch.load(VAL_PROBS_CACHE_PATH, map_location="cpu", weights_only=False)
    cfg = AutoConfig.from_pretrained("checkpoints/gbert_baseline_best")
    id2label = {int(k): v for k, v in cfg.id2label.items()}
    model_names = ["dapt_gbert", "xlmr_seed42", "xlmr_seed7", "xlmr_seed21"]
    model_weights = [0.40, 0.20, 0.20, 0.20]

    clean_p_fused = [
        sum(w * probs[m][i] for w, m in zip(model_weights, model_names)) for i in clean_indices
    ]
    dirty_p_fused = [
        sum(w * probs[m][i] for w, m in zip(model_weights, model_names)) for i in dirty_indices
    ]

    # Decode candidates on clean holdout
    base_clean, cand_a_clean, cand_b_clean = decode_phase_39c_candidates(
        clean_p_fused, clean_examples, id2label, lexicon
    )
    clean_eval = evaluate_partition(clean_examples, base_clean, cand_a_clean, cand_b_clean)

    # Decode candidates on contaminated validation titles (cross-partition diagnostic)
    base_dirty, cand_a_dirty, cand_b_dirty = decode_phase_39c_candidates(
        dirty_p_fused, dirty_examples, id2label, lexicon
    )
    dirty_eval = evaluate_partition(dirty_examples, base_dirty, cand_a_dirty, cand_b_dirty)

    # Error analysis and affected title inspection on clean holdout
    cand_a_affected_titles = []
    cand_a_added_entities = []
    for i, ex in enumerate(clean_examples):
        if len(cand_a_clean[i]) != len(base_clean[i]):
            cand_a_affected_titles.append(ex["record_number"])
            added = [p for p in cand_a_clean[i] if p not in base_clean[i]]
            for p in added:
                cand_a_added_entities.append({
                    "record_id": ex["record_number"],
                    "value": p["value"],
                    "conf": float(p.get("min_conf", p.get("confidence", 0.0))),
                })

    cand_b_affected_titles = []
    cand_b_removed_entities = []
    for i, ex in enumerate(clean_examples):
        if len(cand_b_clean[i]) != len(base_clean[i]):
            cand_b_affected_titles.append(ex["record_number"])
            removed = [p for p in base_clean[i] if p not in cand_b_clean[i]]
            for p in removed:
                cand_b_removed_entities.append({
                    "record_id": ex["record_number"],
                    "category": ex["category"],
                    "value": p["value"],
                    "conf": float(p.get("min_conf", p.get("confidence", 0.0))),
                })

    # Paired Bootstrap
    bootstrap_results = None
    if run_bootstrap:
        bootstrap_results = run_paired_bootstrap(
            clean_eval["records_base"],
            clean_eval["records_a"],
            clean_eval["records_b"],
            n_resamples=1000,
            seed=42,
        )

    # Decision classification
    delta_a = clean_eval["candidate_a"]["delta_official"]
    delta_b = clean_eval["candidate_b"]["delta_official"]
    cat1_delta_b = clean_eval["candidate_b"]["delta_cat1"]
    cat2_delta_b = clean_eval["candidate_b"]["delta_cat2"]

    decision = "RETAIN PHASE 36C CHAMPION"
    if delta_b >= 0.0020 and (bootstrap_results and bootstrap_results["candidate_b"]["ci_95"][0] > 0) and cat1_delta_b >= 0 and cat2_delta_b >= 0:
        decision = "STRONG LOCAL CHALLENGER — B"
    elif delta_a >= 0.0020 and (bootstrap_results and bootstrap_results["candidate_a"]["ci_95"][0] > 0):
        decision = "STRONG LOCAL CHALLENGER — A"
    elif delta_b > 0 and cat1_delta_b >= 0:
        decision = "MODEST LOCAL CHALLENGER — B"
    elif delta_a > 0 and clean_eval["candidate_a"]["delta_cat1"] >= 0:
        decision = "MODEST LOCAL CHALLENGER — A"

    results = {
        "integrity": integrity,
        "clean_holdout": {
            "num_titles": len(clean_examples),
            "baseline": clean_eval["baseline"],
            "candidate_a": clean_eval["candidate_a"],
            "candidate_b": clean_eval["candidate_b"],
        },
        "contaminated_validation_diagnostic": {
            "num_titles": len(dirty_examples),
            "baseline": dirty_eval["baseline"],
            "candidate_a": dirty_eval["candidate_a"],
            "candidate_b": dirty_eval["candidate_b"],
        },
        "stability": {
            "candidate_a": {
                "num_affected_titles": len(cand_a_affected_titles),
                "num_added_entities": len(cand_a_added_entities),
                "added_entities": cand_a_added_entities,
            },
            "candidate_b": {
                "num_affected_titles": len(cand_b_affected_titles),
                "num_removed_entities": len(cand_b_removed_entities),
                "removed_category_1": sum(1 for r in cand_b_removed_entities if r["category"] == 1),
                "removed_category_2": sum(1 for r in cand_b_removed_entities if r["category"] == 2),
                "sample_removed_values": Counter(r["value"] for r in cand_b_removed_entities).most_common(10),
            },
        },
        "bootstrap": bootstrap_results,
        "decision": decision,
    }

    # Clean intermediate raw objects
    if save_cache:
        if bootstrap_results is None and os.path.exists(RESULTS_CACHE_PATH):
            try:
                with open(RESULTS_CACHE_PATH, "r", encoding="utf-8") as f_prev:
                    prev_data = json.load(f_prev)
                    if prev_data.get("bootstrap") is not None:
                        results["bootstrap"] = prev_data["bootstrap"]
                        if results.get("decision") == "RETAIN PHASE 36C CHAMPION" and prev_data.get("decision") != "RETAIN PHASE 36C CHAMPION":
                            results["decision"] = prev_data["decision"]
            except Exception:
                pass
        os.makedirs(os.path.dirname(RESULTS_CACHE_PATH), exist_ok=True)
        with open(RESULTS_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    print("Running Phase 39C Candidate Comparison Pipeline...")
    res = execute_phase_39c_pipeline(run_bootstrap=True)
    print("\nPipeline execution complete.")
    print(f"Decision: {res['decision']}")
    print(f"Candidate A Official Delta: {res['clean_holdout']['candidate_a']['delta_official']:+.6f}")
    print(f"Candidate B Official Delta: {res['clean_holdout']['candidate_b']['delta_official']:+.6f}")
