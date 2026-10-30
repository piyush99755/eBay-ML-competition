"""
Phase 39B: Precision-Guided Vehicle-Model Recovery Evaluation Module.

Evaluates whether vehicle-model (Kompatibles_Fahrzeug_Modell) predictions suppressed by the
0.80 threshold (lying in [0.60, 0.80)) can be selectively recovered using:
  - Candidate A: Conservative Training-Only Lexicon Matching
  - Candidate B: Lexicon Plus Vehicle-Make Context Window (+/- 3 tokens)

Adheres strictly to Phase 39B specifications:
  - Cache-only research: re-uses existing ensemble probabilities on the 489-title clean holdout.
  - Zero retraining, zero GPU inference, zero DAPT.
  - Training-only lexicon extraction from the 4,000-title training partition (0 holdout leakage).
  - Enforces non-destructive additive invariant: baseline predictions are preserved 100%.
  - Evaluates using both official-style hierarchical F0.2 and global micro F0.2.
  - Runs paired title-level bootstrap resampling (1,000 resamples, seed 42).
"""

import hashlib
import json
import os
import re
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

# Constants & Paths
CHAMPION_TSV_PATH = "submissions/submission_quiz_phase36c_produktart080.tsv"
EXPECTED_CHAMPION_SHA256 = "9F76C2418C60708BC92486F5F39165021000EE9151DCCC7DD8F75CE8BBA9CC87"
CLEAN_HOLDOUT_MANIFEST_PATH = "data/cache/phase_35b_clean_holdout_manifest.json"
VAL_PROBS_CACHE_PATH = "data/cache/phase_35a_val_1k_probs.pt"
CHECKPOINT_PATHS = [
    "checkpoints/phase33b_dapt_gbert_best",
    "checkpoints/xlmr_baseline_best",
    "checkpoints/xlmr_seed_7_best",
    "checkpoints/xlmr_seed_21_best",
]
RESULTS_CACHE_PATH = "data/cache/phase_39b_recovery_results.json"

STOPWORDS: Set[str] = {
    "auch", "fuer", "für", "set", "kit", "typ", "bj", "ab", "bis", "alle", "und", "mit"
}
DECIMAL_REGEX = re.compile(r"^\d+\.\d+$")


def verify_phase_integrity() -> Dict[str, Any]:
    """Verify champion TSV SHA-256, checkpoints, holdout manifest, and probability cache."""
    # Check TSV
    if not os.path.exists(CHAMPION_TSV_PATH):
        raise FileNotFoundError(f"Champion TSV missing: {CHAMPION_TSV_PATH}")
    hasher = hashlib.sha256()
    with open(CHAMPION_TSV_PATH, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    actual_hash = hasher.hexdigest().upper()
    if actual_hash != EXPECTED_CHAMPION_SHA256:
        raise ValueError(f"Champion hash mismatch! Expected {EXPECTED_CHAMPION_SHA256}, got {actual_hash}")

    # Check checkpoints
    for cp in CHECKPOINT_PATHS:
        if not os.path.exists(cp):
            raise FileNotFoundError(f"Checkpoint directory missing: {cp}")

    # Check manifest and probs
    if not os.path.exists(CLEAN_HOLDOUT_MANIFEST_PATH):
        raise FileNotFoundError(f"Manifest missing: {CLEAN_HOLDOUT_MANIFEST_PATH}")
    if not os.path.exists(VAL_PROBS_CACHE_PATH):
        raise FileNotFoundError(f"Probs cache missing: {VAL_PROBS_CACHE_PATH}")

    return {
        "status": "PASSED",
        "champion_sha256": actual_hash,
        "checkpoints_verified": len(CHECKPOINT_PATHS),
        "manifest_path": CLEAN_HOLDOUT_MANIFEST_PATH,
        "probs_cache_path": VAL_PROBS_CACHE_PATH,
    }


def filter_vehicle_model_entry(val: str, cnt: int) -> bool:
    """
    Apply deterministic ambiguity filtering to candidate vehicle-model string:
      - Exclude single digits ('0'-'9')
      - Exclude single letters ('a'-'z', 'A'-'Z')
      - Exclude pure engine displacements / decimals (r'^\\d+\\.\\d+$')
      - Exclude German automotive stopwords / prepositions
      - Require cnt >= 2 OR string length >= 4 characters
    """
    norm_val = val.strip()
    if not norm_val:
        return False
    if norm_val.lower() in STOPWORDS:
        return False
    if len(norm_val) == 1 and (norm_val.isdigit() or norm_val.isalpha()):
        return False
    if DECIMAL_REGEX.match(norm_val):
        return False
    if cnt >= 2 or len(norm_val) >= 4:
        return True
    return False


def build_training_vehicle_lexicon(
    train_examples: List[Dict[str, Any]]
) -> Tuple[Set[str], Set[Tuple[str, str]], Counter, Counter]:
    """
    Extract vehicle models and makes strictly from the canonical training partition (train_4k).
    Applies deterministic ambiguity filtering via filter_vehicle_model_entry.
    
    Returns:
      (candidate_a_lexicon, make_model_pairs, train_model_counts, train_make_counts)
    """
    train_model_counts: Counter = Counter()
    train_make_counts: Counter = Counter()
    make_model_pairs: Set[Tuple[str, str]] = set()

    if train_examples and "tokens" in train_examples[0] and "bio_tags" in train_examples[0]:
        train_spanned = extract_spanned_gold_entities(train_examples)
        for i, ex in enumerate(train_examples):
            ents = train_spanned[i]
            makes = [e["value"].strip().upper() for e in ents if e["aspect"] == "Kompatible_Fahrzeug_Marke"]
            models = [e["value"].strip() for e in ents if e["aspect"] == "Kompatibles_Fahrzeug_Modell"]

            for mk in makes:
                train_make_counts[mk] += 1
            for md in models:
                train_model_counts[md] += 1
                for mk in makes:
                    make_model_pairs.add((mk, md.upper()))
    else:
        for ex in train_examples:
            aspects = ex.get("aspects", {})
            makes = [str(m).strip().upper() for m in aspects.get("Kompatible_Fahrzeug_Marke", [])]
            models = [str(m).strip() for m in aspects.get("Kompatibles_Fahrzeug_Modell", [])]
            for mk in makes:
                train_make_counts[mk] += 1
            for md in models:
                train_model_counts[md] += 1
                for mk in makes:
                    make_model_pairs.add((mk, md.upper()))

    candidate_a_lexicon: Set[str] = set()
    for mod, cnt in train_model_counts.items():
        if filter_vehicle_model_entry(mod, cnt):
            candidate_a_lexicon.add(mod.strip())

    return candidate_a_lexicon, make_model_pairs, train_model_counts, train_make_counts


def decode_predictions_and_candidates(
    clean_p_fused: List[np.ndarray],
    clean_examples: List[Dict[str, Any]],
    id2label: Dict[int, str],
    candidate_a_lexicon: Set[str],
    make_model_pairs: Set[Tuple[str, str]],
    train_model_counts: Counter,
    context_window: int = 3,
) -> Tuple[
    List[List[Dict[str, Any]]],
    List[List[Dict[str, Any]]],
    List[List[Dict[str, Any]]],
    List[Dict[str, Any]],
]:
    """
    Decodes predictions under:
      - Baseline Phase 36C (VM=0.80, PA=0.80, other=0.60)
      - Suppressed VM search (VM=0.60, PA=0.80, other=0.60)
    
    Filters candidates into:
      - Candidate A: Normalized value in candidate_a_lexicon
      - Candidate B: Candidate A OR (Make context within +/- context_window tokens AND valid token)
      
    Asserts non-destructive additive property: baseline entities are preserved exactly.
    """
    base_spanned = decode_spanned_predictions(
        clean_p_fused,
        clean_examples,
        id2label,
        conf_thresh=0.60,
        aspect_thresholds={"Kompatibles_Fahrzeug_Modell": 0.80, "Produktart": 0.80},
    )

    vm060_spanned = decode_spanned_predictions(
        clean_p_fused,
        clean_examples,
        id2label,
        conf_thresh=0.60,
        aspect_thresholds={"Kompatibles_Fahrzeug_Modell": 0.60, "Produktart": 0.80},
    )

    cand_a_spanned: List[List[Dict[str, Any]]] = []
    cand_b_spanned: List[List[Dict[str, Any]]] = []
    audit_candidates: List[Dict[str, Any]] = []

    for i, ex in enumerate(clean_examples):
        b_ents = list(base_spanned[i])
        v_ents = vm060_spanned[i]
        b_keys = {(e["aspect"], e["start"], e["end"]) for e in b_ents}
        pred_makes = [e for e in b_ents if e["aspect"] == "Kompatible_Fahrzeug_Marke"]

        a_add = []
        b_add = []

        for p in v_ents:
            p_key = (p["aspect"], p["start"], p["end"])
            if p_key not in b_keys:
                # This is a suppressed candidate in [0.60, 0.80)
                val = p["value"].strip()
                s = p["start"]
                e = p["end"]

                passes_a = val in candidate_a_lexicon

                has_make_context = False
                matched_make = None
                for pm in pred_makes:
                    dist = min(abs(s - pm["end"]), abs(pm["start"] - e))
                    if dist <= context_window:
                        mk_val = pm["value"].strip().upper()
                        if (mk_val, val.upper()) in make_model_pairs or val in train_model_counts:
                            has_make_context = True
                            matched_make = pm["value"]
                            break

                is_ambiguous = (val.lower() in STOPWORDS) or bool(DECIMAL_REGEX.match(val))
                passes_b = passes_a or (has_make_context and not is_ambiguous)

                if passes_a:
                    a_add.append(p)
                if passes_b:
                    b_add.append(p)

                audit_candidates.append({
                    "record_idx": i,
                    "category": ex["category"],
                    "value": val,
                    "start": s,
                    "end": e,
                    "confidence": float(p.get("confidence", 0.0)),
                    "passes_a": passes_a,
                    "passes_b": passes_b,
                    "has_make_context": has_make_context,
                    "matched_make": matched_make,
                })

        cand_a_spanned.append(b_ents + a_add)
        cand_b_spanned.append(b_ents + b_add)

    # Enforce non-destructive addition invariant
    for i in range(len(clean_examples)):
        base_tuples = [(e["aspect"], e["value"]) for e in base_spanned[i]]
        cand_a_tuples = [(e["aspect"], e["value"]) for e in cand_a_spanned[i]]
        cand_b_tuples = [(e["aspect"], e["value"]) for e in cand_b_spanned[i]]

        b_cnt = Counter(base_tuples)
        a_cnt = Counter(cand_a_tuples)
        b_cnt_b = Counter(cand_b_tuples)

        for k, v in b_cnt.items():
            if a_cnt[k] < v:
                raise AssertionError(f"Candidate A violated non-destructive addition for record {i}, item {k}")
            if b_cnt_b[k] < v:
                raise AssertionError(f"Candidate B violated non-destructive addition for record {i}, item {k}")

    return base_spanned, cand_a_spanned, cand_b_spanned, audit_candidates


def run_paired_bootstrap(
    records_base: List[Dict[str, Any]],
    records_a: List[Dict[str, Any]],
    records_b: List[Dict[str, Any]],
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run paired title-level bootstrap resampling (seed 42) computing official competition scores.
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


def execute_phase_39b_evaluation(run_bootstrap: bool = True) -> Dict[str, Any]:
    """
    Executes the complete Phase 39B evaluation and audit pipeline.
    """
    # 1. Step 1 Integrity
    integrity = verify_phase_integrity()

    # 2. Load clean holdout and examples
    with open(CLEAN_HOLDOUT_MANIFEST_PATH, "r", encoding="utf-8") as f:
        clean_ids = set(json.load(f)["record_ids"])

    df = load_training_data()
    all_examples = build_examples(df)
    train_4k, val_1k = split_examples(all_examples, random_state=42)

    clean_indices = [i for i, ex in enumerate(val_1k) if ex["record_number"] in clean_ids]
    clean_examples = [val_1k[i] for i in clean_indices]

    # 3. Build training lexicon strictly from train_4k
    cand_a_lexicon, make_model_pairs, train_model_counts, train_make_counts = build_training_vehicle_lexicon(train_4k)

    # 4. Load cached fused probabilities
    probs = torch.load(VAL_PROBS_CACHE_PATH, map_location="cpu", weights_only=False)
    cfg = AutoConfig.from_pretrained("checkpoints/gbert_baseline_best")
    id2label = {int(k): v for k, v in cfg.id2label.items()}
    model_names = ["dapt_gbert", "xlmr_seed42", "xlmr_seed7", "xlmr_seed21"]
    model_weights = [0.40, 0.20, 0.20, 0.20]
    clean_p_fused = [
        sum(w * probs[m][i] for w, m in zip(model_weights, model_names)) for i in clean_indices
    ]

    # 5. Decode predictions & recover candidates
    base_spanned, cand_a_spanned, cand_b_spanned, audit_candidates = decode_predictions_and_candidates(
        clean_p_fused,
        clean_examples,
        id2label,
        cand_a_lexicon,
        make_model_pairs,
        train_model_counts,
        context_window=3,
    )

    # Gold entities
    spanned_gold = extract_spanned_gold_entities(clean_examples)
    gold_tuples = [[(e["aspect"], e["value"]) for e in ex] for ex in spanned_gold]
    base_tuples = [[(e["aspect"], e["value"]) for e in ex] for ex in base_spanned]
    cand_a_tuples = [[(e["aspect"], e["value"]) for e in ex] for ex in cand_a_spanned]
    cand_b_tuples = [[(e["aspect"], e["value"]) for e in ex] for ex in cand_b_spanned]

    records_base = [
        {"category": clean_examples[i]["category"], "predicted_entities": base_tuples[i], "gold_entities": gold_tuples[i]}
        for i in range(len(clean_examples))
    ]
    records_a = [
        {"category": clean_examples[i]["category"], "predicted_entities": cand_a_tuples[i], "gold_entities": gold_tuples[i]}
        for i in range(len(clean_examples))
    ]
    records_b = [
        {"category": clean_examples[i]["category"], "predicted_entities": cand_b_tuples[i], "gold_entities": gold_tuples[i]}
        for i in range(len(clean_examples))
    ]

    # Evaluate official & micro
    off_base = calculate_official_competition_metrics(records_base)
    off_a = calculate_official_competition_metrics(records_a)
    off_b = calculate_official_competition_metrics(records_b)

    mic_base = calculate_global_micro_metrics(records_base)
    mic_a = calculate_global_micro_metrics(records_a)
    mic_b = calculate_global_micro_metrics(records_b)

    # Audit candidate breakdown
    total_suppressed = len(audit_candidates)
    added_a = sum(1 for c in audit_candidates if c["passes_a"])
    added_b = sum(1 for c in audit_candidates if c["passes_b"])

    # Reconcile TP and FP of added candidates
    delta_tp_a = mic_a["tp"] - mic_base["tp"]
    delta_fp_a = mic_a["fp"] - mic_base["fp"]
    delta_fn_a = mic_a["fn"] - mic_base["fn"]

    delta_tp_b = mic_b["tp"] - mic_base["tp"]
    delta_fp_b = mic_b["fp"] - mic_base["fp"]
    delta_fn_b = mic_b["fn"] - mic_base["fn"]

    bootstrap_results = None
    if run_bootstrap:
        bootstrap_results = run_paired_bootstrap(records_base, records_a, records_b, n_resamples=1000, seed=42)

    # Decision rule logic
    # PROMISING LOCAL CHALLENGER only if:
    # 1. official delta >= +0.0020
    # 2. bootstrap 95% CI lower bound > 0
    # 3. no significant degradation in either category
    delta_off_a = off_a["official_score"] - off_base["official_score"]
    delta_off_b = off_b["official_score"] - off_base["official_score"]
    cat1_delta_a = off_a["categories"][1]["score"] - off_base["categories"][1]["score"]
    cat1_delta_b = off_b["categories"][1]["score"] - off_base["categories"][1]["score"]

    decision = "RETAIN PHASE 36C CHAMPION"
    decision_reason = []

    if delta_off_a >= 0.0020 and (bootstrap_results and bootstrap_results["candidate_a"]["ci_95"][0] > 0) and cat1_delta_a >= 0:
        decision = "PROMISING LOCAL CHALLENGER — A"
    elif delta_off_b >= 0.0020 and (bootstrap_results and bootstrap_results["candidate_b"]["ci_95"][0] > 0) and cat1_delta_b >= 0:
        decision = "PROMISING LOCAL CHALLENGER — B"
    else:
        decision = "RETAIN PHASE 36C CHAMPION"
        if delta_off_a < 0.0020:
            decision_reason.append(f"Candidate A official score delta ({delta_off_a:+.6f}) < threshold (+0.0020)")
        if cat1_delta_a < 0:
            decision_reason.append(f"Candidate A degrades Category 1 score by {cat1_delta_a:+.6f}")
        if bootstrap_results and bootstrap_results["candidate_a"]["ci_95"][0] <= 0:
            decision_reason.append(f"Candidate A 95% CI [{bootstrap_results['candidate_a']['ci_95'][0]:+.6f}, {bootstrap_results['candidate_a']['ci_95'][1]:+.6f}] spans zero (p={bootstrap_results['candidate_a']['p_val_le_zero']:.4f})")
        if delta_off_b < 0:
            decision_reason.append(f"Candidate B degrades overall official score by {delta_off_b:+.6f}")

    results = {
        "integrity": integrity,
        "lexicon_stats": {
            "total_raw_models": len(train_model_counts),
            "total_model_occurrences": sum(train_model_counts.values()),
            "total_raw_makes": len(train_make_counts),
            "make_model_pairs": len(make_model_pairs),
            "filtered_lexicon_size": len(cand_a_lexicon),
        },
        "suppressed_candidate_audit": {
            "total_suppressed_candidates": total_suppressed,
            "candidate_a_admitted": added_a,
            "candidate_a_tp": delta_tp_a,
            "candidate_a_fp": delta_fp_a,
            "candidate_a_precision": float(delta_tp_a / added_a) if added_a > 0 else 0.0,
            "candidate_b_admitted": added_b,
            "candidate_b_tp": delta_tp_b,
            "candidate_b_fp": delta_fp_b,
            "candidate_b_precision": float(delta_tp_b / added_b) if added_b > 0 else 0.0,
        },
        "metrics": {
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
                "cat1_vm_f02": float(off_base["categories"][1]["aspects"]["Kompatibles_Fahrzeug_Modell"]["f_beta"]),
                "cat2_vm_f02": float(off_base["categories"][2]["aspects"]["Kompatibles_Fahrzeug_Modell"]["f_beta"]),
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
                "cat1_vm_f02": float(off_a["categories"][1]["aspects"]["Kompatibles_Fahrzeug_Modell"]["f_beta"]),
                "cat2_vm_f02": float(off_a["categories"][2]["aspects"]["Kompatibles_Fahrzeug_Modell"]["f_beta"]),
                "delta_official": float(delta_off_a),
                "delta_cat1": float(cat1_delta_a),
                "delta_cat2": float(off_a["categories"][2]["score"] - off_base["categories"][2]["score"]),
                "delta_micro_f02": float(mic_a["global_f_beta"] - mic_base["global_f_beta"]),
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
                "cat1_vm_f02": float(off_b["categories"][1]["aspects"]["Kompatibles_Fahrzeug_Modell"]["f_beta"]),
                "cat2_vm_f02": float(off_b["categories"][2]["aspects"]["Kompatibles_Fahrzeug_Modell"]["f_beta"]),
                "delta_official": float(delta_off_b),
                "delta_cat1": float(cat1_delta_b),
                "delta_cat2": float(off_b["categories"][2]["score"] - off_base["categories"][2]["score"]),
                "delta_micro_f02": float(mic_b["global_f_beta"] - mic_base["global_f_beta"]),
            },
        },
        "bootstrap": bootstrap_results,
        "decision": decision,
        "decision_reasons": decision_reason,
    }

    # Save to local ignored cache
    os.makedirs(os.path.dirname(RESULTS_CACHE_PATH), exist_ok=True)
    with open(RESULTS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    print("Running Phase 39B Vehicle Model Recovery Pipeline...")
    res = execute_phase_39b_evaluation(run_bootstrap=True)
    print("\nPipeline execution complete.")
    print(f"Decision: {res['decision']}")
    print(f"Candidate A Official Delta: {res['metrics']['candidate_a']['delta_official']:+.6f}")
    print(f"Candidate B Official Delta: {res['metrics']['candidate_b']['delta_official']:+.6f}")
