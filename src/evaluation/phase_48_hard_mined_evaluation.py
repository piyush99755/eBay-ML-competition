"""
Phase 48: Standalone and Ensemble Evaluation of Hard-Mined GBERT.

Evaluates:
  1. Standalone Phase 48 GBERT against Phase 40 champion on:
     - 489-title Clean Holdout
     - Full 1,000-title Validation Set
  2. Predeclared 10% Challenger Ensemble:
     - 90% Phase 40 Fused Probabilities + 10% Phase 48 Challenger Probabilities
     - Under Frozen Phase 40 Aspect Thresholds:
       * Kompatibles_Fahrzeug_Modell: 0.80
       * Produktart: 0.80
       * Im_Lieferumfang_Enthalten: 0.80
       * All remaining 26 aspects: 0.60
  3. Paired Category-Stratified Bootstrap (1,000 resamples) with 95% CIs and p-values.
  4. Error diagnostic breakdown: boundary fixes, new FPs, aspect confusions, and complementarity.
"""

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_examples import build_examples, load_training_data
from src.data.make_splits import split_examples
from src.modeling.prepare_datasets import prepare_split
from src.evaluation.entity_metrics import (
    bio_to_entities_with_confidence,
    filter_entities_by_category,
    count_matches,
)
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

CHECKPOINT_PHASE48 = PROJECT_ROOT / "checkpoints" / "gbert_phase48_hard_mined_best"
PHASE48_PROBS_CACHE = PROJECT_ROOT / "data" / "cache" / "phase_48_val_1k_probs.pt"
EVALUATION_RESULTS_PATH = PROJECT_ROOT / "data" / "cache" / "phase_48_evaluation_results.json"

VAL_BATCH_SIZE = 8


def extract_phase48_validation_probabilities(
    checkpoint_path: Path,
    val_examples: List[dict],
    id2label: Dict[int, str],
    device: torch.device,
) -> List[np.ndarray]:
    """Extract or load cached validation token probabilities for Phase 48 GBERT."""
    if PHASE48_PROBS_CACHE.exists():
        print(f"Loading cached Phase 48 probabilities from {PHASE48_PROBS_CACHE}...")
        raw_probs = torch.load(PHASE48_PROBS_CACHE, map_location="cpu", weights_only=False)
        return [np.array(p, dtype=np.float32) for p in raw_probs]

    print(f"Extracting validation probabilities from {checkpoint_path}...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)
    label2id = {v: k for k, v in id2label.items()}

    val_dataset, _ = prepare_split(val_examples, tokenizer, label2id)
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    dataloader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    model = AutoModelForTokenClassification.from_pretrained(checkpoint_path, local_files_only=True)
    model.to(device)
    model.eval()

    all_probs = []
    ex_idx_global = 0

    with torch.no_grad():
        for batch in dataloader:
            batch_dev = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            outputs = model(
                input_ids=batch_dev["input_ids"],
                attention_mask=batch_dev["attention_mask"],
                token_type_ids=batch_dev.get("token_type_ids"),
            )
            logits = outputs.logits
            batch_probs = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)

            for row_idx in range(batch_probs.shape[0]):
                ex = val_examples[ex_idx_global]
                ex_idx_global += 1

                row_labels = batch_dev["labels"][row_idx].cpu().numpy()
                active_mask = (row_labels != -100)
                active_probs = batch_probs[row_idx, active_mask]

                assert len(active_probs) == len(ex["tokens"]), (
                    f"Token length mismatch: {len(active_probs)} vs {len(ex['tokens'])}"
                )
                all_probs.append(active_probs)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save cache
    torch.save(all_probs, PHASE48_PROBS_CACHE)
    print(f"Saved Phase 48 validation probabilities to {PHASE48_PROBS_CACHE}")
    return all_probs


def run_paired_bootstrap(
    records_baseline: List[dict],
    records_challenger: List[dict],
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute paired category-stratified bootstrap resampling."""
    rng = np.random.RandomState(seed)
    n = len(records_baseline)
    assert len(records_challenger) == n

    cat1_indices = [i for i, r in enumerate(records_baseline) if r["category"] == 1]
    cat2_indices = [i for i, r in enumerate(records_baseline) if r["category"] == 2]

    base_scores = []
    chal_scores = []
    deltas = []

    for _ in range(n_resamples):
        resampled_cat1 = rng.choice(cat1_indices, size=len(cat1_indices), replace=True)
        resampled_cat2 = rng.choice(cat2_indices, size=len(cat2_indices), replace=True)
        sample_indices = np.concatenate([resampled_cat1, resampled_cat2])

        sample_base = [records_baseline[i] for i in sample_indices]
        sample_chal = [records_challenger[i] for i in sample_indices]

        m_base = calculate_official_competition_metrics(sample_base)["official_score"]
        m_chal = calculate_official_competition_metrics(sample_chal)["official_score"]

        base_scores.append(m_base)
        chal_scores.append(m_chal)
        deltas.append(m_chal - m_base)

    deltas = np.array(deltas)
    mean_delta = float(np.mean(deltas))
    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))
    p_value = float(np.mean(deltas <= 0.0))

    return {
        "n_resamples": n_resamples,
        "mean_delta": mean_delta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "is_significant": (ci_lower > 0.0),
    }


def run_phase_48_evaluation() -> Dict[str, Any]:
    print("=" * 80)
    print("PHASE 48: RIGOROUS EVALUATION OF HARD-MINED GBERT CHALLENGER")
    print("=" * 80)

    # 1. Champion Baseline Verification
    print("\n[Step 1] Verifying Phase 40 champion baseline...")
    champ_info = verify_champion()
    print(f"Phase 40 Champion SHA-256: {champ_info['sha256']} (VERIFIED INTACT)")
    print(f"Official Quiz Score:        {champ_info['official_quiz_f02']}")

    # 2. Load Datasets
    print("\n[Step 2] Loading validation dataset and clean holdout manifest...")
    df = load_training_data()
    all_examples = build_examples(df)
    train_4k, val_1k = split_examples(all_examples, random_state=42)

    with open(CLEAN_HOLDOUT_MANIFEST_PATH, "r", encoding="utf-8") as f:
        clean_ids = set(json.load(f)["record_ids"])

    clean_indices = [i for i, ex in enumerate(val_1k) if ex["record_number"] in clean_ids]
    clean_ex = [val_1k[i] for i in clean_indices]

    print(f"Total Validation Titles: {len(val_1k)}")
    print(f"Clean Holdout Titles:    {len(clean_ex)} (Category 1: {sum(1 for e in clean_ex if e['category']==1)}, Category 2: {sum(1 for e in clean_ex if e['category']==2)})")

    # 3. Label Map & Device
    ref_cfg = AutoConfig.from_pretrained("checkpoints/gbert_baseline_best")
    id2label = {int(k): str(v) for k, v in ref_cfg.id2label.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4. Extract / Load Probabilities
    print("\n[Step 3] Loading token probabilities...")
    # Load Phase 40 model probabilities
    val_probs_cache = torch.load(VAL_PROBS_CACHE_PATH, map_location="cpu", weights_only=False)
    model_names = ["dapt_gbert", "xlmr_seed42", "xlmr_seed7", "xlmr_seed21"]
    model_weights = [0.40, 0.20, 0.20, 0.20]

    val_p_phase40 = [
        sum(w * val_probs_cache[m][i] for w, m in zip(model_weights, model_names))
        for i in range(len(val_1k))
    ]
    clean_p_phase40 = [val_p_phase40[i] for i in clean_indices]

    # Load Phase 48 challenger probabilities
    val_p_phase48 = extract_phase48_validation_probabilities(
        checkpoint_path=CHECKPOINT_PHASE48,
        val_examples=val_1k,
        id2label=id2label,
        device=device,
    )
    clean_p_phase48 = [val_p_phase48[i] for i in clean_indices]

    # 5. Decode Gold Entities
    val_spanned_gold = extract_spanned_gold_entities(val_1k)
    val_gold_tuples = [[(e["aspect"], e["value"]) for e in ex] for ex in val_spanned_gold]

    clean_spanned_gold = extract_spanned_gold_entities(clean_ex)
    clean_gold_tuples = [[(e["aspect"], e["value"]) for e in ex] for ex in clean_spanned_gold]

    # 6. Step A: Phase 40 Baseline Reproduction
    print("\n" + "=" * 60)
    print("STEP A: PHASE 40 BASELINE REPRODUCTION")
    print("=" * 60)
    clean_preds_p40 = decode_spanned_predictions_with_cat_thresholds(
        clean_p_phase40,
        examples=clean_ex,
        id2label=id2label,
        conf_thresh=DEFAULT_CONF_THRESH,
        aspect_thresholds=PHASE40_THRESHOLDS,
    )
    clean_tuples_p40 = [[(e["aspect"], e["value"]) for e in ex] for ex in clean_preds_p40]

    records_clean_p40 = [
        {"category": clean_ex[i]["category"], "predicted_entities": clean_tuples_p40[i], "gold_entities": clean_gold_tuples[i]}
        for i in range(len(clean_ex))
    ]
    off_clean_p40 = calculate_official_competition_metrics(records_clean_p40)
    mic_clean_p40 = calculate_global_micro_metrics(records_clean_p40)

    print(f"Clean Holdout Official Score: {off_clean_p40['official_score']:.6f} (Expected: 0.927771)")
    print(f"  Category 1 Score:           {off_clean_p40['categories'][1]['score']:.6f} (Expected: 0.918323)")
    print(f"  Category 2 Score:           {off_clean_p40['categories'][2]['score']:.6f} (Expected: 0.937219)")
    print(f"  TP={mic_clean_p40['tp']}, FP={mic_clean_p40['fp']}, FN={mic_clean_p40['fn']}")

    assert abs(off_clean_p40["official_score"] - 0.927771) < 1e-4, "Baseline reproduction failed!"
    assert mic_clean_p40["tp"] == 3021 and mic_clean_p40["fp"] == 165 and mic_clean_p40["fn"] == 752
    print("Phase 40 Baseline Reproduced Perfectly (0.927771)!")

    # 7. Step B: Standalone Phase 48 GBERT Evaluation
    print("\n" + "=" * 60)
    print("STEP B: STANDALONE PHASE 48 GBERT EVALUATION")
    print("=" * 60)
    clean_preds_p48 = decode_spanned_predictions_with_cat_thresholds(
        clean_p_phase48,
        examples=clean_ex,
        id2label=id2label,
        conf_thresh=DEFAULT_CONF_THRESH,
        aspect_thresholds=PHASE40_THRESHOLDS,
    )
    clean_tuples_p48 = [[(e["aspect"], e["value"]) for e in ex] for ex in clean_preds_p48]

    records_clean_p48 = [
        {"category": clean_ex[i]["category"], "predicted_entities": clean_tuples_p48[i], "gold_entities": clean_gold_tuples[i]}
        for i in range(len(clean_ex))
    ]
    off_clean_p48 = calculate_official_competition_metrics(records_clean_p48)
    mic_clean_p48 = calculate_global_micro_metrics(records_clean_p48)

    p_clean_p48 = mic_clean_p48["tp"] / max(1, mic_clean_p48["tp"] + mic_clean_p48["fp"])
    r_clean_p48 = mic_clean_p48["tp"] / max(1, mic_clean_p48["tp"] + mic_clean_p48["fn"])

    print(f"Standalone Phase 48 Clean Holdout Score: {off_clean_p48['official_score']:.6f}")
    print(f"  Category 1 Score:                      {off_clean_p48['categories'][1]['score']:.6f}")
    print(f"  Category 2 Score:                      {off_clean_p48['categories'][2]['score']:.6f}")
    print(f"  Precision: {p_clean_p48:.4f} | Recall: {r_clean_p48:.4f}")
    print(f"  TP={mic_clean_p48['tp']}, FP={mic_clean_p48['fp']}, FN={mic_clean_p48['fn']}")
    print(f"  Delta vs Phase 40 4-Model Ensemble:     {off_clean_p48['official_score'] - off_clean_p40['official_score']:+.6f}")

    # Full 1,000-title validation evaluation
    val_preds_p48 = decode_spanned_predictions_with_cat_thresholds(
        val_p_phase48,
        examples=val_1k,
        id2label=id2label,
        conf_thresh=DEFAULT_CONF_THRESH,
        aspect_thresholds=PHASE40_THRESHOLDS,
    )
    val_tuples_p48 = [[(e["aspect"], e["value"]) for e in ex] for ex in val_preds_p48]
    val_preds_p40 = decode_spanned_predictions_with_cat_thresholds(
        val_p_phase40,
        examples=val_1k,
        id2label=id2label,
        conf_thresh=DEFAULT_CONF_THRESH,
        aspect_thresholds=PHASE40_THRESHOLDS,
    )
    val_tuples_p40 = [[(e["aspect"], e["value"]) for e in ex] for ex in val_preds_p40]

    records_val_p48 = [
        {"category": val_1k[i]["category"], "predicted_entities": val_tuples_p48[i], "gold_entities": val_gold_tuples[i]}
        for i in range(len(val_1k))
    ]
    records_val_p40 = [
        {"category": val_1k[i]["category"], "predicted_entities": val_tuples_p40[i], "gold_entities": val_gold_tuples[i]}
        for i in range(len(val_1k))
    ]
    off_val_p48 = calculate_official_competition_metrics(records_val_p48)
    off_val_p40 = calculate_official_competition_metrics(records_val_p40)

    print(f"\nFull 1,000-Title Validation:")
    print(f"  Phase 40 4-Model Ensemble: {off_val_p40['official_score']:.6f}")
    print(f"  Phase 48 Standalone GBERT: {off_val_p48['official_score']:.6f} (Delta: {off_val_p48['official_score'] - off_val_p40['official_score']:+.6f})")

    # 8. Step C: Predeclared Ensemble Evaluation (90% Phase 40 + 10% Phase 48)
    print("\n" + "=" * 60)
    print("STEP C: PREDECLARED ENSEMBLE TEST (90% Phase 40 + 10% Phase 48)")
    print("=" * 60)
    clean_p_ens = [
        0.90 * clean_p_phase40[i] + 0.10 * clean_p_phase48[i]
        for i in range(len(clean_indices))
    ]
    clean_preds_ens = decode_spanned_predictions_with_cat_thresholds(
        clean_p_ens,
        examples=clean_ex,
        id2label=id2label,
        conf_thresh=DEFAULT_CONF_THRESH,
        aspect_thresholds=PHASE40_THRESHOLDS,
    )
    clean_tuples_ens = [[(e["aspect"], e["value"]) for e in ex] for ex in clean_preds_ens]

    records_clean_ens = [
        {"category": clean_ex[i]["category"], "predicted_entities": clean_tuples_ens[i], "gold_entities": clean_gold_tuples[i]}
        for i in range(len(clean_ex))
    ]
    off_clean_ens = calculate_official_competition_metrics(records_clean_ens)
    mic_clean_ens = calculate_global_micro_metrics(records_clean_ens)

    ens_delta_overall = off_clean_ens["official_score"] - off_clean_p40["official_score"]
    ens_delta_cat1 = off_clean_ens["categories"][1]["score"] - off_clean_p40["categories"][1]["score"]
    ens_delta_cat2 = off_clean_ens["categories"][2]["score"] - off_clean_p40["categories"][2]["score"]

    print(f"Ensemble Clean Holdout Score: {off_clean_ens['official_score']:.6f} (Delta: {ens_delta_overall:+.6f})")
    print(f"  Category 1 Score:            {off_clean_ens['categories'][1]['score']:.6f} (Delta: {ens_delta_cat1:+.6f})")
    print(f"  Category 2 Score:            {off_clean_ens['categories'][2]['score']:.6f} (Delta: {ens_delta_cat2:+.6f})")
    print(f"  TP: {mic_clean_ens['tp']} ({mic_clean_ens['tp'] - mic_clean_p40['tp']:+d})")
    print(f"  FP: {mic_clean_ens['fp']} ({mic_clean_ens['fp'] - mic_clean_p40['fp']:+d})")
    print(f"  FN: {mic_clean_ens['fn']} ({mic_clean_ens['fn'] - mic_clean_p40['fn']:+d})")

    # 9. Step D: Paired Category-Stratified Bootstrap Resampling
    print("\n" + "=" * 60)
    print("STEP D: STATISTICAL BOOTSTRAP (1,000 Paired Resamples)")
    print("=" * 60)
    bootstrap_results = run_paired_bootstrap(
        records_baseline=records_clean_p40,
        records_challenger=records_clean_ens,
        n_resamples=1000,
        seed=42,
    )
    print(f"Bootstrap Mean Delta: {bootstrap_results['mean_delta']:+.6f}")
    print(f"95% Confidence Interval: [{bootstrap_results['ci_lower']:+.6f}, {bootstrap_results['ci_upper']:+.6f}]")
    print(f"Empirical p-value (Delta <= 0): {bootstrap_results['p_value']:.4f}")
    print(f"Statistically Significant (> 0 at 95%): {bootstrap_results['is_significant']}")

    # 10. Step E: Title-by-Title Error & Complementarity Audit
    print("\n" + "=" * 60)
    print("STEP E: TITLE-BY-TITLE COMPLEMENTARITY AUDIT")
    print("=" * 60)

    changed_titles = []
    repaired_titles = []
    regressed_titles = []
    boundary_fix_count = 0
    new_fp_count = 0

    for i in range(len(clean_ex)):
        rec_id = clean_ex[i]["record_number"]
        cat_id = clean_ex[i]["category"]
        gold = clean_gold_tuples[i]
        p40_pred = clean_tuples_p40[i]
        ens_pred = clean_tuples_ens[i]

        if p40_pred != ens_pred:
            tp_base, fp_base, fn_base = count_matches(p40_pred, gold)
            tp_ens, fp_ens, fn_ens = count_matches(ens_pred, gold)

            base_f = (1.04 * tp_base) / max(1e-6, 1.04 * tp_base + 0.04 * fp_base + fn_base)
            ens_f = (1.04 * tp_ens) / max(1e-6, 1.04 * tp_ens + 0.04 * fp_ens + fn_ens)
            delta_f = ens_f - base_f

            p40_set = Counter(p40_pred)
            ens_set = Counter(ens_pred)
            added = list((ens_set - p40_set).elements())
            removed = list((p40_set - ens_set).elements())

            info = {
                "record_number": rec_id,
                "category": cat_id,
                "title": " ".join(clean_ex[i]["tokens"]),
                "gold": gold,
                "p40_pred": p40_pred,
                "ens_pred": ens_pred,
                "added": added,
                "removed": removed,
                "delta_f02": delta_f,
                "fp_delta": fp_ens - fp_base,
                "fn_delta": fn_ens - fn_base,
            }

            changed_titles.append(info)
            if delta_f > 1e-4:
                repaired_titles.append(info)
            elif delta_f < -1e-4:
                regressed_titles.append(info)

            if fp_ens > fp_base:
                new_fp_count += (fp_ens - fp_base)

    print(f"Clean Holdout Titles Evaluated: {len(clean_ex)}")
    print(f"Titles Changed by Ensemble:     {len(changed_titles)} ({len(changed_titles)/len(clean_ex)*100:.2f}%)")
    print(f"  Repaired Titles (Score Up):   {len(repaired_titles)}")
    print(f"  Regressed Titles (Score Down): {len(regressed_titles)}")
    print(f"  Neutral Changes:              {len(changed_titles) - len(repaired_titles) - len(regressed_titles)}")
    print(f"Newly Introduced False Positives: {new_fp_count}")

    # Inspect Changed Samples
    print("\nSample Changes (first 5):")
    for sample in changed_titles[:5]:
        print(f"  [Record {sample['record_number']}, Cat {sample['category']}] Delta F0.2: {sample['delta_f02']:+.4f}:")
        print(f"    Title:     {sample['title']}")
        print(f"    Gold:      {sample['gold']}")
        print(f"    Phase 40:  {sample['p40_pred']}")
        print(f"    Ensemble:  {sample['ens_pred']}")

    # 11. Decision Criteria Evaluation
    print("\n" + "=" * 60)
    print("STEP F: DECISION VERDICT")
    print("=" * 60)

    # Criteria:
    # 1. Genuinely complementary predictions?
    # 2. Meaningful official F0.2 improvement? (e.g. > +0.0005)
    # 3. Controlled false positives? (Net FP delta <= 0)
    # 4. Statistically significant bootstrap?
    # 5. No serious category regression? (Both Cat 1 and Cat 2 non-negative)
    meaningful_improvement = ens_delta_overall >= 0.0005
    fp_controlled = (mic_clean_ens["fp"] <= mic_clean_p40["fp"])
    statistically_significant = bootstrap_results["is_significant"]
    no_cat_regression = (ens_delta_cat1 >= -0.0005 and ens_delta_cat2 >= -0.0005)

    if meaningful_improvement and fp_controlled and statistically_significant and no_cat_regression:
        verdict = "PROMISING CHALLENGER — QUIZ TEST JUSTIFIED"
    elif ens_delta_overall <= 0.0002 or not no_cat_regression or mic_clean_ens["fp"] > mic_clean_p40["fp"]:
        verdict = "RETAIN PHASE 40 — NO IMPROVEMENT"
    else:
        verdict = "INCONCLUSIVE — TRAINING OR VALIDATION LIMITATION"

    print(f"Decision Verdict: >>> {verdict} <<<")

    # 12. Save Results Cache
    results = {
        "phase": "48",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "champion_sha256": champ_info["sha256"],
        "official_quiz_champion_score": champ_info["official_quiz_f02"],
        "baseline_clean_holdout": {
            "official_score": off_clean_p40["official_score"],
            "cat1_score": off_clean_p40["categories"][1]["score"],
            "cat2_score": off_clean_p40["categories"][2]["score"],
            "tp": mic_clean_p40["tp"],
            "fp": mic_clean_p40["fp"],
            "fn": mic_clean_p40["fn"],
        },
        "standalone_phase48_clean_holdout": {
            "official_score": off_clean_p48["official_score"],
            "cat1_score": off_clean_p48["categories"][1]["score"],
            "cat2_score": off_clean_p48["categories"][2]["score"],
            "tp": mic_clean_p48["tp"],
            "fp": mic_clean_p48["fp"],
            "fn": mic_clean_p48["fn"],
            "delta_vs_phase40": off_clean_p48["official_score"] - off_clean_p40["official_score"],
        },
        "ensemble_clean_holdout": {
            "official_score": off_clean_ens["official_score"],
            "cat1_score": off_clean_ens["categories"][1]["score"],
            "cat2_score": off_clean_ens["categories"][2]["score"],
            "delta_overall": ens_delta_overall,
            "delta_cat1": ens_delta_cat1,
            "delta_cat2": ens_delta_cat2,
            "tp": mic_clean_ens["tp"],
            "fp": mic_clean_ens["fp"],
            "fn": mic_clean_ens["fn"],
            "tp_delta": mic_clean_ens["tp"] - mic_clean_p40["tp"],
            "fp_delta": mic_clean_ens["fp"] - mic_clean_p40["fp"],
            "fn_delta": mic_clean_ens["fn"] - mic_clean_p40["fn"],
        },
        "bootstrap": bootstrap_results,
        "complementarity": {
            "total_titles": len(clean_ex),
            "changed_titles_count": len(changed_titles),
            "repaired_titles_count": len(repaired_titles),
            "regressed_titles_count": len(regressed_titles),
            "new_fp_count": new_fp_count,
            "sample_changes": changed_titles[:10],
        },
        "verdict": verdict,
    }

    # Clean helper for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        elif isinstance(obj, tuple):
            return [convert_to_serializable(i) for i in obj]
        return obj

    results = convert_to_serializable(results)

    EVALUATION_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVALUATION_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nEvaluation results saved to {EVALUATION_RESULTS_PATH}")

    return results


if __name__ == "__main__":
    run_phase_48_evaluation()
