"""
Phase 49: Precision-Aware Generalization, Complementarity, and Bootstrap Evaluation.

Evaluates:
  1. Standalone Primary Evaluation on 800-title Internal Dev Partition
     against strictly matched DAPT GBERT baseline (checkpoints/phase33b_dapt_gbert_best).
  2. Paired Bootstrap Analysis on 800-title Internal Dev Partition (1,000 resamples).
  3. Secondary Diagnostic on 489-title Clean Holdout:
     - Standalone Phase 49 vs Phase 40 Champion Baseline
     - Predeclared 10% Ensemble Blend: 90% Phase 40 + 10% Phase 49
     - 1,000 Paired Category-Stratified Bootstrap Resamples
     - Title-by-title complementarity and false-positive audit.
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
from src.data.make_internal_split import get_internal_split
from src.modeling.prepare_datasets import prepare_split
from src.evaluation.entity_metrics import count_matches
from src.evaluation.phase_35b_audit_and_error_analysis import extract_spanned_gold_entities
from src.evaluation.phase_39a_official_metric import (
    calculate_official_competition_metrics,
    calculate_global_micro_metrics,
)
from src.evaluation.phase_43_part_number_calibration import (
    decode_spanned_predictions_with_cat_thresholds,
    PHASE40_THRESHOLDS,
    DEFAULT_CONF_THRESH,
    CLEAN_HOLDOUT_MANIFEST_PATH,
    VAL_PROBS_CACHE_PATH,
    verify_champion,
)

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CHECKPOINT_PHASE49 = PROJECT_ROOT / "checkpoints" / "gbert_phase49_precision_aware_best"
CHECKPOINT_MATCHED_BASELINE = PROJECT_ROOT / "checkpoints" / "phase33b_dapt_gbert_best"
DEV_PROBS_CACHE_PHASE49 = CACHE_DIR / "phase_49_dev_800_probs.pt"
VAL_PROBS_CACHE_PHASE49 = CACHE_DIR / "phase_49_val_1k_probs.pt"
EVALUATION_RESULTS_PATH = CACHE_DIR / "phase_49_evaluation_results.json"

VAL_BATCH_SIZE = 8


def extract_probabilities(
    checkpoint_path: Path,
    examples: List[dict],
    id2label: Dict[int, str],
    device: torch.device,
    cache_path: Optional[Path] = None,
) -> List[np.ndarray]:
    """Extract or load cached token probabilities."""
    if cache_path is not None and cache_path.exists():
        print(f"Loading cached probabilities from {cache_path}...")
        raw_probs = torch.load(cache_path, map_location="cpu", weights_only=False)
        return [np.array(p, dtype=np.float32) for p in raw_probs]

    print(f"Extracting probabilities from {checkpoint_path} ({len(examples)} titles)...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)
    label2id = {v: k for k, v in id2label.items()}

    dataset, _ = prepare_split(examples, tokenizer, label2id)
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
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
            logits = model(**batch_dev).logits
            batch_probs = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)

            for row_idx in range(batch_probs.shape[0]):
                ex = examples[ex_idx_global]
                ex_idx_global += 1

                row_labels = batch_dev["labels"][row_idx].cpu().numpy()
                active_mask = (row_labels != -100)
                active_probs = batch_probs[row_idx, active_mask]

                assert len(active_probs) == len(ex["tokens"])
                all_probs.append(active_probs)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if cache_path is not None:
        torch.save(all_probs, cache_path)
        print(f"Saved probabilities to {cache_path}")

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
        "is_significant": bool(ci_lower > 0.0),
    }


def run_phase_49_evaluation() -> Dict[str, Any]:
    print("=" * 80)
    print("PHASE 49: GENERALIZATION, COMPLEMENTARITY & BOOTSTRAP EVALUATION")
    print("=" * 80)

    # 1. Champion Baseline Verification
    print("\n[Step 1] Verifying Phase 40 champion baseline...")
    champ_info = verify_champion()
    print(f"Phase 40 Champion SHA-256: {champ_info['sha256']} (VERIFIED INTACT)")
    print(f"Official Quiz Score:        {champ_info['official_quiz_f02']}")

    # 2. Setup Data Partitions
    print("\n[Step 2] Setting up data partitions...")
    df = load_training_data()
    all_examples = build_examples(df)
    train_4k, val_1k = split_examples(all_examples, random_state=42)
    train_3200, dev_800, _ = get_internal_split(train_4k, random_state=42)

    with open(CLEAN_HOLDOUT_MANIFEST_PATH, "r", encoding="utf-8") as f:
        clean_ids = set(json.load(f)["record_ids"])

    clean_indices = [i for i, ex in enumerate(val_1k) if ex["record_number"] in clean_ids]
    clean_ex = [val_1k[i] for i in clean_indices]

    print(f"Internal Dev Partition:  {len(dev_800)} titles (400 Cat 1, 400 Cat 2) [PRIMARY]")
    print(f"Clean Holdout Partition: {len(clean_ex)} titles (305 Cat 1, 184 Cat 2) [SECONDARY DIAGNOSTIC]")

    # Label space
    ref_cfg = AutoConfig.from_pretrained("checkpoints/gbert_baseline_best")
    id2label = {int(k): str(v) for k, v in ref_cfg.id2label.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------------------
    # SECTION 1: PRIMARY EVALUATION ON 800 INTERNAL DEV
    # ----------------------------------------------------
    print("\n" + "=" * 60)
    print("SECTION 1: PRIMARY EVALUATION ON 800 INTERNAL DEV")
    print("=" * 60)

    # Gold entities for dev
    dev_gold_spanned = extract_spanned_gold_entities(dev_800)
    dev_gold_tuples = [[(e["aspect"], e["value"]) for e in ex] for ex in dev_gold_spanned]

    # Matched Baseline (Phase 33B DAPT GBERT) on dev
    dev_p_base = extract_probabilities(
        checkpoint_path=CHECKPOINT_MATCHED_BASELINE,
        examples=dev_800,
        id2label=id2label,
        device=device,
        cache_path=CACHE_DIR / "phase_33b_dev_800_probs.pt",
    )
    dev_preds_base = decode_spanned_predictions_with_cat_thresholds(
        dev_p_base, examples=dev_800, id2label=id2label,
        conf_thresh=DEFAULT_CONF_THRESH, aspect_thresholds=PHASE40_THRESHOLDS,
    )
    dev_tuples_base = [[(e["aspect"], e["value"]) for e in ex] for ex in dev_preds_base]
    records_dev_base = [
        {"category": dev_800[i]["category"], "predicted_entities": dev_tuples_base[i], "gold_entities": dev_gold_tuples[i]}
        for i in range(len(dev_800))
    ]
    off_dev_base = calculate_official_competition_metrics(records_dev_base)
    mic_dev_base = calculate_global_micro_metrics(records_dev_base)

    # Phase 49 Precision-Aware GBERT on dev
    dev_p_p49 = extract_probabilities(
        checkpoint_path=CHECKPOINT_PHASE49,
        examples=dev_800,
        id2label=id2label,
        device=device,
        cache_path=DEV_PROBS_CACHE_PHASE49,
    )
    dev_preds_p49 = decode_spanned_predictions_with_cat_thresholds(
        dev_p_p49, examples=dev_800, id2label=id2label,
        conf_thresh=DEFAULT_CONF_THRESH, aspect_thresholds=PHASE40_THRESHOLDS,
    )
    dev_tuples_p49 = [[(e["aspect"], e["value"]) for e in ex] for ex in dev_preds_p49]
    records_dev_p49 = [
        {"category": dev_800[i]["category"], "predicted_entities": dev_tuples_p49[i], "gold_entities": dev_gold_tuples[i]}
        for i in range(len(dev_800))
    ]
    off_dev_p49 = calculate_official_competition_metrics(records_dev_p49)
    mic_dev_p49 = calculate_global_micro_metrics(records_dev_p49)

    delta_dev_overall = off_dev_p49["official_score"] - off_dev_base["official_score"]
    delta_dev_cat1 = off_dev_p49["categories"][1]["score"] - off_dev_base["categories"][1]["score"]
    delta_dev_cat2 = off_dev_p49["categories"][2]["score"] - off_dev_base["categories"][2]["score"]

    print(f"\nInternal Dev (800 titles) Results:")
    print(f"  Matched Baseline DAPT GBERT: {off_dev_base['official_score']:.6f} (Cat 1: {off_dev_base['categories'][1]['score']:.6f}, Cat 2: {off_dev_base['categories'][2]['score']:.6f})")
    print(f"    TP={mic_dev_base['tp']}, FP={mic_dev_base['fp']}, FN={mic_dev_base['fn']}")
    print(f"  Phase 49 Precision-Aware:    {off_dev_p49['official_score']:.6f} (Cat 1: {off_dev_p49['categories'][1]['score']:.6f}, Cat 2: {off_dev_p49['categories'][2]['score']:.6f})")
    print(f"    TP={mic_dev_p49['tp']} ({mic_dev_p49['tp'] - mic_dev_base['tp']:+d}), FP={mic_dev_p49['fp']} ({mic_dev_p49['fp'] - mic_dev_base['fp']:+d}), FN={mic_dev_p49['fn']} ({mic_dev_p49['fn'] - mic_dev_base['fn']:+d})")
    print(f"  Delta on Internal Dev:       Overall: {delta_dev_overall:+.6f} | Cat 1: {delta_dev_cat1:+.6f} | Cat 2: {delta_dev_cat2:+.6f}")

    # Paired Bootstrap on Internal Dev
    boot_dev = run_paired_bootstrap(records_dev_base, records_dev_p49, n_resamples=1000, seed=42)
    print(f"  Internal Dev Bootstrap (1,000 resamples):")
    print(f"    Mean Delta: {boot_dev['mean_delta']:+.6f} | 95% CI: [{boot_dev['ci_lower']:+.6f}, {boot_dev['ci_upper']:+.6f}] | p-value: {boot_dev['p_value']:.4f}")

    # ----------------------------------------------------
    # SECTION 2: SECONDARY DIAGNOSTIC ON 489 CLEAN HOLDOUT
    # ----------------------------------------------------
    print("\n" + "=" * 60)
    print("SECTION 2: SECONDARY DIAGNOSTIC ON 489 CLEAN HOLDOUT")
    print("=" * 60)

    clean_spanned_gold = extract_spanned_gold_entities(clean_ex)
    clean_gold_tuples = [[(e["aspect"], e["value"]) for e in ex] for ex in clean_spanned_gold]

    # Phase 40 champion probabilities
    val_probs_cache = torch.load(VAL_PROBS_CACHE_PATH, map_location="cpu", weights_only=False)
    p40_models = ["dapt_gbert", "xlmr_seed42", "xlmr_seed7", "xlmr_seed21"]
    p40_weights = [0.40, 0.20, 0.20, 0.20]
    val_p_p40 = [
        sum(w * val_probs_cache[m][i] for w, m in zip(p40_weights, p40_models))
        for i in range(len(val_1k))
    ]
    clean_p_p40 = [val_p_p40[i] for i in clean_indices]

    # Phase 40 baseline decoded on clean holdout
    clean_preds_p40 = decode_spanned_predictions_with_cat_thresholds(
        clean_p_p40, examples=clean_ex, id2label=id2label,
        conf_thresh=DEFAULT_CONF_THRESH, aspect_thresholds=PHASE40_THRESHOLDS,
    )
    clean_tuples_p40 = [[(e["aspect"], e["value"]) for e in ex] for ex in clean_preds_p40]
    records_clean_p40 = [
        {"category": clean_ex[i]["category"], "predicted_entities": clean_tuples_p40[i], "gold_entities": clean_gold_tuples[i]}
        for i in range(len(clean_ex))
    ]
    off_clean_p40 = calculate_official_competition_metrics(records_clean_p40)
    mic_clean_p40 = calculate_global_micro_metrics(records_clean_p40)

    # Standalone Phase 49 on clean holdout
    val_p_p49 = extract_probabilities(
        checkpoint_path=CHECKPOINT_PHASE49,
        examples=val_1k,
        id2label=id2label,
        device=device,
        cache_path=VAL_PROBS_CACHE_PHASE49,
    )
    clean_p_p49 = [val_p_p49[i] for i in clean_indices]

    clean_preds_p49 = decode_spanned_predictions_with_cat_thresholds(
        clean_p_p49, examples=clean_ex, id2label=id2label,
        conf_thresh=DEFAULT_CONF_THRESH, aspect_thresholds=PHASE40_THRESHOLDS,
    )
    clean_tuples_p49 = [[(e["aspect"], e["value"]) for e in ex] for ex in clean_preds_p49]
    records_clean_p49 = [
        {"category": clean_ex[i]["category"], "predicted_entities": clean_tuples_p49[i], "gold_entities": clean_gold_tuples[i]}
        for i in range(len(clean_ex))
    ]
    off_clean_p49 = calculate_official_competition_metrics(records_clean_p49)
    mic_clean_p49 = calculate_global_micro_metrics(records_clean_p49)

    print(f"\nClean Holdout (489 titles) Results:")
    print(f"  Phase 40 Champion Baseline:  {off_clean_p40['official_score']:.6f} (Cat 1: {off_clean_p40['categories'][1]['score']:.6f}, Cat 2: {off_clean_p40['categories'][2]['score']:.6f})")
    print(f"    TP={mic_clean_p40['tp']}, FP={mic_clean_p40['fp']}, FN={mic_clean_p40['fn']}")
    print(f"  Phase 49 Standalone GBERT:   {off_clean_p49['official_score']:.6f} (Cat 1: {off_clean_p49['categories'][1]['score']:.6f}, Cat 2: {off_clean_p49['categories'][2]['score']:.6f})")
    print(f"    TP={mic_clean_p49['tp']} ({mic_clean_p49['tp'] - mic_clean_p40['tp']:+d}), FP={mic_clean_p49['fp']} ({mic_clean_p49['fp'] - mic_clean_p40['fp']:+d}), FN={mic_clean_p49['fn']} ({mic_clean_p49['fn'] - mic_clean_p40['fn']:+d})")
    print(f"  Standalone Delta vs P40:     {off_clean_p49['official_score'] - off_clean_p40['official_score']:+.6f}")

    # ----------------------------------------------------
    # SECTION 3: PREDECLARED ENSEMBLE TEST (90/10)
    # ----------------------------------------------------
    print("\n" + "=" * 60)
    print("SECTION 3: PREDECLARED ENSEMBLE TEST (90% Phase 40 + 10% Phase 49)")
    print("=" * 60)

    clean_p_ens = [
        0.90 * clean_p_p40[i] + 0.10 * clean_p_p49[i]
        for i in range(len(clean_indices))
    ]
    clean_preds_ens = decode_spanned_predictions_with_cat_thresholds(
        clean_p_ens, examples=clean_ex, id2label=id2label,
        conf_thresh=DEFAULT_CONF_THRESH, aspect_thresholds=PHASE40_THRESHOLDS,
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

    print(f"Ensemble Official Score: {off_clean_ens['official_score']:.6f} (Delta: {ens_delta_overall:+.6f})")
    print(f"  Category 1 Score:       {off_clean_ens['categories'][1]['score']:.6f} (Delta: {ens_delta_cat1:+.6f})")
    print(f"  Category 2 Score:       {off_clean_ens['categories'][2]['score']:.6f} (Delta: {ens_delta_cat2:+.6f})")
    print(f"  TP={mic_clean_ens['tp']} ({mic_clean_ens['tp'] - mic_clean_p40['tp']:+d})")
    print(f"  FP={mic_clean_ens['fp']} ({mic_clean_ens['fp'] - mic_clean_p40['fp']:+d})")
    print(f"  FN={mic_clean_ens['fn']} ({mic_clean_ens['fn'] - mic_clean_p40['fn']:+d})")

    # Bootstrap on clean holdout ensemble
    boot_clean_ens = run_paired_bootstrap(records_clean_p40, records_clean_ens, n_resamples=1000, seed=42)
    print(f"\nClean Holdout Ensemble Bootstrap (1,000 resamples):")
    print(f"  Mean Delta: {boot_clean_ens['mean_delta']:+.6f}")
    print(f"  95% CI:     [{boot_clean_ens['ci_lower']:+.6f}, {boot_clean_ens['ci_upper']:+.6f}]")
    print(f"  p-value:    {boot_clean_ens['p_value']:.4f}")
    print(f"  Significant: {boot_clean_ens['is_significant']}")

    # ----------------------------------------------------
    # SECTION 4: COMPLEMENTARITY AUDIT
    # ----------------------------------------------------
    print("\n" + "=" * 60)
    print("SECTION 4: COMPLEMENTARITY & FALSE-POSITIVE AUDIT")
    print("=" * 60)

    changed_titles = []
    repaired_titles = []
    regressed_titles = []
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

    print(f"Titles Changed:   {len(changed_titles)} ({len(changed_titles)/len(clean_ex)*100:.2f}%)")
    print(f"  Repaired Titles:   {len(repaired_titles)}")
    print(f"  Regressed Titles:  {len(regressed_titles)}")
    print(f"  Net Title Delta:   {len(repaired_titles) - len(regressed_titles):+d}")
    print(f"New False Positives Introduced: {new_fp_count}")

    # ----------------------------------------------------
    # SECTION 5: OFFICIAL DECISION VERDICT
    # ----------------------------------------------------
    print("\n" + "=" * 60)
    print("SECTION 5: OFFICIAL DECISION VERDICT")
    print("=" * 60)

    # Criteria:
    # 1. Primary evaluation on internal dev improves over matched baseline? (delta_dev_overall > 0)
    # 2. Meaningful ensemble F0.2 improvement? (ens_delta_overall >= 0.0005)
    # 3. Controlled false positives? (Net FP delta <= 0)
    # 4. Bootstrap statistically significant? (ci_lower > 0)
    # 5. No serious category regression? (Both categories non-negative)

    dev_improved = (delta_dev_overall > 0.0005)
    ens_improved = (ens_delta_overall >= 0.0005)
    fps_controlled = (mic_clean_ens["fp"] <= mic_clean_p40["fp"])
    statistically_significant = boot_clean_ens["is_significant"]
    no_cat_regression = (ens_delta_cat1 >= -0.0005 and ens_delta_cat2 >= -0.0005)

    if dev_improved and ens_improved and fps_controlled and statistically_significant and no_cat_regression:
        verdict = "PROMISING CHALLENGER — OFFICIAL QUIZ TEST JUSTIFIED"
    elif not dev_improved or ens_delta_overall <= 0.0002 or not no_cat_regression or mic_clean_ens["fp"] > mic_clean_p40["fp"]:
        verdict = "RETAIN PHASE 40 — NO RELIABLE IMPROVEMENT"
    else:
        verdict = "STOP — VALIDATION OR RESOURCE INTEGRITY FAILURE"

    print(f"Decision Verdict: >>> {verdict} <<<")

    results = {
        "phase": "49",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "champion_sha256": champ_info["sha256"],
        "official_quiz_champion_score": champ_info["official_quiz_f02"],
        "internal_dev_evaluation": {
            "title_count": len(dev_800),
            "matched_baseline": {
                "official_score": off_dev_base["official_score"],
                "cat1_score": off_dev_base["categories"][1]["score"],
                "cat2_score": off_dev_base["categories"][2]["score"],
                "tp": mic_dev_base["tp"],
                "fp": mic_dev_base["fp"],
                "fn": mic_dev_base["fn"],
            },
            "phase49_standalone": {
                "official_score": off_dev_p49["official_score"],
                "cat1_score": off_dev_p49["categories"][1]["score"],
                "cat2_score": off_dev_p49["categories"][2]["score"],
                "tp": mic_dev_p49["tp"],
                "fp": mic_dev_p49["fp"],
                "fn": mic_dev_p49["fn"],
                "delta_overall": delta_dev_overall,
                "delta_cat1": delta_dev_cat1,
                "delta_cat2": delta_dev_cat2,
            },
            "bootstrap": boot_dev,
        },
        "clean_holdout_secondary_diagnostic": {
            "title_count": len(clean_ex),
            "phase40_champion": {
                "official_score": off_clean_p40["official_score"],
                "cat1_score": off_clean_p40["categories"][1]["score"],
                "cat2_score": off_clean_p40["categories"][2]["score"],
                "tp": mic_clean_p40["tp"],
                "fp": mic_clean_p40["fp"],
                "fn": mic_clean_p40["fn"],
            },
            "phase49_standalone": {
                "official_score": off_clean_p49["official_score"],
                "cat1_score": off_clean_p49["categories"][1]["score"],
                "cat2_score": off_clean_p49["categories"][2]["score"],
                "tp": mic_clean_p49["tp"],
                "fp": mic_clean_p49["fp"],
                "fn": mic_clean_p49["fn"],
                "delta_vs_phase40": off_clean_p49["official_score"] - off_clean_p40["official_score"],
            },
            "ensemble_90_10": {
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
            "bootstrap": boot_clean_ens,
            "complementarity": {
                "changed_titles_count": len(changed_titles),
                "repaired_titles_count": len(repaired_titles),
                "regressed_titles_count": len(regressed_titles),
                "new_fp_count": new_fp_count,
                "sample_changes": changed_titles[:10],
            },
        },
        "verdict": verdict,
    }

    # Serialization helper
    def to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(i) for i in obj]
        elif isinstance(obj, tuple):
            return [to_serializable(i) for i in obj]
        return obj

    clean_results = to_serializable(results)
    EVALUATION_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVALUATION_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=2, ensure_ascii=False)
    print(f"\nEvaluation results successfully saved to {EVALUATION_RESULTS_PATH}")

    return results


if __name__ == "__main__":
    run_phase_49_evaluation()
