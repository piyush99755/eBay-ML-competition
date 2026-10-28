"""
Phase 29B: Comprehensive Evaluation for 3-Fold GBERT Ensemble Pilot.

Evaluates:
- Gate A: Out-of-fold (OOF) evaluation across all 4,000 training titles.
- Gate B: Individual fold models on the 1,000 Split-A holdout titles.
- Gate C: 3-fold GBERT-only ensemble on the 1,000 Split-A holdout titles.
- Gate D: Full 6-model challenger ensemble (0.40 GBERT-CV3 + 0.60 XLM-R 3-seed)
          with MIN 0.60 confidence filter on the 1,000 Split-A holdout titles.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["HF_HUB_OFFLINE"] = "1"

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
from src.data.label_mapping import create_label_mappings
from src.data.make_splits import split_examples
from src.data.make_cv_folds import get_cv_folds
from src.modeling.prepare_datasets import prepare_split
from src.evaluation.entity_metrics import (
    bio_to_entities,
    bio_to_entities_with_confidence,
    count_matches,
    filter_entities_by_category,
)
from src.evaluation.competition_metrics import calculate_competition_metrics

# Frozen Champion & Challenger Checkpoint Paths
CHAMPION_CHECKPOINTS = {
    "gbert_baseline": Path("checkpoints/gbert_baseline_best"),
    "xlmr_seed42": Path("checkpoints/xlmr_baseline_best"),
    "xlmr_seed7": Path("checkpoints/xlmr_seed_7_best"),
    "xlmr_seed21": Path("checkpoints/xlmr_seed_21_best"),
}

CV_GBERT_CHECKPOINTS = {
    0: Path("checkpoints/gbert_cv3_fold0_best"),
    1: Path("checkpoints/gbert_cv3_fold1_best"),
    2: Path("checkpoints/gbert_cv3_fold2_best"),
}

BENCHMARK_CHAMPION_F02 = 0.899802
BENCHMARK_SINGLE_GBERT_F02 = 0.867200
VAL_BATCH_SIZE = 8
IGNORE_LABEL_ID = -100


def collect_active_token_probabilities(
    checkpoint_path: Path,
    examples: List[dict],
    label2id: Dict[str, int],
    device: torch.device,
    batch_size: int = VAL_BATCH_SIZE,
) -> List[np.ndarray]:
    """Load model, run validation inference, extract softmax probabilities for active tokens, and free VRAM."""
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForTokenClassification.from_pretrained(checkpoint_path).to(device)
    model.eval()

    dataset, _ = prepare_split(examples, tokenizer, label2id)
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    all_example_probs = []
    with torch.no_grad():
        for batch in dataloader:
            batch_dev = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            outputs = model(**batch_dev)
            logits = outputs.logits
            labels = batch_dev["labels"]

            for row_idx in range(labels.shape[0]):
                row_labels = labels[row_idx]
                row_logits = logits[row_idx]
                active_mask = row_labels != IGNORE_LABEL_ID
                active_logits = row_logits[active_mask]
                active_probs = torch.softmax(active_logits, dim=-1).cpu().numpy().astype(np.float32)
                all_example_probs.append(active_probs)

    del model, tokenizer, dataset, dataloader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_example_probs


def score_predictions_official(
    examples: List[dict],
    predicted_entities_per_example: List[List[Tuple[str, str]]],
) -> dict:
    """Compute official competition metrics from predicted entities."""
    records = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for ex, preds in zip(examples, predicted_entities_per_example):
        true_ents = bio_to_entities(ex["tokens"], ex["bio_tags"])
        filtered_true = filter_entities_by_category(true_ents, ex["category"])
        filtered_preds = filter_entities_by_category(preds, ex["category"])

        tp, fp, fn = count_matches(filtered_preds, filtered_true)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        records.append({
            "category": ex["category"],
            "true_entities": filtered_true,
            "predicted_entities": filtered_preds,
        })

    comp_res = calculate_competition_metrics(records)
    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)

    return {
        "final_score": comp_res["final_score"],
        "cat1_score": comp_res["categories"][1]["score"],
        "cat2_score": comp_res["categories"][2]["score"],
        "precision": precision,
        "recall": recall,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "total_preds": sum(len(p) for p in predicted_entities_per_example),
        "comp_results": comp_res,
    }


def main():
    print("=" * 75)
    print("PHASE 29B: THREE-FOLD GBERT PILOT EVALUATION (GATES A, B, C, D)")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load Data
    df = load_training_data()
    examples = build_examples(df)
    train_examples, val_examples = split_examples(examples, random_state=42)
    label2id, id2label = create_label_mappings(examples)

    folds = get_cv_folds(train_examples, n_splits=3, random_state=42)
    cache_dir = Path("data/cache")

    # ----------------------------------------------------
    # GATE A: Out-of-Fold (OOF) Evaluation (4,000 titles)
    # ----------------------------------------------------
    print("\n" + "=" * 70)
    print("GATE A: OUT-OF-FOLD (OOF) EVALUATION ACROSS 4,000 TRAINING TITLES")
    print("=" * 70)

    oof_examples = []
    oof_predictions = []

    for fold_idx in range(3):
        _, val_f = folds[fold_idx]
        oof_cache_path = cache_dir / f"oof_probs_gbert_fold{fold_idx}.npy"
        print(f"Loading OOF probabilities for Fold {fold_idx} from {oof_cache_path}...")

        flat_probs = np.load(oof_cache_path)
        splits = np.cumsum([len(e["tokens"]) for e in val_f])[:-1]
        fold_probs = np.split(flat_probs, splits)

        assert len(fold_probs) == len(val_f)

        for ex, p in zip(val_f, fold_probs):
            pred_ids = np.argmax(p, axis=-1)
            pred_bio = [id2label[pid] for pid in pred_ids]
            ents = bio_to_entities(ex["tokens"], pred_bio)
            filtered_ents = filter_entities_by_category(ents, ex["category"])

            oof_examples.append(ex)
            oof_predictions.append(filtered_ents)

    assert len(oof_examples) == 4000, f"Expected 4,000 OOF examples, got {len(oof_examples)}"
    gate_a_metrics = score_predictions_official(oof_examples, oof_predictions)

    print("\n>>> GATE A RESULTS (Individual Fold-Model Quality Diagnostic):")
    print(f"  Official OOF F0.2: {gate_a_metrics['final_score']:.6f}")
    print(f"  Category 1 F0.2:   {gate_a_metrics['cat1_score']:.6f}")
    print(f"  Category 2 F0.2:   {gate_a_metrics['cat2_score']:.6f}")
    print(f"  Precision:         {gate_a_metrics['precision']:.4f}")
    print(f"  Recall:            {gate_a_metrics['recall']:.4f}")
    print(f"  Bookkeeping:       TP = {gate_a_metrics['tp']:,} | FP = {gate_a_metrics['fp']:,} | FN = {gate_a_metrics['fn']:,}")
    print(f"  Total Predictions: {gate_a_metrics['total_preds']:,}")

    # ----------------------------------------------------
    # GATE B: Individual Fold Models on Holdout (1,000 titles)
    # ----------------------------------------------------
    print("\n" + "=" * 70)
    print("GATE B: INDIVIDUAL FOLD MODELS ON 1,000-TITLE SPLIT-A HOLDOUT")
    print("=" * 70)

    fold_holdout_probs = {}
    gate_b_results = {}

    for fold_idx in range(3):
        ckpt_path = CV_GBERT_CHECKPOINTS[fold_idx]
        print(f"Evaluating Fold {fold_idx} ({ckpt_path.name}) on 1,000 holdout titles...")
        t0 = time.time()
        probs = collect_active_token_probabilities(ckpt_path, val_examples, label2id, device)
        fold_holdout_probs[fold_idx] = probs

        # Predict
        fold_preds = []
        for ex, p in zip(val_examples, probs):
            pred_ids = np.argmax(p, axis=-1)
            pred_bio = [id2label[pid] for pid in pred_ids]
            ents = bio_to_entities(ex["tokens"], pred_bio)
            filtered_ents = filter_entities_by_category(ents, ex["category"])
            fold_preds.append(filtered_ents)

        m = score_predictions_official(val_examples, fold_preds)
        gate_b_results[fold_idx] = m
        delta_single = m["final_score"] - BENCHMARK_SINGLE_GBERT_F02
        print(f"  Fold {fold_idx} Holdout F0.2: {m['final_score']:.6f} (Delta vs Single GBERT: {delta_single:+.6f}) | Elapsed: {time.time()-t0:.1f}s")

    # ----------------------------------------------------
    # GATE C: GBERT-Only CV Ensemble on Holdout (1,000 titles)
    # ----------------------------------------------------
    print("\n" + "=" * 70)
    print("GATE C: 3-FOLD GBERT-ONLY ENSEMBLE ON 1,000-TITLE SPLIT-A HOLDOUT")
    print("=" * 70)

    gbert_cv_probs = []
    for ex_idx in range(len(val_examples)):
        p_avg = (fold_holdout_probs[0][ex_idx] + fold_holdout_probs[1][ex_idx] + fold_holdout_probs[2][ex_idx]) / 3.0
        gbert_cv_probs.append(p_avg)

    gate_c_preds = []
    for ex, p in zip(val_examples, gbert_cv_probs):
        pred_ids = np.argmax(p, axis=-1)
        pred_bio = [id2label[pid] for pid in pred_ids]
        ents = bio_to_entities(ex["tokens"], pred_bio)
        filtered_ents = filter_entities_by_category(ents, ex["category"])
        gate_c_preds.append(filtered_ents)

    gate_c_metrics = score_predictions_official(val_examples, gate_c_preds)
    delta_c_vs_single = gate_c_metrics["final_score"] - BENCHMARK_SINGLE_GBERT_F02

    print(f"\n>>> GATE C RESULTS (GBERT-Only 3-Fold Ensemble):")
    print(f"  Holdout F0.2:       {gate_c_metrics['final_score']:.6f} (Delta vs Single GBERT: {delta_c_vs_single:+.6f})")
    print(f"  Category 1 F0.2:    {gate_c_metrics['cat1_score']:.6f}")
    print(f"  Category 2 F0.2:    {gate_c_metrics['cat2_score']:.6f}")
    print(f"  Precision:          {gate_c_metrics['precision']:.4f}")
    print(f"  Recall:             {gate_c_metrics['recall']:.4f}")
    print(f"  Bookkeeping:        TP = {gate_c_metrics['tp']:,} | FP = {gate_c_metrics['fp']:,} | FN = {gate_c_metrics['fn']:,}")
    print(f"  Total Predictions:  {gate_c_metrics['total_preds']:,}")

    # ----------------------------------------------------
    # GATE D: Full 6-Model Challenger Ensemble on Holdout
    # ----------------------------------------------------
    print("\n" + "=" * 70)
    print("GATE D: FULL 6-MODEL CHALLENGER ENSEMBLE (0.40 GBERT-CV3 + 0.60 XLM-R 3-SEED)")
    print("=" * 70)

    print("Collecting active probabilities for 3 frozen XLM-R champion models...")
    xlmr_probs = {}
    for name in ["xlmr_seed42", "xlmr_seed7", "xlmr_seed21"]:
        t0_x = time.time()
        ckpt = CHAMPION_CHECKPOINTS[name]
        print(f"  Loading {name} ({ckpt.name})...")
        xlmr_probs[name] = collect_active_token_probabilities(ckpt, val_examples, label2id, device)
        print(f"  {name} collected in {time.time()-t0_x:.1f}s.")

    # 6-Model Probability Fusion & Confidence Filtering (MIN >= 0.60)
    print("\nFusing 6 models (0.40 GBERT-CV3 + 0.60 XLM-R 3-seed) and applying MIN 0.60 filter...")
    gate_d_preds = []
    aspect_counts: Dict[str, int] = {}

    for ex_idx, ex in enumerate(val_examples):
        tokens = ex["tokens"]
        cat_id = ex["category"]

        p_gb = gbert_cv_probs[ex_idx]
        p_xl = (xlmr_probs["xlmr_seed42"][ex_idx] + xlmr_probs["xlmr_seed7"][ex_idx] + xlmr_probs["xlmr_seed21"][ex_idx]) / 3.0

        p_fused = 0.40 * p_gb + 0.60 * p_xl
        pred_ids = np.argmax(p_fused, axis=-1)
        pred_bio = [id2label[pid] for pid in pred_ids]

        selected_probs = [float(p_fused[t_idx, pred_ids[t_idx]]) for t_idx in range(len(tokens))]
        raw_conf_ents = bio_to_entities_with_confidence(tokens, pred_bio, selected_probs)

        # Apply Category Post-Filtering (Method A)
        cat_valid_ents = filter_entities_by_category(raw_conf_ents, cat_id)

        # Apply MIN 0.60 Confidence Threshold
        ents_060 = [(asp, val) for asp, val, _, min_c in cat_valid_ents if min_c >= 0.60]
        gate_d_preds.append(ents_060)

        for asp, _ in ents_060:
            aspect_counts[asp] = aspect_counts.get(asp, 0) + 1

    gate_d_metrics = score_predictions_official(val_examples, gate_d_preds)
    delta_d_vs_champ = gate_d_metrics["final_score"] - BENCHMARK_CHAMPION_F02

    print("\n" + "=" * 70)
    print(">>> GATE D FINAL RESULTS (Full 6-Model Challenger Ensemble vs Champion):")
    print("=" * 70)
    print(f"  Challenger Holdout F0.2: {gate_d_metrics['final_score']:.6f}")
    print(f"  Champion Holdout F0.2:   {BENCHMARK_CHAMPION_F02:.6f}")
    print(f"  Score Delta:             {delta_d_vs_champ:+.6f}")
    print(f"  Category 1 F0.2:         {gate_d_metrics['cat1_score']:.6f}")
    print(f"  Category 2 F0.2:         {gate_d_metrics['cat2_score']:.6f}")
    print(f"  Precision:               {gate_d_metrics['precision']:.4f}")
    print(f"  Recall:                  {gate_d_metrics['recall']:.4f}")
    print(f"  Bookkeeping:             TP = {gate_d_metrics['tp']:,} | FP = {gate_d_metrics['fp']:,} | FN = {gate_d_metrics['fn']:,}")
    print(f"  Total Predictions:       {gate_d_metrics['total_preds']:,}")

    # Aspect Extinction Audit
    print("\n--- Aspect Emission Breakdown (Challenger vs Baseline):")
    aspect_metrics = gate_d_metrics["comp_results"]["categories"][1]["aspects"]
    aspect_metrics_cat2 = gate_d_metrics["comp_results"]["categories"][2]["aspects"]

    extinct_aspects = []
    for asp, count in sorted(aspect_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {asp:<35} : {count} predictions")
    for asp in ["Oe/Oem_Referenznummer(N)", "Modell", "Stärke", "Material"]:
        if aspect_counts.get(asp, 0) == 0:
            extinct_aspects.append(asp)

    if extinct_aspects:
        print(f"\nWARNING: Extinct aspects detected: {extinct_aspects}")
    else:
        print(f"\nAll rare aspects preserved without extinction.")

    # Save summary artifact
    results_summary = {
        "gate_a_oof": {
            "f02": gate_a_metrics["final_score"],
            "cat1_f02": gate_a_metrics["cat1_score"],
            "cat2_f02": gate_a_metrics["cat2_score"],
            "precision": gate_a_metrics["precision"],
            "recall": gate_a_metrics["recall"],
            "tp": gate_a_metrics["tp"],
            "fp": gate_a_metrics["fp"],
            "fn": gate_a_metrics["fn"],
        },
        "gate_b_individual": {
            k: {
                "f02": v["final_score"],
                "cat1_f02": v["cat1_score"],
                "cat2_f02": v["cat2_score"],
                "precision": v["precision"],
                "recall": v["recall"],
            }
            for k, v in gate_b_results.items()
        },
        "gate_c_gbert_cv": {
            "f02": gate_c_metrics["final_score"],
            "cat1_f02": gate_c_metrics["cat1_score"],
            "cat2_f02": gate_c_metrics["cat2_score"],
            "precision": gate_c_metrics["precision"],
            "recall": gate_c_metrics["recall"],
            "delta_vs_single": delta_c_vs_single,
        },
        "gate_d_challenger": {
            "f02": gate_d_metrics["final_score"],
            "cat1_f02": gate_d_metrics["cat1_score"],
            "cat2_f02": gate_d_metrics["cat2_score"],
            "precision": gate_d_metrics["precision"],
            "recall": gate_d_metrics["recall"],
            "tp": gate_d_metrics["tp"],
            "fp": gate_d_metrics["fp"],
            "fn": gate_d_metrics["fn"],
            "delta_vs_champion": delta_d_vs_champ,
            "extinct_aspects": extinct_aspects,
            "aspect_counts": aspect_counts,
        },
    }

    out_json = cache_dir / "experiment_29b_evaluation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nEvaluation summary saved to {out_json}")


if __name__ == "__main__":
    main()
