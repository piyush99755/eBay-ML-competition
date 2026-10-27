"""
Phase 26A: Precision Confidence Calibration Study.

Investigates whether filtering low-confidence predicted entities from the original
Exp 10 ensemble (GBERT 0.40 + 3-seed XLM-R 0.60) can improve official-style entity F0.2
without retraining models or altering the frozen champion.

Fixed Threshold Grid: [0.00, 0.50, 0.60, 0.70, 0.80]
Aggregation Methods:
1. Mean selected-label probability across the entity.
2. Minimum selected-label probability across the entity.
"""

from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_examples import build_examples, load_training_data
from src.data.make_splits import split_examples
from src.modeling.prepare_datasets import prepare_split
from src.evaluation.entity_metrics import (
    bio_to_entities,
    count_matches,
    filter_entities_by_category,
    precision_recall_fbeta,
)
from src.evaluation.competition_metrics import calculate_competition_metrics

# Frozen Champion Checkpoints
CHAMPION_GBERT_CHECKPOINT = Path("checkpoints/gbert_baseline_best")
CHAMPION_XLMR42_CHECKPOINT = Path("checkpoints/xlmr_baseline_best")
CHAMPION_XLMR7_CHECKPOINT = Path("checkpoints/xlmr_seed_7_best")
CHAMPION_XLMR21_CHECKPOINT = Path("checkpoints/xlmr_seed_21_best")

VAL_BATCH_SIZE = 8
IGNORE_LABEL_ID = -100
EXPECTED_CHAMPION_F02 = 0.879852
THRESHOLDS = [0.00, 0.50, 0.60, 0.70, 0.80]
BOOTSTRAP_ROUNDS = 1000
BOOTSTRAP_SEED = 42


def collect_active_probabilities(
    checkpoint_path: Path,
    val_examples: List[dict],
    label2id: Dict[str, int],
    device: torch.device,
    batch_size: int = VAL_BATCH_SIZE,
) -> List[np.ndarray]:
    """Load model, run validation inference, extract softmax probabilities for active tokens, and free VRAM."""
    print(f"Loading {checkpoint_path.name}...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForTokenClassification.from_pretrained(checkpoint_path).to(device)
    model.eval()

    dataset, stats = prepare_split(val_examples, tokenizer, label2id)
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


def bio_to_entities_with_confidence(
    tokens: List[str],
    bio_tags: List[str],
    token_probs: List[float],
) -> List[Tuple[str, str, float, float]]:
    """
    Extract entities and compute mean & min confidence across constituent tokens.
    Returns: [(aspect_name, aspect_value, mean_conf, min_conf), ...]
    """
    if len(tokens) != len(bio_tags) or len(tokens) != len(token_probs):
        raise ValueError("Lengths of tokens, tags, and probs must match.")

    entities = []
    current_aspect = None
    current_tokens = []
    current_probs = []

    def close_current_entity() -> None:
        nonlocal current_aspect, current_tokens, current_probs
        if current_aspect is not None and current_tokens:
            value = " ".join(current_tokens)
            mean_conf = float(np.mean(current_probs))
            min_conf = float(np.min(current_probs))
            entities.append((current_aspect, value, mean_conf, min_conf))
        current_aspect = None
        current_tokens = []
        current_probs = []

    for token, tag, prob in zip(tokens, bio_tags, token_probs):
        if tag == "O":
            close_current_entity()
            continue

        if tag.startswith("B-"):
            close_current_entity()
            current_aspect = tag[2:]
            current_tokens = [token]
            current_probs = [prob]
            continue

        if tag.startswith("I-"):
            aspect = tag[2:]
            if current_aspect == aspect:
                current_tokens.append(token)
                current_probs.append(prob)
            else:
                close_current_entity()
                current_aspect = aspect
                current_tokens = [token]
                current_probs = [prob]
            continue

        raise ValueError(f"Unknown BIO tag: {tag}")

    close_current_entity()
    return entities


def evaluate_records(records: List[dict]) -> dict:
    """Evaluate records and return official F0.2, category scores, and micro P/R."""
    metrics = calculate_competition_metrics(records)
    final_f02 = metrics["final_score"]
    cat1_score = metrics["categories"].get(1, {}).get("score", 0.0)
    cat2_score = metrics["categories"].get(2, {}).get("score", 0.0)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    for r in records:
        tp, fp, fn = count_matches(r["predicted_entities"], r["true_entities"])
        total_tp += tp
        total_fp += fp
        total_fn += fn

    pr_stats = precision_recall_fbeta(total_tp, total_fp, total_fn, beta=0.2)
    return {
        "f02": final_f02,
        "cat1_f02": cat1_score,
        "cat2_f02": cat2_score,
        "precision": pr_stats["precision"],
        "recall": pr_stats["recall"],
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "total_preds": sum(len(r["predicted_entities"]) for r in records),
    }


def run_paired_bootstrap(
    base_records: List[dict],
    filtered_records: List[dict],
    num_rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Run paired bootstrap resampling over validation titles to estimate Delta F0.2 CI."""
    np.random.seed(seed)
    n = len(base_records)
    assert len(filtered_records) == n

    deltas = []
    base_scores = []
    filt_scores = []

    for _ in range(num_rounds):
        indices = np.random.randint(0, n, size=n)
        resampled_base = [base_records[i] for i in indices]
        resampled_filt = [filtered_records[i] for i in indices]

        score_b = calculate_competition_metrics(resampled_base)["final_score"]
        score_f = calculate_competition_metrics(resampled_filt)["final_score"]

        base_scores.append(score_b)
        filt_scores.append(score_f)
        deltas.append(score_f - score_b)

    deltas = np.array(deltas)
    mean_delta = float(np.mean(deltas))
    std_delta = float(np.std(deltas))
    ci_low = float(np.percentile(deltas, 2.5))
    ci_high = float(np.percentile(deltas, 97.5))
    prob_improved = float(np.mean(deltas > 0))

    return {
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "ci_95": (ci_low, ci_high),
        "prob_improved": prob_improved,
    }


def main():
    print("=" * 80)
    print("PHASE 26A: PRECISION CONFIDENCE CALIBRATION STUDY")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Label Mapping from Reference Champion
    ref_cfg = AutoConfig.from_pretrained(CHAMPION_GBERT_CHECKPOINT)
    id2label = {int(k): v for k, v in ref_cfg.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    # 2. Load Split A Validation Set
    df = load_training_data()
    all_examples = build_examples(df)
    _, val_examples = split_examples(all_examples, random_state=42)
    print(f"Loaded {len(val_examples)} Split-A validation titles.\n")

    # 3. Extract Champion Probabilities Sequentially
    print("--- Extracting Frozen Champion Model Probabilities ---")
    p_gbert = collect_active_probabilities(CHAMPION_GBERT_CHECKPOINT, val_examples, label2id, device)
    p_xlmr42 = collect_active_probabilities(CHAMPION_XLMR42_CHECKPOINT, val_examples, label2id, device)
    p_xlmr7 = collect_active_probabilities(CHAMPION_XLMR7_CHECKPOINT, val_examples, label2id, device)
    p_xlmr21 = collect_active_probabilities(CHAMPION_XLMR21_CHECKPOINT, val_examples, label2id, device)

    # 4. Compute Frozen Champion Fused Probabilities
    print("\nFusing champion probabilities: 0.40 GBERT + 0.20 XLM-R(42) + 0.20 XLM-R(7) + 0.20 XLM-R(21)...")
    num_ex = len(val_examples)
    p_fused = [
        0.40 * p_gbert[i] + 0.20 * p_xlmr42[i] + 0.20 * p_xlmr7[i] + 0.20 * p_xlmr21[i]
        for i in range(num_ex)
    ]

    # 5. Extract Entities with Confidences
    print("Extracting candidate entities with token-level selected BIO confidence...")
    extracted_data = []
    base_records = []

    for ex_idx, ex in enumerate(val_examples):
        tokens = ex["tokens"]
        cat_id = ex["category"]
        true_bio = ex["bio_tags"]

        probs = p_fused[ex_idx]
        pred_ids = np.argmax(probs, axis=-1)
        pred_bio = [id2label[pid] for pid in pred_ids]

        # Extract probability assigned to the selected label at each token position
        selected_probs = [probs[tok_idx, pred_ids[tok_idx]] for tok_idx in range(len(tokens))]

        true_entities = bio_to_entities(tokens, true_bio)

        # Raw entities with mean and min confidence
        raw_conf_entities = bio_to_entities_with_confidence(tokens, pred_bio, selected_probs)

        # Apply category-aware post-filtering (Method A)
        cat_filtered_conf_entities = []
        for asp, val, mean_c, min_c in raw_conf_entities:
            # Check if this aspect is allowed for this category
            allowed = filter_entities_by_category([(asp, val)], cat_id)
            if allowed:
                cat_filtered_conf_entities.append((asp, val, mean_c, min_c))

        extracted_data.append({
            "category": cat_id,
            "true_entities": true_entities,
            "entities_with_conf": cat_filtered_conf_entities,
        })

        # Baseline unfiltered entities
        base_entities = [(asp, val) for asp, val, _, _ in cat_filtered_conf_entities]
        base_records.append({
            "category": cat_id,
            "true_entities": true_entities,
            "predicted_entities": base_entities,
        })

    # 6. Verify Baseline Score Exact Reproduction
    base_metrics = evaluate_records(base_records)
    print("\n" + "=" * 50)
    print("BASELINE CHAMPION REPRODUCTION")
    print("=" * 50)
    print(f"Reproduced Baseline F0.2: {base_metrics['f02']:.6f}")
    print(f"Expected Champion F0.2:   {EXPECTED_CHAMPION_F02:.6f}")
    print(f"Category 1 F0.2:          {base_metrics['cat1_f02']:.6f}")
    print(f"Category 2 F0.2:          {base_metrics['cat2_f02']:.6f}")
    print(f"Micro Precision:          {base_metrics['precision']:.4f}")
    print(f"Micro Recall:             {base_metrics['recall']:.4f}")
    print(f"Total True Positives:     {base_metrics['tp']}")
    print(f"Total False Positives:    {base_metrics['fp']}")
    print(f"Total Predicted Entities: {base_metrics['total_preds']}")

    assert abs(base_metrics["f02"] - EXPECTED_CHAMPION_F02) < 1e-5, (
        f"Baseline F0.2 mismatch! Got {base_metrics['f02']:.6f}, expected {EXPECTED_CHAMPION_F02:.6f}"
    )
    print(">>> EXACT BASELINE MATCH VERIFIED (0.879852) <<<\n")

    # 7. Evaluate Fixed Threshold Grid
    grid_results = []
    filtered_records_store = {}

    for agg in ["mean", "min"]:
        for thresh in THRESHOLDS:
            records = []
            for item in extracted_data:
                cat_id = item["category"]
                true_ents = item["true_entities"]
                filtered_ents = []
                for asp, val, mean_c, min_c in item["entities_with_conf"]:
                    score = mean_c if agg == "mean" else min_c
                    if score >= thresh:
                        filtered_ents.append((asp, val))
                records.append({
                    "category": cat_id,
                    "true_entities": true_ents,
                    "predicted_entities": filtered_ents,
                })

            res = evaluate_records(records)
            delta_f02 = res["f02"] - base_metrics["f02"]
            removed_total = base_metrics["total_preds"] - res["total_preds"]
            pct_removed = (removed_total / base_metrics["total_preds"]) * 100.0 if base_metrics["total_preds"] > 0 else 0.0

            removed_tp = base_metrics["tp"] - res["tp"]
            removed_fp = base_metrics["fp"] - res["fp"]

            grid_entry = {
                "agg": agg,
                "thresh": thresh,
                "f02": res["f02"],
                "delta_f02": delta_f02,
                "cat1_f02": res["cat1_f02"],
                "cat2_f02": res["cat2_f02"],
                "precision": res["precision"],
                "recall": res["recall"],
                "total_preds": res["total_preds"],
                "removed_total": removed_total,
                "pct_removed": pct_removed,
                "removed_tp": removed_tp,
                "removed_fp": removed_fp,
            }
            grid_results.append(grid_entry)
            filtered_records_store[(agg, thresh)] = records

    # 8. Display Results Table
    print("=" * 115)
    print("FIXED CONFIDENCE THRESHOLD GRID RESULTS")
    print("=" * 115)
    print(
        f"{'Method':<6}{'Thresh':<8}{'F0.2':<11}{'Delta':<11}{'Cat1':<10}{'Cat2':<10}"
        f"{'Prec':<9}{'Rec':<9}{'Preds':<8}{'RemTot':<8}{'Rem%':<7}{'RemTP':<7}{'RemFP':<7}"
    )
    print("-" * 115)

    for g in grid_results:
        d_str = f"{g['delta_f02']:+.6f}"
        rem_pct_str = f"{g['pct_removed']:.2f}%"
        print(
            f"{g['agg'].upper():<6}{g['thresh']:<8.2f}{g['f02']:<11.6f}{d_str:<11}"
            f"{g['cat1_f02']:<10.6f}{g['cat2_f02']:<10.6f}{g['precision']:<9.4f}{g['recall']:<9.4f}"
            f"{g['total_preds']:<8}{g['removed_total']:<8}{rem_pct_str:<7}{g['removed_tp']:<7}{g['removed_fp']:<7}"
        )
    print("-" * 115)

    # 9. Find Strongest Configuration
    non_baseline_entries = [g for g in grid_results if g["thresh"] > 0.0]
    best_entry = max(non_baseline_entries, key=lambda x: x["f02"])
    print(f"\nStrongest Filtered Configuration: {best_entry['agg'].upper()} Threshold >= {best_entry['thresh']:.2f}")
    print(f"  F0.2:             {best_entry['f02']:.6f} ({best_entry['delta_f02']:+.6f} vs Baseline)")
    print(f"  Category 1 F0.2:  {best_entry['cat1_f02']:.6f} ({best_entry['cat1_f02'] - base_metrics['cat1_f02']:+.6f})")
    print(f"  Category 2 F0.2:  {best_entry['cat2_f02']:.6f} ({best_entry['cat2_f02'] - base_metrics['cat2_f02']:+.6f})")
    print(f"  Precision:        {best_entry['precision']:.4f} (Baseline: {base_metrics['precision']:.4f})")
    print(f"  Recall:           {best_entry['recall']:.4f} (Baseline: {base_metrics['recall']:.4f})")
    print(f"  Entities Removed: {best_entry['removed_total']} ({best_entry['pct_removed']:.2f}%)")
    print(f"  TP Removed:       {best_entry['removed_tp']}")
    print(f"  FP Removed:       {best_entry['removed_fp']}")

    # 10. Paired Bootstrap Resampling for Strongest Configuration
    best_filtered_records = filtered_records_store[(best_entry["agg"], best_entry["thresh"])]
    print(f"\nRunning {BOOTSTRAP_ROUNDS} paired bootstrap iterations to assess statistical significance...")
    boot_stats = run_paired_bootstrap(base_records, best_filtered_records, num_rounds=BOOTSTRAP_ROUNDS)

    print(f"  Mean Delta F0.2:      {boot_stats['mean_delta']:+.6f}")
    print(f"  Std Error:            {boot_stats['std_delta']:.6f}")
    print(f"  95% Confidence Int:   [{boot_stats['ci_95'][0]:+.6f}, {boot_stats['ci_95'][1]:+.6f}]")
    print(f"  P(Filtered > Base):   {boot_stats['prob_improved'] * 100:.1f}%")

    print("\n" + "=" * 80)
    print("PHASE 26A STUDY COMPLETE")
    print("=" * 80)

    return {
        "base_metrics": base_metrics,
        "grid_results": grid_results,
        "best_entry": best_entry,
        "boot_stats": boot_stats,
    }


if __name__ == "__main__":
    main()
