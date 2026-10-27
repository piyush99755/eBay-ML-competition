"""
Phase 25B: Experiment 25 - GELECTRA Challenger Ensemble Evaluation.

Evaluates the German ELECTRA model (checkpoints/gelectra_baseline_best) both standalone
and in combination with the frozen 4-model champion ensemble on the 1,000 Split-A validation titles.

Predefined Evaluation Variants:
- Baseline:   0.40 GBERT + 0.60 XLM-R trio (0.20 each) [Exp 10 Champion]
- Standalone: 1.00 GELECTRA
- Variant A:  0.25 GBERT + 0.20 GELECTRA + 0.55 XLM-R trio (0.1833 each)
- Variant B:  0.25 GBERT + 0.25 GELECTRA + 0.50 XLM-R trio (0.1667 each)
- Variant C:  0.35 GBERT + 0.15 GELECTRA + 0.50 XLM-R trio (0.1667 each)
- Variant D:  0.40 GELECTRA + 0.60 XLM-R trio (0.20 each)
"""

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

# Model Checkpoint Paths
CHAMPION_GBERT_CHECKPOINT = Path("checkpoints/gbert_baseline_best")
CHAMPION_XLMR42_CHECKPOINT = Path("checkpoints/xlmr_baseline_best")
CHAMPION_XLMR7_CHECKPOINT = Path("checkpoints/xlmr_seed_7_best")
CHAMPION_XLMR21_CHECKPOINT = Path("checkpoints/xlmr_seed_21_best")
GELECTRA_CHECKPOINT = Path("checkpoints/gelectra_baseline_best")

VAL_BATCH_SIZE = 8
IGNORE_LABEL_ID = -100
FROZEN_CHAMPION_F02 = 0.879852


def collect_active_probabilities(
    checkpoint_path: Path,
    val_examples: List[dict],
    label2id: Dict[str, int],
    device: torch.device,
    batch_size: int = VAL_BATCH_SIZE,
) -> List[np.ndarray]:
    """
    Sequentially load model, run validation inference, extract softmax probabilities
    for original active tokens, and immediately free GPU memory.
    """
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


def evaluate_variant(
    name: str,
    combined_probabilities: List[np.ndarray],
    val_examples: List[dict],
    id2label: Dict[int, str],
) -> dict:
    """Run argmax decoding, entity reconstruction, category filtering, and scoring."""
    records = []
    total_predicted_entities = 0
    cat1_pred_entities = 0
    cat2_pred_entities = 0

    for ex_idx, ex in enumerate(val_examples):
        tokens = ex["tokens"]
        cat_id = ex["category"]
        true_bio = ex["bio_tags"]

        probs = combined_probabilities[ex_idx]
        assert len(tokens) == len(probs), f"Token mismatch at example {ex_idx}"

        pred_ids = np.argmax(probs, axis=-1)
        pred_bio = [id2label[pid] for pid in pred_ids]

        true_entities = bio_to_entities(tokens, true_bio)
        raw_pred_entities = bio_to_entities(tokens, pred_bio)
        filtered_pred_entities = filter_entities_by_category(raw_pred_entities, cat_id)

        total_predicted_entities += len(filtered_pred_entities)
        if cat_id == 1:
            cat1_pred_entities += len(filtered_pred_entities)
        else:
            cat2_pred_entities += len(filtered_pred_entities)

        records.append({
            "category": cat_id,
            "true_entities": true_entities,
            "predicted_entities": filtered_pred_entities,
        })

    # Competition-style metrics
    metrics = calculate_competition_metrics(records)
    final_f02 = metrics["final_score"]
    cat1_score = metrics["categories"].get(1, {}).get("score", 0.0)
    cat2_score = metrics["categories"].get(2, {}).get("score", 0.0)

    # Micro P and R
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for r in records:
        tp, fp, fn = count_matches(r["predicted_entities"], r["true_entities"])
        total_tp += tp
        total_fp += fp
        total_fn += fn

    pr_stats = precision_recall_fbeta(total_tp, total_fp, total_fn, beta=0.2)
    precision = pr_stats["precision"]
    recall = pr_stats["recall"]

    return {
        "name": name,
        "f02": final_f02,
        "cat1_f02": cat1_score,
        "cat2_f02": cat2_score,
        "precision": precision,
        "recall": recall,
        "total_predicted_entities": total_predicted_entities,
        "cat1_predicted_entities": cat1_pred_entities,
        "cat2_predicted_entities": cat2_pred_entities,
    }


def main():
    print("=" * 70)
    print("PHASE 25B: GELECTRA CHALLENGER ENSEMBLE EVALUATION")
    print("=" * 70)

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

    # 3. Collect Probabilities Sequentially
    print("--- [1/5] Extracting GBERT probabilities ---")
    p_gbert = collect_active_probabilities(CHAMPION_GBERT_CHECKPOINT, val_examples, label2id, device)

    print("--- [2/5] Extracting XLM-R Seed 42 probabilities ---")
    p_xlmr42 = collect_active_probabilities(CHAMPION_XLMR42_CHECKPOINT, val_examples, label2id, device)

    print("--- [3/5] Extracting XLM-R Seed 7 probabilities ---")
    p_xlmr7 = collect_active_probabilities(CHAMPION_XLMR7_CHECKPOINT, val_examples, label2id, device)

    print("--- [4/5] Extracting XLM-R Seed 21 probabilities ---")
    p_xlmr21 = collect_active_probabilities(CHAMPION_XLMR21_CHECKPOINT, val_examples, label2id, device)

    print("--- [5/5] Extracting GELECTRA probabilities ---")
    p_gelectra = collect_active_probabilities(GELECTRA_CHECKPOINT, val_examples, label2id, device)

    num_ex = len(val_examples)
    # Average XLM-R trio probabilities equally (1/3 each)
    p_xlmr_avg = [(p_xlmr42[i] + p_xlmr7[i] + p_xlmr21[i]) / 3.0 for i in range(num_ex)]

    # 4. Define All Evaluation Variants
    variants = [
        (
            "Baseline (Exp 10 Champion)",
            [0.40 * p_gbert[i] + 0.60 * p_xlmr_avg[i] for i in range(num_ex)],
        ),
        (
            "Standalone GELECTRA",
            [1.00 * p_gelectra[i] for i in range(num_ex)],
        ),
        (
            "Standalone GBERT",
            [1.00 * p_gbert[i] for i in range(num_ex)],
        ),
        (
            "Standalone XLM-R 42",
            [1.00 * p_xlmr42[i] for i in range(num_ex)],
        ),
        (
            "Variant A (0.25 GB + 0.20 ELE + 0.55 XLMR)",
            [0.25 * p_gbert[i] + 0.20 * p_gelectra[i] + 0.55 * p_xlmr_avg[i] for i in range(num_ex)],
        ),
        (
            "Variant B (0.25 GB + 0.25 ELE + 0.50 XLMR)",
            [0.25 * p_gbert[i] + 0.25 * p_gelectra[i] + 0.50 * p_xlmr_avg[i] for i in range(num_ex)],
        ),
        (
            "Variant C (0.35 GB + 0.15 ELE + 0.50 XLMR)",
            [0.35 * p_gbert[i] + 0.15 * p_gelectra[i] + 0.50 * p_xlmr_avg[i] for i in range(num_ex)],
        ),
        (
            "Variant D (0.40 ELE + 0.60 XLMR)",
            [0.40 * p_gelectra[i] + 0.60 * p_xlmr_avg[i] for i in range(num_ex)],
        ),
    ]

    print("\n" + "=" * 70)
    print("EVALUATING PREDEFINED ENSEMBLE VARIANTS")
    print("=" * 70)

    results = []
    base_pred_count = None

    for name, combined_probs in variants:
        res = evaluate_variant(name, combined_probs, val_examples, id2label)
        if base_pred_count is None and "Baseline" in name:
            base_pred_count = res["total_predicted_entities"]
        results.append(res)

    # 5. Display Comprehensive Comparison Table
    print(f"\nFrozen Champion Benchmark: {FROZEN_CHAMPION_F02:.6f}")
    print("-" * 105)
    print(
        f"{'Variant':<42}{'F0.2':<11}{'Delta':<10}{'Cat1':<10}{'Cat2':<10}{'Prec':<10}{'Rec':<10}{'Preds':<8}{'Diff':<8}"
    )
    print("-" * 105)

    for r in results:
        delta = r["f02"] - FROZEN_CHAMPION_F02
        delta_str = f"{delta:+.6f}"
        diff_preds = r["total_predicted_entities"] - base_pred_count
        diff_str = f"{diff_preds:+d}" if diff_preds != 0 else "0"

        print(
            f"{r['name']:<42}"
            f"{r['f02']:<11.6f}"
            f"{delta_str:<10}"
            f"{r['cat1_f02']:<10.6f}"
            f"{r['cat2_f02']:<10.6f}"
            f"{r['precision']:<10.4f}"
            f"{r['recall']:<10.4f}"
            f"{r['total_predicted_entities']:<8}"
            f"{diff_str:<8}"
        )

    print("-" * 105)

    # 6. Evaluation Verdict
    best_candidate = max(results[1:], key=lambda x: x["f02"])
    print(f"\nBest Performing Challenger Variant: {best_candidate['name']}")
    print(f"  F0.2:        {best_candidate['f02']:.6f}")
    print(f"  Delta vs CH: {best_candidate['f02'] - FROZEN_CHAMPION_F02:+.6f}")

    if best_candidate["f02"] > FROZEN_CHAMPION_F02:
        print("  >>> CRITERIA MET: Candidate strictly exceeds champion benchmark. <<<")
    else:
        print("  >>> CRITERIA NOT MET: No variant exceeds frozen champion benchmark (0.879852). <<<")

    return results


if __name__ == "__main__":
    main()
