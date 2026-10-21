"""
Experiment 14: Boundary-Aware Multi-Task XLM-RoBERTa Evaluation.

Evaluates the checkpoint: checkpoints/xlmr_boundary_seed42_best
Compares directly against the raw XLM-R seed-42 baseline (0.869284).

Controlled inference rule:
- Final entity predictions come ONLY from the BIO classifier argmax.
- Boundary head predictions are NOT used to alter BIO labels.
- Auxiliary boundary heads are evaluated separately for diagnostic purposes only.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

from src.data.build_examples import (
    build_examples,
    load_training_data,
)
from src.data.label_mapping import create_label_mappings
from src.data.make_splits import split_examples
from src.modeling.prepare_datasets import prepare_split
from src.modeling.xlmr_boundary import XLMRobertaBoundary, generate_boundary_targets
from src.evaluation.entity_metrics import bio_to_entities
from src.evaluation.competition_metrics import calculate_competition_metrics
from src.evaluation.error_analysis import (
    analyze_token_confusions,
    analyze_entity_errors,
    print_top_confusions,
    print_entity_error_summary,
)


CHECKPOINT_DIR = Path("checkpoints/xlmr_boundary_seed42_best")
BASELINE_XLMR_SEED42_SCORE = 0.869284
VAL_BATCH_SIZE = 8
IGNORE_LABEL_ID = -100


def calculate_binary_metrics(true_binary: List[int], pred_binary: List[int]) -> dict:
    """Calculate accuracy, precision, recall, and F1 for binary predictions."""
    tp = sum(1 for t, p in zip(true_binary, pred_binary) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(true_binary, pred_binary) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(true_binary, pred_binary) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(true_binary, pred_binary) if t == 0 and p == 0)

    total = len(true_binary)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "support": tp + fn,
        "total": total,
    }


def evaluate_boundary_model(
    model: XLMRobertaBoundary,
    dataset,
    examples: List[dict],
    tokenizer,
    id2label: dict,
    device: torch.device,
    batch_size: int = VAL_BATCH_SIZE,
) -> dict:
    """
    Run evaluation over validation split.
    - Final entities are produced strictly from BIO classifier argmax.
    - Boundary heads are collected strictly for diagnostic metrics.
    """
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=True,
    )

    model.eval()
    records = []
    true_bio_sequences = []
    predicted_bio_sequences = []

    all_true_starts = []
    all_pred_starts = []
    all_true_ends = []
    all_pred_ends = []

    example_index = 0

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"]
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }

            outputs = model(**batch)
            bio_logits = outputs.bio_logits
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            batch_size_actual = labels.shape[0]

            for row_index in range(batch_size_actual):
                example = examples[example_index]
                row_labels = labels[row_index]

                active_mask = row_labels != IGNORE_LABEL_ID
                active_len = int(active_mask.sum().item())

                # 1. BIO predictions (Strictly from BIO head argmax)
                row_bio_logits = bio_logits[row_index, :active_len]
                pred_ids = torch.argmax(row_bio_logits, dim=-1).cpu().tolist()

                predicted_bio = [id2label[lid] for lid in pred_ids]
                true_bio = example["bio_tags"]
                tokens = example["tokens"]

                if len(predicted_bio) != len(tokens):
                    raise ValueError(
                        f"Predicted BIO count ({len(predicted_bio)}) != token count ({len(tokens)})."
                    )

                true_bio_sequences.append(true_bio)
                predicted_bio_sequences.append(predicted_bio)

                true_entities = bio_to_entities(tokens, true_bio)
                predicted_entities = bio_to_entities(tokens, predicted_bio)

                records.append(
                    {
                        "category": example["category"],
                        "true_entities": true_entities,
                        "predicted_entities": predicted_entities,
                    }
                )

                # 2. Diagnostic Boundary Predictions (Never used for entities)
                row_start_logits = start_logits[row_index, :active_len]
                row_end_logits = end_logits[row_index, :active_len]

                pred_starts = torch.argmax(row_start_logits, dim=-1).cpu().tolist()
                pred_ends = torch.argmax(row_end_logits, dim=-1).cpu().tolist()

                true_starts, true_ends = generate_boundary_targets(true_bio)

                all_true_starts.extend(true_starts)
                all_pred_starts.extend(pred_starts)
                all_true_ends.extend(true_ends)
                all_pred_ends.extend(pred_ends)

                example_index += 1

    if example_index != len(examples):
        raise ValueError("Not all validation examples were evaluated.")

    results = calculate_competition_metrics(records)

    start_metrics = calculate_binary_metrics(all_true_starts, all_pred_starts)
    end_metrics = calculate_binary_metrics(all_true_ends, all_pred_ends)

    return {
        "metrics": results,
        "records": records,
        "true_bio_sequences": true_bio_sequences,
        "predicted_bio_sequences": predicted_bio_sequences,
        "diagnostic_start": start_metrics,
        "diagnostic_end": end_metrics,
    }


def inspect_vehicle_model_aspect(
    evaluation_output: dict,
    aspect_name: str = "Kompatibles_Fahrzeug_Modell",
) -> None:
    """Detailed inspection of Kompatibles_Fahrzeug_Modell performance, errors, and boundary confusions."""
    print(f"\n--- DETAILED INSPECTION: {aspect_name} ---")

    metrics = evaluation_output["metrics"]
    found = False

    for category, cat_data in sorted(metrics["categories"].items()):
        aspects = cat_data.get("aspects", {})
        if aspect_name in aspects:
            found = True
            asp = aspects[aspect_name]
            print(f"Category {category}:")
            print(f"  Count:     {asp['true_count']}")
            print(f"  Precision: {asp['precision']:.4f}")
            print(f"  Recall:    {asp['recall']:.4f}")
            print(f"  F0.2:      {asp['f_beta']:.4f}")
            print(f"  Weight:    {asp['weight']:.4f}")

    if not found:
        print(f"  Aspect '{aspect_name}' not found in validation categories.")

    # Entity errors
    records = evaluation_output["records"]
    aspect_errors = analyze_entity_errors(records)
    if aspect_name in aspect_errors:
        err = aspect_errors[aspect_name]
        fp = err["false_positives"]
        fn = err["false_negatives"]
        print(f"  Entity Errors: FP={fp} | FN={fn} | Total={fp + fn}")

    # Specific B -> I and I -> B boundary confusions
    true_bio_sequences = evaluation_output["true_bio_sequences"]
    predicted_bio_sequences = evaluation_output["predicted_bio_sequences"]
    confusions = analyze_token_confusions(
        true_bio_sequences,
        predicted_bio_sequences,
    )

    b_tag = f"B-{aspect_name}"
    i_tag = f"I-{aspect_name}"

    b_to_i = confusions[(b_tag, i_tag)]
    i_to_b = confusions[(i_tag, b_tag)]

    print(f"  Boundary Confusions for {aspect_name}:")
    print(f"    B -> I confusion count: {b_to_i}")
    print(f"    I -> B confusion count: {i_to_b}")

    print(f"\n  Top token confusions involving {aspect_name}:")
    target_tags = {b_tag, i_tag}
    aspect_confusions = [
        ((t, p), c)
        for (t, p), c in confusions.most_common()
        if t in target_tags or p in target_tags
    ]

    if aspect_confusions:
        for (t, p), count in aspect_confusions[:15]:
            print(f"    {t} -> {p} | count={count}")
    else:
        print("    None detected.")


def print_diagnostic_boundary_report(evaluation_output: dict) -> None:
    """Print diagnostic metrics for auxiliary START and END heads."""
    print("\n--- DIAGNOSTIC BOUNDARY HEAD EVALUATION (NOT USED FOR ENTITIES) ---")
    start = evaluation_output["diagnostic_start"]
    end = evaluation_output["diagnostic_end"]

    print("START Boundary Head:")
    print(f"  Accuracy:  {start['accuracy']:.4f}")
    print(f"  Precision: {start['precision']:.4f}")
    print(f"  Recall:    {start['recall']:.4f}")
    print(f"  F1:        {start['f1']:.4f}")
    print(f"  Support:   {start['support']} positive START tokens (Total: {start['total']})")

    print("\nEND Boundary Head:")
    print(f"  Accuracy:  {end['accuracy']:.4f}")
    print(f"  Precision: {end['precision']:.4f}")
    print(f"  Recall:    {end['recall']:.4f}")
    print(f"  F1:        {end['f1']:.4f}")
    print(f"  Support:   {end['support']} positive END tokens (Total: {end['total']})")


def main() -> None:
    print("==================================================")
    print("EXPERIMENT 14: BOUNDARY-AWARE MULTI-TASK XLM-R")
    print("==================================================")

    # 1. Defensive Checkpoint Check
    weights_file = CHECKPOINT_DIR / "boundary_heads.pt"
    if not CHECKPOINT_DIR.exists() or not weights_file.exists():
        print(
            f"\n[!] Checkpoint not found at: {CHECKPOINT_DIR}\n"
            f"Please run training first before evaluating Experiment 14:\n"
            f"  python -m src.modeling.train_xlmr_boundary\n"
        )
        return

    # 2. CUDA Check
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU (may be slow).")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    # 3. Load Data & Validation Split (Split-A, random_state=42)
    print("\n--- LOADING DATA & VALIDATION SPLIT ---")
    df = load_training_data()
    examples = build_examples(df)
    label2id, id2label = create_label_mappings(examples)
    _, val_examples = split_examples(examples)
    print(f"Validation titles: {len(val_examples)}")

    # 4. Load Model & Tokenizer
    print(f"\nLoading model from {CHECKPOINT_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    model = XLMRobertaBoundary.from_pretrained(CHECKPOINT_DIR)
    model.to(device)

    # 5. Prepare Validation Dataset
    val_dataset, val_stats = prepare_split(
        val_examples,
        tokenizer,
        label2id,
    )
    print(f"Validation alignment mismatches: {val_stats['alignment_mismatches']}")

    # 6. Evaluate Model
    print("\nRunning BIO-argmax evaluation and diagnostic boundary checks...")
    evaluation_output = evaluate_boundary_model(
        model=model,
        dataset=val_dataset,
        examples=val_examples,
        tokenizer=tokenizer,
        id2label=id2label,
        device=device,
        batch_size=VAL_BATCH_SIZE,
    )

    metrics = evaluation_output["metrics"]
    boundary_score = metrics["final_score"]
    delta = boundary_score - BASELINE_XLMR_SEED42_SCORE

    cat1_score = metrics["categories"].get(1, {}).get("score", 0.0)
    cat2_score = metrics["categories"].get(2, {}).get("score", 0.0)

    # 7. Print Results
    print("\n==================================================")
    print("EXPERIMENT 14 RESULTS")
    print("==================================================")
    print(f"Existing XLM-R seed-42 baseline:      {BASELINE_XLMR_SEED42_SCORE:.6f}")
    print(f"Boundary-Aware Multi-Task XLM-R:     {boundary_score:.6f}")
    print(f"Delta vs baseline:                    {delta:+.6f}")
    print(f"Category 1 score:                     {cat1_score:.6f}")
    print(f"Category 2 score:                     {cat2_score:.6f}")

    # 8. Error Analysis & Vehicle Model Aspect Inspection
    print("\n--- GENERAL ERROR ANALYSIS ---")
    confusions = analyze_token_confusions(
        evaluation_output["true_bio_sequences"],
        evaluation_output["predicted_bio_sequences"],
    )
    aspect_errors = analyze_entity_errors(evaluation_output["records"])
    print_top_confusions(confusions, top_n=20)
    print_entity_error_summary(aspect_errors)

    inspect_vehicle_model_aspect(
        evaluation_output,
        aspect_name="Kompatibles_Fahrzeug_Modell",
    )

    # 9. Diagnostic Boundary Head Evaluation
    print_diagnostic_boundary_report(evaluation_output)

    # 10. Clean up VRAM
    del model
    del val_dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
