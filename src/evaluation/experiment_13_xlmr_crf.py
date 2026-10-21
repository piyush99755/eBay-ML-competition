"""
Experiment 13: XLM-RoBERTa + Linear-Chain CRF Evaluation.

Evaluates the isolated structured sequence modeling checkpoint:
checkpoints/xlmr_crf_seed42_best

Compares directly against the raw XLM-R seed-42 baseline (0.869284).
No boundary heuristics, no ensembling, no multi-seed averaging.
"""

from pathlib import Path
import sys
from typing import Dict, List

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
from src.modeling.xlmr_crf import XLMRobertaCRF
from src.evaluation.entity_metrics import bio_to_entities
from src.evaluation.competition_metrics import calculate_competition_metrics
from src.evaluation.error_analysis import (
    analyze_token_confusions,
    analyze_entity_errors,
    print_top_confusions,
    print_entity_error_summary,
)


CHECKPOINT_DIR = Path("checkpoints/xlmr_crf_seed42_best")
BASELINE_XLMR_SEED42_SCORE = 0.869284
VAL_BATCH_SIZE = 8


def evaluate_crf_model(
    model: XLMRobertaCRF,
    dataset,
    examples: List[dict],
    tokenizer,
    id2label: dict,
    device: torch.device,
    batch_size: int = VAL_BATCH_SIZE,
) -> dict:
    """Run Viterbi decoding over validation split and compute competition metrics."""
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
    example_index = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }

            decoded_batch = model.decode(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch.get("labels"),
            )

            batch_size_actual = len(decoded_batch)

            for row_index in range(batch_size_actual):
                example = examples[example_index]
                predicted_ids = decoded_batch[row_index]

                predicted_bio = [id2label[lid] for lid in predicted_ids]
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
                example_index += 1

    if example_index != len(examples):
        raise ValueError("Not all validation examples were evaluated.")

    results = calculate_competition_metrics(records)

    return {
        "metrics": results,
        "records": records,
        "true_bio_sequences": true_bio_sequences,
        "predicted_bio_sequences": predicted_bio_sequences,
    }


def inspect_vehicle_model_aspect(
    evaluation_output: dict,
    aspect_name: str = "Kompatibles_Fahrzeug_Modell",
) -> None:
    """Detailed inspection of Kompatibles_Fahrzeug_Modell performance and errors."""
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

    # Specific token confusions for B- and I- tags
    true_bio_sequences = evaluation_output["true_bio_sequences"]
    predicted_bio_sequences = evaluation_output["predicted_bio_sequences"]
    confusions = analyze_token_confusions(
        true_bio_sequences,
        predicted_bio_sequences,
    )

    print(f"\n  Token confusions involving {aspect_name}:")
    target_tags = {f"B-{aspect_name}", f"I-{aspect_name}"}
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


def main() -> None:
    print("==================================================")
    print("EXPERIMENT 13: XLM-RoBERTa + LINEAR-CHAIN CRF")
    print("==================================================")

    # 1. Defensive Checkpoint Check
    crf_head_file = CHECKPOINT_DIR / "crf_head.pt"
    if not CHECKPOINT_DIR.exists() or not crf_head_file.exists():
        print(
            f"\n[!] Checkpoint not found at: {CHECKPOINT_DIR}\n"
            f"Please run training first before evaluating Experiment 13:\n"
            f"  python -m src.modeling.train_xlmr_crf\n"
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
    model = XLMRobertaCRF.from_pretrained(CHECKPOINT_DIR)
    model.to(device)

    # 5. Prepare Validation Dataset
    val_dataset, val_stats = prepare_split(
        val_examples,
        tokenizer,
        label2id,
    )
    print(f"Validation alignment mismatches: {val_stats['alignment_mismatches']}")

    # 6. Evaluate Model
    print("\nRunning Viterbi decoding and evaluation...")
    evaluation_output = evaluate_crf_model(
        model=model,
        dataset=val_dataset,
        examples=val_examples,
        tokenizer=tokenizer,
        id2label=id2label,
        device=device,
        batch_size=VAL_BATCH_SIZE,
    )

    metrics = evaluation_output["metrics"]
    crf_score = metrics["final_score"]
    delta = crf_score - BASELINE_XLMR_SEED42_SCORE

    cat1_score = metrics["categories"].get(1, {}).get("score", 0.0)
    cat2_score = metrics["categories"].get(2, {}).get("score", 0.0)

    # 7. Print Results
    print("\n==================================================")
    print("EXPERIMENT 13 RESULTS")
    print("==================================================")
    print(f"Existing XLM-R seed-42 baseline:  {BASELINE_XLMR_SEED42_SCORE:.6f}")
    print(f"XLM-R + Linear-Chain CRF:        {crf_score:.6f}")
    print(f"Delta vs baseline:                {delta:+.6f}")
    print(f"Category 1 score:                 {cat1_score:.6f}")
    print(f"Category 2 score:                 {cat2_score:.6f}")

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

    # 9. Clean up VRAM
    del model
    del val_dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
