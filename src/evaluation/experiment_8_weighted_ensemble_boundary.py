from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
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
from src.evaluation.entity_metrics import bio_to_entities
from src.evaluation.competition_metrics import calculate_competition_metrics


GBERT_CHECKPOINT = Path("checkpoints/gbert_baseline_best")
XLMR_CHECKPOINT = Path("checkpoints/xlmr_baseline_best")

VAL_BATCH_SIZE = 8
IGNORE_LABEL_ID = -100

VEHICLE_MODEL_B = "B-Kompatibles_Fahrzeug_Modell"
VEHICLE_MODEL_I = "I-Kompatibles_Fahrzeug_Modell"

THRESHOLDS = [
    0.05,
    
]


def collect_active_logits(
    model,
    dataset,
    tokenizer,
    device,
) -> list[list[list[float]]]:
    """
    Run evaluation and extract logits for active original tokens only.
    """
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=True,
    )

    model.eval()
    all_active_logits = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }

            outputs = model(**batch)
            logits = outputs.logits
            labels = batch["labels"]

            batch_size_actual = labels.shape[0]

            for row_index in range(batch_size_actual):
                row_labels = labels[row_index]
                row_logits = logits[row_index]

                active_mask = row_labels != IGNORE_LABEL_ID
                active_logits = (
                    row_logits[active_mask]
                    .detach()
                    .cpu()
                    .tolist()
                )

                all_active_logits.append(active_logits)

    return all_active_logits


def ensemble_predictions(
    gbert_logits,
    xlmr_logits,
    id2label: dict,
    gbert_weight: float = 0.5,
    xlmr_weight: float = 0.5,
) -> tuple[list[list[str]], list[list[list[float]]]]:
    """
    Convert logits to softmax probabilities, combine with weights at original-token
    level, and return predicted BIO labels and combined probabilities.
    """
    predictions = []
    all_combined_probabilities = []

    if len(gbert_logits) != len(xlmr_logits):
        raise ValueError(
            "GBERT and XLM-R validation example counts do not match."
        )

    for example_index, (gbert_example, xlmr_example) in enumerate(
        zip(gbert_logits, xlmr_logits)
    ):
        if len(gbert_example) != len(xlmr_example):
            raise ValueError(
                f"Token count mismatch at validation example {example_index}: "
                f"GBERT={len(gbert_example)}, XLM-R={len(xlmr_example)}"
            )

        example_predictions = []
        example_probabilities = []

        for gbert_token_logits, xlmr_token_logits in zip(
            gbert_example, xlmr_example
        ):
            gbert_tensor = torch.tensor(
                gbert_token_logits,
                dtype=torch.float32,
            )
            xlmr_tensor = torch.tensor(
                xlmr_token_logits,
                dtype=torch.float32,
            )

            gbert_probabilities = torch.softmax(gbert_tensor, dim=-1)
            xlmr_probabilities = torch.softmax(xlmr_tensor, dim=-1)

            combined_probabilities = (
                gbert_weight * gbert_probabilities
                + xlmr_weight * xlmr_probabilities
            )

            predicted_id = int(torch.argmax(combined_probabilities).item())

            example_predictions.append(id2label[predicted_id])
            example_probabilities.append(combined_probabilities.tolist())

        predictions.append(example_predictions)
        all_combined_probabilities.append(example_probabilities)

    return predictions, all_combined_probabilities


def repair_ensemble_boundaries(
    predicted_bio: list[str],
    token_probabilities: list[list[float]],
    label2id: dict,
    threshold: float,
) -> tuple[list[str], int]:
    """
    Selectively convert:
        B-Kompatibles_Fahrzeug_Modell -> I-Kompatibles_Fahrzeug_Modell
    only when:
    1. previous predicted label is B- or I-Kompatibles_Fahrzeug_Modell
    2. current predicted label is B-Kompatibles_Fahrzeug_Modell
    3. current B probability >= current I probability
    4. probability gap (B - I) is <= threshold
    """
    repaired = predicted_bio.copy()
    changed = 0

    b_id = label2id[VEHICLE_MODEL_B]
    i_id = label2id[VEHICLE_MODEL_I]

    for index in range(1, len(repaired)):
        previous_label = repaired[index - 1]
        current_label = repaired[index]

        previous_is_vehicle_model = (
            previous_label in {VEHICLE_MODEL_B, VEHICLE_MODEL_I}
        )

        if not previous_is_vehicle_model:
            continue

        if current_label != VEHICLE_MODEL_B:
            continue

        current_probabilities = token_probabilities[index]
        b_probability = current_probabilities[b_id]
        i_probability = current_probabilities[i_id]

        confidence_gap = b_probability - i_probability

        if 0.0 <= confidence_gap <= threshold:
            repaired[index] = VEHICLE_MODEL_I
            changed += 1

    return repaired, changed


def logits_to_predictions(
    all_logits,
    id2label: dict,
) -> list[list[str]]:
    """
    Convert raw logits to BIO sequence predictions by argmax.
    """
    predictions = []

    for example_logits in all_logits:
        example_predictions = []
        for token_logits in example_logits:
            predicted_id = int(
                torch.argmax(
                    torch.tensor(token_logits, dtype=torch.float32)
                ).item()
            )
            example_predictions.append(id2label[predicted_id])
        predictions.append(example_predictions)

    return predictions


def build_records(
    examples: list[dict],
    predicted_sequences: list[list[str]],
) -> list[dict]:
    """
    Convert BIO predictions back to entities for competition evaluation.
    """
    records = []

    for example, predicted_bio in zip(examples, predicted_sequences):
        tokens = example["tokens"]
        true_bio = example["bio_tags"]

        true_entities = bio_to_entities(tokens, true_bio)
        predicted_entities = bio_to_entities(tokens, predicted_bio)

        records.append(
            {
                "category": example["category"],
                "true_entities": true_entities,
                "predicted_entities": predicted_entities,
            }
        )

    return records


def calculate_score(
    examples,
    predictions,
) -> float:
    records = build_records(examples, predictions)
    metrics = calculate_competition_metrics(records)
    return metrics["final_score"]


def main() -> None:
    print(
        "--- EXPERIMENT 8: "
        "40/60 ENSEMBLE + FIXED BOUNDARY REPAIR ---"
    )

    if not torch.cuda.is_available():
        print("CUDA is not available. Stopping.")
        return

    device = torch.device("cuda")
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # -----------------------------------------
    # LOAD COMMON VALIDATION DATA
    # -----------------------------------------
    df = load_training_data()
    examples = build_examples(df)
    label2id, id2label = create_label_mappings(examples)
    _, val_examples = split_examples(examples)

    print(f"Validation examples: {len(val_examples)}")

    # =========================================
    # GBERT
    # =========================================
    print("\n--- RUNNING GBERT ---")

    gbert_tokenizer = AutoTokenizer.from_pretrained(GBERT_CHECKPOINT)
    gbert_model = AutoModelForTokenClassification.from_pretrained(
        GBERT_CHECKPOINT
    )
    gbert_model.to(device)

    gbert_dataset, gbert_stats = prepare_split(
        val_examples,
        gbert_tokenizer,
        label2id,
    )
    print(
        f"GBERT alignment mismatches: "
        f"{gbert_stats['alignment_mismatches']}"
    )

    gbert_logits = collect_active_logits(
        model=gbert_model,
        dataset=gbert_dataset,
        tokenizer=gbert_tokenizer,
        device=device,
    )

    gbert_predictions = logits_to_predictions(gbert_logits, id2label)
    gbert_score = calculate_score(val_examples, gbert_predictions)
    print(f"GBERT raw F0.2: {gbert_score:.6f}")

    # Free GPU memory before loading XLM-R
    del gbert_model
    del gbert_dataset
    torch.cuda.empty_cache()

    # =========================================
    # XLM-R
    # =========================================
    print("\n--- RUNNING XLM-R ---")

    xlmr_tokenizer = AutoTokenizer.from_pretrained(XLMR_CHECKPOINT)
    xlmr_model = AutoModelForTokenClassification.from_pretrained(
        XLMR_CHECKPOINT
    )
    xlmr_model.to(device)

    xlmr_dataset, xlmr_stats = prepare_split(
        val_examples,
        xlmr_tokenizer,
        label2id,
    )
    print(
        f"XLM-R alignment mismatches: "
        f"{xlmr_stats['alignment_mismatches']}"
    )

    xlmr_logits = collect_active_logits(
        model=xlmr_model,
        dataset=xlmr_dataset,
        tokenizer=xlmr_tokenizer,
        device=device,
    )

    xlmr_predictions = logits_to_predictions(xlmr_logits, id2label)
    xlmr_score = calculate_score(val_examples, xlmr_predictions)
    print(f"XLM-R raw F0.2: {xlmr_score:.6f}")

    # Free GPU memory
    del xlmr_model
    del xlmr_dataset
    torch.cuda.empty_cache()

    # =========================================
    # 40 / 60 ENSEMBLE
    # =========================================
    print("\n--- 40/60 ENSEMBLE ---")

    (
        ensemble_predictions_result,
        ensemble_probabilities,
    ) = ensemble_predictions(
        gbert_logits,
        xlmr_logits,
        id2label,
        gbert_weight=0.4,
        xlmr_weight=0.6,
    )

    ensemble_score = calculate_score(
        val_examples,
        ensemble_predictions_result,
    )

    print(f"GBERT raw F0.2: {gbert_score:.6f}")
    print(f"XLM-R raw F0.2: {xlmr_score:.6f}")
    print(f"40/60 ensemble F0.2: {ensemble_score:.6f}")

    # =========================================
    # BOUNDARY REPAIR THRESHOLD SWEEP
    # =========================================
    print("\n--- ENSEMBLE BOUNDARY THRESHOLD RESULTS ---")
    print(f"Raw ensemble F0.2: {ensemble_score:.6f}")

    best_threshold = None
    best_repaired_score = ensemble_score

    for threshold in THRESHOLDS:
        repaired_predictions = []
        total_changes = 0

        for predicted_bio, token_probabilities in zip(
            ensemble_predictions_result,
            ensemble_probabilities,
        ):
            repaired_bio, changed = repair_ensemble_boundaries(
                predicted_bio,
                token_probabilities,
                label2id,
                threshold,
            )
            repaired_predictions.append(repaired_bio)
            total_changes += changed

        repaired_score = calculate_score(
            val_examples,
            repaired_predictions,
        )

        change = repaired_score - ensemble_score

        print(
            f"Threshold {threshold:.2f} | "
            f"changes={total_changes} | "
            f"F0.2={repaired_score:.6f} | "
            f"change={change:+.6f}"
        )

        if repaired_score > best_repaired_score:
            best_repaired_score = repaired_score
            best_threshold = threshold

    # =========================================
    # EXPERIMENT 8 SUMMARY
    # =========================================
    print("\n--- EXPERIMENT 8 SUMMARY ---")
    print(f"Raw ensemble F0.2: {ensemble_score:.6f}")

    if best_threshold is None:
        print("REJECT: boundary repair did not improve ensemble.")
    else:
        print(f"Best threshold: {best_threshold:.2f}")
        print(f"Best repaired F0.2: {best_repaired_score:.6f}")
        print(
            f"Improvement vs raw ensemble: "
            f"{best_repaired_score - ensemble_score:+.6f}"
        )

        best_single = max(gbert_score, xlmr_score)
        print(
            f"Change vs best raw single model: "
            f"{best_repaired_score - best_single:+.6f}"
        )


if __name__ == "__main__":
    main()
