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
XLMR_SEED_42_CHECKPOINT = Path("checkpoints/xlmr_baseline_best")
XLMR_SEED_7_CHECKPOINT = Path("checkpoints/xlmr_seed_7_best")
XLMR_SEED_21_CHECKPOINT = Path("checkpoints/xlmr_seed_21_best")

VAL_BATCH_SIZE = 8
IGNORE_LABEL_ID = -100

VEHICLE_MODEL_B = "B-Kompatibles_Fahrzeug_Modell"
VEHICLE_MODEL_I = "I-Kompatibles_Fahrzeug_Modell"

THRESHOLD = 0.05
CURRENT_CHAMPION_SCORE = 0.879852


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


def ensemble_gbert_multiseed_xlmr_predictions(
    gbert_logits,
    seed_42_logits,
    seed_7_logits,
    seed_21_logits,
    id2label: dict,
    gbert_weight: float = 0.40,
    xlmr_weight: float = 0.60,
) -> tuple[list[list[str]], list[list[list[float]]]]:
    """
    Combine GBERT and 3-seed XLM-R probabilities:
    1. Average the 3 XLM-R seed probabilities equally: (p42 + p7 + p21) / 3
    2. Combine: gbert_weight * GBERT_prob + xlmr_weight * XLMR_multiseed_prob
    3. Argmax to predict the final BIO label.
    4. Return predicted BIO sequences and token-level combined probabilities.
    """
    predictions = []
    all_combined_probabilities = []

    num_examples = len(gbert_logits)
    if not (
        len(seed_42_logits) == num_examples
        and len(seed_7_logits) == num_examples
        and len(seed_21_logits) == num_examples
    ):
        raise ValueError("Example counts across models do not match.")

    for example_index, (ex_gbert, ex_42, ex_7, ex_21) in enumerate(
        zip(gbert_logits, seed_42_logits, seed_7_logits, seed_21_logits)
    ):
        if not (len(ex_gbert) == len(ex_42) == len(ex_7) == len(ex_21)):
            raise ValueError(
                f"Token count mismatch at validation example {example_index}: "
                f"gbert={len(ex_gbert)}, seed_42={len(ex_42)}, "
                f"seed_7={len(ex_7)}, seed_21={len(ex_21)}"
            )

        example_predictions = []
        example_probabilities = []

        for gbert_t, t_42, t_7, t_21 in zip(ex_gbert, ex_42, ex_7, ex_21):
            gbert_tensor = torch.tensor(gbert_t, dtype=torch.float32)
            tensor_42 = torch.tensor(t_42, dtype=torch.float32)
            tensor_7 = torch.tensor(t_7, dtype=torch.float32)
            tensor_21 = torch.tensor(t_21, dtype=torch.float32)

            gbert_prob = torch.softmax(gbert_tensor, dim=-1)
            prob_42 = torch.softmax(tensor_42, dim=-1)
            prob_7 = torch.softmax(tensor_7, dim=-1)
            prob_21 = torch.softmax(tensor_21, dim=-1)

            xlmr_multiseed_prob = (prob_42 + prob_7 + prob_21) / 3.0

            combined_probabilities = (
                gbert_weight * gbert_prob + xlmr_weight * xlmr_multiseed_prob
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
        "--- EXPERIMENT 11: "
        "GBERT + MULTI-SEED XLM-R ENSEMBLE + BOUNDARY REPAIR ---"
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
    # 1. GBERT BASELINE
    # =========================================
    print("\n--- RUNNING GBERT BASELINE ---")

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

    del gbert_model
    del gbert_dataset
    torch.cuda.empty_cache()

    # =========================================
    # 2. XLM-R SEED 42
    # =========================================
    print("\n--- RUNNING XLM-R (SEED 42) ---")

    tokenizer_42 = AutoTokenizer.from_pretrained(XLMR_SEED_42_CHECKPOINT)
    model_42 = AutoModelForTokenClassification.from_pretrained(
        XLMR_SEED_42_CHECKPOINT
    )
    model_42.to(device)

    dataset_42, stats_42 = prepare_split(
        val_examples,
        tokenizer_42,
        label2id,
    )
    print(
        f"Seed 42 alignment mismatches: "
        f"{stats_42['alignment_mismatches']}"
    )

    logits_42 = collect_active_logits(
        model=model_42,
        dataset=dataset_42,
        tokenizer=tokenizer_42,
        device=device,
    )

    predictions_42 = logits_to_predictions(logits_42, id2label)
    score_42 = calculate_score(val_examples, predictions_42)
    print(f"XLM-R seed 42 F0.2: {score_42:.6f}")

    del model_42
    del dataset_42
    torch.cuda.empty_cache()

    # =========================================
    # 3. XLM-R SEED 7
    # =========================================
    print("\n--- RUNNING XLM-R (SEED 7) ---")

    tokenizer_7 = AutoTokenizer.from_pretrained(XLMR_SEED_7_CHECKPOINT)
    model_7 = AutoModelForTokenClassification.from_pretrained(
        XLMR_SEED_7_CHECKPOINT
    )
    model_7.to(device)

    dataset_7, stats_7 = prepare_split(
        val_examples,
        tokenizer_7,
        label2id,
    )
    print(
        f"Seed 7 alignment mismatches: "
        f"{stats_7['alignment_mismatches']}"
    )

    logits_7 = collect_active_logits(
        model=model_7,
        dataset=dataset_7,
        tokenizer=tokenizer_7,
        device=device,
    )

    predictions_7 = logits_to_predictions(logits_7, id2label)
    score_7 = calculate_score(val_examples, predictions_7)
    print(f"XLM-R seed 7 F0.2: {score_7:.6f}")

    del model_7
    del dataset_7
    torch.cuda.empty_cache()

    # =========================================
    # 4. XLM-R SEED 21
    # =========================================
    print("\n--- RUNNING XLM-R (SEED 21) ---")

    tokenizer_21 = AutoTokenizer.from_pretrained(XLMR_SEED_21_CHECKPOINT)
    model_21 = AutoModelForTokenClassification.from_pretrained(
        XLMR_SEED_21_CHECKPOINT
    )
    model_21.to(device)

    dataset_21, stats_21 = prepare_split(
        val_examples,
        tokenizer_21,
        label2id,
    )
    print(
        f"Seed 21 alignment mismatches: "
        f"{stats_21['alignment_mismatches']}"
    )

    logits_21 = collect_active_logits(
        model=model_21,
        dataset=dataset_21,
        tokenizer=tokenizer_21,
        device=device,
    )

    predictions_21 = logits_to_predictions(logits_21, id2label)
    score_21 = calculate_score(val_examples, predictions_21)
    print(f"XLM-R seed 21 F0.2: {score_21:.6f}")

    del model_21
    del dataset_21
    torch.cuda.empty_cache()

    # =========================================
    # 5. EXPERIMENT 10 RAW ENSEMBLE
    # =========================================
    print("\n--- EXPERIMENT 10 RAW ENSEMBLE (40/60) ---")

    (
        raw_ensemble_predictions,
        ensemble_probabilities,
    ) = ensemble_gbert_multiseed_xlmr_predictions(
        gbert_logits,
        logits_42,
        logits_7,
        logits_21,
        id2label,
        gbert_weight=0.40,
        xlmr_weight=0.60,
    )

    raw_ensemble_score = calculate_score(
        val_examples,
        raw_ensemble_predictions,
    )
    print(f"Experiment 10 raw ensemble F0.2: {raw_ensemble_score:.6f}")

    # =========================================
    # 6. FROZEN 0.05 BOUNDARY REPAIR
    # =========================================
    print(f"\n--- FROZEN VEHICLE-MODEL BOUNDARY REPAIR (THRESHOLD {THRESHOLD:.2f}) ---")

    repaired_predictions = []
    total_changes = 0

    for predicted_bio, token_probabilities in zip(
        raw_ensemble_predictions,
        ensemble_probabilities,
    ):
        repaired_bio, changed = repair_ensemble_boundaries(
            predicted_bio,
            token_probabilities,
            label2id,
            THRESHOLD,
        )
        repaired_predictions.append(repaired_bio)
        total_changes += changed

    repaired_score = calculate_score(
        val_examples,
        repaired_predictions,
    )

    delta_vs_raw = repaired_score - raw_ensemble_score
    delta_vs_champion = repaired_score - CURRENT_CHAMPION_SCORE

    # =========================================
    # 7. EXPERIMENT 11 SUMMARY
    # =========================================
    print("\n--- EXPERIMENT 11 SUMMARY ---")
    print(f"Experiment 10 raw ensemble F0.2: {raw_ensemble_score:.6f}")
    print(f"Number of BIO boundary changes: {total_changes}")
    print(f"Repaired F0.2 (threshold {THRESHOLD:.2f}): {repaired_score:.6f}")
    print(f"Delta vs raw Experiment 10: {delta_vs_raw:+.6f}")
    print(
        f"Delta vs current champion ({CURRENT_CHAMPION_SCORE:.6f}): "
        f"{delta_vs_champion:+.6f}"
    )


if __name__ == "__main__":
    main()
