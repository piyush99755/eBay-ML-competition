from pathlib import Path

import torch

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from src.data.build_examples import (
    build_examples,
    load_training_data,
)
from src.data.label_mapping import (
    create_label_mappings,
)
from src.data.make_splits import (
    split_examples,
)
from src.modeling.prepare_datasets import (
    prepare_split,
)

from src.evaluation.evaluate_model import (
    evaluate_model_entities,
)
from src.evaluation.entity_metrics import (
    bio_to_entities,
)
from src.evaluation.competition_metrics import (
    calculate_competition_metrics,
)


CHECKPOINT_DIR = Path(
    "checkpoints/gbert_baseline_best"
)

VAL_BATCH_SIZE = 8

VEHICLE_MODEL_B = (
    "B-Kompatibles_Fahrzeug_Modell"
)

VEHICLE_MODEL_I = (
    "I-Kompatibles_Fahrzeug_Modell"
)


def repair_vehicle_model_boundaries(
    predicted_bio: list[str],
) -> list[str]:
    """
    Experiment 1:

    If GBERT predicts B-VehicleModel immediately
    after another VehicleModel token, convert
    the new B into I.

    Example:

        B-VehicleModel
        B-VehicleModel

    becomes:

        B-VehicleModel
        I-VehicleModel
    """

    repaired = predicted_bio.copy()

    for index in range(
        1,
        len(repaired),
    ):
        current_label = repaired[index]
        previous_label = repaired[index - 1]

        previous_is_vehicle_model = (
            previous_label == VEHICLE_MODEL_B
            or
            previous_label == VEHICLE_MODEL_I
        )

        if (
            current_label == VEHICLE_MODEL_B
            and previous_is_vehicle_model
        ):
            repaired[index] = VEHICLE_MODEL_I

    return repaired


def build_records(
    examples: list[dict],
    predicted_sequences: list[list[str]],
) -> list[dict]:
    """
    Convert true and predicted BIO sequences
    back into entities for competition scoring.
    """

    if len(examples) != len(
        predicted_sequences
    ):
        raise ValueError(
            "Example count and prediction "
            "count do not match."
        )

    records = []

    for example, predicted_bio in zip(
        examples,
        predicted_sequences,
    ):
        tokens = example["tokens"]
        true_bio = example["bio_tags"]

        true_entities = bio_to_entities(
            tokens,
            true_bio,
        )

        predicted_entities = bio_to_entities(
            tokens,
            predicted_bio,
        )

        records.append(
            {
                "category": example["category"],
                "true_entities": true_entities,
                "predicted_entities": predicted_entities,
            }
        )

    return records


def main() -> None:

    print(
        "--- EXPERIMENT 1: "
        "VEHICLE-MODEL BOUNDARY MERGE ---"
    )

    # ---------------------------------------------
    # 1. GPU CHECK
    # ---------------------------------------------

    if not torch.cuda.is_available():
        print(
            "CUDA is not available. Stopping."
        )
        return

    device = torch.device("cuda")

    print(f"Device: {device}")
    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    # ---------------------------------------------
    # 2. LOAD SAVED BEST CHECKPOINT
    # ---------------------------------------------

    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{CHECKPOINT_DIR}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT_DIR
    )

    model = (
        AutoModelForTokenClassification
        .from_pretrained(
            CHECKPOINT_DIR
        )
    )

    model.to(device)

    print(
        f"Loaded checkpoint: "
        f"{CHECKPOINT_DIR}"
    )

    # ---------------------------------------------
    # 3. LOAD SAME VALIDATION SPLIT
    # ---------------------------------------------

    df = load_training_data()

    examples = build_examples(df)

    label2id, id2label = (
        create_label_mappings(
            examples
        )
    )

    _, val_examples = split_examples(
        examples
    )

    val_dataset, val_stats = prepare_split(
        val_examples,
        tokenizer,
        label2id,
    )

    print(
        f"Validation examples: "
        f"{len(val_examples)}"
    )

    print(
        f"Alignment mismatches: "
        f"{val_stats['alignment_mismatches']}"
    )

    # ---------------------------------------------
    # 4. GET BASELINE PREDICTIONS
    # ---------------------------------------------

    evaluation_output = (
        evaluate_model_entities(
            model=model,
            dataset=val_dataset,
            examples=val_examples,
            tokenizer=tokenizer,
            id2label=id2label,
            device=device,
            batch_size=VAL_BATCH_SIZE,
        )
    )

    baseline_metrics = (
        evaluation_output["metrics"]
    )

    baseline_score = (
        baseline_metrics["final_score"]
    )

    original_predictions = (
        evaluation_output[
            "predicted_bio_sequences"
        ]
    )

    print(
        f"\nBaseline F0.2: "
        f"{baseline_score:.6f}"
    )

    # ---------------------------------------------
    # 5. APPLY EXPERIMENTAL REPAIR
    # ---------------------------------------------

    repaired_predictions = []

    changed_labels = 0

    for predicted_bio in original_predictions:

        repaired_bio = (
            repair_vehicle_model_boundaries(
                predicted_bio
            )
        )

        changed_labels += sum(
            original != repaired
            for original, repaired in zip(
                predicted_bio,
                repaired_bio,
            )
        )

        repaired_predictions.append(
            repaired_bio
        )

    print(
        f"Boundary labels changed: "
        f"{changed_labels}"
    )

    # ---------------------------------------------
    # 6. REBUILD ENTITIES
    # ---------------------------------------------

    repaired_records = build_records(
        val_examples,
        repaired_predictions,
    )

    # ---------------------------------------------
    # 7. RECALCULATE COMPETITION METRIC
    # ---------------------------------------------

    repaired_metrics = (
        calculate_competition_metrics(
            repaired_records
        )
    )

    repaired_score = (
        repaired_metrics["final_score"]
    )

    score_change = (
        repaired_score - baseline_score
    )

    print(
        f"Experiment F0.2: "
        f"{repaired_score:.6f}"
    )

    print(
        f"Score change: "
        f"{score_change:+.6f}"
    )

    # ---------------------------------------------
    # 8. CATEGORY COMPARISON
    # ---------------------------------------------

    print(
        "\n--- CATEGORY COMPARISON ---"
    )

    for category in sorted(
        baseline_metrics["categories"]
    ):
        baseline_category_score = (
            baseline_metrics[
                "categories"
            ][category]["score"]
        )

        repaired_category_score = (
            repaired_metrics[
                "categories"
            ][category]["score"]
        )

        difference = (
            repaired_category_score
            - baseline_category_score
        )

        print(
            f"Category {category}: "
            f"{baseline_category_score:.6f} "
            f"-> "
            f"{repaired_category_score:.6f} "
            f"({difference:+.6f})"
        )

    # ---------------------------------------------
    # 9. FINAL DECISION
    # ---------------------------------------------

    print(
        "\n--- EXPERIMENT RESULT ---"
    )

    if score_change > 0:
        print(
            "KEEP: boundary repair improved "
            "validation F0.2."
        )

    elif score_change < 0:
        print(
            "REJECT: boundary repair reduced "
            "validation F0.2."
        )

    else:
        print(
            "NO CHANGE: boundary repair had "
            "no effect."
        )


if __name__ == "__main__":
    main()