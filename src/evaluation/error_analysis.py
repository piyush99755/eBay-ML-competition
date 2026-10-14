from collections import Counter, defaultdict


def analyze_token_confusions(
    true_bio_sequences: list[list[str]],
    predicted_bio_sequences: list[list[str]],
) -> Counter:
    """
    Count token-level true-label -> predicted-label confusions.

    Example:
        B-Kompatibles_Fahrzeug_Modell -> O
    """

    confusion_counts = Counter()

    for true_bio, predicted_bio in zip(
        true_bio_sequences,
        predicted_bio_sequences,
    ):
        if len(true_bio) != len(predicted_bio):
            raise ValueError(
                "True and predicted BIO sequence lengths must match."
            )

        for true_label, predicted_label in zip(
            true_bio,
            predicted_bio,
        ):
            if true_label != predicted_label:
                confusion_counts[
                    (true_label, predicted_label)
                ] += 1

    return confusion_counts


def analyze_entity_errors(
    records: list[dict],
) -> dict:
    """
    Aggregate false positives and false negatives
    by aspect.

    Each record should contain:
    {
        "category": int,
        "true_entities": [...],
        "predicted_entities": [...],
    }
    """

    aspect_errors = defaultdict(
        lambda: {
            "false_positives": 0,
            "false_negatives": 0,
        }
    )

    for record in records:
        true_counter = Counter(
            record["true_entities"]
        )

        predicted_counter = Counter(
            record["predicted_entities"]
        )

        false_positive_counter = (
            predicted_counter - true_counter
        )

        false_negative_counter = (
            true_counter - predicted_counter
        )

        for (
            aspect,
            _value,
        ), count in false_positive_counter.items():

            aspect_errors[aspect][
                "false_positives"
            ] += count

        for (
            aspect,
            _value,
        ), count in false_negative_counter.items():

            aspect_errors[aspect][
                "false_negatives"
            ] += count

    return dict(aspect_errors)


def print_top_confusions(
    confusion_counts: Counter,
    top_n: int = 25,
) -> None:
    """
    Print the most common token-level label confusions.
    """

    print(
        "\n--- TOP TOKEN-LEVEL CONFUSIONS ---"
    )

    for (
        true_label,
        predicted_label,
    ), count in confusion_counts.most_common(
        top_n
    ):

        print(
            f"{true_label} "
            f"-> {predicted_label} "
            f"| count={count}"
        )


def print_entity_error_summary(
    aspect_errors: dict,
) -> None:
    """
    Print entity-level FP/FN counts by aspect.
    """

    print(
        "\n--- ENTITY ERROR SUMMARY ---"
    )

    sorted_aspects = sorted(
        aspect_errors.items(),
        key=lambda item: (
            item[1]["false_positives"]
            + item[1]["false_negatives"]
        ),
        reverse=True,
    )

    for aspect, errors in sorted_aspects:

        fp = errors["false_positives"]
        fn = errors["false_negatives"]

        print(
            f"{aspect}: "
            f"FP={fp} | "
            f"FN={fn} | "
            f"total_errors={fp + fn}"
        )