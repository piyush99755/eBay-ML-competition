from collections import Counter


BETA = 0.2


def bio_to_entities(
    tokens: list[str],
    bio_tags: list[str],
) -> list[tuple[str, str]]:
    """
    Convert BIO labels back into extracted entities.

    Returns:
        [
            (aspect_name, aspect_value),
            ...
        ]

    Example:
        tokens:
            ["Bosch", "brake", "disc"]

        labels:
            ["B-Hersteller", "B-Produktart", "I-Produktart"]

        becomes:
            [
                ("Hersteller", "Bosch"),
                ("Produktart", "brake disc"),
            ]
    """

    if len(tokens) != len(bio_tags):
        raise ValueError(
            "Token count and BIO label count must match."
        )

    entities = []

    current_aspect = None
    current_tokens = []

    def close_current_entity() -> None:
        nonlocal current_aspect, current_tokens

        if current_aspect is not None and current_tokens:
            value = " ".join(current_tokens)

            entities.append(
                (
                    current_aspect,
                    value,
                )
            )

        current_aspect = None
        current_tokens = []

    for token, tag in zip(tokens, bio_tags):

        # Outside / ignored token
        if tag == "O":
            close_current_entity()
            continue

        # Beginning of a new entity
        if tag.startswith("B-"):
            close_current_entity()

            current_aspect = tag[2:]
            current_tokens = [token]

            continue

        # Continuation of an entity
        if tag.startswith("I-"):
            aspect = tag[2:]

            # Normal continuation:
            # I-X follows B-X or I-X.
            if current_aspect == aspect:
                current_tokens.append(token)

            # Invalid BIO sequence produced by model:
            # treat I-X as the beginning of a new X entity.
            else:
                close_current_entity()

                current_aspect = aspect
                current_tokens = [token]

            continue

        raise ValueError(f"Unknown BIO tag: {tag}")

    # If title ended while an entity was still open.
    close_current_entity()

    return entities


def filter_entities_by_category(
    entities: list[tuple[str, str]],
    category: int,
    allowed_cat1: set[str] = None,
    allowed_cat2: set[str] = None,
) -> list[tuple[str, str]]:
    """
    Filter extracted entities to only those allowed for the specific listing category.
    Category 1 allows 23 aspects; Category 2 allows 20 aspects (per Annexure.pdf pp. 11-14).
    """
    if allowed_cat1 is None or allowed_cat2 is None:
        from scripts.validate_submission import (
            OFFICIAL_ASPECTS_CAT1,
            OFFICIAL_ASPECTS_CAT2,
        )
        allowed_cat1 = OFFICIAL_ASPECTS_CAT1
        allowed_cat2 = OFFICIAL_ASPECTS_CAT2

    allowed = allowed_cat1 if category == 1 else allowed_cat2
    return [(asp, val) for asp, val in entities if asp in allowed]


def count_matches(
    predicted_entities: list[tuple[str, str]],
    true_entities: list[tuple[str, str]],
) -> tuple[int, int, int]:
    """
    Count:
    - true positives
    - false positives
    - false negatives

    Counter is used instead of set because duplicate
    entities can legitimately occur.
    """

    predicted_counter = Counter(predicted_entities)
    true_counter = Counter(true_entities)

    matched_counter = (
        predicted_counter & true_counter
    )

    true_positives = sum(
        matched_counter.values()
    )

    predicted_total = sum(
        predicted_counter.values()
    )

    true_total = sum(
        true_counter.values()
    )

    false_positives = (
        predicted_total - true_positives
    )

    false_negatives = (
        true_total - true_positives
    )

    return (
        true_positives,
        false_positives,
        false_negatives,
    )


def precision_recall_fbeta(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
    beta: float = BETA,
) -> dict:
    """
    Calculate precision, recall and F-beta.
    """

    precision_denominator = (
        true_positives + false_positives
    )

    recall_denominator = (
        true_positives + false_negatives
    )

    precision = (
        true_positives / precision_denominator
        if precision_denominator > 0
        else 0.0
    )

    recall = (
        true_positives / recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    beta_squared = beta ** 2

    f_denominator = (
        beta_squared * precision + recall
    )

    if f_denominator > 0:
        f_beta = (
            (1 + beta_squared)
            * precision
            * recall
            / f_denominator
        )
    else:
        f_beta = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f_beta": f_beta,
    }


def main() -> None:
    """
    Test the evaluation logic with synthetic data only.
    """

    tokens = [
        "Bosch",
        "brake",
        "disc",
        "front",
        "kit",
    ]

    true_bio = [
        "B-Hersteller",
        "B-Produktart",
        "I-Produktart",
        "B-Einbauposition",
        "O",
    ]

    predicted_bio = [
        "B-Hersteller",
        "B-Produktart",
        "I-Produktart",
        "O",
        "O",
    ]

    true_entities = bio_to_entities(
        tokens,
        true_bio,
    )

    predicted_entities = bio_to_entities(
        tokens,
        predicted_bio,
    )

    tp, fp, fn = count_matches(
        predicted_entities,
        true_entities,
    )

    metrics = precision_recall_fbeta(
        tp,
        fp,
        fn,
    )

    print("--- SYNTHETIC ENTITY TEST ---")

    print(f"True entities: {true_entities}")
    print(
        f"Predicted entities: "
        f"{predicted_entities}"
    )

    print("\n--- COUNTS ---")
    print(f"True positives: {tp}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")

    print("\n--- METRICS ---")
    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F{BETA}: "
        f"{metrics['f_beta']:.4f}"
    )


if __name__ == "__main__":
    main()