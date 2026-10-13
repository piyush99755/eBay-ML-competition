from collections import Counter, defaultdict

from src.evaluation.entity_metrics import (
    BETA,
    count_matches,
    precision_recall_fbeta,
)


def calculate_competition_metrics(
    records: list[dict],
) -> dict:
    """
    Calculate a local competition-style F-beta score.

    Each record must contain:
    {
        "category": int,
        "true_entities": list[tuple[str, str]],
        "predicted_entities": list[tuple[str, str]],
    }

    Scoring flow:
    - group by category
    - group by aspect
    - calculate TP / FP / FN
    - calculate F0.2 per aspect
    - weight aspects by true entity frequency
    - average the category scores
    """

    category_aspect_counts = defaultdict(
        lambda: defaultdict(
            lambda: {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "true_count": 0,
            }
        )
    )

    for record in records:
        category = record["category"]

        true_entities = record["true_entities"]
        predicted_entities = record["predicted_entities"]

        true_by_aspect = defaultdict(list)
        predicted_by_aspect = defaultdict(list)

        for aspect, value in true_entities:
            true_by_aspect[aspect].append(
                (aspect, value)
            )

        for aspect, value in predicted_entities:
            predicted_by_aspect[aspect].append(
                (aspect, value)
            )

        all_aspects = (
            set(true_by_aspect)
            | set(predicted_by_aspect)
        )

        for aspect in all_aspects:
            true_for_aspect = true_by_aspect[aspect]
            predicted_for_aspect = predicted_by_aspect[aspect]

            tp, fp, fn = count_matches(
                predicted_for_aspect,
                true_for_aspect,
            )

            counts = category_aspect_counts[
                category
            ][aspect]

            counts["tp"] += tp
            counts["fp"] += fp
            counts["fn"] += fn
            counts["true_count"] += len(
                true_for_aspect
            )

    category_results = {}

    for category, aspect_counts in (
        category_aspect_counts.items()
    ):
        total_true_entities = sum(
            counts["true_count"]
            for counts in aspect_counts.values()
        )

        weighted_score = 0.0
        aspect_results = {}

        for aspect, counts in aspect_counts.items():
            metrics = precision_recall_fbeta(
                counts["tp"],
                counts["fp"],
                counts["fn"],
                beta=BETA,
            )

            if total_true_entities > 0:
                weight = (
                    counts["true_count"]
                    / total_true_entities
                )
            else:
                weight = 0.0

            weighted_f_beta = (
                weight * metrics["f_beta"]
            )

            weighted_score += weighted_f_beta

            aspect_results[aspect] = {
                "tp": counts["tp"],
                "fp": counts["fp"],
                "fn": counts["fn"],
                "true_count": counts["true_count"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f_beta": metrics["f_beta"],
                "weight": weight,
                "weighted_f_beta": weighted_f_beta,
            }

        category_results[category] = {
            "score": weighted_score,
            "aspects": aspect_results,
        }

    if category_results:
        final_score = sum(
            result["score"]
            for result in category_results.values()
        ) / len(category_results)
    else:
        final_score = 0.0

    return {
        "beta": BETA,
        "categories": category_results,
        "final_score": final_score,
    }


def main() -> None:
    """
    Synthetic test only.
    """

    records = [
        {
            "category": 1,
            "true_entities": [
                ("Hersteller", "Bosch"),
                ("Produktart", "brake disc"),
                ("Einbauposition", "front"),
            ],
            "predicted_entities": [
                ("Hersteller", "Bosch"),
                ("Produktart", "brake disc"),
            ],
        },
        {
            "category": 2,
            "true_entities": [
                ("Hersteller", "ExampleBrand"),
                ("Produktart", "timing kit"),
            ],
            "predicted_entities": [
                ("Hersteller", "ExampleBrand"),
                ("Produktart", "wrong value"),
            ],
        },
    ]

    results = calculate_competition_metrics(
        records
    )

    print("--- LOCAL COMPETITION METRIC TEST ---")

    for category, result in sorted(
        results["categories"].items()
    ):
        print(
            f"Category {category} score: "
            f"{result['score']:.6f}"
        )

        for aspect, metrics in sorted(
            result["aspects"].items()
        ):
            print(
                f"  {aspect}: "
                f"P={metrics['precision']:.4f}, "
                f"R={metrics['recall']:.4f}, "
                f"F{BETA}={metrics['f_beta']:.4f}, "
                f"weight={metrics['weight']:.4f}"
            )

    print(
        f"\nFinal averaged score: "
        f"{results['final_score']:.6f}"
    )


if __name__ == "__main__":
    main()