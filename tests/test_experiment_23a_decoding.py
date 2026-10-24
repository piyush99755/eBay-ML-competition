"""
Unit tests for Phase 23A: Category-Constrained BIO Decoding.

Validates decoding logic, category label masking, and entity reconstruction
on synthetic inputs without modifying models or datasets.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.evaluation.experiment_23a_constrained_decoding import (
    decode_method_a,
    decode_method_b,
    get_category_allowed_label_indices,
    get_category_forbidden_mask,
    count_invalid_predictions,
)
from scripts.validate_submission import (
    OFFICIAL_ASPECTS,
    OFFICIAL_ASPECTS_CAT1,
    OFFICIAL_ASPECTS_CAT2,
)


def create_synthetic_label_space():
    """Create a minimal 8-label BIO space for testing."""
    # Common: Hersteller, Produktart
    # Cat 1 only: Bremsscheibenart
    # Cat 2 only: Anwendung
    labels = [
        "B-Hersteller", "I-Hersteller",
        "B-Produktart", "I-Produktart",
        "B-Bremsscheibenart", "I-Bremsscheibenart",
        "B-Anwendung",
        "O",
    ]
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}
    return label2id, id2label


def test_allowed_label_indices():
    label2id, _ = create_synthetic_label_space()

    cat1_allowed = get_category_allowed_label_indices(label2id, category=1)
    cat2_allowed = get_category_allowed_label_indices(label2id, category=2)

    # 'O' must be allowed in both
    assert label2id["O"] in cat1_allowed
    assert label2id["O"] in cat2_allowed

    # Common aspects allowed in both
    assert label2id["B-Hersteller"] in cat1_allowed
    assert label2id["B-Hersteller"] in cat2_allowed
    assert label2id["B-Produktart"] in cat1_allowed
    assert label2id["B-Produktart"] in cat2_allowed

    # Bremsscheibenart: Cat 1 allowed, Cat 2 forbidden
    assert label2id["B-Bremsscheibenart"] in cat1_allowed
    assert label2id["B-Bremsscheibenart"] not in cat2_allowed
    assert label2id["I-Bremsscheibenart"] in cat1_allowed
    assert label2id["I-Bremsscheibenart"] not in cat2_allowed

    # Anwendung: Cat 1 forbidden, Cat 2 allowed
    assert label2id["B-Anwendung"] not in cat1_allowed
    assert label2id["B-Anwendung"] in cat2_allowed


def test_forbidden_mask():
    label2id, _ = create_synthetic_label_space()
    cat1_mask = get_category_forbidden_mask(label2id, category=1)
    cat2_mask = get_category_forbidden_mask(label2id, category=2)

    assert cat1_mask[label2id["B-Anwendung"]] == True
    assert cat1_mask[label2id["B-Hersteller"]] == False
    assert cat1_mask[label2id["O"]] == False

    assert cat2_mask[label2id["B-Bremsscheibenart"]] == True
    assert cat2_mask[label2id["I-Bremsscheibenart"]] == True
    assert cat2_mask[label2id["B-Hersteller"]] == False
    assert cat2_mask[label2id["O"]] == False


def test_allowed_predictions_identical():
    """When the model predicts allowed aspects, Method A and Method B produce identical results."""
    label2id, id2label = create_synthetic_label_space()

    # Title: Bosch Bremsscheibe (Category 1)
    # Probabilities highest for B-Hersteller, B-Produktart
    ex = [{"tokens": ["Bosch", "Bremsscheibe"], "category": 1, "bio_tags": ["B-Hersteller", "B-Produktart"]}]
    probs = np.zeros((1, 2, len(label2id)), dtype=np.float32)
    probs[0, 0, label2id["B-Hersteller"]] = 0.95
    probs[0, 0, label2id["O"]] = 0.05
    probs[0, 1, label2id["B-Produktart"]] = 0.90
    probs[0, 1, label2id["O"]] = 0.10

    res_a = decode_method_a(ex, [probs[0]], id2label)
    res_b = decode_method_b(ex, [probs[0]], id2label, label2id)

    assert res_a == [[("Hersteller", "Bosch"), ("Produktart", "Bremsscheibe")]]
    assert res_b == [[("Hersteller", "Bosch"), ("Produktart", "Bremsscheibe")]]
    assert res_a == res_b


def test_forbidden_aspect_runner_up_is_O():
    """In Cat 2, if model's top choice is Bremsscheibenart and runner-up is O, both methods yield no entity."""
    label2id, id2label = create_synthetic_label_space()

    # Category 2 listing
    ex = [{"tokens": ["gelocht"], "category": 2, "bio_tags": ["O"]}]
    probs = np.zeros((1, 1, len(label2id)), dtype=np.float32)
    # 1st: B-Bremsscheibenart (0.60), 2nd: O (0.35), 3rd: B-Produktart (0.05)
    probs[0, 0, label2id["B-Bremsscheibenart"]] = 0.60
    probs[0, 0, label2id["O"]] = 0.35
    probs[0, 0, label2id["B-Produktart"]] = 0.05

    res_a = decode_method_a(ex, [probs[0]], id2label)
    res_b = decode_method_b(ex, [probs[0]], id2label, label2id)

    # Method A: predicts Bremsscheibenart, then filter_entities_by_category drops it -> empty
    assert res_a == [[]]
    # Method B: masks Bremsscheibenart, picks O (0.35 > 0.05) -> empty
    assert res_b == [[]]
    assert res_a == res_b


def test_forbidden_aspect_runner_up_is_allowed_aspect():
    """In Cat 2, if top choice is Bremsscheibenart and runner-up is Produktart: Method A drops, Method B recovers."""
    label2id, id2label = create_synthetic_label_space()

    # Category 2 listing
    ex = [{"tokens": ["Bremsscheibe"], "category": 2, "bio_tags": ["B-Produktart"]}]
    probs = np.zeros((1, 1, len(label2id)), dtype=np.float32)
    # 1st: B-Bremsscheibenart (0.55), 2nd: B-Produktart (0.35), 3rd: O (0.10)
    probs[0, 0, label2id["B-Bremsscheibenart"]] = 0.55
    probs[0, 0, label2id["B-Produktart"]] = 0.35
    probs[0, 0, label2id["O"]] = 0.10

    res_a = decode_method_a(ex, [probs[0]], id2label)
    res_b = decode_method_b(ex, [probs[0]], id2label, label2id)

    # Method A drops Bremsscheibenart -> empty
    assert res_a == [[]]
    # Method B masks Bremsscheibenart -> picks B-Produktart (0.35 > 0.10) -> recovered!
    assert res_b == [[("Produktart", "Bremsscheibe")]]
    assert res_a != res_b


def test_multi_token_continuity_and_boundary():
    """Verify that multi-token entities continue properly and adjacent entities don't merge."""
    label2id, id2label = create_synthetic_label_space()

    # Category 1 listing: "Bosch", "Super", "Kit"
    ex = [{"tokens": ["Bosch", "Super", "Kit"], "category": 1, "bio_tags": ["B-Hersteller", "B-Produktart", "I-Produktart"]}]
    probs = np.zeros((1, 3, len(label2id)), dtype=np.float32)
    probs[0, 0, label2id["B-Hersteller"]] = 0.9
    probs[0, 1, label2id["B-Produktart"]] = 0.8
    probs[0, 2, label2id["I-Produktart"]] = 0.85

    res_b = decode_method_b(ex, [probs[0]], id2label, label2id)
    assert res_b == [[("Hersteller", "Bosch"), ("Produktart", "Super Kit")]]


def test_invalid_aspect_counter():
    """Confirm count_invalid_predictions detects invalid aspects."""
    ex = [
        {"tokens": ["A"], "category": 2, "bio_tags": ["O"]},
        {"tokens": ["B"], "category": 1, "bio_tags": ["O"]},
    ]
    # In Cat 2, Bremsscheibenart is invalid
    preds_invalid = [
        [("Bremsscheibenart", "A")],
        [("Produktart", "B")],
    ]
    assert count_invalid_predictions(ex, preds_invalid) == 1

    # In Cat 1, Anwendung is invalid
    preds_invalid_cat1 = [
        [("Produktart", "A")],
        [("Anwendung", "B")],
    ]
    assert count_invalid_predictions(ex, preds_invalid_cat1) == 1


if __name__ == "__main__":
    test_allowed_label_indices()
    test_forbidden_mask()
    test_allowed_predictions_identical()
    test_forbidden_aspect_runner_up_is_O()
    test_forbidden_aspect_runner_up_is_allowed_aspect()
    test_multi_token_continuity_and_boundary()
    test_invalid_aspect_counter()
    print("ALL 6 SYNTHETIC DECODING TESTS PASSED!")
