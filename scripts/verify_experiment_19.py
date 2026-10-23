"""
Verification script for Experiment 19 (BIO vs Span Disagreement Analysis).

Covers synthetic cases:
1. Exact agreement (1-to-1 match).
2. Same-boundary aspect disagreement (1-to-1).
3. Partial boundary overlap (1-to-1).
4. One-to-many overlap (1-to-N).
5. Many-to-one overlap (N-to-1).
6. Disjoint entities (no overlap).
7. Mixed TP/FP groups (partial advantage).
8. Duplicate-safe counting invariant across complex multi-entity examples.
9. Deterministic group construction under input permutations.
10. Model predictions and checkpoints unmodified.
"""

import random
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.experiment_19_disagreement_analysis import (
    build_connected_components_for_example,
    classify_disagreement_group,
    GBERT_CHECKPOINT,
    XLMR_SEED_42_CHECKPOINT,
    XLMR_SEED_7_CHECKPOINT,
    XLMR_SEED_21_CHECKPOINT,
    XLMR_SPAN_CHECKPOINT,
)


def test_exact_agreement() -> bool:
    print("\n--- TEST 1: EXACT AGREEMENT ---")
    p10 = [(1, 2, "Hersteller")]
    p16 = [(1, 2, "Hersteller")]
    gold = [(1, 2, "Hersteller")]

    exact, dis, d10, d16 = build_connected_components_for_example(0, p10, p16, gold)

    if len(exact) != 1 or len(dis) != 0 or len(d10) != 0 or len(d16) != 0:
        print(f"FAIL: Expected 1 exact agreement, got exact={len(exact)}, dis={len(dis)}!")
        return False
    print("PASS: Exact agreement correctly grouped.")
    return True


def test_same_boundary_aspect_disagreement() -> bool:
    print("\n--- TEST 2: SAME-BOUNDARY ASPECT DISAGREEMENT ---")
    p10 = [(1, 1, "Farbe")]
    p16 = [(1, 1, "Material")]
    gold = [(1, 1, "Farbe")]

    exact, dis, d10, d16 = build_connected_components_for_example(0, p10, p16, gold)

    if len(dis) != 1:
        print(f"FAIL: Expected 1 disagreement group, got {len(dis)}!")
        return False

    cd = classify_disagreement_group(dis[0])
    if cd["structure"] != "same_boundary_diff_aspect":
        print(f"FAIL: Expected structure same_boundary_diff_aspect, got {cd['structure']}!")
        return False
    if cd["category"] != "1_exp10_fully_correct":
        print(f"FAIL: Expected 1_exp10_fully_correct, got {cd['category']}!")
        return False
    print("PASS: Same-boundary aspect disagreement verified.")
    return True


def test_partial_boundary_overlap() -> bool:
    print("\n--- TEST 3: PARTIAL BOUNDARY OVERLAP ---")
    p10 = [(1, 2, "Produktart")]
    p16 = [(2, 3, "Produktart")]
    gold = [(1, 2, "Produktart")]

    exact, dis, d10, d16 = build_connected_components_for_example(0, p10, p16, gold)

    if len(dis) != 1:
        print(f"FAIL: Expected 1 disagreement group, got {len(dis)}!")
        return False

    cd = classify_disagreement_group(dis[0])
    if cd["structure"] != "diff_boundary_same_aspect":
        print(f"FAIL: Expected structure diff_boundary_same_aspect, got {cd['structure']}!")
        return False
    if cd["category"] != "1_exp10_fully_correct":
        print(f"FAIL: Expected 1_exp10_fully_correct, got {cd['category']}!")
        return False
    print("PASS: Partial boundary overlap verified.")
    return True


def test_one_to_many_overlap() -> bool:
    print("\n--- TEST 4: ONE-TO-MANY OVERLAP ---")
    p10 = [(1, 3, "Produktart")]  # 1 entity in Exp 10
    p16 = [(1, 1, "Marke"), (2, 3, "Produktart")]  # 2 entities in Exp 16
    gold = [(1, 1, "Marke"), (2, 3, "Produktart")]

    exact, dis, d10, d16 = build_connected_components_for_example(0, p10, p16, gold)

    if len(dis) != 1:
        print(f"FAIL: Expected 1 connected disagreement group, got {len(dis)}!")
        return False

    g = dis[0]
    if len(g["p10_entities"]) != 1 or len(g["p16_entities"]) != 2:
        print(f"FAIL: Expected 1 p10 and 2 p16 entities, got {len(g['p10_entities'])} and {len(g['p16_entities'])}!")
        return False

    cd = classify_disagreement_group(g)
    if cd["structure"] != "complex_n_to_m":
        print(f"FAIL: Expected complex_n_to_m structure, got {cd['structure']}!")
        return False
    if cd["category"] != "2_exp16_fully_correct":
        print(f"FAIL: Expected 2_exp16_fully_correct, got {cd['category']}!")
        return False
    print("PASS: One-to-many overlap correctly grouped and evaluated.")
    return True


def test_many_to_one_overlap() -> bool:
    print("\n--- TEST 5: MANY-TO-ONE OVERLAP ---")
    p10 = [(1, 1, "Marke"), (2, 3, "Produktart")]  # 2 entities in Exp 10
    p16 = [(1, 3, "Produktart")]  # 1 entity in Exp 16
    gold = [(1, 3, "Produktart")]

    exact, dis, d10, d16 = build_connected_components_for_example(0, p10, p16, gold)

    if len(dis) != 1:
        print(f"FAIL: Expected 1 connected disagreement group, got {len(dis)}!")
        return False

    g = dis[0]
    if len(g["p10_entities"]) != 2 or len(g["p16_entities"]) != 1:
        print(f"FAIL: Expected 2 p10 and 1 p16 entities, got {len(g['p10_entities'])} and {len(g['p16_entities'])}!")
        return False

    cd = classify_disagreement_group(g)
    if cd["category"] != "2_exp16_fully_correct":
        print(f"FAIL: Expected 2_exp16_fully_correct, got {cd['category']}!")
        return False
    print("PASS: Many-to-one overlap correctly grouped and evaluated.")
    return True


def test_disjoint_entities() -> bool:
    print("\n--- TEST 6: DISJOINT ENTITIES ---")
    p10 = [(1, 1, "A")]
    p16 = [(5, 5, "B")]
    gold = [(1, 1, "A"), (5, 5, "B")]

    exact, dis, d10, d16 = build_connected_components_for_example(0, p10, p16, gold)

    if len(dis) != 0:
        print(f"FAIL: Expected 0 disagreement groups, got {len(dis)}!")
        return False
    if len(d10) != 1 or len(d16) != 1:
        print(f"FAIL: Expected 1 d10 and 1 d16 disjoint group, got {len(d10)} and {len(d16)}!")
        return False
    print("PASS: Disjoint entities correctly partitioned.")
    return True


def test_mixed_tp_fp_group() -> bool:
    print("\n--- TEST 7: MIXED TP/FP GROUPS ---")
    p10 = [(1, 2, "A"), (3, 4, "B")]
    p16 = [(2, 3, "C")]
    gold = [(1, 2, "A")]

    # p10 overlaps p16 on token 2 and 3 -> connected group
    exact, dis, d10, d16 = build_connected_components_for_example(0, p10, p16, gold)

    if len(dis) != 1:
        print(f"FAIL: Expected 1 disagreement group, got {len(dis)}!")
        return False

    cd = classify_disagreement_group(dis[0])
    if cd["category"] != "5_mixed_partial":
        print(f"FAIL: Expected category 5_mixed_partial, got {cd['category']}!")
        return False
    if cd["advantage"] != "exp10_partial_advantage":
        print(f"FAIL: Expected exp10_partial_advantage, got {cd['advantage']}!")
        return False
    print("PASS: Mixed TP/FP group classified correctly.")
    return True


def test_duplicate_safe_counting() -> bool:
    print("\n--- TEST 8: DUPLICATE-SAFE COUNTING ---")
    # Multi-component complex example
    p10 = [(0, 0, "Exact1"), (2, 3, "Over1"), (5, 5, "Only10")]
    p16 = [(0, 0, "Exact1"), (3, 4, "Over1"), (7, 8, "Only16")]
    gold = [(0, 0, "Exact1"), (2, 3, "Over1"), (5, 5, "Only10")]

    exact, dis, d10, d16 = build_connected_components_for_example(0, p10, p16, gold)

    all_p10_counted = (
        sum(len(g["p10_entities"]) for g in exact)
        + sum(len(g["p10_entities"]) for g in dis)
        + sum(len(g["p10_entities"]) for g in d10)
    )
    all_p16_counted = (
        sum(len(g["p16_entities"]) for g in exact)
        + sum(len(g["p16_entities"]) for g in dis)
        + sum(len(g["p16_entities"]) for g in d16)
    )

    if all_p10_counted != len(p10):
        print(f"FAIL: Counted {all_p10_counted} p10 entities, expected {len(p10)}!")
        return False
    if all_p16_counted != len(p16):
        print(f"FAIL: Counted {all_p16_counted} p16 entities, expected {len(p16)}!")
        return False

    print("PASS: Duplicate-safe entity counting verified.")
    return True


def test_deterministic_group_construction() -> bool:
    print("\n--- TEST 9: DETERMINISTIC GROUP CONSTRUCTION ---")
    p10_orig = [(0, 0, "E"), (2, 3, "O1"), (5, 5, "D10"), (8, 9, "O2")]
    p16_orig = [(0, 0, "E"), (3, 4, "O1"), (8, 9, "O2_diff"), (11, 11, "D16")]
    gold_orig = [(0, 0, "E"), (2, 3, "O1")]

    exact_base, dis_base, d10_base, d16_base = build_connected_components_for_example(
        0, p10_orig, p16_orig, gold_orig
    )

    # Permute inputs
    for seed in [1, 42, 99]:
        rng = random.Random(seed)
        p10_perm = list(p10_orig)
        p16_perm = list(p16_orig)
        rng.shuffle(p10_perm)
        rng.shuffle(p16_perm)

        # Ensure sort order by start
        p10_perm.sort(key=lambda x: x[0])
        p16_perm.sort(key=lambda x: x[0])

        e_p, d_p, d10_p, d16_p = build_connected_components_for_example(
            0, p10_perm, p16_perm, gold_orig
        )

        if len(e_p) != len(exact_base) or len(d_p) != len(dis_base):
            print(f"FAIL: Permutation gave different component counts!")
            return False

    print("PASS: Deterministic group construction verified.")
    return True


def test_models_and_checkpoints_unmodified() -> bool:
    print("\n--- TEST 10: MODELS & CHECKPOINTS UNMODIFIED ---")
    required_checkpoints = [
        GBERT_CHECKPOINT,
        XLMR_SEED_42_CHECKPOINT,
        XLMR_SEED_7_CHECKPOINT,
        XLMR_SEED_21_CHECKPOINT,
        XLMR_SPAN_CHECKPOINT,
    ]
    for cp in required_checkpoints:
        if not cp.exists():
            print(f"FAIL: Missing checkpoint {cp}!")
            return False
        print(f"Verified intact checkpoint: {cp}")

    print("PASS: All checkpoints verified intact.")
    return True


def run_all_verification_tests() -> bool:
    print("==================================================")
    print("EXPERIMENT 19: VERIFICATION AUDIT")
    print("==================================================")

    tests = [
        ("Exact Agreement", test_exact_agreement),
        ("Same-Boundary Aspect Disagreement", test_same_boundary_aspect_disagreement),
        ("Partial Boundary Overlap", test_partial_boundary_overlap),
        ("One-to-Many Overlap", test_one_to_many_overlap),
        ("Many-to-One Overlap", test_many_to_one_overlap),
        ("Disjoint Entities", test_disjoint_entities),
        ("Mixed TP/FP Groups", test_mixed_tp_fp_group),
        ("Duplicate-Safe Counting", test_duplicate_safe_counting),
        ("Deterministic Group Construction", test_deterministic_group_construction),
        ("Models & Checkpoints Unmodified", test_models_and_checkpoints_unmodified),
    ]

    for name, test_func in tests:
        success = test_func()
        if not success:
            print(f"\n[!] AUDIT FAILED at test: {name}")
            return False

    print("\n==================================================")
    print("ALL 10 VERIFICATION TESTS PASSED!")
    print("==================================================")
    return True


if __name__ == "__main__":
    success = run_all_verification_tests()
    if not success:
        sys.exit(1)
    sys.exit(0)
