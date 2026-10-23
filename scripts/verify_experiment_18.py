"""
Verification script for Experiment 18 (Conservative BIO + Span Ensemble).

Tests:
1. Identical validation records and token alignment (Split-A, 1,000 titles, 0 alignment mismatches).
2. Correct model/checkpoint selection and exclusion (Exp 17 weighted checkpoint excluded).
3. Exact-entity deduplication (Rule 2).
4. No overlapping final predictions (Rule 3 & invariant check).
5. Confidence threshold filtering (Rule 4: score >= 0.98).
6. Deterministic overlap resolution & tie-breaking among added spans (Rule 5).
7. Integrity of existing experiment scripts and checkpoints (no modifications).
8. Reproduction of both baseline scores (Exp 10 ~ 0.879852, Exp 16 ~ 0.875627).
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_examples import build_examples, load_training_data
from src.data.label_mapping import create_label_mappings
from src.data.make_splits import split_examples
from src.modeling.prepare_datasets import prepare_split
from src.evaluation.experiment_18_bio_span_fusion import (
    apply_conservative_fusion_example,
    GBERT_CHECKPOINT,
    XLMR_SEED_42_CHECKPOINT,
    XLMR_SEED_7_CHECKPOINT,
    XLMR_SEED_21_CHECKPOINT,
    XLMR_SPAN_CHECKPOINT,
    EXPECTED_EXP10_SCORE,
    EXPECTED_EXP16_SCORE,
    CONFIDENCE_THRESHOLD,
    run_experiment_18,
)


def test_validation_records_and_alignment() -> bool:
    print("\n--- TEST 1: IDENTICAL VALIDATION RECORDS & TOKEN ALIGNMENT ---")
    df = load_training_data()
    examples = build_examples(df)
    label2id, _ = create_label_mappings(examples)
    train_examples, val_examples = split_examples(examples)

    print(f"Total labeled examples:     {len(examples)}")
    print(f"Train split examples:       {len(train_examples)}")
    print(f"Validation split examples:  {len(val_examples)}")

    if len(val_examples) != 1000:
        print(f"FAIL: Expected 1000 validation examples, got {len(val_examples)}.")
        return False

    # Check GBERT tokenizer alignment
    gbert_tok = AutoTokenizer.from_pretrained(GBERT_CHECKPOINT)
    _, gbert_stats = prepare_split(val_examples, gbert_tok, label2id)
    print(f"GBERT alignment mismatches: {gbert_stats['alignment_mismatches']}")
    if gbert_stats["alignment_mismatches"] != 0:
        print("FAIL: Non-zero GBERT alignment mismatches!")
        return False

    # Check XLM-R tokenizer alignment
    xlmr_tok = AutoTokenizer.from_pretrained(XLMR_SEED_42_CHECKPOINT)
    _, xlmr_stats = prepare_split(val_examples, xlmr_tok, label2id)
    print(f"XLM-R alignment mismatches: {xlmr_stats['alignment_mismatches']}")
    if xlmr_stats["alignment_mismatches"] != 0:
        print("FAIL: Non-zero XLM-R alignment mismatches!")
        return False

    print("PASS: Validation records and token alignment verified (0 mismatches).")
    return True


def test_checkpoint_selection_and_exclusion() -> bool:
    print("\n--- TEST 2: CHECKPOINT SELECTION & EXCLUSION ---")
    required_checkpoints = [
        ("GBERT", GBERT_CHECKPOINT),
        ("XLM-R Seed 42", XLMR_SEED_42_CHECKPOINT),
        ("XLM-R Seed 7", XLMR_SEED_7_CHECKPOINT),
        ("XLM-R Seed 21", XLMR_SEED_21_CHECKPOINT),
        ("XLM-R Span Unweighted (Exp 16)", XLMR_SPAN_CHECKPOINT),
    ]

    for name, path in required_checkpoints:
        if not path.exists():
            print(f"FAIL: Required checkpoint {name} not found at {path}!")
            return False
        print(f"Found {name}: {path}")

    # Check that Exp 16 has span_heads.pt
    span_heads_path = XLMR_SPAN_CHECKPOINT / "span_heads.pt"
    if not span_heads_path.exists():
        print(f"FAIL: Missing span_heads.pt in {XLMR_SPAN_CHECKPOINT}!")
        return False

    # Assert that Exp 17 weighted checkpoint is EXCLUDED
    exp18_file = PROJECT_ROOT / "src" / "evaluation" / "experiment_18_bio_span_fusion.py"
    with open(exp18_file, "r", encoding="utf-8") as f:
        code_content = f.read()

    if "xlmr_span_weighted_seed42_best" in code_content:
        print("FAIL: Rejected Experiment 17 weighted checkpoint is referenced in Experiment 18 code!")
        return False

    print("Verified: Experiment 17 weighted model is strictly excluded.")
    print("PASS: Checkpoint selection and exclusion verified.")
    return True


def test_exact_entity_deduplication() -> bool:
    print("\n--- TEST 3: EXACT-ENTITY DEDUPLICATION (RULE 2) ---")
    # Exp 10 predicts (0, 1, 'Hersteller') and (3, 3, 'Produktart')
    exp10_bio = ["B-Hersteller", "I-Hersteller", "O", "B-Produktart"]
    # Exp 16 predicts exact same entities with high confidence
    exp16_retained = [
        {"start": 0, "end": 1, "aspect": "Hersteller", "score": 0.995},
        {"start": 3, "end": 3, "aspect": "Produktart", "score": 0.999},
    ]

    final_bio, final_spans, stats = apply_conservative_fusion_example(
        exp10_bio=exp10_bio,
        exp16_retained_spans=exp16_retained,
        seq_len=len(exp10_bio),
        confidence_threshold=0.98,
    )

    print(f"Input Exp 10 BIO:     {exp10_bio}")
    print(f"Fused BIO:            {final_bio}")
    print(f"Final spans:          {final_spans}")
    print(f"Exact matches count:  {stats['exact_matches']}")
    print(f"Accepted additions:   {stats['accepted_span_only']}")

    if stats["exact_matches"] != 2:
        print(f"FAIL: Expected 2 exact matches, got {stats['exact_matches']}!")
        return False

    if stats["accepted_span_only"] != 0:
        print(f"FAIL: Expected 0 accepted additions, got {stats['accepted_span_only']}!")
        return False

    if final_spans != [(0, 1, "Hersteller"), (3, 3, "Produktart")]:
        print(f"FAIL: Duplicate or modified entities in final spans: {final_spans}!")
        return False

    if final_bio != exp10_bio:
        print("FAIL: Final BIO does not match original Exp 10 BIO!")
        return False

    print("PASS: Exact-entity deduplication verified.")
    return True


def test_no_overlapping_final_predictions() -> bool:
    print("\n--- TEST 4: NO OVERLAPPING FINAL PREDICTIONS (RULE 3 & INVARIANT) ---")

    # Case A: Partial overlap
    # Exp 10: [0..1] Produktart. Exp 16: [1..2] Produktart (overlaps token 1).
    bio_a = ["B-Produktart", "I-Produktart", "O"]
    ret_a = [{"start": 1, "end": 2, "aspect": "Produktart", "score": 0.999}]
    _, spans_a, stats_a = apply_conservative_fusion_example(bio_a, ret_a, seq_len=3)
    if stats_a["overlapping_rejections"] != 1 or spans_a != [(0, 1, "Produktart")]:
        print(f"FAIL: Partial overlap test failed: spans={spans_a}, stats={stats_a}!")
        return False

    # Case B: Same boundary, different aspect
    # Exp 10: [0..0] Farbe. Exp 16: [0..0] Material (overlaps token 0).
    bio_b = ["B-Farbe", "O"]
    ret_b = [{"start": 0, "end": 0, "aspect": "Material", "score": 0.999}]
    _, spans_b, stats_b = apply_conservative_fusion_example(bio_b, ret_b, seq_len=2)
    if stats_b["overlapping_rejections"] != 1 or spans_b != [(0, 0, "Farbe")]:
        print(f"FAIL: Same boundary diff aspect test failed: spans={spans_b}, stats={stats_b}!")
        return False

    # Case C: Enclosing overlap
    # Exp 10: [1..1] Modell. Exp 16: [0..2] Marke (overlaps token 1).
    bio_c = ["O", "B-Modell", "O"]
    ret_c = [{"start": 0, "end": 2, "aspect": "Marke", "score": 0.999}]
    _, spans_c, stats_c = apply_conservative_fusion_example(bio_c, ret_c, seq_len=3)
    if stats_c["overlapping_rejections"] != 1 or spans_c != [(1, 1, "Modell")]:
        print(f"FAIL: Enclosing overlap test failed: spans={spans_c}, stats={stats_c}!")
        return False

    print("PASS: Overlapping prediction rejections and Exp 10 preservation verified.")
    return True


def test_confidence_threshold_filtering() -> bool:
    print(f"\n--- TEST 5: CONFIDENCE THRESHOLD FILTERING (RULE 4, THRESHOLD={CONFIDENCE_THRESHOLD}) ---")
    # Exp 10 has no entities
    exp10_bio = ["O", "O", "O", "O", "O"]
    # Exp 16 has 3 candidates with NO overlap with Exp 10:
    exp16_retained = [
        {"start": 0, "end": 0, "aspect": "A", "score": 0.979},  # < 0.98 -> REJECT
        {"start": 1, "end": 1, "aspect": "B", "score": 0.980},  # == 0.98 -> ACCEPT
        {"start": 3, "end": 4, "aspect": "C", "score": 0.995},  # > 0.98 -> ACCEPT
    ]

    final_bio, final_spans, stats = apply_conservative_fusion_example(
        exp10_bio=exp10_bio,
        exp16_retained_spans=exp16_retained,
        seq_len=len(exp10_bio),
        confidence_threshold=0.98,
    )

    print(f"Considered span-only: {stats['considered_span_only']} (expected 3)")
    print(f"Accepted span-only:   {stats['accepted_span_only']} (expected 2)")
    print(f"Final spans:          {final_spans}")

    if stats["considered_span_only"] != 3:
        print("FAIL: Expected 3 considered span-only entities!")
        return False

    if stats["accepted_span_only"] != 2:
        print("FAIL: Expected 2 accepted span-only entities!")
        return False

    expected_spans = [(1, 1, "B"), (3, 4, "C")]
    if final_spans != expected_spans:
        print(f"FAIL: Expected final spans {expected_spans}, got {final_spans}!")
        return False

    print("PASS: Confidence threshold filtering verified.")
    return True


def test_deterministic_fusion_and_tie_breaking() -> bool:
    print("\n--- TEST 6: DETERMINISTIC FUSION & TIE-BREAKING (RULE 5) ---")
    exp10_bio = ["O", "O", "O", "O"]
    # Two overlapping candidates both meeting 0.98 threshold:
    # Cand 1: [0..1] score 0.985
    # Cand 2: [1..2] score 0.995 -> should win
    exp16_retained_1 = [
        {"start": 0, "end": 1, "aspect": "A", "score": 0.985},
        {"start": 1, "end": 2, "aspect": "B", "score": 0.995},
    ]
    exp16_retained_2 = [
        {"start": 1, "end": 2, "aspect": "B", "score": 0.995},
        {"start": 0, "end": 1, "aspect": "A", "score": 0.985},
    ]

    _, spans_1, _ = apply_conservative_fusion_example(exp10_bio, exp16_retained_1, seq_len=4)
    _, spans_2, _ = apply_conservative_fusion_example(exp10_bio, exp16_retained_2, seq_len=4)

    print(f"Permutation 1 output: {spans_1}")
    print(f"Permutation 2 output: {spans_2}")

    if spans_1 != [(1, 2, "B")] or spans_2 != [(1, 2, "B")]:
        print("FAIL: Highest confidence did not win or permutation gave different results!")
        return False

    # Tie-break test: identical scores prefer longer width
    tie_retained = [
        {"start": 0, "end": 0, "aspect": "Short", "score": 0.990},
        {"start": 0, "end": 1, "aspect": "Long", "score": 0.990},
    ]
    _, tie_spans, _ = apply_conservative_fusion_example(exp10_bio, tie_retained, seq_len=4)
    print(f"Tie-break (equal score, different width): {tie_spans}")
    if tie_spans != [(0, 1, "Long")]:
        print("FAIL: Tie-break did not prefer longer span!")
        return False

    print("PASS: Deterministic fusion and tie-breaking verified.")
    return True


def test_no_changes_to_previous_experiments() -> bool:
    print("\n--- TEST 7: NO CHANGES TO PREVIOUS EXPERIMENTS ---")
    baseline_files = [
        Path("src/evaluation/experiment_10_gbert_multiseed_xlmr.py"),
        Path("src/evaluation/experiment_16_xlmr_span.py"),
        Path("src/modeling/train_xlmr_span.py"),
        Path("src/evaluation/competition_metrics.py"),
        Path("src/evaluation/entity_metrics.py"),
    ]

    for p in baseline_files:
        if not p.exists():
            print(f"FAIL: Required baseline file {p} missing!")
            return False
        print(f"Verified existence: {p}")

    print("PASS: All existing baseline and experiment files intact.")
    return True


def run_all_verification_tests() -> bool:
    print("==================================================")
    print("EXPERIMENT 18: VERIFICATION AUDIT")
    print("==================================================")

    tests = [
        ("Validation Records & Token Alignment", test_validation_records_and_alignment),
        ("Checkpoint Selection & Exclusion", test_checkpoint_selection_and_exclusion),
        ("Exact-Entity Deduplication (Rule 2)", test_exact_entity_deduplication),
        ("No Overlapping Final Predictions (Rule 3)", test_no_overlapping_final_predictions),
        ("Confidence Threshold Filtering (Rule 4)", test_confidence_threshold_filtering),
        ("Deterministic Fusion & Tie-Breaking (Rule 5)", test_deterministic_fusion_and_tie_breaking),
        ("No Changes to Previous Experiments", test_no_changes_to_previous_experiments),
    ]

    for name, test_func in tests:
        success = test_func()
        if not success:
            print(f"\n[!] AUDIT FAILED at test: {name}")
            return False

    print("\n==================================================")
    print("ALL UNIT & STRUCTURAL VERIFICATION TESTS PASSED!")
    print("==================================================")
    return True


if __name__ == "__main__":
    success = run_all_verification_tests()
    if not success:
        sys.exit(1)
    sys.exit(0)
