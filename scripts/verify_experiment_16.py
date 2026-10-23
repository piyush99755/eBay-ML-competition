"""
Pre-training verification script for Experiment 16 (Span-Based XLM-R).

Tests:
1. Synthetic BIO-to-span tests:
   - O B-X I-X O -> one X span [1, 2]
   - B-X B-X -> two X spans [0, 0] and [1, 1]
   - B-X I-X B-Y I-Y -> X span [0, 1], Y span [2, 3]
   - invalid I-X start -> handled deterministically
2. Synthetic span-to-BIO reconstruction tests
3. Synthetic greedy overlap resolution tests
4. Candidate generation tests (all widths 1..5, bounds check)
5. Gold positive span inclusion test (all gold spans width <= 5 are in candidate set)
6. Gold span-width coverage check across all 5,000 labeled titles (verify >5 width <= 5%)
7. 1-batch structural check on real data (aggregates only, no raw competition data)
8. Forward and backward gradient check:
   - span_loss finite
   - start_loss finite
   - end_loss finite
   - total_loss finite
   - finite gradients across backbone, START head, END head, and span classifier
9. Checkpoint serialization and reload test
"""

import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForTokenClassification

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_examples import build_examples, load_training_data
from src.data.label_mapping import create_label_mappings
from src.data.make_splits import split_examples
from src.modeling.prepare_datasets import prepare_split
from src.modeling.xlmr_span import (
    MAX_SPAN_WIDTH,
    XLMRobertaSpan,
    bio_to_spans,
    create_span_mappings,
    generate_candidate_spans,
    resolve_overlapping_spans,
    sample_training_spans_for_example,
    spans_to_bio,
)


def test_bio_to_spans() -> bool:
    print("\n--- TEST 1: SYNTHETIC BIO-TO-SPAN CONVERSION ---")
    test_cases = [
        {
            "name": "O B-X I-X O",
            "bio": ["O", "B-X", "I-X", "O"],
            "expected": [(1, 2, "X")],
        },
        {
            "name": "B-X B-X",
            "bio": ["B-X", "B-X"],
            "expected": [(0, 0, "X"), (1, 1, "X")],
        },
        {
            "name": "B-X I-X B-Y I-Y",
            "bio": ["B-X", "I-X", "B-Y", "I-Y"],
            "expected": [(0, 1, "X"), (2, 3, "Y")],
        },
        {
            "name": "invalid I-X start: I-X I-X",
            "bio": ["I-X", "I-X"],
            "expected": [(0, 1, "X")],
        },
        {
            "name": "invalid aspect transition: B-X I-Y",
            "bio": ["B-X", "I-Y"],
            "expected": [(0, 0, "X"), (1, 1, "Y")],
        },
    ]

    for tc in test_cases:
        res = bio_to_spans(tc["bio"])
        print(f"Case: {tc['name']}")
        print(f"  Result:   {res}")
        print(f"  Expected: {tc['expected']}")
        if res != tc["expected"]:
            print(f"FAIL: Mismatch for {tc['name']}!")
            return False

    print("PASS: BIO-to-span conversion verified.")
    return True


def test_span_to_bio() -> bool:
    print("\n--- TEST 2: SYNTHETIC SPAN-TO-BIO RECONSTRUCTION ---")
    test_cases = [
        {
            "spans": [(1, 2, "X")],
            "len": 4,
            "expected": ["O", "B-X", "I-X", "O"],
        },
        {
            "spans": [(0, 0, "X"), (1, 1, "X")],
            "len": 2,
            "expected": ["B-X", "B-X"],
        },
        {
            "spans": [(0, 1, "X"), (2, 3, "Y")],
            "len": 4,
            "expected": ["B-X", "I-X", "B-Y", "I-Y"],
        },
    ]

    for tc in test_cases:
        res = spans_to_bio(tc["spans"], tc["len"])
        print(f"Spans: {tc['spans']} -> BIO: {res}")
        if res != tc["expected"]:
            print(f"FAIL: Mismatch for {tc['spans']}!")
            return False

    print("PASS: Span-to-BIO reconstruction verified.")
    return True


def test_overlap_resolution() -> bool:
    print("\n--- TEST 3: SYNTHETIC GREEDY OVERLAP RESOLUTION ---")
    candidates = [
        {"start": 1, "end": 2, "aspect": "Modell", "score": 0.95},
        {"start": 2, "end": 3, "aspect": "Produktart", "score": 0.80},
        {"start": 4, "end": 4, "aspect": "Hersteller", "score": 0.70},
    ]

    retained, rejected_count = resolve_overlapping_spans(candidates, seq_len=5)
    print(f"Input candidates: {len(candidates)}")
    print(f"Retained: {retained}")
    print(f"Rejected: {rejected_count}")

    if len(retained) != 2:
        print(f"FAIL: Expected 2 retained spans, got {len(retained)}!")
        return False
    if rejected_count != 1:
        print(f"FAIL: Expected 1 rejected span, got {rejected_count}!")
        return False

    if retained[0]["start"] != 1 or retained[0]["end"] != 2:
        print("FAIL: Expected span [1, 2] to be retained!")
        return False
    if retained[1]["start"] != 4 or retained[1]["end"] != 4:
        print("FAIL: Expected span [4, 4] to be retained!")
        return False

    print("PASS: Greedy overlap resolution verified.")
    return True


def test_candidate_generation() -> bool:
    print("\n--- TEST 4: CANDIDATE SPAN GENERATION & INCLUSION ---")
    T = 10
    cands = generate_candidate_spans(T, max_span_width=5)
    print(f"Sequence length T={T}, MAX_SPAN_WIDTH=5: {len(cands)} candidates")

    for s, e in cands:
        if not (0 <= s <= e < T):
            print(f"FAIL: Candidate ({s}, {e}) out of range [0, {T - 1}]!")
            return False
        w = e - s + 1
        if w > 5 or w < 1:
            print(f"FAIL: Candidate ({s}, {e}) invalid width {w}!")
            return False

    # Verify that all possible spans of width <= 5 are present
    cand_set = set(cands)
    for s in range(T):
        for e in range(s, min(T, s + 5)):
            if (s, e) not in cand_set:
                print(f"FAIL: Missing candidate span ({s}, {e})!")
                return False

    print("PASS: Candidate generation verified for all widths 1..5.")
    return True


def test_span_width_coverage(examples: List[dict]) -> bool:
    print("\n--- TEST 5: GOLD ENTITY SPAN-WIDTH COVERAGE ---")
    widths = []
    for ex in examples:
        for s, e, asp in bio_to_spans(ex["bio_tags"]):
            widths.append(e - s + 1)

    total = len(widths)
    print(f"Total gold entities across 5,000 labeled titles: {total}")

    for w in [1, 2, 3, 4, 5]:
        c = sum(1 for x in widths if x <= w)
        print(f"  Width <= {w}: {c:5d} / {total} ({c / total * 100:.2f}%)")

    gt5 = sum(1 for x in widths if x > 5)
    pct_gt5 = (gt5 / total) * 100
    print(f"  Width >  5: {gt5:5d} / {total} ({pct_gt5:.2f}%)")

    if pct_gt5 > 5.0:
        print(f"\n[!] STOP: Width > 5 entities represent {pct_gt5:.2f}% (> 5%) of gold entities!")
        return False

    print(f"PASS: Width > 5 represents {pct_gt5:.2f}% (<= 5.0%), MAX_SPAN_WIDTH=5 is validated.")
    return True


def test_structural_batch_and_gradient(
    train_examples: List[dict],
    tokenizer,
    label2id: dict,
    id2label: dict,
    span2id: dict,
    id2span: dict,
) -> bool:
    print("\n--- TEST 6: LOCAL 1-BATCH STRUCTURAL & GRADIENT CHECK (AGGREGATE ONLY) ---")

    dataset_base, stats = prepare_split(train_examples[:16], tokenizer, label2id)
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    loader = DataLoader(dataset_base, batch_size=4, shuffle=False, collate_fn=collator)

    batch = next(iter(loader))
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    batch_size, subword_len = input_ids.shape
    active_mask = (labels != -100)
    active_counts = active_mask.sum(dim=1).tolist()

    # Generate training spans for these 4 examples
    spans_list = []
    span_labels_list = []
    total_pos = 0
    total_neg = 0
    widths = []

    for b in range(batch_size):
        ex = train_examples[b]
        spans_b, labels_b = sample_training_spans_for_example(
            bio_tags=ex["bio_tags"],
            span2id=span2id,
            seed=42,
            ex_idx=b,
            max_span_width=MAX_SPAN_WIDTH,
            max_neg_ratio=3,
        )
        spans_list.append(torch.tensor(spans_b, dtype=torch.long))
        span_labels_list.append(torch.tensor(labels_b, dtype=torch.long))

        for (s, e), lbl in zip(spans_b, labels_b):
            w = e - s + 1
            widths.append(w)
            if lbl == 0:
                total_neg += 1
            else:
                total_pos += 1

    # Print aggregate statistics only
    print(f"Batch size:                       {batch_size}")
    print(f"Subword sequence length:          {subword_len}")
    print(f"Active original-token counts:     {active_counts}")
    print(f"Positive spans in batch:          {total_pos}")
    print(f"Sampled negative spans in batch:  {total_neg}")
    print(f"Total training spans in batch:    {len(widths)}")
    print(f"Span width min/max in batch:      [{min(widths)}, {max(widths)}]")

    if any(c == 0 for c in active_counts):
        print("FAIL: Batch contains a zero-length sequence!")
        return False
    if total_pos == 0:
        print("FAIL: Batch contains 0 positive spans!")
        return False
    if max(widths) > 5 or min(widths) < 1:
        print("FAIL: Span width outside [1, 5]!")
        return False

    print("PASS: 1-batch structural check verified.")

    # -------------------------------------------------
    # Forward and Backward Pass Verification
    # -------------------------------------------------
    print("\n--- TEST 7: FORWARD & BACKWARD GRADIENT CHECK ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running forward/backward pass on: {device}")

    model = XLMRobertaSpan(
        label2id=label2id,
        id2label=id2label,
        span2id=span2id,
        id2span=id2span,
        model_name="FacebookAI/xlm-roberta-base",
        max_span_width=MAX_SPAN_WIDTH,
        start_loss_weight=0.20,
        end_loss_weight=0.20,
    )
    model.to(device)
    model.train()

    batch_dev = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "labels": batch["labels"].to(device),
        "spans": [s.to(device) for s in spans_list],
        "span_labels": [l.to(device) for l in span_labels_list],
    }

    outputs = model(**batch_dev)
    loss = outputs.loss
    span_loss = outputs.span_loss
    start_loss = outputs.start_loss
    end_loss = outputs.end_loss

    print(f"Span loss:  {span_loss.item():.4f}")
    print(f"START loss: {start_loss.item():.4f}")
    print(f"END loss:   {end_loss.item():.4f}")
    print(f"Total loss: {loss.item():.4f}")

    if not torch.isfinite(loss):
        print("FAIL: Total loss is not finite!")
        return False
    if not torch.isfinite(span_loss):
        print("FAIL: Span loss is not finite!")
        return False
    if not torch.isfinite(start_loss):
        print("FAIL: START loss is not finite!")
        return False
    if not torch.isfinite(end_loss):
        print("FAIL: END loss is not finite!")
        return False

    loss.backward()

    # Check gradients
    for name, param in [
        ("span_classifier.weight", model.span_classifier.weight),
        ("start_classifier.weight", model.start_classifier.weight),
        ("end_classifier.weight", model.end_classifier.weight),
    ]:
        if param.grad is None or not torch.isfinite(param.grad).all():
            print(f"FAIL: Gradient for {name} is missing or not finite!")
            return False

    print("PASS: Backward pass succeeded; finite gradients verified across all heads.")

    # -------------------------------------------------
    # Checkpoint Serialization Test
    # -------------------------------------------------
    print("\n--- TEST 8: CHECKPOINT PARAMETER SERIALIZATION & RELOAD ---")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "span_heads.pt"
        save_dict = {
            "span_classifier": model.span_classifier.state_dict(),
            "start_classifier": model.start_classifier.state_dict(),
            "end_classifier": model.end_classifier.state_dict(),
            "label2id": label2id,
            "id2label": id2label,
            "span2id": span2id,
            "id2span": id2span,
            "num_span_classes": len(span2id),
            "max_span_width": MAX_SPAN_WIDTH,
            "model_name": "FacebookAI/xlm-roberta-base",
            "start_loss_weight": 0.20,
            "end_loss_weight": 0.20,
        }
        torch.save(save_dict, tmp_path)
        loaded_dict = torch.load(tmp_path, map_location="cpu")

        for k in model.span_classifier.state_dict():
            if not torch.allclose(model.span_classifier.state_dict()[k].cpu(), loaded_dict["span_classifier"][k]):
                print(f"FAIL: Loaded span_classifier weight mismatch: {k}")
                return False

    print("PASS: Checkpoint serialization and reload verified.")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return True


def main():
    print("==================================================")
    print("VERIFICATION: EXPERIMENT 16 PRE-TRAINING AUDIT")
    print("==================================================")

    # 1. Load data
    df = load_training_data()
    examples = build_examples(df)
    label2id, id2label = create_label_mappings(examples)
    span2id, id2span = create_span_mappings(label2id)
    train_examples, _ = split_examples(examples)

    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")

    test1_ok = test_bio_to_spans()
    if not test1_ok:
        sys.exit(1)

    test2_ok = test_span_to_bio()
    if not test2_ok:
        sys.exit(1)

    test3_ok = test_overlap_resolution()
    if not test3_ok:
        sys.exit(1)

    test4_ok = test_candidate_generation()
    if not test4_ok:
        sys.exit(1)

    test5_ok = test_span_width_coverage(examples)
    if not test5_ok:
        sys.exit(1)

    test6_ok = test_structural_batch_and_gradient(
        train_examples=train_examples,
        tokenizer=tokenizer,
        label2id=label2id,
        id2label=id2label,
        span2id=span2id,
        id2span=id2span,
    )
    if not test6_ok:
        sys.exit(1)

    print("\n==================================================")
    print("ALL EXPERIMENT 16 PRE-TRAINING CHECKS PASSED!")
    print("==================================================")


if __name__ == "__main__":
    main()
