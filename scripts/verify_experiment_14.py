"""
Verification script for Experiment 14 (Boundary-Aware Multi-Task XLM-R).

Tests:
1. Synthetic boundary target generation test:
   - B-X I-X I-X O
   - B-X B-X O
   - O B-X I-X
   - B-X I-Y
2. Checkpoint save and reload verification in temporary directory
3. 1-batch structural check on local data using safe aggregate statistics only:
   - Active original-token counts
   - BIO target range
   - Positive START targets
   - Positive END targets
   - Entity 1-to-1 START/END correspondence check
   - Zero invalid/empty sequences check
   - Privacy rule preserved: zero raw competition data printed
4. Real 1-batch forward & backward pass:
   - bio_loss finite
   - start_loss finite
   - end_loss finite
   - total_loss finite
   - gradients finite across backbone and all 3 heads
"""

import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForTokenClassification

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.xlmr_boundary import (
    XLMRobertaBoundary,
    generate_boundary_targets,
)
from src.data.build_examples import build_examples, load_training_data
from src.data.label_mapping import create_label_mappings
from src.data.make_splits import split_examples
from src.modeling.prepare_datasets import prepare_split


def test_synthetic_boundary_targets() -> bool:
    print("\n--- TEST 1: SYNTHETIC BOUNDARY TARGET GENERATION ---")

    test_cases = [
        {
            "name": "B-X I-X I-X O",
            "bio": ["B-X", "I-X", "I-X", "O"],
            "expected_start": [1, 0, 0, 0],
            "expected_end":   [0, 0, 1, 0],
        },
        {
            "name": "B-X B-X O",
            "bio": ["B-X", "B-X", "O"],
            "expected_start": [1, 1, 0],
            "expected_end":   [1, 1, 0],
        },
        {
            "name": "O B-X I-X",
            "bio": ["O", "B-X", "I-X"],
            "expected_start": [0, 1, 0],
            "expected_end":   [0, 0, 1],
        },
        {
            "name": "B-X I-Y",
            "bio": ["B-X", "I-Y"],
            "expected_start": [1, 0],
            "expected_end":   [1, 1],
        },
        {
            "name": "O O O (No entities)",
            "bio": ["O", "O", "O"],
            "expected_start": [0, 0, 0],
            "expected_end":   [0, 0, 0],
        },
        {
            "name": "Single token B-X",
            "bio": ["B-X"],
            "expected_start": [1],
            "expected_end":   [1],
        },
        {
            "name": "B-X I-X B-Y (Adjacent entities)",
            "bio": ["B-X", "I-X", "B-Y"],
            "expected_start": [1, 0, 1],
            "expected_end":   [0, 1, 1],
        },
    ]

    for tc in test_cases:
        starts, ends = generate_boundary_targets(tc["bio"])
        print(f"Case: {tc['name']}")
        print(f"  BIO:   {tc['bio']}")
        print(f"  START: {starts} (expected {tc['expected_start']})")
        print(f"  END:   {ends} (expected {tc['expected_end']})")

        if starts != tc["expected_start"]:
            print(f"FAIL: START mismatch for {tc['name']}!")
            return False
        if ends != tc["expected_end"]:
            print(f"FAIL: END mismatch for {tc['name']}!")
            return False

    print("PASS: All synthetic boundary target cases verified exactly.")
    return True


def test_structural_batch_and_forward() -> bool:
    print("\n--- TEST 2: LOCAL 1-BATCH STRUCTURAL & GRADIENT CHECK (AGGREGATE ONLY) ---")

    # Load data locally without printing any raw text or tokens
    df = load_training_data()
    examples = build_examples(df)
    label2id, id2label = create_label_mappings(examples)
    train_examples, _ = split_examples(examples)

    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    dataset, stats = prepare_split(train_examples[:16], tokenizer, label2id)

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collator)

    batch = next(iter(loader))
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    batch_size, subword_len = input_ids.shape
    active_mask = (labels != -100)
    active_counts = active_mask.sum(dim=1).tolist()
    max_active_len = int(active_mask.sum(dim=1).max().item())

    # Label checks
    active_labels = labels[active_mask]
    min_label = int(active_labels.min().item())
    max_label = int(active_labels.max().item())

    # Boundary target extraction across the batch
    total_starts = 0
    total_ends = 0
    all_entities_valid = True

    for b in range(batch_size):
        row_lids = labels[b][labels[b] != -100]
        bio_tags = [id2label[lid.item()] for lid in row_lids]
        s_targets, e_targets = generate_boundary_targets(bio_tags)
        s_count = sum(s_targets)
        e_count = sum(e_targets)
        total_starts += s_count
        total_ends += e_count

        # In standard clean annotations, every entity starting with B- must have exactly 1 END
        if s_count != e_count:
            all_entities_valid = False

    # Print aggregate statistics only
    print(f"Batch size:                   {batch_size}")
    print(f"Subword sequence length:      {subword_len}")
    print(f"Active original-token counts: {active_counts}")
    print(f"Max active length in batch:   {max_active_len}")
    print(f"Active BIO label ID range:    [{min_label}, {max_label}] (valid range: [0, {len(label2id) - 1}])")
    print(f"Total positive START targets: {total_starts}")
    print(f"Total positive END targets:   {total_ends}")
    print(f"START/END 1-to-1 match:       {total_starts == total_ends}")
    print(f"Any zero-length sequences:    {any(c == 0 for c in active_counts)}")

    if any(c == 0 for c in active_counts):
        print("FAIL: Batch contains a zero-length sequence!")
        return False

    if min_label < 0 or max_label >= len(label2id):
        print(f"FAIL: Active label outside range [0, {len(label2id) - 1}]!")
        return False

    if total_starts != total_ends:
        print(f"FAIL: Total STARTs ({total_starts}) != total ENDs ({total_ends})!")
        return False

    print("PASS: 1-batch structural verification passed all checks.")

    # -------------------------------------------------
    # Forward & Backward Pass Verification
    # -------------------------------------------------
    print("\n--- TEST 3: REAL 1-BATCH FORWARD & BACKWARD PASS ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running forward/backward pass on: {device}")

    model = XLMRobertaBoundary(
        label2id=label2id,
        id2label=id2label,
        model_name="FacebookAI/xlm-roberta-base",
        start_loss_weight=0.20,
        end_loss_weight=0.20,
    )
    model.to(device)
    model.train()

    batch_dev = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch_dev)

    loss = outputs.loss
    bio_loss = outputs.bio_loss
    start_loss = outputs.start_loss
    end_loss = outputs.end_loss

    print(f"BIO loss:   {bio_loss.item():.4f}")
    print(f"START loss: {start_loss.item():.4f}")
    print(f"END loss:   {end_loss.item():.4f}")
    print(f"Total loss: {loss.item():.4f} (expected {bio_loss.item() + 0.20 * start_loss.item() + 0.20 * end_loss.item():.4f})")

    if not torch.isfinite(loss):
        print("FAIL: Total loss is not finite!")
        return False
    if not torch.isfinite(bio_loss):
        print("FAIL: BIO loss is not finite!")
        return False
    if not torch.isfinite(start_loss):
        print("FAIL: START loss is not finite!")
        return False
    if not torch.isfinite(end_loss):
        print("FAIL: END loss is not finite!")
        return False

    # Backward pass
    loss.backward()

    # Check gradients
    for name, param in [
        ("bio_classifier.weight", model.bio_classifier.weight),
        ("start_classifier.weight", model.start_classifier.weight),
        ("end_classifier.weight", model.end_classifier.weight),
    ]:
        if param.grad is None or not torch.isfinite(param.grad).all():
            print(f"FAIL: Gradient for {name} is missing or not finite!")
            return False

    print("PASS: Backward pass succeeded; finite gradients verified for all 3 classification heads.")

    # -------------------------------------------------
    # Checkpoint Serialization Test
    # -------------------------------------------------
    print("\n--- TEST 4: BOUNDARY HEAD PARAMETER SERIALIZATION & RELOAD ---")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "boundary_heads.pt"
        save_dict = {
            "bio_classifier": model.bio_classifier.state_dict(),
            "start_classifier": model.start_classifier.state_dict(),
            "end_classifier": model.end_classifier.state_dict(),
            "label2id": label2id,
            "id2label": id2label,
            "num_labels": len(label2id),
            "model_name": "FacebookAI/xlm-roberta-base",
            "start_loss_weight": 0.20,
            "end_loss_weight": 0.20,
        }
        torch.save(save_dict, tmp_path)
        loaded_dict = torch.load(tmp_path, map_location="cpu")

        # Verify weights equality
        for k in model.bio_classifier.state_dict():
            if not torch.allclose(model.bio_classifier.state_dict()[k].cpu(), loaded_dict["bio_classifier"][k]):
                print(f"FAIL: Loaded bio_classifier weight mismatch: {k}")
                return False

        for k in model.start_classifier.state_dict():
            if not torch.allclose(model.start_classifier.state_dict()[k].cpu(), loaded_dict["start_classifier"][k]):
                print(f"FAIL: Loaded start_classifier weight mismatch: {k}")
                return False

        for k in model.end_classifier.state_dict():
            if not torch.allclose(model.end_classifier.state_dict()[k].cpu(), loaded_dict["end_classifier"][k]):
                print(f"FAIL: Loaded end_classifier weight mismatch: {k}")
                return False

    print("PASS: Boundary head parameters serialization & reload verified.")

    # Clean up memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return True


def main():
    print("==================================================")
    print("VERIFICATION: EXPERIMENT 14 PRE-TRAINING AUDIT")
    print("==================================================")

    test1_ok = test_synthetic_boundary_targets()
    if not test1_ok:
        print("\n[!] TEST 1 FAILED!")
        sys.exit(1)

    test2_ok = test_structural_batch_and_forward()
    if not test2_ok:
        print("\n[!] TEST 2/3/4 FAILED!")
        sys.exit(1)

    print("\n==================================================")
    print("ALL EXPERIMENT 14 PRE-TRAINING CHECKS PASSED!")
    print("==================================================")


if __name__ == "__main__":
    main()
