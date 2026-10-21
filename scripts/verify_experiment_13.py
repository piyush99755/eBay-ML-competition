"""
Verification script for Experiment 13 (XLM-RoBERTa + Linear-Chain CRF).

Tests:
1. Synthetic CRF sanity test (no competition data used):
   - LinearChainCRF forward pass, loss calculation, backward gradient flow, and Viterbi decoding
   - Checkpoint save & reload verification
2. 1-batch structural verification on local data using safe aggregate statistics only:
   - Subword sequence length
   - Active original-token counts
   - Gathered CRF sequence lengths
   - Label ID range check (0..47)
   - Zero-length sequence check
   - No raw titles, tokens, or competition data printed
"""

import sys
from pathlib import Path
import tempfile
import torch
import torch.nn as nn

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.xlmr_crf import LinearChainCRF, XLMRobertaCRF
from src.data.build_examples import build_examples, load_training_data
from src.data.label_mapping import create_label_mappings
from src.data.make_splits import split_examples
from src.modeling.prepare_datasets import prepare_split
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from torch.utils.data import DataLoader


def run_synthetic_crf_test() -> bool:
    print("\n--- TEST 1: SYNTHETIC CRF SANITY TEST ---")
    batch_size = 3
    seq_len = 6
    num_labels = 48  # Exact competition BIO label space size

    torch.manual_seed(42)
    crf = LinearChainCRF(num_labels=num_labels)

    # Random emissions [B, S, N]
    emissions = torch.randn(batch_size, seq_len, num_labels, requires_grad=True)

    # Synthetic tags [B, S] within valid label range [0, 47]
    tags = torch.randint(0, num_labels, (batch_size, seq_len), dtype=torch.long)

    # Synthetic mask with varying lengths: 6, 4, 2
    mask = torch.tensor([
        [True, True, True, True, True, True],
        [True, True, True, True, False, False],
        [True, True, False, False, False, False],
    ], dtype=torch.bool)

    expected_lengths = [6, 4, 2]

    # 1. Forward NLL Loss
    loss = crf.loss(emissions, tags, mask)
    print(f"Emissions shape:       {list(emissions.shape)}")
    print(f"Tags shape:            {list(tags.shape)}")
    print(f"Mask shape:            {list(mask.shape)}")
    print(f"CRF loss value:        {loss.item():.4f}")

    if not torch.isfinite(loss):
        print("FAIL: Loss is not finite!")
        return False
    print("PASS: CRF loss is finite.")

    # 2. Backward Pass
    loss.backward()
    if emissions.grad is None or not torch.isfinite(emissions.grad).all():
        print("FAIL: Emissions gradients are missing or not finite!")
        return False
    if crf.transitions.grad is None or not torch.isfinite(crf.transitions.grad).all():
        print("FAIL: CRF transitions gradient is missing or not finite!")
        return False
    print("PASS: Backward pass succeeded, finite gradients verified for emissions and transitions.")

    # 3. Viterbi Decoding
    decoded = crf.decode(emissions.detach(), mask)
    print(f"Decoded batch size:    {len(decoded)}")
    for b, path in enumerate(decoded):
        print(f"  Sequence {b} decoded length: {len(path)} (expected {expected_lengths[b]})")
        if len(path) != expected_lengths[b]:
            print(f"FAIL: Decoded length {len(path)} does not match expected length {expected_lengths[b]}!")
            return False
        if not all(0 <= tag < num_labels for tag in path):
            print("FAIL: Decoded tag outside valid label range!")
            return False

    print("PASS: Viterbi decoding returned exactly one label per active timestep in [0, 47].")

    # 4. Save and reload test using a temporary directory
    print("\nTesting CRF parameter checkpoint serialization...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "crf_head.pt"
        save_dict = {
            "crf": crf.state_dict(),
            "num_labels": num_labels,
        }
        torch.save(save_dict, tmp_path)
        loaded_dict = torch.load(tmp_path, map_location="cpu")
        new_crf = LinearChainCRF(num_labels=loaded_dict["num_labels"])
        new_crf.load_state_dict(loaded_dict["crf"])

        # Check parameter equality
        if not torch.allclose(crf.transitions, new_crf.transitions):
            print("FAIL: Loaded transitions do not match original!")
            return False
        if not torch.allclose(crf.start_transitions, new_crf.start_transitions):
            print("FAIL: Loaded start transitions do not match original!")
            return False
        if not torch.allclose(crf.end_transitions, new_crf.end_transitions):
            print("FAIL: Loaded end transitions do not match original!")
            return False
    print("PASS: CRF parameter checkpoint serialization & reload verified.")

    return True


def run_structural_batch_test() -> bool:
    print("\n--- TEST 2: LOCAL 1-BATCH STRUCTURAL CHECK (AGGREGATE ONLY) ---")

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

    # Check for labels outside 0..47
    active_labels = labels[active_mask]
    min_label = int(active_labels.min().item())
    max_label = int(active_labels.max().item())

    # Aggregates only
    print(f"Batch size:                   {batch_size}")
    print(f"Subword sequence length:      {subword_len}")
    print(f"Active original-token counts: {active_counts}")
    print(f"Max active length in batch:   {max_active_len}")
    print(f"Active label ID range:        [{min_label}, {max_label}] (valid range: [0, {len(label2id) - 1}])")
    print(f"Any zero-length sequences:    {any(c == 0 for c in active_counts)}")

    if any(c == 0 for c in active_counts):
        print("FAIL: Batch contains a zero-length sequence!")
        return False

    if min_label < 0 or max_label >= len(label2id):
        print(f"FAIL: Active label outside range [0, {len(label2id) - 1}]!")
        return False

    if max_active_len > subword_len:
        print("FAIL: Active token count exceeds subword length!")
        return False

    print("PASS: 1-batch structural verification passed all checks.")
    return True


def main():
    print("==================================================")
    print("VERIFICATION: EXPERIMENT 13 PRE-TRAINING AUDIT")
    print("==================================================")

    test1_ok = run_synthetic_crf_test()
    if not test1_ok:
        print("\n[!] TEST 1 FAILED!")
        sys.exit(1)

    test2_ok = run_structural_batch_test()
    if not test2_ok:
        print("\n[!] TEST 2 FAILED!")
        sys.exit(1)

    print("\n==================================================")
    print("ALL PRE-TRAINING VERIFICATION CHECKS PASSED!")
    print("==================================================")


if __name__ == "__main__":
    main()
