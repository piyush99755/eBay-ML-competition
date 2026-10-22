"""
Verification script for Experiment 15 (Boundary-Guided Joint Decoding).

Tests:
1. Decoding matrix construction:
   - Shapes of is_start, is_entity, and E (48x48)
   - Boundary transition semantics
2. Synthetic decoder test (no competition data used):
   - Invented logits with varying sequence lengths T in {1, 2, 5, 12, 25}
   - Backtracking returns exactly T labels
   - All labels strictly in range 0..47
   - Zero-weight equivalence: start_weight=0 & end_weight=0 strictly reproduces bio_logits.argmax(-1)
   - Boundary influence test: boundary probability correctly guides ambiguous BIO logits
3. Checkpoint & model integrity:
   - Checkpoint exists at checkpoints/xlmr_boundary_seed42_best
   - Weights and configurations load cleanly
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_examples import build_examples, load_training_data
from src.data.label_mapping import create_label_mappings
from src.evaluation.experiment_15_boundary_guided_decoding import (
    CHECKPOINT_DIR,
    build_boundary_decoding_matrices,
    boundary_guided_viterbi_decode,
)
from src.modeling.xlmr_boundary import XLMRobertaBoundary


def test_decoding_matrices(id2label: dict) -> bool:
    print("\n--- TEST 1: DECODING MATRIX CONSTRUCTION ---")
    is_start, is_entity, E = build_boundary_decoding_matrices(id2label)
    num_labels = len(id2label)

    print(f"Number of labels:             {num_labels}")
    print(f"is_start shape:               {list(is_start.shape)}")
    print(f"is_entity shape:              {list(is_entity.shape)}")
    print(f"Transition matrix E shape:    {list(E.shape)}")

    if is_start.shape != (num_labels,) or is_entity.shape != (num_labels,) or E.shape != (num_labels, num_labels):
        print("FAIL: Incorrect shapes for decoding matrices!")
        return False

    o_id = None
    for lid, tag in id2label.items():
        if tag == "O":
            o_id = lid
            break

    if o_id is None:
        print("FAIL: 'O' label not found in id2label!")
        return False

    # For 'O', is_start and is_entity must be 0
    if is_start[o_id].item() != 0 or is_entity[o_id].item() != 0:
        print("FAIL: 'O' has non-zero start or entity flag!")
        return False

    # Row o_id in E must be all zeros (O does not terminate an entity)
    if E[o_id].sum().item() != 0:
        print("FAIL: 'O' transitions should not terminate an entity!")
        return False

    # Check a B-X and I-X pair if present
    b_id = None
    i_id = None
    for lid, tag in id2label.items():
        if tag.startswith("B-"):
            b_tag = tag
            i_tag = "I-" + tag[2:]
            for lid2, tag2 in id2label.items():
                if tag2 == i_tag:
                    b_id = lid
                    i_id = lid2
                    break
            if b_id is not None and i_id is not None:
                break

    if b_id is not None and i_id is not None:
        # B-X to I-X should NOT terminate (E[b_id, i_id] == 0)
        if E[b_id, i_id].item() != 0:
            print("FAIL: B-X -> I-X should have E=0 (entity continues)!")
            return False

        # B-X to O SHOULD terminate (E[b_id, o_id] == 1)
        if E[b_id, o_id].item() != 1:
            print("FAIL: B-X -> O should have E=1 (entity terminates)!")
            return False

    print("PASS: Decoding matrix shapes and semantics verified.")
    return True


def test_synthetic_viterbi_decoder(id2label: dict) -> bool:
    print("\n--- TEST 2: SYNTHETIC VITERBI DECODER TESTS ---")
    is_start, is_entity, E = build_boundary_decoding_matrices(id2label)
    num_labels = len(id2label)

    torch.manual_seed(42)

    # 1. Test varying lengths: length T in {1, 2, 5, 12, 25}
    for T in [1, 2, 5, 12, 25]:
        bio_logits = torch.randn(T, num_labels)
        start_logits = torch.randn(T, 2)
        end_logits = torch.randn(T, 2)

        decoded = boundary_guided_viterbi_decode(
            bio_logits=bio_logits,
            start_logits=start_logits,
            end_logits=end_logits,
            is_start=is_start,
            is_entity=is_entity,
            E=E,
            start_weight=0.20,
            end_weight=0.20,
        )

        if len(decoded) != T:
            print(f"FAIL: Decoded length {len(decoded)} != expected {T}!")
            return False

        if not all(0 <= lid < num_labels for lid in decoded):
            print(f"FAIL: Decoded label outside 0..{num_labels - 1}!")
            return False

    print("PASS: Length matching and label range [0, 47] verified across varying sequence lengths.")

    # 2. Test zero-weight equivalence: start_weight=0 and end_weight=0 must reproduce bio_logits.argmax(-1)
    for T in [1, 3, 10]:
        bio_logits = torch.randn(T, num_labels)
        start_logits = torch.randn(T, 2)
        end_logits = torch.randn(T, 2)

        decoded_unweighted = boundary_guided_viterbi_decode(
            bio_logits=bio_logits,
            start_logits=start_logits,
            end_logits=end_logits,
            is_start=is_start,
            is_entity=is_entity,
            E=E,
            start_weight=0.0,
            end_weight=0.0,
        )

        expected_argmax = bio_logits.argmax(dim=-1).tolist()
        if decoded_unweighted != expected_argmax:
            print("FAIL: Zero-weight Viterbi does not match plain BIO argmax!")
            return False

    print("PASS: Zero-weight decoding strictly reproduces independent BIO argmax.")

    # 3. Test boundary head guidance on ambiguous logits
    # Find a B-X and I-X pair
    b_id, i_id = None, None
    for lid, tag in id2label.items():
        if tag.startswith("B-"):
            i_candidate = "I-" + tag[2:]
            for lid2, tag2 in id2label.items():
                if tag2 == i_candidate:
                    b_id, i_id = lid, lid2
                    break
            if b_id is not None:
                break

    if b_id is not None and i_id is not None:
        # Create a 1-token scenario where I-X has slightly higher BIO logit than B-X,
        # but START head has a strong positive START signal.
        bio_l = torch.zeros(1, num_labels)
        bio_l[0, i_id] = 10.0  # Slightly preferred by BIO
        bio_l[0, b_id] = 9.9   # Close second

        start_l = torch.zeros(1, 2)
        start_l[0, 1] = 50.0   # Extremely strong START signal
        start_l[0, 0] = -50.0

        end_l = torch.zeros(1, 2)  # Neutral END signal

        # Without boundary guidance (weight=0): should predict i_id
        dec_unweighted = boundary_guided_viterbi_decode(
            bio_l, start_l, end_l, is_start, is_entity, E, start_weight=0.0, end_weight=0.0
        )
        # With boundary guidance: should tip to b_id because it is a start token
        dec_guided = boundary_guided_viterbi_decode(
            bio_l, start_l, end_l, is_start, is_entity, E, start_weight=0.20, end_weight=0.20
        )

        if dec_unweighted != [i_id]:
            print(f"FAIL: Unweighted expected {[i_id]}, got {dec_unweighted}!")
            return False

        if dec_guided != [b_id]:
            print(f"FAIL: Guided expected {[b_id]}, got {dec_guided}!")
            return False

        print("PASS: Boundary head successfully guides ambiguous BIO logits.")

    return True


def test_checkpoint_integrity() -> bool:
    print("\n--- TEST 3: CHECKPOINT INTEGRITY ---")
    if not CHECKPOINT_DIR.exists():
        print(f"FAIL: Checkpoint directory {CHECKPOINT_DIR} does not exist!")
        return False

    weights_file = CHECKPOINT_DIR / "boundary_heads.pt"
    if not weights_file.exists():
        print(f"FAIL: Boundary weights file {weights_file} does not exist!")
        return False

    print(f"Found checkpoint at: {CHECKPOINT_DIR}")
    print(f"Found weights file:  {weights_file}")

    checkpoint_weights = torch.load(weights_file, map_location="cpu")
    required_keys = ["bio_classifier", "start_classifier", "end_classifier", "label2id", "id2label"]
    for k in required_keys:
        if k not in checkpoint_weights:
            print(f"FAIL: Missing key '{k}' in {weights_file}!")
            return False

    print(f"Checkpoint contains all required keys: {required_keys}")
    print(f"BIO classes in checkpoint: {len(checkpoint_weights['label2id'])}")
    print("PASS: Checkpoint structure and weights verified.")
    return True


def main():
    print("==================================================")
    print("VERIFICATION: EXPERIMENT 15 PRE-EVALUATION AUDIT")
    print("==================================================")

    # Load safe label mappings without printing raw data
    df = load_training_data()
    examples = build_examples(df)
    label2id, id2label = create_label_mappings(examples)

    test1_ok = test_decoding_matrices(id2label)
    if not test1_ok:
        print("\n[!] TEST 1 FAILED!")
        sys.exit(1)

    test2_ok = test_synthetic_viterbi_decoder(id2label)
    if not test2_ok:
        print("\n[!] TEST 2 FAILED!")
        sys.exit(1)

    test3_ok = test_checkpoint_integrity()
    if not test3_ok:
        print("\n[!] TEST 3 FAILED!")
        sys.exit(1)

    print("\n==================================================")
    print("ALL EXPERIMENT 15 PRE-EVALUATION CHECKS PASSED!")
    print("==================================================")


if __name__ == "__main__":
    main()
