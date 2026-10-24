"""
Inspect the 3 category-invalid aspect cases found in Split-A validation.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import torch

from src.data.build_examples import build_examples, load_training_data
from src.data.label_mapping import create_label_mappings
from src.data.make_splits import split_examples
from src.evaluation.experiment_23a_constrained_decoding import (
    GBERT_CHECKPOINT,
    XLMR_SEED_42_CHECKPOINT,
    XLMR_SEED_7_CHECKPOINT,
    XLMR_SEED_21_CHECKPOINT,
    collect_active_logits,
    compute_ensemble_probabilities,
    decode_unconstrained_raw,
)
from scripts.validate_submission import (
    OFFICIAL_ASPECTS_CAT1,
    OFFICIAL_ASPECTS_CAT2,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_training_data()
    examples = build_examples(df)
    l2i, i2l = create_label_mappings(examples)
    _, val_examples = split_examples(examples, random_state=42)

    gb = collect_active_logits(GBERT_CHECKPOINT, val_examples, l2i, device)
    s42 = collect_active_logits(XLMR_SEED_42_CHECKPOINT, val_examples, l2i, device)
    s7 = collect_active_logits(XLMR_SEED_7_CHECKPOINT, val_examples, l2i, device)
    s21 = collect_active_logits(XLMR_SEED_21_CHECKPOINT, val_examples, l2i, device)
    probs = compute_ensemble_probabilities(gb, s42, s7, s21)

    raw_preds = decode_unconstrained_raw(val_examples, probs, i2l)

    print("\n--- DETAILED INSPECTION OF THE 3 INVALID ASPECT PREDICTIONS ---")
    case_num = 0
    for idx, (ex, r_ents) in enumerate(zip(val_examples, raw_preds)):
        cat = ex["category"]
        allowed = OFFICIAL_ASPECTS_CAT1 if cat == 1 else OFFICIAL_ASPECTS_CAT2
        for asp, val in r_ents:
            if asp not in allowed:
                case_num += 1
                first_tok = val.split()[0]
                tok_idx = ex["tokens"].index(first_tok) if first_tok in ex["tokens"] else -1
                print(f"\nCase {case_num}: Example {idx} (Category {cat})")
                print(f"  Title Tokens: {' '.join(ex['tokens'])}")
                print(f"  Ground Truth: {ex['bio_tags']}")
                print(f"  Predicted Invalid Entity: ({asp}, '{val}')")
                if tok_idx >= 0:
                    p = probs[idx][tok_idx]
                    top_indices = np.argsort(p)[-5:][::-1]
                    top_ranked = [(i2l[i], round(float(p[i]), 4)) for i in top_indices]
                    print(f"  Token '{first_tok}' Probabilities (Top 5): {top_ranked}")


if __name__ == "__main__":
    main()
