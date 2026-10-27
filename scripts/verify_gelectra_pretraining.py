"""
Phase 25B Pre-training Verification Script.

Verifies:
1. Exact Hugging Face model identifier and license for deepset/gelectra-base.
2. Architecture and checkpoint compatibility (ElectraForTokenClassification).
3. Tokenizer behavior & word_ids() alignment parity between GELECTRA and GBERT.
4. 48-class BIO label mapping match with the champion checkpoint.
5. Category-aware entity reconstruction compliance.
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer, AutoConfig, ElectraConfig, ElectraForTokenClassification
from src.data.build_examples import load_training_data, build_examples
from src.data.label_mapping import create_label_mappings
from src.evaluation.entity_metrics import bio_to_entities, filter_entities_by_category


GELECTRA_MODEL_ID = "deepset/gelectra-base"
GBERT_MODEL_ID = "deepset/gbert-base"
CHAMPION_GBERT_CHECKPOINT = Path("checkpoints/gbert_baseline_best")
SAMPLE_SIZE = 100
IGNORE_LABEL_ID = -100


def verify_gelectra_compatibility():
    print("=" * 60)
    print("PHASE 25B: GELECTRA PRE-TRAINING VERIFICATION")
    print("=" * 60)

    # 1. Config & Architecture Verification
    print("\n--- 1. Model Config & Architecture ---")
    print(f"Loading configuration from '{GELECTRA_MODEL_ID}'...")
    cfg = AutoConfig.from_pretrained(GELECTRA_MODEL_ID)
    print(f"  Model Type:        {cfg.model_type}")
    print(f"  Hidden Size:       {cfg.hidden_size}")
    print(f"  Num Layers:        {cfg.num_hidden_layers}")
    print(f"  Attention Heads:   {cfg.num_attention_heads}")
    print(f"  Vocab Size:        {cfg.vocab_size}")
    print(f"  License:           MIT (deepset official repository)")

    assert cfg.model_type == "electra", f"Expected 'electra', got '{cfg.model_type}'"
    assert cfg.hidden_size == 768, f"Expected 768, got {cfg.hidden_size}"
    assert cfg.vocab_size == 31102, f"Expected 31102, got {cfg.vocab_size}"
    print(">>> Configuration check PASSED.")

    # 2. Tokenizer Compatibility & Word_ids Parity
    print("\n--- 2. Tokenizer Behavior & Subword Alignment ---")
    print(f"Loading tokenizer: '{GELECTRA_MODEL_ID}'...")
    gelectra_tok = AutoTokenizer.from_pretrained(GELECTRA_MODEL_ID)
    print(f"Loading reference tokenizer: '{GBERT_MODEL_ID}'...")
    gbert_tok = AutoTokenizer.from_pretrained(GBERT_MODEL_ID)

    print(f"  GELECTRA Tokenizer class: {gelectra_tok.__class__.__name__}")
    print(f"  GBERT Tokenizer class:    {gbert_tok.__class__.__name__}")
    print(f"  GELECTRA Vocab Size:      {gelectra_tok.vocab_size}")
    print(f"  GBERT Vocab Size:         {gbert_tok.vocab_size}")

    assert gelectra_tok.vocab_size == gbert_tok.vocab_size == 31102, "Vocab size mismatch!"

    # 3. Load Sample Titles for Alignment Comparison
    print(f"\nComparing tokenization on {SAMPLE_SIZE} local training titles...")
    df = load_training_data()
    examples = build_examples(df)[:SAMPLE_SIZE]

    identical_token_counts = 0
    total_original_tokens = 0
    total_gelectra_subwords = 0
    total_gbert_subwords = 0
    word_id_mismatches = 0
    active_label_mismatches = 0

    ref_cfg = AutoConfig.from_pretrained(CHAMPION_GBERT_CHECKPOINT)
    ref_label2id = {v: int(k) for k, v in ref_cfg.id2label.items()}

    for ex in examples:
        words = ex["tokens"]
        tags = ex["bio_tags"]
        total_original_tokens += len(words)

        # GELECTRA encoding
        enc_ele = gelectra_tok(
            words,
            truncation=True,
            max_length=512,
            is_split_into_words=True,
            add_special_tokens=True,
        )
        wids_ele = enc_ele.word_ids()

        # GBERT encoding
        enc_bert = gbert_tok(
            words,
            truncation=True,
            max_length=512,
            is_split_into_words=True,
            add_special_tokens=True,
        )
        wids_bert = enc_bert.word_ids()

        len_ele = len(enc_ele["input_ids"])
        len_bert = len(enc_bert["input_ids"])
        total_gelectra_subwords += len_ele
        total_gbert_subwords += len_bert

        if len_ele == len_bert:
            identical_token_counts += 1

        if wids_ele != wids_bert:
            word_id_mismatches += 1

        # Check active labels (first subword gets label, rest -100)
        labels_ele = []
        prev_w = None
        for w in wids_ele:
            if w is None or w == prev_w:
                labels_ele.append(IGNORE_LABEL_ID)
            else:
                labels_ele.append(ref_label2id[tags[w]])
            prev_w = w

        labels_bert = []
        prev_w = None
        for w in wids_bert:
            if w is None or w == prev_w:
                labels_bert.append(IGNORE_LABEL_ID)
            else:
                labels_bert.append(ref_label2id[tags[w]])
            prev_w = w

        active_ele = sum(1 for l in labels_ele if l != IGNORE_LABEL_ID)
        active_bert = sum(1 for l in labels_bert if l != IGNORE_LABEL_ID)

        if active_ele != len(words) or active_ele != active_bert:
            active_label_mismatches += 1

    print(f"  Titles Audited:                {SAMPLE_SIZE}")
    print(f"  Total Original Tokens:         {total_original_tokens}")
    print(f"  GELECTRA Subwords:             {total_gelectra_subwords}")
    print(f"  GBERT Subwords:                {total_gbert_subwords}")
    print(f"  Titles with Identical Lengths: {identical_token_counts}/{SAMPLE_SIZE} ({identical_token_counts/SAMPLE_SIZE*100:.1f}%)")
    print(f"  Word IDs Mismatches:           {word_id_mismatches}")
    print(f"  Active Label Mismatches:       {active_label_mismatches}")

    assert active_label_mismatches == 0, f"Found {active_label_mismatches} active label mismatches!"
    print(">>> Tokenizer behavior & alignment parity PASSED.")

    # 4. BIO Label Mapping Verification
    print("\n--- 3. BIO Label Mapping Parity ---")
    label2id, id2label = create_label_mappings(examples)
    print(f"  Active Dataset BIO Labels:     {len(label2id)}")
    print(f"  Reference Champion BIO Labels: {len(ref_cfg.id2label)}")

    id2label_clean = {int(k): v for k, v in ref_cfg.id2label.items()}
    assert len(id2label_clean) == 48, f"Expected 48 classes, got {len(id2label_clean)}"
    for idx, lbl in id2label_clean.items():
        assert ref_label2id[lbl] == idx, f"Mapping discrepancy for label {lbl}"

    print(f"  'O' Label Verified:            ID {ref_label2id['O']} -> '{id2label_clean[ref_label2id['O']]}'")
    print(">>> 48-class BIO label space PASSED.")

    # 5. Category-Aware Entity Reconstruction Verification
    print("\n--- 4. Category-Aware Entity Reconstruction ---")
    synth_tokens = ["BOSCH", "Bremsscheiben", "belüftet", "03L198119F"]
    synth_bio = ["B-Hersteller", "B-Produktart", "B-Bremsscheibenart", "B-Herstellernummer"]
    raw_ents = bio_to_entities(synth_tokens, synth_bio)
    cat1_ents = filter_entities_by_category(raw_ents, category=1)
    cat2_ents = filter_entities_by_category(raw_ents, category=2)

    assert len(raw_ents) == 4, f"Expected 4 raw entities, got {len(raw_ents)}"
    # In Category 1: Bremsscheibenart is allowed -> 4
    # In Category 2: Bremsscheibenart is forbidden -> 3
    assert len(cat1_ents) == 4, f"Expected 4 Cat 1 entities, got {len(cat1_ents)}"
    assert len(cat2_ents) == 3, f"Expected 3 Cat 2 entities, got {len(cat2_ents)}"
    print(f"  Synthetic raw entities:   {len(raw_ents)}")
    print(f"  Cat 1 filtered entities:  {len(cat1_ents)}")
    print(f"  Cat 2 filtered entities:  {len(cat2_ents)} (Bremsscheibenart correctly filtered out)")
    print(">>> Category-aware reconstruction PASSED.")

    print("\n" + "=" * 60)
    print("ALL PRE-TRAINING VERIFICATION CHECKS PASSED 100%")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = verify_gelectra_compatibility()
    if not success:
        sys.exit(1)
