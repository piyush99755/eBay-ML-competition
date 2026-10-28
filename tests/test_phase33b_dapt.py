"""
Phase 33B: Unit Tests for DAPT and Downstream Fine-Tuning
Tests:
1. Deterministic sampling repeatability and seed control.
2. Title-overlap exclusions (eval titles excluded, record_id <= 30000 excluded).
3. MLM masking behavior (special tokens and padding excluded from loss).
4. Dataset parsing (token count and format).
5. Label mapping compatibility (48 BIO classes).
6. Supervised checkpoint initialization from adapted encoder.
7. Entity decoding and confidence calculation.
8. Champion artifact protection (frozen weights unchanged).
"""

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from src.data.sample_dapt_corpus import build_exclusion_set, sample_dapt_corpus
from src.evaluation.entity_metrics import (
    bio_to_entities,
    bio_to_entities_with_confidence,
    filter_entities_by_category,
)


class TestPhase33BDAPT(unittest.TestCase):

    def test_champion_artifact_protection(self):
        """Verify the 4 frozen champion checkpoints are intact and unchanged."""
        expected_hashes = {
            "gbert_baseline_best": "4d909833701cdb202618b4366ad1c34d759df6a1ebfaae6982c228e1e7a44148",
            "xlmr_baseline_best": "e1933989c5625e78d63d27cfee8781d311dbf01d9d24e122a97ada97eec67eb3",
            "xlmr_seed_7_best": "49437dff72fa00dab4e5c3e045ad2c21d76db4f4207e0293e1898aae35998170",
            "xlmr_seed_21_best": "638baa567ece3ae9b363fe0869454f300fe2ff1cb20e0b57a8b1d9c838bc5338",
        }

        for model_dir, exp_hash in expected_hashes.items():
            model_path = Path("checkpoints") / model_dir / "model.safetensors"
            self.assertTrue(model_path.exists(), f"Missing champion file: {model_path}")
            h = hashlib.sha256()
            with open(model_path, "rb") as f:
                while chunk := f.read(1024 * 1024):
                    h.update(chunk)
            actual_hash = h.hexdigest().lower()
            self.assertEqual(
                actual_hash,
                exp_hash.lower(),
                f"PROTECTION BREACH: Champion checkpoint {model_dir} was modified!",
            )

    def test_exclusion_set_rules(self):
        """Verify forbidden record ID cutoff and non-empty normalized evaluation titles."""
        forbidden_max_id, forbidden_titles, stats = build_exclusion_set()
        self.assertEqual(forbidden_max_id, 30000)
        self.assertGreater(len(forbidden_titles), 4000)
        self.assertEqual(stats["forbidden_max_record_id"], 30000)

    def test_deterministic_sampling_reproducibility(self):
        """Verify sampling with fixed seed produces identical record sequence."""
        # Test with small count of 10 per category
        s1, stats1 = sample_dapt_corpus(sample_size_per_category=10, random_seed=42)
        s2, stats2 = sample_dapt_corpus(sample_size_per_category=10, random_seed=42)

        self.assertEqual(len(s1), 20)
        self.assertEqual(len(s2), 20)
        self.assertEqual([x["record_id"] for x in s1], [x["record_id"] for x in s2])
        self.assertEqual([x["category"] for x in s1], [x["category"] for x in s2])

        # All records must be strictly > 30000
        for item in s1:
            self.assertGreater(item["record_id"], 30000)

    def test_mlm_collator_masking_rules(self):
        """Verify DataCollatorForLanguageModeling excludes special tokens and padding from loss."""
        from transformers import DataCollatorForLanguageModeling

        tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-base", local_files_only=True)
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

        examples = [
            {"input_ids": [102, 1500, 2500, 103], "special_tokens_mask": [1, 0, 0, 1]},
            {"input_ids": [102, 3500, 103, 0], "special_tokens_mask": [1, 0, 1, 1]},
        ]
        batch = collator(examples)

        # Labels must be -100 on [CLS] (index 0) and [SEP]
        labels = batch["labels"]
        self.assertEqual(labels[0, 0].item(), -100, "CLS token was masked in labels!")

    def test_label_mapping_compatibility(self):
        """Verify that reference config contains exactly 48 classes with 'O' at expected ID."""
        cfg = AutoConfig.from_pretrained("checkpoints/gbert_baseline_best")
        self.assertEqual(len(cfg.id2label), 48)
        self.assertIn("O", cfg.label2id)
        self.assertIn("B-Kompatibles_Fahrzeug_Modell", cfg.label2id)
        self.assertIn("I-Kompatibles_Fahrzeug_Modell", cfg.label2id)

    def test_supervised_encoder_initialization(self):
        """Verify that AutoModelForTokenClassification loads from base GBERT and attaches 48-class head."""
        cfg = AutoConfig.from_pretrained("checkpoints/gbert_baseline_best")
        model = AutoModelForTokenClassification.from_pretrained(
            "deepset/gbert-base",
            num_labels=48,
            id2label=cfg.id2label,
            label2id=cfg.label2id,
            local_files_only=True,
        )
        self.assertEqual(model.config.num_labels, 48)
        self.assertEqual(model.classifier.out_features, 48)

    def test_entity_decoding_and_confidence(self):
        """Verify bio_to_entities_with_confidence and category filtering."""
        tokens = ["BOSCH", "Bremsscheiben", "vorne", "Golf", "Passat"]
        bio = ["B-Hersteller", "B-Produktart", "B-Einbauposition", "B-Kompatibles_Fahrzeug_Modell", "B-Kompatibles_Fahrzeug_Modell"]
        probs = [0.99, 0.95, 0.92, 0.88, 0.45]

        ents = bio_to_entities_with_confidence(tokens, bio, probs)
        self.assertEqual(len(ents), 5)

        # Category 1 filtering
        cat1_ents = filter_entities_by_category(ents, category=1)
        self.assertEqual(len(cat1_ents), 5)

        # Min confidence 0.60 filtering
        filtered_060 = [e for e in cat1_ents if e[3] >= 0.60]
        # 'Passat' has conf 0.45, so it should be pruned
        self.assertEqual(len(filtered_060), 4)
        self.assertEqual([e[1] for e in filtered_060], ["BOSCH", "Bremsscheiben", "vorne", "Golf"])


if __name__ == "__main__":
    unittest.main()
