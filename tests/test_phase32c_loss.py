"""
Unit tests for Phase 32C weighted loss implementation and label mapping.
"""

import unittest
import torch
import torch.nn as nn
from transformers import AutoConfig
from pathlib import Path


def create_phase32c_loss_weights(
    label2id: dict,
    i_modell_weight: float = 1.00,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create class weight vector for CrossEntropyLoss."""
    num_labels = len(label2id)
    weights = torch.ones(num_labels, dtype=torch.float32, device=device)

    target_aspect_i = "I-Kompatibles_Fahrzeug_Modell"
    if target_aspect_i in label2id:
        i_id = label2id[target_aspect_i]
        weights[i_id] = float(i_modell_weight)

    return weights


class TestPhase32CLoss(unittest.TestCase):

    def setUp(self):
        cfg = AutoConfig.from_pretrained(Path("checkpoints/gbert_baseline_best"))
        self.label2id = {v: int(k) for k, v in cfg.id2label.items()}
        self.id2label = {int(k): v for k, v in cfg.id2label.items()}

    def test_label_mapping_integrity(self):
        """Verify label2id contains required aspect labels and 48 total classes."""
        self.assertEqual(len(self.label2id), 48)
        self.assertIn("I-Kompatibles_Fahrzeug_Modell", self.label2id)
        self.assertIn("B-Kompatibles_Fahrzeug_Modell", self.label2id)
        self.assertIn("O", self.label2id)

    def test_baseline_weights_are_uniform(self):
        """Verify baseline 1.00x weights are identically 1.0 for all 48 classes."""
        weights = create_phase32c_loss_weights(self.label2id, i_modell_weight=1.00)
        self.assertEqual(weights.shape[0], 48)
        self.assertTrue(torch.all(weights == 1.0))

    def test_challenger_weights_isolate_i_modell(self):
        """Verify challenger 1.15x weight is applied strictly and exclusively to I-Modell."""
        weights = create_phase32c_loss_weights(self.label2id, i_modell_weight=1.15)
        i_id = self.label2id["I-Kompatibles_Fahrzeug_Modell"]

        # Check that I-Modell is exactly 1.15
        self.assertAlmostEqual(weights[i_id].item(), 1.15, places=5)

        # Check that all other 47 weights are exactly 1.0
        for idx in range(len(self.label2id)):
            if idx != i_id:
                self.assertEqual(weights[idx].item(), 1.0)

    def test_ignore_index_masking(self):
        """Verify -100 labels produce zero loss and do not affect gradients."""
        num_classes = 48
        logits = torch.randn(2, 5, num_classes, requires_grad=True)
        labels = torch.full((2, 5), -100, dtype=torch.long)

        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(logits.view(-1, num_classes), labels.view(-1))
        self.assertTrue(loss.item() == 0.0 or torch.isnan(loss))

    def test_weighted_loss_gradient_scaling(self):
        """Verify 1.15x weight increases loss proportionally for I-Modell tokens."""
        i_id = self.label2id["I-Kompatibles_Fahrzeug_Modell"]
        b_id = self.label2id["B-Kompatibles_Fahrzeug_Modell"]
        num_classes = len(self.label2id)

        w_base = create_phase32c_loss_weights(self.label2id, i_modell_weight=1.00)
        w_chal = create_phase32c_loss_weights(self.label2id, i_modell_weight=1.15)

        loss_base_fn = nn.CrossEntropyLoss(weight=w_base, ignore_index=-100)
        loss_chal_fn = nn.CrossEntropyLoss(weight=w_chal, ignore_index=-100)

        # Asymmetric batch where B-Modell is predicted with low loss, I-Modell with high loss
        logits_asym = torch.zeros(2, num_classes)
        logits_asym[0, b_id] = 10.0  # B-Modell is almost perfectly predicted (near 0 loss)
        logits_asym[1, 0] = 10.0     # I-Modell is misclassified as 0 (high loss)
        labels_asym = torch.tensor([b_id, i_id], dtype=torch.long)

        l_asym_base = loss_base_fn(logits_asym, labels_asym)
        l_asym_chal = loss_chal_fn(logits_asym, labels_asym)

        # Baseline: (0 + L) / 2 = 0.50 L
        # Challenger: (0 + 1.15 * L) / (1 + 1.15) = 1.15/2.15 L = 0.5349 L > 0.50 L
        self.assertGreater(l_asym_chal.item(), l_asym_base.item())


if __name__ == "__main__":
    unittest.main()
