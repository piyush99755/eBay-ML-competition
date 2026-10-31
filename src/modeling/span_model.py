"""
Span-Based NER Architecture for Phase 50 Boundary-Aware Span Learning.

Features:
- DAPT GBERT encoder (768 hidden size)
- Span representation: [h_start ; h_end ; width_embedding]
- Multi-layer perceptron classification head
- Vectorized batched forward pass and cross-entropy loss
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from src.modeling.span_dataset import NUM_CLASSES


class SpanNERModel(nn.Module):
    """Boundary-Aware Span NER Model built on top of DAPT GBERT."""

    def __init__(
        self,
        encoder_path: str,
        num_classes: int = NUM_CLASSES,
        max_width: int = 8,
        width_embed_dim: int = 64,
        hidden_dim: int = 256,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.encoder_path = encoder_path
        self.num_classes = num_classes
        self.max_width = max_width
        self.width_embed_dim = width_embed_dim

        self.encoder = AutoModel.from_pretrained(encoder_path, local_files_only=True)
        encoder_hidden = self.encoder.config.hidden_size

        self.width_embedding = nn.Embedding(max_width + 1, width_embed_dim)

        # Span representation: start (768) + end (768) + width (64) = 1600
        span_repr_dim = encoder_hidden * 2 + width_embed_dim

        self.classifier = nn.Sequential(
            nn.Linear(span_repr_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        span_batch_idx: torch.Tensor,
        span_sub_start: torch.Tensor,
        span_sub_end: torch.Tensor,
        span_width: torch.Tensor,
        span_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass with vectorized span extraction.
        """
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = encoder_outputs.last_hidden_state  # [B, seq_len, 768]

        if span_batch_idx.numel() == 0:
            # Edge case: no candidate spans
            return {
                "logits": torch.empty((0, self.num_classes), device=input_ids.device),
                "loss": torch.tensor(0.0, device=input_ids.device, requires_grad=True),
            }

        # Vectorized candidate span extraction across all batch elements
        h_start = sequence_output[span_batch_idx, span_sub_start]  # [N_spans, 768]
        h_end = sequence_output[span_batch_idx, span_sub_end]      # [N_spans, 768]
        w_emb = self.width_embedding(span_width)                  # [N_spans, width_embed_dim]

        span_repr = torch.cat([h_start, h_end, w_emb], dim=-1)   # [N_spans, span_repr_dim]
        logits = self.classifier(span_repr)                     # [N_spans, num_classes]

        loss = None
        if span_labels is not None:
            loss = F.cross_entropy(logits, span_labels)

        return {
            "loss": loss,
            "logits": logits,
        }

    def save_pretrained(self, save_directory: Path):
        """Save model weights and architectural configuration."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Save encoder weights and config
        self.encoder.save_pretrained(save_directory)
        model_state = self.state_dict()
        torch.save(
            {
                "state_dict": model_state,
                "num_classes": self.num_classes,
                "max_width": self.max_width,
                "width_embed_dim": self.width_embed_dim,
                "encoder_path": self.encoder_path,
            },
            save_directory / "span_model_checkpoint.pt",
        )

    @classmethod
    def from_pretrained(cls, checkpoint_dir: Path, device: torch.device = torch.device("cpu")):
        """Load trained SpanNERModel from checkpoint directory."""
        checkpoint_dir = Path(checkpoint_dir)
        pt_path = checkpoint_dir / "span_model_checkpoint.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"Missing {pt_path}")

        checkpoint_data = torch.load(pt_path, map_location=device, weights_only=False)
        model = cls(
            encoder_path=str(checkpoint_dir),
            num_classes=checkpoint_data["num_classes"],
            max_width=checkpoint_data["max_width"],
            width_embed_dim=checkpoint_data["width_embed_dim"],
        )
        model.load_state_dict(checkpoint_data["state_dict"])
        model.to(device)
        return model
