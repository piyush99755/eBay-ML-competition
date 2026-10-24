"""
Audit all Phase 22 full-data champion checkpoints and confirm original checkpoints remain intact.
"""

from pathlib import Path
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

FULLDATA_CHECKPOINTS = [
    ("gbert_seed42", Path("checkpoints/gbert_fulldata_seed42_final")),
    ("xlmr_seed42", Path("checkpoints/xlmr_fulldata_seed42_final")),
    ("xlmr_seed7", Path("checkpoints/xlmr_fulldata_seed7_final")),
    ("xlmr_seed21", Path("checkpoints/xlmr_fulldata_seed21_final")),
]

ORIGINAL_CHAMPION_CHECKPOINTS = [
    Path("checkpoints/gbert_baseline_best"),
    Path("checkpoints/xlmr_baseline_best"),
    Path("checkpoints/xlmr_seed_7_best"),
    Path("checkpoints/xlmr_seed_21_best"),
    Path("checkpoints/xlmr_span_seed42_best"),
]


def audit_checkpoints():
    print("=" * 60)
    print("AUDITING PHASE 22 FULL-DATA CHAMPION CHECKPOINTS")
    print("=" * 60)

    device = torch.device("cpu")
    total_size_mb = 0.0

    for model_id, cp_path in FULLDATA_CHECKPOINTS:
        assert cp_path.exists(), f"Missing checkpoint directory: {cp_path}"
        weights = cp_path / "model.safetensors"
        assert weights.exists(), f"Missing weights: {weights}"
        config = cp_path / "config.json"
        assert config.exists(), f"Missing config: {config}"

        cp_size_mb = sum(f.stat().st_size for f in cp_path.iterdir() if f.is_file()) / (1024 ** 2)
        total_size_mb += cp_size_mb

        tokenizer = AutoTokenizer.from_pretrained(cp_path)
        model = AutoModelForTokenClassification.from_pretrained(cp_path).to(device)

        assert len(model.config.id2label) == 48, f"{model_id}: expected 48 labels"
        assert model.config.id2label[47] == "O", f"{model_id}: index 47 is not 'O'"
        assert model.config.num_labels == 48, f"{model_id}: num_labels != 48"

        # Synthetic forward pass
        tokens = ["Bremsscheibe", "vorne", "links", "Passat"]
        enc = tokenizer(tokens, is_split_into_words=True, return_tensors="pt")
        with torch.no_grad():
            out = model(**enc)
        assert out.logits.shape[-1] == 48, f"{model_id}: logits shape != 48"

        print(f"PASS: {model_id:<14} -> {cp_path.name:<32} (Size: {cp_size_mb:6.1f} MB, Vocab: {len(tokenizer):,})")

        del model, tokenizer

    print(f"\nTotal full-data checkpoint storage: {total_size_mb:.1f} MB ({total_size_mb / 1024:.2f} GB)")

    print("\n" + "=" * 60)
    print("VERIFYING ORIGINAL CHAMPION & EXPLORATORY CHECKPOINTS")
    print("=" * 60)
    for cp_path in ORIGINAL_CHAMPION_CHECKPOINTS:
        assert cp_path.exists(), f"Original checkpoint missing: {cp_path}"
        weights = cp_path / "model.safetensors"
        assert weights.exists(), f"Original weights missing: {weights}"
        size_mb = sum(f.stat().st_size for f in cp_path.iterdir() if f.is_file()) / (1024 ** 2)
        print(f"INTACT: {cp_path.name:<30} (Size: {size_mb:6.1f} MB)")

    print("\nALL CHECKPOINTS VERIFIED AND INTACT!")


if __name__ == "__main__":
    audit_checkpoints()
