# Experiment 20: Final Submission Readiness and Pipeline Audit

**Date**: October 2026  
**Status**: **READY** (All technical, schema, and inference compliance checks passed; 0 blockers).  
**Target Submission Platform**: EvalAI (eBay 2026 Machine Learning Competition).  
**Current Champion**: Experiment 10 (GBERT 0.40 + 3-seed XLM-R 0.60 ensemble; Split-A local $F_{0.2} = 0.879852$).

---

## 1. Executive Summary & Readiness Verdict

* **Overall Status**: **READY**
* **Active Champion**: Experiment 10 4-model ensemble:
  * 0.40 GBERT (`checkpoints/gbert_baseline_best`)
  * 0.20 XLM-R seed 42 (`checkpoints/xlmr_baseline_best`)
  * 0.20 XLM-R seed 7 (`checkpoints/xlmr_seed_7_best`)
  * 0.20 XLM-R seed 21 (`checkpoints/xlmr_seed_21_best`)
* **Local Split-A Validation Benchmark**: $F_{0.2} = 0.879852$ (reproduced bit-for-bit across Experiments 10, 11, 18, and 19).
* **Official Compliance**: Verified 100% against official competition specification in `data/raw/Annexure.pdf`.
* **Zero Blockers**: All 4 champion checkpoints exist, share identical 48-label mappings, and run sequentially within 6 GB GPU VRAM.

---

## 2. Official Competition Specification Compliance (Annexure.pdf)

The pipeline was audited against the official guidelines provided in `data/raw/Annexure.pdf`:

| Specification Requirement | Official Rule (Annexure.pdf) | Implementation Status | Verification Details |
| :--- | :--- | :---: | :--- |
| **Output File Structure** | Literal Tab-Separated Values (TSV), 1 row per extracted entity. | **COMPLIANT** | Evaluated by `scripts/validate_submission.py`. Uses `\t` separator without quotes. |
| **Column Schema** | Exactly 4 values: `Record Number`, `Category`, `Aspect Name`, `Aspect Value`. | **COMPLIANT** | Validated against official header `Record Number\tCategory\tAspect Name\tAspect Value`. |
| **Character Encoding** | UTF-8 encoded text with literal characters (e.g. `€`, `ä`, `ö`, `ü`, `ß`, `Ø`). | **COMPLIANT** | Explicitly written with `encoding="utf-8"`. Non-ASCII German characters preserved. |
| **Escaping & Quoting** | Literal text, **no CSV-style quoting or escaping** (`csv.QUOTE_NONE`). | **COMPLIANT** | Avoids pandas default quotation. Verified with synthetic tests. |
| **Special Tag "O"** | Semantic entities tagged with `O` must be omitted from submissions. | **COMPLIANT** | `O` is strictly filtered out during row generation; validator rejects any `O` aspect. |
| **Whitespace Semantics** | Original tokens separated by whitespace in titles; multi-token entities joined by ASCII `0x20`. | **COMPLIANT** | `bio_to_entities` joins tokens using single ASCII space `" ".join(tokens)`. |
| **Consecutive Identical Tags** | Consecutive non-empty tags represent distinct entities and must not be combined. | **COMPLIANT** | `convert_tags_to_bio` produces consecutive `B-<tag>` labels; `bio_to_entities` splits them. |
| **Empty Continuation Tags** | Empty tag `""` continues previous entity. | **COMPLIANT** | Handled correctly by `convert_tags_to_bio` into `I-<tag>` labels. |
| **Zero-Aspect Records** | Titles producing zero non-O entities emit 0 submission rows. | **COMPLIANT** | Validator permits titles with zero aspect rows (no 1-row-per-title enforcement). |
| **Candidate Ranges** | Quiz Data: records `5,001` to `30,000` (25,000 titles, ~2,000 scored). Test Data: 25,000 titles. | **COMPLIANT** | Validator enforces integer boundaries `[5001, 30000]` for Quiz submissions. |
| **Official Aspect Names** | Exactly 29 semantic aspect types permitted. | **COMPLIANT** | Set of 29 aspects matches Annexure.pdf pages 11-14; unknown aspects trigger errors. |

---

## 3. Checkpoint and Model Inventory

The 4 champion checkpoints were verified on disk for configuration, model weights, and tokenizer files:

| Checkpoint Identifier | Model Architecture | Seed | Weight in Ensemble | Files Present | Disk Size |
| :--- | :--- | :---: | :---: | :--- | :---: |
| `checkpoints/gbert_baseline_best` | `BertForTokenClassification` (`deepset/gbert-base`) | 42 | **0.40** | `config.json`, `model.safetensors`, `vocab.txt`, `tokenizer.json` | 418.2 MB |
| `checkpoints/xlmr_baseline_best` | `XLMRobertaForTokenClassification` (`FacebookAI/xlm-roberta-base`) | 42 | **0.20** | `config.json`, `model.safetensors`, `sentencepiece.bpe.model`, `tokenizer.json` | 1,079.7 MB |
| `checkpoints/xlmr_seed_7_best` | `XLMRobertaForTokenClassification` (`FacebookAI/xlm-roberta-base`) | 7 | **0.20** | `config.json`, `model.safetensors`, `sentencepiece.bpe.model`, `tokenizer.json` | 1,079.7 MB |
| `checkpoints/xlmr_seed_21_best` | `XLMRobertaForTokenClassification` (`FacebookAI/xlm-roberta-base`) | 21 | **0.20** | `config.json`, `model.safetensors`, `sentencepiece.bpe.model`, `tokenizer.json` | 1,079.7 MB |
| **Total 4-Model Checkpoints** | — | — | **1.00** | — | **3,657.3 MB (~3.57 GB)** |

### Label Mapping Verification
* All 4 checkpoints share an identical **48-label mapping** (`len(label2id) == 48`).
* Every integer ID $0 \le i \le 47$ corresponds to the exact same BIO label across GBERT and all three XLM-R seeds.
* Covers all 29 official semantic aspect types ($29 \times B\text{-} + 18 \times I\text{-} + 1 \times O = 48$).

---

## 4. Synthetic Submission Dry Run (Phase 3)

The pipeline was tested using invented synthetic examples in `scripts/verify_submission_synthetic.py` without using real competition titles or tokens:

1. **Single-token entity**: `["B-Produktart"]` -> `("Produktart", "Bremsscheibe")` (PASSED).
2. **Multi-token entity**: `["B-Produktart", "I-Produktart", "I-Produktart"]` -> `("Produktart", "Zahnriemen Satz Kit")` (PASSED).
3. **Multiple aspects within one title**: 3 distinct aspects extracted and written as separate rows (PASSED).
4. **Adjacent entities with same aspect name**: Consecutive `B-Im_Lieferumfang_Enthalten` tokens correctly produced separate rows (PASSED).
5. **Zero-entity titles**: Title tagged entirely as `O` produced 0 rows without failing validation (PASSED).
6. **Duplicate-looking distinct entities**: Repeated mentions of `Kompatible_Fahrzeug_Marke` preserved as distinct rows (PASSED).
7. **Non-ASCII German characters**: `ä`, `ö`, `ü`, `ß`, `Ø`, `€` written and parsed cleanly in UTF-8 (PASSED).
8. **Special punctuation and spacing**: `+`, `/`, `-`, `&`, `(`, `)` preserved accurately (PASSED).
9. **Invalid formatting rejection**: Accidental CSV quoting, irregular field counts, and consecutive spaces were caught and rejected by the validator (PASSED).
10. **Unauthorized aspect rejection**: Submission of `O` or non-standard aspect names was caught and rejected (PASSED).

---

## 5. Inference and GPU Resource Readiness

* **GPU Hardware**: NVIDIA GeForce GTX 1660 Ti (6.00 GB VRAM).
* **VRAM Safety**:
  * Holding all 4 models simultaneously in GPU memory would exceed 10 GB and trigger CUDA OOM.
  * **Strict Sequential Strategy**: The inference runner loads GBERT -> collects active logits -> deletes model and runs `torch.cuda.empty_cache()`, then repeats sequentially for XLM-R seeds 42, 7, and 21. Peak VRAM utilization remains under **2.4 GB**.
* **Probability Combination**:
  $$P_{\text{final}} = 0.40 \cdot P_{\text{GBERT}} + 0.60 \cdot \left(\frac{P_{\text{XLMR, 42}} + P_{\text{XLMR, 7}} + P_{\text{XLMR, 21}}}{3}\right)$$
  Verified mathematically: probabilities sum to $1.00000$ across all token positions.

---

## 6. Storage and Disk-Space Risks

* **Drive C: Status**:
  * Total Capacity: `464.04 GB`
  * Available Space: `7.14 GB`
* **Storage Consumption Analysis**:
  * Current 4 champion models: `3.57 GB`.
  * Retraining on full 5,000 labeled titles will generate 4 new full-data checkpoints:
    * 1 GBERT checkpoint: `~440 MB`
    * 3 XLM-R checkpoints: `3 x 1.1 GB = ~3.3 GB`
    * Total new storage required: `~3.74 GB`.
  * **Projected remaining disk space after full retraining**: `7.14 - 3.74 = 3.40 GB`.
* **Risk Mitigation Strategy**:
  1. Do not duplicate checkpoints across scratch directories.
  2. Intermediate epoch checkpoints must be deleted automatically, saving only the final 3-epoch full-data weights.
  3. Pre-create clean target directories (`checkpoints/*_fulldata_*`).

---

## 7. Full-Data Retraining Plan (5,000 Labeled Titles)

The proposed retraining plan utilizes **100% of available labeled data (5,000 titles)** to maximize vocabulary coverage and parameter quality for final competition submission:

### Hyperparameters (Locked from Champion Validation Runs)
* **Dataset**: `data/raw/Tagged_Titles_Train.tsv.gz` (5,000 titles, 56,812 tokens).
* **Train / Val Split**: None (100% used for training).
* **Batch Size**: `4` (train batch size), `AdamW` optimizer.
* **Learning Rate**: `2e-5`.
* **Epochs**: Fixed at **3 epochs** (identical to champion convergence point).
* **Tokenizer**: Standard HuggingFace tokenizers from `deepset/gbert-base` and `FacebookAI/xlm-roberta-base`.

### Execution Schedule & Target Checkpoints
1. **Model 1**: GBERT seed 42 -> `checkpoints/gbert_fulldata_seed42_best` (~15 min, 440 MB).
2. **Model 2**: XLM-R seed 42 -> `checkpoints/xlmr_fulldata_seed42_best` (~22 min, 1.1 GB).
3. **Model 3**: XLM-R seed 7 -> `checkpoints/xlmr_fulldata_seed7_best` (~22 min, 1.1 GB).
4. **Model 4**: XLM-R seed 21 -> `checkpoints/xlmr_fulldata_seed21_best` (~22 min, 1.1 GB).
* **Total Estimated Retraining Time**: **~81 minutes**.

> [!CAUTION]
> **Validation Integrity Notice**:
> The original Split-A validation checkpoints (`*_baseline_best`, `*_seed_7_best`, `*_seed_21_best`) will **not** be overwritten. Full-data checkpoints will use separate dedicated paths (`*_fulldata_*`).
> The Split-A validation score ($F_{0.2} = 0.879852$) cannot be used as an empirical validation score for full-data models because full-data models are trained on all 5,000 titles.

---

## 8. Unresolved Blockers and Risk Assessment

* **Critical Blockers**: **0**.
* **Advisory Warnings**:
  1. **Disk Headroom**: Free space on Drive C is 7.14 GB. Full-data retraining must save only the final weights per model.
  2. **Candidate Titles Loading**: In `data/raw/Listing_Titles.tsv.gz`, Quiz Data corresponds to record numbers `5,001` through `30,000`. When generating submission predictions, ensure only the exact candidate range is processed.

---

## 9. Recommended Next Action

The repository is fully verified and submission-ready. The recommended next actions (pending user approval) are:
1. **Option A (Zero Retraining / Direct Inference)**: Run the existing 4-model champion ensemble directly on the Quiz Data (`records 5,001..30,000`) and validate the output TSV with `scripts/validate_submission.py`.
2. **Option B (Full-Data Retraining)**: Execute the 81-minute sequential retraining of the 4 champion models on all 5,000 labeled titles, then run Quiz inference using the new full-data ensemble.
