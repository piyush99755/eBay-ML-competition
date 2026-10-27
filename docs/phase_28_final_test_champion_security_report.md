# Phase 28: Secure Official Quiz Champion for Final Test

**Date**: October 2026  
**Status**: **VERIFICATION COMPLETE & SECURED**  
**Pipeline**: `src/submission/predict_test_champion.py`  
**Active Official Champion**: Exp 10 Ensemble with MIN 0.60 Confidence Threshold  
- **Official EvalAI Quiz $F_{0.2}$**: **0.901245410919943**  
- **Official Quiz Submission**: `submissions/submission_quiz_exp26b_min060.tsv`  
- **Verified SHA-256**: `049CF5EF60D88D86C26DEC5A815F8DB820A306989E383843E6A9B3108AD7A7FA`  
- **Unfiltered Baseline Quiz $F_{0.2}$**: **0.8824567479388766**  
- **Baseline SHA-256**: `51F880C8A6DEBB4EAF1A83594ECA2D4D2D4A92201D6CB64F9F63EAE913B65115`  

---

## 1. Executive Summary

Phase 28 establishes a fully secured, reproducible, and configurable final inference pipeline for the eBay 2026 Machine Learning Competition.

1. **Configurable Pipeline**:
   The final inference pipeline (`src/submission/predict_test_champion.py`) was enhanced with a configurable `--min-confidence` parameter defaulting to `0.60` (our official Quiz champion), while `--min-confidence 0.00` cleanly restores the unfiltered Exp 10 baseline.
2. **Dual Bit-for-Bit & Byte-for-Byte Reproductions**:
   Both the official MIN 0.60 champion (161,381 rows) and the original unfiltered Exp 10 baseline (186,164 rows) were reproduced from scratch across the complete 25,000 Quiz candidate records (5,001..30,000). Both generated files matched their reference files **100% byte-for-byte** and produced **identical SHA-256 hashes**.
3. **Annexure.pdf & Validation Compliance**:
   Both generated files passed 100% of checks in the standalone competition validator (`scripts/validate_submission.py`), with zero format violations, zero malformed records, and strict adherence to Category 1 and Category 2 aspect sets.
4. **Frozen Assets Preserved**:
   All 4 validation-trained champion checkpoints (`checkpoints/gbert_baseline_best`, `checkpoints/xlmr_baseline_best`, `checkpoints/xlmr_seed_7_best`, `checkpoints/xlmr_seed_21_best`), exploration checkpoints, raw datasets, caches, and submission files remain completely intact.

---

## 2. Dual SHA-256 & Byte-for-Byte Reproduction Audit

The complete reproduction suite was executed via:
```bash
python src/submission/predict_test_champion.py --verify-all-quiz --clean-cache
```

| Verification Dimension | Part 1: Official Champion (MIN 0.60) | Part 2: Unfiltered Baseline (0.00) |
| :--- | :---: | :---: |
| **Output Path** | `submissions/submission_quiz_exp28_reproduced_min060.tsv` | `submissions/submission_quiz_exp28_reproduced_unfiltered.tsv` |
| **Target Reference File** | `submissions/submission_quiz_exp26b_min060.tsv` | `submissions/submission_quiz_exp10_baseline_corrected.tsv` |
| **Expected SHA-256** | `049CF5EF60D88D86C26DEC5A815F8DB820A306989E383843E6A9B3108AD7A7FA` | `51F880C8A6DEBB4EAF1A83594ECA2D4D2D4A92201D6CB64F9F63EAE913B65115` |
| **Observed SHA-256** | `049CF5EF60D88D86C26DEC5A815F8DB820A306989E383843E6A9B3108AD7A7FA` | `51F880C8A6DEBB4EAF1A83594ECA2D4D2D4A92201D6CB64F9F63EAE913B65115` |
| **SHA-256 Status** | **EXACT MATCH (100% Identical)** | **EXACT MATCH (100% Identical)** |
| **File Size (Bytes)** | **6,091,833 bytes** | **7,009,806 bytes** |
| **Byte-for-Byte Comparison** | **100% Byte-for-Byte Match** | **100% Byte-for-Byte Match** |
| **Data Rows** | 161,381 | 186,164 |
| **Titles Covered** | 24,958 (99.83%) | 25,000 (100.00%) |
| **Zero-Entity Titles** | 42 (0.17%) | 0 (0.00%) |
| **Category 1 Rows** | 85,832 | 101,843 |
| **Category 2 Rows** | 75,549 | 84,321 |
| **Official Validator Status** | **PASSED (100% Compliant)** | **PASSED (100% Compliant)** |

---

## 3. Pipeline Invariance & Compliance Checks

1. **Title Sequencing & Zero-Entity Processing**:
   - The candidate loading and inference loops iterate over every listing title strictly in sequential record order.
   - When a title produces zero entities (e.g. 42 titles in MIN 0.60 mode, and synthetic record 10010), it is not skipped or dropped; the pipeline correctly advances to the next title without altering row ordering or omitting record bookkeeping.
2. **Category-Aware Post-Filtering (Method A)**:
   - Reconstructed entity aspects are filtered per listing category according to Annexure.pdf pp. 11-14:
     - Category 1: 23 permitted aspects.
     - Category 2: 20 permitted aspects.
   - Forbidden aspects are filtered prior to confidence thresholding.
3. **Confidence Calculation Semantics**:
   - For every reconstructed entity spanning original tokens $[t_{\text{start}} .. t_{\text{end}}]$, each token's selected-label probability is extracted from the fused ensemble distribution:
     $$p_{\text{token}} = P_{\text{ensemble}}(t, \text{pred\_label}_t)$$
   - The entity minimum confidence is:
     $$\text{conf}_{\min} = \min_{t \in \text{entity}} p_{\text{token}}$$
   - If $\text{conf}_{\min} < \text{threshold}$, the entity is pruned.
4. **Annexure.pdf Formatting Specifications**:
   - Strict TSV: tab character `\t` delimiter, UNIX line endings `\n`.
   - Literal text: `csv.QUOTE_NONE`, zero quote characters injected.
   - Encoding: UTF-8.
   - Header: Exactly `Record Number\tCategory\tAspect Name\tAspect Value`.
   - No `O` (Obscure) tags ever emitted as aspect names.

---

## 4. Test Suite & Validation Results

| Test Suite / Script | Scope | Result | Details |
| :--- | :--- | :---: | :--- |
| `tests/test_confidence_filtering.py` | Unit tests for confidence math, threshold boundaries, 4-tuple category filtering, and bio_to_entities equivalence | **10 / 10 PASSED** | All unit tests passed in 0.001s. |
| `tests/test_experiment_23a_decoding.py` | Category masking, multi-token boundary continuity, and invalid aspect counter | **6 / 6 PASSED** | All synthetic decoding tests passed. |
| `predict_test_champion.py --dry-run` | Synthetic end-to-end dry-run (10 titles, 1 zero-entity record, unfiltered + filtered modes) | **PASSED (100%)** | Verified 39 unfiltered rows and 37 filtered rows; validator compliant. |
| `predict_test_champion.py --verify-all-quiz` | Full 25,000-title Quiz reproduction for MIN 0.60 and unfiltered modes | **PASSED (100%)** | Both SHA-256 hashes matched expected values bit-for-bit and byte-for-byte. |

---

## 5. Hardware Performance & Resource Usage

- **GPU**: NVIDIA GeForce GTX 1660 Ti (6.00 GB GDDR6 VRAM, Compute Capability 7.5).
- **Peak VRAM Consumption**: ~3.5 GB (safe 2.5 GB buffer remaining during batched inference with batch size 32).
- **Inference Throughput**:
  - GBERT: ~79.3 titles/sec (total 317.6s for 25k titles)
  - XLM-R seed 42: ~74.2 titles/sec (total 341.1s for 25k titles)
  - XLM-R seed 7: ~62.1 titles/sec (total 407.3s for 25k titles)
  - XLM-R seed 21: ~70.2 titles/sec (total 363.2s for 25k titles)
- **Probability Fusion & TSV Writing**: ~20.9s for 25,000 titles.
- **Disk Free Space**: **6.87 GB free** on Drive C:. Intermediate probability caches (~206 MB) were cleaned automatically with `--clean-cache`.

---

## 6. Final Test Phase Readiness & Risk Assessment

| Risk Factor | Assessment | Mitigation in Pipeline |
| :--- | :---: | :--- |
| **Unseen Record IDs** | LOW | `--start-record`, `--end-record`, or `--record-ids-file` allow arbitrary record ID specifications. |
| **File Format Variations** | LOW | Pipeline accepts either compressed `.tsv.gz` or uncompressed `.tsv` seamlessly. |
| **Memory Exhaustion (OOM)** | LOW | Sequential loading with explicit `del model` and `torch.cuda.empty_cache()` guarantees VRAM remains $< 3.5$ GB. |
| **Confidence Threshold Shift** | MINIMAL | `--min-confidence` is fully configurable at CLI invocation (default `0.60`, fallback `0.00`). |
| **Disk Exhaustion** | LOW | Probability caching is optional (`--cache-dir` with `--clean-cache`); TSV stream writes direct to disk. |

---

## 7. Final Test Invocation Commands Reference

When the official Test dataset instructions become available:

1. **Synthetic Dry Run** (fast pre-flight check):
   ```bash
   python src/submission/predict_test_champion.py --dry-run
   ```

2. **Official Champion Test Inference** (Default: MIN 0.60 confidence):
   ```bash
   python src/submission/predict_test_champion.py \
       --input-file data/raw/Test_Titles.tsv.gz \
       --output-file submissions/submission_test_champion_min060.tsv \
       --clean-cache
   ```

3. **Fallback Unfiltered Test Inference** (Raw Exp 10 ensemble):
   ```bash
   python src/submission/predict_test_champion.py \
       --input-file data/raw/Test_Titles.tsv.gz \
       --output-file submissions/submission_test_champion_unfiltered.tsv \
       --min-confidence 0.00 \
       --clean-cache
   ```
