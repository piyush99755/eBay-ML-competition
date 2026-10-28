# Phase 33C — DAPT Reproducibility Audit & Evaluation Integrity Report

**Project**: eBay 2026 University Machine Learning Competition  
**Date**: October 28, 2026  
**Status**: AUDIT COMPLETED — TRAINING HALTED  
**Mandatory Audit Decision**: `AUDIT FAILED — STOP`  
**Final Competition Strategy Decision**: `FULL-DATA DAPT FAILED — RETAIN CHAMPION`  
**Active Production Champion**: `submissions/submission_quiz_exp26b_min060.tsv`  
- Official EvalAI Quiz $F_{0.2}$: **0.901245410919943** (Rank 13)
- Original Split-A Validation $F_{0.2}$: **0.899802**
- Production Inference Pipeline: `src/submission/predict_test_champion.py` (Frozen, 100% Intact)
- Free Disk Space on Drive C: **7.99 GB** (strictly $\ge 4.5$ GB)

---

## 1. Executive Summary

Phase 33C conducted the mandatory evaluation-integrity and reproducibility audit of the Phase 33B Domain-Adaptive Pretraining (DAPT) experiment prior to any authorized full-data training.

### The Forensic Finding: Severe Partition Leakage in Phase 33B
The audit has uncovered a critical evaluation flaw that completely invalidates the purported "+0.0450" gain reported in Phase 33B:
1. **The Partition Slicing Bug**: In `src/modeling/train_phase33b_dapt_gbert.py` (lines 124–126), training examples were created via sequential array slicing:
   ```python
   split_a_train_examples = all_examples[:4000]
   train_examples, dev_examples, _ = get_internal_split(split_a_train_examples, random_state=42)
   ```
   instead of the project's canonical stratified function:
   ```python
   train_4k, val_1k = split_examples(all_examples, random_state=42)
   train_examples, dev_examples, _ = get_internal_split(train_4k, random_state=42)
   ```
2. **Massive Evaluation-Set Contamination**:
   Because `train_test_split(..., random_state=42)` samples randomly across the dataset, `all_examples[:4000]` contained **62.5% of the Split-A validation holdout** (`val_1k`) and **66.8% of the canonical internal dev partition** (`dev_800`).
   Consequently, the 3,200 training examples used in Phase 33B Stage C included:
   - **534 of the 800 titles (66.8%)** from the internal dev set used in Phase 33B Stage D evaluation.
   - **625 of the 1,000 titles (62.5%)** from the Split-A validation holdout.
3. **The Score Inflation Illusion**:
   When Phase 33B Stage D evaluated the DAPT model on `dev_800` and reported $F_{0.2} = \mathbf{0.911893}$, **the model was evaluating on its own training data for two-thirds of the evaluation split**. The apparent surge was an artifact of training-data memorization, not generalization.
4. **Honest Out-of-Sample Reality**:
   When DAPT GBERT is evaluated strictly on its true, uncontaminated validation holdouts:
   - On its true unseen 800 dev titles (`s_dev`), DAPT scores **0.864194**, trailing the matched baseline (**0.882806**) by $\mathbf{-0.018612}$.
   - On the true unseen 1,000 holdout titles (`all_examples[4000:5000]`), DAPT scores **0.869557**, trailing the matched baseline (**0.891887**) by $\mathbf{-0.022330}$.
   - Continued MLM pretraining softened boundary representations and caused lower exact-match precision on genuine unseen data.
5. **Ensemble Contamination**:
   The internal 4-model ensemble was doubly contaminated: the 3 frozen XLM-R models were trained on all 4,000 Split-A training titles (including 100% of `dev_800`), and DAPT GBERT had seen 66.8% of `dev_800`.
6. **Audit Action Taken**:
   Per the mandatory protocol, full-data retraining was **IMMEDIATELY HALTED**. The slicing bug was corrected in research code, no official files were modified, and the frozen champion was firmly retained.

```
=============================================================================
AUDIT DECISION: AUDIT FAILED — STOP AND INVESTIGATE
FINAL ACTION:   FULL-DATA DAPT FAILED — RETAIN CHAMPION
=============================================================================
```

---

## 2. Stage 33B Audit Findings

The audit systematically investigated the 14 integrity checks stipulated in the mission:

| Check # | Audit Item | Verification Status | Forensic Detail |
| :---: | :--- | :---: | :--- |
| **1** | **Same 800-title internal dev partition** | **FAILED** | Stage C evaluated on `s_dev` (derived from `all_examples[:4000]`). Stage D evaluated on `sp_dev` (derived from `split_examples`). Overlap was only 120 / 800 titles (15.0%). |
| **2** | **Same gold annotations** | **PASSED** | Identical raw token annotations from `train_data.tsv`. |
| **3** | **Same aspect vocabulary** | **PASSED** | Exact 48 BIO classes across both evaluations. |
| **4** | **Same category assignments** | **PASSED** | Category 1 and 2 integer mappings matched. |
| **5** | **Same saved checkpoint** | **PASSED** | Both loaded `checkpoints/phase33b_dapt_gbert_best`. |
| **6** | **Identical entity reconstruction** | **PASSED** | Both use standard `bio_to_entities`. |
| **7** | **Consistent BIO decoding** | **PASSED** | Argmax over active logits; no heuristic shifts. |
| **8** | **Correct $F_{0.2}$ formula** | **PASSED** | Official competition formula ($\beta=0.2$, precision penalty 25). |
| **9** | **Matching by record, aspect, value** | **PASSED** | Exact string matching verified via `count_matches`. |
| **10** | **Duplicate prediction protection** | **PASSED** | `Counter` intersection enforces exact frequency matching. |
| **11** | **Gold entities completeness** | **PASSED** | All true entities evaluated; zero omissions. |
| **12** | **Category filtering applied correctly** | **PASSED** | Applied properly, but removes only 1 entity (explaining $<0.000001$ $F_{0.2}$). |
| **13** | **No label / test leakage** | **FAILED (CRITICAL)** | **534 / 800 dev titles (66.8%)** and **625 / 1,000 val titles (62.5%)** were present in DAPT's training set `s_tr`! |
| **14** | **No self-comparison** | **PASSED** | Gold targets and predictions are distinct objects. |

---

## 3. Explanation of Stage C vs Stage D Score Discrepancy

Phase 33B reported two conflicting metrics for the same checkpoint:
- Stage C Training History: Dev $F_{0.2} = \mathbf{0.864194}$
- Stage D Internal Evaluation: Dev $F_{0.2} = \mathbf{0.911893}$

### What Caused the 0.0477 Discrepancy?
It was initially hypothesized that category post-filtering (Method A) caused the jump from 0.864 to 0.911. **Our empirical audit disproved this hypothesis completely**:
- When category post-filtering is applied to Stage C predictions on `phase33b_dapt_gbert_best`, only **1 single prediction** is removed (`Bremsscheiben-Aussendurchmesser`: 1).
- The raw score before filtering is **0.911893**, and after filtering is **0.911893**.

### The Real Cause: Split Desynchronization
1. During Stage C training, `train_phase33b_dapt_gbert.py` evaluated the model at the end of Epoch 3 on its own internal partition `s_dev` (which had zero overlap with its 3,200 training examples). On that clean, out-of-sample data, DAPT achieved **0.864194**.
2. In Stage D, `phase_33b_evaluate_dapt.py` evaluated the checkpoint on `sp_dev` (the canonical split from `split_examples`). Because of the slicing bug, **534 of those 800 titles (66.8%) were in the model's training set**.
3. The jump from 0.864194 to 0.911893 was **pure training-set memorization**.

```
+-----------------------------------------------------------------------------------+
|               STAGE C VS STAGE D SCORE DISCREPANCY FORENSIC AUDIT                |
+------------------------------------+-----------+------------+--------------------+
| Evaluation Condition               | Split     | Dev F0.2   | Leakage Status     |
+------------------------------------+-----------+------------+--------------------+
| Stage C Epoch 3 (In-Training)      | s_dev     | 0.864194   | 0% Leakage (Clean) |
| Stage D Evaluation (Phase 33B)     | sp_dev    | 0.911893   | 66.8% Leakage      |
| Clean True Dev Holdout             | s_dev     | 0.864194   | 0% Leakage (Clean) |
| Clean True Unseen Holdout (4k-5k)  | unseen_1k | 0.869557   | 0% Leakage (Clean) |
+------------------------------------+-----------+------------+--------------------+
```

---

## 4. Ensemble Contamination Analysis

The audit evaluated the fairness and validity of the 4-model ensemble on internal partitions:

### 1. XLM-R Branch Contamination
- The three frozen XLM-R models (`xlmr_baseline_best`, `xlmr_seed_7_best`, `xlmr_seed_21_best`) were trained on all 4,000 Split-A training titles.
- The 800-title internal dev partition (`sp_dev`) is an internal 20% subset of those 4,000 titles.
- **Result**: The three XLM-R models (which constitute 60% of the ensemble voting weight) have seen **100% of the internal dev titles** during their supervised fine-tuning.
- Therefore, the baseline ensemble score of **0.919551** was largely an artifact of in-sample training data evaluation on the XLM-R branches.

### 2. DAPT GBERT Branch Contamination
- In Phase 33B, DAPT GBERT was trained on `s_tr` (3,200 titles).
- As proven above, `s_tr` contained 534 of the 800 `sp_dev` titles (66.8%).
- In the "Challenger Ensemble", **all four models had seen the internal dev titles**: 3 models saw 100% of the titles, and 1 model saw 66.8% of the titles.

### 3. Reclassification of Results
Under Section 4 guidelines, **all internal 4-model ensemble scores on `dev_800` are formally reclassified as contaminated in-sample diagnostic checks**. They provide zero statistical evidence of out-of-distribution generalization.

---

## 5. Reproduced Phase 33B Metrics

Using the existing saved checkpoint `checkpoints/phase33b_dapt_gbert_best`, we independently reproduced the exact metrics reported in Phase 33B:

| Partition / Metric | Reported in Phase 33B | Re-evaluated in Audit | Audit Match |
| :--- | :---: | :---: | :---: |
| **Internal Dev DAPT Standalone $F_{0.2}$** | 0.911893 | 0.911893 | **EXACT MATCH** |
| **Internal Dev Baseline Standalone $F_{0.2}$** | 0.866940 | 0.866940 | **EXACT MATCH** |
| **Internal Dev DAPT Ensemble $F_{0.2}$** | 0.937209 | 0.937209 | **EXACT MATCH** |
| **Internal Dev Baseline Ensemble $F_{0.2}$** | 0.919551 | 0.919551 | **EXACT MATCH** |
| **Split-A Val DAPT Standalone $F_{0.2}$** | 0.901654 | 0.901654 | **EXACT MATCH** |
| **Split-A Val Baseline Standalone $F_{0.2}$** | 0.851460 | 0.851460 | **EXACT MATCH** |
| **Split-A Val DAPT Ensemble $F_{0.2}$** | 0.917130 | 0.917130 | **EXACT MATCH** |
| **Split-A Val Baseline Ensemble $F_{0.2}$** | 0.897179 | 0.897179 | **EXACT MATCH** |

The code execution and arithmetic in Phase 33B were perfectly reproducible. However, the data partitions fed into the script suffered from the catastrophic slicing bug documented in Section 1.

---

## 6. Full-Data Training Configuration (Why It Was Halted)

Under Section 6:
> *"Only proceed to Part B if the audit passes. Do not rationalize discrepancies or silently change evaluation definitions to produce higher scores. If there is an evaluation bug, fix it in research-only evaluation code, recompute all Phase 33B metrics, and stop before retraining."*

Because the audit conclusively proved that:
1. The Phase 33B Challenger's apparent superiority was caused by data leakage, and
2. On truly clean, out-of-sample data, DAPT GBERT trails the matched baseline ($0.864194$ vs. $0.882806$),

**Full-data supervised retraining was NOT authorized and was safely halted**. No GPU cycles or storage space were wasted training an unviable architecture on the 4,000-title dataset.

---

## 7. True Uncontaminated Standalone Comparison

Here is the true, clean head-to-head comparison between Baseline GBERT and DAPT GBERT on genuinely unseen data:

### On True Unseen Dev Split `s_dev` (800 Titles):
| Metric | Matched Baseline GBERT | DAPT Challenger GBERT | Absolute Delta |
| :--- | :---: | :---: | :---: |
| **$F_{0.2}$ Score** | **0.882806** | 0.864194 | $\mathbf{-0.018612}$ |
| **Precision** | **0.883865** | 0.867484 | $-0.016381$ |
| **Recall** | **0.882098** | 0.860117 | $-0.021981$ |
| **True Positives (TP)** | **5,297** | 5,165 | $-132$ |
| **False Positives (FP)** | **696** | 789 | $+93$ (Worse) |
| **False Negatives (FN)** | **708** | 840 | $+132$ (Worse) |

### On True Unseen Holdout Records 4000..5000 (1,000 Titles):
| Metric | Matched Baseline GBERT | DAPT Challenger GBERT | Absolute Delta |
| :--- | :---: | :---: | :---: |
| **$F_{0.2}$ Score** | **0.891887** | 0.869557 | $\mathbf{-0.022330}$ |
| **Precision** | **0.892110** | 0.871225 | $-0.020885$ |
| **Recall** | **0.886873** | 0.858324 | $-0.028549$ |
| **True Positives (TP)** | **6,648** | 6,434 | $-214$ |
| **False Positives (FP)** | **804** | 951 | $+147$ (Worse) |
| **False Negatives (FN)** | **848** | 1,062 | $+214$ (Worse) |

**Conclusion**: On genuine out-of-sample data, DAPT GBERT consistently produces lower precision ($-0.016$ to $-0.021$) and lower recall ($-0.022$ to $-0.028$), generating **93 to 147 more false positives** and missing **132 to 214 true entities**.

---

## 8. Why Continued MLM Pretraining Regressed Supervised NER

Why did self-supervised pretraining on automotive titles fail to translate into downstream NER gains?
1. **Loss of Discriminative Sharpness**: Standard `gbert-base` was pretrained on millions of structured German Wikipedia and news texts, establishing sharp, high-entropy representation boundaries between standard German grammatical categories.
2. **Grammar Erosion on Title Shorthand**: eBay listing titles are ungrammatical keyword strings (e.g. *"2x Bremsscheibe bel. vorne BMW E46 320d Bj 01-05"*). Two epochs of MLM on 50,000 ungrammatical titles biased the self-attention heads toward loose keyword co-occurrence rather than structural boundary parsing.
3. **Softening of Continuation Token Logits**: In token-level BIO classification, continuation tags (`I-Modell`) require crisp discrimination against beginning tags (`B-Modell`) and outside tokens (`O`). The adapted encoder blurred these transitions, causing 93–147 additional false positive boundary mistakes.

---

## 9. Scientific Limitations & Methodological Lessons

1. **Array Slicing is an Anti-Pattern for Stratified Datasets**: Using `all_examples[:4000]` instead of an explicit partitioning function (`split_examples`) broke the cardinal rule of experimental design and introduced severe cross-fold leakage.
2. **Never Trust an Anomaly Without Verifying Splits**: An overnight jump of $+0.0450$ in a highly mature competition pipeline should always be treated as data leakage until proven otherwise.
3. **Diagnostic Ensembles Require Component-Level Split Auditing**: Even if an ensemble evaluation code is syntactically sound, if individual component models were trained on data intersecting the evaluation split, the ensemble score is invalid.

---

## 10. Hardware, GPU and Disk Telemetry

All operations maintained strict system safety:
- **Host GPU**: NVIDIA GeForce GTX 1660 Ti (6 GB VRAM, 0 MB memory leak).
- **Free Disk Space on Drive C:**: **7.99 GB** (strict floor $\ge 4.5$ GB maintained at all times).
- **Automated Test Suite**: **43 / 43 tests passing** in 24.91s (`Ran 43 tests - OK`).
- **Production Pipeline**: `src/submission/predict_test_champion.py` completely untouched.
- **Champion Checkpoints**: All 4 models verified bit-for-bit intact via SHA-256 hashes:
  - `checkpoints/gbert_baseline_best` (`4d909833701cdb20...`)
  - `checkpoints/xlmr_baseline_best` (`e1933989c5625e78...`)
  - `checkpoints/xlmr_seed_7_best` (`49437dff72fa00da...`)
  - `checkpoints/xlmr_seed_21_best` (`638baa567ece3ae9...`)

---

## 11. Debugging and Fixes Applied

1. **Fixed `src/modeling/train_phase33b_dapt_gbert.py`**:
   Replaced sequential slicing (`all_examples[:4000]`) with canonical stratified splitting (`split_examples(all_examples, random_state=42)`), guaranteeing that future experiments using this script cannot accidentally leak validation data.
2. **Created Diagnostic Scripts**:
   - `scratch/audit_discrepancy.py`: Re-evaluates both Stage C and Stage D logic with and without category filtering.
   - `scratch/audit_true_metrics.py`: Evaluates DAPT and Baseline models strictly on genuine out-of-sample data.

---

## 12. Final Decision and Strategic Recommendation

```
=============================================================================
FINAL DECISION: FULL-DATA DAPT FAILED — RETAIN CHAMPION
=============================================================================
```

### Strategic Recommendation for Test Window (November 1–6, 2026):
1. **Discard DAPT**: Domain-Adaptive Pretraining has been rigorously audited and proven to regress true out-of-sample precision and $F_{0.2}$. Do not train or deploy full-data DAPT models.
2. **Firmly Retain Active Champion**:
   - Model: Experiment 10 (GBERT 0.40 + 3x XLM-R 0.20) with Phase 26B MIN 0.60 filtering.
   - Proven Local Validation $F_{0.2}$: **0.899802**
   - Official EvalAI Quiz $F_{0.2}$: **0.901245410919943** (Rank 13)
   - Production Pipeline: `src/submission/predict_test_champion.py`
3. **Lock Down Submission Readiness**: Focus all remaining effort on ensuring flawless execution, file validation, and format adherence for when the official hidden Test set is released.
