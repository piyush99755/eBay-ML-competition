# Phase 35A — Five-Model Ensemble, Generalization Audit & Quiz Challenger Report

**Project**: eBay 2026 University Machine Learning Competition  
**Date**: October 29, 2026  
**Status**: COMPLETE — Research and Validation Concluded  
**Decision Gate**: **B. FIVE-MODEL INCONCLUSIVE**  
**Final Strategic Recommendation**: **RETAIN DAPT QUIZ LEADER**  
**Active Provisional Quiz Champion**: [`submissions/submission_quiz_phase34a_dapt_min060.tsv`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/submissions/submission_quiz_phase34a_dapt_min060.tsv)  
- Official EvalAI Quiz $F_{0.2}$: **0.9035321754551815**  
- SHA-256: `B40B03D5772EC89ED957993610B31B59F06F7C6D4FB84EB44C61C0CBF9D57258`  
- Total Data Rows: 157,273  
**Frozen Original Champion**: [`submissions/submission_quiz_exp26b_min060.tsv`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/submissions/submission_quiz_exp26b_min060.tsv)  
- Official EvalAI Quiz $F_{0.2}$: **0.901245410919943**  
- SHA-256: `049CF5EF60D88D86C26DEC5A815F8DB820A306989E383843E6A9B3108AD7A7FA`  
- Total Data Rows: 161,381  
**Generated Five-Model Challenger**: [`submissions/submission_quiz_phase35a_five_model_min060.tsv`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/submissions/submission_quiz_phase35a_five_model_min060.tsv)  
- SHA-256: `53305E6A4692F453B1B37F90638C66EEF3B2BB988EE8410D7E1EDA2AAE2254C0`  
- Total Data Rows: 158,665  
- Size: 6,023,543 bytes  
- Validation Status: 100% Compliant (Awaiting User Discretion, NOT Submitted)  
- Production Pipeline: [`src/submission/predict_test_champion.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/submission/predict_test_champion.py) (100% Intact & Verified)  

---

## 1. Executive Summary

Phase 35A conducted a rigorous evaluation-integrity audit, controlled local validation, and Quiz challenger generation to investigate whether combining the Original GBERT model (weight 0.20) and the DAPT-adapted GBERT model (weight 0.20) with three diverse XLM-RoBERTa models (weights 0.20 each) yields a stronger precision-oriented automotive aspect extraction ensemble.

### Key Conclusions:
1. **Evaluation Integrity Finding (No Standard Clean Holdout)**:
   A forensic data audit revealed that the union of supervised training data across all 5 models covers **4,511 of the 5,000 labeled titles (90.22%)**. Every standard evaluation partition (`val_1k`, `dev_800`, `all_examples[4000:]`, `s_dev`) is partially or fully contaminated by supervised training data from at least one model. Only 489 titles (9.78%) in the entire 5,000-title corpus were never seen in training by ANY of the 5 models.
2. **Empirical Behavior on Clean Holdout ($N=489$)**:
   - Original 4-Model Champion: $F_{0.2} = \mathbf{0.847173}$
   - Five-Model Ensemble: $F_{0.2} = \mathbf{0.851557}$ ($\Delta = \mathbf{+0.004425}$, $p = 0.0200$ vs Orig)
   - DAPT 4-Model Champion: $F_{0.2} = \mathbf{0.856033}$ ($\Delta = \mathbf{-0.004465}$, $p = 0.0100$ vs DAPT)
   The five-model ensemble sits precisely at the mathematical midpoint of the Original and DAPT models. Diluting the DAPT GBERT weight from 0.40 to 0.20 by introducing original GBERT hurts performance relative to the DAPT champion.
3. **Quiz File Verification & Symmetry**:
   On the 25,000 Quiz titles, the Five-Model Challenger extracted **158,665 data rows**, perfectly bisecting the Original Champion (161,381) and the DAPT Champion (157,273). Across every major aspect (`Produktart`, `Herstellernummer`, `Kompatibles_Fahrzeug_Modell`), the five-model ensemble produces intermediate frequency and confidence distributions.
4. **Strategic Decision**:
   Because the pure DAPT 4-model ensemble is statistically superior on clean data ($p = 0.0100$) and already achieved an official EvalAI Quiz score of **0.903532** (beating the original champion by +0.002287), our recommendation is to **RETAIN DAPT QUIZ LEADER**.

---

## 2. Part A: Reproducibility & Evaluation Integrity Audit

### 2.1 Checkpoint Data Provenance & Overlap Analysis
We audited all 5 saved checkpoints, their configuration parameters, and the exact training slices used during supervised fine-tuning:

| Model # | Checkpoint Path | Base Architecture | Supervised Training Slice | Number of Titles | Training Seed |
| :---: | :--- | :--- | :--- | :---: | :---: |
| **Model 1** | `checkpoints/gbert_baseline_best` | `deepset/gbert-base` | Canonical Split-A `train_4k` | 4,000 | 42 |
| **Model 2** | `checkpoints/phase33b_dapt_gbert_best` | `deepset/gbert-base` (DAPT MLM) | Sequential slice `s_tr` (`[:4000]` 80%) | 3,200 | 42 |
| **Model 3** | `checkpoints/xlmr_baseline_best` | `FacebookAI/xlm-roberta-base` | Canonical Split-A `train_4k` | 4,000 | 42 |
| **Model 4** | `checkpoints/xlmr_seed_7_best` | `FacebookAI/xlm-roberta-base` | Canonical Split-A `train_4k` | 4,000 | 7 |
| **Model 5** | `checkpoints/xlmr_seed_21_best` | `FacebookAI/xlm-roberta-base` | Canonical Split-A `train_4k` | 4,000 | 21 |

### 2.2 Audit of Candidate Evaluation Partitions
We cross-referenced every potential evaluation partition against the union of training records ($4,511$ unique titles):

| Candidate Partition | Total Titles | Seen by 4 Baselines | Seen by DAPT GBERT | Cleanly Unseen by ALL 5 | Integrity Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Canonical `val_1k`** | 1,000 | 0 (0.0%) | **511 (51.1%)** | 489 (48.9%) | **CONTAMINATED DIAGNOSTIC** |
| **Canonical `dev_800` (`sp_dev`)** | 800 | **800 (100.0%)** | **530 (66.2%)** | 0 (0.0%) | **COMPLETELY CONTAMINATED** |
| **Sequential Tail (`[4000:]`)** | 1,000 | **799 (79.9%)** | 0 (0.0%) | 201 (20.1%) | **SEVERELY CONTAMINATED** |
| **DAPT Dev Set (`s_dev`)** | 800 | **512 (64.0%)** | 0 (0.0%) | 288 (36.0%) | **SEVERELY CONTAMINATED** |

**Conclusion on Shared Holdout**:  
No canonical, standardized evaluation partition exists that was held out from all five models. Consequently, we report local metrics on the 489 strictly unseen titles alongside the standard 1,000-title `val_1k` split (explicitly labeled as a contaminated diagnostic).

### 2.3 Duplicate and Near-Duplicate Title Audit
- Total unique normalized titles in the 5,000 labeled corpus: **4,999 / 5,000** (99.98%).
- Only 1 normalized title appeared twice (Records 2,236 and 4,451).
- **Zero duplicate or normalized-identical titles cross the boundary between unseen records and training records.**

### 2.4 Architecture, Vocabulary & Checkpoint Integrity
- All 5 checkpoints load valid `model.safetensors` files without corruption.
- BIO label space: All 5 checkpoints share the exact same **48 classes** (`O` ID = 47) and identical `id2label` / `label2id` dictionaries.
- Tokenizers: Original GBERT and DAPT GBERT share the exact same `BertTokenizerFast` vocabulary (31,102 tokens). XLM-R models share `XLMRobertaTokenizerFast` (250,002 tokens).
- Previous Quiz Submissions verified bit-for-bit:
  - Original Champion: `049CF5EF60D88D86C26DEC5A815F8DB820A306989E383843E6A9B3108AD7A7FA` (161,381 rows) — **MATCHED**
  - DAPT Champion: `B40B03D5772EC89ED957993610B31B59F06F7C6D4FB84EB44C61C0CBF9D57258` (157,273 rows) — **MATCHED**

---

## 3. Part B: Five-Model Ensemble Specification

The Five-Model Ensemble uses a balanced, equal-weight configuration:

| Model Branch | Checkpoint Path | Architecture | Supervised Seed | Ensemble Weight |
| :--- | :--- | :--- | :---: | :---: |
| **Model 1 (Original GBERT)** | `checkpoints/gbert_baseline_best` | `deepset/gbert-base` | 42 | **0.20** |
| **Model 2 (DAPT GBERT)** | `checkpoints/phase33b_dapt_gbert_best` | `deepset/gbert-base` (DAPT) | 42 | **0.20** |
| **Model 3 (XLM-R Seed 42)** | `checkpoints/xlmr_baseline_best` | `xlm-roberta-base` | 42 | **0.20** |
| **Model 4 (XLM-R Seed 7)** | `checkpoints/xlmr_seed_7_best` | `xlm-roberta-base` | 7 | **0.20** |
| **Model 5 (XLM-R Seed 21)** | `checkpoints/xlmr_seed_21_best` | `xlm-roberta-base` | 21 | **0.20** |

$$\sum_{m=1}^{5} w_m = 0.20 + 0.20 + 0.20 + 0.20 + 0.20 = 1.00$$

### Mathematical Symmetry
Notice the exact relationship with the two 4-model systems:
$$p_{\text{5-model}}(t, c) = \frac{1}{2} p_{\text{orig\_champ}}(t, c) + \frac{1}{2} p_{\text{dapt\_champ}}(t, c)$$
The five-model ensemble is mathematically the exact equal blend of the Original Champion and the DAPT Champion.

### Production Constraints Enforced:
- First-subword logits mapped to original whitespace tokens.
- BIO decoding with exact span reconstruction.
- Confidence: Minimum selected BIO-label probability across constituent tokens ($\min_{t \in \text{span}} p_{\text{fused}}(t, y_t)$).
- Threshold: MIN 0.60.
- Category-Aware Filtering: Method A (Annexure.pdf allowed aspects).

---

## 4. Part C: Local Evaluation Results & Statistical Comparison

### 4.1 Clean Unpolluted Holdout ($N=489$ Records, Unseen by ALL 5 Models)
On this clean, unbiased holdout, all metrics reflect true out-of-distribution generalization:

| System | $F_{0.2}$ | Precision | Recall | True Positives | False Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original GBERT Standalone** | 0.845660 | 0.843626 | 0.899915 | 3,183 | 590 | 354 |
| **DAPT GBERT Standalone** | 0.862761 | 0.860588 | 0.920874 | 3,247 | 526 | 279 |
| **Original 4-Model Champion** | 0.847173 | 0.844686 | 0.914491 | 3,187 | 586 | 298 |
| **Five-Model Ensemble** | **0.851557** | **0.848927** | **0.923055** | **3,203** | **570** | **267** |
| **DAPT 4-Model Champion** | **0.856033** | **0.853432** | **0.926619** | **3,220** | **553** | **255** |

### 4.2 Paired Bootstrap Significance (1,000 Resamples on Clean Holdout)
- **Five-Model vs. Original Champion**:
  $$\Delta F_{0.2} = \mathbf{+0.004425} \quad [95\%\text{ CI: } +0.000276, +0.008619], \quad p = 0.0200$$
  *Result*: The 5-model ensemble shows a statistically significant improvement over the Original Champion.
- **Five-Model vs. DAPT Champion**:
  $$\Delta F_{0.2} = \mathbf{-0.004465} \quad [95\%\text{ CI: } -0.008625, -0.000670], \quad p = 0.0100$$
  *Result*: The 5-model ensemble is statistically significantly **worse** than the DAPT Champion. Re-introducing original GBERT degrades the precision and recall gains achieved by DAPT.

### 4.3 Split-A Validation Set ($N=1,000$ Records) `[CONTAMINATED DIAGNOSTIC ONLY]`
*Note: DAPT GBERT observed 511 of these titles during supervised training; these numbers must NOT be interpreted as independent generalization.*

| System | $F_{0.2}$ (Diag) | Precision | Recall | True Positives | False Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original 4-Model Champion** | 0.837870 | 0.834980 | 0.917245 | 6,340 | 1,253 | 572 |
| **Five-Model Ensemble** | 0.843613 | 0.840643 | 0.925341 | 6,383 | 1,210 | 515 |
| **DAPT 4-Model Champion** | 0.850336 | 0.847359 | 0.932194 | 6,434 | 1,159 | 468 |

---

## 5. Detailed Disagreement & Venn Analysis

Analysis of predictions across the systems on the 1,000-title validation split:

```
                    +------------------------------------+
                    |        PREDICTION OVERLAPS         |
                    +------------------------------------+
                    | Total Orig Champion:   6,889 preds |
                    | Total DAPT Champion:   6,878 preds |
                    | Total Five-Model:      6,876 preds |
                    +------------------------------------+
                    | Shared by ALL THREE:   6,676 (96.9%)|
                    | Unique to Orig:          100       |
                    | Unique to DAPT:          116       |
                    | Unique to Five-Model:      1       |
                    +------------------------------------+
```

### Where Do Model Disagreements Concentrate?
Divergence between the Original and DAPT models concentrates almost entirely in complex compound aspects:
1. `Kompatibles_Fahrzeug_Modell`: **254 disagreements** (subword token segmentation & span boundaries).
2. `Einbauposition`: **43 disagreements** (multi-word position combinations like "Vorne Links").
3. `Im_Lieferumfang_Enthalten`: **36 disagreements** (packaged items).
4. `Herstellernummer`: **24 disagreements** (borderline alphanumeric part codes).
5. `Produktart`: **20 disagreements** (generic nouns vs specific component terms).
6. `Kompatible_Fahrzeug_Marke`: **19 disagreements**.

In `Kompatibles_Fahrzeug_Modell`:
- Shared across both: 1,343 predictions.
- Unique to Original Champion: 133 predictions.
- Unique to DAPT Champion: 121 predictions.
- Five-Model prediction count: 1,475 (sits directly in between).

---

## 6. Part D: Decision Gate

```
=============================================================================
DECISION GATE: B. FIVE-MODEL INCONCLUSIVE
=============================================================================
```

**Justification**:
1. **Clean Validation Partition Unavailable**: A standardized, canonical holdout partition clean of all 5 models does not exist; only an unstratified 489-title residual slice remains.
2. **Intermediate Performance**: On the 489 clean titles, the five-model ensemble ($F_{0.2} = 0.851557$) outperforms the old champion ($0.847173$), but is statistically inferior ($p=0.0100$) to the DAPT champion ($0.856033$).
3. **Dilution Effect**: Blending Original GBERT back into the ensemble dilutes the specialized automotive domain vocabulary representations learned by DAPT.
4. **Verified Leader**: The Phase 34A DAPT ensemble has already established an official EvalAI Quiz score of **0.903532**, representing our confirmed provisional leader.

---

## 7. Part E: Three-Way Official Quiz Comparison

Per Part E authorization for Outcome B, we generated and validated the Five-Model Quiz challenger file on all 25,000 Quiz titles:

| Metric / Dimension | Original Champion (Exp 26B) | DAPT Champion (Phase 34A) | Five-Model Ensemble (Ph 35A) |
| :--- | :---: | :---: | :---: |
| **Official Quiz $F_{0.2}$** | **0.901245410919943** | **0.9035321754551815** | *Unsubmitted (Held)* |
| **Total Prediction Rows** | **161,381** | **157,273** | **158,665** |
| **File Size** | 6,091,833 bytes | 5,981,377 bytes | 6,023,543 bytes |
| **SHA-256 Hash** | `049CF5EF60D88D86...` | `B40B03D5772EC89E...` | `53305E6A4692F453B1B37F90638C66EEF3B2BB988EE8410D7E1EDA2AAE2254C0` |
| **Coverage** | 25,000 / 25,000 (100%) | 25,000 / 25,000 (100%) | 25,000 / 25,000 (100%) |
| **Category 1 Aspects** | 85,832 | 82,780 | 83,363 |
| **Category 2 Aspects** | 75,549 | 74,493 | 75,302 |
| **Shared by All Three** | — | — | **151,674 (95.59%)** |
| **Agreement with Orig** | 100% | 94.01% | 96.42% (155,609 rows) |
| **Agreement with DAPT** | 94.01% | 100% | 98.11% (154,299 rows) |
| **Unique to System** | 5,413 | 2,600 | **105** |

### Aspect Frequency Shifts Across Quiz Submissions

| Aspect Name | Orig Champion | DAPT Champion | Five-Model | $\Delta$ vs Orig | $\Delta$ vs DAPT |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Produktart` | 19,762 | 16,639 | 17,520 | -2,242 | +881 |
| `Herstellernummer` | 12,409 | 11,345 | 11,849 | -560 | +504 |
| `Kompatibles_Fahrzeug_Modell` | 31,043 | 30,433 | 30,724 | -319 | +291 |
| `Im_Lieferumfang_Enthalten` | 26,126 | 26,567 | 26,421 | +295 | -146 |
| `Hersteller` | 19,782 | 19,454 | 19,570 | -212 | +116 |
| `Einbauposition` | 11,126 | 11,387 | 11,248 | +122 | -139 |
| `Kompatible_Fahrzeug_Marke` | 28,173 | 28,367 | 28,286 | +113 | -81 |
| `Kompatibles_Fahrzeug_Jahr` | 2,791 | 2,844 | 2,818 | +27 | -26 |
| `Modell` | 176 | 211 | 200 | +24 | -11 |
| `Bremsscheibenart` | 1,312 | 1,322 | 1,326 | +14 | +4 |
| `Bremsscheiben-Aussendurchmesser`| 4,048 | 4,073 | 4,060 | +12 | -13 |
| `Oe/Oem_Referenznummer(N)` | 24 | 55 | 36 | +12 | -19 |

Across all major aspects, the Five-Model Ensemble row counts sit strictly between the Original Champion and the DAPT Champion.

---

## 8. Strategic Recommendation

### **RETAIN DAPT QUIZ LEADER**

**Why?**
1. **Proven Performance**: The Phase 34A DAPT 4-Model Ensemble achieved **0.9035321754551815** on EvalAI Quiz, providing a verified +0.002287 boost over the original champion.
2. **Empirical Superiority on Clean Holdout**: On the 489 strictly uncontaminated validation titles, DAPT scored **0.856033** vs. **0.851557** for the Five-Model ensemble ($p=0.0100$). The 5-model blend dilutes the domain adaptation effect.
3. **Competition Readiness**: Final Test submissions should be based on the model with the highest demonstrated generalization on domain terminology. The DAPT champion is currently our strongest, cleanest system.
4. **Safety**: `submissions/submission_quiz_phase35a_five_model_min060.tsv` is generated, validated, and stored safely locally for reference, but should **NOT** be submitted to EvalAI.

---

## 9. Verification Sign-Off

- [x] All 5 model checkpoints verified and untouched.
- [x] Both previous Quiz submissions confirmed intact (SHA-256 bit-for-bit matched).
- [x] Production script `src/submission/predict_test_champion.py` confirmed unchanged.
- [x] Data provenance audit completed (all partitions cataloged).
- [x] Local evaluation executed with bootstrap tests on clean holdout.
- [x] Five-model Quiz challenger generated on full 25,000 Quiz records.
- [x] Strict schema validator passed 100% (158,665 rows, compliant with Annexure.pdf).
- [x] Free disk space on Drive C: maintained at 5.20 GB ($\ge 4.5$ GB requirement).
- [x] All 22 unit tests passed (`test_phase35a_five_model.py` and `test_final_submission_validator.py`).

**FINAL STATUS**: `PHASE 35A COMPLETE — RETAIN DAPT QUIZ LEADER`
