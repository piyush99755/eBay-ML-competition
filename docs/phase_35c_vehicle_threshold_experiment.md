# Phase 35C: Precision-Optimized Vehicle Model Thresholding Experiment Report

**Date:** October 29, 2026  
**Status:** COMPLETED & VERIFIED  
**Auditor / System:** Antigravity (DeepMind Advanced Agentic Coding)  
**Workspace:** `C:\Users\piyus\Downloads\eBay-ML-competition`  
**Current Official Quiz Leader:** Phase 34A DAPT Four-Model Ensemble ($F_{0.2} = \mathbf{0.9035321754551815}$)  
**Challenger Submission:** `submissions/submission_quiz_phase35c_vehicle080.tsv`  
**Challenger SHA-256:** `4B591EF6A9072C9319FF1C0134DFC9736E5A436431ED512FE47B980ECCC01382`  
**File Size:** 5,537,316 bytes (147,794 prediction rows)  
**Associated Unit Tests:** [`tests/test_phase_35c_vehicle_threshold.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/tests/test_phase_35c_vehicle_threshold.py) (All 3 passed; full test suite 64/64 passed)  
**Drive C: Free Space:** 4.95 GB ($\ge 4.5$ GB maintained)

---

## Executive Summary

Phase 35C executed a targeted, zero-retraining inference optimization to address the primary failure mode identified in Phase 35B: disproportionate False Positives in vehicle model extraction (`Kompatibles_Fahrzeug_Modell`).

By elevating the entity confidence filtering threshold specifically for `Kompatibles_Fahrzeug_Modell` from **MIN 0.60 to MIN 0.80**, while preserving all other 28 aspects at the verified champion threshold of **MIN 0.60**, the challenger selectively prunes low-confidence, boundary-ambiguous vehicle models.

### Key Results
1. **Mandatory Subset Invariant 100% Satisfied:**
   - Every single prediction in Phase 35C exists in the Phase 34A DAPT submission (`submission_quiz_phase34a_dapt_min060.tsv`).
   - Exactly **0 new predictions** were added.
   - Exactly **9,479 vehicle-model predictions** were removed across the 25,000 Quiz titles.
   - Exactly **0 non-vehicle predictions** were altered or removed across all other 18 extracted aspects.
2. **Baseline Exact Match Verification:**
   When evaluated at MIN 0.60 across all aspects, the Phase 35C inference pipeline reproduced the 157,273 rows of the Phase 34A leader with 100% row-for-row identity, confirming checkpoint and pipeline equivalence.
3. **Official Submission Compliance:**
   Validated via [`scripts/validate_submission.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/scripts/validate_submission.py) with zero warnings or errors.

---

## Part A — Preflight Integrity and System Verification

Prior to inference, all assets were verified byte-exact:

### 1. Model Checkpoints
All four safetensors files match their verified SHA-256 signatures:
- `dapt_gbert` (`checkpoints/phase33b_dapt_gbert_best/model.safetensors`): `446637E3BB3B1CA09FC998290C5576619A0E68E26F1503BD9F8ADC20277DF0E0`
- `xlmr_seed42` (`checkpoints/xlmr_baseline_best/model.safetensors`): `E1933989C5625E78D63D27CFEE8781D311DBF01D9D24E122A97ADA97EEC67EB3`
- `xlmr_seed7` (`checkpoints/xlmr_seed_7_best/model.safetensors`): `49437DFF72FA00DAB4E5C3E045AD2C21D76DB4F4207E0293E1898AAE35998170`
- `xlmr_seed21` (`checkpoints/xlmr_seed_21_best/model.safetensors`): `638BAA567ECE3AE9B363FE0869454F300FE2FF1CB20E0B57A8B1D9C838BC5338`

### 2. Protected Previous Submissions
- Original Champion Quiz: `submissions/submission_quiz_exp26b_min060.tsv`  
  SHA-256: `049CF5EF60D88D86C26DEC5A815F8DB820A306989E383843E6A9B3108AD7A7FA` (Intact)
- Phase 34A DAPT Quiz Leader: `submissions/submission_quiz_phase34a_dapt_min060.tsv`  
  SHA-256: `B40B03D5772EC89ED957993610B31B59F06F7C6D4FB84EB44C61C0CBF9D57258` (Intact)

---

## Part B — Hypothesis & Fixed Threshold Configuration

### 1. Scientific Hypothesis
In the competition metric:
$$F_{0.2} = (1 + 0.2^2) \frac{\text{Precision} \cdot \text{Recall}}{0.2^2 \text{Precision} + \text{Recall}} = \frac{1.04 \cdot P \cdot R}{0.04 P + R}$$
Precision is weighted approximately $25\times$ more heavily than Recall ($\beta^2 = 0.04$).

Phase 35B precision error analysis revealed:
- `Kompatibles_Fahrzeug_Modell` accounts for **130 out of 255 (50.98%)** of all False Positives on the 489-title holdout.
- **96.15% (125/130)** of these False Positives are boundary mismatches (e.g., chassis codes, trim identifiers, and engine tags like `"Vectra B"` vs. `"Vectra"`).
- Pruning vehicle model candidates with confidence between $0.60 \le c < 0.80$ eliminates borderline boundary errors with high precision selectivity.

### 2. Predetermined Aspect-Specific Configuration
- **Aspect Threshold Filter:**
  $$\text{Threshold}(\text{aspect}) = \begin{cases} 0.80 & \text{if aspect} = \text{"Kompatibles\_Fahrzeug\_Modell"} \\ 0.60 & \text{otherwise} \end{cases}$$
- **Entity Confidence Definition:** Minimum selected BIO-label probability across constituent tokens:
  $$\text{EntityConfidence}(E) = \min_{t \in E} P(y_t = \hat{y}_t)$$
- **Ensemble Fusion Weights:**
  - DAPT GBERT: 0.40
  - XLM-R seed 42: 0.20
  - XLM-R seed 7: 0.20
  - XLM-R seed 21: 0.20
- **Decoding & Alignment:** Original-token alignment, BIO sequence decoding, category-aware filtering per Annexure.pdf (Category 1: 23 aspects, Category 2: 20 aspects).

---

## Part C — Local Diagnostic Verification & Acknowledgment of Selection Bias

### 1. Local Holdout Metrics (489 Clean Titles)
Using the corrected Phase 35B evaluator, holdout metrics were independently reproduced:

| System Configuration | Holdout F0.2 | Holdout Precision | Holdout Recall | TP | FP | FN | $\Delta F_{0.2}$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Phase 34A DAPT (Global MIN 0.60) | 0.923573 | 0.926619 | 0.853432 | 3,220 | 255 | 553 | Baseline |
| **Phase 35C Challenger (Vehicle 0.80 / Other 0.60)** | **0.933651** | **0.939162** | **0.814206** | **3,072** | **199** | **701** | **+0.010079** |

On the holdout:
- False Positives dropped by **56** (255 $\rightarrow$ 199, a 22.0% FP reduction).
- Precision increased from **92.66% to 93.92%** (+1.25%).
- True Positives dropped by 148 (3,220 $\rightarrow$ 3,072).
- Net $F_{0.2}$ increased by **$+0.010079$** due to the $25\times$ precision penalty on false alarms.

### 2. Explicit Acknowledgment of Known Selection Bias
> [!IMPORTANT]
> The 0.80 threshold on `Kompatibles_Fahrzeug_Modell` was selected based on exploratory analysis of the 489-title holdout in Phase 35B. Consequently, the local holdout gain of $+0.010079$ reflects development-set optimization and **cannot be cited as unbiased or independent proof of generalization**. The official EvalAI Quiz evaluation serves as the true unbiased external test of this threshold modification.

---

## Part D — Quiz Challenger Submission Analysis

### 1. Mandatory Subset Invariant Verification
Because Phase 35C only increases the threshold on one aspect, the set of predictions must be a strict mathematical subset of Phase 34A:
$$\mathcal{P}_{\text{Phase 35C}} \subset \mathcal{P}_{\text{Phase 34A}}$$

```
================================================================================
MANDATORY SUBSET INVARIANT VERIFICATION REPORT
================================================================================
Phase 34A DAPT Leader Total Predictions:  157,273
Phase 35C Challenger Total Predictions:    147,794
Total Predictions Removed:                  9,479
Total Predictions Added:                        0 (EXACTLY ZERO)
Subset Invariant Status:                    PASS (100% Verified)

Removed Predictions by Aspect:
  Kompatibles_Fahrzeug_Modell:              9,479 (100.0% of removed)
  All Other 18 Aspects:                         0 (0.0% removed)
  
Removed Predictions by Category:
  Category 1 (Bremsenteile):                4,180
  Category 2 (Motorenteile):                5,299
  
Titles Affected:                            7,160 out of 25,000 (28.64%)
Titles with at least one prediction:       24,944 out of 25,000 (99.78%)
================================================================================
```

### 2. Aspect Breakdown of Phase 35C Challenger vs. Phase 34A Leader

| Aspect Name | Phase 34A (MIN 0.60) | Phase 35C (Vehicle 0.80) | Net Change ($\Delta$) | Action |
|:---|:---:|:---:|:---:|:---:|
| `Kompatibles_Fahrzeug_Modell` | 30,433 | **20,954** | **-9,479** | Pruned ($0.60 \le c < 0.80$) |
| `Kompatible_Fahrzeug_Marke` | 28,367 | 28,367 | 0 | Untouched |
| `Im_Lieferumfang_Enthalten` | 26,567 | 26,567 | 0 | Untouched |
| `Hersteller` | 19,454 | 19,454 | 0 | Untouched |
| `Produktart` | 16,639 | 16,639 | 0 | Untouched |
| `Einbauposition` | 11,387 | 11,387 | 0 | Untouched |
| `Herstellernummer` | 11,345 | 11,345 | 0 | Untouched |
| `Bremsscheiben-Aussendurchmesser` | 4,073 | 4,073 | 0 | Untouched |
| `Anzahl_Der_Einheiten` | 3,695 | 3,695 | 0 | Untouched |
| `Kompatibles_Fahrzeug_Jahr` | 2,844 | 2,844 | 0 | Untouched |
| `Bremsscheibenart` | 1,322 | 1,322 | 0 | Untouched |
| `Maßeinheit` | 578 | 578 | 0 | Untouched |
| `Modell` | 211 | 211 | 0 | Untouched |
| `Material` | 146 | 146 | 0 | Untouched |
| `Produktlinie` | 136 | 136 | 0 | Untouched |
| `Oe/Oem_Referenznummer(N)` | 55 | 55 | 0 | Untouched |
| `Stärke` | 15 | 15 | 0 | Untouched |
| `Menge` | 4 | 4 | 0 | Untouched |
| `Anwendung` | 2 | 2 | 0 | Untouched |
| **Total** | **157,273** | **147,794** | **-9,479** | **Strict Subset** |

---

## Part E — Hardware & Runtime Telemetry

- **Compute Platform:** NVIDIA GeForce GTX 1660 Ti (6.00 GB total VRAM)
- **Host System:** Windows 11, Python 3.11, PyTorch 2.6.0+cu124
- **Sequential Execution Profile:**
  - Model 1 (DAPT GBERT seed 42): 346.4s (72.6 titles/sec)
  - Model 2 (XLM-R seed 42): 453.6s (55.5 titles/sec)
  - Model 3 (XLM-R seed 7): 521.7s (48.4 titles/sec)
  - Model 4 (XLM-R seed 21): 582.2s (43.4 titles/sec)
  - **Total 4-Model Inference Time:** 1,939.4s (32.32 minutes)
- **Peak Host Memory Allocation:** 72 MB for fused probability accumulator.
- **Cache Persistence:** Decoded candidate entities with confidence scores saved to `data/cache/phase_35c_quiz_dapt_decoded_entities.json` (22.8 MB) for zero-compute future threshold exploration.

---

## Part F — Deliverable Checksums & Verification Summary

| File / Component | Path | SHA-256 Checksum | Size |
|:---|:---|:---|:---:|
| **Phase 35C Challenger TSV** | `submissions/submission_quiz_phase35c_vehicle080.tsv` | `4B591EF6A9072C9319FF1C0134DFC9736E5A436431ED512FE47B980ECCC01382` | 5,537,316 B |
| Phase 34A Leader TSV | `submissions/submission_quiz_phase34a_dapt_min060.tsv` | `B40B03D5772EC89ED957993610B31B59F06F7C6D4FB84EB44C61C0CBF9D57258` | 5,981,377 B |
| Original Champion TSV | `submissions/submission_quiz_exp26b_min060.tsv` | `049CF5EF60D88D86C26DEC5A815F8DB820A306989E383843E6A9B3108AD7A7FA` | 6,091,833 B |
| Challenger Stats Cache | `data/cache/phase_35c_quiz_challenger_stats.json` | `3E206DFBFB030CF9CD3869DE4B1DE19CF1EBD23899FF00AC668D65691F6D4408` | 1,162 B |

---

## Part G — Operational Next Steps (Manual User Action)

1. Upload `submissions/submission_quiz_phase35c_vehicle080.tsv` to the **EvalAI Quiz** phase.
2. Record the resulting official Quiz F0.2 score and compare against:
   - Current Leader (Phase 34A DAPT): **0.9035321754551815**
   - Original Champion: **0.901245410919943**
3. If the official Quiz score increases, Phase 35C establishes a new official competition leader for the final Test submission.
