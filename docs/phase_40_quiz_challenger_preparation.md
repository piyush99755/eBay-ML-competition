# Phase 40 Deployment Report: Official Quiz Challenger Preparation

## Executive Summary

Phase 40 successfully generated and validated a new competition-compliant EvalAI Quiz challenger based on the proven Phase 39C Candidate B configuration.

The new submission file:
`submissions/submission_quiz_phase40_lieferumfang080.tsv`
differs from the official Quiz champion (Phase 36C, EvalAI $F_{0.2} = \mathbf{0.9206543463689059}$) **strictly and exclusively** in the confidence threshold for:
$$\text{Im\_Lieferumfang\_Enthalten}: 0.60 \to 0.80$$

All models, ensemble weights, token alignments, category filtering, and other aspect thresholds remain frozen.

---

## 1. Baseline Champion Protection & Verification

Before and after the generation process, the active EvalAI Quiz champion was verified bit-for-bit:
- **Champion File:** `submissions/submission_quiz_phase36c_produktart080.tsv`
- **Official EvalAI Quiz F0.2:** **$0.9206543463689059$**
- **Expected SHA-256:** `9F76C2418C60708BC92486F5F39165021000EE9151DCCC7DD8F75CE8BBA9CC87`
- **Pre-Generation SHA-256:** `9F76C2418C60708BC92486F5F39165021000EE9151DCCC7DD8F75CE8BBA9CC87` (**Exact Match**)
- **Post-Generation SHA-256:** `9F76C2418C60708BC92486F5F39165021000EE9151DCCC7DD8F75CE8BBA9CC87` (**Exact Match**)
- **Total Champion Prediction Rows:** $145,396$

The champion TSV was never overwritten or modified.

---

## 2. Frozen Ensemble Architecture & Checkpoints

The underlying model ensemble remains strictly frozen:
- **DAPT GBERT seed 42** (`checkpoints/phase33b_dapt_gbert_best`): Weight = $0.40$
- **XLM-RoBERTa seed 42** (`checkpoints/xlmr_baseline_best`): Weight = $0.20$
- **XLM-RoBERTa seed 7** (`checkpoints/xlmr_seed_7_best`): Weight = $0.20$
- **XLM-RoBERTa seed 21** (`checkpoints/xlmr_seed_21_best`): Weight = $0.20$

All four checkpoint directories were verified intact.

---

## 3. Threshold Configuration

Exactly three aspects employ the elevated $0.80$ threshold; all remaining 26 aspects employ the baseline $0.60$ threshold:
1. `Kompatibles_Fahrzeug_Modell`: **$0.80$** (established in Phase 35C)
2. `Produktart`: **$0.80$** (established in Phase 36C)
3. `Im_Lieferumfang_Enthalten`: **$0.80$** (newly established in Phase 40)
4. All remaining 26 aspects: **$0.60$**

---

## 4. Cache Reuse & Mathematical Determinism

Inference cache inspection confirmed that `data/cache/phase_35c_quiz_dapt_decoded_entities.json` contains full pre-threshold decoded Quiz predictions across all $25,000$ titles ($5001$ to $30000$) with individual entity `min_conf` scores:
- Reconstructing Phase 36C from this cache produced **$145,396$ rows**, matching `submissions/submission_quiz_phase36c_produktart080.tsv` **row-for-row with 100% bit-for-bit identity**.
- By reusing this verified decoded cache, Phase 40 was generated with **100% mathematical determinism, zero GPU inference, and zero numerical jitter**.

---

## 5. Phase 40 Submission File Details

- **File Path:** [`submissions/submission_quiz_phase40_lieferumfang080.tsv`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/submissions/submission_quiz_phase40_lieferumfang080.tsv)
- **File Size:** $5,383,305\text{ bytes}$
- **SHA-256 Digest:** **`BF125FF2F4D7CBA79636739CF4C7FF66C7F42A4462531245E3104045961461E8`**
- **Total Data Rows:** **$143,715$**
- **Header:** `Record Number\tCategory\tAspect Name\tAspect Value`
- **Record Universe:** $25,000$ records spanning $[5001, 30000]$ ($24,894$ unique titles produced at least one prediction row).

---

## 6. Prediction-Level Difference vs Phase 36C Champion

Rigorous verification between Phase 36C and Phase 40 confirms:

| Metric | Phase 36C Champion | Phase 40 Challenger | Difference |
|---|---:|---:|---:|
| **Total Prediction Rows** | $145,396$ | $143,715$ | **$-1,681$ ($-1.156\%$)** |
| **Category 1 Predictions** | $77,471$ | $76,404$ | $-1,067$ |
| **Category 2 Predictions** | $67,925$ | $67,311$ | $-614$ |
| **`Im_Lieferumfang_Enthalten` Predictions** | $27,477$ | $25,796$ | **$-1,681$** |
| **Non-`Im_Lieferumfang_Enthalten` Predictions** | $117,919$ | $117,919$ | **$0$ (100% identical)** |
| **New Predictions Added** | — | — | **$0$** |
| **Affected Quiz Titles** | — | — | **$1,513$ ($6.05\%$)** |

### Verified Invariants:
1. Every record ID in Phase 40 matches Phase 36C in exact identical order.
2. Every non-`Im_Lieferumfang_Enthalten` prediction is 100% row-for-row identical.
3. Every removed prediction belongs exclusively to `Im_Lieferumfang_Enthalten` with confidence in $[0.60, 0.80)$.
4. Zero new predictions were added ($\text{Phase 40} \subset \text{Phase 36C}$).
5. Zero aspect names or boundaries were altered.

---

## 7. Submission Format Validation

The repository's official validator ([`scripts/validate_submission.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/scripts/validate_submission.py)) was executed:
```
==================================================
SUBMISSION VALIDATION REPORT
==================================================
File Path:                C:\Users\piyus\Downloads\eBay-ML-competition\submissions\submission_quiz_phase40_lieferumfang080.tsv
File Valid:               YES (COMPLIANT)
Gzip Compressed:          False
Header Present:           True
Total Lines Read:         143716
Data Aspect Rows:         143715
Unique Record Numbers:    24894
Record Range:             [5001, 30000]
Category Distribution:    {1: 76404, 2: 67311}
Aspect Types Extracted:   19 of 29 official aspects

All schema, encoding, and Annexure.pdf formatting checks passed!
```

---

## 8. Test Suite & Provenance

- **Focused Challenger Test:** [`tests/test_phase_40_quiz_challenger.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/tests/test_phase_40_quiz_challenger.py) (5/5 tests passed in 2.135s).
- **Full Regression Suite:** 112/112 tests passed in 31.792s with zero regressions.
- **Generator Script:** [`src/submission/predict_quiz_phase40_lieferumfang080.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/submission/predict_quiz_phase40_lieferumfang080.py).
- **Free Disk Space:** **7.98 GiB (8.57 GB)**.

---

## 9. Manual EvalAI Upload Instructions

> [!IMPORTANT]
> Automatic submission is disabled. Follow these exact steps to submit the Phase 40 challenger manually:

1. **Log in to EvalAI:** Navigate to the eBay 2026 University ML Competition page on [EvalAI](https://eval.ai).
2. **Select Phase:** Choose **Quiz Phase** (or Test Phase when open).
3. **Locate Submission File:**
   `c:\Users\piyus\Downloads\eBay-ML-competition\submissions\submission_quiz_phase40_lieferumfang080.tsv`
4. **Verify SHA-256 Before Uploading:**
   ```powershell
   Get-FileHash submissions/submission_quiz_phase40_lieferumfang080.tsv -Algorithm SHA256
   ```
   Must match: `BF125FF2F4D7CBA79636739CF4C7FF66C7F42A4462531245E3104045961461E8`.
5. **Submission Metadata:**
   - **Method Name:** `DAPT GBERT + 3x XLM-R Ensemble (VM 0.80, PA 0.80, Lieferumfang 0.80)`
   - **Description:** `Four-model precision ensemble with elevated thresholds on Kompatibles_Fahrzeug_Modell (0.80), Produktart (0.80), and Im_Lieferumfang_Enthalten (0.80).`
6. **Upload and Monitor:** Upload the TSV file and await the official evaluation result.
