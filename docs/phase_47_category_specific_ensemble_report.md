# Phase 47 — Category-Specific Boundary-Specialist Ensemble Report

## Executive Summary

- **Objective:** Investigate whether selectively routing Category 2 predictions through a blend with the existing Phase 32C boundary-specialist GBERT (`gbert_phase32c_challenger_best` at 10% weight) provides a genuine complementary advantage while keeping Category 1 strictly invariant.
- **Champion Protected:** Phase 40 Quiz submission [`submissions/submission_quiz_phase40_lieferumfang080.tsv`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/submissions/submission_quiz_phase40_lieferumfang080.tsv) verified bit-for-bit:
  - Expected SHA-256: `BF125FF2F4D7CBA79636739CF4C7FF66C7F42A4462531245E3104045961461E8`
  - Official EvalAI Quiz $F_{0.2}$: `0.9233317116970503`
  - Local Clean-Holdout Baseline (489 titles): Overall **`0.927771`** (Category 1: `0.918323`, Category 2: `0.937219`; Micro TP: `3021`, FP: `165`, FN: `752`).
- **Phase 46 Verification & Discrepancy Resolution:**
  - In Phase 46, blending `gbert_phase32c_challenger_best` at global weight 0.10 yielded: Overall **`0.928892`** ($\Delta = +0.001121$), Category 1: `0.916020` ($\Delta = -0.002303$), and Category 2: `0.941765` ($\Delta = +0.004546$).
  - *Report Discrepancy Resolved:* Phase 46 text in line 339 cited `+0.000239` as the best observed delta, which corresponded to the separate `orig_gbert` sweep. The specialist blend actually achieved $+0.001121$ globally, driven entirely by Category 2 but severely damaging Category 1.
- **Independent Evidence Audit:**
  - Before adopting this hypothesis, we searched historical validation splits for independent evidence of a Category 2 advantage.
  - On the independent 800-title internal development set from Phase 32C:
    - Category 1 gained $+0.004225$ ($0.884556 \to 0.888781$).
    - Category 2 **actively regressed by $-0.002831$** ($0.849324 \to 0.846493$).
  - On the 1,000-title exploratory ensemble holdout, the specialist ensemble trailed baseline by $-0.000635$, gaining only $+0.000686$ on Category 2.
  - **Conclusion:** There is **zero independent evidence** that `gbert_phase32c_challenger_best` provides a robust advantage for Category 2. In fact, independent development data showed Category 2 actively degraded.
- **Candidate A Evaluation on Clean Holdout ($N=489$):**
  - Category 1: Phase 40 unchanged $\to$ **`0.918323`** ($\mathbf{\Delta = 0.000000}$, verified bit-for-bit identical).
  - Category 2: 90% Phase 40 + 10% Specialist $\to$ **`0.941764`** ($\mathbf{\Delta = +0.004546}$).
  - Overall Score: **`0.930044`** ($\mathbf{\Delta = +0.002273}$).
  - Micro Metrics: TP: 3,024 ($+3$), FP: 161 ($-4$), FN: 749 ($-3$), Micro $F_{0.2}$: `0.942756` ($+0.001218$).
- **Mechanism & Fragility Findings:**
  - In Category 2 (184 titles total), **only 13 titles changed**.
  - The score improvement did NOT stem from the intended vehicle model boundary repair.
  - Instead, the 10% blend slightly lowered confidence on four borderline false positives (`Corsa`, `Motor`, `Satz`, `Water`), pushing them below the 0.80/0.60 thresholds ($-4$ FPs), and elevated three single-token part numbers (`092.750`, `06F133062Q`, `11121485838`) above 0.60 ($+3$ TPs).
- **Paired Bootstrap Analysis (1,000 resamples, seed 42):**
  - Overall Mean Delta: **`+0.002228`**
  - 95% Percentile Confidence Interval: **`[+0.000402, +0.004503]`**
  - Fraction with $\Delta \le 0$: **`0.80%`**
  - Category 1 Mean Delta: `+0.000000` (strictly invariant, CI `[0.0, 0.0]`)
  - Category 2 Mean Delta: `+0.004455` (CI `[+0.000803, +0.009006]`).
- **Generalization Verdict:**
  While Candidate A clears the local $+0.0020$ threshold on the repeatedly inspected holdout and protects Category 1, the lack of independent evidence and concentration in just 13 titles indicates that the gain may be an artifact of repeated holdout inspection. Spending an official Quiz submission is not justified without independent confirmation.
- **Final Decision Verdict:** **`PROMISING LOCAL CHALLENGER — GENERALIZATION UNCONFIRMED`**.

---

## 1. Champion Protection & Baseline Verification

Before conducting any diagnostics or evaluation, champion submission integrity and local baseline metrics were verified:

| Check / Metric | Target Value | Observed Value | Verification Status |
| :--- | :--- | :--- | :---: |
| **Champion TSV** | `submissions/submission_quiz_phase40_lieferumfang080.tsv` | Present | **PASSED** |
| **Champion SHA-256** | `BF125FF2F4D7CBA79636739CF4C7FF66C7F42A4462531245E3104045961461E8` | Exact Match | **PASSED** |
| **Official EvalAI Quiz $F_{0.2}$** | `0.9233317116970503` | `0.9233317116970503` | **VERIFIED** |
| **Local Clean-Holdout Overall** | `0.927771` | `0.927771` | **REPRODUCED** |
| **Local Clean-Holdout Cat 1** | `0.918323` | `0.918323` | **REPRODUCED** |
| **Local Clean-Holdout Cat 2** | `0.937219` | `0.937219` | **REPRODUCED** |
| **Local Micro TP / FP / FN** | 3,021 / 165 / 752 | 3,021 / 165 / 752 | **EXACT MATCH** |

The champion file and historical submissions remained unmodified throughout the phase.

---

## 2. Phase 46 Verification & Discrepancy Resolution

### 2.1 Metric Reproduction
We verified the exact scores from the Phase 46 global fusion experiments using the established evaluator:

| Metric | Phase 40 Baseline | Global Specialist Blend ($w=0.10$) | Delta ($\Delta$) | Impact Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Overall Official Score** | **`0.927771`** | **`0.928892`** | $\mathbf{+0.001121}$ | Moderate Overall Gain |
| **Category 1 Score** | **`0.918323`** | **`0.916020`** | $\mathbf{-0.002303}$ | **Severe Regression** |
| **Category 2 Score** | **`0.937219`** | **`0.941765`** | $\mathbf{+0.004546}$ | Large Gain |
| Micro TP / FP / FN | 3,021 / 165 / 752 | 3,028 / 165 / 745 | $+7$ TP, 0 FP, $-7$ FN | — |

### 2.2 Phase 46 Report Discrepancy Resolution
In Phase 46 line 339, the text stated: *"best observed delta is +0.000239 at weight 0.10"*. 
- **Cause:** That figure was from the `orig_gbert` candidate sweep (where weight 0.10 achieved $+0.000239$).
- **Correction:** The `gbert_phase32c_challenger_best` blend at weight 0.10 actually produced a global delta of **`+0.001121`**. However, it was rejected in Phase 46 because Category 1 suffered an unacceptable regression of $-0.002303$, directly violating the category non-regression criterion.

---

## 3. Independent Evidence Audit

Before testing a category-specific blend, we audited historical experiments conducted prior to Phase 46 to check whether independent data supports a genuine Category 2 advantage for `gbert_phase32c_challenger_best`:

| Independent Evaluation Set | Size | Baseline Cat 1 | Baseline Cat 2 | Challenger Cat 1 | Challenger Cat 2 | Cat 1 Delta | Cat 2 Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 32C Internal Dev Set** | 800 titles | 0.884556 | 0.849324 | 0.888781 | 0.846493 | **`+0.004225`** | **`-0.002831`** |
| **Phase 32C Exploratory Holdout** | 1,000 titles | 0.893547 | 0.900812 | 0.891590 | 0.901498 | `-0.001957` | `+0.000686` |

### Critical Finding:
1. When `gbert_phase32c_challenger_best` was originally trained and evaluated on the internal 800-title validation set, its advantage was **strictly on Category 1 (+0.004225)**. On Category 2, it **actively regressed by $-0.002831$**.
2. On the 1,000-title exploratory holdout ensemble, Category 2 gained only a marginal $+0.000686$ while overall score dropped by $-0.000635$.
3. **Conclusion:** There is **zero independent validation evidence** supporting a Category 2-specific advantage. The Category 2 boost observed on the 489-title holdout is completely contradicted by prior independent experimental splits.

---

## 4. Predeclared Candidate A Configuration

To test whether category routing isolates the Category 2 gain while eliminating Category 1 regression, Candidate A was predeclared without post-hoc tuning:

- **Category 1 Routing:**
  $$P_{\text{Candidate A}} = P_{\text{Phase 40 Baseline}}$$
  All Phase 40 confidence thresholds preserved (`Kompatibles_Fahrzeug_Modell`: 0.80, `Produktart`: 0.80, `Im_Lieferumfang_Enthalten`: 0.80, others: 0.60).
- **Category 2 Routing:**
  $$P_{\text{Candidate A}} = 0.90 \times P_{\text{Phase 40 Baseline}} + 0.10 \times P_{\text{gbert\_phase32c\_challenger}}$$
  Same Phase 40 confidence thresholds applied.

---

## 5. Candidate A Evaluation on Clean Holdout ($N=489$)

All evaluations were executed using cached probabilities from `data/cache/phase_35a_val_1k_probs.pt` and `data/cache/phase_32c_challenger_clean_holdout_probs.pt`:

### 5.1 Official Competition Metric Comparison

| Evaluation Metric | Phase 40 Baseline | Candidate A | Delta ($\Delta$) | Gate Requirement | Gate Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Official Score ($F_{0.2}$)** | **`0.927771`** | **`0.930044`** | $\mathbf{+0.002273}$ | $\ge +0.0020$ | **PASSED** |
| **Category 1 Score** | **`0.918323`** | **`0.918323`** | $\mathbf{+0.000000}$ | $\ge 0.0$ (Invariant) | **PASSED** |
| **Category 2 Score** | **`0.937219`** | **`0.941764`** | $\mathbf{+0.004546}$ | $> 0.0$ | **PASSED** |
| Micro True Positives (TP) | 3,021 | 3,024 | $+3$ | $\ge 0$ | **PASSED** |
| Micro False Positives (FP) | 165 | 161 | $-4$ | $\le 0$ | **PASSED** |
| Micro False Negatives (FN) | 752 | 749 | $-3$ | $\le 0$ | **PASSED** |
| Global Micro $F_{0.2}$ | `0.941539` | `0.942756` | $+0.001218$ | $> 0$ | **PASSED** |

### 5.2 Category 1 Bit-for-Bit Invariance Check
Across all 305 Category 1 titles in the clean holdout, predicted entity spans, aspects, and confidence scores between Candidate A and Phase 40 were compared:
- **Category 1 Predictions Changed:** **0 / 305 titles (100% Bit-for-Bit Identical)**.

---

## 6. Detailed Error Decomposition in Category 2

To determine what caused the $+0.004546$ jump in Category 2, every prediction change was traced:

### 6.1 Title and Entity Volume Changes
- **Total Category 2 Titles:** 184
- **Titles with Changed Predictions:** **13 / 184 (7.07%)**
- **Titles Unchanged:** 171 / 184 (92.93%)

### 6.2 Aspect-Level Changes in Category 2

| Aspect | Aspect Weight | Baseline Score | Candidate A Score | Score Delta | TP Delta | FP Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Kompatibles_Fahrzeug_Modell` | 0.2623 | 0.883324 | 0.896375 | **`+0.013051`** | $0$ (175 $\to$ 175) | $\mathbf{-3}$ (18 $\to$ 15) |
| `Produktart` | 0.1448 | 0.981730 | 0.987745 | **`+0.006015`** | $0$ (155 $\to$ 155) | $\mathbf{-1}$ (2 $\to$ 1) |
| `Herstellernummer` | 0.0861 | 0.936483 | 0.939405 | **`+0.002922`** | $\mathbf{+3}$ (93 $\to$ 96) | $0$ (6 $\to$ 6) |
| *Other 12 Aspects* | 0.5068 | Various | Various | **`+0.000000`** | $0$ | $0$ |

### 6.3 Qualitative Root Cause of the Gain
The Phase 32C specialist was fine-tuned specifically to extend multi-token vehicle model spans (`I-Kompatibles_Fahrzeug_Modell`). However, on Category 2:
1. **Zero Vehicle Model Boundaries Were Corrected:**
   - In `Kompatibles_Fahrzeug_Modell`, True Positives remained completely unchanged at 175.
   - The $+0.013051$ gain came entirely from **suppressing 3 false positives**:
     - Rec 4967: Suppressed `'Corsa'` (FP because gold was `'Corsa A B'`).
     - Rec 4384: Suppressed `'Motor'` (FP because gold was `'Motor 3.0 diesel'`).
     - Rec 1320: Suppressed `'Satz'` as `Produktart` (FP because gold was `'Satz Riementrieb Klimakompressor'`).
2. **Part Number Confidence Shift:**
   - In Rec 776, 2867, and 3898, the specialist slightly elevated confidence on three single-token part numbers (`092.750`, `06F133062Q`, `11121485838`), pushing them above the 0.60 threshold ($+3$ TPs).
3. **Assessment:** The gain did not occur via the hypothesized architectural mechanism (boundary recovery). Instead, it arose from an accidental suppression of 4 borderline false positives and threshold crossing of 3 part numbers in a cohort of just 13 titles.

---

## 7. Paired Category-Stratified Bootstrap Analysis

We ran 1,000 paired category-stratified bootstrap resamples (seed 42) recomputing all official category and aspect weights per resample:

| Metric | Overall ($F_{0.2}$) | Category 1 | Category 2 |
| :--- | :---: | :---: | :---: |
| **Empirical Delta** | `+0.002273` | `+0.000000` | `+0.004546` |
| **Bootstrap Mean Delta** | `+0.002228` | `+0.000000` | `+0.004455` |
| **95% Percentile Confidence Interval** | **`[+0.000402, +0.004503]`** | **`[+0.000000, +0.000000]`** | **`[+0.000803, +0.009006]`** |
| **Fraction of Resamples with $\Delta \le 0$** | **`0.80%`** | `0.00%` (Invariant) | **`0.80%`** |

### Statistical Caveat:
While the bootstrap interval is strictly positive (`[+0.000402, +0.004503]`), this metric is exploratory because the 489-title clean holdout has been used repeatedly across multiple phases.

---

## 8. Generalization Assessment

We evaluate Candidate A across the three required dimensions:

### A. Does Candidate A improve the already-inspected local holdout?
**YES.** On the 489 clean holdout titles, Candidate A achieves an official-style score of `0.930044` ($\Delta = +0.002273$), with Category 1 unchanged and Category 2 gaining $+0.004546$.

### B. Is there independent evidence that the specialist genuinely helps Category 2?
**NO.** In the Phase 32C internal 800-title development set, `gbert_phase32c_challenger_best` actively regressed Category 2 by $-0.002831$. In the 1,000-title exploratory holdout, it showed near-zero ensemble impact on Category 2 ($+0.000686$). There is no independent evidence supporting a Category 2-specific advantage.

### C. Is there enough evidence to justify spending an official Quiz submission?
**NO.** 
1. The entire Category 2 improvement on the holdout is concentrated in just **13 titles**.
2. The mechanism is unrelated to boundary specialization (it operated via accidental false-positive threshold suppression).
3. The historical independent data contradicts the holdout observation.
4. With approximately two days remaining before the competition deadline and limited EvalAI submissions available, risking an official submission on an unconfirmed 13-title artifact is strategically unsound.

---

## 9. Promotion Gate Synthesis

| Promotion Gate | Requirement | Candidate A Result | Status |
| :--- | :--- | :--- | :---: |
| **1. Net Score Delta** | Overall local $\Delta \ge +0.0020$ | $\Delta = +0.002273$ | **PASSED** |
| **2. Category 1 Invariance** | Category 1 predictions exactly unchanged | Bit-for-bit identical ($\Delta = 0.0$) | **PASSED** |
| **3. Category 2 Improvement** | Category 2 score improves | $\Delta = +0.004546$ (0.937219 $\to$ 0.941764) | **PASSED** |
| **4. Exact-Match Verification** | Net positive entity changes | $+3$ TP, $-4$ FP, $-3$ FN | **PASSED** |
| **5. False Positive Control** | No substantial new FPs introduced | Total FPs reduced from 165 to 161 | **PASSED** |
| **6. Independent Generalization** | Supported by external independent split | Independent dev set showed Category 2 regression | **FAILED** |

Because all local quantitative gates passed but independent external validation is absent, Candidate A qualifies strictly as an unconfirmed local challenger.

---

## 10. Deliverables & Repository Verification

1. **Evaluation Pipeline:** [`src/evaluation/phase_47_category_specific_ensemble.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/evaluation/phase_47_category_specific_ensemble.py)
   - Reproducible evaluation computing baseline reproduction, Candidate A routing, independent evidence audit, error decomposition, and 1,000 paired bootstrap resamples.
   - Cached to `data/cache/phase_47_category_ensemble_results.json`.
2. **Unit Test Suite:** [`tests/test_phase_47_category_specific_ensemble.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/tests/test_phase_47_category_specific_ensemble.py)
   - 9 test cases covering champion integrity, baseline reproduction, Phase 46 verification, Category 1 invariance, Candidate A metrics, independent evidence audit, and decision verdict.
   - **Full test suite passes:** 165 tests passing in 31.2s (`OK`).
3. **Audit Documentation:** [`docs/phase_47_category_specific_ensemble_report.md`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/docs/phase_47_category_specific_ensemble_report.md).
4. **Champion File Invariant:** Preserved bit-for-bit (`BF125FF2F4D7CBA79636739CF4C7FF66C7F42A4462531245E3104045961461E8`).
5. **Disk Space Invariant:** **7.91 GB free** (well above 5.0 GB limit).

---

## Final Recommendation

Candidate A demonstrates a valid local score of `0.930044` ($\Delta = +0.002273$) on the clean holdout while completely protecting Category 1. However, because this gain is driven by only 13 titles and directly contradicts the Category 2 regression observed on the independent 800-title development set, it cannot be confirmed as a generalizable improvement.

We recommend **retaining Phase 40 as the active official champion** and withholding Candidate A from submission unless explicit user discretion decides to take an exploratory risk.

**FINAL VERDICT:**

**PROMISING LOCAL CHALLENGER — GENERALIZATION UNCONFIRMED**
