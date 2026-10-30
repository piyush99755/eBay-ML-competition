# Phase 45 — Precision-Gated Recovery of Rejected Entities: Audit & Feasibility Report

## Executive Summary

- **Objective:** Investigate whether high-quality entity predictions rejected by the Phase 40 confidence thresholds can be selectively recovered without damaging precision, utilizing independent 4-model agreement, non-overlapping span safety, and supervised training annotations (`train_4k`).
- **Champion Verification:** Phase 40 official champion submission [`submissions/submission_quiz_phase40_lieferumfang080.tsv`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/submissions/submission_quiz_phase40_lieferumfang080.tsv) verified bit-for-bit:
  - SHA-256: `BF125FF2F4D7CBA79636739CF4C7FF66C7F42A4462531245E3104045961461E8`
  - Official EvalAI Quiz $F_{0.2}$: `0.9233317116970503`
  - Local Clean-Holdout Baseline (489 titles): Overall **`0.927771`** (Category 1: `0.918323`, Category 2: `0.937219`; Micro TP: `3021`, FP: `165`, FN: `752`).
- **Predeclared Recovery Gate Result:**
  - Recovered Entities Passing Gate: **94**
  - Recovered True Positives: **65**
  - Recovered False Positives: **29**
  - Recovered Entity Precision: **69.15%**
- **Metric Impact ($F_{0.2}$ Score):**
  - Overall Score: **`0.923803`** ($\mathbf{\Delta = -0.003968}$, severe regression)
  - Category 1: **`0.914228`** ($\Delta = -0.004095$)
  - Category 2: **`0.933378`** ($\Delta = -0.003841$)
  - Micro $F_{0.2}$: **`0.935446`** ($\Delta = -0.006093$)
- **Bootstrap Uncertainty (1,000 paired resamples, seed 42):**
  - Mean Overall Delta: **`-0.003858`**
  - 95% Percentile Confidence Interval: **`[-0.006966, -0.000700]`** (strictly negative!)
  - Fraction $\le 0$: **`99.0%`** (over 99% of bootstrap iterations actively degrade performance).
- **Core Scientific Finding:**
  Under the official $F_{0.2}$ metric ($\beta = 0.2, \beta^2 = 0.04$), false positives are penalized **25 times more severely** than false negatives. The mathematical break-even precision required for any entity recovery mechanism to improve the score is:
  $$\text{Precision}_{\text{break-even}} = \frac{1}{1 + \beta^2} = \frac{1}{1.04} \approx 96.15\%$$
  Even with a strict 5-point gate requiring $\ge 0.70$ confidence, $\ge 3$ agreeing independent models, zero overlap with accepted entities, and identical exact string occurrences in supervised training annotations, the recovered candidate precision reaches only **69.15%**. Introducing 29 false positives overwhelms the gain of 65 true positives, resulting in a net penalty of $-0.003968$.
- **Final Decision Verdict:** **`REJECT — RETAIN PHASE 40`**.

---

## 1. Champion Verification & Local Baseline Reproduction

Before conducting any diagnostics or evaluation, the repository and submission integrity were verified against the official benchmarks.

| Metric / Check | Target Value | Measured Value | Status |
| :--- | :--- | :--- | :--- |
| Champion File | `submissions/submission_quiz_phase40_lieferumfang080.tsv` | Present | **PASSED** |
| Champion SHA-256 | `BF125FF2F4D7CBA79636739CF4C7FF66C7F42A4462531245E3104045961461E8` | Exact Match | **PASSED** |
| Official Quiz $F_{0.2}$ | `0.9233317116970503` | Exact Match | **PASSED** |
| Local Holdout Titles | 489 clean titles | 489 titles | **PASSED** |
| Local Baseline Overall | `0.927771` | `0.927771` | **PASSED** |
| Local Baseline Cat 1 | `0.918323` | `0.918323` | **PASSED** |
| Local Baseline Cat 2 | `0.937219` | `0.937219` | **PASSED** |
| Local Micro TP / FP / FN | 3,021 / 165 / 752 | 3,021 / 165 / 752 | **PASSED** |

The champion file was untouched throughout the audit, and fallback Phase 36C (`9F76C2418C60708BC92486F5F39165021000EE9151DCCC7DD8F75CE8BBA9CC87`) remains intact.

---

## 2. Audit of Rejected Predictions

Using cached individual model out-of-fold probabilities and fused probabilities from the frozen ensemble (DAPT GBERT seed 42: 0.40, XLM-R seed 42: 0.20, XLM-R seed 7: 0.20, XLM-R seed 21: 0.20), raw sequence predictions were reconstructed prior to confidence thresholding.

### 2.1 Reconstructed Entity Counts
- **Total Raw Entities Generated:** 3,755
- **Accepted Entities (Phase 40 Thresholds):** 3,186
- **Rejected Entities (Below Thresholds):** 569

### 2.2 Exact Entity Matching vs Ground Truth
Each rejected entity was evaluated against unmatched ground-truth entities on the clean holdout:
- **True Positives (Exact aspect & exact string span):** 327 (57.47%)
- **False Positives:** 242 (42.53%)

### 2.3 Reconciliation with Phase 44
Phase 44 reported 394 token-tag aligned False Negatives in Buckets A and B (under-confidence and threshold suppression). When translated into exact non-overlapping multi-token entity spans and matched under the official multi-set metric rules, exactly 327 distinct true entities were rejected by the Phase 40 confidence thresholds. The remaining 67 gold entities in Buckets A and B represent partial span overlaps or tokens where the decoder generated fragmented spans that failed exact span matching.

### 2.4 Breakdown of Rejected Entities by Aspect
The 569 rejected entities are distributed across aspects as follows:

| Aspect | Rejected Count | Default Thresh | Phase 40 Thresh | Primary Cause for Rejection |
| :--- | :---: | :---: | :---: | :--- |
| `Kompatibles_Fahrzeug_Modell` | 303 | 0.60 | 0.80 | Elevated 0.80 threshold (Phase 35C) |
| `Produktart` | 82 | 0.60 | 0.80 | Elevated 0.80 threshold (Phase 36C) |
| `Im_Lieferumfang_Enthalten` | 56 | 0.60 | 0.80 | Elevated 0.80 threshold (Phase 40) |
| `Einbauposition` | 39 | 0.60 | 0.60 | Below standard 0.60 confidence |
| `Modell` | 20 | 0.60 | 0.60 | Ambiguity with generic series names |
| `Herstellernummer` | 16 | 0.60 | 0.60 | Alphanumeric code truncation |
| `Kompatible_Fahrzeug_Marke` | 12 | 0.60 | 0.60 | Below standard 0.60 confidence |
| `Oe/Oem_Referenznummer(N)` | 12 | 0.60 | 0.60 | Complex punctuation / formatting |
| `Hersteller` | 9 | 0.60 | 0.60 | Brand ambiguity |
| `Bremsscheibenart` | 6 | 0.60 | 0.60 | Rare terminology |
| `Besonderheiten` | 4 | 0.60 | 0.60 | Unstructured descriptor noise |
| *Other 7 Aspects* | 10 | 0.60 | 0.60 | Rare aspects (< 3 instances each) |
| **Total** | **569** | — | — | — |

---

## 3. Predeclared Precision-Gated Recovery Rule

To test whether the 327 rejected true positives could be harvested without admitting the 242 false positives, a strict 5-condition gate was predeclared prior to evaluation:

1. **Confidence Gate:** Candidate minimum token confidence $\ge 0.70$ (`min_conf >= 0.70`).
2. **Model Consensus Gate:** At least 3 of the 4 ensemble models independently predict the exact same entity span and aspect tag (`agree_models >= 3`).
3. **Collision Safety Gate:** Candidate span does not overlap any entity accepted by Phase 40 (`not overlaps_accepted`).
4. **Supervised Knowledge Gate:** The exact `(aspect, value)` pair appears in the supervised training annotations (`(aspect, value) in train_gold`).
5. **Boundary Preservation Gate:** Original token boundaries decoded by the model are preserved without truncation or expansion.

---

## 4. Exact Trade-Offs & Official Metric Evaluation

### 4.1 Filter Pipeline Attrition
Applying the 5 gating criteria progressively to the 569 rejected candidates:
- Raw Rejected Candidates: 569 (327 TP, 242 FP; precision = 57.5%)
- After Collision Gate (`not overlaps_accepted`): 492 candidates
- After Confidence Gate (`min_conf >= 0.70`): 228 candidates
- After Model Agreement Gate (`agree_models >= 3`): 142 candidates
- After Training Knowledge Gate (`in_train_exact`): **94 candidates**

### 4.2 Candidate Recovery Breakdown
Among the 94 candidates admitted by the gate:
- **Recovered True Positives:** 65
- **Recovered False Positives:** 29
- **Recovered Entity Precision:** **69.15%**

### 4.3 Per-Aspect Recovered Candidates

| Aspect | Recovered Candidates | True Positives | False Positives | Recovered Precision |
| :--- | :---: | :---: | :---: | :---: |
| `Kompatibles_Fahrzeug_Modell` | 52 | 43 | 9 | 82.69% |
| `Produktart` | 23 | 9 | 14 | **39.13%** |
| `Im_Lieferumfang_Enthalten` | 19 | 13 | 6 | 68.42% |
| **Total** | **94** | **65** | **29** | **69.15%** |

### 4.4 The "Training Support" Double-Edged Sword
A critical qualitative vulnerability discovered in the audit is that training set support (`in_train_exact`) acts as a double-edged sword for generic German vocabulary:
- Tokens like `"Satz"`, `"Rep.-Satz"`, `"Bremsen"`, and `"Handbremse"` occur hundreds of times in `train_4k` labeled as valid `Produktart` entities.
- In holdout titles, sellers frequently include `"Satz"` or `"Bremsen"` as generic filler, context descriptions, or noise (e.g., when the true product is an accessory or caliper).
- The models assign moderate confidence ($\approx 0.72 - 0.76$) and agree that the token matches the training pattern.
- However, in ground truth annotations for those specific listings, `"Satz"` was rejected by the annotators as non-salient.
- Consequently, `Produktart` admitted **14 false positives against only 9 true positives (39.1% precision)**, devastating the score.

---

## 5. Metric Impact and Mathematical Proof of Failure Under $F_{0.2}$

### 5.1 Official Competition Metric Comparison

| Evaluation Metric | Phase 40 Baseline | Phase 45 Gated Recovery | Delta ($\Delta$) | Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Official Score ($F_{0.2}$)** | **`0.927771`** | **`0.923803`** | $\mathbf{-0.003968}$ | **Severe Regression** |
| Category 1 Score | `0.918323` | `0.914228` | $-0.004095$ | **Severe Regression** |
| Category 2 Score | `0.937219` | `0.933378` | $-0.003841$ | **Severe Regression** |
| Micro True Positives (TP) | 3,021 | 3,086 | $+65$ | Gain |
| Micro False Positives (FP) | 165 | 194 | $+29$ | **Penalty ($25\times$)** |
| Micro False Negatives (FN) | 752 | 687 | $-65$ | Reduction |
| Global Micro $F_{0.2}$ | `0.941539` | `0.935446` | $-0.006093$ | Severe Regression |

### 5.2 Mathematical Proof: Why 69.15% Precision Fails
The official competition metric is $F_{\beta}$ with $\beta = 0.2$:
$$F_{0.2} = (1 + \beta^2) \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}} = \frac{1.04 \cdot \text{TP}}{1.04 \cdot \text{TP} + \text{FP} + 0.04 \cdot \text{FN}}$$

Taking the partial derivatives with respect to adding a True Positive ($\text{TP} \to \text{TP} + 1, \text{FN} \to \text{FN} - 1$) versus adding a False Positive ($\text{FP} \to \text{FP} + 1$):
- Adding 1 True Positive reduces FN by 1 and increases the numerator and denominator, with net positive derivative proportional to $\beta^2 = 0.04$.
- Adding 1 False Positive increases the denominator with coefficient $1.0$, imposing a penalty proportional to $1.0$.
- The ratio of penalties is:
  $$\frac{\text{FP Penalty}}{\text{TP Benefit}} = \frac{1.0}{\beta^2} = \frac{1.0}{0.04} = 25$$
- For a recovered cohort to achieve $\Delta F_{0.2} > 0$, the ratio of recovered TPs to introduced FPs must exceed $25:1$, corresponding to a break-even precision of:
  $$\text{Precision}_{\text{break-even}} = \frac{25}{25 + 1} = \frac{1}{1.04} \approx 96.15\%$$

With an empirical precision of **69.15%** ($65 \text{ TP} : 29 \text{ FP} \approx 2.24 : 1$), the ratio falls an entire order of magnitude below the required $25:1$ threshold. Every introduced false positive wipes out the gain of 25 recovered true positives.

---

## 6. Paired Category-Stratified Bootstrap Analysis

To confirm that this result is not an artifact of random sampling, 1,000 paired category-stratified bootstrap resamples were executed with seed 42.

| Bootstrap Metric (1,000 resamples) | Overall ($F_{0.2}$) | Category 1 | Category 2 |
| :--- | :---: | :---: | :---: |
| Mean Delta | **`-0.003858`** | `-0.004052` | `-0.003664` |
| 95% Percentile Confidence Interval | **`[-0.006966, -0.000700]`** | `[-0.008273, +0.000104]` | `[-0.008537, +0.000778]` |
| Fraction of Resamples with $\Delta \le 0$ | **`99.0%`** | `96.9%` | `94.8%` |

The 95% confidence interval for the overall score is **strictly negative** (`[-0.006966, -0.000700]`). In 990 out of 1,000 resamples, recovering these rejected entities degrades performance.

---

## 7. Decision Synthesis Against Promotion Criteria

| Criterion | Requirement | Phase 45 Result | Status |
| :--- | :--- | :--- | :--- |
| **1. Overall Delta** | $\Delta \ge +0.0020$ | $\Delta = -0.003968$ | **FAILED** |
| **2. Category Non-Regression** | $\Delta \text{Cat 1} \ge 0$ AND $\Delta \text{Cat 2} \ge 0$ | Cat 1: $-0.004095$, Cat 2: $-0.003841$ | **FAILED** |
| **3. Precision Break-Even** | Recovered entity precision $\ge 96.15\%$ | Recovered precision $= 69.15\%$ | **FAILED** |
| **4. Bootstrap Statistical Confidence** | 95% CI strictly positive ($\text{CI}_{\text{lower}} > 0$) | 95% CI is $[-0.006966, -0.000700]$ (strictly negative) | **FAILED** |
| **5. Data Leakage / Integrity** | Zero contamination, reproducible seed | Clean holdout untouched, seeds fixed | **PASSED** |

All four promotion gates failed conclusively.

---

## 8. Deliverables & Repository State

1. **Diagnostic Script:** [`src/evaluation/phase_45_rejected_entity_recovery.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/evaluation/phase_45_rejected_entity_recovery.py)
   - Self-contained implementation computing raw entity extraction, 5-point gating, exact multi-set metric evaluation, and 1,000 paired bootstrap resamples.
   - Caches structured outputs to `data/cache/phase_45_recovery_results.json`.
2. **Unit Test Suite:** [`tests/test_phase_45_rejected_entity_recovery.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/tests/test_phase_45_rejected_entity_recovery.py)
   - 7 test cases covering champion integrity, baseline reproduction, raw decoding counts, gating results, metric regression, bootstrap confidence intervals, and decision verdicts.
   - Full test suite: **148 tests passing** in 30.6s.
3. **Audit Report:** [`docs/phase_45_rejected_entity_recovery_audit.md`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/docs/phase_45_rejected_entity_recovery_audit.md).
4. **Disk Space Invariant:** 7.92 GB free disk space (well above 5.0 GB minimum).
5. **Champion File Invariant:** Unaltered (`BF125FF2F4D7CBA79636739CF4C7FF66C7F42A4462531245E3104045961461E8`).

---

## 9. Final Recommendation

The diagnostic audit demonstrates conclusively that predictions rejected by the Phase 40 confidence thresholds cannot be safely recovered. The $25\times$ penalty ratio of $F_{0.2}$ renders any recovery mechanism with $< 96.15\%$ precision destructive. Attempting to relax thresholds or recover rejected entities directly violates the core mathematical properties of precision-weighted competition scoring.

Future research before the November 6 deadline should focus on:
1. **Precision pruning** of residual false positives (e.g., in low-confidence single-token aspects).
2. **Exact boundary refinement** where a true positive is already accepted but spans an extra or missing token, which eliminates an FP and creates a TP simultaneously (a $26\times$ value swing).

**FINAL VERDICT:**

**REJECT — RETAIN PHASE 40**
