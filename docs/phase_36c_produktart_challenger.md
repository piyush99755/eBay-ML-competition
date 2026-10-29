# Phase 36C: Produktart 0.80 Quiz Challenger Report

## Executive Summary

Phase 36C created and validated exactly one new official Quiz challenger submission:
`submissions/submission_quiz_phase36c_produktart080.tsv`

This challenger is derived from our verified official EvalAI Quiz champion (**Phase 35C**, official Quiz $F_{0.2} = \mathbf{0.9159834927416699}$). In accordance with the Phase 36B research findings, the only modification is elevating the confidence threshold for **`Produktart`** from $0.60$ to $0.80$.

All other aspects, model checkpoints, probability fusion weights, BIO decoding rules, and category-aware post-filters remain completely frozen and identical to Phase 35C.

Using the verified Phase 35C decoded entity cache (`data/cache/phase_35c_quiz_dapt_decoded_entities.json`), generation executed with **zero GPU compute** in under 2 seconds. The pipeline mathematically reproduced the Phase 35C baseline row-for-row down to the exact SHA-256 before filtering.

---

## 1. Frozen Baseline Champion

- **Official EvalAI Quiz Leader**: **$F_{0.2} = 0.9159834927416699$**
- **Baseline File**: `submissions/submission_quiz_phase35c_vehicle080.tsv`
- **Baseline SHA-256**: `4B591EF6A9072C9319FF1C0134DFC9736E5A436431ED512FE47B980ECCC01382` (100% verified intact)
- **Baseline Data Rows**: 147,794
- **Checkpoints and Weights**:
  - DAPT GBERT Seed 42: `checkpoints/phase33b_dapt_gbert_best` (Weight 0.40)
  - XLM-R Seed 42: `checkpoints/xlmr_baseline_best` (Weight 0.20)
  - XLM-R Seed 7: `checkpoints/xlmr_seed_7_best` (Weight 0.20)
  - XLM-R Seed 21: `checkpoints/xlmr_seed_21_best` (Weight 0.20)
- **Baseline Thresholds**:
  - `Kompatibles_Fahrzeug_Modell`: Minimum token confidence $\ge 0.80$
  - All other 28 aspects: Minimum token confidence $\ge 0.60$

---

## 2. Phase 36C Challenger Configuration

- **Target File**: `submissions/submission_quiz_phase36c_produktart080.tsv`
- **Challenger SHA-256**: `9F76C2418C60708BC92486F5F39165021000EE9151DCCC7DD8F75CE8BBA9CC87`
- **File Size**: 5,460,127 bytes
- **Total Output Rows**: 145,396
- **Thresholds Applied**:
  - `Kompatibles_Fahrzeug_Modell`: Minimum token confidence $\ge 0.80$
  - `Produktart`: Minimum token confidence $\ge 0.80$
  - All other 27 aspects: Minimum token confidence $\ge 0.60$
- **Category-Aware Filter**: Method A (per official Annexure category definitions).

---

## 3. Strict Comparative Validation and Invariants

Every required submission invariant was verified through automated test suites:

1. **Input Universe Verification**:
   - Total Quiz records represented: exactly **25,000** records (Record IDs 5,001 to 30,000).
   - Records with extracted entities: 24,937 titles.
   - Records with zero extracted entities (all tokens below threshold): 63 titles.
2. **Multi-set Subset Invariant**:
   - For every prediction $(r, c, a, v)$ in Phase 36C, its count in Phase 36C is $\le$ its count in Phase 35C.
   - **Zero new predictions were introduced.**
3. **Aspect Invariance**:
   - Predictions for all aspects other than `Produktart` are **100% identical row-for-row** between Phase 35C and Phase 36C.
4. **Targeted Filtering Invariant**:
   - Phase 35C total rows: 147,794
   - Phase 36C total rows: 145,396
   - Row difference: **2,398**
   - Number of `Produktart` predictions removed: exactly **2,398**
   - Confidence range of removed `Produktart` predictions: strictly in $[0.6000, 0.7999]$.
5. **Official Submission Validator**:
   - Output from `scripts/validate_submission.py`:
     - File Valid: `YES (COMPLIANT)`
     - Encoding: UTF-8 without BOM
     - Header: `Record Number\tCategory\tAspect Name\tAspect Value`
     - Columns: exactly 4 tab-separated fields per line
     - No blank lines, trailing tabs, or duplicate headers.

---

## 4. Quantitative Impact Analysis

### Category Distribution of Removed Predictions
The 2,398 filtered `Produktart` predictions are evenly distributed across both competition categories:
- **Category 1**: 1,129 predictions removed ($78,600 \to 77,471$)
- **Category 2**: 1,269 predictions removed ($69,194 \to 67,925$)

### Most Frequent Removed Tokens
Inspection of the removed Quiz entities confirms that the filter primarily purges noisy, generic, and composite bag-of-words values that are frequently unannotated in the competition ground truth:
1. `'ZAHNRIEMENSATZ SET KIT'`: 367 instances
2. `'Satz'`: 363 instances
3. `'Zahnriemen Satz'`: 237 instances
4. `'ZAHNRIEMEN-SATZ KIT SET'`: 236 instances
5. `'SET'`: 170 instances
6. `'Set'`: 108 instances
7. `'Bremsen'`: 104 instances
8. `'ZAHNRIEMEN SATZ'`: 60 instances
9. `'Kit'`: 53 instances
10. `'SATZ'`: 52 instances

### Projected Performance (from Phase 36B Calibration)
On the 489 clean holdout titles:
- $\text{Precision}$: $0.939162 \to \mathbf{0.944479}$ ($+0.005317$)
- $\text{Recall}$: $0.814206 \to 0.807050$ ($-0.007156$)
- $F_{0.2}$: $0.933651 \to \mathbf{0.938333}$ ($\Delta = \mathbf{+0.004682}$)
- Paired Title-Level Bootstrap 95% CI: **$[+0.002288, +0.007141]$** ($p < 0.0001$)
- Category 1 $\Delta F_{0.2}$: $+0.005088$; Category 2 $\Delta F_{0.2}$: $+0.003791$.

---

## 5. Artifact Summary

| Artifact | Location | Status |
|---|---|---|
| **Pipeline Script** | `src/submission/predict_quiz_phase36c_produktart080.py` | Committed to Git |
| **Unit & Regression Tests** | `tests/test_phase_36c_produktart_challenger.py` | 5/5 Tests Passing |
| **Documentation Report** | `docs/phase_36c_produktart_challenger.md` | Committed to Git |
| **Challenger Submission TSV** | `submissions/submission_quiz_phase36c_produktart080.tsv` | Generated locally (git-ignored) |

---

## 6. Official EvalAI Submission Instructions

To evaluate this challenger on the official EvalAI Quiz leaderboard:

1. Navigate to the competition submission portal on EvalAI.
2. Select the **Quiz Phase** submission track.
3. Upload:
   ```
   C:\Users\piyus\Downloads\eBay-ML-competition\submissions\submission_quiz_phase36c_produktart080.tsv
   ```
4. Verify file SHA-256 prior to upload:
   `9F76C2418C60708BC92486F5F39165021000EE9151DCCC7DD8F75CE8BBA9CC87`
5. Record the official Quiz $F_{0.2}$ score returned by EvalAI.
6. Compare directly against the frozen champion baseline score (**0.9159834927416699**).
