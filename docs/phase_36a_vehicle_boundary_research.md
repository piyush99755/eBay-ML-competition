# Phase 36A: Vehicle-Model Boundary Correction Research Report

**Date:** October 29, 2026  
**Status:** COMPLETED & SCIENTIFICALLY CONCLUDED  
**Auditor / System:** Antigravity (DeepMind Advanced Agentic Coding)  
**Workspace:** `C:\Users\piyus\Downloads\eBay-ML-competition`  
**Current Official Quiz Champion:** Phase 35C Vehicle 0.80 Challenger ($F_{0.2} = \mathbf{0.9159834927416699}$)  
**Research Report File:** `docs/phase_36a_vehicle_boundary_research.md`  
**Research Data Artifact:** `data/cache/phase_36a_boundary_research_results.json`  
**Unit Test File:** [`tests/test_phase_36a_vehicle_boundary_research.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/tests/test_phase_36a_vehicle_boundary_research.py) (All 5 passed; full test suite 69/69 passed)  
**Drive C: Free Space:** 4.53 GB ($\ge 4.5$ GB maintained)

---

## Executive Summary

Phase 36A investigated whether conservative entity-boundary post-processing rules could improve the precision-oriented $F_{0.2}$ score of our current official competition champion (Phase 35C DAPT Four-Model Ensemble with vehicle-model threshold 0.80, which achieved an official EvalAI Quiz score of **0.9159834927416699**).

Across the 489-title clean holdout, we performed an exhaustive error analysis of all vehicle-model predictions at threshold 0.80, and independently simulated three conservative boundary correction rules:
1. **Rule A (Parenthetical Chassis Suffix):** Strip trailing parenthetical chassis codes (e.g., `"VITO (W447)"` $\rightarrow$ `"VITO"`).
2. **Rule B (Generation Suffix):** Strip trailing generation indicators (e.g., `"Vectra B"` $\rightarrow$ `"Vectra"`, `"Golf IV"` $\rightarrow$ `"Golf"`).
3. **Rule C (Punctuation Cleanup):** Strip extraneous leading/trailing punctuation (e.g., `"C-Max,"` $\rightarrow$ `"C-Max"`, `"-Golf-"` $\rightarrow$ `"Golf"`).

### Key Scientific Findings

1. **Every Proposed Rule Causes Severe Net Regressions:**
   - **Rule A (Chassis Suffix):** Fixes 6 errors, but **damages 14 already-correct predictions**, causing a net loss of $-8$ True Positives and dropping $F_{0.2}$ by **$-0.002431$** ($0.933651 \rightarrow 0.931220$).
   - **Rule B (Generation Suffix):** Fixes 3 errors, but **damages 55 already-correct predictions**, causing a net loss of $-52$ True Positives and dropping $F_{0.2}$ by **$-0.015804$** ($0.933651 \rightarrow 0.917847$).
   - **Rule C (Punctuation Cleanup):** Fixes 0 errors, but **damages 28 already-correct predictions**, causing a net loss of $-28$ True Positives and dropping $F_{0.2}$ by **$-0.008510$** ($0.933651 \rightarrow 0.925141$).
2. **Root Cause of Rule Failure is Token-Level Ground-Truth Grounding:**
   In the competition dataset, titles are evaluated against exact human token spans over whitespace-delimited words. Analysis of all 9,357 vehicle models in the 5,000 labeled titles reveals that human annotators frequently and deliberately **include** chassis codes (193 gold entities), generation codes (653 gold entities), and token-attached punctuation (87 gold entities). Heuristic post-processing rules cannot predict annotator boundary choices and inevitably destroy legitimate exact matches.
3. **Definitive Decision:**
   $$\mathbf{RETAIN\ PHASE\ 35C\ CHAMPION}$$
   No boundary correction rule satisfies the acceptance criteria. The Phase 35C champion remains our frozen, optimal configuration for final competition submission.

---

## Part A — Verification of Research Inputs

All research inputs were verified prior to running simulations:
1. **Evaluator Correctness:** Evaluator ordering strictly confirmed as `count_matches(predicted_entities, true_entities)` in [`src/evaluation/entity_metrics.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/evaluation/entity_metrics.py#L185).
2. **Holdout Provenance:** 489-title clean holdout verified via `data/cache/phase_35b_clean_holdout_manifest.json` (SHA-256 of IDs: `d37f2e9642d1908c48f8f2e71cfafc7fc446a678484e565985b98b955c45e548`), confirmed 100% free of supervised training data across all five models.
3. **Cached Predictions:** Loaded token probabilities from `data/cache/phase_35a_val_1k_probs.pt` (no new GPU inference required).
4. **Frozen Champion Configuration:**
   - DAPT GBERT seed 42: 0.40
   - XLM-R seed 42: 0.20
   - XLM-R seed 7: 0.20
   - XLM-R seed 21: 0.20
   - `Kompatibles_Fahrzeug_Modell` threshold: **0.80**
   - All other 28 aspects: **0.60**
   - Category-aware filtering: Method A (Category 1: 23 aspects, Category 2: 20 aspects).
5. **Baseline Holdout Reproduction:**
   - True Positives (TP): **3,072**
   - False Positives (FP): **199**
   - False Negatives (FN): **701**
   - Total Gold Entities: **3,773** ($3,072 + 701 \equiv 3,773$)
   - Precision: **0.939162**
   - Recall: **0.814206**
   - **Holdout $F_{0.2}$:** **0.933651** (Exact reproduction of Phase 35C baseline).

---

## Part B — Vehicle-Model Error Analysis at Threshold 0.80

Under the current champion threshold of 0.80 on `Kompatibles_Fahrzeug_Modell`, the model generated **547 vehicle-model predictions** on the 489 clean holdout titles:
- **473 True Positives (86.47%)**
- **74 False Positives (13.53%)**

All 74 False Positives were classified into 8 mutually exclusive failure categories:

| Error Category | FP Count | Share of FPs | Description / Typical Failure Mechanism |
|:---|:---:|:---:|:---|
| **Missing Model Tokens** | **29** | 39.2% | Model extracted partial string (e.g. `"Grand"` vs. gold `"Grand Scenic III"`). |
| **Multi-Token Boundary Error** | **24** | 32.4% | Model merged adjacent trim, engine, or multiple models (e.g. `"FOCUS 3 2.0 ST"` vs. gold `"FOCUS"`). |
| **Parenthetical Chassis Code** | **7** | 9.5% | Model included trailing chassis code when gold omitted it (e.g. `"Croma (154)"` vs. gold `"Croma"`). |
| **Generation Suffix** | **5** | 6.8% | Model included generation letter when gold omitted it (e.g. `"Vectra B"` vs. gold `"Vectra"`). |
| **Leading / Trailing Punctuation** | **4** | 5.4% | Trailing delimiter boundary mismatch (e.g. `"A4 B5 Lim."` vs. gold `"A4 B5"`). |
| **Incorrect Aspect Label** | **3** | 4.1% | Classification confusion (e.g. `"Cupra"` predicted as Model, but gold tagged as Marke). |
| **Extra Model Variants** | **1** | 1.4% | Minor trim variant addition (e.g. `"500 312"` vs. gold `"500"`). |
| **Genuine False Alarm** | **1** | 1.4% | Entity predicted where no vehicle model exists in ground truth (e.g. `"Avant"`). |
| **Total** | **74** | **100.0%** | |

### Key Observation
Chassis and generation suffix errors account for only **12 out of 74 FPs (16.2%)** at threshold 0.80. The vast majority of remaining errors (71.6%) are missing model components or complex multi-token merges.

---

## Part C — Independent Simulation of Three Boundary Rules

Each rule was evaluated independently on the 489 clean holdout titles by applying it strictly to `Kompatibles_Fahrzeug_Modell` predictions:

### 1. Overall Metric Comparison Table

| Rule / Configuration | Changed | Fixed | Damaged | TP | FP | FN | Precision | Recall | Holdout F0.2 | $\Delta F_{0.2}$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Phase 35C Champion (Baseline)** | — | — | — | **3,072** | **199** | **701** | **0.939162** | **0.814206** | **0.933651** | **Baseline** |
| **Rule A: Parenthetical Chassis** | 22 | 6 | **14** | 3,064 | 207 | 709 | 0.936717 | 0.812086 | 0.931220 | **-0.002431** |
| **Rule B: Generation Suffix** | 63 | 3 | **55** | 3,020 | 251 | 753 | 0.923265 | 0.800424 | 0.917847 | **-0.015804** |
| **Rule C: Punctuation Cleanup** | 41 | 0 | **28** | 3,044 | 227 | 729 | 0.930602 | 0.806785 | 0.925141 | **-0.008510** |

*All three rules result in a strict decline in True Positives, an increase in False Positives and False Negatives, and a net drop in $F_{0.2}$.*

### 2. Category Breakdown

#### Category 1 (Bremsenteile — 305 Titles, 2,530 Gold Entities)
- **Baseline:** TP = 2,050, FP = 144, FN = 480, P = 0.934366, R = 0.810277, $F_{0.2} = \mathbf{0.928895}$
- **Rule A:** TP = 2,044 ($-6$), FP = 150 ($+6$), $F_{0.2} = \mathbf{0.926176}$ ($\Delta = \mathbf{-0.002719}$)
- **Rule B:** TP = 2,006 ($-44$), FP = 188 ($+44$), $F_{0.2} = \mathbf{0.908958}$ ($\Delta = \mathbf{-0.019937}$)
- **Rule C:** TP = 2,024 ($-26$), FP = 170 ($+26$), $F_{0.2} = \mathbf{0.917114}$ ($\Delta = \mathbf{-0.011781}$)

#### Category 2 (Motorenteile — 184 Titles, 1,243 Gold Entities)
- **Baseline:** TP = 1,022, FP = 55, FN = 221, P = 0.948932, R = 0.822204, $F_{0.2} = \mathbf{0.943340}$
- **Rule A:** TP = 1,020 ($-2$), FP = 57 ($+2$), $F_{0.2} = \mathbf{0.941494}$ ($\Delta = \mathbf{-0.001846}$)
- **Rule B:** TP = 1,014 ($-8$), FP = 63 ($+8$), $F_{0.2} = \mathbf{0.935956}$ ($\Delta = \mathbf{-0.007384}$)
- **Rule C:** TP = 1,020 ($-2$), FP = 57 ($+2$), $F_{0.2} = \mathbf{0.941494}$ ($\Delta = \mathbf{-0.001846}$)

*Both Category 1 and Category 2 suffer regressions across all three rules.*

---

## Part D — Detailed Failure Analysis and Counterexamples

### 1. Why Rule A (Chassis Suffix) Failed: Damaged 14 Predictions to Fix 6
While Rule A successfully fixed 6 predictions where the annotator split the model name from the chassis code (e.g. `"VITO (W447)"` $\rightarrow$ `"VITO"`, Record 4561), it destroyed **14 already-correct predictions** where the ground truth explicitly required the parenthetical chassis:

#### Counterexamples Damaged by Rule A
- **Record 1332:** Title: `"... für BMW 3er (E91)"`  
  - Baseline Prediction: `"3er (E91)"` | Gold: `["3er (E91)"]` $\rightarrow$ **100% Correct Match!**  
  - Rule A Output: `"3er"` $\rightarrow$ **Damaged** (Created 1 FP and 1 FN).
- **Record 3374:** Title: `"... passend für Fiat Panda (169)"`  
  - Baseline Prediction: `"Panda (169)"` | Gold: `["Panda (169)"]` $\rightarrow$ **100% Correct Match!**  
  - Rule A Output: `"Panda"` $\rightarrow$ **Damaged**.
- **Record 458:** Title: `"... für Alfa Romeo Mito (955) Fiorino Qubo"`  
  - Baseline Prediction: `"Mito (955)"` | Gold: `["Mito (955)", "Fiorino", "Qubo"]` $\rightarrow$ **100% Correct Match!**  
  - Rule A Output: `"Mito"` $\rightarrow$ **Damaged**.
- **Record 1957:** Title: `"... Mercedes-Benz GLK-Klasse (X204)"`  
  - Baseline Prediction: `"GLK-Klasse (X204)"` | Gold: `["GLK-Klasse (X204)"]` $\rightarrow$ **100% Correct Match!**  
  - Rule A Output: `"GLK-Klasse"` $\rightarrow$ **Damaged**.
- **Record 2981:** Title: `"... für AUDI S3 (8V) Golf 7 Gti R"`  
  - Baseline Prediction: `"S3 (8V)"` | Gold: `["S3 (8V)", "Golf 7 Gti R"]` $\rightarrow$ **100% Correct Match!**  
  - Rule A Output: `"S3"` $\rightarrow$ **Damaged**.

### 2. Why Rule B (Generation Suffix) Failed: Damaged 55 Predictions to Fix 3
Rule B had a catastrophic damage-to-fix ratio of **18.3 : 1** (55 damaged vs. 3 fixed).
In the vast majority of cases in automotive listings, the generation code is considered integral to the vehicle model designation:

#### Counterexamples Damaged by Rule B
- **Record 4904:** Title: `"... AUDI A3,Q3 und Skoda Superb III ..."`  
  - Baseline Prediction: `"Superb III"` | Gold: `["A3,Q3", "Superb III"]` $\rightarrow$ **Correct!**  
  - Rule B Output: `"Superb"` $\rightarrow$ **Damaged**.
- **Record 4534:** Title: `"... für NISSAN QASHQAI II RENAULT KADJAR"`  
  - Baseline Prediction: `"QASHQAI II"` | Gold: `["QASHQAI II", "KADJAR"]` $\rightarrow$ **Correct!**  
  - Rule B Output: `"QASHQAI"` $\rightarrow$ **Damaged**.
- **Record 64:** Title: `"... Ford C-Max, Focus II Satz für vorne"`  
  - Baseline Prediction: `"Focus II"` | Gold: `["C-Max,", "Focus II"]` $\rightarrow$ **Correct!**  
  - Rule B Output: `"Focus"` $\rightarrow$ **Damaged**.
- **Record 3400:** Title: `"... passend für KIA PICANTO 3 JA III"`  
  - Baseline Prediction: `"PICANTO 3 JA III"` | Gold: `["PICANTO 3 JA III"]` $\rightarrow$ **Correct!**  
  - Rule B Output: `"PICANTO 3 JA"` $\rightarrow$ **Damaged**.

### 3. Why Rule C (Punctuation Cleanup) Failed: Damaged 28 Predictions to Fix 0
Rule C achieved **zero fixed errors** while damaging 28 correct predictions.
Because ground-truth tokens are strictly whitespace-delimited, trailing punctuation is often legitimately part of the token sequence:

#### Counterexamples Damaged by Rule C
- **Record 4958:** Title: `"Bosch Bremsscheiben und Bremsbeläge Mercedes A- und B-Klasse vorne u. hinten"`  
  - Baseline Prediction: `"A-"` | Gold: `["A-", "B-Klasse"]` $\rightarrow$ **Correct!** (German hyphenated elision).  
  - Rule C Output: `"A"` $\rightarrow$ **Damaged**.
- **Record 64:** Title: `"... Ford C-Max, Focus II ..."`  
  - Baseline Prediction: `"C-Max,"` | Gold: `["C-Max,", "Focus II"]` $\rightarrow$ **Correct!**  
  - Rule C Output: `"C-Max"` $\rightarrow$ **Damaged**.

---

## Part E — Corpus-Level Ground Truth Consistency Analysis

To confirm that these findings are not holdout-specific artifacts, we evaluated the entire 5,000 labeled training dataset (**9,357 total gold vehicle-model entities**):

| Pattern in Ground Truth | Total Gold Entities | Percentage of Vehicle Models | Representative Ground-Truth Examples |
|:---|:---:|:---:|:---|
| **Parenthetical Chassis Code** | **193** | 2.06% | `"A4 (B9)"`, `"X5 (E53)"`, `"E-Klasse (S211)"`, `"5 Limousine (E39)"` |
| **Roman Numeral Generation** | **327** | 3.49% | `"Golf V"`, `"Superb III"`, `"Focus II"`, `"Qashqai II"` |
| **Single Letter Generation** | **326** | 3.48% | `"Zafira B"`, `"Astra H"`, `"Corsa E"`, `"Astra J"`, `"Omega B"` |
| **Leading / Trailing Punctuation** | **87** | 0.93% | `"A-"`, `",A5 (F5)"`, `"C-Max,"` |

### Synthesis
In over **933 gold entities** across the training data, chassis codes, roman numerals, and single letters are explicitly tagged as part of the vehicle model name. Stripping them with post-processing rules guarantees widespread damage on unseen competition data.

---

## Part F — Decision and Final Recommendation

```
================================================================================
FINAL PHASE 36A RECOMMENDATION:
RETAIN PHASE 35C CHAMPION
================================================================================
```

### Decision Justification
1. **Rule Performance Criterion Not Met:** None of the three rules fixed more errors than it damaged. In fact, Rule A had a 2.3 : 1 damage ratio, Rule B had an 18.3 : 1 damage ratio, and Rule C had a 28 : 0 damage ratio.
2. **Category Degradation:** All rules degraded performance in Category 1 and Category 2 simultaneously.
3. **Generalization Risk:** The rules attempt to enforce an artificial standardization on ground truth that does not exist in the competition annotations.
4. **Current Champion Strength:** The Phase 35C DAPT four-model ensemble with vehicle threshold 0.80 already achieved an exceptional **0.9159834927416699** on official EvalAI Quiz. Applying heuristic string stripping would degrade this performance.

---

## Artifacts and Git Summary

- **Research Simulation Pipeline:** [`src/evaluation/phase_36a_boundary_research.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/evaluation/phase_36a_boundary_research.py)
- **Unit Tests:** [`tests/test_phase_36a_vehicle_boundary_research.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/tests/test_phase_36a_vehicle_boundary_research.py) (5 passed)
- **Simulation Results JSON:** `data/cache/phase_36a_boundary_research_results.json` (git-ignored)
- **Report:** `docs/phase_36a_vehicle_boundary_research.md`
