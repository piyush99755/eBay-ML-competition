# Phase 46 — Entity Boundary Specialist and Sixth-Model Ensemble Feasibility & Diagnostic Report

## Executive Summary

- **Objective:** Investigate whether adding a fifth or sixth model to our active ensemble—specifically targeted at boundary and exact-span endings—can provide complementary signal and improve official $F_{0.2}$ performance without endangering precision.
- **Official Champion Protected:** Phase 40 Quiz Champion TSV [`submissions/submission_quiz_phase40_lieferumfang080.tsv`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/submissions/submission_quiz_phase40_lieferumfang080.tsv) verified bit-for-bit:
  - Expected SHA-256: `BF125FF2F4D7CBA79636739CF4C7FF66C7F42A4462531245E3104045961461E8`
  - Official EvalAI Quiz $F_{0.2}$: `0.9233317116970503`
  - Local Clean-Holdout Baseline (489 titles): Overall **`0.927771`** (Category 1: `0.918323`, Category 2: `0.937219`; Micro TP: `3021`, FP: `165`, FN: `752`).
- **Trained Model Inventory:**
  - Active Phase 40 Ensemble: **4 models** (DAPT GBERT seed 42: 0.40, XLM-R seed 42: 0.20, XLM-R seed 7: 0.20, XLM-R seed 21: 0.20).
  - Existing Unused Checkpoints:
    1. `gbert_baseline_best` (Original GBERT seed 42, 418 MB) — Unused 5th model, predictions cached.
    2. `gelectra_baseline_best` (GELECTRA seed 42, 418 MB) — Unused 6th candidate.
    3. `gbert_phase32c_challenger_best` (GBERT with 1.15x vehicle continuation loss, 418 MB) — Unused boundary candidate.
    4. `xlmr_span_seed42_best` (XLM-R span head model, 1,082 MB) — Unused span candidate.
- **Boundary Failure Modes Audit (Clean Holdout $N=489$):**
  - Mode 1 (Missing final token): **47 cases** (Cat 1: 37, Cat 2: 10; 24 in `Kompatibles_Fahrzeug_Modell`).
  - Mode 2 (Extra final token): **1 case** (Cat 1: 1, Cat 2: 0).
  - Mode 3 (Missing leading token): **6 cases** (Cat 1: 4, Cat 2: 2).
  - Mode 4 (Extra leading token): **0 cases**.
  - Mode 5 (Incorrect merging of adjacent entities): **42 cases** (Cat 1: 31, Cat 2: 11; 35 in `Kompatibles_Fahrzeug_Modell`).
  - Mode 6 (Incorrect splitting of a single entity): **15 cases** (Cat 1: 11, Cat 2: 4).
  - Mode 7 (Wrong aspect despite correct boundaries): **31 cases** (Cat 1: 21, Cat 2: 10).
- **Root Cause Analysis:**
  Boundary errors in German automotive listings do NOT stem from subword tokenization truncation or simple sequence cutoff. They stem directly from **human annotation multimodality** in sub-model designations (e.g. `Golf 4`, `A6 C7`, `Corsa A`, `Mondeo III`). In the ground truth training annotations, these are labeled inconsistently—sometimes as a single unified `B-I-` span and sometimes as consecutive separate `B-B-` spans. The ensemble's probability mass is evenly split between `B-` and `I-` tags, causing accidental merging when `I-` dominates and accidental splitting/missing tokens when `B-` dominates.
- **Model Complementarity & Ensemble Comparison:**
  - Standalone `orig_gbert` achieves $F_{0.2} = 0.904415$ with 263 FPs (vs. 165 in Phase 40). While it fixes 130 errors in gold, it simultaneously damages 114 correct entities in gold.
  - Standalone `gelectra_baseline` achieves $F_{0.2} = 0.872451$ with 296 FPs. When fused at weight 0.05, it drops overall score by $\Delta = -0.002183$.
  - Standalone `gbert_phase32c_challenger` achieves $F_{0.2} = 0.898950$. When fused at weight 0.10, Category 1 drops sharply by $\Delta = -0.002303$.
  - Fusing `orig_gbert` at weight 0.10 yields Overall $0.928010$ ($\Delta = +0.000239$, far below the $+0.0020$ gate), with Category 1 regressing by $-0.000821$.
  - Fusing `orig_gbert` at weight 0.20 (Phase 35A 5-model equal weighting) degrades the score to $0.926162$ ($\Delta = -0.001608$, with Category 1 dropping by $-0.003738$).
- **Bootstrap Uncertainty (1,000 resamples):**
  - Candidate 1 (weight 0.10): 95% CI is `[-0.001894, +0.002691]` (crosses zero; 43.2% probability of non-positive delta; Category 1 mean delta is $-0.000708$).
  - Candidate 2 (weight 0.20): 95% CI is `[-0.005750, +0.001966]` (78.6% probability of degradation; Category 1 mean delta is $-0.003618$).
- **Training Feasibility & Constraints:**
  - Hardware: NVIDIA GTX 1660 Ti (6 GB VRAM), 16 GB RAM.
  - Disk Space: 7.91 GB free (tightly constrained; standard safety floor is $\ge 5.0$ GB).
  - Time Budget: Approximately 2 days remaining before final competition cutoff.
  - Risk Assessment: Under the $25\times$ false-positive penalty of $F_{0.2}$, any newly trained model must exceed 96.15% break-even precision to avoid destroying the ensemble. Past experience with boundary-weighted loss (Phase 32C) and auxiliary boundary heads (Exp 14) confirms that forced boundary modification increases false positives and degrades the ensemble.
- **Final Decision Verdict:** **`RETAIN PHASE 40 — INSUFFICIENT EVIDENCE`**.

---

## 1. Verified Model Inventory

We audited all 17 directories in `checkpoints/` and cataloged their exact architectures, seeds, training datasets, and sizes on disk:

| Checkpoint Name | Base Architecture | Seed | Training Data Slice | Size (MB) | Active in Phase 40? | Role / Status |
| :--- | :--- | :---: | :--- | :---: | :---: | :--- |
| `phase33b_dapt_gbert_best` | `deepset/gbert-base` (DAPT MLM) | 42 | Sequential `s_tr` (3,200) | 418.2 | **YES (0.40)** | **Active Model 1** |
| `xlmr_baseline_best` | `FacebookAI/xlm-roberta-base` | 42 | Split-A `train_4k` (4,000) | 1,079.7 | **YES (0.20)** | **Active Model 2** |
| `xlmr_seed_7_best` | `FacebookAI/xlm-roberta-base` | 7 | Split-A `train_4k` (4,000) | 1,079.7 | **YES (0.20)** | **Active Model 3** |
| `xlmr_seed_21_best` | `FacebookAI/xlm-roberta-base` | 21 | Split-A `train_4k` (4,000) | 1,079.7 | **YES (0.20)** | **Active Model 4** |
| `gbert_baseline_best` | `deepset/gbert-base` | 42 | Split-A `train_4k` (4,000) | 418.2 | No (0.00) | **Unused 5th Model** (Exp 10 / Phase 35A) |
| `gelectra_baseline_best` | `deepset/gelectra-base` | 42 | Split-A `train_4k` (4,000) | 418.2 | No (0.00) | **Unused 6th Model** (Phase 25B) |
| `gbert_phase32c_challenger_best` | `deepset/gbert-base` (1.15x I-Modell) | 42 | Internal train (3,200) | 418.2 | No (0.00) | **Unused Boundary Model** (Phase 32C) |
| `gbert_phase32c_baseline_best` | `deepset/gbert-base` (1.00x loss) | 42 | Internal train (3,200) | 418.2 | No (0.00) | Unused Baseline (Phase 32C) |
| `xlmr_span_seed42_best` | `xlm-roberta-base` + span heads | 42 | Split-A `train_4k` (4,000) | 1,082.1 | No (0.00) | Unused Span Model (Exp 16) |
| `phase33b_dapt_encoder` | `BertForMaskedLM` | 42 | 50k unlabelled titles | 420.4 | No (0.00) | Pretrained backbone (no token classifier) |
| `gbert_cv3_fold[0-2]_best` (3 models) | `deepset/gbert-base` | 0 | 3-Fold CV on `train_4k` | 418.2 ea | No (0.00) | CV Fold Models (Phase 29B) |
| `[gbert/xlmr]_fulldata_*` (4 models) | GBERT / XLM-R (Seeds 42, 7, 21) | 42, 7, 21 | Full 5,000 titles | ~418–1080 | No (0.00) | Leaked on validation holdout (Phase 22) |

### Key Inventory Takeaway:
Phase 40 contains **exactly 4 models** (DAPT GBERT: 0.40, XLM-R seeds 42, 7, 21: 0.20 each).
An unused 5th model (`gbert_baseline_best`) and 6th model (`gelectra_baseline_best`) already exist with full weights on disk and cached validation probabilities.

---

## 2. Quantitative Boundary Failure Modes Audit

Using the 489 clean holdout titles (never seen by any baseline model during supervised training), all predicted entity spans from Phase 40 were compared with ground-truth entity spans to classify all boundary discrepancies into 7 mutually exclusive failure modes:

| Failure Mode Description | Category 1 | Category 2 | Total Errors | Primary Affected Aspects |
| :--- | :---: | :---: | :---: | :--- |
| **1. Correct entity, missing final token** | **37** | **10** | **47** | `Kompatibles_Fahrzeug_Modell` (24), `Im_Lieferumfang_Enthalten` (12), `Einbauposition` (5), `Produktart` (4) |
| **2. Correct entity, extra final token** | **1** | **0** | **1** | `Kompatibles_Fahrzeug_Modell` (1) |
| **3. Correct entity, missing leading token** | **4** | **2** | **6** | `Kompatibles_Fahrzeug_Modell` (3), `Im_Lieferumfang_Enthalten` (2), `Kompatibles_Fahrzeug_Jahr` (1) |
| **4. Correct entity, extra leading token** | **0** | **0** | **0** | None |
| **5. Incorrect merging of adjacent entities** | **31** | **11** | **42** | `Kompatibles_Fahrzeug_Modell` (35), `Produktart` (6), `Kompatible_Fahrzeug_Marke` (1) |
| **6. Incorrect splitting of a single entity** | **11** | **4** | **15** | `Im_Lieferumfang_Enthalten` (6), `Kompatibles_Fahrzeug_Modell` (5), `Produktart` (1) |
| **7. Wrong aspect despite correct boundaries** | **21** | **10** | **31** | `Im_Lieferumfang_Enthalten` vs `Produktart` (9), `Herstellernummer` vs `OE/OEM` (4) |
| **Total Boundary Discrepancies** | **105** | **37** | **142** | — |

### Asymmetry of Errors: Missing vs. Extra
- Missing final token (**47 cases**) dwarfs extra final token (**1 case**).
- Extra leading token is **0 cases**, while missing leading token is only **6 cases**.
- Merging adjacent entities (**42 cases**) and splitting entities (**15 cases**) represent significant structural boundary errors.

---

## 3. Qualitative Root Cause Analysis of Boundary Errors

A forensic token-level inspection of the 47 missing final token cases and 42 merging cases revealed the underlying mechanisms:

### 3.1 Why Final Tokens Are Missing (Mode 1: 47 cases)
Contrary to initial assumptions, missing final tokens do NOT occur because the model's vocabulary or context window cut off the word, nor because the token was simply classified as `O`:
1. **Sub-Entity Re-initialization via `B-` Collision (64% of cases):**
   - Example (Rec 4299): Gold is `Kompatibles_Fahrzeug_Modell` = `"Grand Scenic III"`.
   - The model predicts `"Grand"` as `B-Kompatibles_Fahrzeug_Modell`.
   - At token `"Scenic"`, the fused prediction is `B-Kompatibles_Fahrzeug_Modell` (confidence 0.522) against `I-` (confidence 0.468).
   - Under standard BIO semantics, receiving a new `B-` tag immediately forces the decoder to terminate `"Grand"` and start a new entity `"Scenic III"`.
   - The original entity is now missing its final tokens (`"Scenic III"`), and the second entity (`"Scenic III"`) is either filtered out if below 0.80 confidence or accepted as an ungrounded split.
2. **Annotation Ambiguity in Ground Truth (26% of cases):**
   - Example (Rec 1332): Gold has `"Beläge Scheiben"` tagged as a single `Im_Lieferumfang_Enthalten` span.
   - All 4 models assign `"Beläge"` to `B-Im_Lieferumfang_Enthalten` and `"Scheiben"` to `B-Im_Lieferumfang_Enthalten` (91% confidence).
   - The models split the two items, causing `"Beläge"` to be scored as missing its final token.
3. **True Low Confidence / Threshold Suppression (10% of cases):**
   - In rare cases (e.g. Rec 1558: `"GT AB"`), the token `"AB"` receives only 0.34 confidence on `I-Modell` and falls below threshold.

### 3.2 Why Adjacent Entities Are Merged (Mode 5: 42 cases)
- In 35 out of 42 merging cases, the aspect is `Kompatibles_Fahrzeug_Modell`.
- In German automotive listings, titles frequently string together a car model and its generation or chassis code (e.g. `"Golf"` + `"4 Variant"`, `"Corolla"` + `"E12"`, `"Q5"` + `"(8RB)"`, `"E87"` + `"118 i"`).
- In human ground truth, annotators split these into two adjacent entities: `[Golf] (B-Modell)` and `[4 Variant] (B-Modell)`.
- However, the ensemble models predict the entire sequence continuously: `[Golf] (B-Modell)`, `[4] (I-Modell)`, `[Variant] (I-Modell)`, merging both into a single entity `"Golf 4 Variant"`.
- This single predicted entity fails exact multi-set matching against both gold entities, creating 1 False Positive and 2 False Negatives simultaneously.

---

## 4. Existing Model Complementarity Check

We evaluated three existing, fully trained, unused checkpoints on the clean holdout titles to determine whether their predictions provide genuine boundary complementarity:

### 4.1 Standalone Performance vs Phase 40 Baseline

| Model / Checkpoint | Checkpoint Path | Overall $F_{0.2}$ | Cat 1 $F_{0.2}$ | Cat 2 $F_{0.2}$ | TP | FP | FN | Micro $F_{0.2}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 40 Baseline (4-Model)** | Frozen Ensemble | **`0.927771`** | **`0.918323`** | **`0.937219`** | **3,021** | **165** | **752** | **0.941539** |
| **Original GBERT seed 42** | `gbert_baseline_best` | `0.904415` | `0.896473` | `0.912357` | 3,037 | **263** | 736 | 0.915257 |
| **GELECTRA seed 42** | `gelectra_baseline_best` | `0.872451` | `0.850800` | `0.894102` | 3,017 | **296** | 756 | 0.905818 |
| **GBERT Phase 32C Challenger** | `gbert_phase32c_challenger_best` | `0.898950` | `0.883139` | `0.914761` | 3,081 | **301** | 692 | 0.906966 |

### 4.2 Error Complementarity Analysis: Fixed vs. Damaged Spans
For `orig_gbert` vs. Phase 40:
- **Errors in Gold Fixed by `orig_gbert`:** 130 spans (Top: `Kompatibles_Fahrzeug_Modell`: 74, `Produktart`: 21, `Einbauposition`: 13).
- **Correct Gold Spans Damaged by `orig_gbert`:** 114 spans (Top: `Kompatibles_Fahrzeug_Modell`: 57, `Im_Lieferumfang_Enthalten`: 28, `Einbauposition`: 15).
- **Net Spans Gained:** $+16$ spans.
- **False Positive Overhead:** `orig_gbert` introduces **+98 extra False Positives** (263 vs. 165). Under $F_{0.2}$, where 1 FP cancels 25 TPs, 98 extra FPs would require $+2,450$ additional TPs to break even!

---

## 5. Controlled Ensemble Comparisons

We evaluated blending each candidate checkpoint with the Phase 40 ensemble across predeclared small candidate weights ($w \in \{0.05, 0.10, 0.15, 0.20\}$):

### 5.1 Fusing Original GBERT (`gbert_baseline_best`)

| Weight ($w$) | Overall $F_{0.2}$ | Delta ($\Delta$) | Cat 1 Score | Cat 1 $\Delta$ | Cat 2 Score | Cat 2 $\Delta$ | TP | FP | FN |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.00 (Baseline)** | **`0.927771`** | `+0.000000` | **`0.918323`** | `+0.000000` | **`0.937219`** | `+0.000000` | 3,021 | 165 | 752 |
| **0.05** | `0.926581` | $-0.001189$ | `0.916551` | $-0.001772$ | `0.936612` | $-0.000607$ | 3,013 | 168 | 760 |
| **0.10** | `0.928010` | $+0.000239$ | `0.917501` | **$-0.000821$** | `0.938519` | $+0.001300$ | 3,012 | 164 | 761 |
| **0.15** | `0.927320` | $-0.000451$ | `0.916569` | $-0.001754$ | `0.938071` | $+0.000852$ | 3,009 | 166 | 764 |
| **0.20 (5-Model Equal)** | `0.926162` | $-0.001608$ | `0.914585` | **$-0.003738$** | `0.937740` | $+0.000521$ | 3,010 | 170 | 763 |

*Key Result:*
- The highest score observed is at weight 0.10 ($\Delta = +0.000239$), which is an order of magnitude smaller than the required $+0.0020$ promotion threshold.
- Crucially, at weight 0.10, **Category 1 suffers a regression of $-0.000821$**, and overall true positives drop from 3,021 to 3,012.
- At weight 0.20 (the classic 5-model ensemble evaluated in Phase 35A), the ensemble regresses by $-0.001608$, with Category 1 collapsing by $-0.003738$.

### 5.2 Fusing GELECTRA (`gelectra_baseline_best`)
- Weight 0.05: Overall `0.925588` ($\Delta = -0.002183$, Cat 1 $\Delta = -0.003379$).
- Weight 0.10: Overall `0.923324` ($\Delta = -0.004447$, Cat 1 $\Delta = -0.006708$).
- GELECTRA consistently degrades the ensemble across all weights due to excessive false positives (+8 to +23 FPs).

### 5.3 Fusing Phase 32C Boundary Challenger (`gbert_phase32c_challenger_best`)
- Weight 0.05: Overall `0.928207` ($\Delta = +0.000437$, Cat 1 $\Delta = -0.001113$).
- Weight 0.10: Overall `0.928892` ($\Delta = +0.001121$, Cat 1 $\Delta = \mathbf{-0.002303}$).
- Although Category 2 gains slightly, Category 1 experiences a severe regression ($-0.002303$), failing the non-regression criterion.

---

## 6. Paired Category-Stratified Bootstrap Analysis

1,000 paired category-stratified bootstrap resamples (seed 42) were conducted on Candidate 1 (weight 0.10) and Candidate 2 (weight 0.20):

| Bootstrap Metric (1,000 resamples) | Candidate 1 ($w=0.10$) | Candidate 2 ($w=0.20$, Equal 5-Model) |
| :--- | :---: | :---: |
| **Mean Overall Delta** | `+0.000266` | `-0.001630` |
| **95% Confidence Interval (Overall)** | **`[-0.001894, +0.002691]`** | **`[-0.005750, +0.001966]`** |
| **Fraction of Resamples with $\Delta \le 0$** | **`43.2%`** | **`78.6%`** |
| **Mean Category 1 Delta** | **`-0.000708`** | **`-0.003618`** |
| **95% Confidence Interval (Category 1)** | `[-0.004440, +0.003171]` | `[-0.008162, +0.000830]` |

The bootstrap confidence interval for Candidate 1 crosses zero broadly and includes substantial negative density (`-0.001894`). More than 43% of resamples show active degradation, and the mean delta on Category 1 is negative ($-0.000708$). Candidate 2 degrades performance in 78.6% of resamples.

---

## 7. Sixth-Model Training Feasibility Assessment

We evaluated the four proposed architectural approaches for training a new sixth model:

| Approach | Description | Technical Feasibility | Estimated Runtime | Risk Assessment under $F_{0.2}$ |
| :--- | :--- | :---: | :---: | :--- |
| **A. Different Architecture** (e.g. ModernBERT / German-GPT) | Download and fine-tune novel architecture | Low | 4–6 hours | **High Risk**: Subword alignment mismatches, token representation incompatibility, high VRAM requirements, unvalidated on competition data. |
| **B. Boundary-Objective Fine-Tuning** (Loss re-weighting on `I-Modell`) | Re-tune GBERT with boundary loss | Medium | 1.5–2.5 hours | **High Risk**: Empirically tested in Phase 32C. Caused asymmetric precision drop (-0.0016 precision) that erased all recall gains under $25\times$ FP penalty. |
| **C. Supervised Boundary Post-Tagger** | Train CRF / secondary classifier on training annotations | Low | 3–5 hours | **High Risk**: Secondary tagger introduces boundary churn; tested in Phase 42B and Phase 45 with severe negative deltas. |
| **D. Existing Historical Checkpoints** | Evaluate `orig_gbert`, `gelectra`, `phase32c` | **High (Complete)** | **Immediate** | **Empirically Evaluated**: All three existing checkpoints fail the promotion criteria. |

### Resource & Safety Limits:
- **Free Disk Space:** 7.91 GB. Initiating full model training with optimizer states, intermediate checkpoints, and logging risks violating the 5.0 GB safety threshold.
- **Remaining Time:** ~2 days before final test preparation. Committing 4–6 hours to training a high-variance model with a known precision bottleneck is strategically contraindicated.

---

## 8. Promotion Rule Evaluation

| Promotion Criterion | Requirement | Observed Result (Best Candidate: $w=0.10$) | Status |
| :--- | :--- | :--- | :---: |
| **1. Net Score Delta** | Overall $\Delta \ge +0.0020$ | $\Delta = +0.000239$ | **FAILED** |
| **2. Boundary Error Ratio** | Boundary corrections $>$ new boundary errors | Fixed 130 vs Damaged 114 (+16 net), but +98 FPs | **FAILED** |
| **3. Category Protection** | $\Delta \text{Cat 1} \ge 0$ AND $\Delta \text{Cat 2} \ge 0$ | Cat 1: **$-0.000821$**, Cat 2: $+0.001300$ | **FAILED** |
| **4. True Positive Preservation** | No substantial loss of existing TPs | TP decreased from 3,021 to 3,012 ($-9$ TPs) | **FAILED** |
| **5. Statistical Confidence** | 95% Bootstrap CI strictly positive | 95% CI is `[-0.001894, +0.002691]` (43.2% $\le 0$) | **FAILED** |
| **6. Resource & Safety** | $\ge 5.0$ GB disk, feasible within 2 days | 7.91 GB disk intact, 0 MB GPU allocated | **PASSED** |
| **7. Champion Integrity** | Champion files completely unaltered | SHA-256 `BF125FF2F4...` verified intact | **PASSED** |

---

## 9. Deliverables & Repository Verification

1. **Evaluation Pipeline:** [`src/evaluation/phase_46_boundary_specialist_evaluation.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/evaluation/phase_46_boundary_specialist_evaluation.py)
   - Reproducible script computing full checkpoint inventory, boundary failure mode counts, unused model complementarity, ensemble fusion sweeps, and 1,000 paired bootstrap resamples.
   - Outputs cached to `data/cache/phase_46_boundary_evaluation_results.json`.
2. **Unit Test Suite:** [`tests/test_phase_46_boundary_specialist.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/tests/test_phase_46_boundary_specialist.py)
   - 8 tests covering champion integrity, baseline reproduction, inventory verification, failure modes, complementarity, ensemble fusion, bootstrap metrics, and decision verdict.
   - **Full test suite passes:** 156 tests passing in 31.9s (`OK`).
3. **Documentation:** [`docs/phase_46_boundary_specialist_feasibility.md`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/docs/phase_46_boundary_specialist_feasibility.md).
4. **Champion Submission Preservation:** `submissions/submission_quiz_phase40_lieferumfang080.tsv` untouched (SHA-256 `BF125FF2F4D7CBA79636739CF4C7FF66C7F42A4462531245E3104045961461E8`).
5. **Disk Space Invariant:** **7.91 GB free** (well above 5.0 GB limit).

---

## Final Recommendation

Adding a fifth or sixth model to the Phase 40 ensemble does not provide sufficient complementary precision to improve the official score. The underlying boundary errors (primarily sub-model splitting and merging) arise from human annotation inconsistencies across automotive series designations, rather than model capacity limitations. Diluting the frozen 4-model DAPT champion with existing or newly trained lower-precision models introduces residual false positives that actively degrade the precision-heavy $F_{0.2}$ metric.

With approximately two days remaining before the competition deadline, all engineering resources should be reserved for:
1. Final submission pipeline verification (`src/submission/predict_test_champion.py`).
2. Test set packaging and dry-run execution.
3. Strict preservation of the Phase 40 champion configuration.

**FINAL VERDICT:**

**RETAIN PHASE 40 — INSUFFICIENT EVIDENCE**
