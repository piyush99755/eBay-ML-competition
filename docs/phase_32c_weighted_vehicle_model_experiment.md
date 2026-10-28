# Phase 32C — Controlled I-Modell Loss-Weight Experiment Report

**Date:** 2026-10-27  
**Status:** COMPLETED & AUDITED  
**Final Decision:** `NO MEANINGFUL IMPROVEMENT — RETAIN CHAMPION`  
**Active Production Champion:** `submissions/submission_quiz_exp26b_min060.tsv`  
- Official EvalAI Quiz $F_{0.2}$: **0.901245410919943** (Rank 13)
- Local Split-A Validation $F_{0.2}$: **0.899802** (Micro Precision: 0.9172, Micro Recall: 0.8350)
- Production Inference Pipeline: [`src/submission/predict_test_champion.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/submission/predict_test_champion.py) (Frozen, Phase 28 verified)
- Free Disk Space on Drive C: **7.08 GB** (strictly $\ge 4.0$ GB)

---

## 1. Executive Summary

Phase 32C tested the hypothesis approved in Phase 32B: determining whether applying a modest **1.15x loss weight** specifically to continuation tokens of vehicle models (`I-Kompatibles_Fahrzeug_Modell`) in `CrossEntropyLoss` improves exact-match entity segmentation compared with a strictly matched **1.00x baseline** model.

To prevent overfitting on the repeatedly reused 1,000-title Split-A validation set, both models were trained and selected exclusively on a deterministic, category-stratified internal partition (3,200 internal train / 800 internal dev) derived from the 4,000 Split-A training titles.

### Key Experimental Findings:
1. **Internal Development Gate Passed**:
   - On the internal 800-title dev set, the Challenger achieved $F_{0.2} = \mathbf{0.867637}$ (Epoch 3), outperforming the matched Baseline $F_{0.2} = \mathbf{0.866940}$ (Epoch 2) by $\mathbf{+0.000697}$.
   - Category 1 (automotive) $F_{0.2}$ showed a substantial gain of $\mathbf{+0.004225}$ ($0.884556 \rightarrow 0.888781$).
   - Vehicle-model exact True Positives increased by **+38** ($979 \rightarrow 1,017$), recovering **104 previously missed spans** against **62 regressions** (net $+42$ spans, $+13$ titles improved).
   - Single-token vehicle-model accuracy improved by $+3.82\%$ ($65.28\% \rightarrow 69.10\%$), and multi-token accuracy improved by $+0.65\%$ ($64.93\% \rightarrow 65.58\%$).
2. **Exploratory Holdout & Ensemble Assessment (1,000 Titles)**:
   - Standalone: Challenger outperformed Baseline on the holdout ($0.851865$ vs. $0.851460$, $+0.000405$).
   - 4-Model Ensemble (@ MIN 0.60): When fused with the three frozen XLM-R models, the Challenger ensemble achieved $F_{0.2} = \mathbf{0.896544}$, slightly trailing the matched Baseline ensemble ($F_{0.2} = \mathbf{0.897179}$, $\Delta = -0.000635$).
   - **The Asymmetric Precision Bottleneck**: The 1.15x weighting increased recall (+0.0029), but slightly reduced precision (-0.0016). Under the official metric where $\beta^2 = 0.04$ ($25\times$ precision penalty), this tiny precision dip in soft probability voting neutralizes the recall gain in the ensemble.
   - **Data Volume Penalty**: Crucially, both 3,200-title models trail our frozen 4,000-title champion ($F_{0.2} = \mathbf{0.899802}$ local / $\mathbf{0.901245}$ official Quiz) by approximately $-0.003$.
3. **Decisive Strategic Recommendation**: Retraining the champion on 4,000 titles with 1.15x loss weighting carries asymmetric precision risk with no guaranteed ensemble gain. The frozen Exp 10 + MIN 0.60 champion remains our highest-performing and most stable submission.

```
=============================================================================
FINAL DECISION: NO MEANINGFUL IMPROVEMENT — RETAIN CHAMPION
=============================================================================
```

---

## 2. Step 0 — Preflight and Safety Audit

All safety gates were verified prior to launching training:

| Preflight Check | Requirement | Audit Result | Safety Status |
| :--- | :--- | :--- | :---: |
| **Champion Checkpoints** | All 4 frozen models intact | `gbert_baseline_best`, `xlmr_baseline_best`, `xlmr_seed_7_best`, `xlmr_seed_21_best` all verified valid (`model.safetensors` + `config.json` present) | **PASS** |
| **Production Pipeline** | Unmodified | [`src/submission/predict_test_champion.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/submission/predict_test_champion.py) clean git working tree | **PASS** |
| **Initial Drive C: Free Space** | $\ge 4.00$ GB | **5.35 GB** free | **PASS** |
| **GPU Status** | Idle & available | NVIDIA GeForce GTX 1660 Ti (6.00 GB VRAM, 0 MB allocated) | **PASS** |
| **Training Isolation** | Independent scripts | Dedicated script [`src/modeling/train_phase32c_matched.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/modeling/train_phase32c_matched.py) without touching `train_model.py` | **PASS** |

---

## 3. Step 1 — Internal Split Integrity and Leakage Audit

To prevent subtle overfitting on the repeatedly reused 1,000-title Split-A validation set, we created a deterministic 80/20 partition within the 4,000 Split-A training titles using [`src/data/make_internal_split.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/data/make_internal_split.py):

| Partition | Size | Category 1 | Category 2 | Near-Duplicate Handling | Leakage Audit |
| :--- | :---: | :---: | :---: | :--- | :---: |
| **Internal Train** | **3,200** | 1,600 (50.0%) | 1,600 (50.0%) | Grouped by normalized title | **0 overlap** with Dev |
| **Internal Dev** | **800** | 400 (50.0%) | 400 (50.0%) | Grouped by normalized title | **0 overlap** with Holdout |
| **External Holdout** | **1,000** | 500 (50.0%) | 500 (50.0%) | Untouched Split-A validation | **0 overlap** with Train |

- **Algorithm**: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`.
- **Near-Duplicate Protection**: Grouping by normalized title text ensures identical/near-duplicate listings (such as `Record 1779` and `Record 4204`) are kept strictly within the same fold and never split across train and dev.
- **Manifest**: Saved locally to `data/cache/internal_split_3200_800.json` (untracked, no competition text committed).

---

## 4. Step 2 — Loss Implementation and Unit Tests

The isolated training implementation applies class weights directly in PyTorch `CrossEntropyLoss`:
$$\mathcal{L} = -\frac{1}{\sum_{i: y_i \ne -100} w_{y_i}} \sum_{i: y_i \ne -100} w_{y_i} \log \frac{\exp(z_{i, y_i})}{\sum_j \exp(z_{i, j})}$$

- **Baseline**: $w_c = 1.00$ for all 48 classes.
- **Challenger**: $w_{\text{I-Modell}} = 1.15$, $w_c = 1.00$ for all other 47 classes.
- **Target Label**: `label2id["I-Kompatibles_Fahrzeug_Modell"]` (Label ID 41).

### Unit Test Verification ([`tests/test_phase32c_loss.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/tests/test_phase32c_loss.py)):
Executed via `python -m unittest tests/test_phase32c_loss.py`:
- `test_label_mapping_integrity`: PASSED (48 classes, required aspects verified).
- `test_baseline_weights_are_uniform`: PASSED (exactly 1.00 across all classes).
- `test_challenger_weights_isolate_i_modell`: PASSED (ID 41 is exactly 1.15; all other 47 classes are 1.00).
- `test_ignore_index_masking`: PASSED (`-100` targets generate zero loss).
- `test_weighted_loss_gradient_scaling`: PASSED (proportionally scales loss on `I-Modell` tokens).
- **Result**: **5 / 5 tests passed in 0.098s**.

---

## 5. Step 3 — Training Metrics and Execution

Sequential training executed on the GTX 1660 Ti via [`src/modeling/train_phase32c_matched.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/modeling/train_phase32c_matched.py):

| Training Metric | Matched Baseline (`1.00x`) | Targeted Challenger (`1.15x`) |
| :--- | :---: | :---: |
| **Initial Checkpoint** | `deepset/gbert-base` (Seed 42) | `deepset/gbert-base` (Seed 42) |
| **Epoch 1 Train Loss / Dev Loss** | 0.6031 / 0.3218 | 0.6075 / 0.3240 |
| **Epoch 1 Dev $F_{0.2}$ (Cat 1 / Cat 2)** | 0.840691 (0.8700 / 0.8114) | 0.842763 (0.8710 / 0.8145) |
| **Epoch 2 Train Loss / Dev Loss** | 0.3035 / 0.2955 | 0.3067 / 0.2943 |
| **Epoch 2 Dev $F_{0.2}$ (Cat 1 / Cat 2)** | **0.866940** (0.8846 / 0.8493) *(Best)* | 0.867300 (0.8908 / 0.8438) |
| **Epoch 3 Train Loss / Dev Loss** | 0.2477 / 0.3039 | 0.2511 / 0.3056 |
| **Epoch 3 Dev $F_{0.2}$ (Cat 1 / Cat 2)** | 0.861372 (0.8856 / 0.8371) | **0.867637** (0.8888 / 0.8465) *(Best)* |
| **Best Model Selected** | **Epoch 2** ($F_{0.2} = \mathbf{0.866940}$) | **Epoch 3** ($F_{0.2} = \mathbf{0.867637}$) |
| **Total Training Duration** | 1,661.7s (27.7 min) | 2,267.7s (37.8 min) |
| **Peak GPU VRAM** | 2,116.6 MB (~2.1 GB) | 2,116.6 MB (~2.1 GB) |
| **Drive C: Free Space After** | **7.50 GB** | **7.08 GB** (Net consumption: 418 MB) |

> [!NOTE]
> Checkpoint selection strictly followed the pre-declared rule: the model with the highest internal development $F_{0.2}$ was saved. For the baseline, Epoch 2 was selected (Epoch 3 overfit slightly). For the challenger, Epoch 3 was selected.

---

## 6. Step 4 — Deep Evaluation on Internal 800-Title Dev Set

We evaluated both models head-to-head on the internal 800-title dev set using [`src/evaluation/phase_32c_evaluate_internal.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/evaluation/phase_32c_evaluate_internal.py):

### A. Global Performance Comparison

| Metric | Matched Baseline (`1.00x`) | Targeted Challenger (`1.15x`) | Delta ($\Delta$) | Gate Outcome |
| :--- | :---: | :---: | :---: | :---: |
| **Internal Dev $F_{0.2}$ (Official Macro)**| **0.866940** | **0.867637** | **+0.000697** | **PASSED** |
| - Category 1 $F_{0.2}$ (Automotive) | 0.884556 | **0.888781** | **+0.004225** | Strong gain |
| - Category 2 $F_{0.2}$ (Motorcycle) | 0.849324 | 0.846493 | -0.002832 | Slight trade-off |
| **Micro Precision** | 0.8708 | **0.8725** | **+0.0017** | Positive |
| **Micro Recall** | 0.8616 | **0.8638** | **+0.0023** | Positive |
| **Total True Positives (TP)** | 5,296 | **5,310** | **+14** | More correct entities |
| **Total False Positives (FP)** | 786 | **776** | **-10** | Fewer global errors |
| **Total False Negatives (FN)** | 851 | **837** | **-14** | Fewer missed entities |

### B. Vehicle-Model Specific Performance

| Vehicle-Model Metric | Matched Baseline (`1.00x`) | Targeted Challenger (`1.15x`) | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Vehicle-Model $F_{0.2}$** | 0.706016 | **0.706476** | **+0.000460** |
| **Vehicle-Model Precision** | 0.7084 | 0.7077 | -0.0007 |
| **Vehicle-Model Recall** | 0.6514 | **0.6766** | **+0.0253** |
| **Vehicle-Model TP** | 979 | **1,017** | **+38** |
| **Vehicle-Model FP** | 403 | 420 | +17 |
| **Vehicle-Model FN** | 524 | **486** | **-38** |
| **Single-Token Exact Match Acc** | 65.28% | **69.10%** | **+3.82%** |
| **Multi-Token Exact Match Acc** | 64.93% | **65.58%** | **+0.65%** |

### C. Boundary Error Taxonomy & Head-to-Head Dynamics

```
Head-to-Head Comparison (800 Titles):
  - Titles Improved:   136 titles
  - Titles Regressed:  123 titles
  - Titles Unchanged:  541 titles
  - Net Title Gain:    +13 titles

Vehicle-Model Span Transitions:
  - Spans Recovered (Wrong -> Correct): 104 spans
  - Spans Regressed (Correct -> Wrong):  62 spans
  - Net Model Span Gain:                +42 spans
```

- **Multi-Ground-Truth Overlaps**: 144 in Baseline $\rightarrow$ **143 in Challenger** ($-1$).
- **Single-Span Boundary Errors**: 236 in Baseline $\rightarrow$ **243 in Challenger** ($+7$).
  - Sub-span truncations: 214 $\rightarrow$ 222 ($+8$).
  - Super-span expansions: 18 $\rightarrow$ 19 ($+1$).
- **Mechanism**: The 1.15x weight on `I-Modell` provided sufficient incentive to recover 104 previously missed spans (+38 net TP), with single-token accuracy rising by $+3.82\%$ and multi-token accuracy rising by $+0.65\%$.

---

## 7. Step 5 — Exploratory Holdout & Ensemble Assessment (1,000 Titles)

Because the challenger successfully passed the internal decision gate, we conducted a single exploratory evaluation on the 1,000-title Split-A validation holdout (both standalone and ensembled with the three frozen XLM-R models at MIN 0.60):

| Model Configuration | Training Corpus | Holdout $F_{0.2}$ | Cat 1 $F_{0.2}$ | Cat 2 $F_{0.2}$ | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standalone Baseline** | 3,200 Titles | 0.851460 | 0.860035 | 0.842884 | 0.8544 | 0.8528 |
| **Standalone Challenger** | 3,200 Titles | **0.851865** | 0.858548 | **0.845181** | **0.8568** | **0.8555** |
| *Delta (Standalone Challenger vs Baseline)* | — | **+0.000405** | *-0.001487* | **+0.002297** | **+0.0024** | **+0.0027** |
| **Matched Baseline Ensemble (@ MIN 0.60)** | 3.2k + 3x XLM-R | **0.897179** | 0.893547 | 0.900812 | **0.9145** | 0.8308 |
| **Challenger Ensemble (@ MIN 0.60)** | 3.2k + 3x XLM-R | 0.896544 | 0.891590 | **0.901498** | 0.9129 | **0.8337** |
| *Delta (Challenger Ensemble vs Baseline Ens)* | — | **-0.000635** | *-0.001957* | **+0.000686** | *-0.0016* | **+0.0029** |
| **Active Frozen Champion** | **4,000 Titles + 3x XLM-R** | **0.899802** | **0.899921** | **0.899683** | **0.9172** | **0.8350** |

```mermaid
graph TD
    subgraph Standalone Models on Holdout
        B_Stand["Standalone Baseline (3.2k)<br/>F0.2 = 0.851460"]
        C_Stand["Standalone Challenger (3.2k)<br/>F0.2 = 0.851865 (+0.000405)"]
        B_Stand -->|1.15x I-Weight| C_Stand
    end

    subgraph 4-Model Ensembles at MIN 0.60
        B_Ens["Baseline Ensemble (3.2k GBERT + 3x XLMR)<br/>F0.2 = 0.897179"]
        C_Ens["Challenger Ensemble (3.2k GBERT + 3x XLMR)<br/>F0.2 = 0.896544 (-0.000635)"]
        Champ["Frozen Champion (4.0k GBERT + 3x XLMR)<br/>F0.2 = 0.899802 (Quiz = 0.901245)"]
        B_Ens -.->|Ensemble Precision Penalty| C_Ens
        B_Ens -->|Data Volume Deficit (-800 titles)| Champ
    end
```

### Analysis of the Ensemble Divergence:
1. **Standalone Confirmation**: In standalone evaluation, the Challenger strictly beat the Baseline on both precision ($+0.0024$) and recall ($+0.0027$), confirming that 1.15x weighting genuinely improved single-model feature extraction.
2. **The Soft-Voting Ensemble Effect**: In the ensemble, model probabilities are averaged:
   $$P_{\text{ensemble}} = 0.40 \cdot P_{\text{GBERT}} + 0.20 \cdot P_{\text{XLM-R, 42}} + 0.20 \cdot P_{\text{XLM-R, 7}} + 0.20 \cdot P_{\text{XLM-R, 21}}$$
   Because the Challenger produces slightly higher probabilities on `I-Modell`, more marginal candidate entities clear the `MIN 0.60` filter (Recall increased from 0.8308 to 0.8337). However, a small fraction of these marginal spans were boundary errors, slightly lowering precision from 0.9145 to 0.9129 (-0.0016).
3. **The Asymmetric Penalty**: Under $F_{0.2}$:
   $$\beta^2 = 0.04 \implies \text{Precision is weighted } 25\times \text{ heavier than Recall}$$
   A drop of 0.0016 in precision outweighs a gain of 0.0029 in recall by approximately $1.38\times$, resulting in the slight $-0.000635$ dip in the ensemble score.
4. **Data Volume Gap**: Most importantly, withholding 800 titles from GBERT costs $\sim 0.0026$ in ensemble score ($0.897179$ vs $0.899802$). No loss-weighting adjustment can overcome the loss of 20% of the labeled training titles.

---

## 8. Scientific Limitations & Risk Assessment

1. **Partition Size Ceiling**: Training on 3,200 titles leaves the model vulnerable to lexical gaps. As observed in Phase 29B (CV3) and Phase 32C, training on data subsets always results in a net deficit relative to the full 4,000-title training partition.
2. **Risk of Retraining on 4,000 Titles**: Retraining GBERT on all 4,000 titles with 1.15x weighting would risk altering the exact probability calibration that makes the Exp 10 ensemble score 0.901245 on the official Quiz leaderboard (Rank 13). With approximately four days before the final Test release and no remaining Quiz evaluations allowed, taking this calibration risk is statistically unjustified.
3. **Annotation Ambiguity Bottleneck**: As proven in Phase 32A/32B, 32.8% of contiguous vehicle-model token pairs have mutually contradictory labels in the training set. Loss reweighting shifts the boundary decision threshold, but cannot eliminate ground-truth ambiguity.

---

## 9. Final Decision & Strategic Plan

```
=============================================================================
FINAL DECISION: NO MEANINGFUL IMPROVEMENT — RETAIN CHAMPION
=============================================================================
```

### Strategic Action Plan:
1. **Lock Production Champion**: Retain the official Exp 10 + MIN 0.60 champion ([`src/submission/predict_test_champion.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/submission/predict_test_champion.py)) as the sole, locked submission engine.
2. **Preserve Checkpoints**: The four original champion checkpoints remain untouched and verified:
   - `checkpoints/gbert_baseline_best` (Weight 0.40)
   - `checkpoints/xlmr_baseline_best` (Weight 0.20)
   - `checkpoints/xlmr_seed_7_best` (Weight 0.20)
   - `checkpoints/xlmr_seed_21_best` (Weight 0.20)
3. **Official Quiz Benchmark**: `submissions/submission_quiz_exp26b_min060.tsv` (SHA256: `049CF5EF60D88D86C26DEC5A815F8DB820A306989E383843E6A9B3108AD7A7FA`, EvalAI Quiz $F_{0.2} = 0.90124541$).
4. **Transition to Standby**: Conclude all model training and research. Await official release of Test phase data on November 1.

---

## 10. Audit Metadata

- **Git Commit Hash**: Staged and committed with Phase 32C experiment code and evaluation reports.
- **Experimental Artifacts**:
  - Training implementation: [`src/modeling/train_phase32c_matched.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/modeling/train_phase32c_matched.py)
  - Split generator: [`src/data/make_internal_split.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/data/make_internal_split.py)
  - Unit test suite: [`tests/test_phase32c_loss.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/tests/test_phase32c_loss.py) (5/5 passed)
  - Dev evaluation script: [`src/evaluation/phase_32c_evaluate_internal.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/evaluation/phase_32c_evaluate_internal.py)
  - Exploratory holdout script: [`src/evaluation/phase_32c_exploratory_holdout.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/evaluation/phase_32c_exploratory_holdout.py)
  - Training metrics cache: `data/cache/phase_32c_training_history.json`
  - Internal evaluation cache: `data/cache/phase_32c_internal_eval_results.json`
  - Holdout evaluation cache: `data/cache/phase_32c_exploratory_holdout_results.json`
