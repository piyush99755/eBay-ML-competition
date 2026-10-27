# Phase 25B: German ELECTRA Challenger Evaluation and Negative Result Analysis

**Date**: October 2026  
**Status**: **REJECT CANDIDATE** (Frozen Exp 10 Champion Retained)  
**Evaluated Architecture**: `deepset/gelectra-base` (German ELECTRA Discriminator, Seed 42)  
**Evaluation Set**: Local Split-A Validation ($N = 1,000$ titles, 11,349 tokens, Seed 42)  

---

## 1. Executive Summary & Verdict

* **Verdict**: **REJECT CANDIDATE**.
* **Reasoning**:
  * Standalone GELECTRA achieved an entity $F_{0.2} = 0.844756$, trailing standalone GBERT ($0.867200$) by $-0.022444$ and standalone XLM-R ($0.869284$) by $-0.024528$.
  * All four ensemble variants incorporating GELECTRA (Variants A, B, C, D) strictly underperformed the frozen Exp 10 champion benchmark ($F_{0.2} = 0.879852$).
  * The best variant (Variant C: $0.35$ GBERT + $0.15$ GELECTRA + $0.50$ XLM-R) scored $0.874976$ ($\Delta = -0.004876$).
  * Directly replacing GBERT with GELECTRA (Variant D) caused a drop of $-0.009841$ ($0.870011$).
  * Under the official competition $F_{0.2}$ metric, precision is weighted 25× higher than recall ($\beta^2 = 0.04$). Adding GELECTRA increased false-positive entity emissions ($+32$ to $+78$ entities), diluting precision from $0.8846$ down to $0.8780 - 0.8818$ and lowering the composite score.
* **Champion Status**: The frozen Experiment 10 ensemble (0.40 GBERT + 0.60 3-seed XLM-R) remains our active, verified champion for the final competition Test phase.

---

## 2. Pre-Training & Tokenizer Compatibility Verification

Before fine-tuning, compatibility was verified using `scripts/verify_gelectra_pretraining.py` against `deepset/gbert-base` and the champion label mapping:

| Check | Expected | Observed | Status |
| :--- | :--- | :--- | :---: |
| **Model Type** | `electra` | `electra` | **PASS** |
| **Hidden Size / Layers / Heads** | 768 / 12 / 12 | 768 / 12 / 12 | **PASS** |
| **Tokenizer Vocab Size** | 31,102 (WordPiece) | 31,102 (WordPiece) | **PASS** |
| **Subword Parity on Sample (N=100)**| 100% Identical | 2,913 / 2,913 subwords (100% match) | **PASS** |
| **Word IDs Alignment Mismatches** | 0 | 0 | **PASS** |
| **Active Label Mismatches** | 0 | 0 | **PASS** |
| **BIO Label Space** | 48 classes | 48 classes (identical IDs to champion) | **PASS** |
| **Category-Aware Post-Filtering** | Cat 1: 4 / Cat 2: 3 | Verified via `filter_entities_by_category` | **PASS** |

---

## 3. Training History by Epoch

Fine-tuning was executed via `src/modeling/train_gelectra.py` using Split A (4,000 train / 1,000 val), seed 42, AdamW ($lr = 2 \times 10^{-5}$), batch size 4:

| Epoch | Train Loss | Val Loss | Official $F_{0.2}$ | Cat 1 $F_{0.2}$ | Cat 2 $F_{0.2}$ | Micro Precision | Micro Recall | Training Time | Peak VRAM |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 1.4881 | 0.5518 | 0.802191 | 0.824612 | 0.779770 | 0.8247 | 0.8180 | 963.2s | 2.07 GB |
| **2** | 0.4889 | 0.4276 | 0.841426 | 0.845215 | 0.837638 | 0.8556 | 0.8391 | 1434.5s | 2.07 GB |
| **3** | **0.3853** | **0.4041** | **0.844756** | **0.851980** | **0.837532** | **0.8574** | **0.8510** | 1226.6s | 2.07 GB |

*Best checkpoint saved to*: `checkpoints/gelectra_baseline_best` (Epoch 3).

---

## 4. Standalone and Ensemble Validation Comparison

Validation was executed on the exact 1,000 Split-A holdout titles via `src/evaluation/experiment_25_gelectra_ensemble.py`:

| Model / Variant Configuration | Local $F_{0.2}$ | Delta vs Champion | Cat 1 $F_{0.2}$ | Cat 2 $F_{0.2}$ | Precision | Recall | Total Entities | Entity Diff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Exp 10 Champion)** | **0.879852** | **+0.000000** | **0.878888** | **0.880815** | **0.8846** | **0.8669** | **7,441** | **0** |
| **Standalone GELECTRA** | 0.844756 | -0.035096 | 0.851980 | 0.837532 | 0.8591 | 0.8510 | 7,522 | +81 |
| **Standalone GBERT** | 0.867200 | -0.012652 | 0.874798 | 0.859602 | 0.8713 | 0.8570 | 7,468 | +27 |
| **Standalone XLM-R (Seed 42)** | 0.869284 | -0.010568 | 0.869246 | 0.869322 | 0.8756 | 0.8568 | 7,430 | -11 |
| **Variant A** (0.25 GB + 0.20 ELE + 0.55 XLMR) | 0.873589 | -0.006263 | 0.873885 | 0.873293 | 0.8806 | 0.8692 | 7,495 | +54 |
| **Variant B** (0.25 GB + 0.25 ELE + 0.50 XLMR) | 0.871497 | -0.008355 | 0.871863 | 0.871132 | 0.8786 | 0.8700 | 7,519 | +78 |
| **Variant C** (0.35 GB + 0.15 ELE + 0.50 XLMR) | 0.874976 | -0.004876 | 0.877763 | 0.872188 | 0.8818 | 0.8679 | 7,473 | +32 |
| **Variant D** (0.40 ELE + 0.60 XLMR) | 0.870011 | -0.009841 | 0.870513 | 0.869509 | 0.8780 | 0.8688 | 7,514 | +73 |

---

## 5. Technical Dynamics & Metric Sensitivity Analysis

### Why Did GELECTRA Underperform GBERT and Hurt the Ensemble?
1. **Lower Base Discriminator Calibration**:
   Although ELECTRA's replaced-token detection objective is effective for classification benchmarks (e.g. GLUE), in fine-grained 48-class token classification on specialized German e-commerce text, `gelectra-base` produced a higher rate of false-positive entity boundaries ($+81$ predicted entities on 1,000 titles).
2. **Extreme Asymmetry of the $F_{0.2}$ Competition Metric**:
   The competition metric uses $\beta = 0.2$, which weights precision 25× more heavily than recall:
   $$F_{0.2} = \frac{(1 + 0.04) \cdot P \cdot R}{0.04 \cdot P + R} \approx \frac{1.04 \cdot P \cdot R}{R} \approx P$$
   In Variant A, adding GELECTRA slightly increased recall ($0.8669 \to 0.8692$, $+0.0023$), but decreased precision ($0.8846 \to 0.8806$, $-0.0040$). Because precision loss is penalized 25× higher than recall gains, the net effect was an overall drop of $-0.006263$.
3. **GBERT Superiority in German Auto Domain**:
   Standard Masked Language Modeling in `deepset/gbert-base` learned more conservative, high-precision contextual representations for complex technical compounds and part numbers than `gelectra-base`.

---

## 6. Resource Consumption and System Health

* **Peak GPU VRAM**: **2.07 GB** (on 6.00 GB GTX 1660 Ti, 65% headroom).
* **Storage Consumption**:
  * HF Cache: ~441 MB
  * Checkpoint (`checkpoints/gelectra_baseline_best`): 438 MB
  * Final free space on Drive C: **9.66 GB free** (comfortably above the 5.0 GB limit).
* **Errors or Hardware Faults**: Zero errors or warnings encountered.

---

## 7. Recommendation & Conclusion

* **Formal Recommendation**: **REJECT CANDIDATE**.
* Do **NOT** incorporate `deepset/gelectra-base` into the champion ensemble.
* Do **NOT** upload any submission derived from GELECTRA to EvalAI.
* Retain the frozen Experiment 10 champion:
  $$\mathbf{P}_{\text{final}} = 0.40 \cdot \mathbf{P}_{\text{GBERT}} + 0.20 \cdot \mathbf{P}_{\text{XLMR, 42}} + 0.20 \cdot \mathbf{P}_{\text{XLMR, 7}} + 0.20 \cdot \mathbf{P}_{\text{XLMR, 21}}$$
* The production test pipeline in `src/submission/predict_test_champion.py` remains 100% frozen, validated, and ready for official Test data.
