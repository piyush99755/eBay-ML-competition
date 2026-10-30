# Phase 39B Research Report: Precision-Guided Vehicle-Model Recovery

## Executive Summary

Phase 39B conducted a focused, cache-only research investigation to evaluate whether vehicle-model (`Kompatibles_Fahrzeug_Modell`) predictions suppressed by the Phase 35C/36C confidence threshold of $0.80$ (i.e. lying in $[0.60, 0.80)$) could be selectively recovered without degrading precision under the official competition metric ($F_{0.2}$).

Two predeclared challenger configurations were implemented and tested on the verified 489-title clean holdout:
1. **Candidate A (Conservative Lexicon Recovery):** Admits suppressed predictions whose normalized value strictly matches an ambiguity-filtered training-only lexicon ($3,662$ entries derived exclusively from `train_4k`).
2. **Candidate B (Lexicon Plus Vehicle-Make Context):** Extends Candidate A by also admitting suppressed candidates occurring within $\pm 3$ tokens of a predicted vehicle make matching training co-occurrence patterns.

### Key Results
- **Official-Style Hierarchical $F_{0.2}$ Score:**
  - **Baseline (Phase 36C):** $0.924945$ (Category 1: $0.915723$, Category 2: $0.934168$)
  - **Candidate A:** $0.925095$ ($\Delta = +0.000149$, Category 1: $0.913934$ [$\Delta = -0.001789$], Category 2: $0.936255$ [$\Delta = +0.002088$])
  - **Candidate B:** $0.924039$ ($\Delta = -0.000906$, Category 1: $0.913550$ [$\Delta = -0.002173$], Category 2: $0.934528$ [$\Delta = +0.000360$])
- **Paired Bootstrap Resampling (1,000 resamples, seed 42):**
  - **Candidate A:** Mean $\Delta = +0.000226$, $95\%$ CI = $[-0.003149, +0.003320]$, $p(\Delta \le 0) = 0.4340$.
  - **Candidate B:** Mean $\Delta = -0.000800$, $95\%$ CI = $[-0.005108, +0.002976]$, $p(\Delta \le 0) = 0.6400$.
- **Final Verdict:** **`RETAIN PHASE 36C CHAMPION`**. Neither candidate achieves the required $+0.0020$ official delta, both degrade Category 1 performance, and the bootstrap confidence intervals substantially overlap zero.

---

## 1. Baseline & Research Integrity Verification

Before running experiments, all critical assets were verified:
- **Active Champion TSV:** `submissions/submission_quiz_phase36c_produktart080.tsv`
  - SHA-256: `9F76C2418C60708BC92486F5F39165021000EE9151DCCC7DD8F75CE8BBA9CC87` (**Verified Bit-for-Bit Exact Match**)
  - Official Quiz Score: $F_{0.2} = 0.9206543463689059$
- **Checkpoints Verified (4/4 intact):**
  - `checkpoints/phase33b_dapt_gbert_best`
  - `checkpoints/xlmr_baseline_best`
  - `checkpoints/xlmr_seed_7_best`
  - `checkpoints/xlmr_seed_21_best`
- **Data & Cache Integrity:**
  - Clean holdout manifest: `data/cache/phase_35b_clean_holdout_manifest.json` (489 record IDs strictly unseen in supervised training by all four checkpoints).
  - Out-of-fold probability cache: `data/cache/phase_35a_val_1k_probs.pt` (verified intact).
- **Disk Free Space:** $8.00\text{ GiB}$ ($8.58\text{ GB}$), well above the $5.0\text{ GiB}$ safety threshold.
- **Ensemble Weights:** DAPT GBERT seed 42 ($0.40$), XLM-R seed 42 ($0.20$), XLM-R seed 7 ($0.20$), XLM-R seed 21 ($0.20$).
- **Aspect Thresholds:** `Kompatibles_Fahrzeug_Modell` ($0.80$), `Produktart` ($0.80$), all other 27 aspects ($0.60$).

---

## 2. Suppressed Candidate Audit

Using cached fused probabilities, predictions were decoded with `Kompatibles_Fahrzeug_Modell` threshold set to $0.60$ (while retaining $0.80$ for `Produktart` and $0.60$ for all other aspects). Comparing against the Phase 36C baseline revealed exactly the suppressed candidate population in $[0.60, 0.80)$:

- **Total Suppressed Candidates in $[0.60, 0.80)$:** $204$
- **Reconciliation with Prior Estimates:**
  - True Positives (reconciled via multiset matching): **$148$**
  - False Positives (reconciled via multiset matching): **$56$**
  - Overall Precision across suppressed band: **$72.55\%$**
  - Category 1 Candidates: $136$ ($41$ TP, $17$ FP in lexicon; remainder outside lexicon)
  - Category 2 Candidates: $68$
  - Overlap with baseline predictions: **$0$** (all candidates strictly fill empty spans or non-conflicting slots).
  - Ambiguous short/numeric candidates ($\le 2$ chars or digits): $20$ ($14$ TP, $6$ FP).

*Note on reconciliation:* Phase 39A diagnostics roughly estimated $149$ TPs and $55$ FPs. Exact multiset duplicate matching and strict title-level boundary alignment reconcile the exact count as $148$ TPs and $56$ FPs ($72.55\%$ precision).

---

## 3. Training-Only Vehicle-Model Lexicon

To prevent any data snooping or holdout contamination, the vehicle model lexicon was constructed strictly from `train_4k` (the canonical 4,000-title training partition, records 1 to 4,000). Zero validation titles, holdout records, Quiz, or Test data were used.

### Raw Statistics (from `train_4k`):
- Unique vehicle model strings: $4,046$
- Total vehicle model entity occurrences: $7,480$
- Unique vehicle makes: $178$
- Co-occurring (Make, Model) pairs: $6,182$

### Deterministic Ambiguity Filtering:
A candidate vehicle model was included in `candidate_a_lexicon` if and only if:
1. It is not an automotive connector or stopword: `{'auch', 'fuer', 'für', 'set', 'kit', 'typ', 'bj', 'ab', 'bis', 'alle', 'und', 'mit'}`.
2. It is not a single digit (`0-9`) or single letter (`a-z`, `A-Z`).
3. It does not match pure engine displacement / decimal format (`^\d+\.\d+$`).
4. It satisfies the reliability constraint: `train_count >= 2` OR `string_length >= 4`.

### Resulting Lexicon:
- Filtered Lexicon Size: **$3,662$ entries** ($90.5\%$ of raw training models).
- Eliminated: $384$ noisy, ambiguous, or single-occurrence short fragments.

---

## 4. Predeclared Configurations & Invariant Enforcement

### Predeclared Configurations:
1. **Baseline (Phase 36C):**
   - Vehicle-Model threshold: $0.80$
   - Produktart threshold: $0.80$
   - All other aspects: $0.60$
2. **Candidate A (Conservative Lexicon Recovery):**
   - Base entities identical to Phase 36C.
   - For any suppressed vehicle-model candidate in $[0.60, 0.80)$: admit if and only if normalized value $\in \text{candidate\_a\_lexicon}$.
   - Result: Admitted **$95$ candidates** ($76$ TP, $19$ FP, precision = $80.00\%$).
3. **Candidate B (Lexicon Plus Vehicle-Make Context):**
   - Base entities identical to Phase 36C.
   - Admits Candidate A entities, PLUS candidates that have a predicted vehicle make within $\pm 3$ tokens matching training co-occurrence or model vocabulary (and not matching stopwords or pure decimals).
   - Result: Admitted **$104$ candidates** ($81$ TP, $23$ FP, precision = $77.88\%$).

### Non-Destructive Addition Assertion:
For every title $i \in \{1, \dots, 489\}$, the following invariant was asserted:
$$\text{baseline\_predictions}_i \subseteq_{\text{multiset}} \text{candidate\_predictions}_i$$
**Result: 100% Passed.** No baseline entities, boundaries, or other aspect predictions were altered or suppressed.

---

## 5. Performance Comparison Tables

### Table 1: Official Competition Hierarchical Metrics

| Configuration | Official $F_{0.2}$ | Official $\Delta$ | Category 1 Score | Category 1 $\Delta$ | Category 2 Score | Category 2 $\Delta$ |
|---|---:|---:|---:|---:|---:|---:|
| **Baseline (Phase 36C)** | **0.924945** | — | **0.915723** | — | **0.934168** | — |
| **Candidate A** | 0.925095 | **+0.000149** | 0.913934 | **-0.001789** | 0.936255 | +0.002088 |
| **Candidate B** | 0.924039 | **-0.000906** | 0.913550 | **-0.002173** | 0.934528 | +0.000360 |

### Table 2: Global Micro Metrics (Secondary Diagnostic)

| Configuration | Micro $F_{0.2}$ | Micro $\Delta$ | Micro Precision | Micro Recall | TP | FP | FN | $\Delta$ TP | $\Delta$ FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Baseline (Phase 36C)** | **0.938333** | — | **0.944479** | 0.807050 | 3,045 | 179 | 728 | — | — |
| **Candidate A** | 0.935422 | -0.002911 | 0.940343 | 0.827193 | 3,121 | 198 | 652 | +76 | +19 |
| **Candidate B** | 0.934497 | -0.003836 | 0.939303 | 0.828518 | 3,126 | 202 | 647 | +81 | +23 |

### Table 3: Vehicle-Model Aspect-Level Metrics

| Configuration | Cat 1 VM Precision | Cat 1 VM Recall | Cat 1 VM $F_{0.2}$ | Cat 2 VM Precision | Cat 2 VM Recall | Cat 2 VM $F_{0.2}$ |
|---|---:|---:|---:|---:|---:|---:|
| **Baseline (Phase 36C)** | **0.8423** | 0.4983 | **0.820068** | **0.9067** | 0.5368 | 0.883324 |
| **Candidate A** | 0.8329 | 0.5694 | 0.812500 ($-0.007568$) | 0.9142 | 0.6053 | 0.891283 ($+0.007959$) |
| **Candidate B** | 0.8309 | 0.5729 | 0.810875 ($-0.009193$) | 0.9070 | 0.6105 | 0.884696 ($+0.001372$) |

---

## 6. Category-Specific Impact & The $F_{0.2}$ Precision Dilution Trap

The results reveal the mathematical mechanics of the $F_{0.2}$ metric in entity extraction:

1. **The 25:1 Asymmetry:**
   $$\beta^2 = 0.2^2 = 0.04 = \frac{1}{25}$$
   Under $F_{0.2}$, one False Positive incurs **25 times** the penalty of one False Negative.
2. **Category 1 Degradation:**
   In Category 1, vehicle model predictions in $[0.60, 0.80)$ that matched the training lexicon had a precision of only $70.7\%$ ($41$ TP, $17$ FP). Adding predictions at $70.7\%$ precision into an aspect whose baseline precision was $84.23\%$ mechanically diluted precision, causing Category 1 VM $F_{0.2}$ to fall from $0.820068$ to $0.812500$ ($-0.007568$). Because Category 1 has high ground-truth model frequency ($288$ gold models), this degraded overall Category 1 score by $-0.001789$.
3. **Category 2 Asymmetry:**
   In Category 2, vehicle models are cleaner; lexicon matching recovered $35$ TP with only $2$ FP ($94.6\%$ precision), improving Category 2 VM $F_{0.2}$ from $0.883324$ to $0.891283$ ($+0.007959$).
4. **Net Cancellation:**
   Because the official competition score unweighted macro-averages Category 1 and Category 2:
   $$\text{Official Score} = \frac{\text{Score}_{\text{Cat 1}} + \text{Score}_{\text{Cat 2}}}{2}$$
   The $-0.001789$ loss in Category 1 almost entirely cancelled out the $+0.002088$ gain in Category 2, leaving an insignificant net gain of only $+0.000149$ for Candidate A, and causing a net loss of $-0.000906$ for Candidate B.

---

## 7. Paired Title-Level Bootstrap Resampling

To establish statistical significance, $1,000$ paired title-level bootstrap resamples were executed with seed 42. In each resample, gold-frequency aspect weights were recomputed dynamically per category:

| Challenger | Mean Bootstrap $\Delta$ | Bootstrap Std | 95% Confidence Interval | $p(\Delta \le 0)$ | Statistically Significant? |
|---|---:|---:|:---:|---:|:---:|
| **Candidate A** | $+0.000226$ | $0.001631$ | $[-0.003149, +0.003320]$ | $0.4340$ | **NO** |
| **Candidate B** | $-0.000800$ | $0.002082$ | $[-0.005108, +0.002976]$ | $0.6400$ | **NO** |

The 95% bootstrap confidence interval for Candidate A spans from $-0.0031$ to $+0.0033$, with $43.4\%$ of bootstrap iterations yielding zero or negative delta. This confirms that the nominal $+0.000149$ delta is indistinguishable from random noise.

---

## 8. Leakage, Generalization & Historical Context

1. **Zero Data Leakage:**
   The lexicon was strictly constructed using `train_4k`. No holdout records, Quiz predictions, or Test inputs contributed to lexicon formation or threshold boundaries.
2. **Validation of Phase 35C Threshold Decision:**
   In Phase 35C, elevating the vehicle model threshold from $0.60$ to $0.80$ boosted Quiz $F_{0.2}$ from $0.903532$ to $0.915983$ ($+0.012451$). Phase 39B proves that this elevation was not an aggressive over-filtering; rather, predictions in $[0.60, 0.80)$ inherently contain $27.45\%$ false positives, which is too noisy for an $F_{0.2}$ objective. Attempting to selectively re-admit them merely reintroduces the precision penalties that Phase 35C successfully eliminated.
3. **Consistency with Phase 36A & Phase 38A:**
   Phase 36A showed that boundary heuristics regress $F_{0.2}$, and Phase 38A demonstrated that model agreement does not reliably salvage low-confidence predictions. Phase 39B confirms that training-lexicon membership alone cannot distinguish true automotive models from noisy false positives in Category 1.

---

## 9. Decision & Recommended Next Actions

### Predeclared Decision Criteria:
- **PROMISING LOCAL CHALLENGER:**
  - Official-style delta $\ge +0.0020$ (**FAILED: $+0.000149$**)
  - Bootstrap 95% CI lower bound $> 0$ (**FAILED: $-0.003149$**)
  - No significant degradation in either category (**FAILED: Category 1 $-0.001789$**)
- **RETAIN PHASE 36C CHAMPION:**
  - Met when both candidates fail to clear the improvement threshold or inflict unacceptable precision damage.

### Final Decision:
**`RETAIN PHASE 36C CHAMPION`**

### Next Strategic Steps:
1. **Preserve Frozen Champion:** Keep Phase 36C (`submissions/submission_quiz_phase36c_produktart080.tsv`, official Quiz $F_{0.2} = 0.9206543463689059$) as our protected competition leader.
2. **Do Not Lower Vehicle Model Thresholds:** Discontinue attempts to recover vehicle models in $[0.60, 0.80)$ via string matching.
3. **Explore High-Precision Opportunities:** As identified in Phase 39A, explore aspects where false positives can be *suppressed* (such as precision filtering on `Hersteller` or category-specific calibration) rather than recalling noisy entities.
