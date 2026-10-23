# Phase 21A: Baseline Quiz Submission Report

**Date:** 2026-10-23  
**Status:** Completed & 100% Compliant with Annexure.pdf  
**Pipeline Script:** [`src/submission/predict_quiz_baseline.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/src/submission/predict_quiz_baseline.py)  
**Output Submission File:** [`submissions/submission_quiz_exp10_baseline.tsv`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/submissions/submission_quiz_exp10_baseline.tsv)  
**Validator Script:** [`scripts/validate_submission.py`](file:///c:/Users/piyus/Downloads/eBay-ML-competition/scripts/validate_submission.py)  

---

## 1. Executive Summary

Phase 21A executed unlabelled candidate inference on the full eBay Motors Quiz set (records 5,001 through 30,000, 25,000 listing titles) using our verified **Experiment 10 Champion Ensemble**:
- **0.40 GBERT** (`checkpoints/gbert_baseline_best`)
- **0.20 XLM-R seed 42** (`checkpoints/xlmr_baseline_best`)
- **0.20 XLM-R seed 7** (`checkpoints/xlmr_seed_7_best`)
- **0.20 XLM-R seed 21** (`checkpoints/xlmr_seed_21_best`)

Inference ran sequentially on a single NVIDIA GeForce GTX 1660 Ti (6.00 GB VRAM), loading one model at a time, caching flat token probability arrays to disk (`data/cache/*.npy`), and clearing VRAM between model runs. Peak VRAM utilization remained under 1.3 GB.

The output submission TSV passed all schema, encoding, and Annexure.pdf validation checks with zero errors.

---

## 2. Quantitative Summary

| Metric | Value |
|---|---|
| **Quiz Titles Processed** | 25,000 |
| **Record Number Range** | 5,001 to 30,000 |
| **Total Data Rows Generated** | 186,461 |
| **Average Entities per Title** | 7.46 |
| **Titles with >= 1 Entity** | 25,000 (100.0%) |
| **Zero-Entity Titles** | 0 (0.0%) |
| **Output File Size** | 6.70 MB |
| **Output Encoding** | UTF-8 (Strict literal TSV, `QUOTE_NONE`) |
| **Header Format** | `Record Number\tCategory\tAspect Name\tAspect Value` |
| **Category 1 Predictions** | 101,846 rows (54.6%) |
| **Category 2 Predictions** | 84,615 rows (45.4%) |
| **Distinct Aspect Types** | 23 of 29 official competition aspects |
| **Total Pipeline Runtime** | 3,484.3s (58.07 minutes) |
| **Validation Result** | **PASSED (100% Compliant)** |

---

## 3. Aspect Distribution (Quiz Set)

| Aspect Name | Prediction Count | Share (%) |
|---|---|---|
| `Kompatibles_Fahrzeug_Modell` | 39,964 | 21.43% |
| `Kompatible_Fahrzeug_Marke` | 28,672 | 15.38% |
| `Im_Lieferumfang_Enthalten` | 28,345 | 15.20% |
| `Produktart` | 22,970 | 12.32% |
| `Hersteller` | 22,182 | 11.90% |
| `Herstellernummer` | 15,955 | 8.56% |
| `Einbauposition` | 11,878 | 6.37% |
| `Bremsscheiben-Aussendurchmesser` | 4,322 | 2.32% |
| `Anzahl_Der_Einheiten` | 3,904 | 2.09% |
| `Kompatibles_Fahrzeug_Jahr` | 2,943 | 1.58% |
| `Modell` | 1,857 | 1.00% |
| `Bremsscheibenart` | 1,735 | 0.93% |
| `Maßeinheit` | 649 | 0.35% |
| `Material` | 283 | 0.15% |
| `Oe/Oem_Referenznummer(N)` | 267 | 0.14% |
| `Besonderheiten` | 182 | 0.10% |
| `Produktlinie` | 181 | 0.10% |
| `Farbe` | 44 | 0.02% |
| `Oberflächenbeschaffenheit` | 36 | 0.02% |
| `Anwendung` | 28 | 0.02% |
| `Stärke` | 27 | 0.01% |
| `Größe` | 26 | 0.01% |
| `Menge` | 11 | 0.01% |
| **Total** | **186,461** | **100.0%** |

---

## 4. Key Engineering Implementations

### 4.1 Subword-to-Original-Token Alignment
- German eBay Motors titles contain noisy unencodable characters (e.g. lone `\ufffd` replacement characters). Exactly 91 out of 25,000 Quiz titles contained such a token, which HuggingFace tokenizers stripped to 0 subwords.
- `QuizCandidateDataset` maps original whitespace tokens to the first subword index.
- If a token index is dropped by the tokenizer, its probability vector defaults to 100% `'O'` (`label_id = 47`).
- **Nature of the Fallback**: Defaulting dropped/unaligned tokens to the `'O'` class is an **engineering design choice** to preserve matrix shape integrity and prevent false entity hallucinations, rather than an official competition rule requiring dropped tokens to be `'O'`. Annexure.pdf defines `'O'` ("Obscure" / "Other") for tokens that do not belong to an aspect entity, but contains no rule dictating tokenizer-level subword drop behavior. This fallback guarantees exact alignment with original whitespace tokens without shifting any subsequent token indices.

### 4.2 Intermediate Disk Caching
- To guard against memory pressure or unexpected interruption during long sequential inference runs, each model's full probability tensor is flattened (`282,444` tokens × 48 classes) and saved to `data/cache/quiz_probs_<model>.npy` (~51.7 MB each).
- Re-running the script automatically detects and loads existing cached probabilities without re-running GPU models.

### 4.3 Sequential GPU Execution
- Models are loaded, evaluated, and unallocated one by one.
- Explicit `del model` and `torch.cuda.empty_cache()` ensure that peak GPU memory is limited to ~1.27 GB VRAM, remaining well within the 6 GB hardware budget.

---

## 5. Standalone Validation Results

Running `python scripts/validate_submission.py submissions/submission_quiz_exp10_baseline.tsv --min-record 5001 --max-record 30000` yielded:
```text
==================================================
SUBMISSION VALIDATION REPORT
==================================================
File Path:                submissions\submission_quiz_exp10_baseline.tsv
File Valid:               YES (COMPLIANT)
Gzip Compressed:          False
Header Present:           True
Total Lines Read:         186462
Data Aspect Rows:         186461
Unique Record Numbers:    25000
Record Range:             [5001, 30000]
Category Distribution:    {1: 101846, 2: 84615}
Aspect Types Extracted:   23 of 29 official aspects

All schema, encoding, and Annexure.pdf formatting checks passed!
```

---

## 6. Official EvalAI Submission & Debugging History

### 6.1 Initial Submission Rejection
The initial submission file `submissions/submission_quiz_exp10_baseline.tsv` was rejected by EvalAI with:
`ValueError: encountered aspect name not in allowed aspect name list for category 2: Bremsscheibenart`

### 6.2 Root Cause Analysis
Per `Annexure.pdf` (pages 11–14), 9 aspects are restricted to Category 1 only (`Bremsscheibenart`, `Bremsscheiben-Aussendurchmesser`, `Farbe`, `Herstellungsland_Und_-Region`, `Material`, `Oberflächenbeschaffenheit`, `Produktlinie`, `Stärke`, `Technologie`) and 6 aspects are restricted to Category 2 only (`Anwendung`, `Breite`, `Länge`, `Menge`, `SAE_Viskosität`, `Zähnezahl`).
Unconstrained token classification produced 297 invalid category-aspect pairs (including 200 instances of `Bremsscheibenart` in Category 2).

### 6.3 Resolution & Corrected Submission
- Extended `scripts/validate_submission.py` to enforce `OFFICIAL_ASPECTS_CAT1` and `OFFICIAL_ASPECTS_CAT2`.
- Generated `submissions/submission_quiz_exp10_baseline_corrected.tsv` by removing only the 297 invalid rows (186,164 valid prediction rows remain across all 25,000 Quiz titles).
- Checksum: `51f880c8a6debb4eaf1a83594eca2d4d2d4a92201d6cb64f9f63eae913b65115`.

### 6.4 Official Leaderboard Results
- **Team**: TeamPiyush
- **Official EvalAI Quiz F0.2 Score**: `0.8824567479388766`
- **Leaderboard Position at Submission**: 13
- **Local Split-A Validation Benchmark**: `0.879852` (Ensemble generalized strongly, +0.0026 above local validation).

