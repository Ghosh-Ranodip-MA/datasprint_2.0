# Script-Bridged, Shift-Robust Hinglish Intent Classifier
### Data Sprint 2.0 (Anveshan, The National Institute of Engineering, Mysuru) — NLP Track

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Metric](https://img.shields.io/badge/Metric-Unweighted%20Macro--F1-success.svg)](https://www.kaggle.com/competitions/anveshan-data-sprint-2-nlp)
[![Parameters](https://img.shields.io/badge/Trainable%20Params-4.51M%20%28%E2%89%A4%20300M%29-brightgreen.svg)]()
[![Latency](https://img.shields.io/badge/Latency%20p95-22.8ms%20%28%3C%2050ms%29-brightgreen.svg)]()
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

---

## 📌 Table of Contents
1. [Executive Summary & Verified Results](#executive-summary--verified-results)
2. [Dataset Overview & Inversion Shift](#dataset-overview--inversion-shift)
3. [Methodological Novelty & Architecture](#methodological-novelty--architecture)
   - [N1: Calibrated Phonetic Devanagari $\rightarrow$ Roman Transliteration Bridge](#n1-calibrated-phonetic-devanagari--roman-transliteration-bridge)
   - [N2: Dual-View Representation & Test-Time Augmentation (TTA)](#n2-dual-view-representation--test-time-augmentation-tta)
   - [N3: Non-Leaky Twin-Grouped Cross-Validation Protocol](#n3-non-leaky-twin-grouped-cross-validation-protocol)
   - [N4: Two-Tier Fallback Calibration (Class 55 In-Domain + Open-Set)](#n4-two-tier-fallback-calibration-class-55-in-domain--open-set)
   - [N5: Class-Balanced Prior Logit Adjustment](#n5-class-balanced-prior-logit-adjustment)
4. [Script-Stratified Ablation Table (Tasks 1 & 2)](#script-stratified-ablation-table-tasks-1--2)
5. [Strict Compliance & Hardware Verification](#strict-compliance--hardware-verification)
6. [Repository Structure & Reproduction](#repository-structure--reproduction)
7. [Live Interactive Inference Demo](#live-interactive-inference-demo)
8. [Known Limitations & Boundary Conditions](#known-limitations--boundary-conditions)

---

## 📊 Executive Summary & Verified Results

This repository contains the complete, reproducible submission for **Data Sprint 2.0 (Anveshan, NIE Mysuru) NLP Track**. All numbers reported below are generated directly by [`notebook.ipynb`](notebook.ipynb) and [`audit_and_compute.py`](audit_and_compute.py), exported to `results.json`, and injected automatically into documentation.

| Evaluation Metric / Constraint | Competition Threshold | Baseline | Our Audited Result | Status |
|---|---|---|---|---|
| **Macro-F1 (Non-Leaky CV)** | Maximize | `0.7537` | **`0.8157`** (Nested CV: **`0.8154`**) | **+6.20% Improvement** |
| **Test-Weighted Macro-F1** | Maximize | `0.7378` | **`0.8174`** | **+7.96% Improvement** |
| **Trainable Parameters** | $\le 300\text{ M}$ | — | **`4,509,680` ($4.51\text{ M}$)** | **$66\times$ under budget** |
| **Full-Pipeline Latency (p95)** | $< 50\text{ ms}$ | — | **`22.82 ms`** (p50: `15.34 ms`) | **$2.1\times$ faster than limit** |
| **Submission Integrity** | 4,912 rows, matching IDs | — | Exact match (`87dcc95f...`) | **100% Verified** |

---

## ⚠️ Dataset Overview & Inversion Shift

The dataset consists of Hindi and Hinglish conversational voice assistant utterances across **56 classes** (55 in-scope intents + Class 55 Fallback).

```
TRAIN SPLIT (16,936 rows):
  ████████████████████████████████████████ 61.1% Devanagari (10,353 rows)
  █████████████████████ 33.3% Roman (5,645 rows)
  ███ 5.5% Mixed (938 rows)

TEST SPLIT (4,912 rows):
  ████████████████████████████████████████████ 64.2% Roman (3,154 rows)
  █████████████████████ 32.7% Devanagari (1,604 rows)
  ██ 3.1% Mixed (154 rows)
```

- **The Script Inversion Shift:** 64.2% of test queries are in Roman Hinglish, while the training set is predominantly Devanagari (61.1%). Standard models trained without script bridging experience severe domain degradation.
- **The Twin Pair Phenomenon:** 36% of Roman queries in train have exact or near-identical Devanagari translation twins. Naive random/stratified splits allow twins to straddle folds, producing an artificially optimistic CV. We designed a connected-component twin clustering protocol to eliminate cross-split leakage.

---

## 🔬 Methodological Novelty & Architecture

### N1: Calibrated Phonetic Devanagari $\rightarrow$ Roman Script Bridge
A deterministic mapper tuned to the dataset's Romanization conventions:
- Long vowels: `aa/ee/oo` (`रसोईघर` $\rightarrow$ `rasoeeghar`, `शुरू होंगी` $\rightarrow$ `shuru hongee`).
- Nukta normalization (`क़, ख़, ग़, ज़, ड़, ढ़, फ़`).
- Schwa deletion at word boundaries.
- Increases unseen Roman test token coverage from **85.5% to 89.8%**, and unique word type coverage from **54.6% to 63.7%**.

### N2: Dual-View Representation & Test-Time Augmentation (TTA)
- Training utterances in Devanagari/mixed script are augmented with their transliterated Roman view.
- At inference time, queries are vectorized across both views in a single unified call:
  $$P(\text{class} \mid x) = \frac{1}{2} P(x_{\text{orig}}) + \frac{1}{2} P(x_{\text{trans}})$$
- **TTA Mechanism:** TTA is an exact no-op for Roman rows (where $x_{\text{orig}} = x_{\text{trans}}$). Its entire gain comes from projecting Devanagari queries into the Romanized feature space.

### N3: Non-Leaky Twin-Grouped Cross-Validation Protocol
- Graph partitioning over exact duplicate transliterations and near-duplicate cosine neighbors ($\ge 0.75$) groups train into **13,335 disjoint twin clusters**.
- Evaluated with `StratifiedGroupKFold(n_splits=5)`.
- Enforces strict assertions: zero groups span multiple folds, and zero duplicate transliteration strings span multiple folds.

### N4: Two-Tier Fallback Calibration (Class 55)
- **Tier 1 (In-Domain Fallback):** Real Class 55 voice commands in train (alarms, podcasts, coffee, board games):
  - **OOF Precision = 0.8652 | Recall = 0.7992 | F1 = 0.8309**.
- **Tier 2 (Open-Set Fallback):** Generic narrative prose ($> 250$ chars) has no voice assistant properties and is filtered to Class 55 (changes exactly 1 test row).
- **Pseudo-OOS Multi-Rotation Experiment:** Rotating 3 held-out classes across runs yields a mean maximum-softmax **AUROC of 0.9162**.

### N5: Class-Balanced Prior Logit Adjustment
Severe class imbalance ($14.6\times$) is resolved via post-hoc temperature-scaled prior logit adjustment (Menon et al., 2021):
$$\tilde{P}(y \mid x) \propto P(y \mid x) \cdot \frac{1}{\pi_y^\tau}, \quad \tau = 0.200$$
Verified via nested 5-fold cross-validation ($\tau \in [0.20, 0.27]$, median $= 0.200$), lifting Overall Macro-F1 from **0.8082 to 0.8157**.

---

## 📈 Script-Stratified Ablation Table (Tasks 1 & 2)

Measured strictly on the 13,335 twin-grouped folds:

| Model Stage / Configuration | Overall Macro-F1 | Roman Macro-F1 (5,645 rows) | Devanagari Macro-F1 (10,353 rows) | Mixed Macro-F1 (938 rows, noisy) | Test-Weighted Macro-F1 |
|---|---|---|---|---|---|
| **A. Baseline (Original text only, no augmentation)** | 0.7537 | 0.7272 | 0.7623 | 0.6978 | 0.7378 |
| **B. Augmentation with Transliterated Devanagari (Single-view infer)** | 0.7818 | 0.8139 | 0.7611 | 0.6984 | 0.7931 |
| **C. Stage B + Dual-View TTA** | 0.8082 | 0.8139 | 0.8037 | 0.7395 | 0.8083 |
| **D. Stage C + Prior Logit Adjustment ($\tau = 0.200$)** | **0.8157** | **0.8257** | **0.8097** | **0.7276** | **0.8174** |
| *D (Nested CV tuned $\tau$ per fold: `[0.2, 0.27, 0.2, 0.2, 0.2]`)* | *0.8154* | *0.8253* | *0.8099* | *0.7268* | *0.8172* |

---

## 🛡️ Strict Compliance & Hardware Verification

- **Trainable Parameters:** Exactly **`4,509,680` parameters (4.51M)** ($56 \times 80,529$ coefficients + 56 intercepts) $\ll 300\text{M}$ limit. The 57,246 char and 23,283 word n-gram vocabularies are fixed, non-trainable preprocessing mappings.
- **Inference Latency Profile (600 queries after warmup on 12th Gen Intel i5-12450H):**
  - **p50 (Median):** `15.34 ms`
  - **p95:** `22.82 ms` ($< 50\text{ ms}$ threshold)
  - **p99:** `23.78 ms`

---

## 📁 Repository Structure & Reproduction

```text
datasprint/
├── train.csv                     # Training dataset (16,936 rows, 56 targets)
├── test.csv                      # Test dataset (4,912 rows)
├── sample_submission.csv         # Official submission template
├── submission.csv                # Final competition submission (SHA-256: 87dcc95f...)
├── notebook.ipynb                # Executed reproducible Jupyter Notebook with fresh outputs
├── METHODOLOGY_SUMMARY.md        # Synced defense brief generated from results.json
├── audit_and_compute.py          # Standalone computation script producing results.json
├── run_final_pipeline.py         # Production runner supporting --from-scratch
├── requirements.txt              # Pinned environment dependencies
├── results.json                  # Single source of truth containing all metrics
└── README.md                     # Comprehensive project documentation
```

### Reproduction Steps
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the production pipeline from scratch
python run_final_pipeline.py --from-scratch

# 3. Regenerate documentation from results.json
python generate_summary.py
```

---

## 💻 Live Interactive Inference Demo

```python
from run_final_pipeline import predict

# Verified query outputs from executed notebook:
# 1. Roman Kitchen Light (Class 39)
predict("kripaya rasoeeghar ki battiyaan band kar den")
# -> Label: 39, Conf: 0.882, Latency: 24.96ms, Fallback: False

# 2. Devanagari Kitchen Light (Class 39)
predict("कृपया रसोईघर की बत्तियां बंद कर दें")
# -> Label: 39, Conf: 0.937, Latency: 30.51ms, Fallback: False

# 3. Fallback Domain Command (Class 55)
predict("kya kal subah ke liye koi alarm set hai mera bhai")
# -> Label: 55, Conf: 0.613, Latency: 26.69ms, Fallback: True
```

---

## ⚠️ Known Limitations & Boundary Conditions

1. **Class 55 In-Domain Semantics:** Class 55 in train contains ordinary-looking commands (alarms, podcasts, coffee, chess). The system routes alarm queries to 55 because train explicitly labels them as such.
2. **Mixed-Script Sample Size:** Mixed-script queries account for only 938 training examples (5.5%). Hence, mixed-script F1 carries higher sample variance than Roman or Devanagari subsets.
3. **Hardware Latency:** Profiling was conducted on an Intel i5-12450H CPU. Inference latency may vary across different hardware architectures.

---
*Developed for Data Sprint 2.0 (Anveshan, NIE Mysuru) by Ranodip Ghosh.*
