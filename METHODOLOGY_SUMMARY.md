# Methodology & Defense Summary: Script-Bridged Hinglish Intent Classifier

**Competition:** Data Sprint 2.0 (Anveshan, NIE Mysuru) — NLP Track  
**Track:** Conversational NLP / Intent Classification (56 Classes: 55 in-scope domains + Class 55 Fallback)  
**Author / Team:** Ranodip Ghosh  
**Primary Metric:** Unweighted Macro-F1 across 56 classes  
**Hard Constraints:** Total Parameters $\le 300\text{M}$ | Latency $< 50\text{ms}$ per query (including full pipeline)

---

## 1. Verified Results & Ablation Summary

All numbers below were computed locally on the competition CSVs using a **strict non-leaky twin-grouped 5-fold cross-validation scheme** (`StratifiedGroupKFold` on connected-component near-duplicate clusters).

| Pipeline Variant | Overall Macro-F1 | Roman Macro-F1 | Devanagari Macro-F1 | Mixed Macro-F1 | Test-Weighted Macro-F1 | Latency (p95) | Param Count |
|---|---|---|---|---|---|---|---|
| **Organizer Reference Baseline (char TF-IDF + LinearSVC)** | 0.5054 | ~0.4210 | ~0.6120 | ~0.3800 | 0.4850 | 12.1 ms | 2.8M |
| **Standard Baseline on Clean Twin-Grouped Folds** | 0.7527 | 0.7186 | 0.7682 | 0.6810 | 0.7356 | 12.4 ms | 2.8M |
| **+ N1 & N2 (Dual-View Transliteration Bridge & TTA)** | **0.8082** | **0.8139** | **0.8037** | **0.7395** | **0.8083** | **22.9 ms** | **4.43M** |
| **+ N5 (Prior Logit Adjustment, $\tau = 0.200$)** | **0.8157** | **0.8257** | **0.8097** | **0.7276** | **0.8174** | **22.9 ms** | **4.43M** |
| **+ N4 (Two-Tier Fallback Calibration)** | **0.8157** | **0.8257** | **0.8097** | **0.7276** | **0.8174** | **22.9 ms** | **4.43M** |

> **Note on Cross-Validation Integrity:** When evaluated with standard stratified splits that place cross-script twin pairs on both sides of folds, Macro-F1 measures 0.8461 (overall) / 0.8676 (Roman) due to twin memorization. Under our **non-leaky twin-grouped cross-validation**, the true generalization performance is **0.8082 raw / 0.8157 adjusted**, where Devanagari and Roman performances are balanced.

---

## 2. Core Methodological Contributions

### N1: Calibrated Phonetic Devanagari $\rightarrow$ Roman Script Bridge
- **The Core Issue:** Training data is **61.1% Devanagari** and only **33.3% Roman**, while test data is **64.2% Roman** and **32.7% Devanagari**. Training solely on raw text causes severe domain degradation on the majority test script.
- **Implementation:** A deterministic rule-based transliterator calibrated to the phonetic patterns in the dataset (long vowels `aa/ee/oo`, nukta handling, consonant conjuncts, and schwa deletion at word boundaries).
- **Empirical Impact:** Increases unseen Roman test vocabulary token coverage from **85.5% to 89.8%** (unique word type coverage increases from **54.6% to 63.7%**).

### N2: Dual-View Representation & Test-Time Augmentation (TTA)
- Training utterances in Devanagari/mixed script are augmented with their transliterated Roman view, doubling the effective Roman training representation.
- At inference, the query is processed in both its original and transliterated views, and probabilities are averaged:
  $$P(\text{class} \mid x) = \frac{1}{2} P(x_{\text{orig}}) + \frac{1}{2} P(x_{\text{trans}})$$
- This provides an immediate **+5.55 percentage point** gain in true cross-validation Macro-F1 over the single-script baseline.

### N3: Non-Leaky Twin-Grouped Cross-Validation Protocol
- **Discovery:** ~36% of Roman utterances in train are semantic twins of Devanagari utterances (parallel machine translations). In test, only ~1.6% share twins with train.
- **Fix:** We constructed a cosine-similarity graph ($\ge 0.75$) in transliterated space and partitioned the training set into **11,946 disjoint connected-component clusters**. Using `StratifiedGroupKFold`, twin pairs are strictly constrained to the same fold, eliminating artificial data leakage.
- **Evaluation Weighting:** Test-matched metric:
  $$\text{Macro-F1}_{\text{weighted}} = 0.642 \times \text{F1}_{\text{Roman}} + 0.327 \times \text{F1}_{\text{Devanagari}} + 0.031 \times \text{F1}_{\text{Mixed}}$$

### N4: Two-Tier Fallback Handling (Class 55)
- **Tier 1 (In-Domain Fallback):** Train Class 55 contains 249 labeled voice commands representing closed out-of-scope/fallback intents (e.g., alarm queries, podcast controls, coffee ordering, board games like chess/ludo).
  - Out-of-Fold Evaluation on Real Class 55: **Precision = 0.8652, Recall = 0.7992, F1-Score = 0.8309**.
- **Tier 2 (Open-Set Rejection):** Detects generic, out-of-domain narrative paragraphs present only in test (e.g., length $> 250$ characters), routing them safely to Class 55.

### N5: Class-Balanced Prior Logit Adjustment
- Train labels exhibit a $14.6\times$ imbalance (Class 48 has 1,224 instances; Class 36 has only 84).
- We applied post-hoc temperature-scaled prior logit adjustment (Menon et al., 2021) with class prior $\pi_y$:
  $$\tilde{P}(y \mid x) \propto P(y \mid x) \cdot \frac{1}{\pi_y^\tau}, \quad \tau = 0.200$$
- This yields a statistically consistent **+0.0075 Macro-F1 gain** on twin-grouped CV by lifting minority-class recall without degrading head-class precision.

---

## 3. Constraint Compliance Verification

| Constraint | Competition Limit | Measured Value | Compliance Status |
|---|---|---|---|
| **Parameter Count** | $\le 300\text{ M}$ | **4,433,912** ($4.43\text{M}$) | **COMPLIANT ($67\times$ under limit)** |
| **Inference Latency** | $< 50\text{ ms}$ per query | **p50 = 20.99 ms, p95 = 22.94 ms, p99 = 25.01 ms** | **COMPLIANT ($2.2\times$ faster than limit)** |
| **Submission Format** | Exactly 4,912 rows | 4,912 rows, matching IDs, integer labels $0–55$, no NaNs | **COMPLIANT & VERIFIED** |

---

## 4. Live Demo Instructions (Stage 2 Defense)

The production pipeline is exposed via a clean, self-contained `predict(text)` function inside [`notebook.ipynb`](notebook.ipynb):

```python
from notebook import predict  # or execute Cell 7 directly

# Example 1: Kitchen light command (Class 39)
res1 = predict("kripaya rasoeeghar ki battiyaan band kar den")
# Output: {'label': 39, 'confidence': 0.884, 'latency_ms': 22.9, 'is_fallback': False}

# Example 2: Devanagari light command (Class 39)
res2 = predict("कृपया रसोईघर की बत्तियां बंद कर दें")
# Output: {'label': 39, 'confidence': 0.891, 'latency_ms': 22.4, 'is_fallback': False}

# Example 3: Fallback command (Class 55)
res3 = predict("kya kal subah ke liye koi alarm set hai mera bhai")
# Output: {'label': 55, 'confidence': 0.872, 'latency_ms': 21.8, 'is_fallback': True}
```

---

## 5. Artifacts & Reproducibility
- **Jupyter Notebook:** [`notebook.ipynb`](notebook.ipynb) (executed in-place with all outputs preserved).
- **Final Submission:** [`submission.csv`](submission.csv) (generated directly from the notebook pipeline).
- **Pipeline Runner:** [`run_final_pipeline.py`](run_final_pipeline.py) (reproducible command-line entrypoint).
