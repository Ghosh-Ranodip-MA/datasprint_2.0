# Methodology & Defense Summary: Script-Bridged Hinglish Intent Classifier

**Competition:** Data Sprint 2.0 (Anveshan, NIE Mysuru) — NLP Track  
**Author / Team:** Ranodip Ghosh  
**Primary Metric:** Unweighted Macro-F1 across 56 classes (55 in-scope domains + Class 55 Fallback)  
**Constraints:** Total Parameters $\le 300\text{M}$ | Full-Pipeline Latency $< 50\text{ms}$ per query  
**Hardware Environment:** 12th Gen Intel(R) Core(TM) i5-12450H | Python 3.10.20 | scikit-learn 1.7.2  
**Submission SHA-256:** `87dcc95f347c50ea28c300bf18527a5e0224aa65e65ba188e0231e962fe5d9e2`

---

## 1. Verified Results & Script-Stratified Ablations (Tasks 1 & 2)

All numbers below were computed locally using a **strict non-leaky twin-grouped 5-fold cross-validation scheme** (`StratifiedGroupKFold` over 13335 disjoint connected-component twin clusters).
- **Train row counts by script:** Devanagari = 10353 rows | Roman = 5645 rows | Mixed = 938 rows (Total = 16936).
- **Test row counts by script:** Roman = 3154 rows (64.2%) | Devanagari = 1604 rows (32.7%) | Mixed = 154 rows (3.1%) (Total = 4912).
- **Test-Weighted Metric:** `0.642 * Roman + 0.327 * Devanagari + 0.031 * Mixed` *(Note: mixed-script F1 is based on only 938 training rows and is flagged as noisy)*.
- **Twin-Group Assertion:** 0 groups span multiple folds, and 0 exact-duplicate transliteration strings with the same label span folds (Fold assignment hash: `c57dcf854e4d0ba5`).

### Script-Stratified Ablation Table

| Model Stage / Configuration | Overall Macro-F1 | Roman Macro-F1 (5645 rows) | Devanagari Macro-F1 (10353 rows) | Mixed Macro-F1 (938 rows, noisy) | Test-Weighted Macro-F1 |
|---|---|---|---|---|---|
| **A. Baseline (Original text only, no augmentation)** | 0.7537 | 0.7272 | 0.7623 | 0.6978 | 0.7378 |
| **B. Augmentation with Transliterated Devanagari (Single-view infer)** | 0.7818 | 0.8139 | 0.7611 | 0.6984 | 0.7931 |
| **C. Stage B + Dual-View TTA** | 0.8082 | 0.8139 | 0.8037 | 0.7395 | 0.8083 |
| **D. Stage C + Prior Logit Adjustment (tau = 0.2)** | **0.8157** | **0.8257** | **0.8097** | **0.7276** | **0.8174** |
| *D (Nested CV tuned tau per fold: [0.2, 0.27, 0.2, 0.2, 0.2])* | *0.8154* | *0.8253* | *0.8099* | *0.7268* | *0.8172* |

> **Key Mechanistic Insight on TTA (Stage B vs C):**  
> Notice that for Roman rows, Macro-F1 is identical (0.8139 vs 0.8139). TTA is an exact no-op for Roman rows because the transliterated text is identical to the original. The entire performance gain of TTA comes from Devanagari rows (0.7611 $\rightarrow$ 0.8037, +0.0426), as projecting Devanagari into the Romanized feature space allows the classifier to utilize Roman n-gram weights.

---

## 2. Tau Selection & Prior Logit Adjustment (Task 3)

Severe class imbalance exists in the dataset (Class 48 has 1,224 rows; Class 36 has only 84 rows). We evaluated temperature-scaled prior logit adjustment:
$$\tilde{P}(y \mid x) \propto P(y \mid x) \cdot \frac{1}{\pi_y^\tau}$$

### Full OOF Tau Sweep (0.0 to 1.0)
```text
tau = 0.00: Macro-F1 = 0.8082
tau = 0.10: Macro-F1 = 0.8112
tau = 0.15: Macro-F1 = 0.8137
tau = 0.20: Macro-F1 = 0.8157  <-- Peak Macro-F1
tau = 0.25: Macro-F1 = 0.8151
tau = 0.30: Macro-F1 = 0.8148
tau = 0.50: Macro-F1 = 0.8128
tau = 1.00: Macro-F1 = 0.7921
```
To avoid selection bias, we verified this via **nested cross-validation**: in each fold, $\tau$ was chosen on the 4 training folds and evaluated strictly on the unseen fold. The selected values were `[0.2, 0.27, 0.2, 0.2, 0.2]` (median = `0.2`), achieving an honest nested Macro-F1 of **0.8154**.

---

## 3. Class 55 & Fallback Evaluation (Task 4)

- **Tier 1 (In-Domain Supervised Fallback):** Class 55 in train consists of 249 voice commands representing unhandled/fallback domains (alarms, podcasts, coffee, board games).
  - **Out-of-Fold Real Class 55 Precision:** **0.8652**
  - **Out-of-Fold Real Class 55 Recall:** **0.7992**
  - **Out-of-Fold Real Class 55 F1-Score:** **0.8309**
- **Tier 2 (Open-Set Fallback Rule):** A strict length rule ($> 250$ characters) flags generic long narrative prose present in test that has no voice assistant properties. It changes exactly **1 test row** (Row 901594).
- **Pseudo-OOS Multi-Rotation Experiment:** In a 3-rotation held-out domain experiment (holding out 3 real classes per rotation to simulate open-set queries), the maximum-softmax detector achieved an average **AUROC of 0.9162** (Rotations: [0.9385, 0.8908, 0.9194]).

---

## 4. Test Prediction Skew Findings (Task 5)

The submission predicts Class 55 on 309 of 4,912 test queries (6.3%), compared to 1.47% in train. Our empirical investigation reveals:
1. Test Class 55 counts (309 rows, 6.3%) reflect that the test set contains a substantial density of alarm, podcast, and coffee queries that train explicitly labeled as Class 55.
2. Over 85% of test queries routed to Class 55 contain explicit keywords ('alarm', 'alaarm', 'अलार्म', 'podcast', 'podkaast', 'kophee', 'khel').
3. Over-predicted classes 7, 28, 34, and 40 are small tail classes (each <= 95 train examples); their test queries exhibit high vocabulary density in short intent commands.
4. Under-predicted classes 15 and 37 correspond to specific transactional requests (e.g., specific radio/media queries) that appear with lower relative frequency in the test distribution.
5. Prior logit adjustment with tau=0.200 was verified via nested 5-fold cross-validation to maximize Macro-F1 across all 56 classes without distortion.
6. No manual post-hoc class clamping is justified, as modifying test frequencies without ground truth hurts unweighted Macro-F1 on genuine test distribution shifts.

---

## 5. Constraint Compliance & Honesty (Task 7)

| Criterion | Requirement | Measured Value | Status |
|---|---|---|---|
| **Trainable Parameters** | $\le 300\text{M}$ | **4,509,680 (4.51M)** | **COMPLIANT ($66\times$ under limit)** |
| **Inference Latency (p50)** | $< 50\text{ms}$ | **15.335 ms** | **COMPLIANT** |
| **Inference Latency (p95)** | $< 50\text{ms}$ | **22.821 ms** | **COMPLIANT ($2.1\times$ faster than limit)** |
| **Inference Latency (p99)** | $< 50\text{ms}$ | **23.779 ms** | **COMPLIANT** |

*Note on Parameter Counting:* Exactly 4,509,680 parameters represent the logistic regression coefficients ($56 \times 80530$) plus 56 intercepts. The 57,246 character n-gram and 23,283 word n-gram vectorizer dictionary tokens and IDF vectors are fixed, non-trainable preprocessing mappings.

---

## 6. Live Interactive Demo Outputs (Task 6)

The following outputs were produced directly by the production `predict(text)` function in the executed notebook:

```python
# Query: 'kripaya rasoeeghar ki battiyaan band kar den'
# -> Label: 39, Confidence: 0.882, Latency: 24.96ms, Fallback: False
# Query: 'कृपया रसोईघर की बत्तियां बंद कर दें'
# -> Label: 39, Confidence: 0.937, Latency: 30.51ms, Fallback: False
# Query: 'kya kal subah ke liye koi alarm set hai mera bhai'
# -> Label: 55, Confidence: 0.613, Latency: 26.69ms, Fallback: True
# Query: 'volyum star ko adhiktam mein badlen'
# -> Label: 28, Confidence: 0.791, Latency: 27.26ms, Fallback: False
# Query: 'bina tel ke svadisht snaiks kaise pakaaen'
# -> Label: 36, Confidence: 0.909, Latency: 20.82ms, Fallback: False
```

---

## 7. Robustness Experiment (Task 8)

We evaluated phonetic and ASR-noise injection (matra drops, `h` for `hai`, `kia` for `kya`, `mjhe`, vowel collapses `aa->a`, `ee->i`, `oo->u`) on Roman queries:
- **Clean Validation Macro-F1:** **0.7431**
- **Noised Validation Macro-F1:** **0.7464**
- *Decision:* The noise model improved noised validation performance from 0.7431 to 0.7464 without destabilizing clean validation. However, to keep the final pipeline minimalist, deterministic, and 100% reproducible, the primary submission relies on the clean dual-view phonetic bridge.

---

## 8. Known Limitations (Task 6)

1. **Class 55 In-Domain Semantics:** In the competition dataset, Class 55 is not purely synthetic out-of-domain noise; it contains legitimate voice commands for alarms, podcasts, and coffee. Consequently, queries like *"kya kal subah ke liye koi alarm set hai"* are routed to Class 55 because train explicitly labels them as such.
2. **Mixed-Script Variance:** Mixed-script training queries account for only 938 rows (5.5% of train). Hence, mixed-script F1 carries higher sample variance than Roman or Devanagari subsets.
3. **Hardware Latency:** Full-pipeline latency was profiled on an `12th Gen Intel(R) Core(TM) i5-12450H` at 22.821ms (p95). On lower-power machines or under heavy multiprocessing contention, latency may vary.

---

## 9. Reference Floor Comparison (Task 6)

The organizer's reference baseline (char n-gram TF-IDF + LinearSVC) published leaderboard scores:
- **Organizer Public Leaderboard Floor:** `0.52000`
- **Organizer Private Leaderboard Floor:** `0.50542`
- **Our System Non-Leaky CV Score:** **0.8157** (Test-Weighted: **0.8174**)
