# Data Sprint 2.0 (Anveshan, NIE Mysuru) — NLP Track
## Project Brief: Script-Bridged, Shift-Robust Hinglish Intent Classifier

> **Audience:** an AI coding agent (Antigravity) that will turn this brief into a single, reproducible Jupyter notebook (`.ipynb`).
> **Owner:** Ranodip. **Compute available:** NVIDIA DGX Spark class machine (128 GB unified memory) — plenty for everything below. Confirm GPU/CPU at runtime and adapt.
> **Status of claims:** Numbers marked **[MEASURED]** were computed on the real CSVs. Everything marked **[HYPOTHESIS]** must be verified by the notebook before being relied upon. Do not present hypotheses as facts in the final write-up.

---

## 1. Task in one paragraph

Build a compact NLU pipeline (**≤ 300M parameters total**) that classifies conversational Hindi / Hinglish voice-assistant utterances into **56 classes**: 55 supported domains (IDs 0–54) plus **Class 55 = Out-of-Scope / fallback**. The metric is **unweighted Macro-F1 over all 56 classes** (each class weighs 1/56). Competition: <https://www.kaggle.com/competitions/anveshan-data-sprint-2-nlp>.

### Hard constraints
| Constraint | Detail |
|---|---|
| Parameters | ≤ 300M for the whole pipeline. Assume ensemble members are **summed** unless organizers say otherwise. |
| Latency | **< 50 ms** per query at inference (judged in the live demo). |
| Submissions | Max **5** on Kaggle. Keep 1–2 in reserve. Do not tune to the public leaderboard. |
| Leaderboard | Public = 19.83% (974 rows, ~17/class → very noisy). Private = 80.17% (3,938 rows) decides Stage 1. |
| Scoring | Stage 1 (40%): private Macro-F1. Stage 2 (60%): live 5-min demo — Hindi/Hinglish robustness, Class 55 fallback, latency, code verification. |
| License | Data is CC BY-NC-SA 4.0. |
| Reference floor | Organizer baseline: char n-gram TF-IDF + LinearSVC → **0.52000 public / 0.50542 private**. |

**Therefore the deliverable is a deployable system plus a CSV, not just a CSV.** The notebook must end with a working inference function and a latency benchmark.

---

## 2. Data (files in `/mnt/user-data/uploads/` locally; on Kaggle `/kaggle/input/anveshan-data-sprint-2-nlp/`)

| File | Rows | Columns |
|---|---|---|
| `train.csv` | 16,936 | `example_id` (100001–116936), `query_text`, `target` (0–55) |
| `test.csv` | 4,912 | `example_id` (900001–904912), `query_text` |
| `sample_submission.csv` | 4,912 | `example_id`, `target` (placeholder = 48 everywhere) |

The submission must contain exactly the test `example_id`s in order, with integer `target` in 0–55.

### 2.1 Verified facts **[MEASURED]**
- No missing values; **0 duplicate train queries; 0 conflicting labels**.
- **0% of test queries appear exactly in train** (no memorization shortcut).
- Class sizes: min 84 (class 36), max 1,224 (class 48). Largest: 48 (1,224), 44 (1,009), 12 (902), 13 (896), 31 (861). Smallest: 36 (84), 28 (85), 40 (85), 23 (87), 15 (90). Class 55: 249 rows. Imbalance ≈ 14.6×.
- Script composition (by character class): 

| | Devanagari-only | Roman-only | Mixed |
|---|---|---|---|
| train | 61.1% | 33.3% | 5.5% |
| test | 32.7% | **64.2%** | 3.1% |

- Length: train mean 37.7 chars / 7.6 words (max 204 chars); test mean 40.3 chars / 7.9 words (max **462** chars). Rows >100 chars: train 123, test 34. Rows >250 chars: train 0, test 1.
- Per-class Roman share in train is nearly constant (mean 0.34, std 0.05) → script is not leaking into labels.

### 2.2 Qualitative observations
- Text looks like **machine-translated** English intents (MASSIVE-style) into Hindi. **[HYPOTHESIS]**
- The Roman text looks **machine-transliterated from Devanagari with a systematic scheme plus noise**: vowel lengthening (`aa`, `ee`, `oo`), phonetic spellings (`rasoeeghar`, `imel`, `baidmintan`, `myujik pleyar`, `kuki resipi`, `seerij podkaast`), and colloquial shortenings (`h` for `hai`, `kia` for `kya`, `mjhe`, `plz`). **[HYPOTHESIS but strongly suggested by samples]**
- Code-mixed rows exist in train, e.g. `alexa लिविंग रूम की लाइट्स को बंद कर दो`, `please facebook पर एक status सबमिट करो`.
- **Class 55 in train does not look like "crypto / coding / homework".** A sample of 25 rows was mostly ordinary commands (alarm set/list, podcast control, coffee, chess, "next episode"). **[HYPOTHESIS]:** train's Class 55 is one or more whole intents relabeled as fallback, i.e. a closed, learnable class.
- **Test contains a different OOS style:** at least one very long Roman paragraph of generic prose (e.g. about young drivers and road accidents) that has no analogue in train. **[HYPOTHESIS]:** the organizers injected generic-text OOS in test only.

---

## 3. The central problem: script shift

Baseline experiments with char_wb (2–5)-gram TF-IDF + LinearSVC(C=0.5), my own splits **[MEASURED]**:

| Setup | Macro-F1 |
|---|---|
| Random 80/20 split of train | **0.776** |
| Train on Devanagari+mixed only → evaluate on Roman | **0.057** |
| Train on Roman only → evaluate on held-out Roman | 0.713 |

Interpretation: a character model trained on one script transfers almost nothing to the other. Test is 64% Roman while train is 33% Roman, so the model sees a minority of the script it will mostly be tested on. The gap between 0.776 (random CV) and 0.505 (organizer private score) is largely this shift. **Random K-fold CV will over-estimate performance. Do not use it as the model-selection criterion.**

---

## 4. Proposed approach (the novelty)

Working title: **"Script-Bridged Dual-View Learning with Two-Tier Fallback for Hinglish Intent Detection."**

The novelty is the combination, each part motivated by a measured property of this data:

### N1 — Script bridge via a learned/engineered Devanagari→Roman transliterator that mimics the *test* romanization
- Goal: convert the 61% Devanagari train rows into Roman text that looks like the Roman test text, doubling effective Roman training data (33% → ~94% of rows available in Roman form), and optionally the reverse for a Devanagari view.
- **Step A (rule-based, always do this first):** write a deterministic Devanagari→Roman mapper (consonants, matras, nukta, anusvara/chandrabindu, halant, conjuncts, digits). Calibrate its conventions to the observed Roman data: long vowels `aa/ee/oo`, `kh/gh/chh/jh`, `ph` vs `f`, `v` vs `w`, `sh/s`, `n` vs `m` for anusvara.
- **Step B (calibrate against reality):** pick an objective that needs no parallel data: **token-level coverage** — the fraction of Roman *test* tokens found in the vocabulary of transliterated-Devanagari-train. Tune the mapper's choices to maximize it. Also report character-trigram distribution distance between transliterated-Dev and real Roman train.
- **Step C (optional, if allowed offline/on Kaggle):** AI4Bharat IndicXlit as an alternative/ensemble transliterator; compare coverage vs the rule-based one. Verify license/internet availability first; if unavailable, skip.
- **Step D (noise model):** apply ASR/phonetic noise to the generated Roman text: matra drop, vowel lengthening/shortening, `v/w`, `ph/f`, `sh/s`, `t/d` retroflex merging, `h` for `hai`, `kya→kia`, word split/merge, repeated letters, dropped letters. Make noise rates configurable, and **fit them to the observed Roman train/test noise** (e.g., compare token-frequency of known variants) rather than guessing.
- Optional extra: Roman→Devanagari back-view only if a decent reverse mapper is feasible; otherwise skip.

### N2 — Dual-view training and test-time augmentation (TTA)
- Train on the **original** text and the **transliterated** view (each row represented in both scripts where available), so the encoder sees every intent in both scripts.
- At inference, predict on the original text and on the normalized view and **average probabilities** (weights tuned on validation).

### N3 — Test-matched validation (this is part of the contribution, not an afterthought)
- Stratify folds by **(class × script)**, so every fold has Roman rows for every class.
- Report **Macro-F1 separately on Roman-only and Devanagari-only validation subsets**, plus a **re-weighted Macro-F1 that mimics test composition** (≈64% Roman / 33% Devanagari / 3% mixed).
- **Adversarial validation:** train a classifier to distinguish train vs test; use its scores to (a) find train rows most test-like and (b) report a "test-likeness-weighted" CV score.
- Model selection is done on these, **never on the public leaderboard**.

### N4 — Two-tier fallback (Class 55)
1. **Tier 1 — learned Class 55** as an ordinary class (it has 249 labeled rows).
2. **Tier 2 — open-set rejection** for generic, off-domain text that train never shows. Candidates, in increasing cost: (a) maximum-softmax / energy score thresholds, (b) kNN or Mahalanobis distance in the encoder's embedding space, (c) a length/atypicality heuristic combined with low confidence (long prose is not a voice command). Choose by validation.
- **Pseudo-OOS evaluation protocol:** since test OOS may be unlike train OOS, hold out several *real in-scope classes* (e.g. 3–5 per run, rotating) and relabel them as OOS during training/evaluation, to measure whether Tier 2 generalizes to unseen topics. Also build **synthetic OOS** (generic Hinglish chit-chat, finance/crypto, coding, homework, long paragraphs) — write them with a local LLM if permitted, or hand-write 100–300 templates — as outlier-exposure negatives.
- Report OOS precision/recall explicitly; the demo will probe this.

### N5 — Macro-F1-targeted decision layer
- Logit adjustment or class-balanced / focal loss for tail classes.
- After training, **per-class additive logit biases** tuned on out-of-fold predictions to maximize Macro-F1 (coordinate ascent, with regularization to avoid overfitting the ~17 rows/class scale of the public split).

### N6 — Constraint-aware engineering (needed for Stage 2)
- Backbone shortlist (verify param counts in code with `sum(p.numel())`): MuRIL-base, IndicBERT v2, XLM-R base, mDeBERTa-v3-base, multilingual-e5-base, HingBERT/HingRoBERTa. Models >300M (LaBSE, BGE-M3, XLM-R large) are out.
- **Vocabulary trimming** of the embedding matrix to tokens occurring in train+test (Hindi/Hinglish) to shrink multilingual models substantially, enabling a small ensemble under 300M.
- **Knowledge distillation** of the ensemble into one student for speed.
- **ONNX Runtime + int8 dynamic quantization**, max sequence length ~48–64 tokens (dynamic padding), batch size 1; benchmark p50/p95 latency on CPU and GPU.
- Include the char-ngram TF-IDF+linear model as a cheap ensemble member and count its parameters honestly (coef matrix size).

### N7 — Semi-supervised adaptation on test text (**only if the rules allow it; verify with organizers**)
- Domain-adaptive MLM on train+test text, and/or self-training with high-confidence pseudo-labels on test. Expected to help because 64% of test is in the minority-script of train. Provide a flag `USE_TEST_TEXT = False` default; enable only after confirming legality. Never use any test labels (none exist).

---

## 5. Notebook structure (what the agent should build)

One notebook, top-to-bottom reproducible, with a `CONFIG` cell at the top (paths, seeds, flags, max_len, model names, noise rates). Use **seed control** everywhere and log package versions.

1. **Setup & config** — imports, device detection, paths (support both local uploads path and Kaggle input path), seeds.
2. **Data loading & sanity checks** — shapes, ID ranges, submission template compatibility; assert the numbers in §2.1 (print any mismatch).
3. **EDA** — class distribution plot, script-composition by split and by class, length distributions, rare-class table, 20 random rows per class for the 10 smallest classes + classes 48 and 55 (to understand label meaning; the class IDs have no names). Include the **Class-55 cluster analysis**: embed the 249 rows (TF-IDF or sentence embeddings), cluster, and print cluster exemplars to confirm/refute the "relabeled intents" hypothesis.
4. **Shift diagnostics** — train/test adversarial validation; baseline reproduction (random split vs script-held-out); show the 0.776 / 0.057 / 0.713 pattern.
5. **Transliteration bridge (N1)** — mapper, coverage metric, noise model, side-by-side examples (Devanagari → generated Roman vs real Roman), parameter fit.
6. **Validation framework (N3)** — (class × script) stratified K-fold (K=5), metrics: overall Macro-F1, Roman-only, Devanagari-only, test-composition-weighted Macro-F1, per-class F1 table, confusion pairs.
7. **Baselines** — char TF-IDF + LinearSVC (organizer-style), then word+char TF-IDF + LR/SVM with dual-view data. Show the gain from N1/N2 *with ablations*.
8. **Transformer fine-tuning** — each shortlisted backbone with: dual-view data, label smoothing, layer-wise LR decay, EMA, optional R-Drop / SupCon+CE, class-balanced loss or logit adjustment. 5-fold OOF predictions saved to disk. Mixed precision on GPU.
9. **OOS module (N4)** — Tier 1 + Tier 2, pseudo-OOS evaluation, synthetic-OOS exposure, threshold selection, reported OOS precision/recall on held-out pseudo-OOS and on the real Class 55.
10. **Ensemble + Macro-F1 bias tuning (N5)** — weight search on OOF, per-class biases with a regularization/holdout check.
11. **(Optional) Self-training / DAPT (N7)** — behind the flag.
12. **Compression (N6)** — vocab trim, distill, ONNX export, int8 quantize; verify accuracy drop (<~1 point Macro-F1 tolerated), count parameters of the final pipeline, benchmark latency (p50/p95/p99, 1000 runs, warmup excluded).
13. **Final inference** — single `predict(texts: list[str]) -> list[dict]` returning `label`, `confidence`, `is_fallback`, `view_scores`; fallback message string for class 55 ("I don't know how to do that").
14. **Submission** — write `submission.csv` with exact IDs/order/dtypes; validate with assertions (length 4,912, ids match `sample_submission.csv`, targets in 0–55, no NaNs). Save 2–3 candidate files (best CV, best ensemble, best compressed model).
15. **Demo cell** — interactive widget or simple function-call loop showing Devanagari, Roman, code-mixed, noisy, and OOS examples with confidences and latency.
16. **Results summary table** — ablation of every novelty component (N1…N6) with the CV metrics from step 6.

---

## 6. Evaluation & reporting rules for the notebook
- Primary selection metric: **test-composition-weighted Macro-F1 on (class × script)-stratified OOF**, with Roman-only and Devanagari-only numbers always shown alongside.
- Every claimed improvement needs an **ablation row** (with vs without) and, where cheap, a seed-variance estimate (≥3 seeds for the final candidates).
- Do **not** report Kaggle public-leaderboard score as evidence of quality (974 rows, ≈17/class).
- State clearly which hypotheses in §2.2 were confirmed or rejected by the notebook.

---

## 7. Risks & open questions (resolve early)
1. **Is the first-24-hour submission rule already violated?** The announcement requires one valid submission in the first 24 hours; the overview says the competition began 2 days ago. Ask organizers about shortlisting eligibility. Submit a baseline immediately regardless.
2. **Parameter counting:** per-model or total across ensemble? Count the TF-IDF model too? Get a ruling.
3. **Finalist count mismatch:** overview says top 10–12; announcement says top 15 per track.
4. **External resources:** are pretrained models, external transliteration tools, LLM-generated augmentation, and test-text usage (DAPT / pseudo-labels) allowed? Default to the *conservative* interpretation (public pretrained models only; no test-text training) until confirmed.
5. **Class-name unknown:** all labels are integers; understand domains from samples. Do not guess names in the write-up beyond what the data supports.
6. **Transliterator overfitting:** a mapper tuned too tightly to this one noise pattern may fail on the live demo's hand-typed Hinglish. Keep a small hand-made "demo robustness set" (~100 realistic Hinglish queries with common spelling variants) and evaluate on it separately.
7. **Latency:** 50 ms must hold including tokenization and the transliteration view if TTA is on. Measure the full path, not just the model forward pass.

---

## 8. Deliverables checklist
- [ ] `notebook.ipynb` (clean, runs top-to-bottom, seeds fixed, config at top)
- [ ] `submission.csv` (validated against `sample_submission.csv`)
- [ ] Final model artifacts (ONNX int8 + tokenizer + thresholds + per-class biases) and a short `predict()` demo
- [ ] Parameter-count and latency report (p50/p95/p99)
- [ ] Ablation table and per-class F1 table
- [ ] One-page methodology summary for the live defense, listing which of N1–N7 were used and what each contributed

---

## 9. Literature pointers (for the write-up; verify details before citing)
- MuRIL (Khanuja et al., 2021); IndicBERT v2 / IndicXlit / Aksharantar (AI4Bharat); HingBERT / HingRoBERTa (L3Cube; Nayak & Joshi, 2022); XLM-R (Conneau et al., 2020).
- Code-mixed data generation: Pratapa et al., 2018; Tarunesh et al., 2021.
- OOS / open-set intent detection: CLINC150 (Larson et al., 2019); Hendrycks & Gimpel, 2017 (MSP); Liu et al., 2020 (energy); Podolskiy et al., 2021 (Mahalanobis for transformers); Zhang et al., 2021 (adaptive decision boundary); Hendrycks et al., 2019 (outlier exposure).
- Imbalance: Cui et al., 2019 (class-balanced loss); Menon et al., 2021 (logit adjustment).
- Training: Gunel et al., 2021 (SupCon + CE); Wu et al., 2021 (R-Drop).
- Domain adaptation / self-training: Gururangan et al., 2020 (DAPT); Xie et al., 2020 (Noisy Student).
- Compression: Abdaoui et al., 2020 (vocabulary trimming for multilingual models); Hinton et al., 2015 (distillation).
- Label noise: Northcutt et al., 2021 (confident learning / cleanlab).

---

## 10. Instruction summary for the coding agent
1. Reproduce §2.1 numbers first; stop and report if they don't match.
2. Build the validation framework (N3) **before** any modelling.
3. Prioritize in this order: **N3 → N1/N2 → backbone fine-tuning → N4 → N5 → N6**; N7 only if confirmed allowed.
4. Keep the notebook modular (functions, not copy-paste), deterministic, and runnable on a single GPU; fall back to CPU gracefully for the TF-IDF parts.
5. Never claim an improvement without the ablation that shows it.
