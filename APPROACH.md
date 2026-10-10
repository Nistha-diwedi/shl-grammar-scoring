# SHL Grammar Scoring Engine — Approach Overview

## Problem
Map 45–60s spoken English audio to continuous grammar score 0–5.
Framed as supervised regression: `Audio → Speech → Language → Grammar → Score`, not generic audio classification, because rubric targets syntax, structure, accuracy, complexity, completeness, self-correction.

## Data Audit
- Train 769 labeled, Test 216 (labels -1 placeholder), all 16kHz mono.
- Labels include 37×0.0 despite 1–5 rubric; imbalance 3.0=174, 5.0=133, 4.0=124 vs 1.0=1; mean 3.31.
- Durations break spec: train 20–61 mean 55.8, test 6.5–61 mean 48.8; duration-label Pearson 0.087 weak.
- `sample_submission.csv` stale (204 rows) → submit 216 rows from `test.csv` format `filename,label`.
- Same filenames in train/test are different content (different hashes) → resolve by split folder.

## EDA
- Label histogram (imbalance to 3–5), duration histogram, score-vs-duration scatter, score-vs-energy, train-vs-test shift check.
- Used to choose stratified-binned CV and to distrust duration as grammar signal.

## Validation
- Stratified-binned 5-fold CV, SEED 42. Report RMSE + Pearson per fold, mean±std.
- Public 60% / private 40% → trust CV over public, submit sparingly.

## Baselines
- Mean 3.31: CV RMSE 1.2397, Pearson NaN. Minimum bar.

## Acoustic Features
- librosa 5: duration, RMS energy, zero-crossing rate, spectral centroid, MFCC-13 mean.
- Why: prosody/fluency/pauses correlate with scores, cheap, works on 769 samples.
- XGB: CV 0.9336 / 0.6617. Proved sound helps.

## Transcription
- Whisper `base` then `small` for cleaner text, transcripts cached once, manual QA on 20 samples across 1–5.
- Why: grammar lives in words; pretrained ASR allowed, external data not.

## Linguistic Features
- Regex counts: nwords, nsentences, avg length, type-token ratio; sentence completeness, filler/repeat rate.
- Why: direct map to rubric sentence-structure, complexity, vocabulary, incomplete/self-correct.
- Fusion acoustic+linguistic XGB: CV 0.8103 / 0.7573, Train 0.0959. First submit 0.6225 public.

## Tried and Dropped (Why)
- MiniLM 384-d embeddings + Ridge: CV 0.98 / 0.62. Captures topic, not grammar.
- TF-IDF 2000 + LightGBM: 1.04 / 0.53. Sparse overfits 769.
- CoLA RoBERTa grammaticality: Pearson 0.29. Dragged by ASR noise/disfluencies.
- LLM judges (Gemini, Groq gpt-oss, local Qwen2.5-3B): all ~3.0 mode-collapse / quota 20/day / 200k TPD / empty reasoning. Dropped for quota and no CV gain.

## Final Model — What Used and Why
- **Wav2Vec2-base frozen embeddings (768-d)** mean-pooled: captures pronunciation, fluency, prosody without transcript noise. Alone RidgeCV: CV 0.6809 / 0.8363.
- **Acoustic 5 + linguistic-small 4**: light rubric signals, complement audio embedding.
- **Stack RidgeCV + CatBoost (w=0.5/0.5)**: Ridge generalizes linear audio grammar, CatBoost captures non-linear interactions; OOF blend CV 0.6181 / 0.8699 beats singles (0.644/0.650).
- Why this wins: audio embedding carries most signal, handcrafted adds structure, stacking reduces variance on 769 samples. No fine-tuning to avoid overfit.
- Train RMSE 0.2165 (compulsory), CV 0.6181, Public 0.5447.

## Results
| Model | CV RMSE | CV Pearson | Public |
|---|---|---|---|
| Mean | 1.2397 | — | — |
| Acoustic XGB | 0.9336 | 0.6617 | — |
| Fusion | 0.8103 | 0.7573 | 0.6225 |
| W2V Ridge | 0.6809 | 0.8363 | 0.5800 |
| Full W2V+Ac+Ling | 0.6439 | 0.8547 | 0.5534 |
| Stack Ridge+Cat | 0.6181 | 0.8699 | 0.5447 |

## Submission
- `submission.csv` 216 rows `filename,label`, clipped 0–5.
- Notebook re-runs top-to-bottom with plots: label hist, predicted-vs-actual, residual hist, worst-10 table.
