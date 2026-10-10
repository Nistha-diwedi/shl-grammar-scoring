# SHL Grammar Scoring Engine

Spoken English audio (45–60s) → continuous grammar score 0–5.

## Approach
`Audio → Speech → Language → Grammar → Score`. Rubric targets syntax, structure, accuracy, complexity, completeness, self-correction, so text matters more than raw sound.

- **EDA:** label imbalance (3.0/5.0 heavy, 37×0.0), duration shift train/test, weak duration-label link.
- **Validation:** stratified-binned 5-fold CV, RMSE + Pearson. Public 60% / private 40% — trust CV.
- **Acoustic:** librosa duration, RMS, ZCR, centroid, MFCC mean + XGBoost.
- **ASR:** Whisper base → small, transcripts cached, manual QA.
- **Linguistic:** word/sentence counts, avg length, type-token ratio.
- **Audio embeddings:** frozen wav2vec2-base mean-pooled 768-d + RidgeCV (no fine-tune on 769 samples).
- **Final:** RidgeCV + CatBoost stack w=0.5/0.5 on w2v + acoustic + ling-small.

Dropped: MiniLM (topic not grammar), TF-IDF (sparse overfit), CoLA (ASR noise), LLM judges (quota/mode-collapse).

## Results
| Model | CV RMSE | Pearson | Public |
|---|---|---|---|
| Mean | 1.2397 | — | — |
| Acoustic | 0.9336 | 0.66 | — |
| Fusion | 0.8103 | 0.76 | 0.6225 |
| W2V | 0.6809 | 0.84 | 0.5800 |
| Full | 0.6439 | 0.85 | 0.5534 |
| Stack | 0.6181 | 0.87 | **0.5447** |

Train RMSE 0.2165. Submission 216 rows, clipped 0–5.

## Repo
```
data/train.csv, test.csv (+ audio via Kaggle shl-hiring-assessment-2026)
features/audio_features.csv, ling_*.csv, transcripts_*.csv (+ w2v *.npy local-only)
models/fusion.json
notebooks/grammar_scoring.ipynb
submissions/submission.csv
requirements.txt
```

## Reproduce
1. `pip install -r requirements.txt`
2. Data from Kaggle `shl-hiring-assessment-2026` → `data/`
3. Run `notebooks/grammar_scoring.ipynb` top-to-bottom (Kaggle GPU T4 for wav2vec2/Whisper-small).
4. Output `submissions/submission.csv`.
