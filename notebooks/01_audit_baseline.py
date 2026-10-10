"""Stage 1-4: Audit + EDA + Mean baseline (CPU, laptop-safe). Run: python 01_audit_baseline.py"""
import os, wave, contextlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

BASE = r"C:\Users\DELL\Desktop\SHL Project\data"
OUT = r"C:\Users\DELL\Desktop\SHL Project\notebooks"
train = pd.read_csv(os.path.join(BASE, "train.csv"))
test = pd.read_csv(os.path.join(BASE, "test.csv"))

print("Train:", train.shape, "Test:", test.shape)
print(train["label"].value_counts().sort_index())
print("Mean:", train["label"].mean(), "Std:", train["label"].std())

# 1. label hist
plt.figure(); train["label"].hist(bins=20); plt.title("Train label distribution (n=769)"); plt.xlabel("label"); plt.ylabel("count")
plt.savefig(os.path.join(OUT, "label_hist.png")); print("saved label_hist.png")

# 2. durations (wave stdlib, no librosa)
def durations(folder):
    d = os.path.join(BASE, folder); out = {}
    for f in os.listdir(d):
        if not f.endswith(".wav"): continue
        p = os.path.join(d, f)
        with contextlib.closing(wave.open(p)) as w:
            out[f] = w.getnframes() / w.getframerate()
    return out
dtr = durations("train"); dte = durations("test")
tr_map = train.set_index("filename")["label"].to_dict()
vals = [(f, dtr[f], tr_map.get(f)) for f in dtr if f in tr_map]
df = pd.DataFrame(vals, columns=["file", "dur", "label"])
print(f"train dur mean {df['dur'].mean():.1f} min {df['dur'].min():.1f} max {df['dur'].max():.1f}")
print(f"test dur mean {np.mean(list(dte.values())):.1f} min {min(dte.values()):.1f} max {max(dte.values()):.1f}")

plt.figure(); plt.hist(df["dur"], bins=20); plt.title("Train durations"); plt.xlabel("sec")
plt.savefig(os.path.join(OUT, "train_dur.png"))
plt.figure(); plt.scatter(df["dur"], df["label"], alpha=0.4); plt.title("Score vs duration"); plt.xlabel("dur"); plt.ylabel("label")
plt.savefig(os.path.join(OUT, "score_vs_dur.png"))
print("saved train_dur.png score_vs_dur.png")
print("Pearson(dur,label):", pearsonr(df["dur"], df["label"]))

# 3. Mean baseline 5-fold CV
y = train["label"].values
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rms, prs = [], []
for tr, va in kf.split(y):
    pred = np.full(len(va), y[tr].mean())
    rms.append(mean_squared_error(y[va], pred) ** 0.5)
    try: prs.append(pearsonr(y[va], pred)[0])
    except: prs.append(np.nan)
print(f"Mean baseline CV RMSE {np.mean(rms):.4f} +- {np.std(rms):.4f} | Pearson {np.nanmean(prs)} (expected ~0/NaN for constant)")
print("Train RMSE (fit on full train, predict mean):", (mean_squared_error(y, np.full_like(y, y.mean())) ** 0.5))
