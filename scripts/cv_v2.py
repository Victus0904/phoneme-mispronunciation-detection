"""Model selection for the v2 head: speaker-grouped 5-fold CV on the TRAIN
split only (the test split is never touched here -- see train_v2.py).

Usage: python scripts/cv_v2.py [config ...]   (default: all configs)
Writes out-of-fold predictions to cache/v2/oof_<config>.npy.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

from src.head_v2 import CAT_FEATURES, SCALAR_FEATURES, V1_CAT, V1_NUMERIC, GBMHead, SeqHead

V2 = os.path.join(ROOT, "cache", "v2")
LAYERS = (8, 16, 24, "sfvec", "postvec")


def load(split=None):
    df = pd.read_parquet(os.path.join(V2, "features.parquet"))
    emb = {L: np.load(os.path.join(V2, f"emb_{'h' if isinstance(L, int) else ''}{L}.npy"), mmap_mode="r")
           for L in LAYERS}
    if split:
        idx = np.nonzero((df["split"] == split).to_numpy())[0]
        df = df.iloc[idx].reset_index(drop=True)
        emb = {L: np.asarray(e[idx]) for L, e in emb.items()}
    return df, emb


def metrics(y, p):
    return {"pearson": float(pearsonr(y, p).statistic), "spearman": float(spearmanr(y, p).statistic),
            "ap": float(average_precision_score((y < 2.0).astype(int), -p)),
            "mse": float(np.mean((y - p) ** 2))}


def sub(emb, idx):
    return {L: e[idx] for L, e in emb.items()}


CONFIGS = {
    "v1_gbm": lambda: GBMHead(scalar=V1_NUMERIC, cats=V1_CAT),
    "gbm": lambda: GBMHead(),
    "gbm_pca16": lambda: GBMHead(emb_layers=(16,)),
    "gbm_pca24": lambda: GBMHead(emb_layers=(24,)),
    "seq_noemb": lambda: SeqHead(layers_used=()),
    "seq_h8": lambda: SeqHead(layers_used=(8,)),
    "seq_h16": lambda: SeqHead(layers_used=(16,)),
    "seq_h24": lambda: SeqHead(layers_used=(24,)),
    "seq_h16_24": lambda: SeqHead(layers_used=(16, 24)),
    "seq_h8_16_24": lambda: SeqHead(layers_used=(8, 16, 24)),
    "seq_vec": lambda: SeqHead(layers_used=("sfvec", "postvec")),
    "seq_vec_h16": lambda: SeqHead(layers_used=("sfvec", "postvec", 16)),
    "gbm_vec": lambda: GBMHead(emb_layers=("sfvec", "postvec")),
    # final recipe (selected on fold 0, then confirmed by full 5-fold CV)
    "gbm_vec_h16": lambda: GBMHead(emb_layers=("sfvec", "postvec", 16), n_pca=48),
    "seq_vec_h16_s0": lambda: SeqHead(layers_used=("sfvec", "postvec", 16), epochs=20, lr=3e-3, seed=0),
    "seq_vec_h16_s1": lambda: SeqHead(layers_used=("sfvec", "postvec", 16), epochs=20, lr=3e-3, seed=1),
}


def run(name, df, emb, folds):
    y = df["human_score"].to_numpy()
    oof = np.zeros(len(df))
    t0 = time.time()
    for tr, va in folds:
        m = CONFIGS[name]().fit(df.iloc[tr].reset_index(drop=True), sub(emb, tr))
        oof[va] = m.predict(df.iloc[va].reset_index(drop=True), sub(emb, va))
    np.save(os.path.join(V2, f"oof_{name}.npy"), oof)
    r = metrics(y, oof)
    print(f"{name:14s} pearson={r['pearson']:.4f} spearman={r['spearman']:.4f} ap={r['ap']:.4f} "
          f"mse={r['mse']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    return oof


def main():
    names = sys.argv[1:] or list(CONFIGS)
    df, emb = load("train")
    print(f"train: {len(df)} rows, {df['uid'].nunique()} utts, {df['speaker'].nunique()} speakers", flush=True)
    folds = list(GroupKFold(n_splits=5).split(df, groups=df["speaker"]))
    for n in names:
        run(n, df, emb, folds)


if __name__ == "__main__":
    main()
