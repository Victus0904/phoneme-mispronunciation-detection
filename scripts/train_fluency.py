"""Fluency and prosody (rhythm/intonation) scorers on speechocean762's expert
utterance scores (0-10), from cache/v2/utt_features.parquet.

Same discipline as the phone scorer: config chosen by speaker-grouped
5-fold CV on the train split only, then fit on all of train and scored
ONCE on the speaker-disjoint test split.

Usage: python scripts/train_fluency.py           (CV + final + save)
Writes models/utt_v2.joblib, cache/v2/metrics_fluency.json
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import GroupKFold

from src.utterance_v2 import EMB_KEYS, SCALARS, UttHead

V2 = os.path.join(ROOT, "cache", "v2")
TARGETS = ["fluency", "prosodic"]


def load():
    df = pd.read_parquet(os.path.join(V2, "utt_features.parquet"))
    emb = {k: np.load(os.path.join(V2, f"uttemb_{k}.npy")) for k in EMB_KEYS}
    return df, emb


def sub(emb, idx):
    return {k: v[idx] for k, v in emb.items()}


def m(y, p):
    return {"pearson": float(pearsonr(y, p).statistic), "spearman": float(spearmanr(y, p).statistic),
            "mae": float(np.mean(np.abs(y - p)))}


CONFIGS = {
    "scalars_only": dict(emb_keys=[], use_scalars=True),
    "emb_only": dict(use_scalars=False),
    "scalars+emb": dict(),
    "scalars+emb_gbm_only": dict(w_gbm=1.0),
    "scalars+emb_ridge_only": dict(w_gbm=0.0),
}


def main():
    df, emb = load()
    tr = np.nonzero((df["split"] == "train").to_numpy())[0]
    te = np.nonzero((df["split"] == "test").to_numpy())[0]
    dtr, dte = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
    etr, ete = sub(emb, tr), sub(emb, te)
    assert not set(dtr["speaker"]) & set(dte["speaker"])
    folds = list(GroupKFold(5).split(dtr, groups=dtr["speaker"]))

    print("=== train-split CV (Pearson) ===")
    cv = {}
    for name, kw in CONFIGS.items():
        row = {}
        for t in TARGETS:
            y, oof = dtr[t].to_numpy(float), np.zeros(len(dtr))
            for a, b in folds:
                oof[b] = UttHead(**kw).fit(dtr.iloc[a], sub(etr, a), y[a]).predict(dtr.iloc[b], sub(etr, b))
            row[t] = m(y, oof)["pearson"]
        cv[name] = row
        print(f"{name:24s} " + "  ".join(f"{t}={row[t]:.3f}" for t in TARGETS), flush=True)
    best = max(cv, key=lambda n: np.mean(list(cv[n].values())))
    print(f"chosen on CV: {best}")

    print("\n=== test split (scored once) ===")
    heads, results = {}, {"cv": cv, "chosen": best, "test": {}}
    for t in TARGETS:
        h = UttHead(**CONFIGS[best]).fit(dtr, etr, dtr[t].to_numpy(float))
        p = h.predict(dte, ete)
        base = m(dte[t].to_numpy(float), UttHead(**CONFIGS["scalars_only"]).fit(dtr, etr, dtr[t].to_numpy(float)).predict(dte, ete))
        r = m(dte[t].to_numpy(float), p)
        results["test"][t] = {"chosen": r, "scalars_only": base, "label_std": float(dte[t].std())}
        heads[t] = h
        print(f"{t:9s} chosen: r={r['pearson']:.3f} rho={r['spearman']:.3f} MAE={r['mae']:.2f}/10   "
              f"(timing/pitch scalars only: r={base['pearson']:.3f})")
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    joblib.dump({"heads": heads, "targets": TARGETS, "emb_keys": EMB_KEYS, "scalars": SCALARS,
                 "metrics": {t: results["test"][t]["chosen"] for t in TARGETS}},
                os.path.join(ROOT, "models", "utt_v2.joblib"))
    with open(os.path.join(V2, "metrics_fluency.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("saved models/utt_v2.joblib")


if __name__ == "__main__":
    main()
