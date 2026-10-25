"""Strictness levels for the app: flag thresholds chosen on train out-of-fold
predictions only (never test), then each level evaluated once on the
speaker-disjoint test split with the shipped head.

  relaxed   -- only flag what we're quite sure about (OOF precision >= 0.80)
  standard  -- the v2 default (OOF precision >= 0.70)
  strict    -- catch most real errors (OOF recall >= 0.70), more false alarms
  very_strict -- also counts accent-level deviations (OOF recall >= 0.85)

Usage: python scripts/calibrate_strictness.py
Writes models/strictness.json (read by src/scorer.py).
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
from sklearn.metrics import precision_recall_curve

from cv_v2 import V2, load, sub
from train_v2 import FINAL_MEMBERS, FINAL_WEIGHTS, at_threshold

LEVELS = [("relaxed", "precision", 0.80), ("standard", "precision", 0.70),
          ("strict", "recall", 0.70), ("very_strict", "recall", 0.85)]


def threshold_for(y, pred, kind, target):
    """flag = pred < threshold. precision: max recall with precision >= target;
    recall: the highest-precision threshold that still reaches recall >= target."""
    prec, rec, thr = precision_recall_curve((y < 2.0).astype(int), -pred)
    prec, rec = prec[:-1], rec[:-1]
    ok = np.nonzero(prec >= target if kind == "precision" else rec >= target)[0]
    i = ok[np.argmax(rec[ok])] if kind == "precision" else ok[np.argmax(prec[ok])]
    return float(-thr[i]) + 1e-9


def main():
    df, emb = load()
    tr = np.nonzero((df["split"] == "train").to_numpy())[0]
    te = np.nonzero((df["split"] == "test").to_numpy())[0]
    ytr = df["human_score"].to_numpy()[tr]
    yte = df["human_score"].to_numpy()[te]
    oof = np.average([np.load(os.path.join(V2, f"oof_{n}.npy")) for n, _ in FINAL_MEMBERS], axis=0,
                     weights=FINAL_WEIGHTS)
    head = joblib.load(os.path.join(ROOT, "models", "head_v2.joblib"))["head"]
    pred = head.predict(df.iloc[te].reset_index(drop=True), sub(emb, te))

    out = {"levels": {}, "default": "standard", "error_base_rate_test": float((yte < 2.0).mean())}
    print(f"test: {(yte < 2.0).mean():.1%} of phones rated below 2 by experts")
    for name, kind, target in LEVELS:
        t = threshold_for(ytr, oof, kind, target)
        o, s = at_threshold(ytr, oof, t), at_threshold(yte, pred, t)
        out["levels"][name] = {"threshold": t, "rule": f"train OOF {kind} >= {target}",
                               "train_oof": o, "test": s}
        print(f"{name:12s} t={t:.3f}  OOF p={o['precision']:.3f} r={o['recall']:.3f}  |  "
              f"TEST p={s['precision']:.3f} r={s['recall']:.3f} flags {s['flag_rate']:.1%}")
    with open(os.path.join(ROOT, "models", "strictness.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
