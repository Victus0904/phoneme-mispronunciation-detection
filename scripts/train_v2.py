"""Final v2 head: train on the full train split, evaluate ONCE on the test
split (speaker-disjoint), pick the flag threshold from train out-of-fold
predictions (never from test), save the bundle for the local app, and
write metrics + figures.

Usage: python scripts/train_v2.py
Reads the config chosen by cv_v2.py (FINAL below), writes
  models/head_v2.joblib, cache/v2/metrics_v2.json, figures/v2_*.png
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
from sklearn.metrics import precision_recall_curve

from cv_v2 import CONFIGS, V2, load, metrics, sub
from src.head_v2 import EnsembleHead

FIGURES = os.path.join(ROOT, "figures")
MODELS = os.path.join(ROOT, "models")

# chosen from cv_v2.py results (train-split CV only); see REPORT_v2.md
FINAL_MEMBERS = [("gbm_vec_h16", None), ("seq_vec_h16_s0", None), ("seq_vec_h16_s1", None)]
FINAL_WEIGHTS = [0.5, 0.25, 0.25]
FLAG_MIN_PRECISION = 0.70  # flag threshold: max recall s.t. OOF precision >= this


def v1_subset_uids(split, n=500, n_total=2500, seed=42):
    """The exact 500-utterance subsample v1 used (src/features._select_utterance_indices)."""
    rng = np.random.RandomState(seed)
    return {f"{split}_{i}" for i in sorted(rng.choice(n_total, size=n, replace=False).tolist())}


def make(name, seed):
    m = CONFIGS[name]()
    if seed is not None:
        m.seed = seed
    return m


def fit_ensemble(df, emb, members=FINAL_MEMBERS, weights=FINAL_WEIGHTS, log=print):
    fitted = []
    for name, seed in members:
        t0 = time.time()
        fitted.append(make(name, seed).fit(df, emb))
        log(f"  fit {name} (seed={seed}) in {time.time()-t0:.0f}s")
    return EnsembleHead(fitted, weights)


def pick_threshold(y, pred, min_precision=FLAG_MIN_PRECISION):
    """Largest predicted-score threshold region giving max recall with
    precision >= min_precision (flag = pred < threshold)."""
    prec, rec, thr = precision_recall_curve((y < 2.0).astype(int), -pred)
    ok = np.nonzero(prec[:-1] >= min_precision)[0]
    if len(ok) == 0:
        return None
    i = ok[np.argmax(rec[ok])]
    return float(-thr[i]) + 1e-9, float(prec[i]), float(rec[i])


def at_threshold(y, pred, t):
    flag = pred < t
    err = y < 2.0
    tp = int((flag & err).sum())
    return {"precision": tp / max(1, flag.sum()), "recall": tp / max(1, err.sum()), "flag_rate": float(flag.mean())}


def main():
    os.makedirs(MODELS, exist_ok=True)
    df, emb = load()
    tr = np.nonzero((df["split"] == "train").to_numpy())[0]
    te = np.nonzero((df["split"] == "test").to_numpy())[0]
    dtr, dte = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
    etr, ete = sub(emb, tr), sub(emb, te)
    assert not set(dtr["speaker"]) & set(dte["speaker"]), "speaker leakage"
    yte = dte["human_score"].to_numpy()
    sub_mask = dte["uid"].isin(v1_subset_uids("test")).to_numpy()
    results = {"n_train_rows": len(dtr), "n_test_rows": len(dte), "n_test_subset_rows": int(sub_mask.sum())}

    # flag threshold from train OOF predictions of the same ensemble recipe
    oof = np.average([np.load(os.path.join(V2, f"oof_{n}.npy")) for n, _ in FINAL_MEMBERS], axis=0,
                     weights=FINAL_WEIGHTS)
    t, p_oof, r_oof = pick_threshold(dtr["human_score"].to_numpy(), oof)
    print(f"flag threshold from train OOF: {t:.3f} (OOF precision {p_oof:.3f}, recall {r_oof:.3f})")

    print("=== fitting final ensemble on full train split ===")
    head = fit_ensemble(dtr, etr)
    t0 = time.time()
    pred = head.predict(dte, ete)
    print(f"test inference (head only): {time.time()-t0:.1f}s for {len(dte)} phones")

    # baselines on the same test rows
    v1 = CONFIGS["v1_gbm"]().fit(dtr, etr).predict(dte, ete)
    v1_sub_train = dtr["uid"].isin(v1_subset_uids("train")).to_numpy()
    v1_capped = CONFIGS["v1_gbm"]().fit(dtr[v1_sub_train].reset_index(drop=True)).predict(dte, ete)
    v2_capped = fit_ensemble(dtr[v1_sub_train].reset_index(drop=True), sub(etr, np.nonzero(v1_sub_train)[0]),
                             log=lambda *_: None).predict(dte, ete)
    naive = dte["mean_log_posterior"].to_numpy()
    naive_sf = dte["gop_sf"].to_numpy()

    rows = [("naive: mean log-posterior (v1 baseline)", naive), ("naive: GOP-SF alone", naive_sf),
            ("v1 head, v1 features, 500-utt train (v1 as shipped)", v1_capped),
            ("v1 head, v1 features, full train", v1),
            ("v2 ensemble, 500-utt train", v2_capped),
            ("v2 ensemble, full train (shipped)", pred)]
    results["table"] = {}
    print("\n=== test results ===")
    print(f"{'model':55s} {'subset r':>9s} {'subset AP':>9s} {'full r':>8s} {'full rho':>8s} {'full AP':>8s}")
    for name, p in rows:
        ms, mf = metrics(yte[sub_mask], p[sub_mask]), metrics(yte, p)
        results["table"][name] = {"subset": ms, "full": mf}
        print(f"{name:55s} {ms['pearson']:9.3f} {ms['ap']:9.3f} {mf['pearson']:8.3f} {mf['spearman']:8.3f} {mf['ap']:8.3f}")

    op = at_threshold(yte, pred, t)
    print(f"\nshipped flag threshold {t:.3f}: test precision {op['precision']:.3f}, recall {op['recall']:.3f}, "
          f"flags {op['flag_rate']:.1%} of phones")
    prec, rec, _ = precision_recall_curve((yte < 2.0).astype(int), -pred)
    p10 = float(prec[:-1][np.argmin(np.abs(rec[:-1] - 0.10))])
    p50 = float(prec[:-1][np.argmin(np.abs(rec[:-1] - 0.50))])
    print(f"precision at recall 0.10: {p10:.3f}   at recall 0.50: {p50:.3f}")
    results.update(flag_threshold=t, flag_op_test=op, flag_op_train_oof={"precision": p_oof, "recall": r_oof},
                   precision_at_recall_0_10=p10, precision_at_recall_0_50=p50,
                   error_base_rate=float((yte < 2.0).mean()))

    # per-phone breakdown
    per = []
    for ph, g in dte.assign(pred=pred).groupby("arpa_base"):
        if len(g) >= 60 and g["human_score"].std() > 0:
            per.append((ph, len(g), metrics(g["human_score"].to_numpy(), g["pred"].to_numpy())["pearson"]))
    results["per_phone_pearson"] = {p: {"n": n, "r": r} for p, n, r in sorted(per, key=lambda x: x[2])}

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5.5, 5))
    for name, p in [("v1 (as shipped)", v1_capped), ("v2 (shipped)", pred)]:
        pr, rc, _ = precision_recall_curve((yte < 2.0).astype(int), -p)
        ap = metrics(yte, p)["ap"]
        plt.plot(rc, pr, label=f"{name}  AP={ap:.3f}")
    plt.scatter([op["recall"]], [op["precision"]], color="black", zorder=5, label="v2 flag threshold")
    plt.axhline((yte < 2.0).mean(), color="gray", ls=":", label="random")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Phone error detection, full test set")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(FIGURES, "v2_pr_curve.png"), dpi=150); plt.close()

    plt.figure(figsize=(5.5, 5))
    jit = np.random.RandomState(0).uniform(-0.05, 0.05, len(yte))
    plt.scatter(yte + jit, pred, s=4, alpha=0.08)
    plt.xlabel("Human phone score (0-2, jittered)"); plt.ylabel("v2 predicted score")
    plt.title(f"v2 vs human, full test set (r={results['table'][rows[-1][0]]['full']['pearson']:.3f})")
    plt.tight_layout(); plt.savefig(os.path.join(FIGURES, "v2_correlation_scatter.png"), dpi=150); plt.close()

    layers = sorted({L for m in head.members for L in getattr(m, "layers_used", ()) + getattr(m, "emb_layers", ())},
                    key=str)
    full = results["table"][rows[-1][0]]["full"]
    bundle = {
        "head": head, "layers": tuple(layers), "flag_threshold": t, "version": "v2",
        "description": "GBM + 2-seed GOPT-style transformer ensemble on frozen wav2vec2 GOP/GOP-SF features, "
                       "GOPT-style LLR/posterior vectors and layer-16 embeddings, trained on all 2,500 "
                       "speechocean762 train utterances",
        "metrics": {"pearson": full["pearson"], "spearman": full["spearman"], "ap": full["ap"],
                    "flag_precision": op["precision"], "flag_recall": op["recall"]},
    }
    joblib.dump(bundle, os.path.join(MODELS, "head_v2.joblib"))
    with open(os.path.join(V2, "metrics_v2.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved models/head_v2.joblib (layers={layers}) and cache/v2/metrics_v2.json")


if __name__ == "__main__":
    main()
