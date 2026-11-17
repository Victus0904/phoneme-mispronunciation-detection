"""Fine-tuning report: training curves, per-accent chart and a model card.

Reads what a fine-tuning run already produced and writes nothing new to
the test splits:
  runs/<name>/log.json          per-epoch validation r (+ frozen baseline)
  runs/<name>/metrics_v2.json   one-time speechocean762 test scoring (rebuild)
  runs/<name>/strictness.json   strictness levels on test speakers (rebuild)
  runs/accents_summary.json     L2-ARCTIC held-out test speakers (eval_accents)

Writes:
  figures/finetune_curves.png   validation r per epoch vs the frozen model, train losses
  figures/finetune_accents.png  per-first-language r, frozen vs fine-tuned
  runs/<name>/model_card.json   the numbers the website shows ("How accurate is this?")

  python scripts/finetune_report.py --name ft_l2
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Frozen model, L2-ARCTIC test speakers, GOP-SF only. Measured with
# scripts/eval_accents.py on 2026-11-17 (runs/accents_summary.json only keeps
# the last model evaluated, so the baseline is recorded here; see docs/FINETUNING.md).
FROZEN_ACCENTS = {"overall": (0.286, 0.323), "Arabic": 0.198, "Hindi": 0.149, "Korean": 0.221,
                  "Mandarin": 0.216, "Spanish": 0.202, "Vietnamese": 0.474}
# Frozen v2 pipeline on the speechocean762 test split (models/head_v2.joblib, REPORT_v2.md)
FROZEN_SO = {"pearson": 0.753, "ap": 0.642, "recall": 0.40, "precision": 0.69}


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="ft_l2")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    run = os.path.join(ROOT, "runs", a.name)
    log = load(os.path.join(run, "log.json"))
    mv2 = load(os.path.join(run, "metrics_v2.json"))
    strict = load(os.path.join(run, "strictness.json"))
    acc_all = load(os.path.join(ROOT, "runs", "accents_summary.json"))
    acc = acc_all.get(f"runs/{a.name}/best")
    if acc is None:
        raise SystemExit(f"runs/accents_summary.json has no entry for {a.name}; run eval_accents first")

    shipped = mv2["table"]["v2 ensemble, full train (shipped)"]["full"]
    std = strict["levels"][strict.get("default", "standard")]["test"]
    epochs = log["epochs"]
    best = max(epochs, key=lambda e: e["r"])
    card = {
        "name": a.name,
        "label": "Multi-accent fine-tuned model",
        "base_model": "facebook/wav2vec2-lv-60-espeak-cv-ft",
        "training_data": ["speechocean762 (Mandarin-L1 learners)",
                          "L2-ARCTIC v5.0 (Arabic, Mandarin, Hindi, Korean, Spanish, Vietnamese L1)"],
        "license": "CC BY-NC 4.0 (non-commercial, inherited from L2-ARCTIC)",
        "trained_layers": f"top {log['args']['unfreeze']} of 24 transformer layers",
        "epochs": len(epochs), "best_epoch": best["epoch"] + 1,
        "train_minutes": round(sum(e["minutes"] for e in epochs), 1),
        "speechocean_test": {
            "frozen": FROZEN_SO,
            "finetuned": {"pearson": round(shipped["pearson"], 3), "ap": round(shipped["ap"], 3),
                          "recall": round(std["recall"], 3), "precision": round(std["precision"], 3)},
        },
        "accents_test": {
            "note": "One held-out speaker per first language, GOP-SF feature only (the scoring head is trained on speechocean762).",
            "frozen": {"r": FROZEN_ACCENTS["overall"][0], "ap": FROZEN_ACCENTS["overall"][1]},
            "finetuned": {"r": round(acc["overall"]["r"], 3), "ap": round(acc["overall"]["ap"], 3)},
            "by_l1": {k: {"frozen": FROZEN_ACCENTS[k], "finetuned": round(v["r"], 3)}
                      for k, v in acc.items() if k != "overall"},
        },
    }
    out = os.path.join(run, "model_card.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    print("wrote", out)
    print(json.dumps({"speechocean_test": card["speechocean_test"], "accents": card["accents_test"]["finetuned"]}, indent=1))
    if a.no_figures:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = os.path.join(ROOT, "figures")
    os.makedirs(figdir, exist_ok=True)
    x = [e["epoch"] + 1 for e in epochs]
    base = log["baseline_frozen"]
    sets = [k for k in ("speechocean", "l2arctic") if k in epochs[0]]
    names = {"speechocean": "speechocean762 val", "l2arctic": "L2-ARCTIC val"}
    colors = {"speechocean": "#2a6fdb", "l2arctic": "#d9822b"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for k in sets:
        ax1.plot(x, [e[k]["r"] for e in epochs], "o-", color=colors[k], label=f"{names[k]} (fine-tuned)")
        ax1.axhline(base[k]["r"], color=colors[k], ls="--", lw=1, label=f"{names[k]} (frozen)")
    ax1.axvline(best["epoch"] + 1, color="#888", ls=":", lw=1)
    ax1.set(xlabel="epoch", ylabel="GOP-SF r with human phone scores",
            title="Validation speakers (checkpoint picked on mean r)")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(alpha=.3)
    ax2.plot(x, [e["train_ctc"] for e in epochs], "o-", label="CTC loss")
    ax2.plot(x, [e["train_gop"] for e in epochs], "o-", label="GOP-SF score loss")
    ax2.set(xlabel="epoch", ylabel="mean training loss", title="Training losses")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "finetune_curves.png"), dpi=130)
    plt.close(fig)

    l1s = list(card["accents_test"]["by_l1"])
    fr = [card["accents_test"]["by_l1"][k]["frozen"] for k in l1s]
    ft = [card["accents_test"]["by_l1"][k]["finetuned"] for k in l1s]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    xs = range(len(l1s))
    ax.bar([i - .2 for i in xs], fr, .4, label="frozen", color="#9aa5b1")
    ax.bar([i + .2 for i in xs], ft, .4, label=f"fine-tuned ({a.name})", color="#2a6fdb")
    for i, (u, v) in enumerate(zip(fr, ft)):
        ax.text(i + .2, v + .01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(list(xs), l1s)
    ax.set(ylabel="GOP-SF r", title="Held-out L2-ARCTIC test speakers, by first language")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "finetune_accents.png"), dpi=130)
    plt.close(fig)
    print("wrote figures/finetune_curves.png, figures/finetune_accents.png")


if __name__ == "__main__":
    main()
