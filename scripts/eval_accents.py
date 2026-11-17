"""Per-accent evaluation on L2-ARCTIC's held-out TEST speakers.

For each acoustic model given (the frozen base model and/or fine-tuned
runs/<name>/best directories), computes GOP-SF features for every phone,
fits the same 4-parameter GOP->score map on L2-ARCTIC's *train* speakers,
and reports on the *test* speakers (one per L1, never used for training):
Pearson r with the annotation (2 = correct, 0/1 = wrong) and average
precision for detecting a wrong phone, overall and per L1.

This measures the acoustic model alone (GOP-SF), not the full v2 head,
which is trained only on speechocean762.

  python scripts/eval_accents.py base runs/ft_top8/best runs/ft_l2/best
Writes results/accents.json next to each run (or prints only for 'base').
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from sklearn.metrics import average_precision_score
from transformers import Wav2Vec2ForCTC

import finetune_acoustic as fa

L2 = os.path.join(ROOT, "datasets", "l2arctic", "l2arctic_annotated.parquet")


@torch.no_grad()
def features(model, ds, cand_cols, C, device, amp_dtype, batch=4):
    F, Y, G = [], [], []
    model.eval()
    for idx in fa.length_bucketed(ds, batch, False, 0):
        b = fa.collate([ds[i] for i in idx])
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(input_values=b["x"].to(device), attention_mask=b["mask"].to(device)).logits
        frames = model._get_feat_extract_output_lengths(b["mask"].sum(1)).tolist()
        for j, it in enumerate(b["items"]):
            lp = torch.log_softmax(logits[j, :int(frames[j])].float(), -1)[:, cand_cols]
            F.append(fa.gop_sf_torch(lp, it["canon"], C).cpu())
            Y += it["scores"]
            G += [it["group"]] * len(it["scores"])
    return torch.cat(F), torch.tensor(Y), np.array(G)


def fit_map(f, y, err_weight=4.0):
    head = fa.GopToScore()
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    wt = torch.where(y < 2, torch.full_like(y, err_weight), torch.ones_like(y))
    for _ in range(500):
        loss = ((head(f) - y) ** 2 * wt).sum() / wt.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return head


def metrics(p, y):
    err = (y < 2).numpy().astype(int)
    return {"r": fa.pearson(p, y), "ap": float(average_precision_score(err, -p.numpy())),
            "n_phones": int(len(y)), "error_rate": float(err.mean())}


def main():
    targets = sys.argv[1:] or ["base"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = None
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    vocab = json.load(open(hf_hub_download(fa.BASE_MODEL, "vocab.json"), encoding="utf-8"))
    blank = vocab["<pad>"]
    cand_idx = [vocab[s] for s in fa.target_inventory(vocab)]
    cand_cols = torch.tensor([blank] + cand_idx, device=device)
    C = len(cand_idx)
    rows = pq.read_table(L2).to_pylist()
    stats = {k: 0 for k in ("sub", "del", "unresolved", "bad_index", "no_ctc", "skipped_unmapped")}
    tr = fa.SOUtterances([r for r in rows if r["split"] == "train"], vocab, cand_idx, blank, stats)
    te = fa.SOUtterances([r for r in rows if r["split"] == "test"], vocab, cand_idx, blank, stats)
    del rows
    print(f"L2-ARCTIC train utts={len(tr)} (for the 4-param map), test utts={len(te)}", flush=True)

    all_res = {}
    for t in targets:
        path = fa.BASE_MODEL if t == "base" else t
        model = Wav2Vec2ForCTC.from_pretrained(path).to(device)
        fa.fp32_frontend(model)
        f_tr, y_tr, _ = features(model, tr, cand_cols, C, device, amp_dtype)
        f_te, y_te, g_te = features(model, te, cand_cols, C, device, amp_dtype)
        head = fit_map(f_tr, y_tr)
        with torch.no_grad():
            p = head(f_te)
        res = {"overall": metrics(p, y_te)}
        for g in sorted(set(g_te)):
            m = torch.from_numpy(g_te == g)
            res[g] = metrics(p[m], y_te[m])
        all_res[t] = res
        print(f"\n== {t}")
        for k, v in res.items():
            print(f"  {k:11s} r={v['r']:.3f}  AP={v['ap']:.3f}  (phones={v['n_phones']}, wrong={v['error_rate']:.1%})",
                  flush=True)
        if t != "base":
            os.makedirs(os.path.join(os.path.dirname(t), "results"), exist_ok=True)
            json.dump(res, open(os.path.join(os.path.dirname(t), "results", "accents.json"), "w"), indent=2)
        del model
        torch.cuda.empty_cache()
    out = os.path.join(ROOT, "runs", "accents_summary.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(all_res, open(out, "w"), indent=2)


if __name__ == "__main__":
    main()
