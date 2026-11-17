"""Fine-tune the acoustic model (wav2vec2-lv-60-espeak-cv-ft) for scoring.

Everything downstream (GOP-SF, the GBM + transformer head) is computed from
this model's phone posteriors, so the fine-tune optimises exactly that:

  loss = ctc_weight * CTC(what the speaker actually said)
       + gop_weight * weighted MSE(score_from_gop_sf, human phone score)

- CTC target: the canonical phones with speechocean762's annotated
  mispronunciations applied (substitutions swapped in, deletions removed).
  Utterances with an error we can't resolve to a phone (e.g. "<unk>") get
  no CTC term, only the GOP term. This keeps the posteriors a calibrated
  phone recognizer instead of drifting to whatever minimises the score loss.
- GOP term: GOP-SF exactly as in src/features_v2.py (canonical sequence vs.
  every single-phone substitution / deletion, scored by full CTC likelihood),
  but computed in torch so it is differentiable. A 4-parameter map turns
  (gop_sf, sub_llr, del_llr) into a 0-2 score. Phones the raters marked
  wrong are upweighted, since 81% of phones score 2.0.

Evaluation discipline (same as REPORT_v2.md): the test split is never read
here. Model selection uses a speaker-disjoint validation set carved out of
the train split (--val-frac of train speakers, fixed seed). The test split is
scored once, later, by scripts/train_v2.py on re-extracted features.

Only the top --unfreeze transformer layers (plus the final layer norm and
CTC head) are trained; the CNN front end and lower layers stay frozen. With
the default 8, layers 16-23 train, so the layer-8/16 hidden states the head
uses are unchanged and only the posteriors / layer 24 move.

Runs on any CUDA GPU (bf16 where supported, else fp16) and, slowly, on CPU.

  # smoke test (a few minutes on a GTX 1650)
  python scripts/finetune_acoustic.py --limit 120 --epochs 1 --out runs/smoke
  # real run (Lightning AI / any 16 GB+ GPU)
  python scripts/finetune_acoustic.py --batch 8 --out runs/ft_top8

Output dir: best/ (a HF model dir usable via ACOUSTIC_MODEL=<out>/best),
last.pt (resume state; rerun the same command to resume), log.json.
"""
import argparse
import io
import json
import math
import os
import random
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import Wav2Vec2ForCTC

from src.acoustic import trim_silence
from src.features_v2 import build_targets_with_words, target_inventory
from src.phone_mapping import ARPABET_TO_IPA, map_word_phones_to_ipa_targets

BASE_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
TRAIN_PARQUET = os.path.join(ROOT, "datasets", "speechocean762", "data", "train-00000-of-00001.parquet")
SR = 16000
PROCESSOR_FILES = ("preprocessor_config.json", "special_tokens_map.json", "tokenizer_config.json", "vocab.json")
SEED = 42


# ---------------------------------------------------------------- data

def ensure_dataset():
    if not os.path.exists(TRAIN_PARQUET):
        print("downloading speechocean762 train split ...", flush=True)
        hf_hub_download("mispeech/speechocean762", "data/train-00000-of-00001.parquet",
                        repo_type="dataset", local_dir=os.path.join(ROOT, "datasets", "speechocean762"))


def _resolve_arpa(p, canonical):
    """One annotated pronounced phone -> an ARPAbet key we can map, or None.
    Annotations carry no stress digit ('AH'), so a vowel takes the canonical
    phone's stress; 'X*' (pronounced as X but distorted) is treated as X."""
    p = p.strip().upper().rstrip("*")
    if not p:
        return None
    digit = canonical[-1] if canonical and canonical[-1].isdigit() else "1"
    for cand in (p, p + digit, p + "1", p + "0"):
        if cand in ARPABET_TO_IPA:
            return cand
    return None


def pronounced_ipa(words, stats):
    """IPA sequence of what was actually said, or None if any annotated
    error can't be resolved to a phone ('<unk>').

    speechocean762 'pronounced-phone' values seen in train: an ARPAbet phone
    without stress ('AH'), the same with '*' (distorted), several phones
    ('IH R', an insertion), '<DEL>' (deleted) and '<unk>' (unidentifiable)."""
    seq = []
    for w in words:
        slots = [[p] for p in w["phones"]]
        for m in (w.get("mispronunciations") or []):
            i = int(m["index"])
            said = str(m.get("pronounced-phone", "")).strip()
            if i < 0 or i >= len(slots):
                stats["bad_index"] += 1
                return None
            if said.upper() == "<DEL>":
                slots[i] = []
                stats["del"] += 1
                continue
            canon = w["phones"][i]
            arpa = [_resolve_arpa(p, canon) for p in said.split()]
            if not arpa or any(a is None for a in arpa):
                stats["unresolved"] += 1
                return None
            slots[i] = arpa
            stats["sub"] += 1
        phones = [p for s in slots for p in s]
        seq += [t["ipa"] for t in map_word_phones_to_ipa_targets(phones)]
    return seq


def load_rows(limit=None):
    ensure_dataset()
    rows = pq.read_table(TRAIN_PARQUET).to_pylist()
    rng = random.Random(SEED)
    if limit:
        rng.shuffle(rows)
        rows = rows[:limit]
    return rows


def speaker_split(rows, val_frac):
    spk = sorted({str(r["speaker"]) for r in rows})
    rng = random.Random(SEED)
    rng.shuffle(spk)
    n_val = max(1, round(len(spk) * val_frac))
    val_spk = set(spk[:n_val])
    tr = [r for r in rows if str(r["speaker"]) not in val_spk]
    va = [r for r in rows if str(r["speaker"]) in val_spk]
    return tr, va, sorted(val_spk)


class SOUtterances(torch.utils.data.Dataset):
    """Decodes audio and precomputes everything the losses need on the CPU
    side: CTC target ids, GOP-SF hypothesis sequences, human scores."""

    def __init__(self, rows, sym2idx, cand_idx, blank, stats):
        self.items = []
        self.blank = blank
        self.cand = cand_idx
        full2local = {v: k + 1 for k, v in enumerate(cand_idx)}
        for r in rows:
            targets = build_targets_with_words(r["words"])
            if any(t["ipa"] is None or t["ipa"] not in sym2idx for t in targets):
                stats["skipped_unmapped"] += 1
                continue
            canon = [full2local[sym2idx[t["ipa"]]] for t in targets]
            scores = [float(np.mean(t["human_scores"])) for t in targets]
            said = pronounced_ipa(r["words"], stats)
            ctc_ids = None
            if said is not None and all(s in sym2idx for s in said) and len(said) > 0:
                ctc_ids = [sym2idx[s] for s in said]
            else:
                stats["no_ctc"] += 1
            self.items.append({"audio": r["audio"]["bytes"], "canon": canon, "scores": scores,
                               "ctc": ctc_ids})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        wav, sr = sf.read(io.BytesIO(it["audio"]), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        assert sr == SR, sr
        wav = trim_silence(wav, sr)
        wav = (wav - wav.mean()) / np.sqrt(wav.var() + 1e-7)  # Wav2Vec2FeatureExtractor do_normalize
        return {"wav": wav, **it}


def gop_hypotheses(canon, C):
    """Same hypothesis order as src/features_v2.gop_sf: canonical, then per
    phone i: every other candidate substituted at i, then deletion of i."""
    N = len(canon)
    seqs = [canon]
    for i in range(N):
        for c in range(1, C + 1):
            if c != canon[i]:
                s = list(canon); s[i] = c; seqs.append(s)
        if N > 1:
            seqs.append(canon[:i] + canon[i + 1:])
    return seqs


class BatchLoader:
    """index list -> collated batch (a class, not a lambda, so DataLoader
    workers can pickle it on Windows)."""

    def __init__(self, ds):
        self.ds = ds

    def __call__(self, idx):
        return collate([self.ds[i] for i in idx])


def collate(batch):
    lens = [len(b["wav"]) for b in batch]
    x = torch.zeros(len(batch), max(lens))
    mask = torch.zeros(len(batch), max(lens), dtype=torch.long)
    for j, b in enumerate(batch):
        x[j, :lens[j]] = torch.from_numpy(b["wav"])
        mask[j, :lens[j]] = 1
    return {"x": x, "mask": mask, "items": batch}


def length_bucketed(ds, batch, shuffle, seed):
    idx = list(range(len(ds)))
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(idx)
    # sort within windows of 50 batches so padding stays small but order stays random
    w = batch * 50
    out = []
    for k in range(0, len(idx), w):
        chunk = sorted(idx[k:k + w], key=lambda i: len(ds.items[i]["audio"]))
        out += [chunk[j:j + batch] for j in range(0, len(chunk), batch)]
    if shuffle:
        rng.shuffle(out)
    return out


# ---------------------------------------------------------------- losses

class GopToScore(nn.Module):
    """(gop_sf, sub_llr, del_llr) -> 0-2 score. Init so higher GOP = higher score."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 1)
        with torch.no_grad():
            self.lin.weight.copy_(torch.tensor([[0.5, 0.1, 0.1]]))
            self.lin.bias.fill_(2.0)

    def forward(self, f):
        return 2 * torch.sigmoid(self.lin(f).squeeze(-1))


def gop_sf_torch(lp, canon, C):
    """Differentiable GOP-SF for one utterance. lp: [T, 1+C] log-probs over
    blank + candidates (not renormalised, matching features_v2). Returns
    [N, 3] features (gop_sf, sub_llr, del_llr), clipped like features_v2."""
    T = lp.shape[0]
    N = len(canon)
    seqs = gop_hypotheses(canon, C)
    B = len(seqs)
    L = torch.tensor([len(s) for s in seqs])
    tg = torch.zeros(B, int(L.max()), dtype=torch.long)
    # a CTC path needs len + (#adjacent repeats) frames; mask impossible ones
    need = torch.empty(B)
    for j, s in enumerate(seqs):
        tg[j, :len(s)] = torch.tensor(s)
        need[j] = len(s) + sum(1 for a, b in zip(s, s[1:]) if a == b)
    feasible = (need <= T).to(lp.device)
    nll = F.ctc_loss(lp.unsqueeze(1).expand(T, B, lp.shape[1]), tg.to(lp.device),
                     torch.full((B,), T, dtype=torch.long), L, blank=0,
                     reduction="none", zero_infinity=True)
    ll = torch.where(feasible, -nll, torch.full_like(nll, -1e4))
    ll_can = ll[0]
    per = (C - 1) + (1 if N > 1 else 0)
    block = ll[1:].view(N, per)
    subs = block[:, :C - 1]
    if N > 1:
        dele = block[:, C - 1]
        alts = torch.cat([ll_can.expand(N, 1), subs, dele[:, None]], 1)
        del_llr = ll_can - dele
    else:
        alts = torch.cat([ll_can.expand(N, 1), subs], 1)
        del_llr = torch.full((N,), 20.0, device=lp.device)
    gop = ll_can - torch.logsumexp(alts, 1)
    sub_llr = ll_can - subs.max(1).values
    f = torch.stack([gop, sub_llr, del_llr], 1)
    return torch.clamp(f, -50, 50)


def batch_losses(model, gop_head, batch, cand_cols, blank, C, device, amp_dtype, err_weight):
    x = batch["x"].to(device)
    mask = batch["mask"].to(device)
    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        logits = model(input_values=x, attention_mask=mask).logits
    frames = model._get_feat_extract_output_lengths(mask.sum(1)).tolist()
    ctc_terms, se, w, preds, gold = [], [], [], [], []
    for j, it in enumerate(batch["items"]):
        T = int(frames[j])
        lp_full = torch.log_softmax(logits[j, :T].float(), -1)
        if it["ctc"] is not None:
            tgt = torch.tensor(it["ctc"], device=device)
            ctc_terms.append(F.ctc_loss(lp_full[:, None], tgt[None], torch.tensor([T]),
                                        torch.tensor([len(tgt)]), blank=blank,
                                        reduction="sum", zero_infinity=True) / len(tgt))
        f = gop_sf_torch(lp_full[:, cand_cols], it["canon"], C)
        s = gop_head(f)
        y = torch.tensor(it["scores"], device=device)
        wt = torch.where(y < 2, torch.full_like(y, err_weight), torch.ones_like(y))
        se.append((s - y) ** 2 * wt); w.append(wt)
        preds.append(s.detach().cpu()); gold.append(y.cpu())
    ctc = torch.stack(ctc_terms).mean() if ctc_terms else torch.zeros((), device=device)
    gop = torch.cat(se).sum() / torch.cat(w).sum()
    return ctc, gop, torch.cat(preds), torch.cat(gold)


# ---------------------------------------------------------------- train

def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


@torch.no_grad()
def evaluate(model, gop_head, ds, args, cand_cols, blank, C, device, amp_dtype):
    model.eval()
    P, G, ctc_sum, n = [], [], 0.0, 0
    for idx in length_bucketed(ds, args.batch, False, 0):
        b = collate([ds[i] for i in idx])
        ctc, _, p, g = batch_losses(model, gop_head, b, cand_cols, blank, C, device, amp_dtype,
                                    args.err_weight)
        P.append(p); G.append(g); ctc_sum += float(ctc) * len(idx); n += len(idx)
    model.train()
    P, G = torch.cat(P), torch.cat(G)
    flag = P < 1.75
    err = G < 2
    tp = float((flag & err).sum())
    return {"r": pearson(P, G), "ctc": ctc_sum / max(n, 1), "n_phones": int(len(G)),
            "precision@1.75": tp / max(float(flag.sum()), 1), "recall@1.75": tp / max(float(err.sum()), 1)}


def fp32_frontend(model):
    """The positional conv (encoder.pos_conv_embed, a 128-tap grouped conv)
    overflows in fp16 on real speech: NaN before the first transformer
    layer. Run it, and the CNN front end feeding it, in fp32 even under
    autocast. All three are frozen, so this costs little."""
    for mod in (model.wav2vec2.feature_extractor, model.wav2vec2.feature_projection,
                model.wav2vec2.encoder.pos_conv_embed):
        fwd = mod.forward

        def wrapped(*a, _fwd=fwd, **k):
            with torch.autocast("cuda", enabled=False):
                a = [x.float() if torch.is_tensor(x) else x for x in a]
                return _fwd(*a, **k)
        mod.forward = wrapped


def setup_trainable(model, n_top):
    for p in model.parameters():
        p.requires_grad = False
    layers = model.wav2vec2.encoder.layers
    for l in layers[len(layers) - n_top:]:
        for p in l.parameters():
            p.requires_grad = True
    for mod in (model.wav2vec2.encoder.layer_norm, model.lm_head):
        for p in mod.parameters():
            p.requires_grad = True


def save_hf(model, out):
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    # processor files (tokenizer vocab, feature-extractor config) so
    # AutoProcessor.from_pretrained(out) works for the rest of the pipeline
    # (only these: the snapshot dir also holds the cached ORIGINAL weights)
    for fn in PROCESSOR_FILES:
        shutil.copy(hf_hub_download(BASE_MODEL, fn), os.path.join(out, fn))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "ft"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=0, help="grad accumulation; default = 8 // batch")
    ap.add_argument("--unfreeze", type=int, default=8, help="top transformer layers to train (of 24)")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=1e-2)
    ap.add_argument("--ctc-weight", type=float, default=1.0)
    ap.add_argument("--gop-weight", type=float, default=1.0)
    ap.add_argument("--err-weight", type=float, default=4.0, help="loss weight of phones scored < 2")
    ap.add_argument("--val-frac", type=float, default=0.1, help="share of TRAIN speakers held out")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="use only N train utterances (debugging)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-checkpointing", action="store_true", help="faster, more memory")
    ap.add_argument("--max-hours", type=float, default=0, help="stop cleanly after this long (budget guard)")
    args = ap.parse_args()
    accum = args.accum or max(1, 8 // args.batch)

    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = None
    if device.type == "cuda":
        bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        amp_dtype = torch.bfloat16 if bf16 else torch.float16
    print(f"device={device} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''} amp={amp_dtype}",
          flush=True)

    vocab = json.load(open(hf_hub_download(BASE_MODEL, "vocab.json"), encoding="utf-8"))
    blank = vocab["<pad>"]
    cands = target_inventory(vocab)
    cand_idx = [vocab[s] for s in cands]
    cand_cols = torch.tensor([blank] + cand_idx, device=device)
    C = len(cand_idx)

    rows = load_rows(args.limit or None)
    tr_rows, va_rows, val_spk = speaker_split(rows, args.val_frac)
    stats = {k: 0 for k in ("sub", "del", "unresolved", "bad_index", "no_ctc", "skipped_unmapped")}
    tr = SOUtterances(tr_rows, vocab, cand_idx, blank, stats)
    va = SOUtterances(va_rows, vocab, cand_idx, blank, stats)
    del rows, tr_rows, va_rows
    print(f"train utts={len(tr)} val utts={len(va)} ({len(val_spk)} speakers)  annotations: {stats}", flush=True)

    model = Wav2Vec2ForCTC.from_pretrained(BASE_MODEL, layerdrop=0.0, mask_time_prob=0.05,
                                           ctc_zero_infinity=True).to(device)
    assert model.config.vocab_size >= max(vocab.values()) + 1
    fp32_frontend(model)
    setup_trainable(model, args.unfreeze)
    if not args.no_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    gop_head = GopToScore().to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params) / 1e6:.1f}M  batch={args.batch} accum={accum}",
          flush=True)
    opt = torch.optim.AdamW([{"params": params, "lr": args.lr, "weight_decay": 0.01},
                             {"params": gop_head.parameters(), "lr": args.head_lr, "weight_decay": 0.0}])
    steps_per_epoch = math.ceil(math.ceil(len(tr) / args.batch) / accum)
    total = steps_per_epoch * args.epochs
    warm = max(1, int(0.1 * total))
    def lr_factor(s):  # linear warmup, then linear decay to 0
        return (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    scaler = torch.amp.GradScaler(enabled=amp_dtype == torch.float16)

    os.makedirs(args.out, exist_ok=True)
    last_path = os.path.join(args.out, "last.pt")
    log = {"args": vars(args), "val_speakers": val_spk, "annotation_stats": stats, "epochs": []}
    start_epoch, best_r, bad = 0, -1.0, 0
    if os.path.exists(last_path):
        st = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(st["trainable"], strict=False)
        gop_head.load_state_dict(st["gop_head"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        start_epoch, best_r, bad, log = st["epoch"] + 1, st["best_r"], st["bad"], st["log"]
        print(f"resumed from epoch {st['epoch']} (best val r {best_r:.4f})", flush=True)
    else:
        # frozen-model baseline on the same validation speakers (only the 4-param map is untrained,
        # so fit it first for a fair comparison)
        # one pass over train: frozen forward, GOP features cached, then fit the map on them
        feats, ys = [], []
        model.eval()
        with torch.no_grad():
            for idx in length_bucketed(tr, args.batch, False, 0):
                b = collate([tr[i] for i in idx])
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    logits = model(input_values=b["x"].to(device), attention_mask=b["mask"].to(device)).logits
                frames = model._get_feat_extract_output_lengths(b["mask"].sum(1)).tolist()
                for j, it in enumerate(b["items"]):
                    lp = torch.log_softmax(logits[j, :int(frames[j])].float(), -1)[:, cand_cols]
                    feats.append(gop_sf_torch(lp, it["canon"], C))
                    ys.append(torch.tensor(it["scores"], device=device))
        model.train()
        f_all, y_all = torch.cat(feats), torch.cat(ys)
        wt = torch.where(y_all < 2, torch.full_like(y_all, args.err_weight), torch.ones_like(y_all))
        head_opt = torch.optim.Adam(gop_head.parameters(), lr=0.05)
        for _ in range(500):
            loss = ((gop_head(f_all) - y_all) ** 2 * wt).sum() / wt.sum()
            head_opt.zero_grad(); loss.backward(); head_opt.step()
        del feats, ys, f_all, y_all
        base = evaluate(model, gop_head, va, args, cand_cols, blank, C, device, amp_dtype)
        log["baseline_frozen"] = base
        if not math.isfinite(base["r"]):
            sys.exit(f"frozen baseline is not finite ({base}); refusing to train")
        best_r = base["r"]
        print(f"frozen baseline (val): {base}", flush=True)

    t_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        batches = length_bucketed(tr, args.batch, True, SEED + epoch)
        to_batch = BatchLoader(tr)
        if args.workers:
            loader = torch.utils.data.DataLoader(batches, batch_size=None, shuffle=False,
                                                 num_workers=args.workers, collate_fn=to_batch)
        else:
            loader = (to_batch(idx) for idx in batches)
        run_ctc = run_gop = 0.0
        n_skipped = 0
        for k, b in enumerate(loader):
            ctc, gop, _, _ = batch_losses(model, gop_head, b, cand_cols, blank, C, device, amp_dtype,
                                          args.err_weight)
            loss = (args.ctc_weight * ctc + args.gop_weight * gop) / accum
            if not torch.isfinite(loss):
                n_skipped += 1
                print(f"ep {epoch} batch {k}: non-finite loss, skipped ({n_skipped} so far)", flush=True)
                if n_skipped > 20:
                    sys.exit("too many non-finite losses; stopping")
                continue
            scaler.scale(loss).backward()
            if (k + 1) % accum == 0 or k + 1 == len(batches):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            run_ctc += ctc.item(); run_gop += gop.item()
            if (k + 1) % 50 == 0:
                el = time.time() - t0
                print(f"ep {epoch} {k + 1}/{len(batches)} ctc={run_ctc / (k + 1):.3f} gop={run_gop / (k + 1):.4f} "
                      f"{(k + 1) * args.batch / el:.2f} utt/s", flush=True)
        ev = evaluate(model, gop_head, va, args, cand_cols, blank, C, device, amp_dtype)
        ev.update(epoch=epoch, train_ctc=run_ctc / len(batches), train_gop=run_gop / len(batches),
                  minutes=(time.time() - t0) / 60)
        log["epochs"].append(ev)
        improved = ev["r"] > best_r
        print(f"epoch {epoch}: val r={ev['r']:.4f} (best {max(best_r, ev['r']):.4f}) ctc={ev['ctc']:.3f} "
              f"P={ev['precision@1.75']:.3f} R={ev['recall@1.75']:.3f}  {ev['minutes']:.1f} min", flush=True)
        if improved:
            best_r, bad = ev["r"], 0
            save_hf(model, os.path.join(args.out, "best"))
            torch.save(gop_head.state_dict(), os.path.join(args.out, "best", "gop_head.pt"))
        else:
            bad += 1
        train_names = {n for n, p in model.named_parameters() if p.requires_grad}
        trainable = {k: v for k, v in model.state_dict().items() if k in train_names}
        torch.save({"trainable": trainable, "gop_head": gop_head.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                    "best_r": best_r, "bad": bad, "log": log}, last_path)
        json.dump(log, open(os.path.join(args.out, "log.json"), "w"), indent=2)
        if bad >= args.patience:
            print(f"early stop: no val improvement for {bad} epochs", flush=True)
            break
        if args.max_hours and (time.time() - t_start) / 3600 > args.max_hours:
            print("stopping: --max-hours reached (rerun the same command to resume)", flush=True)
            break
    json.dump(log, open(os.path.join(args.out, "log.json"), "w"), indent=2)
    if not os.path.isdir(os.path.join(args.out, "best")):
        print("fine-tuning never beat the frozen baseline on validation; no best/ saved", flush=True)
    else:
        print(f"best val r={best_r:.4f}  -> {os.path.join(args.out, 'best')}", flush=True)


if __name__ == "__main__":
    main()
