"""v2 per-phone features: everything in v1 (src/features.py) plus three
families that address v1's diagnosed weaknesses (REPORT.md Sections 2.2/5):

1. Alignment-free GOP ("GOP-SF", substitution/deletion likelihood ratios).
   v1's GOP reads posteriors only inside the forced-aligned span, which for
   this peaky CTC model is almost always a single 20ms frame -- one noisy
   sample. GOP-SF instead asks the CTC model how much more likely the
   *whole canonical phone sequence* is than the same sequence with phone i
   substituted by each other candidate phone (or deleted), marginalizing
   over all alignments. No segmentation, so no 1-frame bottleneck:
     gop_sf       = log P(canonical) - logsumexp(P(canonical), P(each sub), P(del))
     sf_sub_llr   = log P(canonical) - max_c log P(sub_c)
     sf_del_llr   = log P(canonical) - log P(del)
     sf_best_sub  = the most likely substitute phone (categorical)
   Computed with torch's ctc_loss over the 48 target symbols + blank.

2. Windowed / blank-aware posteriors around the aligned spike (CTC spikes
   can land a frame off the Viterbi span), phone identity, position in the
   word, neighbor-phone GOP, and utterance-level GOP statistics (a
   per-utterance proficiency proxy -- human raters' per-phone scores are
   strongly correlated within a speaker/utterance).

3. Frozen wav2vec2 hidden states (layers 8/16/24), mean-pooled over the
   aligned span +/-1 frame. Read-only representations from the same frozen
   model; nothing is fine-tuned.

All inputs come from cache/v2/utt/*.npz (scripts/extract_full.py), or, at
inference time, from the same model call (see src/scorer.py).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.acoustic import content_log_posteriors
from src.align import align
from src.phone_mapping import map_word_phones_to_ipa_targets

HIDDEN_LAYERS = (8, 16, 24)
EMB_WINDOW = 1  # frames either side of the aligned span to pool over
POST_WINDOW = 2  # frames either side for windowed-posterior features


def target_inventory(sym2idx):
    """The 48 distinct IPA targets any ARPAbet phone can map to (the GOP-SF
    substitution candidate set), in a fixed order."""
    from src.phone_mapping import _CONSONANT_MAP, _VOWEL_MAP, RHOTIC_R_FUSION
    syms = set(_CONSONANT_MAP.values()) | set(_VOWEL_MAP.values()) | set(RHOTIC_R_FUSION.values())
    syms = sorted(s for s in syms if s in sym2idx)
    return syms


def build_targets_with_words(words):
    """Like features.build_utterance_targets, but also records which word
    each target came from and its position inside that word."""
    out = []
    for wi, w in enumerate(words):
        word_targets = map_word_phones_to_ipa_targets(w["phones"])
        scores = w.get("phones-accuracy") or [None] * len(w["phones"])
        i = 0
        for pi, t in enumerate(word_targets):
            n = len(t["arpa"])
            out.append({"ipa": t["ipa"], "arpa": t["arpa"], "human_scores": scores[i:i + n],
                        "word_index": wi, "word_text": w.get("text", ""),
                        "pos_in_word": pi, "word_len": len(word_targets)})
            i += n
    return out


def gop_sf(full_logpost, target_full_idx, cand_full_idx, blank_idx, chunk=384):
    """Alignment-free GOP via CTC likelihoods of substituted/deleted
    canonical sequences.

    full_logpost: [T, V] log-posteriors. target_full_idx: [N] canonical
    target vocab indices. cand_full_idx: [C] candidate vocab indices.
    Returns dict of [N] arrays (+ best-substitute candidate index).
    """
    T = full_logpost.shape[0]
    N = len(target_full_idx)
    C = len(cand_full_idx)
    # restrict to blank + candidates (CTC path scores need no renormalization:
    # every hypothesis is scored on the same frames, so ratios are unchanged)
    cols = [blank_idx] + list(cand_full_idx)
    lp = torch.from_numpy(np.ascontiguousarray(full_logpost[:, cols], dtype=np.float32))  # [T, 1+C]
    full2local = {v: k + 1 for k, v in enumerate(cand_full_idx)}
    canon = [full2local[v] for v in target_full_idx]

    seqs = [canon]
    for i in range(N):
        for c in range(1, C + 1):
            if c == canon[i]:
                continue
            s = list(canon)
            s[i] = c
            seqs.append(s)
        if N > 1:
            seqs.append(canon[:i] + canon[i + 1:])

    lls = np.empty(len(seqs), dtype=np.float64)
    for b0 in range(0, len(seqs), chunk):
        batch = seqs[b0:b0 + chunk]
        B = len(batch)
        lens = torch.tensor([len(s) for s in batch], dtype=torch.long)
        tg = torch.zeros(B, max(len(s) for s in batch), dtype=torch.long)
        for j, s in enumerate(batch):
            tg[j, :len(s)] = torch.tensor(s)
        loss = torch.nn.functional.ctc_loss(
            lp.unsqueeze(1).expand(T, B, lp.shape[1]), tg,
            torch.full((B,), T, dtype=torch.long), lens,
            blank=0, reduction="none", zero_infinity=False)
        lls[b0:b0 + B] = -loss.double().numpy()

    ll_can = lls[0]
    per = (C - 1) + (1 if N > 1 else 0)
    out = {"gop_sf": np.zeros(N), "sf_sub_llr": np.zeros(N), "sf_del_llr": np.zeros(N),
           "sf_best_sub": np.zeros(N, dtype=np.int64), "sf_vec": np.zeros((N, C + 1), dtype=np.float32)}
    for i in range(N):
        block = lls[1 + i * per: 1 + (i + 1) * per]
        subs = block[:C - 1]
        dele = block[C - 1] if N > 1 else -np.inf
        alts = np.concatenate([[ll_can], subs, [dele]])
        alts = alts[np.isfinite(alts)]
        m = alts.max()
        lse = m + np.log(np.exp(alts - m).sum())
        out["gop_sf"][i] = ll_can - lse
        out["sf_sub_llr"][i] = ll_can - subs.max()
        out["sf_del_llr"][i] = (ll_can - dele) if np.isfinite(dele) else 20.0
        cand_order = [c for c in range(1, C + 1) if c != canon[i]]
        out["sf_best_sub"][i] = cand_full_idx[cand_order[int(np.argmax(subs))] - 1]
        # GOPT-style vector: LLR of canonical vs. each candidate (0 for the
        # canonical itself), plus deletion, in fixed candidate order
        vec = np.zeros(C + 1)
        vec[np.array(cand_order) - 1] = ll_can - subs
        vec[C] = (ll_can - dele) if np.isfinite(dele) else 20.0
        out["sf_vec"][i] = np.clip(np.nan_to_num(vec, nan=0.0, posinf=20.0, neginf=-20.0), -50, 50)
    for k in ("gop_sf", "sf_sub_llr", "sf_del_llr"):
        out[k] = np.clip(np.nan_to_num(out[k], nan=0.0, posinf=20.0, neginf=-20.0), -50, 50)
    return out


def utterance_features(full_logpost, words, sym2idx, idx2sym, cand_syms, hidden=None):
    """Per-ORIGINAL-ARPAbet-phone feature rows for one utterance.

    full_logpost: [T, V]. words: [{'text', 'phones', optional 'phones-accuracy'}].
    hidden: optional {layer: [T, D]} -> also returns pooled embeddings.
    Returns (rows: list[dict], emb: {key: [n_rows, D]}) where key is a hidden
    layer number, "sfvec" (GOP-SF LLR vs. each of the 48 candidates +
    deletion) or "postvec" (windowed max content log-posterior of each
    candidate) -- the latter two are GOPT-style per-phone vectors.
    """
    blank_idx = sym2idx["<pad>"]
    targets = build_targets_with_words(words)
    tidx = [sym2idx[t["ipa"]] for t in targets]
    T = full_logpost.shape[0]
    bounds = align(full_logpost, tidx, blank_idx)
    content_lp, c_idx2sym, c_sym2idx = content_log_posteriors(full_logpost, idx2sym)
    sf = gop_sf(full_logpost, tidx, [sym2idx[s] for s in cand_syms], blank_idx)
    blank_lp = full_logpost[:, blank_idx]

    base = []
    for k, (t, b) in enumerate(zip(targets, bounds)):
        ci = c_sym2idx[t["ipa"]]
        seg = content_lp[b["start"]:b["end"] + 1]
        segm = seg.mean(axis=0)
        mlp = float(segm[ci])
        comp = segm.copy(); comp[ci] = -np.inf
        cidx = int(np.argmax(comp))
        ent = float((-(np.exp(seg) * seg).sum(axis=1)).mean())

        w0, w1 = max(0, b["start"] - POST_WINDOW), min(T, b["end"] + 1 + POST_WINDOW)
        full_win = full_logpost[w0:w1, tidx[k]]
        cont_win = content_lp[w0:w1, ci]
        base.append({
            "ipa_target": t["ipa"], "start_frame": b["start"], "end_frame": b["end"],
            "duration_frames": b["duration"],
            "mean_log_posterior": mlp, "margin": mlp - float(segm[cidx]), "entropy": ent,
            "top_competitor": c_idx2sym[cidx],
            "full_lp_max_win": float(full_win.max()),
            "full_lp_lse_win": float(np.logaddexp.reduce(full_win)),
            "content_lp_max_win": float(cont_win.max()),
            "blank_lp_span": float(blank_lp[b["start"]:b["end"] + 1].mean()),
            "gop_sf": float(sf["gop_sf"][k]), "sf_sub_llr": float(sf["sf_sub_llr"][k]),
            "sf_del_llr": float(sf["sf_del_llr"][k]),
            "sf_best_sub": idx2sym[int(sf["sf_best_sub"][k])],
            "word_index": t["word_index"], "word_text": t["word_text"],
            "pos_in_word": t["pos_in_word"], "word_len": t["word_len"],
            "is_word_initial": int(t["pos_in_word"] == 0),
            "is_word_final": int(t["pos_in_word"] == t["word_len"] - 1),
            "n_targets": len(targets), "rel_position": k / max(1, len(targets) - 1),
        })

    gsf = np.array([r["gop_sf"] for r in base])
    mlps = np.array([r["mean_log_posterior"] for r in base])
    for k, r in enumerate(base):
        r["prev_gop_sf"] = float(gsf[k - 1]) if k > 0 else 0.0
        r["next_gop_sf"] = float(gsf[k + 1]) if k + 1 < len(base) else 0.0
        r["prev_mlp"] = float(mlps[k - 1]) if k > 0 else 0.0
        r["next_mlp"] = float(mlps[k + 1]) if k + 1 < len(base) else 0.0
        r["prev_ipa"] = base[k - 1]["ipa_target"] if k > 0 else "<s>"
        r["next_ipa"] = base[k + 1]["ipa_target"] if k + 1 < len(base) else "</s>"
        r["utt_mean_gop_sf"] = float(gsf.mean())
        r["utt_min_gop_sf"] = float(gsf.min())
        r["utt_mean_mlp"] = float(mlps.mean())
        r["utt_frac_low_gop_sf"] = float((gsf < -1.0).mean())
        r["frames_per_target"] = T / len(base)
        same_word = [j for j in range(len(base)) if base[j]["word_index"] == r["word_index"]]
        r["word_mean_gop_sf"] = float(gsf[same_word].mean())

    cand_content = [c_sym2idx[c] for c in cand_syms]
    post_vec = np.zeros((len(base), len(cand_syms)), dtype=np.float32)
    for k, b in enumerate(bounds):
        w0, w1 = max(0, b["start"] - POST_WINDOW), min(T, b["end"] + 1 + POST_WINDOW)
        post_vec[k] = content_lp[w0:w1][:, cand_content].max(axis=0)

    rows, emb_rows = [], []
    for k, (t, r) in enumerate(zip(targets, base)):
        for arpa, hs in zip(t["arpa"], t["human_scores"]):
            row = dict(r)
            row["arpa_phone"] = arpa
            row["arpa_base"] = arpa.rstrip("012")
            row["human_score"] = hs
            row["fused"] = int(len(t["arpa"]) > 1)
            rows.append(row)
            emb_rows.append(k)

    emb = {"sfvec": sf["sf_vec"][emb_rows], "postvec": np.maximum(post_vec, -30.0)[emb_rows]}
    if hidden is not None:
        for L, H in hidden.items():
            pooled = np.zeros((len(base), H.shape[1]), dtype=np.float32)
            for k, b in enumerate(bounds):
                w0, w1 = max(0, b["start"] - EMB_WINDOW), min(H.shape[0], b["end"] + 1 + EMB_WINDOW)
                pooled[k] = H[w0:w1].astype(np.float32).mean(axis=0)
            emb[L] = pooled[emb_rows].astype(np.float16)
    return rows, emb
