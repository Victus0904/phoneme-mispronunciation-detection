"""Per-phone Goodness-of-Pronunciation (GOP) features from an aligned
utterance: mean log-posterior, margin vs. best competitor, posterior
entropy, duration, and the competitor's identity.

Margin/entropy/competitor are computed on the content-phone-only
renormalized distribution (src.acoustic.content_log_posteriors), not the
raw CTC output -- comparing a phone against the ever-present blank/<pad>
would make "margin vs. best competitor" and "entropy" trivially dominated
by blank rather than reflecting real phone-vs-phone confusability.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.acoustic import content_log_posteriors, get_log_posteriors, trim_silence, vocab_maps, load_model
from src.phone_mapping import map_word_phones_to_ipa_targets


def build_utterance_targets(example):
    """Flatten an utterance's words into (target, human_score) pairs.

    Returns: list of {'ipa', 'arpa': [phone(s)], 'human_scores': [score(s)]}
    -- 'arpa'/'human_scores' have length 2 for a fused rhotic (vowel+R)
    target, length 1 otherwise, and both lists are always the same length.

    'words' entries without a 'phones-accuracy' key (e.g. demo sentences
    with no ground truth, see app.py's G2P path) get human_scores of None.
    """
    out = []
    for w in example["words"]:
        word_targets = map_word_phones_to_ipa_targets(w["phones"])
        scores = w.get("phones-accuracy")
        if scores is None:
            scores = [None] * len(w["phones"])
        i = 0
        for t in word_targets:
            n = len(t["arpa"])
            out.append({"ipa": t["ipa"], "arpa": t["arpa"], "human_scores": scores[i:i + n]})
            i += n
    return out


def compute_gop_features(full_log_posteriors: np.ndarray, idx2sym: dict, targets: list, boundaries: list):
    """
    full_log_posteriors: [T, V_full] raw log-posteriors (blank included).
    idx2sym: full-vocab index -> symbol.
    targets: from build_utterance_targets (or map_word_phones_to_ipa_targets).
    boundaries: from align.align(), one per target, same order.

    Returns: list of per-ORIGINAL-ARPAbet-phone feature dicts (a fused
    rhotic target expands back into 2 rows sharing identical acoustic
    features but carrying their own human_score).
    """
    content_lp, content_idx2sym, content_sym2idx = content_log_posteriors(full_log_posteriors, idx2sym)

    rows = []
    for target, b in zip(targets, boundaries):
        target_idx = content_sym2idx[target["ipa"]]
        seg = content_lp[b["start"]:b["end"] + 1, :]  # [dur_or_1, V_content]

        seg_mean_vec = seg.mean(axis=0)
        mean_log_posterior = float(seg_mean_vec[target_idx])

        probs = np.exp(seg)
        entropy = float((-(probs * seg).sum(axis=1)).mean())

        competitor_vec = seg_mean_vec.copy()
        competitor_vec[target_idx] = -np.inf
        competitor_idx = int(np.argmax(competitor_vec))
        competitor_log_posterior = float(seg_mean_vec[competitor_idx])
        margin = mean_log_posterior - competitor_log_posterior

        base_feats = {
            "ipa_target": target["ipa"],
            "start_frame": b["start"],
            "end_frame": b["end"],
            "duration_frames": b["duration"],
            "mean_log_posterior": mean_log_posterior,
            "margin": margin,
            "entropy": entropy,
            "top_competitor": content_idx2sym[competitor_idx],
        }

        human_scores = target.get("human_scores", [None] * len(target["arpa"]))
        for arpa_phone, human_score in zip(target["arpa"], human_scores):
            row = dict(base_feats)
            row["arpa_phone"] = arpa_phone
            row["human_score"] = human_score
            row["fused_with"] = [p for p in target["arpa"] if p != arpa_phone] or None
            rows.append(row)

    return rows


CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
SEED = 42
N_PER_SPLIT = 500  # 500 train + 500 test = 1000 utterances max, preserving
                    # speechocean762's native speaker-disjoint train/test split


def _select_utterance_indices(ds, split, n, seed=SEED):
    """Deterministic subsample of row indices within one native split."""
    rng = np.random.RandomState(seed)
    n_total = len(ds[split])
    n = min(n, n_total)
    return sorted(rng.choice(n_total, size=n, replace=False).tolist())


def extract_corpus_features(ds, model, processor, sym2idx, idx2sym, blank_idx,
                             n_per_split=N_PER_SPLIT, cache_dir=CACHE_DIR, seed=SEED,
                             progress_every=50, log=print):
    """Batch GOP feature extraction over a subsample of speechocean762, with
    posteriors and the final feature table cached to disk so re-running
    (e.g. after a features.py tweak) doesn't require redoing model
    inference. Returns a pandas DataFrame, one row per (utterance, phone).
    """
    import pandas as pd
    from src.align import align

    posterior_dir = os.path.join(cache_dir, "posteriors")
    os.makedirs(posterior_dir, exist_ok=True)
    features_path = os.path.join(cache_dir, "features.parquet")

    if os.path.exists(features_path):
        log(f"[cache hit] {features_path} already exists, loading instead of recomputing.")
        return pd.read_parquet(features_path)

    all_rows = []
    n_done = 0
    for split in ("train", "test"):
        indices = _select_utterance_indices(ds, split, n_per_split, seed=seed)
        log(f"split={split}: processing {len(indices)} utterances")

        for row_idx in indices:
            example = ds[split][row_idx]
            uid = f"{split}_{row_idx}"
            posterior_path = os.path.join(posterior_dir, f"{uid}.npy")

            audio = example["audio"]
            waveform = audio["array"].astype(np.float32)
            sr = audio["sampling_rate"]

            if os.path.exists(posterior_path):
                full_logpost = np.load(posterior_path)
            else:
                trimmed = trim_silence(waveform, sr)
                full_logpost = get_log_posteriors(trimmed, sr, model, processor)
                np.save(posterior_path, full_logpost)

            targets = build_utterance_targets(example)
            if len(targets) == 0 or full_logpost.shape[0] < len(targets):
                log(f"  [skip] {uid}: {len(targets)} targets vs {full_logpost.shape[0]} frames")
                continue

            target_vocab_indices = [sym2idx[t["ipa"]] for t in targets]
            boundaries = align(full_logpost, target_vocab_indices, blank_idx)
            rows = compute_gop_features(full_logpost, idx2sym, targets, boundaries)

            for r in rows:
                r["uid"] = uid
                r["split"] = split
                r["speaker"] = example["speaker"]
                r["gender"] = example["gender"]
                r["age"] = example["age"]
                all_rows.append(r)

            n_done += 1
            if n_done % progress_every == 0:
                log(f"  {n_done} utterances processed...")

    df = pd.DataFrame(all_rows)
    df.to_parquet(features_path, index=False)
    log(f"wrote {len(df)} phone-rows from {n_done} utterances to {features_path}")
    return df


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        from src.data import load_speechocean762
        ds = load_speechocean762(decode_audio=True)
        model, processor = load_model()
        sym2idx, idx2sym = vocab_maps(processor)
        blank_idx = sym2idx["<pad>"]
        extract_corpus_features(ds, model, processor, sym2idx, idx2sym, blank_idx)
        raise SystemExit(0)

    from src.data import load_speechocean762
    from src.acoustic import load_model, get_log_posteriors, trim_silence, vocab_maps
    from src.align import align, alignment_score

    ds = load_speechocean762(decode_audio=True)
    example = ds["train"][0]
    audio = example["audio"]
    waveform = audio["array"].astype(np.float32)
    sr = audio["sampling_rate"]

    model, processor = load_model()
    sym2idx, idx2sym = vocab_maps(processor)
    blank_idx = sym2idx["<pad>"]
    trimmed = trim_silence(waveform, sr)
    full_logpost = get_log_posteriors(trimmed, sr, model, processor)

    targets = build_utterance_targets(example)
    target_vocab_indices = [sym2idx[t["ipa"]] for t in targets]
    boundaries = align(full_logpost, target_vocab_indices, blank_idx)
    score = alignment_score(full_logpost, target_vocab_indices, boundaries)

    rows = compute_gop_features(full_logpost, idx2sym, targets, boundaries)

    ms_per_frame = len(trimmed) / sr / full_logpost.shape[0] * 1000
    print(f"text: {example['text']}  (alignment log-score: {score:.1f})")
    print(f"{'arpa':8s} {'human':6s} {'ipa':6s} {'dur(ms)':8s} {'mean_logp':10s} {'margin':8s} {'entropy':8s} {'competitor':10s}")
    for r in rows:
        fused = f" (fused w/ {r['fused_with']})" if r["fused_with"] else ""
        print(f"{r['arpa_phone']:8s} {str(r['human_score']):6s} {r['ipa_target']:6s} "
              f"{r['duration_frames']*ms_per_frame:6.1f}   {r['mean_log_posterior']:9.3f}  "
              f"{r['margin']:7.3f}  {r['entropy']:7.3f}  {r['top_competitor']:10s}{fused}")
