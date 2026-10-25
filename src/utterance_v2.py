"""Utterance-level features for the fluency and prosody (rhythm/intonation)
scorers. These are trained on speechocean762's expert utterance scores
(`fluency`, `prosodic`, 0-10) exactly like the phone scorer: frozen model,
speaker-disjoint split, test scored once (scripts/train_fluency.py).

Everything comes from the same single frozen-model pass the phone scorer
already makes -- the CTC posteriors, the forced alignment (via the phone
feature rows' start/end frames) and the pooled hidden states -- plus a
simple pitch/energy track from the waveform for intonation:

  timing   speech rate, pauses between words (count, length, share of time),
           regularity of word durations, leading/trailing silence
  hesitation  extra CTC spikes beyond the expected phones (repeats/fillers)
  accuracy GOP statistics (strongly correlated with fluency in the data)
  pitch    voiced F0 range/variability/slope and energy variability
  embedding mean/std of hidden states over speech frames (PCA'd in the head)
"""
import numpy as np

HIDDEN_LAYERS = (8, 16, 24)
SR = 16000


def _f0_track(wav, sr=SR, frame=0.04, hop=0.01, fmin=70.0, fmax=400.0):
    """Light autocorrelation pitch tracker (no extra dependencies): returns
    (f0 per frame in Hz with 0 for unvoiced, rms per frame)."""
    n, h = int(frame * sr), int(hop * sr)
    if len(wav) < n:
        return np.zeros(0), np.zeros(0)
    frames = np.lib.stride_tricks.sliding_window_view(wav, n)[::h].astype(np.float64)
    frames = frames - frames.mean(axis=1, keepdims=True)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    lo, hi = int(sr / fmax), int(sr / fmin)
    spec = np.fft.rfft(frames * np.hanning(n), 2 * n, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :n]
    ac = ac / (ac[:, :1] + 1e-12)
    seg = ac[:, lo:hi]
    lag = seg.argmax(axis=1) + lo
    peak = seg.max(axis=1)
    voiced = (peak > 0.45) & (rms > 0.1 * rms.max())
    f0 = np.where(voiced, sr / lag, 0.0)
    return f0, rms


def utterance_features(rows, logpost, hidden, n_samples, wav=None, blank_idx=0):
    """rows: the per-phone feature dicts/DataFrame rows for one utterance
    (src/features_v2.utterance_features output). logpost: [T, V].
    hidden: {layer: [T, D]}. n_samples: length of the trimmed audio that
    produced logpost. wav: that trimmed audio (for pitch), optional.
    Returns (scalar feature dict, {f"{stat}_h{L}": vector}).
    """
    T = logpost.shape[0]
    fs = n_samples / SR / T  # seconds per frame
    # one span per alignment target (fused vowel+R rows share a span)
    spans, seen = [], set()
    for r in rows:
        k = (r["start_frame"], r["end_frame"], r["word_index"])
        if k not in seen:
            seen.add(k)
            spans.append(k)
    words = {}
    for s, e, w in spans:
        a, b, n = words.get(w, (s, e, 0))
        words[w] = (min(a, s), max(b, e), n + 1)
    wl = [words[k] for k in sorted(words)]
    first, last = spans[0][0], max(e for _, e, _ in spans)
    speech_s = max(fs, (last - first + 1) * fs)
    gaps = np.array([max(0, wl[i + 1][0] - wl[i][1] - 1) * fs for i in range(len(wl) - 1)]) if len(wl) > 1 else np.zeros(1)
    in_gaps = []
    srt = sorted(spans)
    for i in range(len(srt) - 1):
        if srt[i][2] == srt[i + 1][2]:
            in_gaps.append(max(0, srt[i + 1][0] - srt[i][1] - 1) * fs)
    wdur = np.array([(b - a + 1) * fs / n for a, b, n in wl])

    am = logpost.argmax(axis=1)
    nonblank = am != blank_idx
    runs = int(np.sum(nonblank[1:] & ~nonblank[:-1]) + nonblank[0]) if T else 0
    blank_p = np.exp(logpost[:, blank_idx])

    g = np.array([r["gop_sf"] for r in rows]); m = np.array([r["mean_log_posterior"] for r in rows])
    f = {
        "dur_s": n_samples / SR, "speech_s": speech_s, "n_words": len(wl), "n_targets": len(spans),
        "lead_sil_s": first * fs, "trail_sil_s": (T - 1 - last) * fs,
        "rate_tps": len(spans) / speech_s, "rate_wps": len(wl) / speech_s,
        "gap_mean": float(gaps.mean()), "gap_max": float(gaps.max()), "gap_std": float(gaps.std()),
        "n_pause_03": int((gaps > 0.3).sum()), "n_pause_06": int((gaps > 0.6).sum()),
        "pause_ratio": float(gaps.sum() / speech_s),
        "in_gap_mean": float(np.mean(in_gaps)) if in_gaps else 0.0, "in_gap_max": float(np.max(in_gaps)) if in_gaps else 0.0,
        "wdur_cv": float(wdur.std() / (wdur.mean() + 1e-9)), "wdur_mean": float(wdur.mean()),
        "spikes_per_target": runs / max(1, len(spans)), "extra_spikes": runs - len(spans),
        "blank_mean": float(blank_p.mean()),
        "gop_mean": float(g.mean()), "gop_min": float(g.min()), "gop_std": float(g.std()),
        "gop_frac_low": float((g < -1).mean()), "mlp_mean": float(m.mean()),
    }
    if wav is not None and len(wav) > 0:
        f0, rms = _f0_track(np.asarray(wav, dtype=np.float64))
        v = f0[f0 > 0]
        st = 12 * np.log2(v / np.median(v)) if len(v) > 4 else np.zeros(1)
        f.update({
            "f0_voiced_frac": float((f0 > 0).mean()) if len(f0) else 0.0,
            "f0_range_st": float(np.percentile(st, 95) - np.percentile(st, 5)) if len(v) > 4 else 0.0,
            "f0_std_st": float(st.std()),
            "f0_slope": float(np.polyfit(np.arange(len(st)), st, 1)[0]) if len(v) > 4 else 0.0,
            "f0_jump": float(np.abs(np.diff(st)).mean()) if len(v) > 5 else 0.0,
            "rms_cv": float(rms.std() / (rms.mean() + 1e-9)) if len(rms) else 0.0,
        })
    else:
        f.update({k: 0.0 for k in ("f0_voiced_frac", "f0_range_st", "f0_std_st", "f0_slope", "f0_jump", "rms_cv")})

    emb = {}
    sl = slice(first, last + 1)
    for L in HIDDEN_LAYERS:
        if hidden is not None and L in hidden:
            H = np.asarray(hidden[L][sl], dtype=np.float32)
            emb[f"mean_h{L}"] = H.mean(axis=0)
            emb[f"std_h{L}"] = H.std(axis=0)
    return f, emb


SCALARS = ["dur_s", "speech_s", "n_words", "n_targets", "lead_sil_s", "trail_sil_s", "rate_tps", "rate_wps",
           "gap_mean", "gap_max", "gap_std", "n_pause_03", "n_pause_06", "pause_ratio", "in_gap_mean", "in_gap_max",
           "wdur_cv", "wdur_mean", "spikes_per_target", "extra_spikes", "blank_mean",
           "gop_mean", "gop_min", "gop_std", "gop_frac_low", "mlp_mean",
           "f0_voiced_frac", "f0_range_st", "f0_std_st", "f0_slope", "f0_jump", "rms_cv"]


EMB_KEYS = [f"{s}_h{L}" for L in HIDDEN_LAYERS for s in ("mean", "std")]


class UttHead:
    """Scalars + PCA of pooled hidden states -> GBM, averaged with a ridge
    on the same inputs (the two make different errors)."""

    def __init__(self, emb_keys=EMB_KEYS, n_pca=24, use_scalars=True, w_gbm=0.5, seed=0):
        self.emb_keys, self.n_pca, self.use_scalars, self.w_gbm, self.seed = list(emb_keys), n_pca, use_scalars, w_gbm, seed

    def _X(self, df, emb, fit=False):
        parts = [df[SCALARS].to_numpy(np.float64)] if self.use_scalars else []
        for k in self.emb_keys:
            E = np.asarray(emb[k], dtype=np.float32)
            if fit:
                from sklearn.decomposition import PCA
                self.pcas[k] = PCA(self.n_pca, random_state=self.seed).fit(E)
            parts.append(self.pcas[k].transform(E))
        return np.hstack(parts)

    def fit(self, df, emb, y):
        self.pcas = {}
        X = self._X(df, emb, fit=True)
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.gbm = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.04, max_leaf_nodes=15,
                                                 min_samples_leaf=30, l2_regularization=1.0, random_state=self.seed).fit(X, y)
        self.ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 16))).fit(X, y)
        return self

    def predict(self, df, emb):
        X = self._X(df, emb)
        p = self.w_gbm * self.gbm.predict(X) + (1 - self.w_gbm) * self.ridge.predict(X)
        return np.clip(p, 0, 10)
