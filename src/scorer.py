"""v2 inference: (waveform, target sentence) -> per-phone scores, with
word/phone timestamps for playback. Used by server.py (the local app).

Exactly the evaluated v2 pipeline: frozen wav2vec2 (posteriors + hidden
states in one forward pass) -> CTC forced alignment -> v2 features
(src/features_v2.py) -> trained head bundle (models/head_v2.joblib,
written by scripts/train_v2.py). The same pass also feeds the fluency /
prosody scorers (src/utterance_v2.py, models/utt_v2.joblib, written by
scripts/train_fluency.py) and the minimal-pair comparison (which_word).
No fine-tuning, no LLM.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from src.acoustic import load_model, trim_bounds, vocab_maps
from src.features_v2 import target_inventory, utterance_features
from src.phone_mapping import ARPABET_TO_IPA, map_word_phones_to_ipa_targets
from src.utterance_v2 import HIDDEN_LAYERS as UTT_LAYERS, utterance_features as utt_features
from src.text import text_to_words
from src.tips import get_example, get_tip

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BUNDLE = os.path.join(ROOT, "models", "head_v2.joblib")
UTT_BUNDLE = os.path.join(ROOT, "models", "utt_v2.joblib")
STRICTNESS_FILE = os.path.join(ROOT, "models", "strictness.json")
TARGET_SR = 16000
MAX_AUDIO_SECONDS = 20.0
SILENCE_RMS_THRESHOLD = 1e-4


class ScoringError(Exception):
    """A user-facing problem with the input (message is shown as-is)."""


def _resample(wav, sr):
    if sr == TARGET_SR:
        return wav
    from scipy.signal import resample_poly
    g = np.gcd(int(sr), TARGET_SR)
    return resample_poly(wav, TARGET_SR // g, int(sr) // g).astype(np.float32)


class Scorer:
    def __init__(self, bundle_path=DEFAULT_BUNDLE):
        import joblib
        from g2p_en import G2p

        self.bundle = joblib.load(bundle_path)
        self.head = self.bundle["head"]
        # hidden-state layers the head reads (ints); "sfvec"/"postvec" are
        # computed from posteriors and always available
        self.layers = tuple(L for L in self.bundle["layers"] if isinstance(L, int))
        self.flag_threshold = float(self.bundle["flag_threshold"])
        # strictness levels: flag thresholds picked on train OOF (scripts/calibrate_strictness.py)
        self.strictness = {"standard": {"threshold": self.flag_threshold}}
        if os.path.exists(STRICTNESS_FILE):
            with open(STRICTNESS_FILE) as f:
                self.strictness = json.load(f)["levels"]
        self.utt = joblib.load(UTT_BUNDLE) if os.path.exists(UTT_BUNDLE) else None
        self.hidden_layers = tuple(sorted(set(self.layers) | (set(UTT_LAYERS) if self.utt else set())))
        self.model, self.processor = load_model()
        self.sym2idx, self.idx2sym = vocab_maps(self.processor)
        self.cands = target_inventory(self.sym2idx)
        self.g2p = G2p()

    def info(self):
        return {"version": self.bundle.get("version", "v2"), "metrics": self.bundle.get("metrics", {}),
                "flag_threshold": self.flag_threshold, "description": self.bundle.get("description", ""),
                "fluency_metrics": self.utt["metrics"] if self.utt else None,
                "strictness": {k: v.get("test") for k, v in self.strictness.items()}}

    @torch.no_grad()
    def _forward(self, wav):
        inputs = self.processor(wav, sampling_rate=TARGET_SR, return_tensors="pt")
        out = self.model(inputs.input_values, output_hidden_states=bool(self.hidden_layers))
        logpost = torch.log_softmax(out.logits[0], dim=-1).numpy()
        hidden = {L: out.hidden_states[L][0].numpy() for L in self.hidden_layers}
        return logpost, hidden

    def _prepare(self, wav, sr):
        """Validate + mono/float/16 kHz + silence-trim. Returns (wav16k, s0, s1)."""
        wav = np.asarray(wav)
        if wav.size == 0:
            raise ScoringError("Recording is empty -- please try again.")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if np.issubdtype(wav.dtype, np.integer):
            wav = wav.astype(np.float32) / np.iinfo(wav.dtype).max
        wav = _resample(wav.astype(np.float32), sr)
        dur = len(wav) / TARGET_SR
        if dur > MAX_AUDIO_SECONDS:
            raise ScoringError(f"That recording is {dur:.1f}s long; please keep it under {MAX_AUDIO_SECONDS:.0f}s "
                               "(one sentence at a time works best).")
        if float(np.sqrt(np.mean(wav.astype(np.float64) ** 2))) < SILENCE_RMS_THRESHOLD:
            raise ScoringError("The recording seems silent -- check your microphone and try again.")
        s0, s1 = trim_bounds(wav, TARGET_SR)
        if s1 - s0 < 1600:
            raise ScoringError("Recording too short or too quiet -- please try again.")
        return wav, s0, s1

    def score(self, wav, sr, sentence, strictness="standard"):
        t_start = time.perf_counter()
        if not sentence or not sentence.strip():
            raise ScoringError("Please type the sentence you read.")
        wav, s0, s1 = self._prepare(wav, sr)
        dur = len(wav) / TARGET_SR
        trimmed = wav[s0:s1]

        words, unresolved = text_to_words(sentence, self.g2p)
        if not words and unresolved:
            raise ScoringError("Couldn't split that sentence into words -- try plain words without numerals "
                               "or unusual punctuation.")
        if not words:
            raise ScoringError("Couldn't find any recognizable words in that sentence.")
        if unresolved:
            raise ScoringError("Couldn't work out a pronunciation for: " + ", ".join(unresolved))

        logpost, hidden = self._forward(trimmed)
        n_targets_upper = sum(len(w["phones"]) for w in words)
        if logpost.shape[0] < n_targets_upper:
            raise ScoringError("The recording is too short for that sentence -- did you read all of it?")

        try:
            rows, emb = utterance_features(logpost, words, self.sym2idx, self.idx2sym, self.cands,
                                           {L: hidden[L] for L in self.layers} or None)
        except KeyError as e:
            raise ScoringError(f"Unsupported sound in that sentence ({e}) -- please try a different sentence.")
        df = pd.DataFrame(rows)
        df["uid"] = "live"
        pred = self.head.predict(df, emb)

        if strictness not in self.strictness:
            strictness = "standard"
        threshold = float(self.strictness[strictness]["threshold"])
        frame_s = len(trimmed) / TARGET_SR / logpost.shape[0]
        off = s0 / TARGET_SR
        out_words = [{"text": w["text"], "phones": [], "start_s": None, "end_s": None} for w in words]
        for r, p in zip(rows, pred):
            w = out_words[r["word_index"]]
            flagged = bool(p < threshold)
            st, en = off + r["start_frame"] * frame_s, off + (r["end_frame"] + 1) * frame_s
            w["phones"].append({
                "arpa": r["arpa_phone"],
                # display each ARPAbet phone's own symbol (a fused vowel+R target
                # would otherwise show the same rhotic symbol twice)
                "ipa": ARPABET_TO_IPA.get(r["arpa_phone"], r["ipa_target"]), "aligned_ipa": r["ipa_target"], "score": round(float(p), 3),
                "flagged": flagged,
                "heard_as": (r["sf_best_sub"] if r["sf_best_sub"] != r["ipa_target"] and not r["fused"] else None) if flagged else None,
                "start_s": round(st, 3), "end_s": round(en, 3),
                "tip": get_tip(r["arpa_phone"]), "example": get_example(r["arpa_phone"]),
            })
            w["start_s"] = st if w["start_s"] is None else min(w["start_s"], st)
            w["end_s"] = en if w["end_s"] is None else max(w["end_s"], en)

        # word spans from 1-frame CTC spikes are tight; widen each word to
        # meet its neighbours halfway so word playback sounds natural
        for i, w in enumerate(out_words):
            prev_end = out_words[i - 1]["end_s"] if i > 0 else off
            next_start = out_words[i + 1]["start_s"] if i + 1 < len(out_words) else (s1 / TARGET_SR)
            w["start_s"] = round(max(off, (prev_end + w["start_s"]) / 2 if i > 0 else max(off, w["start_s"] - 0.1)), 3)
            w["end_s"] = round((w["end_s"] + next_start) / 2 if i + 1 < len(out_words) else min(s1 / TARGET_SR, w["end_s"] + 0.15), 3)
            w["score"] = round(float(np.mean([p["score"] for p in w["phones"]])), 3)

        flat = [(wi, pi, p) for wi, w in enumerate(out_words) for pi, p in enumerate(w["phones"])]
        wi, pi, _ = min(flat, key=lambda x: x[2]["score"])
        n_flagged = int(sum(p["flagged"] for _, _, p in flat))
        return {
            "ok": True, "sentence": sentence.strip(), "duration_s": round(dur, 2),
            "processing_s": round(time.perf_counter() - t_start, 3),
            # overall = share of sounds on target at the chosen strictness, so a
            # stricter level gives a lower, more demanding score
            "overall": {"score_100": round(100.0 * (1 - n_flagged / len(flat)), 1), "n_phones": len(flat),
                        "n_flagged": n_flagged, "strictness": strictness},
            "words": out_words, "worst": {"w": wi, "p": pi},
            "fluency": self._fluency(rows, logpost, hidden, trimmed),
        }

    def _fluency(self, rows, logpost, hidden, trimmed):
        """Fluency + prosody (0-10, trained on expert ratings) plus the raw
        timing/pitch measurements behind them, shown as-is."""
        if not self.utt:
            return None
        f, e = utt_features(rows, logpost, {L: hidden[L] for L in UTT_LAYERS}, len(trimmed), trimmed,
                            self.sym2idx["<pad>"])
        df = pd.DataFrame([f])
        emb = {k: v[None, :] for k, v in e.items()}
        out = {t: round(float(self.utt["heads"][t].predict(df, emb)[0]), 1) for t in self.utt["targets"]}
        return {"fluency_10": out["fluency"], "prosody_10": out["prosodic"],
                "words_per_min": round(f["rate_wps"] * 60), "long_pauses": int(f["n_pause_06"]),
                "short_pauses": int(f["n_pause_03"] - f["n_pause_06"]), "longest_pause_s": round(f["gap_max"], 2),
                "pitch_range_semitones": round(f["f0_range_st"], 1), "extra_sounds": max(0, int(f["extra_spikes"]))}

    # ------------------------------------------------------------------ minimal pairs
    def check_pair(self, wav, sr, set_id, pair_index, target):
        """Minimal-pair drill: did the recording sound more like `target` or
        like the other word of the pair? Compares the two words' full CTC
        sequence likelihoods (the GOP-SF machinery -- no alignment), minus a
        per-contrast offset calibrated on synthetic voices
        (models/pairs_calibration.json, scripts/check_pairs.py). Inside the
        +/- margin the answer is "unsure" rather than a guess."""
        import json
        from src.pairs import PAIR_SETS
        st = next((x for x in PAIR_SETS if x["id"] == set_id), None)
        if st is None or not (0 <= pair_index < len(st["pairs"])):
            raise ScoringError("Unknown word pair.")
        a, b = st["pairs"][pair_index]
        if target not in (a, b):
            raise ScoringError("Unknown target word.")
        if not hasattr(self, "_pair_cal"):
            path = os.path.join(ROOT, "models", "pairs_calibration.json")
            self._pair_cal = json.load(open(path)) if os.path.exists(path) else {"bias_ab": {}, "margin": 1.0}
        d_ab = self._pair_llr(wav, sr, a, b) - self._pair_cal["bias_ab"].get(set_id, 0.0)
        margin = self._pair_cal["margin"]
        heard = a if d_ab > margin else b if d_ab < -margin else None
        d_target = d_ab if target == a else -d_ab
        return {"ok": True, "a": a, "b": b, "target": target, "heard": heard, "correct": heard == target,
                "confidence": round(float(1 / (1 + np.exp(-abs(d_ab)))), 3), "margin_llr": round(float(d_target), 2)}

    def _pair_llr(self, wav, sr, word_a, word_b):
        """log P(audio | word A's phones) - log P(audio | word B's phones)."""
        wav, s0, s1 = self._prepare(wav, sr)
        seqs = []
        for w in (word_a, word_b):
            words, unresolved = text_to_words(w, self.g2p)
            if not words or unresolved or len(words) != 1:
                raise ScoringError(f"Couldn't work out a pronunciation for '{w}'.")
            targets = map_word_phones_to_ipa_targets(words[0]["phones"])
            if any(t["ipa"] not in self.sym2idx for t in targets):
                raise ScoringError(f"'{w}' contains a sound this model can't check.")
            seqs.append([self.sym2idx[t["ipa"]] for t in targets])
        if seqs[0] == seqs[1]:
            raise ScoringError("Those two words sound the same to this model -- try a different pair.")
        with torch.no_grad():
            inputs = self.processor(wav[s0:s1], sampling_rate=TARGET_SR, return_tensors="pt")
            logpost = torch.log_softmax(self.model(inputs.input_values).logits[0], dim=-1).numpy()
        blank = self.sym2idx["<pad>"]
        cols = sorted({blank} | set(seqs[0]) | set(seqs[1]) | {self.sym2idx[c] for c in self.cands})
        lp = torch.from_numpy(np.ascontiguousarray(logpost[:, cols], dtype=np.float32))
        local = {c: i for i, c in enumerate(cols)}
        T = lp.shape[0]
        ll = []
        for idx in seqs:
            tg = torch.tensor([[local[i] for i in idx]])
            loss = torch.nn.functional.ctc_loss(lp.unsqueeze(1), tg, torch.tensor([T]), torch.tensor([len(idx)]),
                                                blank=local[blank], reduction="sum", zero_infinity=True)
            ll.append(-float(loss))
        return float(np.clip(ll[0] - ll[1], -50, 50))
