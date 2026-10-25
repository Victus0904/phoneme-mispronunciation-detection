"""v2 inference: (waveform, target sentence) -> per-phone scores, with
word/phone timestamps for playback. Used by server.py (the local app).

Exactly the evaluated v2 pipeline: frozen wav2vec2 (posteriors + hidden
states in one forward pass) -> CTC forced alignment -> v2 features
(src/features_v2.py) -> trained head bundle (cache/v2/head_v2.joblib,
written by scripts/train_v2.py). No fine-tuning, no LLM.
"""
import os
import time

import numpy as np
import pandas as pd
import torch

from src.acoustic import load_model, trim_bounds, vocab_maps
from src.features_v2 import target_inventory, utterance_features
from src.phone_mapping import ARPABET_TO_IPA
from src.text import text_to_words
from src.tips import get_example, get_tip

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BUNDLE = os.path.join(ROOT, "models", "head_v2.joblib")
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
        self.model, self.processor = load_model()
        self.sym2idx, self.idx2sym = vocab_maps(self.processor)
        self.cands = target_inventory(self.sym2idx)
        self.g2p = G2p()

    def info(self):
        return {"version": self.bundle.get("version", "v2"), "metrics": self.bundle.get("metrics", {}),
                "flag_threshold": self.flag_threshold, "description": self.bundle.get("description", "")}

    @torch.no_grad()
    def _forward(self, wav):
        inputs = self.processor(wav, sampling_rate=TARGET_SR, return_tensors="pt")
        out = self.model(inputs.input_values, output_hidden_states=bool(self.layers))
        logpost = torch.log_softmax(out.logits[0], dim=-1).numpy()
        hidden = {L: out.hidden_states[L][0].numpy() for L in self.layers}
        return logpost, hidden

    def score(self, wav, sr, sentence):
        t_start = time.perf_counter()
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
        if not sentence or not sentence.strip():
            raise ScoringError("Please type the sentence you read.")
        if float(np.sqrt(np.mean(wav.astype(np.float64) ** 2))) < SILENCE_RMS_THRESHOLD:
            raise ScoringError("The recording seems silent -- check your microphone and try again.")

        s0, s1 = trim_bounds(wav, TARGET_SR)
        trimmed = wav[s0:s1]
        if len(trimmed) < 1600:
            raise ScoringError("Recording too short or too quiet -- please try again.")

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
                                           hidden or None)
        except KeyError as e:
            raise ScoringError(f"Unsupported sound in that sentence ({e}) -- please try a different sentence.")
        df = pd.DataFrame(rows)
        df["uid"] = "live"
        pred = self.head.predict(df, emb)

        frame_s = len(trimmed) / TARGET_SR / logpost.shape[0]
        off = s0 / TARGET_SR
        out_words = [{"text": w["text"], "phones": [], "start_s": None, "end_s": None} for w in words]
        for r, p in zip(rows, pred):
            w = out_words[r["word_index"]]
            flagged = bool(p < self.flag_threshold)
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
        scores = np.array([p["score"] for _, _, p in flat])
        return {
            "ok": True, "sentence": sentence.strip(), "duration_s": round(dur, 2),
            "processing_s": round(time.perf_counter() - t_start, 3),
            "overall": {"score_100": round(float(scores.mean() / 2 * 100), 1), "n_phones": len(flat),
                        "n_flagged": int(sum(p["flagged"] for _, _, p in flat))},
            "words": out_words, "worst": {"w": wi, "p": pi},
        }
