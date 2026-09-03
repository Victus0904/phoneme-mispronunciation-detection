"""Gradio demo: record audio + type the target sentence -> per-phone
predicted scores, a below-threshold flag per phone, and an articulatory
coaching tip for the worst-scoring phone.

Reuses the exact same pipeline as training (src/acoustic.py -> src/align.py
-> src/features.py) plus g2p_en to turn an arbitrary typed sentence into
the ARPAbet word/phone structure speechocean762 supplies natively. The
trained head (cache/head_model.joblib, saved by src/train_head.py) scores
each phone; there is no fine-tuning, no LLM anywhere in this file.
"""
import os
import re

import numpy as np
import gradio as gr

from src.acoustic import load_model, get_log_posteriors, trim_silence, vocab_maps
from src.align import align
from src.features import build_utterance_targets, compute_gop_features
from src.tips import get_tip

FLAG_THRESHOLD = 1.5  # predicted score (0-2 scale) below which a phone is flagged
NUMERIC_FEATURES = ["mean_log_posterior", "margin", "entropy", "duration_frames"]
CATEGORICAL_FEATURES = ["top_competitor"]

_state = {}


def _lazy_init():
    if _state:
        return _state
    import joblib
    from g2p_en import G2p

    model, processor = load_model()
    sym2idx, idx2sym = vocab_maps(processor)
    head_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "head_model.joblib")
    head_model = joblib.load(head_path)

    _state.update({
        "model": model, "processor": processor,
        "sym2idx": sym2idx, "idx2sym": idx2sym,
        "blank_idx": sym2idx["<pad>"],
        "head_model": head_model,
        "g2p": G2p(),
    })
    return _state


def resample_linear(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wav
    n_target = int(round(len(wav) * target_sr / orig_sr))
    x_old = np.linspace(0, 1, len(wav), endpoint=False)
    x_new = np.linspace(0, 1, n_target, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)


def text_to_words(text: str, g2p):
    phones_flat = g2p(text)
    groups, current = [], []
    for tok in phones_flat:
        if tok == " ":
            if current:
                groups.append(current)
                current = []
        elif re.match(r"^[A-Z]+[0-2]?$", tok):
            current.append(tok)
    if current:
        groups.append(current)

    word_texts = re.findall(r"[A-Za-z']+", text)
    words = []
    for w, phones in zip(word_texts, groups):
        if phones:
            words.append({"text": w.upper(), "phones": phones})
    return words


def score_pronunciation(audio, sentence):
    if audio is None:
        return "Please record or upload audio first.", None, ""
    if not sentence or not sentence.strip():
        return "Please type the target sentence.", None, ""

    state = _lazy_init()
    sr, wav = audio

    wav = np.asarray(wav)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if np.issubdtype(wav.dtype, np.integer):
        wav = wav.astype(np.float32) / np.iinfo(wav.dtype).max
    else:
        wav = wav.astype(np.float32)

    wav16k = resample_linear(wav, sr, 16000)
    trimmed = trim_silence(wav16k, 16000)
    if len(trimmed) < 1600:  # < 0.1s
        return "Recording too short or too quiet -- please try again.", None, ""

    words = text_to_words(sentence, state["g2p"])
    if not words:
        return "Couldn't find any recognizable words in that sentence.", None, ""

    targets = build_utterance_targets({"words": words})
    target_vocab_indices = [state["sym2idx"][t["ipa"]] for t in targets]

    full_logpost = get_log_posteriors(trimmed, 16000, state["model"], state["processor"])
    if full_logpost.shape[0] < len(targets):
        return ("Recording is too short for that many words/phones -- "
                "please speak a little slower or record more audio.", None, "")

    boundaries = align(full_logpost, target_vocab_indices, state["blank_idx"])
    rows = compute_gop_features(full_logpost, state["idx2sym"], targets, boundaries)

    import pandas as pd
    feat_df = pd.DataFrame(rows)[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    predicted = state["head_model"].predict(feat_df)
    predicted = np.clip(predicted, 0.0, 2.0)
    for r, p in zip(rows, predicted):
        r["predicted_score"] = float(p)

    table = [[r["arpa_phone"], round(r["predicted_score"], 2),
              "⚠ below threshold" if r["predicted_score"] < FLAG_THRESHOLD else ""]
             for r in rows]

    worst = min(rows, key=lambda r: r["predicted_score"])
    tip = get_tip(worst["arpa_phone"])
    overall = float(np.mean(predicted))
    n_flagged = sum(1 for r in rows if r["predicted_score"] < FLAG_THRESHOLD)

    summary = (f"Overall score: {overall:.2f} / 2.0   "
               f"({n_flagged}/{len(rows)} phones flagged below {FLAG_THRESHOLD})\n\n"
               f"Worst phone: {worst['arpa_phone']} (predicted {worst['predicted_score']:.2f}/2.0)\n\n"
               f"Coaching tip: {tip}")

    return "", table, summary


with gr.Blocks(title="Pronunciation Scorer") as demo:
    gr.Markdown(
        "# Pronunciation scoring (research demo)\n"
        "Record yourself reading the target sentence. No fine-tuned model, no LLM -- "
        "frozen wav2vec2 phone posteriors + a small ridge/GBM head trained on "
        "speechocean762, exactly as evaluated in REPORT.md."
    )
    with gr.Row():
        audio_in = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Your recording")
        sentence_in = gr.Textbox(label="Target sentence", placeholder="e.g. We call it a bear.")
    btn = gr.Button("Score pronunciation", variant="primary")
    error_out = gr.Markdown()
    table_out = gr.Dataframe(headers=["phone", "predicted score", "flag"], label="Per-phone scores")
    summary_out = gr.Textbox(label="Summary + coaching tip", lines=6)

    btn.click(score_pronunciation, inputs=[audio_in, sentence_in], outputs=[error_out, table_out, summary_out])

if __name__ == "__main__":
    demo.launch()
