"""Gradio demo: record audio + type the target sentence -> per-phone
predicted scores, a below-threshold flag per phone, and an articulatory
coaching tip for the worst-scoring phone.

Reuses the exact same pipeline as training (src/acoustic.py -> src/align.py
-> src/features.py) plus g2p_en to turn an arbitrary typed sentence into
the ARPAbet word/phone structure speechocean762 supplies natively. The
trained head (cache/head_model.joblib, saved by src/train_head.py) scores
each phone; there is no fine-tuning, no LLM anywhere in this file.

`score_utterance()` is the seam: it's the only function that touches the
scoring pipeline, and it returns a plain structured result (`ScoringResult`).
The Gradio layer below it does no scoring of its own -- it only renders
that structure.
"""
import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import gradio as gr

from src.acoustic import load_model, get_log_posteriors, trim_silence, vocab_maps
from src.align import align
from src.features import build_utterance_targets, compute_gop_features
from src.phone_mapping import map_word_phones_to_ipa_targets
from src.tips import get_tip

FLAG_THRESHOLD = 1.5  # predicted score (0-2 scale) below which a phone is flagged
MAX_AUDIO_SECONDS = 15.0  # reject longer clips outright -- unbounded input is unbounded compute on a metered service
SILENCE_RMS_THRESHOLD = 1e-4  # below this, treat the recording as silent regardless of length
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


def _load_audio(audio):
    """Normalizes score_utterance()'s `audio` argument to (sr, wav) or None.

    Accepts either a (sr, wav_ndarray) pair (the direct/programmatic and
    Checkpoint 1-3 test path) or a filepath string (what the Gradio UI now
    passes, via gr.Audio(type="filepath")) -- decoded with `soundfile`
    rather than Gradio's own ffmpeg-based decoding, per Task 4: this is the
    one runtime place audio gets decoded from a file, and it deliberately
    depends only on libsndfile, not a system ffmpeg install. Browser
    microphone recordings arrive pre-transcoded to WAV by Gradio's
    in-browser (WASM) ffmpeg before upload, so this covers both sources;
    upload formats libsndfile can't read (e.g. mp3) will raise, which the
    caller turns into a clean error rather than a crash.
    """
    if audio is None:
        return None
    if isinstance(audio, str):
        import soundfile as sf
        wav, sr = sf.read(audio, dtype="float32", always_2d=False)
        return sr, wav
    return audio


def resample_linear(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wav
    n_target = int(round(len(wav) * target_sr / orig_sr))
    x_old = np.linspace(0, 1, len(wav), endpoint=False)
    x_new = np.linspace(0, 1, n_target, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)


@dataclass
class PhoneRecord:
    word_index: int          # index into the resolved word list, for grouping in the UI
    word_text: str
    arpa_phone: str          # source ARPAbet symbol (speechocean762 / g2p_en's inventory)
    ipa_symbol: str          # the acoustic model's target symbol this phone was aligned to
    predicted_score: float   # 0-2, GBM head output, clipped
    flagged: bool            # predicted_score < FLAG_THRESHOLD
    top_competitor: str      # highest-posterior non-target symbol in this phone's aligned span


@dataclass
class ScoringResult:
    ok: bool
    error: Optional[str] = None
    phones: List[PhoneRecord] = field(default_factory=list)
    worst_index: Optional[int] = None   # index into `phones`, None iff not ok
    worst_tip: Optional[str] = None
    processing_time_s: float = 0.0


def _error(msg: str) -> ScoringResult:
    return ScoringResult(ok=False, error=msg)


def text_to_words(text: str, g2p):
    """Sentence -> (words, unresolved).

    `words` is [{'text': str, 'phones': [ARPAbet...]}, ...] for every word
    g2p_en resolved cleanly. `unresolved` lists the raw word substrings it
    didn't: either the regex word count didn't match g2p's own tokenization
    (numerals, unusual punctuation -- can't safely pair them up positionally)
    or a specific word came back with zero phones. Returning this instead of
    silently dropping mismatches means the caller can reject the whole
    sentence explicitly, rather than quietly scoring a shorter sentence than
    the one the speaker was shown.
    """
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
    if len(word_texts) != len(groups):
        return [], word_texts  # tokenization mismatch -- can't safely zip these up

    words, unresolved = [], []
    for w, phones in zip(word_texts, groups):
        if phones:
            words.append({"text": w.upper(), "phones": phones})
        else:
            unresolved.append(w)
    return words, unresolved


def score_utterance(audio, sentence: str) -> ScoringResult:
    """audio: a (sample_rate, waveform) pair, a filepath string (as delivered
    by gr.Audio(type="filepath") -- decoded via soundfile, see _load_audio),
    or None. sentence: the target sentence text.

    Runs the full pipeline (acoustic model -> CTC forced alignment -> GOP
    features -> trained GBM head) exactly as evaluated in REPORT.md, and
    returns a structured result rather than raising on bad input. This is
    the single seam between the scoring pipeline and the UI.
    """
    if audio is None:
        return _error("Please record or upload audio first.")

    try:
        sr, wav = _load_audio(audio)
    except Exception:
        return _error(
            "Couldn't read that audio file -- try a WAV, FLAC, or OGG recording "
            "(some formats like mp3 aren't supported)."
        )
    wav = np.asarray(wav)
    if wav.size == 0:
        return _error("Recording is empty -- please try again.")

    duration_s = len(wav) / sr
    if duration_s > MAX_AUDIO_SECONDS:
        return _error(
            f"That recording is {duration_s:.1f}s long; this demo caps input at "
            f"{MAX_AUDIO_SECONDS:.0f}s (unbounded audio is unbounded compute on a "
            f"metered service). Please record or upload a shorter clip."
        )

    if not sentence or not sentence.strip():
        return _error("Please type the target sentence.")

    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if np.issubdtype(wav.dtype, np.integer):
        wav = wav.astype(np.float32) / np.iinfo(wav.dtype).max
    else:
        wav = wav.astype(np.float32)

    wav16k = resample_linear(wav, sr, 16000)
    rms = float(np.sqrt(np.mean(wav16k.astype(np.float64) ** 2))) if len(wav16k) else 0.0
    if rms < SILENCE_RMS_THRESHOLD:
        return _error("Recording appears to be silent -- please check your microphone and try again.")

    trimmed = trim_silence(wav16k, 16000)
    if len(trimmed) < 1600:  # < 0.1s of voiced audio
        return _error("Recording too short or too quiet -- please try again.")

    state = _lazy_init()

    words, unresolved = text_to_words(sentence, state["g2p"])
    if not words and unresolved:
        return _error(
            "Couldn't line up that sentence's words with G2P's output -- try "
            "plain words, without numerals or unusual punctuation."
        )
    if not words:
        return _error("Couldn't find any recognizable words in that sentence.")
    if unresolved:
        return _error(
            "Couldn't work out a pronunciation for: " + ", ".join(unresolved) +
            " -- try rephrasing without that word."
        )

    targets = build_utterance_targets({"words": words})
    unmapped = sorted({t["arpa"][0] for t in targets if t["ipa"] is None})
    if unmapped:
        return _error(
            "Some sounds in that sentence aren't in this demo's supported phone "
            "set (" + ", ".join(unmapped) + ") -- please try a different sentence."
        )

    try:
        target_vocab_indices = [state["sym2idx"][t["ipa"]] for t in targets]
    except KeyError as e:
        return _error(f"Unsupported sound in that sentence ({e}) -- please try a different sentence.")

    t0 = time.perf_counter()

    full_logpost = get_log_posteriors(trimmed, 16000, state["model"], state["processor"])
    if full_logpost.shape[0] < len(targets):
        return _error(
            "Recording is too short for that many words/phones -- please speak "
            "a little slower or record a bit more audio."
        )

    boundaries = align(full_logpost, target_vocab_indices, state["blank_idx"])
    rows = compute_gop_features(full_logpost, state["idx2sym"], targets, boundaries)

    import pandas as pd
    feat_df = pd.DataFrame(rows)[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    predicted = np.clip(state["head_model"].predict(feat_df), 0.0, 2.0)

    processing_time_s = time.perf_counter() - t0

    # `rows` is flat, in the exact order build_utterance_targets walked the
    # words (word by word, each target's arpa phone(s) in order). Re-derive
    # that same per-word row count here -- without changing features.py's
    # return shape -- so each row can be tagged with which word it belongs
    # to, for the UI's word-grouped phone strip.
    phones: List[PhoneRecord] = []
    row_i = 0
    for word_idx, w in enumerate(words):
        n_rows = sum(len(t["arpa"]) for t in map_word_phones_to_ipa_targets(w["phones"]))
        for _ in range(n_rows):
            r = rows[row_i]
            score = float(predicted[row_i])
            phones.append(PhoneRecord(
                word_index=word_idx,
                word_text=w["text"],
                arpa_phone=r["arpa_phone"],
                ipa_symbol=r["ipa_target"],
                predicted_score=score,
                flagged=score < FLAG_THRESHOLD,
                top_competitor=r["top_competitor"],
            ))
            row_i += 1

    worst_index = min(range(len(phones)), key=lambda i: phones[i].predicted_score)
    worst_tip = get_tip(phones[worst_index].arpa_phone)

    return ScoringResult(
        ok=True,
        phones=phones,
        worst_index=worst_index,
        worst_tip=worst_tip,
        processing_time_s=processing_time_s,
    )


# Load the model once here, at import time, rather than lazily on the first
# request (Task 4): on a metered container, the first user shouldn't pay for
# a multi-second cold load that every later request skips via _state's
# guard. _lazy_init() is idempotent, so this is safe to leave in
# score_utterance() too.
_lazy_init()


# --- Gradio layer -----------------------------------------------------
# Everything below renders the ScoringResult produced by score_utterance()
# above -- no scoring happens down here. The design follows REPORT.md
# Section 4's argument directly: at 0.63 precision, a binary right/wrong
# verdict is the wrong interface, so nothing here renders one. Flagged
# phones are "worth another take," never "wrong"; colour is a continuous
# ramp (never a threshold-based traffic light, and never red/green); the
# honesty footer with the measured precision/recall and its Mandarin-L1-only
# scope is permanent, not tucked into an accordion.

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPORT_PATH = os.path.join(_APP_DIR, "REPORT.md").replace("\\", "/")
EXAMPLES_DIR = os.path.join(_APP_DIR, "examples")
EXAMPLES_MANIFEST = os.path.join(EXAMPLES_DIR, "manifest.json")


def _load_examples():
    """Reads examples/manifest.json (added in Task 3) -> [[wav_path, sentence], ...]
    for gr.Examples. Missing manifest or missing wav files are silently
    skipped rather than crashing the UI -- this lets app.py run before
    Task 3's clips exist."""
    if not os.path.exists(EXAMPLES_MANIFEST):
        return []
    import json
    with open(EXAMPLES_MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)
    out = []
    for ex in manifest:
        wav_path = os.path.join(EXAMPLES_DIR, ex["wav"])
        if os.path.exists(wav_path):
            out.append([wav_path, ex["sentence"]])
    return out


EXAMPLES = _load_examples()


def _score_color(score: float) -> "tuple[str, str]":
    """Continuous single-hue (blue) lightness ramp, keyed to the 0-2
    predicted score: near-white at 2.0 (fades into the background -- nothing
    to see here), deep blue at 0.0 (stands out -- worth a look). Deliberately
    not red-to-green (unreadable for colour-blind viewers) and not a
    threshold -- every score gets its own point on the ramp, not a bucket.
    Returns (background, text) CSS colors with the text colour chosen for
    contrast against the background."""
    t = max(0.0, min(1.0, score / 2.0))
    lightness = 40 + t * 52  # 40% at score=0 .. 92% at score=2
    bg = f"hsl(210, 62%, {lightness:.0f}%)"
    fg = "#ffffff" if lightness < 62 else "#1a1a1a"
    return bg, fg


def _esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _render_phone_strip(result: ScoringResult) -> str:
    groups = []
    current_idx, current_text, current = None, None, []
    for i, p in enumerate(result.phones):
        if p.word_index != current_idx:
            if current:
                groups.append((current_text, current))
            current_idx, current_text, current = p.word_index, p.word_text, []
        current.append((i, p))
    if current:
        groups.append((current_text, current))

    word_html = []
    for word_text, phones in groups:
        chips = []
        for i, p in phones:
            bg, fg = _score_color(p.predicted_score)
            worst_cls = " phone-worst" if i == result.worst_index else ""
            note = "worth another take" if p.flagged else "solid"
            tooltip = f"{_esc(p.arpa_phone)} / {_esc(p.ipa_symbol)} -- {p.predicted_score:.2f}/2.0 ({note})"
            chips.append(
                f'<div class="phone-chip{worst_cls}" style="background:{bg};color:{fg};" title="{tooltip}">'
                f'<span class="phone-sym">{_esc(p.ipa_symbol)}</span>'
                f'<span class="phone-score">{p.predicted_score:.2f}</span>'
                f'</div>'
            )
        word_html.append(
            f'<div class="word-group"><div class="word-phones">{"".join(chips)}</div>'
            f'<div class="word-label">{_esc(word_text)}</div></div>'
        )
    return f'<div class="phone-strip">{"".join(word_html)}</div>'


def _render_diagnosis(result: ScoringResult) -> str:
    flagged = [(i, p) for i, p in enumerate(result.phones) if p.flagged]
    if not flagged:
        return '<div class="diagnosis-empty">Nothing flagged here -- no phone came back worth a second take.</div>'
    items = []
    for i, p in flagged:
        cls = ' class="diagnosis-worst"' if i == result.worst_index else ""
        items.append(
            f'<li{cls}><b>{_esc(p.word_text)}</b> -- expected /{_esc(p.ipa_symbol)}/ '
            f'(ARPAbet {_esc(p.arpa_phone)}), closest match /{_esc(p.top_competitor)}/ '
            f'<span class="diagnosis-score">({p.predicted_score:.2f}/2.0, worth another take)</span></li>'
        )
    return f'<div class="diagnosis-heading">Worth another take:</div><ul class="diagnosis-list">{"".join(items)}</ul>'


def _render_tip(result: ScoringResult) -> str:
    worst = result.phones[result.worst_index]
    return (
        f'<div class="tip-box"><b>Articulation tip</b> for /{_esc(worst.ipa_symbol)}/ in '
        f'{_esc(worst.word_text)}:<br>{_esc(result.worst_tip)}</div>'
    )


def _handle(audio, sentence):
    result = score_utterance(audio, sentence)
    if not result.ok:
        return (
            gr.update(value=f'<div class="error-box">{_esc(result.error)}</div>', visible=True),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
        )

    n_flagged = sum(1 for p in result.phones if p.flagged)
    meta = (
        f'<div class="meta-line">{len(result.phones)} phones scored &middot; '
        f'{n_flagged} worth another take &middot; processed in {result.processing_time_s:.2f}s</div>'
    )
    return (
        gr.update(value="", visible=False),
        gr.update(value=_render_phone_strip(result), visible=True),
        gr.update(value=_render_diagnosis(result), visible=True),
        gr.update(value=_render_tip(result), visible=True),
        gr.update(value=meta, visible=True),
    )


CUSTOM_CSS = """
.gradio-container { max-width: 900px !important; margin: auto; }
.phone-strip { display: flex; flex-wrap: wrap; gap: 16px; padding: 10px 0 4px; }
.word-group { display: flex; flex-direction: column; align-items: center; gap: 5px; }
.word-phones { display: flex; gap: 4px; }
.phone-chip {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  min-width: 42px; padding: 6px 8px; border-radius: 8px; cursor: default;
  transition: transform 0.12s ease;
}
.phone-chip:hover { transform: translateY(-2px); }
.phone-chip.phone-worst { outline: 2px solid #d9822b; outline-offset: 2px; }
.phone-sym { font-size: 18px; font-weight: 600; line-height: 1.15; }
.phone-score { font-size: 10px; opacity: 0.85; margin-top: 2px; }
.word-label { font-size: 11px; letter-spacing: 0.03em; color: #777; text-transform: uppercase; }
.diagnosis-heading { font-weight: 600; margin-top: 4px; }
.diagnosis-list { padding-left: 18px; margin: 4px 0; }
.diagnosis-list li { margin-bottom: 4px; }
.diagnosis-worst { font-weight: 600; }
.diagnosis-score { color: #777; font-size: 12px; }
.diagnosis-empty { color: #777; font-style: italic; }
.tip-box { background: #eef3f8; border-left: 3px solid #3478be; padding: 10px 14px; border-radius: 4px; margin-top: 10px; }
.error-box { background: #fdf1e6; border-left: 3px solid #d9822b; padding: 10px 14px; border-radius: 4px; }
.meta-line { color: #777; font-size: 12px; margin-top: 8px; }
.honesty-footer { border-top: 1px solid #ddd; margin-top: 26px; padding-top: 10px; font-size: 12px; color: #555; line-height: 1.55; }
"""

HONESTY_FOOTER = f"""
<div class="honesty-footer">
<b>How honest is this score?</b> Correlation with human raters: <b>r = 0.431</b> (Pearson,
held-out speaker-disjoint test set). At the shipped flagging threshold, a flagged phone is
right about <b>63% of the time</b> (precision &asymp; 0.63, recall &asymp; 0.10) -- it stays quiet
rather than guessing on phones it's unsure about. Trained and evaluated <b>only on Mandarin-L1
speakers</b> (speechocean762); accuracy for other native-language backgrounds is unmeasured.
Treat every score here as a <b>triage signal, not a verdict</b>. Full methodology and limitations:
<a href="/file={_REPORT_PATH}" target="_blank">REPORT.md</a>.
</div>
"""

with gr.Blocks(title="Pronunciation Scorer") as demo:
    gr.Markdown(
        "# Pronunciation scoring (research demo)\n"
        "Record or upload yourself reading the target sentence. No fine-tuned model, no LLM -- "
        "frozen wav2vec2 phone posteriors + a small GBM head trained on speechocean762, exactly "
        "as evaluated in REPORT.md."
    )
    with gr.Row():
        audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Your recording")
        sentence_in = gr.Textbox(label="Target sentence", placeholder="e.g. We call it a bear.")
    if EXAMPLES:
        gr.Examples(examples=EXAMPLES, inputs=[audio_in, sentence_in], label="Or try an example clip")
    btn = gr.Button("Score pronunciation", variant="primary")

    error_out = gr.HTML(visible=False)
    strip_out = gr.HTML(visible=False, label="Per-phone scores")
    diagnosis_out = gr.HTML(visible=False)
    tip_out = gr.HTML(visible=False)
    meta_out = gr.HTML(visible=False)

    gr.HTML(HONESTY_FOOTER)

    btn.click(
        _handle,
        inputs=[audio_in, sentence_in],
        outputs=[error_out, strip_out, diagnosis_out, tip_out, meta_out],
    )

# Gradio's queue holds long-lived SSE connections open; App Runner's request
# timeout may not tolerate that (see DEPLOY.md). USE_QUEUE=0 disables the
# queue without a rebuild -- the escape hatch if requests start hanging.
USE_QUEUE = os.environ.get("USE_QUEUE", "1").lower() not in ("0", "false", "no")

if __name__ == "__main__":
    if USE_QUEUE:
        demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=8080,
        ssr_mode=False,
        allowed_paths=[_APP_DIR],
        css=CUSTOM_CSS,
    )
