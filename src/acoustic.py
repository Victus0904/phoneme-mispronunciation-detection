"""Frozen acoustic model loading + frame x phone log-posterior extraction.

Model: facebook/wav2vec2-lv-60-espeak-cv-ft (fallback per src/phone_mapping.py
docstring -- the originally-specified charsiu/en_w2v2_fc_10ms has no usable
label set). This is a standard Wav2Vec2ForCTC; "posteriors" here means the
per-frame log_softmax(logits) over its 392-symbol IPA/eSpeak vocabulary,
taken *before* any CTC collapsing/decoding -- we need the raw per-frame
distribution for GOP, not a decoded string.
"""
import glob
import os
import sys


def _register_espeak_env():
    """The model's tokenizer (Wav2Vec2PhonemeCTCTokenizer) eagerly
    initializes a phonemizer espeak backend on load, even though we never
    call it to phonemize text (audio -> posteriors only). On Windows the
    `phonemizer` package looks for a plain `espeak` binary via PATH by
    default; point it at the actual eSpeak NG install (same PATH-not-yet-
    refreshed problem as the FFmpeg fix in src/data.py)."""
    if sys.platform != "win32":
        return
    if os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
        return

    candidates = ["C:\\Program Files\\eSpeak NG"]
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates += glob.glob(
            os.path.join(local_app_data, "Microsoft", "WinGet", "Packages",
                         "eSpeak-NG.eSpeak-NG_*", "**"), recursive=False)

    for d in candidates:
        dll = os.path.join(d, "libespeak-ng.dll")
        exe = os.path.join(d, "espeak-ng.exe")
        if os.path.isfile(dll):
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = dll
            if os.path.isfile(exe):
                os.environ["PHONEMIZER_ESPEAK_PATH"] = exe
            return


_register_espeak_env()

import numpy as np
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
SEED = 42

torch.manual_seed(SEED)

_model = None
_processor = None


def load_model():
    global _model, _processor
    if _model is None:
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
        _model.eval()
    return _model, _processor


def trim_silence(waveform: np.ndarray, sr: int, frame_ms: float = 20.0,
                  rel_threshold: float = 0.03, pad_ms: float = 80.0) -> np.ndarray:
    """Simple energy-based leading/trailing silence trim.

    Why: the DP alignment (align.py) assigns every frame in the audio to one
    of the canonical content phones -- there's no silence class. Without
    trimming, leading/trailing silence gets force-assigned to the first/last
    phone's segment, inflating its duration and corrupting its GOP features.
    This is a standard VAD-lite trim (RMS energy vs. a fraction of the
    utterance's peak energy), not a model -- keeps the alignment step itself
    simple, per the "avoid alignment tooling" instruction.
    """
    frame_len = int(sr * frame_ms / 1000)
    if frame_len < 1 or len(waveform) < frame_len:
        return waveform

    n_frames = len(waveform) // frame_len
    trimmed = waveform[: n_frames * frame_len].reshape(n_frames, frame_len)
    energy = np.sqrt((trimmed ** 2).mean(axis=1))
    threshold = energy.max() * rel_threshold

    voiced = np.where(energy > threshold)[0]
    if len(voiced) == 0:
        return waveform  # all "silence" by this measure -- don't trim, let it fail loudly downstream

    pad_frames = int(pad_ms / frame_ms)
    start = max(0, voiced[0] - pad_frames) * frame_len
    end = min(n_frames, voiced[-1] + 1 + pad_frames) * frame_len
    return waveform[start:end]


SPECIAL_SYMBOLS = {"<pad>", "<s>", "</s>", "<unk>"}


def get_log_posteriors(waveform: np.ndarray, sr: int, model=None, processor=None) -> np.ndarray:
    """Returns [T, V] numpy array of log_softmax(logits): frame x IPA-phone
    log-posteriors, over the model's FULL vocabulary (including the CTC
    blank/<pad> and other special tokens)."""
    if model is None or processor is None:
        model, processor = load_model()

    inputs = processor(waveform, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits[0]  # [T, V]
    return torch.log_softmax(logits, dim=-1).numpy()


def vocab_maps(processor):
    vocab = processor.tokenizer.get_vocab()
    sym2idx = dict(vocab)
    idx2sym = {v: k for k, v in vocab.items()}
    return sym2idx, idx2sym


def content_log_posteriors(full_log_posteriors: np.ndarray, idx2sym: dict):
    """Drop the CTC blank (<pad>) and other special tokens, then
    renormalize each frame's distribution over the remaining phone symbols.

    Why: this is a CTC model, so its raw per-frame softmax puts ~99%+ of
    the mass on <pad> (blank) at almost every frame -- CTC models are
    "peaky" by design, only spiking the real phone briefly. Feeding the raw
    full-vocab log-posteriors into a plain monotonic segmental alignment
    (no explicit blank modeling) starves every phone target of a
    meaningful score except at rare spike frames, and the DP ends up
    dumping large blank-dominated stretches onto whichever phone target
    happens to look (locally, relatively) least bad -- producing degenerate
    durations. Renormalizing over content phones only restores a
    meaningful, comparable distribution at every frame and lets the
    original simple (no-blank) monotonic DP in align.py work as specified.

    Returns: (content_log_posteriors [T, V_content], content_idx2sym dict
    mapping new local index -> symbol, content_sym2idx dict).
    """
    content_indices = [i for i in range(full_log_posteriors.shape[1]) if idx2sym[i] not in SPECIAL_SYMBOLS]
    sub = full_log_posteriors[:, content_indices]
    sub = sub - np.log(np.exp(sub).sum(axis=1, keepdims=True))  # renormalize (log-sum-exp)

    content_idx2sym = {new_i: idx2sym[old_i] for new_i, old_i in enumerate(content_indices)}
    content_sym2idx = {sym: new_i for new_i, sym in content_idx2sym.items()}
    return sub, content_idx2sym, content_sym2idx


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, ".")
    from src.data import load_speechocean762

    ds = load_speechocean762(decode_audio=True)
    example = ds["train"][0]
    audio = example["audio"]
    waveform = audio["array"].astype(np.float32)
    sr = audio["sampling_rate"]

    model, processor = load_model()
    trimmed = trim_silence(waveform, sr)
    print(f"raw waveform: {len(waveform)} samples ({len(waveform)/sr:.2f}s)")
    print(f"trimmed waveform: {len(trimmed)} samples ({len(trimmed)/sr:.2f}s)")

    logpost = get_log_posteriors(trimmed, sr, model, processor)
    print(f"log-posterior matrix shape: {logpost.shape}  (T frames x V={logpost.shape[1]} vocab)")
    print(f"effective frame rate: {len(trimmed)/sr/logpost.shape[0]*1000:.2f} ms/frame")

    sym2idx, idx2sym = vocab_maps(processor)
    print(f"vocab size: {len(sym2idx)}")
