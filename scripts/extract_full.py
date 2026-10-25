"""Full-corpus frozen-model extraction (v2 pipeline).

For every utterance in speechocean762 (2,500 train + 2,500 test, the
native speaker-disjoint split), runs the frozen acoustic model ONCE and
caches to cache/v2/utt/<split>_<idx>.npz:

  logpost  [T, V]   float32  full-vocab log_softmax (blank included)
  h{L}     [T, D]   float16  hidden states of transformer layer L, for the
                             layers in HIDDEN_LAYERS (frozen -- read-only
                             representations, nothing is fine-tuned)

Everything downstream (alignment, GOP / alignment-free GOP features,
embedding pooling, head training) reads only from this cache, so feature
experiments never need model inference again. Resumable: existing files
are skipped.

Usage: python scripts/extract_full.py [--threads N]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.acoustic import load_model, trim_silence
from src.data import load_speechocean762

HIDDEN_LAYERS = (8, 16, 24)
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "v2", "utt")


@torch.no_grad()
def extract(waveform, sr, model, processor):
    inputs = processor(waveform, sampling_rate=sr, return_tensors="pt")
    out = model(inputs.input_values, output_hidden_states=True)
    logpost = torch.log_softmax(out.logits[0], dim=-1).numpy().astype(np.float32)
    hidden = {f"h{L}": out.hidden_states[L][0].numpy().astype(np.float16) for L in HIDDEN_LAYERS}
    return logpost, hidden


def main():
    threads = 6
    if "--threads" in sys.argv:
        threads = int(sys.argv[sys.argv.index("--threads") + 1])
    torch.set_num_threads(threads)
    os.makedirs(OUT_DIR, exist_ok=True)

    ds = load_speechocean762(decode_audio=True)
    model, processor = load_model()

    total = sum(len(ds[s]) for s in ("train", "test"))
    done, audio_s, t0 = 0, 0.0, time.time()
    for split in ("train", "test"):
        for idx in range(len(ds[split])):
            path = os.path.join(OUT_DIR, f"{split}_{idx}.npz")
            done += 1
            if os.path.exists(path):
                continue
            ex = ds[split][idx]
            wav = ex["audio"]["array"].astype(np.float32)
            sr = ex["audio"]["sampling_rate"]
            trimmed = trim_silence(wav, sr)
            logpost, hidden = extract(trimmed, sr, model, processor)
            tmp = path + ".tmp.npz"
            np.savez(tmp, logpost=logpost, n_samples=np.int64(len(trimmed)), **hidden)
            os.replace(tmp, path)
            audio_s += len(trimmed) / sr
            if done % 50 == 0:
                el = time.time() - t0
                print(f"{done}/{total}  audio={audio_s:.0f}s wall={el:.0f}s  rtf={audio_s/max(el,1e-9):.2f}x",
                      flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
