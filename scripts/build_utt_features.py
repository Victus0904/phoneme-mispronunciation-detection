"""Utterance-level fluency/prosody features for all 5,000 utterances, from
the v2 cache (cache/v2/utt/*.npz + cache/v2/features.parquet) plus the
trimmed waveform for pitch. No model inference.

Writes cache/v2/utt_features.parquet (scalars + labels) and
cache/v2/uttemb_<stat>_h<L>.npy (row-aligned pooled hidden states).
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from transformers import AutoProcessor

from src.acoustic import MODEL_ID, trim_silence, vocab_maps
from src.data import load_speechocean762
from src.utterance_v2 import HIDDEN_LAYERS, utterance_features

V2 = os.path.join(ROOT, "cache", "v2")


def main():
    sym2idx, _ = vocab_maps(AutoProcessor.from_pretrained(MODEL_ID))
    blank = sym2idx["<pad>"]
    phones = pd.read_parquet(os.path.join(V2, "features.parquet"))
    by_uid = {u: g.to_dict("records") for u, g in phones.groupby("uid", sort=False)}
    ds = load_speechocean762(decode_audio=True)
    out, embs, t0 = [], {}, time.time()
    for split in ("train", "test"):
        for idx in range(len(ds[split])):
            uid = f"{split}_{idx}"
            if uid not in by_uid:
                continue
            ex = ds[split][idx]
            z = np.load(os.path.join(V2, "utt", f"{uid}.npz"))
            wav = trim_silence(ex["audio"]["array"].astype(np.float32), ex["audio"]["sampling_rate"])
            f, e = utterance_features(by_uid[uid], z["logpost"], {L: z[f"h{L}"] for L in HIDDEN_LAYERS},
                                      int(z["n_samples"]), wav, blank)
            f.update(uid=uid, split=split, speaker=ex["speaker"], fluency=ex["fluency"], prosodic=ex["prosodic"],
                     accuracy=ex["accuracy"], total=ex["total"])
            out.append(f)
            for k, v in e.items():
                embs.setdefault(k, []).append(v.astype(np.float16))
            if len(out) % 500 == 0:
                print(f"{len(out)} utterances, {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(out).to_parquet(os.path.join(V2, "utt_features.parquet"), index=False)
    for k, v in embs.items():
        np.save(os.path.join(V2, f"uttemb_{k}.npy"), np.stack(v))
    print(f"wrote {len(out)} utterances in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
