"""Build the v2 feature table from cache/v2/utt/*.npz (no model inference).

Writes:
  cache/v2/features.parquet   one row per (utterance, ARPAbet phone)
  cache/v2/emb_h{L}.npy       [n_rows, 1024] float16 pooled hidden states,
                              row-aligned with features.parquet

Usage: python scripts/build_features_v2.py [--limit N]
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from transformers import AutoProcessor

from src.acoustic import MODEL_ID, vocab_maps
from src.data import load_speechocean762
from src.features_v2 import HIDDEN_LAYERS, target_inventory, utterance_features

V2 = os.path.join(ROOT, "cache", "v2")


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    sym2idx, idx2sym = vocab_maps(processor)
    cands = target_inventory(sym2idx)
    print(f"GOP-SF candidate set: {len(cands)} symbols")

    ds = load_speechocean762(decode_audio=False)
    keys = list(HIDDEN_LAYERS) + ["sfvec", "postvec"]
    rows, embs = [], {k: [] for k in keys}
    t0, n = time.time(), 0
    for split in ("train", "test"):
        meta = ds[split].remove_columns("audio")
        for idx in range(len(meta)):
            path = os.path.join(V2, "utt", f"{split}_{idx}.npz")
            if not os.path.exists(path):
                continue
            if limit and n >= limit:
                break
            z = np.load(path)
            ex = meta[idx]
            hidden = {L: z[f"h{L}"] for L in HIDDEN_LAYERS}
            r, e = utterance_features(z["logpost"], ex["words"], sym2idx, idx2sym, cands, hidden)
            uid = f"{split}_{idx}"
            for row in r:
                row.update(uid=uid, split=split, speaker=ex["speaker"], gender=ex["gender"],
                           age=ex["age"], utt_accuracy=ex["accuracy"])
            rows.extend(r)
            for k in keys:
                embs[k].append(e[k])
            n += 1
            if n % 250 == 0:
                print(f"{n} utterances, {len(rows)} rows, {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    suffix = "" if not limit else f"_limit{limit}"
    df.to_parquet(os.path.join(V2, f"features{suffix}.parquet"), index=False)
    for k in keys:
        name = f"h{k}" if isinstance(k, int) else k
        np.save(os.path.join(V2, f"emb_{name}{suffix}.npy"), np.concatenate(embs[k]))
    print(f"wrote {len(df)} rows from {n} utterances in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
