"""Sanity check for minimal-pair drills: synthesize both words of every pair
(e.g. with the OS text-to-speech voices) into <dir>/<voice>_<word>.wav and
check the raw (uncalibrated) pair comparison picks the word that was
actually said; the dump feeds the per-contrast calibration in
models/pairs_calibration.json (fit offsets so LL(A)-LL(B) is centred between
the two classes; leave-one-voice-out estimates are recorded there). Synthetic speech is
clean and native-like, so this verifies the mechanism -- it is NOT an
accuracy estimate for real learners (no labelled minimal-pair data exists in
speechocean762).

Usage: python scripts/check_pairs.py <wav_dir> [dump.json]
"""
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import soundfile as sf

from src.pairs import PAIR_SETS
from src.scorer import Scorer


def main():
    d = sys.argv[1]
    voices = sorted({os.path.basename(p).split("_")[0] for p in glob.glob(os.path.join(d, "*_*.wav"))})
    s = Scorer()
    n = ok = unsure = 0
    misses, dump = [], []
    for st in PAIR_SETS:
        for a, b in st["pairs"]:
            for said, other in ((a, b), (b, a)):
                for v in voices:
                    path = os.path.join(d, f"{v}_{said}.wav")
                    if not os.path.exists(path):
                        continue
                    wav, sr = sf.read(path, dtype="float32")
                    d_ab = s._pair_llr(wav, sr, a, b)  # raw, uncalibrated
                    llr = d_ab if said == a else -d_ab
                    r = {"llr": round(llr, 2), "heard": said if llr > 1.0 else other if llr < -1.0 else None}
                    dump.append({"set": st["id"], "voice": v, "said": said, "other": other, "a": a, "b": b,
                                 "llr_said": r["llr"]})
                    n += 1
                    if r["heard"] == said:
                        ok += 1
                    elif r["heard"] is None:
                        unsure += 1
                    else:
                        misses.append(f"{v}: said '{said}', heard '{other}' (llr {r['llr']})")
    print(f"{n} recordings: correct {ok} ({ok/n:.1%}), unsure {unsure} ({unsure/n:.1%}), wrong {n-ok-unsure} ({(n-ok-unsure)/n:.1%})")
    for m in misses:
        print("  ", m)
    if len(sys.argv) > 2:
        import json
        with open(sys.argv[2], "w") as f:
            json.dump(dump, f)


if __name__ == "__main__":
    main()
