"""Fit models/pairs_calibration.json from a scripts/check_pairs.py dump.

Per pair set, the offset on LL(word A) - LL(word B) is the midpoint between
the mean raw difference when A was said and when B was said, so the decision
is centred between the two classes. Accuracy is estimated leave-one-voice-out
(the offset is refit without the held-out voice). Synthetic voices are clean,
native-like speech: this checks the mechanism, it is not an accuracy figure
for learners.

Windows: synthesize the words first with scripts/synth_pairs.ps1.

  python scripts/check_pairs.py <wav_dir> dump.json
  python scripts/fit_pairs_calibration.py dump.json <out.json> [margin]
"""
import collections
import json
import sys


def offsets(rows):
    by = collections.defaultdict(lambda: {"a": [], "b": []})
    for r in rows:
        d_ab = r["llr_said"] if r["said"] == r["a"] else -r["llr_said"]
        by[r["set"]]["a" if r["said"] == r["a"] else "b"].append(d_ab)
    out = {}
    for s, v in by.items():
        if v["a"] and v["b"]:
            out[s] = round((sum(v["a"]) / len(v["a"]) + sum(v["b"]) / len(v["b"])) / 2, 3)
    return out


def main():
    rows = json.load(open(sys.argv[1]))
    out_path = sys.argv[2]
    margin = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    voices = sorted({r["voice"] for r in rows})
    n = ok = unsure = 0
    for v in voices:
        bias = offsets([r for r in rows if r["voice"] != v])
        for r in rows:
            if r["voice"] != v:
                continue
            d_ab = (r["llr_said"] if r["said"] == r["a"] else -r["llr_said"]) - bias.get(r["set"], 0.0)
            heard = r["a"] if d_ab > margin else r["b"] if d_ab < -margin else None
            n += 1
            ok += heard == r["said"]
            unsure += heard is None
    check = {"correct": round(ok / n, 3), "unsure": round(unsure / n, 3), "wrong": round((n - ok - unsure) / n, 3),
             "n_recordings": n}
    cal = {"bias_ab": offsets(rows), "margin": margin,
           "note": (f"Per-contrast offset on LL(word A) - LL(word B), fitted on Windows SAPI voices "
                    f"{'/'.join(voices)} saying both words of every pair (scripts/check_pairs.py, "
                    f"scripts/fit_pairs_calibration.py). Leave-one-voice-out: {check['correct']:.1%} correct, "
                    f"{check['unsure']:.1%} unsure, {check['wrong']:.1%} wrong."),
           "check": check}
    json.dump(cal, open(out_path, "w"), indent=1)
    print(cal["note"])


if __name__ == "__main__":
    main()
