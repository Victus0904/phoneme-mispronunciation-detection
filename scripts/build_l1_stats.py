"""Which English sounds speakers of each first language actually get wrong,
measured from L2-ARCTIC's hand annotations (train + validation speakers
only; the test speakers stay unseen).

Writes models/l1_stats.json: per L1, per ARPAbet phone (stress stripped):
phones seen, share marked wrong, and the most common substitutes
(what annotators heard instead). src/l1_profiles.py reads it to rank each
language's focus sounds.

  python scripts/build_l1_stats.py [datasets/l2arctic/l2arctic_annotated.parquet]
"""
import collections
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(ROOT, "datasets", "l2arctic", "l2arctic_annotated.parquet")
OUT = os.path.join(ROOT, "models", "l1_stats.json")
MIN_PHONES = 60  # ignore sounds seen too rarely to rank


def base(p):
    b = re.sub(r"\d$", "", p.strip().upper().rstrip("*"))
    return "AH" if b == "AX" else b  # AX: ARPAbet schwa, same as our AH0


def main():
    import pyarrow.parquet as pq
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    rows = [r for r in pq.read_table(path, columns=["l1", "split", "speaker", "words"]).to_pylist()
            if r["split"] in ("train", "val")]
    seen = collections.defaultdict(collections.Counter)
    wrong = collections.defaultdict(collections.Counter)
    subs = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    speakers = collections.defaultdict(set)
    for r in rows:
        l1 = r["l1"]
        speakers[l1].add(r["speaker"])
        for w in r["words"]:
            mis = {m["index"]: m["pronounced-phone"] for m in (w["mispronunciations"] or [])}
            for i, (p, acc) in enumerate(zip(w["phones"], w["phones-accuracy"])):
                b = base(p)
                seen[l1][b] += 1
                if acc < 2:
                    wrong[l1][b] += 1
                    said = mis.get(i, "")
                    said = "(dropped)" if said == "<DEL>" else " ".join(base(x) for x in said.split()[:2])
                    if said and said != b and "ERR" not in said:  # ERR: annotator couldn't name it
                        subs[l1][b][said] += 1
    out = {"source": "L2-ARCTIC v5.0 hand annotations, train+val speakers (CC BY-NC 4.0)", "languages": {}}
    for l1 in sorted(seen):
        phones = {}
        for b, n in seen[l1].items():
            if n < MIN_PHONES:
                continue
            phones[b] = {"n": n, "error_rate": round(wrong[l1][b] / n, 4),
                         "heard_as": [s for s, _ in subs[l1][b].most_common(3)]}
        out["languages"][l1] = {"speakers": sorted(speakers[l1]), "phones": phones}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    for l1, d in out["languages"].items():
        top = sorted(d["phones"].items(), key=lambda kv: -kv[1]["error_rate"])[:6]
        print(f"{l1:10s} " + "  ".join(f"{b} {v['error_rate']:.0%}→{'/'.join(v['heard_as'][:2])}" for b, v in top))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
