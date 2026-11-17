"""L2-ARCTIC -> the speechocean762 row format used by finetune_acoustic.py.

L2-ARCTIC (Zhao et al., 2018; CC BY-NC 4.0, non-commercial use only) has 24
speakers, four each with L1 Arabic, Mandarin, Hindi, Korean, Spanish and
Vietnamese. 150 utterances per speaker (3,599 in all) have hand-annotated
TextGrids in <speaker>/annotation/, whose "phones" tier labels each phone:

  "AH0"            pronounced as expected
  "CPL,PPL,s"      substitution: canonical CPL said as PPL
  "CPL,sil,d"      deletion
  "sil,PPL,a"      addition (insertion) of PPL

Each annotated utterance becomes one row like speechocean762's:
  words: [{text, phones (canonical ARPAbet), phones-accuracy,
           mispronunciations: [{canonical-phone, index, pronounced-phone}]}]
  phones-accuracy: 2 = correct, 0 = substituted/deleted, 1 = substituted by
                   a distorted version of itself ("AH,AH*,s"). L2-ARCTIC has
                   no graded scores, so these are coarser than speechocean's.
  audio: 16 kHz mono WAV bytes (L2-ARCTIC ships 44.1 kHz)
  speaker, l1, split

Speaker split (fixed, by L1): of each L1's four speakers, the third is
validation and the fourth is test. The test speakers are never used by the
fine-tuning script; they are there to measure accuracy per accent later.

  python scripts/l2arctic.py inspect <l2arctic_root>     # label stats, no output
  python scripts/l2arctic.py pack <l2arctic_root> [out.parquet]
  # default out: datasets/l2arctic/l2arctic_annotated.parquet
"""
import collections
import glob
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from src.phone_mapping import ARPABET_TO_IPA, VOWELS

DEFAULT_OUT = os.path.join(ROOT, "datasets", "l2arctic", "l2arctic_annotated.parquet")

SPEAKERS = {
    "Arabic": ["ABA", "SKA", "YBAA", "ZHAA"],
    "Mandarin": ["BWC", "LXC", "NCC", "TXHC"],
    "Hindi": ["ASI", "RRBI", "SVBI", "TNI"],
    "Korean": ["HJK", "HKK", "YDCK", "YKWK"],
    "Spanish": ["EBVS", "ERMS", "MBMPS", "NJS"],
    "Vietnamese": ["HQTV", "PNV", "THV", "TLV"],
}
SPLIT = {}
L1_OF = {}
for _l1, _spk in SPEAKERS.items():
    for _i, _s in enumerate(_spk):
        SPLIT[_s] = "val" if _i == 2 else "test" if _i == 3 else "train"
        L1_OF[_s] = _l1

SILENCE = {"", "sil", "sp", "spn", "<unk>", "<sil>"}


# ---------------------------------------------------------------- TextGrid

def _read_text(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "utf-16"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def read_textgrid(path):
    """Minimal Praat TextGrid reader (long and short format) -> {tier: [(xmin, xmax, text)]}."""
    txt = _read_text(path)
    tiers = {}
    if "item [" in txt:  # long format
        for block in re.split(r"\n\s*item \[\d+\]:", txt)[1:]:
            name = re.search(r'name = "(.*?)"', block)
            if not name:
                continue
            ivs = re.findall(r'xmin = ([\d.eE+-]+)\s*\n\s*xmax = ([\d.eE+-]+)\s*\n\s*text = "((?:[^"]|"")*)"', block)
            tiers[name.group(1).strip().lower()] = [(float(a), float(b), t.replace('""', '"').strip())
                                                    for a, b, t in ivs]
    else:  # short format: quoted strings and numbers in order
        toks = re.findall(r'"((?:[^"]|"")*)"|([\d.eE+-]+)', txt)
        vals = [t[0] if t[0] or not t[1] else t[1] for t in toks]
        i = vals.index("IntervalTier") if "IntervalTier" in vals else -1
        while 0 <= i < len(vals):
            name, n = vals[i + 1], int(float(vals[i + 4]))
            j = i + 5
            ivs = []
            for _ in range(n):
                ivs.append((float(vals[j]), float(vals[j + 1]), vals[j + 2].strip()))
                j += 3
            tiers[name.strip().lower()] = ivs
            i = vals.index("IntervalTier", j) if "IntervalTier" in vals[j:] else -1
    return tiers


# ---------------------------------------------------------------- labels

def _clean(p):
    return p.strip().upper().replace(" ", "")


def _stress_inserted(p):
    """An inserted (epenthetic) vowel is nearly always reduced: AH -> AH0
    (schwa), other bare vowels -> unstressed 0. Distortion marks are kept."""
    star = "*" if p.endswith("*") else ""
    b = p.rstrip("*")
    if b in VOWELS:
        b += "0"
    return b + star


def parse_phone_label(label):
    """-> (kind, canonical, pronounced); kind in correct/sub/del/add/sil/bad."""
    lab = label.strip()
    if lab.lower() in SILENCE:
        return "sil", None, None
    parts = [x.strip() for x in lab.split(",")]
    if len(parts) == 1:
        return "correct", _clean(parts[0]), _clean(parts[0])
    if len(parts) >= 3:
        cpl, ppl, typ = parts[0], parts[1], parts[2].lower()
        if typ.startswith("s"):
            return "sub", _clean(cpl), _clean(ppl)
        if typ.startswith("d"):
            return "del", _clean(cpl), None
        if typ.startswith("a"):
            return "add", None, _clean(ppl)
    return "bad", None, None


_CMU = None


def _cmudict():
    global _CMU
    if _CMU is None:
        try:
            import nltk
            try:
                from nltk.corpus import cmudict
                _CMU = cmudict.dict()
            except LookupError:
                nltk.download("cmudict", quiet=True)
                from nltk.corpus import cmudict
                _CMU = cmudict.dict()
        except Exception:
            _CMU = {}
    return _CMU


def _base(p):
    return re.sub(r"\d$", "", p)


def add_stress(word, phones, stats):
    """Annotation phones may lack stress digits, which the ARPAbet->IPA map
    needs for vowels (AH0 schwa vs AH1 strut). Take them from CMUdict when a
    pronunciation with the same base phones exists, else default to 1
    (AH -> AH0, the far more common unstressed schwa)."""
    if all(p in ARPABET_TO_IPA for p in phones):
        return phones
    bases = [_base(p) for p in phones]
    for pron in _cmudict().get(word.lower(), []):
        if [_base(p) for p in pron] == bases:
            stats["stress_cmudict"] += 1
            return [p if p in ARPABET_TO_IPA else q for p, q in zip(phones, pron)]
    stats["stress_default"] += 1
    out = []
    for p in phones:
        if p in ARPABET_TO_IPA or _base(p) not in VOWELS:
            out.append(p)
        else:
            out.append(p + ("0" if _base(p) == "AH" else "1"))
    return out


def utterance_words(tiers, stats):
    """TextGrid tiers -> speechocean-style word list, or None if unusable."""
    words_t = tiers.get("words")
    phones_t = tiers.get("phones")
    if not words_t or not phones_t:
        stats["no_tiers"] += 1
        return None
    words = []
    for wx0, wx1, wtext in words_t:
        if wtext.lower() in SILENCE:
            continue
        canon, acc, mis, pending_add = [], [], [], []
        for px0, px1, plab in phones_t:
            mid = (px0 + px1) / 2
            if not (wx0 <= mid < wx1):
                continue
            kind, cpl, ppl = parse_phone_label(plab)
            stats[kind] += 1
            if kind == "bad":
                stats["bad_examples"] = stats.get("bad_examples", []) [:20] + [plab]
                return None
            if kind == "sil":
                continue
            if kind == "add":
                ppl = _stress_inserted(ppl)
                if canon:  # attach the inserted phone to the previous phone's slot
                    prev = next((m for m in mis if m["index"] == len(canon) - 1), None)
                    if prev is None:
                        mis.append({"canonical-phone": canon[-1], "index": len(canon) - 1,
                                    "pronounced-phone": f"{canon[-1]} {ppl}"})
                    elif prev["pronounced-phone"] != "<DEL>":
                        prev["pronounced-phone"] += f" {ppl}"
                else:
                    pending_add.append(ppl)
                continue
            i = len(canon)
            canon.append(cpl)
            if kind == "correct":
                acc.append(2.0)
                if pending_add:
                    mis.append({"canonical-phone": cpl, "index": i, "pronounced-phone": " ".join(pending_add + [cpl])})
            elif kind == "del":
                acc.append(0.0)
                mis.append({"canonical-phone": cpl, "index": i, "pronounced-phone": "<DEL>"})
            else:  # sub
                distorted = ppl.endswith("*") and _base(ppl.rstrip("*")) == _base(cpl)
                acc.append(1.0 if distorted else 0.0)
                said = " ".join(pending_add + [ppl])
                mis.append({"canonical-phone": cpl, "index": i, "pronounced-phone": said})
            pending_add = []
        if not canon:
            continue
        canon = add_stress(wtext, canon, stats)
        for m in mis:
            m["canonical-phone"] = canon[m["index"]]
        if any(p not in ARPABET_TO_IPA for p in canon):
            stats["unmapped_canonical"] += 1
            return None
        words.append({"text": wtext.upper(), "phones": canon, "phones-accuracy": acc,
                      "mispronunciations": mis})
    return words or None


def iter_annotated(root):
    for tg in sorted(glob.glob(os.path.join(root, "*", "annotation", "*.TextGrid"))):
        spk = os.path.basename(os.path.dirname(os.path.dirname(tg)))
        utt = os.path.splitext(os.path.basename(tg))[0]
        wav = os.path.join(root, spk, "wav", utt + ".wav")
        yield spk.upper(), utt, tg, wav


def wav16k_bytes(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != 16000:
        g = np.gcd(int(sr), 16000)
        wav = resample_poly(wav, 16000 // g, int(sr) // g).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, wav, 16000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def build(root, with_audio=True):
    stats = collections.Counter()
    rows = []
    for spk, utt, tg, wav in iter_annotated(root):
        if spk not in SPLIT:
            stats["unknown_speaker"] += 1
            continue
        if not os.path.exists(wav):
            stats["missing_wav"] += 1
            continue
        words = utterance_words(read_textgrid(tg), stats)
        if words is None:
            stats["skipped_utts"] += 1
            continue
        row = {"id": f"{spk}_{utt}", "speaker": spk, "l1": L1_OF[spk], "split": SPLIT[spk],
               "text": " ".join(w["text"] for w in words), "words": words}
        if with_audio:
            row["audio"] = {"bytes": wav16k_bytes(wav), "path": None}
        rows.append(row)
    return rows, stats


def summarize(rows, stats):
    bad = stats.pop("bad_examples", [])
    print(f"utterances kept: {len(rows)}")
    by = collections.Counter((r["split"], r["l1"]) for r in rows)
    for split in ("train", "val", "test"):
        print(f"  {split:5s}: " + ", ".join(f"{l1}={by[(split, l1)]}" for l1 in SPEAKERS))
    n_ph = sum(len(w["phones"]) for r in rows for w in r["words"])
    n_err = sum(a < 2 for r in rows for w in r["words"] for a in w["phones-accuracy"])
    print(f"phones: {n_ph}, marked wrong: {n_err} ({100 * n_err / max(n_ph, 1):.1f}%)")
    print("label stats:", dict(stats))
    if bad:
        print("unparsed labels (first 20):", bad)


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("inspect", "pack"):
        sys.exit(__doc__)
    cmd, root = sys.argv[1], sys.argv[2]
    if cmd == "inspect":
        rows, stats = build(root, with_audio=False)
        summarize(rows, stats)
        if rows:
            r = rows[0]
            print("\nexample:", r["id"], r["text"])
            for w in r["words"][:6]:
                print("  ", w)
        return
    out = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_OUT
    rows, stats = build(root)
    summarize(rows, stats)
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out)
    print(f"wrote {out} ({os.path.getsize(out) / 2**20:.0f} MB)")


if __name__ == "__main__":
    main()
