"""First-language (L1) profiles: which English sounds to focus on, and how
to customise lessons, minimal pairs, suggested sentences and coach notes.

Two kinds of evidence, always labelled to the user:
- measured: Arabic, Hindi, Korean, Mandarin, Spanish, Vietnamese. Focus
  sounds are ranked by how often L2-ARCTIC's annotators marked them wrong
  for speakers of that language, and what they heard instead
  (models/l1_stats.json, scripts/build_l1_stats.py). The scoring model was
  also checked on held-out speakers of these languages.
- described: other languages. Focus sounds are hand-picked from commonly
  described patterns for that language. Not measured, and the scorer has
  not been validated on those accents.

Coaching tips themselves stay in src/tips.py (hand-written, the same for
everyone); this module only adds a short "why this sound, for you" note.
"""
import json
import os

from src.tips import get_tip

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATS_FILE = os.path.join(ROOT, "models", "l1_stats.json")

# sound -> lesson id (web/index.html LESSONS) and minimal-pair set ids (src/pairs.py)
SOUND_LESSON = {"TH": "th", "DH": "th", "R": "rl", "L": "rl", "V": "vw", "W": "vw", "IH": "ee", "IY": "ee",
                "AE": "vowels", "EH": "vowels", "AH": "vowels", "AA": "vowels", "Z": "endings", "S": "endings",
                "D": "endings", "T": "endings", "NG": "endings", "JH": "jch", "CH": "jch", "SH": "jch",
                "ZH": "jch", "P": "pbf", "B": "pbf", "F": "pbf", "ER": "er", "OW": "oh", "AO": "oh",
                "UH": "oh", "UW": "oh"}
SOUND_PAIRS = {"TH": ["th-s", "th-t"], "DH": ["dh-d"], "R": ["r-l"], "L": ["r-l"], "V": ["v-w", "v-b"],
               "W": ["v-w"], "IH": ["ih-iy"], "IY": ["ih-iy"], "AE": ["ae-eh", "ae-ah"], "EH": ["ae-eh"],
               "AH": ["ae-ah"], "NG": ["n-ng"], "SH": ["s-sh", "ch-sh"], "Z": ["z-s"], "S": ["z-s"],
               "D": ["final-dt"], "T": ["final-dt"], "UH": ["uh-uw"], "UW": ["uh-uw"], "P": ["p-b"],
               "B": ["p-b"], "F": ["f-p"], "CH": ["ch-sh", "jh-ch"], "JH": ["jh-ch", "y-jh"]}
SOUND_NAMES = {"TH": "th (think)", "DH": "th (this)", "R": "r", "L": "l", "V": "v", "W": "w",
               "IH": "short i (sit)", "IY": "long ee (seat)", "AE": "a (cat)", "EH": "e (bed)",
               "AH": "u (cup)", "AA": "ah (father)", "Z": "z", "S": "s", "D": "d", "T": "t", "NG": "ng",
               "JH": "j (jam)", "CH": "ch", "SH": "sh", "ZH": "zh (measure)", "P": "p", "B": "b", "F": "f",
               "ER": "er (bird)", "OW": "oh (go)", "AO": "aw (thought)", "UH": "oo (book)", "UW": "oo (food)",
               "K": "k", "G": "g", "N": "n", "M": "m", "HH": "h", "Y": "y", "EY": "ay (day)",
               "AY": "eye (my)", "AW": "ow (now)", "OY": "oy (boy)", "(dropped)": "left out"}

# Example words for headings, where the bare letter is ambiguous ("z" -> "z (zoo)")
EXAMPLE_WORDS = {"R": "red", "L": "light", "V": "van", "W": "wet", "Z": "zoo", "S": "see", "D": "day",
                 "T": "tea", "NG": "sing", "CH": "chair", "SH": "she", "P": "pen", "B": "bed", "F": "fan",
                 "K": "key", "G": "go", "N": "no", "M": "me", "HH": "hat", "Y": "yes"}


def sound_label(arpa):
    name = SOUND_NAMES.get(arpa, arpa.lower())
    return f"{name} ({EXAMPLE_WORDS[arpa]})" if arpa in EXAMPLE_WORDS and "(" not in name else name


LANGUAGES = [
    # code, label, L2-ARCTIC name (measured) or None, hand-picked focus sounds for "described" ones,
    # and optional short notes (only where the pattern is well established)
    {"code": "ar", "label": "Arabic", "data": "Arabic",
     "notes": {"P": "Arabic has no P sound, so it often comes out as B ('bark' for 'park').",
               "V": "Arabic has no V sound, so it can come out as F."}},
    {"code": "zh", "label": "Mandarin Chinese", "data": "Mandarin",
     "notes": {"TH": "Mandarin has no TH sound; S or F are the usual stand-ins.",
               "V": "Mandarin has no V sound; it tends to drift toward W or F."}},
    {"code": "hi", "label": "Hindi / Urdu", "data": "Hindi",
     "notes": {"V": "Hindi and Urdu have one sound between V and W, so English V and W can blur together.",
               "TH": "TH is often made as a soft dental T: keep the tongue between the teeth and let air flow."}},
    {"code": "ko", "label": "Korean", "data": "Korean",
     "notes": {"F": "Korean has no F sound, so it often comes out as P.",
               "Z": "Korean has no Z sound; it often comes out as J or S."}},
    {"code": "es", "label": "Spanish", "data": "Spanish",
     "notes": {"Z": "Spanish has no buzzing Z, so it usually comes out as S.",
               "IH": "Spanish has one 'i' vowel, so ship and sheep can sound the same.",
               "V": "In Spanish, B and V are the same sound, so English V can come out as B."}},
    {"code": "vi", "label": "Vietnamese", "data": "Vietnamese",
     "notes": {"D": "Final consonants are often softened or dropped; make the closure at the end of the word.",
               "T": "Final consonants are often softened or dropped; make the closure at the end of the word."}},
    {"code": "de", "label": "German", "data": None, "focus": ["W", "TH", "DH", "Z", "AE", "JH"],
     "notes": {"W": "German W sounds like English V, so 'west' can come out as 'vest'.",
               "Z": "German devoices final consonants, so 'eyes' can sound like 'ice'.",
               "TH": "German has no TH; S or Z are the usual stand-ins."}},
    {"code": "fr", "label": "French", "data": None, "focus": ["TH", "DH", "HH", "IH", "R", "UH"],
     "notes": {"HH": "French doesn't pronounce H, so it's easy to drop it ('ouse' for 'house').",
               "TH": "French has no TH; S or Z are the usual stand-ins.",
               "IH": "French has one 'i' vowel, so ship and sheep can sound the same."}},
    {"code": "ja", "label": "Japanese", "data": None, "focus": ["R", "L", "TH", "V", "F", "IH"],
     "notes": {"R": "Japanese has one sound between R and L, so the two can blur together.",
               "V": "Japanese has no V sound, so it often comes out as B."}},
    {"code": "pt", "label": "Portuguese", "data": None, "focus": ["TH", "DH", "IH", "HH", "AE", "NG"],
     "notes": {"TH": "Portuguese has no TH; T, F or S are the usual stand-ins.",
               "IH": "Portuguese has no short 'i', so ship and sheep can sound the same."}},
    {"code": "ru", "label": "Russian", "data": None, "focus": ["TH", "DH", "W", "IH", "AE", "HH"],
     "notes": {"W": "Russian has no W sound, so it often comes out as V.",
               "TH": "Russian has no TH; S, Z or F are the usual stand-ins."}},
    {"code": "other", "label": "Another language", "data": None, "focus": ["TH", "DH", "R", "V", "IH", "Z"],
     "notes": {}},
]
BY_CODE = {l["code"]: l for l in LANGUAGES}

_stats = None


def _load_stats():
    global _stats
    if _stats is None:
        try:
            with open(STATS_FILE, encoding="utf-8") as f:
                _stats = json.load(f)["languages"]
        except (FileNotFoundError, KeyError, ValueError):
            _stats = {}
    return _stats


def languages():
    return [{"code": l["code"], "label": l["label"], "measured": bool(l["data"] and l["data"] in _load_stats())}
            for l in LANGUAGES]


def valid(code):
    return code in BY_CODE


def _focus_sounds(lang, k=5):
    stats = _load_stats().get(lang["data"] or "", {}).get("phones")
    if stats:
        ranked = sorted((b for b in stats if b in SOUND_LESSON or b in SOUND_PAIRS),
                        key=lambda b: -stats[b]["error_rate"])
        return [(b, stats[b]) for b in ranked[:k]]
    return [(b, None) for b in lang.get("focus", [])[:k]]


def profile(code):
    """Everything the page needs to customise itself for one L1."""
    lang = BY_CODE.get(code) or BY_CODE["other"]
    measured = bool(lang["data"] and lang["data"] in _load_stats())
    focus = []
    for b, st in _focus_sounds(lang):
        item = {"sound": b, "name": SOUND_NAMES.get(b, b.lower()), "label": sound_label(b), "note": lang["notes"].get(b),
                "tip": get_tip(b).replace(" -- ", "; "), "lesson": SOUND_LESSON.get(b),
                "pairs": SOUND_PAIRS.get(b, [])}
        if st:
            heard = [SOUND_NAMES.get(h, h.lower()) for h in st["heard_as"][:2]]
            item["error_rate"] = st["error_rate"]
            item["evidence"] = (f"In recordings of {lang['label']} speakers, this sound was marked wrong "
                                f"{round(st['error_rate'] * 100)}% of the time"
                                + (f", most often heard as {' or '.join(heard)}." if heard else "."))
        focus.append(item)
    lessons, pairs = [], []
    for f in focus:
        if f["lesson"] and f["lesson"] not in lessons:
            lessons.append(f["lesson"])
        for p in f["pairs"]:
            if p not in pairs:
                pairs.append(p)
    return {"code": lang["code"], "label": lang["label"], "measured": measured, "data": lang["data"],
            "evidence_label": ("Based on how speakers of this language were rated in a research dataset"
                               if measured else "Based on commonly described patterns; not measured for this language"),
            "focus": focus, "lessons": lessons, "pairs": pairs}


def note_for(code, arpa):
    """Short L1-specific note for one sound, or None."""
    lang = BY_CODE.get(code)
    if not lang:
        return None
    b = arpa.rstrip("012")
    if b in lang["notes"]:
        return lang["notes"][b]
    st = _load_stats().get(lang["data"] or "", {}).get("phones", {}).get(b)
    if st and st["error_rate"] >= 0.2:
        return f"A common one for {lang['label']} speakers: in research recordings it was marked wrong {round(st['error_rate'] * 100)}% of the time."
    return None
