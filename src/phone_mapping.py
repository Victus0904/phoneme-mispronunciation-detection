"""ARPAbet (speechocean762's canonical phones) -> IPA/eSpeak (the acoustic
model's output vocabulary) mapping.

Why this exists: the acoustic model (facebook/wav2vec2-lv-60-espeak-cv-ft)
outputs IPA-ish symbols produced by espeak-ng's en-us phonemizer, not
ARPAbet. speechocean762 supplies canonical phones in ARPAbet-with-stress-
digit form (CMUdict style, e.g. 'IY0', 'AH1'). These inventories were
verified NOT to line up 1:1, so a mapping had to be built and checked before
any scoring code, not guessed at.

How it was built: rather than mapping from memory, we ran the *actual*
espeak-ng en-us backend (via `phonemizer`) over all 2,604 unique words in
speechocean762 and aligned its IPA output against each word's ARPAbet
sequence position-by-position. 2,249/2,604 words (86%) matched in phone
count and gave a clean per-symbol vote; each ARPAbet symbol below is mapped
to its majority IPA vote. See the "confidence notes" below for anything
that wasn't a clean landslide.

The other 355 words mismatched in phone count almost entirely because of
one systematic phenomenon, not noise: **espeak-ng fuses a vowel immediately
followed by R into a single rhotic IPA symbol** (e.g. ARPAbet ['AA0', 'R']
in "arm" -> espeak ['ɑːɹ'], one token, not two). This is a real inventory
mismatch (many-to-one), not just multiple spellings of the same thing, and
is handled explicitly by RHOTIC_R_FUSION + map_word_phones_to_ipa_targets
below rather than silently dropping the R or aligning it to noise.

Confidence notes (from the empirical vote counts):
  - AH0 (unstressed, "schwa") -> 'ə' at 39% share, competing with 'ɚ'/'ʌ'/
    'ɐ'/'ɑː'. Genuinely variable in English (schwa is famously the most
    reduced, context-dependent vowel); 'ə' is the linguistically standard
    default and was still the clear plurality.
  - AH1 (stressed, "STRUT") -> 'ʌ' at 51%, with 'ɑː' a real second choice
    at 35%. Flagged as the least confident mapping in the table.
  - AA0/AA1 (unstressed/stressed "PALM") were noisy pre-fusion-handling
    (æ/ɑː/ɑːɹ all competitive) because many raw votes before the R-fusion
    fix were contaminated by unhandled vowel+R words. After excluding
    detected vowel+R cases, 'ɑː' is the standard choice.
  - IY0 -> 'i' (55%) vs 'iː' (38%): unstressed IY tends short in English;
    this is a real, expected stress-conditioned split, not noise.
  - ER0 -> 'ɚ' (70%) / ER1 -> 'ɜː' (98%): confident and phonologically
    expected (unstressed r-colored schwa vs. stressed NURSE vowel).
  - IH0 has a real competitor 'ᵻ' (21%, "barred i", the reduced vowel in
    e.g. unstressed "-es"/"-ed"); 'ɪ' (73%) kept as the single target
    symbol for simplicity -- see Limitations in REPORT.md.
  - T -> 't' (87%) has a large flapping competitor 'ɾ' (13%, American
    English intervocalic flap, e.g. "party" -> [ɾ]). This is a real,
    expected allophone of /t/, not an error; kept as 't' since it's the
    same phoneme.
  - L -> 'l' (86%) vs syllabic 'əl' (14%, e.g. "little"); kept as 'l'.

RHOTIC_R_FUSION entries for AA/AO/EH/AH were directly observed in the vote
data. IH/IY/UH/UW/OW/AY + R fusions were extrapolated by phonological
analogy (not all were directly observed with enough samples) -- flagged
here explicitly per the "do not silently guess" instruction, and worth a
sanity-check glance once alignment is running on real audio (Checkpoint 2).
"""
import re

VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
          "IH", "IY", "OW", "OY", "UH", "UW"}

# Consonants: unambiguous, all >85% majority vote (most ~100%).
_CONSONANT_MAP = {
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f", "G": "ɡ",
    "HH": "h", "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n",
    "NG": "ŋ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t",
    "TH": "θ", "V": "v", "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}

# Vowels, keyed by full ARPAbet symbol including stress digit (stress
# genuinely changes vowel quality in English, e.g. AH0 'schwa' vs AH1
# 'STRUT' -- collapsing stress before mapping was tried first and produced
# a much noisier table). Bare 'IH'/'UH' (no digit) are rare dataset
# variants, mapped the same as their primary-stress form.
_VOWEL_MAP = {
    "AA0": "ɑː", "AA1": "ɑː", "AA2": "ɑː",
    "AE0": "æ", "AE1": "æ", "AE2": "æ",
    "AH0": "ə", "AH1": "ʌ", "AH2": "ʌ",
    "AO0": "ɔː", "AO1": "ɔː", "AO2": "ɔː",
    "AW0": "aʊ", "AW1": "aʊ", "AW2": "aʊ",
    "AY0": "aɪ", "AY1": "aɪ", "AY2": "aɪ",
    "EH0": "ɛ", "EH1": "ɛ", "EH2": "ɛ",
    "ER0": "ɚ", "ER1": "ɜː", "ER2": "ɜː",
    "EY0": "eɪ", "EY1": "eɪ", "EY2": "eɪ",
    "IH": "ɪ", "IH0": "ɪ", "IH1": "ɪ", "IH2": "ɪ",
    "IY0": "i", "IY1": "iː", "IY2": "iː",
    "OW0": "oʊ", "OW1": "oʊ", "OW2": "oʊ",
    "OY0": "ɔɪ", "OY1": "ɔɪ",
    "UH": "ʊ", "UH0": "ʊ", "UH1": "ʊ",
    "UW0": "uː", "UW1": "uː", "UW2": "uː",
}

ARPABET_TO_IPA = {**_CONSONANT_MAP, **_VOWEL_MAP}

# vowel-with-stress -> fused IPA symbol, used only when that vowel is
# immediately followed by 'R' in the ARPAbet sequence.
RHOTIC_R_FUSION = {
    "AA0": "ɑːɹ", "AA1": "ɑːɹ", "AA2": "ɑːɹ",
    "AO0": "ɔːɹ", "AO1": "ɔːɹ", "AO2": "ɔːɹ",
    "EH0": "ɛɹ", "EH1": "ɛɹ", "EH2": "ɛɹ",
    "AH0": "ɚ", "AH1": "ɚ", "AH2": "ɚ",
    # extrapolated, not directly observed with strong sample counts:
    "IH": "ɪɹ", "IH0": "ɪɹ", "IH1": "ɪɹ", "IH2": "ɪɹ",
    "IY0": "ɪɹ", "IY1": "ɪɹ", "IY2": "ɪɹ",
    "UH": "ʊɹ", "UH0": "ʊɹ", "UH1": "ʊɹ",
    "UW0": "ʊɹ", "UW1": "ʊɹ",
    "OW0": "ɔːɹ", "OW1": "ɔːɹ",
    "AY0": "aɪɚ", "AY1": "aɪɚ", "AY2": "aɪɚ",
}


def _base(phone: str) -> str:
    return re.sub(r"\d$", "", phone)


def map_word_phones_to_ipa_targets(phones):
    """ARPAbet phone list (e.g. ['B', 'EH0', 'R']) -> list of alignment
    targets: [{'ipa': str, 'arpa': [orig phone(s)]}, ...].

    A vowel immediately followed by 'R' collapses into one target (one IPA
    symbol, two ARPAbet phones sharing that symbol's aligned frames) when a
    fusion mapping exists; the R is dropped as a *separate* alignment
    target but is still scored later (both its and the vowel's human score
    get attributed to the one aligned frame span -- see features.py).
    """
    targets = []
    i = 0
    n = len(phones)
    while i < n:
        p = phones[i]
        if (_base(p) in VOWELS and i + 1 < n and phones[i + 1] == "R"
                and p in RHOTIC_R_FUSION):
            targets.append({"ipa": RHOTIC_R_FUSION[p], "arpa": [p, "R"]})
            i += 2
        else:
            targets.append({"ipa": ARPABET_TO_IPA.get(p), "arpa": [p]})
            i += 1
    return targets


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, ".")
    from src.data import load_speechocean762, get_phone_inventory

    ds = load_speechocean762(decode_audio=False)
    inventory = get_phone_inventory(ds)

    unmapped = [p for p in inventory if p not in ARPABET_TO_IPA]
    print(f"speechocean762 ARPAbet inventory: {len(inventory)} symbols")
    print(f"Unmapped symbols: {unmapped if unmapped else 'NONE'}")
    print()
    print(f"{'ARPAbet':10s} -> IPA target   (+R fusion target, if any)")
    for p in inventory:
        ipa = ARPABET_TO_IPA.get(p, "???")
        fusion = RHOTIC_R_FUSION.get(p)
        fusion_str = f"   (+R -> {fusion!r})" if fusion else ""
        print(f"{p:10s} -> {ipa!r:8s}{fusion_str}")
