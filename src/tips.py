"""Hand-written articulatory coaching-tip lookup table, keyed by ARPAbet
phone (stress digit stripped). No model, no LLM -- deliberately a static,
deterministic, boring dictionary (see REPORT.md for the reasoning: a wrong
LLM-generated articulatory instruction is worse than a correct generic one,
and this is a place where "boring and correct" beats "flexible and
occasionally wrong").

Covers the ~10 commonly-problematic contrasts named in the brief: TH
(voiced/unvoiced), R, L, AE/EH (cat/bed), IH/IY (bit/beat), V/W, NG, Z/S.
"""
import re

TIPS = {
    "TH": "Put your tongue tip lightly between your top and bottom teeth and push air through without voicing it -- if your tongue pulls back, it turns into a T or S.",
    "DH": "Same tongue-between-the-teeth position as TH, but voice it -- you should feel your throat buzz as the air passes ('this', not 'dis').",
    "R": "Curl or bunch the tip of your tongue up and back without letting it touch the roof of your mouth, and round your lips slightly -- don't tap the tongue like a D or L.",
    "L": "Touch just the tip of your tongue to the ridge behind your upper front teeth and let air flow around both sides -- don't let the whole tongue bunch back like an R.",
    "AE": "Open your jaw wider than usual and keep your tongue low and forward -- this is the 'a' in 'cat', not the narrower 'e' in 'bed'.",
    "EH": "Keep your jaw less open than for 'cat' and your tongue mid-front -- this is the 'e' in 'bed', not the wider-open 'a' in 'cat'.",
    "IH": "Relax your tongue and jaw -- this is the short, lax 'i' in 'bit'; don't tense and lengthen it into the 'ee' of 'beat'.",
    "IY": "Tense your tongue high and forward and hold the vowel a beat longer -- this is the long 'ee' in 'beat', not the shorter, relaxed 'ih' in 'bit'.",
    "V": "Touch your top teeth to your bottom lip and voice it (feel the buzz) -- don't round your lips like a W, and don't let it go silent like an F.",
    "W": "Round your lips into a small circle and glide -- don't touch your teeth to your lip the way you would for V.",
    "NG": "Raise the back of your tongue to touch the soft palate at the back of the roof of your mouth -- don't add a hard 'g' sound after it.",
    "Z": "Same tongue position as S, but voice it -- you should feel your throat buzz as the air hisses through, not stay silent like S.",
    "S": "Keep your throat silent (no voicing) while air hisses between your tongue and teeth -- don't let it buzz like Z.",
}

DEFAULT_TIP = "Listen closely to a native-speaker recording of this sound and try to match its tongue and lip position -- no specific tip is curated for this phone yet."


def _base(phone: str) -> str:
    return re.sub(r"\d$", "", phone)


def get_tip(arpa_phone: str) -> str:
    return TIPS.get(_base(arpa_phone), DEFAULT_TIP)


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    for p in ["TH", "DH", "R", "L", "AE1", "EH0", "IH1", "IY0", "V", "W", "NG", "Z", "S", "K"]:
        print(f"{p:5s}: {get_tip(p)}")
