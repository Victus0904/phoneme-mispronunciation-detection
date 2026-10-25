"""Hand-written articulatory coaching-tip lookup table, keyed by ARPAbet
phone (stress digit stripped). No model, no LLM -- deliberately a static,
deterministic, boring dictionary (see REPORT.md for the reasoning: a wrong
LLM-generated articulatory instruction is worse than a correct generic one,
and this is a place where "boring and correct" beats "flexible and
occasionally wrong").

v1 covered the ~10 commonly-problematic contrasts named in the brief (TH,
DH, R, L, AE/EH, IH/IY, V/W, NG, Z/S). v2 extends the same hand-written
style to every ARPAbet phone in speechocean762's inventory, plus an example
word per phone for the UI. Tips for a few contrasts that are especially
common for Mandarin-L1 learners (the only population in the training data)
mention that contrast explicitly.
"""
import re

TIPS = {
    # --- v1 entries (unchanged) ---
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
    # --- v2 additions: remaining consonants ---
    "P": "Press both lips together, build up air, and release it with a small puff -- hold a paper in front of your mouth at the start of a word and it should move (unlike B).",
    "B": "Press both lips together and release them while your voice is already on -- no strong puff of air, otherwise it sounds like P.",
    "T": "Tap the tongue tip on the ridge just behind your upper front teeth and release with a small puff -- at the end of a word, still make the closure instead of dropping it.",
    "D": "Same tongue-tip spot as T, but voiced and without the puff of air -- at the end of a word ('bed'), keep the voicing instead of turning it into a T or dropping it.",
    "K": "Lift the back of your tongue against the soft palate and release with a puff of air -- at the end of a word ('back'), make the closure clearly rather than swallowing it.",
    "G": "Same back-of-tongue closure as K, but voiced and without the puff -- don't let it turn into K at the end of a word ('bag', not 'back').",
    "F": "Lightly touch your top teeth to your bottom lip and blow air through without voicing -- don't use both lips like a P or W.",
    "SH": "Round your lips slightly and pull the tongue a little further back than for S -- the hiss should sound broader and lower ('she', not 'see').",
    "ZH": "Same position as SH but voiced -- the middle sound of 'measure' and 'vision'; keep it a smooth buzz, not a 'j' as in 'jam'.",
    "CH": "Start with the tongue blade pressed behind your upper teeth like T, then release it straight into SH ('church') -- one quick combined sound.",
    "JH": "Like CH but voiced -- press, then release into a buzzing ZH sound, as in 'judge'; keep your voice on the whole time.",
    "HH": "Just breathe out through an open mouth already shaped for the next vowel -- a light, voiceless breath, without scraping the back of the throat.",
    "M": "Close both lips and hum through your nose -- at the end of a word ('time'), keep the lips closed instead of letting it turn into N.",
    "N": "Press the tongue tip on the ridge behind your upper front teeth and hum through your nose -- keep it distinct from NG ('sin' vs. 'sing') and from L.",
    "Y": "Start with your tongue high and forward like the 'ee' in 'see', then glide quickly into the next vowel ('yes').",
    # --- v2 additions: remaining vowels / diphthongs ---
    "AA": "Open your jaw wide and keep your tongue low and relaxed toward the back -- the 'a' in 'father'; don't round your lips.",
    "AH": "Keep your mouth relaxed and only slightly open, tongue in the middle -- the short 'u' in 'cup' (or, unstressed, the quick 'uh' in 'about'); don't make it a full 'ah'.",
    "AO": "Open your jaw and round your lips slightly with the tongue low and back -- the vowel in 'thought' or 'law'.",
    "AW": "Start with an open 'ah' and glide toward a rounded 'oo' -- 'how', 'now'; make sure both ends of the glide are there.",
    "AY": "Start with an open 'ah' and glide up toward 'ee' -- 'my', 'time'; don't stop at the first half.",
    "ER": "Bunch or curl your tongue as for R and hold it through the whole vowel -- 'bird', 'her'; don't add a separate vowel before the R.",
    "EY": "Start at the 'e' of 'bed' and glide up toward 'ee' -- 'day', 'say'; it should move, not stay flat.",
    "OW": "Start with a mid 'o' and glide toward 'oo' while rounding your lips more -- 'go', 'home'; don't keep it as one flat 'o'.",
    "OY": "Start with rounded lips for 'aw' and glide toward 'ee' -- 'boy', 'coin'.",
    "UH": "Round your lips slightly and keep the tongue relaxed -- the short vowel in 'book' and 'good', shorter and looser than the 'oo' in 'food'.",
    "UW": "Round your lips into a small tight circle with the tongue high and back, and hold it -- the long 'oo' in 'food', not the short vowel in 'book'.",
}

EXAMPLES = {
    "AA": "father", "AE": "cat", "AH": "cup", "AO": "thought", "AW": "now", "AY": "my",
    "B": "bed", "CH": "chair", "D": "day", "DH": "this", "EH": "bed", "ER": "bird",
    "EY": "day", "F": "fan", "G": "go", "HH": "hat", "IH": "bit", "IY": "beat",
    "JH": "judge", "K": "cat", "L": "light", "M": "man", "N": "no", "NG": "sing",
    "OW": "go", "OY": "boy", "P": "pen", "R": "red", "S": "see", "SH": "she",
    "T": "top", "TH": "think", "UH": "book", "UW": "food", "V": "very", "W": "we",
    "Y": "yes", "Z": "zoo", "ZH": "measure",
}

DEFAULT_TIP = "Listen closely to a native-speaker recording of this sound and try to match its tongue and lip position -- no specific tip is curated for this phone yet."


def _base(phone: str) -> str:
    return re.sub(r"\d$", "", phone)


def get_tip(arpa_phone: str) -> str:
    return TIPS.get(_base(arpa_phone), DEFAULT_TIP)


def get_example(arpa_phone: str):
    if arpa_phone == "AH0":
        return "about"  # unstressed AH is schwa, not the 'cup' vowel
    return EXAMPLES.get(_base(arpa_phone))


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    for p in ["TH", "DH", "R", "L", "AE1", "EH0", "IH1", "IY0", "V", "W", "NG", "Z", "S", "K"]:
        print(f"{p:5s}: {get_tip(p)}")
