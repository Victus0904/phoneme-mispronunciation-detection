"""Minimal-pair drills: word pairs that differ in one sound English learners
commonly merge. The app asks for one word of a pair and checks which of the
two expected phone sequences the recording fits better (Scorer.which_word).

Grouped by the contrast; `sound` is the ARPAbet phone being practised in
word A (used to pick the hand-written tip and link to a lesson).
"""

PAIR_SETS = [
    {"id": "th-s", "title": "TH vs S", "sound": "TH", "pairs": [("think", "sink"), ("thick", "sick"), ("path", "pass"), ("mouth", "mouse"), ("thumb", "sum")]},
    {"id": "th-t", "title": "TH vs T", "sound": "TH", "pairs": [("three", "tree"), ("thin", "tin"), ("thank", "tank"), ("thought", "taught"), ("both", "boat")]},
    {"id": "dh-d", "title": "TH (voiced) vs D", "sound": "DH", "pairs": [("then", "den"), ("they", "day"), ("though", "dough"), ("those", "doze"), ("breathe", "breed")]},
    {"id": "r-l", "title": "R vs L", "sound": "R", "pairs": [("right", "light"), ("rice", "lice"), ("road", "load"), ("pray", "play"), ("fry", "fly"), ("correct", "collect")]},
    {"id": "v-w", "title": "V vs W", "sound": "V", "pairs": [("vest", "west"), ("vine", "wine"), ("veil", "whale"), ("verse", "worse"), ("vow", "wow")]},
    {"id": "v-b", "title": "V vs B", "sound": "V", "pairs": [("vote", "boat"), ("very", "berry"), ("van", "ban"), ("vest", "best"), ("curve", "curb")]},
    {"id": "ih-iy", "title": "Ship vs sheep", "sound": "IH", "pairs": [("ship", "sheep"), ("sit", "seat"), ("live", "leave"), ("fill", "feel"), ("bit", "beat"), ("hit", "heat")]},
    {"id": "ae-eh", "title": "Bad vs bed", "sound": "AE", "pairs": [("bad", "bed"), ("man", "men"), ("sat", "set"), ("pan", "pen"), ("had", "head")]},
    {"id": "ae-ah", "title": "Cat vs cut", "sound": "AE", "pairs": [("cat", "cut"), ("bag", "bug"), ("cap", "cup"), ("ran", "run"), ("match", "much")]},
    {"id": "n-ng", "title": "N vs NG", "sound": "NG", "pairs": [("sing", "sin"), ("thing", "thin"), ("bang", "ban"), ("rang", "ran"), ("wing", "win")]},
    {"id": "s-sh", "title": "S vs SH", "sound": "SH", "pairs": [("she", "see"), ("ship", "sip"), ("short", "sort"), ("shell", "sell"), ("shoe", "sue")]},
    {"id": "z-s", "title": "Z vs S", "sound": "Z", "pairs": [("zip", "sip"), ("prize", "price"), ("buzz", "bus"), ("eyes", "ice"), ("zoo", "sue")]},
    {"id": "final-dt", "title": "Final D vs T", "sound": "D", "pairs": [("bad", "bat"), ("ride", "write"), ("card", "cart"), ("seed", "seat"), ("bed", "bet")]},
    {"id": "uh-uw", "title": "Full vs fool", "sound": "UH", "pairs": [("full", "fool"), ("pull", "pool"), ("look", "Luke"), ("could", "cooed"), ("should", "shooed")]},
]


def public_sets():
    """Both directions of every pair are practisable; the UI picks which word
    to ask for."""
    return PAIR_SETS
