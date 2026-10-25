"""Spoken feedback from the voice teacher ("Maya" by default).

The scoring pipeline stays LLM-free (REPORT.md). An LLM is used only
*after* scoring, to phrase the scorer's measured results as a short,
natural spoken comment. It is grounded on a compact fact sheet (overall
score, flagged sounds, what each sounded closer to, and the hand-written
tip from src/tips.py). The prompt forbids adding articulation advice that
isn't in those tips, which is REPORT.md's reason for keeping LLMs out:
"a wrong generated articulatory instruction is worse than a correct
generic one".

"Talk to Maya" (answer()) works the same way: the learner's spoken question
is transcribed by Groq's Whisper, answered by the same LLM chain grounded
on the full hand-written tip table, and spoken back.

Providers (all free tiers, keys read from .env, never sent to the browser):
  text:  Groq -> Gemini -> Mistral -> deterministic template (always works)
  voice: Gemini TTS -> Groq Orpheus TTS -> none (the page then uses the
         browser's built-in voice)
Each provider's model is picked from its live model list, so a retired
model id doesn't break the app; env vars can override.
"""
import base64
import io
import json
import os
import re
import time
import wave

import httpx

from src.phone_mapping import ARPABET_TO_IPA
from src.tips import EXAMPLES, TIPS, get_tip

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMEOUT = httpx.Timeout(25.0, connect=8.0)


def load_env(path=os.path.join(ROOT, ".env")):
    """Minimal .env reader (KEY=VALUE lines); real env vars take precedence."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if v and not os.environ.get(k):
                os.environ[k] = v


# ---------------------------------------------------------------- speakable sound names
# TTS engines can't read IPA, so every sound is described in words.
_LETTERS = {"B": "b", "CH": "ch", "D": "d", "F": "f", "G": "g", "HH": "h", "JH": "j", "K": "k", "L": "l",
            "M": "m", "N": "n", "NG": "ng", "P": "p", "R": "r", "S": "s", "SH": "sh", "T": "t", "V": "v",
            "W": "w", "Y": "y", "Z": "z", "ZH": "zh"}
_IPA_TO_ARPA = {}
for _a, _i in ARPABET_TO_IPA.items():
    _IPA_TO_ARPA.setdefault(_i, re.sub(r"\d$", "", _a))
_IPA_TO_ARPA.update({"ɚ": "ER", "ɑːɹ": "AA", "ɔːɹ": "AO", "ɛɹ": "EH", "ɪɹ": "IH", "ʊɹ": "UH", "aɪɚ": "AY"})


def spoken_name(arpa):
    b = re.sub(r"\d$", "", arpa or "")
    if b == "TH":
        return 'the "th" sound in "think"'
    if b == "DH":
        return 'the "th" sound in "this"'
    if arpa == "AH0":
        return 'the quick "uh" vowel in "about"'
    if b in _LETTERS:
        ex = EXAMPLES.get(b)
        return f'the "{_LETTERS[b]}" sound' + (f' in "{ex}"' if ex else "")
    ex = EXAMPLES.get(b)
    return f'the vowel in "{ex}"' if ex else "that sound"


def spoken_name_ipa(ipa):
    return spoken_name(_IPA_TO_ARPA.get(ipa, "")) if ipa in _IPA_TO_ARPA else None


# ---------------------------------------------------------------- facts + template
def build_facts(result, max_sounds=3):
    flagged = []
    for w in result["words"]:
        for p in w["phones"]:
            if p["flagged"]:
                flagged.append((p["score"], w["text"].lower(), p))
    flagged.sort(key=lambda x: x[0])
    seen, focus = set(), []
    for sc, word, p in flagged:
        key = re.sub(r"\d$", "", p["arpa"])
        if key in seen:
            continue
        seen.add(key)
        focus.append({"sound": spoken_name(p["arpa"]), "word": word,
                      "sounded_closer_to": spoken_name_ipa(p.get("heard_as")) if p.get("heard_as") else None,
                      "how_to_fix": get_tip(p["arpa"]).replace(" -- ", "; ")})
        if len(focus) == max_sounds:
            break
    strong = [w["text"].lower() for w in result["words"] if min(p["score"] for p in w["phones"]) >= 1.8]
    facts = {"sentence": result["sentence"], "overall_score_out_of_100": round(result["overall"]["score_100"]),
             "sounds_scored": result["overall"]["n_phones"],
             "scoring_strictness": result["overall"].get("strictness", "standard").replace("_", " "), "sounds_worth_another_take": result["overall"]["n_flagged"],
             "words_that_sounded_clear": strong[:6], "focus_sounds": focus}
    fl = result.get("fluency")
    if fl:
        facts["fluency_out_of_10"] = fl["fluency_10"]
        facts["rhythm_and_intonation_out_of_10"] = fl["prosody_10"]
        facts["words_per_minute"] = fl["words_per_min"]
        facts["long_pauses_between_words"] = fl["long_pauses"]
    return facts


def template_feedback(facts, name):
    s = facts["overall_score_out_of_100"]
    opener = ("That was excellent!" if s >= 92 else "Nice work, that was very clear." if s >= 84
              else "Good effort, most of that came through clearly." if s >= 74 else "Thanks for trying that one.")
    parts = [opener]
    if facts["words_that_sounded_clear"]:
        parts.append(f'I liked how you said "{facts["words_that_sounded_clear"][0]}".')
    if not facts["focus_sounds"]:
        parts.append("Every sound was on target. Try a longer sentence next!")
    else:
        f = facts["focus_sounds"][0]
        line = f'Let\'s polish {f["sound"]} in "{f["word"]}"'
        line += f', it came out closer to {f["sounded_closer_to"]}.' if f["sounded_closer_to"] else "."
        parts += [line, f["how_to_fix"].split(";")[0].rstrip(".") + ".", "Give it another go!"]
    if facts.get("long_pauses_between_words", 0) >= 2:
        parts.insert(-1, "Also try to keep the words flowing, with fewer long pauses.")
    return " ".join(parts)


SYSTEM_PROMPT = """You are {name}, a warm, encouraging English pronunciation teacher speaking to a learner right after they read a sentence aloud. Your words will be converted to speech, so write exactly what you would say out loud.

You receive FACTS measured by an automatic phoneme scorer. Rules:
- 45 to 85 words, 3 to 5 short sentences, plain spoken English. No lists, markdown, emojis, IPA symbols or phonetic notation.
- Open with a genuine one-line reaction that matches the overall score (above 90 excellent, 80-90 very good, 70-80 good, below 70 encouraging).
- The overall score is the share of sounds on target at the learner's chosen scoring_strictness; at strict or very strict a lower score is expected, so don't be discouraging about it, and don't call a sentence perfect if any sound was flagged.
- If a word sounded clear, you may praise it briefly.
- Talk about at most two of the focus sounds, using the sound description and word exactly as given. If "sounded_closer_to" is given, you may mention it.
- For how to fix a sound, paraphrase ONLY the given "how_to_fix" text. Never add any other mouth, tongue or lip instructions of your own.
- If fluency facts are given and fluency is below 7 or there were 2+ long pauses, you may add ONE short sentence about keeping the words flowing.
- Never claim to have heard anything that is not in the facts. Do not mention numbers other than the overall score, and only if it helps.
- End by inviting them to try again (or, if nothing needs work, to try a harder sentence).
{style}"""

STYLES = {
    "gentle": "- Tone: extra warm and encouraging; lead with what went well.",
    "direct": "- Tone: friendly but brief and to the point; skip the praise unless the score is above 90.",
}

QA_PROMPT = """You are {name}, a warm English pronunciation teacher inside a practice app. The learner is talking to you out loud, and your reply will be converted to speech.

Rules:
- Reply in 1 to 4 short spoken sentences, under 80 words. No lists, markdown, emojis, IPA symbols or phonetic notation.
- Stay on English pronunciation, this learner's results, and how to practise with the app (Practice tab: write a sentence, record it, hear feedback; Lessons tab; Pairs tab with ONLY these minimal-pair drills -- never suggest a pair that is not listed: {pairs}; Progress tab). If asked about anything else, kindly steer back.
- For HOW to make a sound (tongue, lips, teeth, voicing), use ONLY the matching entry in the TIP TABLE below, paraphrased. If the sound isn't in the table, say you don't have a specific tip and suggest listening to the reference and practising with a minimal-pair drill. Never invent articulation advice.
- If LAST RESULT is given, you may refer to it, but never claim to have heard anything that is not in it. You cannot hear the learner's question audio as pronunciation practice.
- Be honest that scores come from an automatic scorer and are a practice guide.

TIP TABLE (ARPAbet sound: tip, example word):
{tips}

LAST RESULT:
{context}"""


def _tip_table():
    return "\n".join(f"{k} ({EXAMPLES.get(k, '')}): {v.replace(' -- ', '; ')}" for k, v in TIPS.items())


def _clean(text):
    text = re.sub(r"[*_#`>]+", "", text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip('"')


# ---------------------------------------------------------------- providers
class Teacher:
    PREF = {
        "groq": ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "llama-3.3-70b-versatile", "openai/gpt-oss-20b",
                 "meta-llama/llama-4-maverick-17b-128e-instruct", "llama-3.1-8b-instant"],
        "groq_tts": ["canopylabs/orpheus-v1-english", "playai-tts"],
        "gemini": ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.5-flash-lite", "gemini-2.0-flash"],
        "gemini_tts": ["gemini-2.5-flash-preview-tts", "gemini-2.5-flash-tts", "gemini-2.5-pro-preview-tts"],
        "mistral": ["mistral-small-latest", "mistral-medium-latest", "open-mistral-nemo"],
    }

    def __init__(self):
        load_env()
        self.name = os.environ.get("TEACHER_NAME", "Maya")
        self.voice = os.environ.get("GEMINI_TTS_VOICE", "Aoede")
        self.groq_voice = os.environ.get("GROQ_TTS_VOICE", "")
        self.keys = {"groq": os.environ.get("GROQ_API_KEY"), "gemini": os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
                     "mistral": os.environ.get("MISTRAL_API_KEY")}
        self.models = {}
        self.cooldown = {}  # provider -> time until which it's skipped (rate limits / errors)
        self.client = httpx.Client(timeout=TIMEOUT)
        self._discover()

    # ---- model discovery
    def _pick(self, available, prefs, env):
        if os.environ.get(env):
            return os.environ[env]
        for p in prefs:
            if p in available:
                return p
        return None

    def _discover(self):
        k = self.keys
        if k["groq"]:
            try:
                r = self.client.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {k['groq']}"})
                r.raise_for_status()
                ids = {m["id"] for m in r.json()["data"]}
                self.models["groq"] = self._pick(ids, self.PREF["groq"], "GROQ_MODEL")
                if self.groq_voice:  # Orpheus needs a voice name and one-time terms acceptance in the Groq console
                    self.models["groq_tts"] = self._pick(ids, self.PREF["groq_tts"], "GROQ_TTS_MODEL")
            except Exception as e:
                print(f"[teacher] groq unavailable: {type(e).__name__}: {str(e)[:120]}")
        if k["gemini"]:
            try:
                r = self.client.get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                                    headers={"x-goog-api-key": k["gemini"]})
                r.raise_for_status()
                ids = {m["name"].split("/", 1)[1] for m in r.json().get("models", [])}
                self.models["gemini"] = self._pick(ids, self.PREF["gemini"], "GEMINI_MODEL")
                self.models["gemini_tts"] = self._pick(ids, self.PREF["gemini_tts"], "GEMINI_TTS_MODEL") or \
                    next((i for i in sorted(ids) if "tts" in i and "flash" in i), None)
            except Exception as e:
                print(f"[teacher] gemini unavailable: {type(e).__name__}: {str(e)[:120]}")
        if k["mistral"]:
            try:
                r = self.client.get("https://api.mistral.ai/v1/models", headers={"Authorization": f"Bearer {k['mistral']}"})
                r.raise_for_status()
                ids = {m["id"] for m in r.json()["data"]}
                self.models["mistral"] = self._pick(ids, self.PREF["mistral"], "MISTRAL_MODEL")
            except Exception as e:
                print(f"[teacher] mistral unavailable: {type(e).__name__}: {str(e)[:120]}")
        self.models = {p: m for p, m in self.models.items() if m}
        print(f"[teacher] {self.name}: text via {self.text_chain() or ['template']}, voice via {self.voice_chain() or ['browser']}")

    def text_chain(self):
        return [f"{p}:{self.models[p]}" for p in ("groq", "gemini", "mistral") if p in self.models]

    def voice_chain(self):
        return [f"{p.split('_')[0]}:{self.models[p]}" for p in ("gemini_tts", "groq_tts") if p in self.models]

    def info(self):
        return {"name": self.name, "text": self.text_chain()[0].split(":")[0] if self.text_chain() else "template",
                "voice": self.voice_chain()[0].split(":")[0] if self.voice_chain() else "browser"}

    def _ok(self, p):
        return p in self.models and self.cooldown.get(p, 0) < time.time()

    def _fail(self, p, e, seconds=60):
        self.cooldown[p] = time.time() + seconds
        print(f"[teacher] {p} failed ({type(e).__name__}: {str(e)[:160]}); skipping it for {seconds}s")

    # ---- text
    def _chat_openai_style(self, url, key, model, system, user):
        body = {"model": model, "temperature": 0.6, "max_tokens": 300,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        # reasoning models spend tokens thinking before answering: keep that
        # short and out of the reply
        if "gpt-oss" in model:
            body.update(reasoning_effort="low", max_tokens=1500)
        elif "qwen3" in model:
            body.update(reasoning_format="hidden", max_tokens=1500)
        r = self.client.post(url, headers={"Authorization": f"Bearer {key}"}, json=body)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def _chat_gemini(self, system, user):
        m = self.models["gemini"]
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": 0.6, "maxOutputTokens": 400,
                                     "thinkingConfig": {"thinkingBudget": 0}}}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
        r = self.client.post(url, headers={"x-goog-api-key": self.keys["gemini"]}, json=body)
        if r.status_code == 400:  # models without "thinking" reject thinkingConfig
            del body["generationConfig"]["thinkingConfig"]
            r = self.client.post(url, headers={"x-goog-api-key": self.keys["gemini"]}, json=body)
        r.raise_for_status()
        return "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])

    def _llm(self, system, user):
        """First provider in the chain that answers sensibly; (text, provider) or (None, None)."""
        for p in ("groq", "gemini", "mistral"):
            if not self._ok(p):
                continue
            try:
                if p == "groq":
                    t = self._chat_openai_style("https://api.groq.com/openai/v1/chat/completions", self.keys[p], self.models[p], system, user)
                elif p == "gemini":
                    t = self._chat_gemini(system, user)
                else:
                    t = self._chat_openai_style("https://api.mistral.ai/v1/chat/completions", self.keys[p], self.models[p], system, user)
                t = _clean(t)
                if 3 <= len(t.split()) <= 140:
                    return t, p
                raise ValueError(f"unexpected length ({len(t.split())} words)")
            except Exception as e:
                self._fail(p, e, 120 if "429" in str(e) else 30)
        return None, None

    def feedback_text(self, result, style="gentle"):
        facts = build_facts(result)
        system = SYSTEM_PROMPT.format(name=self.name, style=STYLES.get(style, STYLES["gentle"]))
        t, p = self._llm(system, "FACTS:\n" + json.dumps(facts, ensure_ascii=False, indent=1))
        if t and len(t.split()) >= 15:
            return t, p
        return template_feedback(facts, self.name), "template"

    # ---- talk to the teacher
    def transcribe(self, audio_bytes, filename="question.wav"):
        """Speech -> text with Groq's free Whisper; None if unavailable."""
        if not self.keys["groq"] or self.cooldown.get("whisper", 0) > time.time():
            return None
        try:
            r = self.client.post("https://api.groq.com/openai/v1/audio/transcriptions",
                                 headers={"Authorization": f"Bearer {self.keys['groq']}"},
                                 files={"file": (filename, audio_bytes, "audio/wav")},
                                 data={"model": "whisper-large-v3-turbo", "language": "en", "temperature": "0"},
                                 timeout=httpx.Timeout(60.0, connect=8.0))
            r.raise_for_status()
            return (r.json().get("text") or "").strip()
        except Exception as e:
            self._fail("whisper", e, 120 if "429" in str(e) else 30)
            return None

    def can_listen(self):
        return bool(self.keys["groq"])

    def answer(self, question, history=(), result=None):
        """Spoken-style reply to a learner's question (text, provider)."""
        context = json.dumps(build_facts(result), ensure_ascii=False) if result else "(none yet)"
        from src.pairs import PAIR_SETS
        system = QA_PROMPT.format(name=self.name, tips=_tip_table(), context=context,
                                  pairs="; ".join(f'{st["title"]} ({", ".join("/".join(p) for p in st["pairs"][:3])})'
                                                  for st in PAIR_SETS))
        convo = "".join(f"{'Learner' if h.get('role') == 'user' else self.name}: {str(h.get('text', ''))[:400]}\n"
                        for h in list(history)[-6:])
        t, p = self._llm(system, convo + f"Learner: {question[:400]}\n{self.name}:")
        if t:
            return t, p
        return self._answer_template(question), "template"

    def _answer_template(self, q):
        ql = q.lower()
        for arpa, word in EXAMPLES.items():
            if re.search(rf"\b{re.escape(_LETTERS.get(arpa, '###'))}\b", ql) or re.search(rf"\b{word}\b", ql):
                return f"For {spoken_name(arpa)}: {get_tip(arpa).replace(' -- ', '; ')}"
        return ("I can explain how to make any English sound. Try asking, for example, how do I say the th sound? "
                "Or practise a sentence in the Practice tab and I'll tell you what to work on.")

    # ---- voice
    def speak(self, text):
        """(WAV bytes, provider), or (None, None): the page then falls back to
        the browser's own voice."""
        for p, fn in (("gemini_tts", self._speak_gemini), ("groq_tts", self._speak_groq)):
            if self._ok(p):
                try:
                    return fn(text), p.split("_")[0]
                except Exception as e:
                    # free-tier TTS quotas are small; back off longer on 429
                    self._fail(p, e, 900 if "429" in str(e) else 60)
        return None, None

    def _speak_groq(self, text):
        r = self.client.post("https://api.groq.com/openai/v1/audio/speech",
                             headers={"Authorization": f"Bearer {self.keys['groq']}"},
                             json={"model": self.models["groq_tts"], "voice": self.groq_voice, "input": text[:1000],
                                   "response_format": "wav"}, timeout=httpx.Timeout(60.0, connect=8.0))
        r.raise_for_status()
        return r.content

    def _speak_gemini(self, text):
        r = self.client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.models['gemini_tts']}:generateContent",
            headers={"x-goog-api-key": self.keys["gemini"]},
            json={"contents": [{"parts": [{"text": "Say warmly and naturally, like a friendly teacher: " + text}]}],
                  "generationConfig": {"responseModalities": ["AUDIO"],
                                       "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": self.voice}}}}},
            timeout=httpx.Timeout(60.0, connect=8.0))
        r.raise_for_status()
        part = r.json()["candidates"][0]["content"]["parts"][0]["inlineData"]
        pcm = base64.b64decode(part["data"])
        rate = int(re.search(r"rate=(\d+)", part.get("mimeType", "")).group(1)) if "rate=" in part.get("mimeType", "") else 24000
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm)
        return buf.getvalue()
