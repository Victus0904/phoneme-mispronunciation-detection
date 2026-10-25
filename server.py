"""Local Pronunciation Coach app: a FastAPI server that serves web/index.html
and scores recordings with the v2 pipeline (src/scorer.py). Everything runs
on this machine -- audio never leaves it.

Run:  python server.py            (opens http://127.0.0.1:7860 in your browser)
      python server.py --no-browser --port 8000
"""
import argparse
import base64
import collections
import io
import json
import os
import sys
import threading
import time
import uuid
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import Body, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from src.phone_mapping import ARPABET_TO_IPA
from src.pairs import PAIR_SETS
from src.scorer import DEFAULT_BUNDLE, Scorer, ScoringError
from src.teacher import Teacher
from src.text import text_to_words

app = FastAPI(title="Pronunciation Coach", docs_url=None, redoc_url=None, openapi_url=None)
_scorer = None
_teacher = None
_results = collections.OrderedDict()  # result id -> scoring result (feedback is only given for these)
MAX_SENTENCE_CHARS = 200

# simple per-visitor rate limits (requests per minute), so one visitor can't
# monopolise this machine or burn through the free API quotas
LIMITS = {"score": 20, "feedback": 12, "phonemes": 60, "ask": 10, "pair": 30}
_hits = collections.defaultdict(collections.deque)


def _limited(request: Request, kind: str) -> bool:
    ip = request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "?")
    q, now = _hits[(ip, kind)], time.time()
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= LIMITS[kind]:
        return True
    q.append(now)
    return False


def get_teacher():
    global _teacher
    if _teacher is None:
        _teacher = Teacher()
    return _teacher
_lock = threading.Lock()  # one inference at a time keeps memory bounded on small machines

EXAMPLES_DIR = os.path.join(ROOT, "examples")
# HOSTED=1 when deployed (e.g. the Hugging Face Space): the page then says
# audio is sent to the server for scoring instead of "runs on this computer"
HOSTED = os.environ.get("HOSTED", "0").lower() in ("1", "true", "yes")
MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # ~4 min of 16 kHz WAV; the scorer caps audio at 20 s anyway


def get_scorer():
    global _scorer
    if _scorer is None:
        _scorer = Scorer(os.environ.get("HEAD_BUNDLE", DEFAULT_BUNDLE))
    return _scorer


def _examples():
    path = os.path.join(EXAMPLES_DIR, "manifest.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [{"id": e["id"], "sentence": e.get("display", e["sentence"]), "url": f"/examples/{e['wav']}"}
                for e in json.load(f) if os.path.exists(os.path.join(EXAMPLES_DIR, e["wav"]))]


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "web", "index.html"))


# ---- PWA shell (installable app): manifest, service worker, icons
@app.get("/manifest.webmanifest")
def manifest():
    return FileResponse(os.path.join(ROOT, "web", "manifest.webmanifest"), media_type="application/manifest+json")


@app.get("/sw.js")
def service_worker():
    return FileResponse(os.path.join(ROOT, "web", "sw.js"), media_type="text/javascript",
                        headers={"Cache-Control": "no-cache"})


@app.get("/icons/{name}")
def icon(name: str):
    path = os.path.join(ROOT, "web", "icons", os.path.basename(name))
    if not name.endswith(".png") or not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="image/png")


@app.get("/examples/{name}")
def example_file(name: str):
    path = os.path.join(EXAMPLES_DIR, os.path.basename(name))
    if not name.endswith(".wav") or not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="audio/wav")


@app.get("/api/info")
def info():
    s = get_scorer().info()
    m = s["metrics"]
    t = get_teacher()
    return {
        "model_label": f"model {s['version']} · r = {m.get('pearson', float('nan')):.2f}",
        "metrics": m, "flag_threshold": s["flag_threshold"], "description": s["description"],
        "examples": _examples(), "hosted": HOSTED, "teacher": {**t.info(), "can_listen": t.can_listen()},
        "fluency_metrics": s.get("fluency_metrics"), "strictness": s.get("strictness"),
    }


@app.get("/api/phonemes")
def phonemes(request: Request, sentence: str = ""):
    """Expected pronunciation (IPA per word) shown before recording."""
    if _limited(request, "phonemes") or len(sentence) > MAX_SENTENCE_CHARS:
        return {"ok": False}
    words, unresolved = text_to_words(sentence, get_scorer().g2p) if sentence.strip() else ([], [])
    if not words or unresolved:
        return {"ok": False}
    return {"ok": True, "words": [{"text": w["text"], "ipa": [ARPABET_TO_IPA.get(p, p) for p in w["phones"]]}
                                  for w in words]}


@app.post("/api/score")
async def score(request: Request, audio: UploadFile = File(...), sentence: str = Form(""),
                strictness: str = Form("standard")):
    if _limited(request, "score"):
        return {"ok": False, "error": "You're going a little fast -- please wait a moment and try again."}
    if len(sentence) > MAX_SENTENCE_CHARS:
        return {"ok": False, "error": f"Please keep the sentence under {MAX_SENTENCE_CHARS} characters."}
    data = await audio.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        return {"ok": False, "error": "That file is too large -- please upload a short clip (one sentence)."}
    try:
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception:
        return {"ok": False, "error": "Couldn't read that audio. The page converts recordings to WAV automatically; "
                                      "if you uploaded a file, try WAV, MP3, M4A or OGG."}
    try:
        with _lock:
            result = get_scorer().score(np.asarray(wav), sr, sentence, strictness[:20])
    except ScoringError as e:
        return {"ok": False, "error": str(e)}
    result["id"] = uuid.uuid4().hex
    _results[result["id"]] = result
    while len(_results) > 300:
        _results.popitem(last=False)
    return result


async def _read_wav(audio: UploadFile):
    """(wav, sr) or an error dict."""
    data = await audio.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        return None, {"ok": False, "error": "That file is too large -- please upload a short clip."}
    try:
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        return (np.asarray(wav), sr), None
    except Exception:
        return None, {"ok": False, "error": "Couldn't read that audio."}


@app.get("/api/pairs")
def pairs():
    path = os.path.join(ROOT, "models", "pairs_calibration.json")
    check = json.load(open(path)).get("check") if os.path.exists(path) else None
    from src.tips import get_tip
    return {"sets": [{**st, "tip": get_tip(st["sound"]).replace(" -- ", "; ")} for st in PAIR_SETS], "check": check}


@app.post("/api/pair")
async def pair(request: Request, audio: UploadFile = File(...), set_id: str = Form(...),
               pair_index: int = Form(...), target: str = Form(...)):
    if _limited(request, "pair"):
        return {"ok": False, "error": "You're going a little fast -- please wait a moment."}
    got, err = await _read_wav(audio)
    if err:
        return err
    try:
        with _lock:
            return get_scorer().check_pair(got[0], got[1], set_id, pair_index, target)
    except ScoringError as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/ask")
async def ask(request: Request, audio: UploadFile = File(None), text: str = Form(""),
              history: str = Form("[]"), result_id: str = Form(""), voice: str = Form("1")):
    """Talk to the teacher: spoken (Whisper) or typed question -> spoken reply."""
    if _limited(request, "ask"):
        return {"ok": False, "error": "Let's slow down a little -- try again in a moment."}
    t = get_teacher()
    question = (text or "").strip()[:400]
    if audio is not None and not question:
        data = await audio.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            return {"ok": False, "error": "That question is too long -- keep it under a minute."}
        question = (t.transcribe(data) or "").strip()[:400]
        if not question:
            return {"ok": False, "error": "I couldn't make out the question -- try again, or type it instead."}
    if not question:
        return {"ok": False, "error": "Ask me something about pronunciation!"}
    try:
        hist = json.loads(history)[-6:] if history else []
        hist = [h for h in hist if isinstance(h, dict)]
    except ValueError:
        hist = []
    reply, source = t.answer(question, hist, _results.get(result_id))
    audio_out, v = t.speak(reply) if voice != "0" else (None, None)
    return {"ok": True, "question": question, "answer": reply, "source": source, "voice": v or "browser",
            "audio": base64.b64encode(audio_out).decode() if audio_out else None}


@app.post("/api/feedback")
def feedback(request: Request, body: dict = Body(...)):
    """The voice teacher's spoken feedback for a result this server produced
    (looked up by id -- clients can't feed arbitrary text to the LLM)."""
    if _limited(request, "feedback"):
        return {"ok": False, "error": "rate limited"}
    result = _results.get(str(body.get("id", "")))
    if result is None:
        return {"ok": False, "error": "unknown result"}
    style = "direct" if body.get("style") == "direct" else "gentle"
    key = f"feedback_{style}"
    if key not in result:
        t = get_teacher()
        text, source = t.feedback_text(result, style)
        audio, voice = (t.speak(text) if body.get("voice", True) else (None, None))
        result[key] = {"ok": True, "name": t.name, "text": text, "source": source, "voice": voice or "browser",
                       "audio": base64.b64encode(audio).decode() if audio else None}
    return result[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    print("Loading the acoustic model (first run downloads ~1.2 GB from Hugging Face)...", flush=True)
    get_scorer()
    get_teacher()
    url = f"http://{args.host}:{args.port}"
    print(f"Ready: {url}", flush=True)
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
