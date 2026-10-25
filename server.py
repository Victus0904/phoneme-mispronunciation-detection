"""Local Pronunciation Coach app: a FastAPI server that serves web/index.html
and scores recordings with the v2 pipeline (src/scorer.py). Everything runs
on this machine -- audio never leaves it.

Run:  python server.py            (opens http://127.0.0.1:7860 in your browser)
      python server.py --no-browser --port 8000
"""
import argparse
import io
import json
import os
import sys
import threading
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from src.scorer import DEFAULT_BUNDLE, Scorer, ScoringError

app = FastAPI(title="Pronunciation Coach")
_scorer = None
_lock = threading.Lock()  # one inference at a time keeps memory bounded on small machines

EXAMPLES_DIR = os.path.join(ROOT, "examples")


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
        return [{"id": e["id"], "sentence": e["sentence"], "url": f"/examples/{e['wav']}"}
                for e in json.load(f) if os.path.exists(os.path.join(EXAMPLES_DIR, e["wav"]))]


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "web", "index.html"))


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
    return {
        "model_label": f"model {s['version']} · r = {m.get('pearson', float('nan')):.2f}",
        "metrics": m, "flag_threshold": s["flag_threshold"], "description": s["description"],
        "examples": _examples(),
    }


@app.post("/api/score")
async def score(audio: UploadFile = File(...), sentence: str = Form("")):
    data = await audio.read()
    try:
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception:
        return {"ok": False, "error": "Couldn't read that audio. The page converts recordings to WAV automatically; "
                                      "if you uploaded a file, try WAV, MP3, M4A or OGG."}
    try:
        with _lock:
            return get_scorer().score(np.asarray(wav), sr, sentence)
    except ScoringError as e:
        return {"ok": False, "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    print("Loading the acoustic model (first run downloads ~1.2 GB from Hugging Face)...", flush=True)
    get_scorer()
    url = f"http://{args.host}:{args.port}"
    print(f"Ready: {url}", flush=True)
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
