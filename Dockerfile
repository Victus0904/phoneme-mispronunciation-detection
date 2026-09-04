# AWS App Runner target: 1 vCPU / 4 GB, ap-south-1, image pulled from ECR.
# CPU-only, fp32 (deliberate -- see README.md: fp32 keeps numerical parity
# with REPORT.md's measured numbers; no quantization, no model substitution).
# Python 3.13 to match the host this was actually tested on (see
# requirements.txt) -- numpy==2.5.2 requires >=3.12, so this isn't optional.
FROM --platform=linux/amd64 python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    NLTK_DATA=/opt/nltk_data \
    USE_QUEUE=1

WORKDIR /app

# libsndfile1: app.py's own audio decoding (src: soundfile, see app.py's
# _load_audio) -- deliberately not the full `ffmpeg` package, see README.md
# / DEPLOY.md for the size trade-off and its one real limitation (mp3
# uploads aren't supported; WAV/FLAC/OGG and browser mic recordings are).
# espeak-ng: NOT optional, despite looking like a phonemizer-only concern --
# transformers' Wav2Vec2PhonemeCTCTokenizer eagerly initializes a real
# espeak-ng backend on load (do_phonemize=True by default) even though this
# pipeline never calls it to phonemize text. Without it, model load crashes
# at container startup. See requirements.txt for how this was verified.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch from the dedicated index: the default PyPI wheel pulls
# ~2GB of CUDA libraries that are dead weight on a CPU-only 4GB instance
# and would blow the image size budget by themselves.
RUN pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

# Bake the acoustic model into the image at build time (needs network --
# this step must run BEFORE HF_HUB_OFFLINE is set below). A container
# downloading ~1.2GB on first request is a broken first impression, and
# App Runner instances don't share a persistent disk cache across restarts
# or scaling events -- this download would otherwise repeat every cold
# start.
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('facebook/wav2vec2-lv-60-espeak-cv-ft')"

# Same reasoning for g2p_en's NLTK data (averaged perceptron tagger +
# cmudict) -- both tagger ids are needed, not one: g2p_en/g2p.py's own
# import-time check does nltk.data.find('taggers/averaged_perceptron_tagger.zip')
# (the old unsuffixed id), while nltk's pos_tag() -- what g2p_en actually
# calls at runtime -- resolves the tagger under a language-suffixed id,
# 'averaged_perceptron_tagger_eng', as of the nltk version resolved here
# (3.10.3). Missing either one means either an import-time nltk.download()
# network call (defeats the whole point of baking this in) or a runtime
# LookupError; verified both ways.
RUN python -c "\
import nltk; \
nltk.download('averaged_perceptron_tagger', download_dir='/opt/nltk_data'); \
nltk.download('averaged_perceptron_tagger_eng', download_dir='/opt/nltk_data'); \
nltk.download('cmudict', download_dir='/opt/nltk_data')"

# Now that both are baked in, force every from_pretrained()/nltk lookup at
# runtime (and in the smoke test right below) to use this cache only --
# no network round-trip, and a build that fails loudly if the bake above
# missed a file, instead of an App Runner instance silently phoning home.
ENV HF_HUB_OFFLINE=1

# Scoring pipeline + demo + example clips + the trained head only --
# .dockerignore keeps datasets/, cached posteriors, and features.parquet
# out regardless, this is belt-and-suspenders for anyone building without it.
COPY src/ src/
COPY app.py .
COPY examples/ examples/
COPY REPORT.md .
COPY cache/head_model.joblib cache/head_model.joblib

# Import-time smoke test: fails the build here, at the cheap/fast point,
# rather than at the first real request on a live App Runner service.
# app.py loads the model at module import (not per-request, see app.py),
# so this both verifies the bake worked AND warms nothing extra.
RUN python -c "import app"

EXPOSE 8080

# app.py's own launch() call already binds 0.0.0.0:8080 with
# ssr_mode=False, and reads USE_QUEUE from the environment
# (default on; set to 0 at deploy time -- no rebuild -- if App Runner's
# request timeout doesn't tolerate the queue's long-lived SSE connections,
# see DEPLOY.md).
CMD ["python", "app.py"]
