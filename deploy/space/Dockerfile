# Hugging Face Space image for the v2 local app (server.py + web/).
# Assembled and uploaded by scripts/deploy_space.py -- the build context is
# that staging folder, not the repo root.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    NLTK_DATA=/opt/nltk_data

# libsndfile: soundfile decoding; espeak-ng: required at model load by the
# tokenizer's phonemizer backend (see requirements.txt in the repo)
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /app

RUN pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

# bake the frozen acoustic model and g2p_en's NLTK data into the image, so
# a cold start never downloads anything
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/wav2vec2-lv-60-espeak-cv-ft')"
RUN python -c "import nltk; [nltk.download(p, download_dir='/opt/nltk_data') for p in ('averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'cmudict')]"

ENV HF_HUB_OFFLINE=1 HOSTED=1 HOST=0.0.0.0 PORT=7860 HOME=/home/user

COPY --chown=user . .
USER user

# fail the build (not the first visitor) if anything can't load
RUN python -c "import sys; sys.path.insert(0, '.'); from src.scorer import Scorer; Scorer(); print('scorer OK')"

EXPOSE 7860
CMD ["python", "server.py", "--no-browser"]
