"""Deploy the Pronunciation Coach website (server.py + web/) on Modal.

Same setup as the Hugging Face Space (deploy/space/Dockerfile): CPU-only
torch, the acoustic model and g2p_en's NLTK data baked into the image, and
HOSTED=1 so the page says audio is sent to the server for scoring.

  modal deploy scripts/modal_web.py      # prints the public https://...modal.run URL
  modal app stop pronunciation-coach     # take it down

Cost: containers scale to zero 5 minutes after the last request, so it
costs nothing while unused. A running container (2 CPU cores, 4 GiB) costs
about $0.13/hour at Modal's rates. max_containers caps the worst case. The
first visit after a quiet spell waits about 20-40 s while the model loads.

The voice coach uses its built-in template unless you add LLM keys as a
Modal secret named "coach-keys" (GROQ_API_KEY etc., see .env.example);
the local .env file is never uploaded.
"""
import os

import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = "/app"

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("libsndfile1", "espeak-ng")
    .pip_install("torch==2.14.0", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install_from_requirements(os.path.join(ROOT, "deploy", "space", "requirements.txt"))
    .env({"HF_HOME": "/opt/hf", "NLTK_DATA": "/opt/nltk_data"})
    .run_commands(
        "python -c \"from huggingface_hub import snapshot_download; "
        "snapshot_download('facebook/wav2vec2-lv-60-espeak-cv-ft')\"",
        "python -c \"import nltk; [nltk.download(p, download_dir='/opt/nltk_data') for p in "
        "('averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'cmudict')]\"",
    )
    .env({"HF_HUB_OFFLINE": "1", "HOSTED": "1"})
    # only what the app needs at runtime; never .env (API keys)
    .add_local_file(os.path.join(ROOT, "server.py"), f"{APP_DIR}/server.py")
    .add_local_dir(os.path.join(ROOT, "src"), f"{APP_DIR}/src", ignore=["**/__pycache__"])
    .add_local_dir(os.path.join(ROOT, "web"), f"{APP_DIR}/web")
    .add_local_dir(os.path.join(ROOT, "models"), f"{APP_DIR}/models")
    .add_local_dir(os.path.join(ROOT, "examples"), f"{APP_DIR}/examples")
)

app = modal.App("pronunciation-coach", image=image)
secrets = [modal.Secret.from_name("coach-keys")] if os.environ.get("COACH_KEYS") == "1" else []


@app.function(cpu=2, memory=4096, min_containers=0, max_containers=2, scaledown_window=300,
              timeout=120, secrets=secrets)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def web():
    import sys
    sys.path.insert(0, APP_DIR)
    os.chdir(APP_DIR)
    import server
    server.get_scorer()  # load the model once per container, before the first request
    server.get_teacher()
    return server.app
