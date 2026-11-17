"""Deploy the Pronunciation Coach website (server.py + web/) on Modal.

Same setup as the Hugging Face Space (deploy/space/Dockerfile): CPU-only
torch, the acoustic model and g2p_en's NLTK data baked into the image, and
HOSTED=1 so the page says audio is sent to the server for scoring.

Model: the multi-accent fine-tuned acoustic model (runs/ft_l2/best, CC BY-NC
4.0 via L2-ARCTIC), read from the "pmd-ft" Modal volume, with its matching
heads, strictness thresholds and pair calibration from local runs/ft_l2/.
Set WEB_MODEL=frozen to deploy the original frozen model instead.

Accounts: stored in the modal.Dict "pmd-accounts"; sessions are signed with
SESSION_SECRET from the Modal Secret "pmd-session" (create it once:
modal secret create pmd-session SESSION_SECRET=<random hex>).

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
RUN = os.environ.get("WEB_RUN", "ft_l2")
FROZEN = os.environ.get("WEB_MODEL") == "frozen"
vol = modal.Volume.from_name("pmd-ft")

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
model_env = {"ACCOUNTS_STORE": "modal-dict:pmd-accounts"}
if not FROZEN:
    # matching heads for the fine-tuned model (small files; the 1.3 GB model stays on the volume)
    image = image.add_local_dir(os.path.join(ROOT, "runs", RUN), f"{APP_DIR}/models_ft",
                                ignore=["best", "best/**", "figures", "figures/**", "log.json", "*.png"])
    model_env.update({"ACOUSTIC_MODEL": f"/vol/runs/{RUN}/best", "MODELS_DIR": f"{APP_DIR}/models_ft"})

app = modal.App("pronunciation-coach", image=image)
secrets = [modal.Secret.from_name("pmd-session")]
if os.environ.get("COACH_KEYS") == "1":
    secrets.append(modal.Secret.from_name("coach-keys"))


@app.function(cpu=2, memory=4096, min_containers=0, max_containers=2, scaledown_window=300,
              timeout=120, secrets=secrets, volumes={} if FROZEN else {"/vol": vol.read_only()})
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def web():
    import sys
    os.environ.update(model_env)  # before src.acoustic / src.scorer read them
    sys.path.insert(0, APP_DIR)
    os.chdir(APP_DIR)
    import server
    server.get_scorer()  # load the model once per container, before the first request
    server.get_teacher()
    return server.app
