"""Run fine-tuning and the v2 rebuild on Modal (modal.com) GPUs.

A Modal Volume ("pmd-ft") keeps everything between runs: the dataset,
runs/<name>/ checkpoints (so a stopped run resumes), cache/<name>/ and
results/<name>/. The repo code is copied into the image on every launch,
so local edits are picked up without a push.

  pip install modal && modal setup                       # once
  # 2-minute speed / sanity check (~$0.05)
  modal run scripts/modal_app.py::train --name smoke --extra "--limit 160 --epochs 1"
  # real run, detached so closing the terminal doesn't stop it
  modal run --detach scripts/modal_app.py::train --name ft_top8
  # rebuild GOP-SF features + heads on the fine-tuned model; scores test ONCE
  modal run --detach scripts/modal_app.py::rebuild --name ft_top8
  # fetch results
  modal volume get pmd-ft results/ft_top8 ./runs/

Each function has a hard timeout, so a mistake can't spend the whole budget.
"""
import os
import subprocess

import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "/root/repo"
VOL = "/vol"
# first available wins; L4 capacity is sometimes short
GPU = os.environ.get("MODAL_GPU", "L4,A10G,T4").split(",")


def _requirements():
    """requirements.txt minus gradio (only the v1 demo, app.py, needs it).
    Only read on the machine that launches the run: this module is imported
    again inside the container, where the image is already built."""
    if not modal.is_local():
        return []
    keep = []
    for line in open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8"):
        line = line.split("#")[0].strip()
        if line and not line.lower().startswith("gradio"):
            keep.append(line)
    return keep


image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("espeak-ng", "ffmpeg", "libsndfile1")
    .pip_install(*_requirements())
    .env({"HF_HOME": f"{VOL}/hf", "PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    .add_local_dir(ROOT, REPO, ignore=[".env", "datasets", "cache", "runs", ".git", ".idea", ".venv", "android",
                                        "**/__pycache__", "docs", "examples"])
)
vol = modal.Volume.from_name("pmd-ft", create_if_missing=True)
app = modal.App("pmd-finetune", image=image)


def _link(name):
    """Point repo-relative datasets/, runs/, cache/ at the volume. cache/
    is per run name, so frozen and fine-tuned features never mix."""
    os.makedirs(f"{VOL}/datasets", exist_ok=True)
    os.makedirs(f"{VOL}/runs", exist_ok=True)
    os.makedirs(f"{VOL}/cache/{name}", exist_ok=True)
    for src, dst in ((f"{VOL}/datasets", "datasets"), (f"{VOL}/runs", "runs"), (f"{VOL}/cache/{name}", "cache")):
        p = os.path.join(REPO, dst)
        if not os.path.lexists(p):
            os.symlink(src, p)
    from huggingface_hub import hf_hub_download
    for split in ("train", "test"):  # no-op once they are on the volume
        hf_hub_download("mispeech/speechocean762", f"data/{split}-00000-of-00001.parquet",
                        repo_type="dataset", local_dir=os.path.join(REPO, "datasets", "speechocean762"))


def _sh(cmd, env=None):
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=REPO, env={**os.environ, **(env or {})})
    vol.commit()


@app.function(gpu=GPU, cpu=4, memory=16384, volumes={VOL: vol}, timeout=6 * 3600)
def train(name: str = "ft_top8", extra: str = ""):
    _link(name)
    _sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
    # --max-hours stops cleanly (and checkpoints) before the hard timeout
    _sh(f"python scripts/finetune_acoustic.py --out runs/{name} --batch 8 --workers 4 "
        f"--no-checkpointing --max-hours 5 {extra}")


@app.function(gpu=GPU, cpu=8, memory=32768, volumes={VOL: vol}, timeout=4 * 3600)
def rebuild(name: str = "ft_top8"):
    """Re-extract with runs/<name>/best, rebuild features and heads, and
    score the test split once. Results land in results/<name>/ on the volume."""
    _link(name)
    best = f"{REPO}/runs/{name}/best"
    if not os.path.isdir(best):
        raise SystemExit(f"{best} missing: fine-tuning never beat the frozen baseline, nothing to rebuild")
    env = {"ACOUSTIC_MODEL": best}
    for step in ("scripts/extract_full.py", "scripts/build_features_v2.py", "scripts/train_v2.py",
                 "scripts/calibrate_strictness.py", "scripts/build_utt_features.py", "scripts/train_fluency.py"):
        _sh(f"python {step}", env)
    out = f"{VOL}/results/{name}"
    _sh(f"mkdir -p {out} && cp models/head_v2.joblib models/utt_v2.joblib models/strictness.json "
        f"cache/v2/metrics*.json {out}/ && cp -r figures {out}/")


L2_PARQUET = "datasets/l2arctic/l2arctic_annotated.parquet"


@app.function(cpu=4, memory=8192, volumes={VOL: vol}, timeout=2 * 3600)
def prep_l2arctic():
    """Download L2-ARCTIC v5.0 (CC BY-NC 4.0; mirror of the official release),
    keep only the hand-annotated utterances, and pack them with
    scripts/l2arctic.py into <volume>/datasets/l2arctic/. Downloads ~7.5 GB
    inside Modal; nothing large touches the local machine."""
    import re
    import shutil
    import zipfile
    from huggingface_hub import hf_hub_download

    _link("l2prep")
    sys_path = f"{REPO}/scripts"
    import sys
    sys.path.insert(0, REPO); sys.path.insert(0, sys_path)
    import l2arctic

    raw = "/tmp/l2raw"   # untrusted archive contents, kept apart from the code
    os.makedirs(raw, exist_ok=True)
    for spk in l2arctic.SPLIT:
        z = hf_hub_download("chikingsley/l2-arctic-release-v5.0", f"{spk}.zip", repo_type="dataset",
                            local_dir="/tmp/l2zips")
        with zipfile.ZipFile(z) as zf:
            names = zf.namelist()
            ann = {m for m in names if re.fullmatch(rf"{spk}/annotation/[\w.-]+\.TextGrid", m)}
            utts = {os.path.splitext(os.path.basename(m))[0] for m in ann}
            wavs = {m for m in names if re.fullmatch(rf"{spk}/wav/[\w.-]+\.wav", m)
                    and os.path.splitext(os.path.basename(m))[0] in utts}
            for m in sorted(ann | wavs):  # whitelisted paths only (no zip-slip)
                dst = os.path.join(raw, *m.split("/"))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with zf.open(m) as src, open(dst, "wb") as out:
                    shutil.copyfileobj(src, out)
        os.remove(z)
        print(f"{spk}: {len(ann)} annotated utterances", flush=True)
    out = os.path.join(REPO, L2_PARQUET)
    _sh(f"python scripts/l2arctic.py pack {raw} {out}")
