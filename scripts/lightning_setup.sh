#!/usr/bin/env bash
# One-time setup on a Lightning AI Studio (or any Ubuntu GPU box).
# Installs system libs, Python deps and the dataset, then checks the GPU.
# Usage (from the repo root):  bash scripts/lightning_setup.sh
set -euo pipefail

sudo apt-get update -qq
# espeak-ng: the model's tokenizer needs it at load time (src/acoustic.py)
# ffmpeg + libsndfile: audio decoding for src/data.py (extract_full.py) and soundfile
sudo apt-get install -y -qq espeak-ng ffmpeg libsndfile1

pip install -q -r requirements.txt
# huggingface_hub pin in requirements.txt is fine for transformers 5.16.1;
# fine-tuning needs nothing beyond requirements.txt on Linux.

python - <<'EOF'
from huggingface_hub import hf_hub_download
for s in ("train", "test"):
    hf_hub_download("mispeech/speechocean762", f"data/{s}-00000-of-00001.parquet",
                    repo_type="dataset", local_dir="datasets/speechocean762")
import torch
print("torch", torch.__version__, "cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("bf16:", torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False))
EOF
echo "setup done"
