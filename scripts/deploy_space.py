"""Deploy the v2 app to a Hugging Face Space (Docker SDK).

Stages only what the app needs (server, web page, inference code, trained
head, example clips, deploy/space/* as the Space's Dockerfile/README) into a
temporary folder and uploads it. Requires `hf auth login` first.

Usage: python scripts/deploy_space.py [--space user/name] [--private]
"""
import argparse
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES = ["server.py", "models/head_v2.joblib", "models/utt_v2.joblib", "models/pairs_calibration.json", "models/strictness.json",
         "web/index.html", "web/manifest.webmanifest", "web/sw.js",
         "web/icons/icon-192.png", "web/icons/icon-512.png", "web/icons/maskable-512.png"]
SRC = ["__init__.py", "acoustic.py", "align.py", "phone_mapping.py", "features_v2.py",
       "head_v2.py", "scorer.py", "text.py", "tips.py", "utterance_v2.py", "pairs.py", "teacher.py"]


def stage(dest):
    for f in FILES:
        os.makedirs(os.path.dirname(os.path.join(dest, f)) or dest, exist_ok=True)
        shutil.copy2(os.path.join(ROOT, f), os.path.join(dest, f))
    os.makedirs(os.path.join(dest, "src"))
    for f in SRC:
        shutil.copy2(os.path.join(ROOT, "src", f), os.path.join(dest, "src", f))
    shutil.copytree(os.path.join(ROOT, "examples"), os.path.join(dest, "examples"))
    for f in os.listdir(os.path.join(ROOT, "deploy", "space")):
        shutil.copy2(os.path.join(ROOT, "deploy", "space", f), os.path.join(dest, f))


def main():
    from huggingface_hub import HfApi

    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default=None, help="user/name (default: <you>/pronunciation-coach)")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    api = HfApi()
    space = args.space or f"{api.whoami()['name']}/pronunciation-coach"
    api.create_repo(space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage(tmp)
        api.upload_folder(folder_path=tmp, repo_id=space, repo_type="space",
                          commit_message="Deploy Pronunciation Coach v2")
    print(f"uploaded -> https://huggingface.co/spaces/{space}")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
