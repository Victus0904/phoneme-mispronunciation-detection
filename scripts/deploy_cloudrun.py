"""Deploy the v2 app to Google Cloud Run.

Stages the same minimal bundle as deploy_space.py (server, web page,
inference code, trained head, example clips, deploy/space/Dockerfile) and
runs `gcloud run deploy --source`, so Cloud Build builds the image remotely
-- no local Docker needed. Requires `gcloud auth login` and a project with
billing enabled.

Settings are chosen to stay inside Cloud Run's free tier for light use:
scale to zero, at most 2 instances, us-central1 (a free-tier region),
CPU only billed while handling requests.

Usage: python scripts/deploy_cloudrun.py --project <project-id> [--region us-central1]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deploy_space import stage  # noqa: E402

SERVICE = "pronunciation-coach"


def gcloud():
    exe = shutil.which("gcloud") or shutil.which("gcloud.cmd")
    if exe:
        return exe
    guess = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Cloud SDK", "google-cloud-sdk", "bin", "gcloud.cmd")
    if os.path.exists(guess):
        return guess
    sys.exit("gcloud not found -- install the Google Cloud CLI first")


def run(*args):
    print("+ gcloud " + " ".join(args), flush=True)
    subprocess.run([gcloud(), *args], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--region", default="us-central1")
    args = ap.parse_args()

    run("services", "enable", "run.googleapis.com", "cloudbuild.googleapis.com",
        "artifactregistry.googleapis.com", "--project", args.project)
    with tempfile.TemporaryDirectory() as tmp:
        stage(tmp)
        run("run", "deploy", SERVICE,
            "--source", tmp,
            "--project", args.project, "--region", args.region,
            "--allow-unauthenticated",
            "--memory", "4Gi", "--cpu", "2",
            "--cpu-boost",              # faster cold start (model load)
            "--min-instances", "0",     # scale to zero when idle -> no idle cost
            "--max-instances", "2",     # cost ceiling
            "--concurrency", "4",
            "--timeout", "120",
            "--quiet")


if __name__ == "__main__":
    main()
