#!/usr/bin/env bash
# Build the pronunciation-scoring demo image and push it to ECR, ready for
# an App Runner service to pull it. This script only builds/tags/pushes --
# it does not create or modify any AWS resource. See DEPLOY.md for the
# console steps to create the ECR repo and the App Runner service itself.
set -euo pipefail

# --- Fill these in ---------------------------------------------------
AWS_ACCOUNT_ID="123456789012"        # <-- your AWS account ID
AWS_REGION="ap-south-1"              # <-- deployment region
ECR_REPO_NAME="pronunciation-scorer" # <-- must match the ECR repo you created (see DEPLOY.md)
IMAGE_TAG="$(date +%Y%m%d-%H%M%S)"   # timestamped tag -- also pushes ':latest' below
# -----------------------------------------------------------------------

ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_URI="${ECR_REGISTRY}/${ECR_REPO_NAME}:${IMAGE_TAG}"
LATEST_URI="${ECR_REGISTRY}/${ECR_REPO_NAME}:latest"

echo "== Logging in to ECR ($ECR_REGISTRY) =="
aws ecr get-login-password --region "$AWS_REGION" \
    | docker login --username AWS --password-stdin "$ECR_REGISTRY"

echo "== Building $IMAGE_URI (linux/amd64) =="
docker build --platform linux/amd64 -t "$IMAGE_URI" -t "$LATEST_URI" .

echo "== Pushing $IMAGE_TAG and latest =="
docker push "$IMAGE_URI"
docker push "$LATEST_URI"

echo
echo "Pushed:"
echo "  $IMAGE_URI"
echo "  $LATEST_URI"
echo
echo "Point the App Runner service at this repo/tag (or 'latest') -- see DEPLOY.md."
