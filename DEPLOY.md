# Deploying the demo to AWS App Runner

Target: 1 vCPU / 4 GB, region `ap-south-1`, image pulled from ECR. This
document is steps only -- nothing here runs an AWS CLI command that
creates or modifies a resource; that's on you, deliberately (see the
project's anti-goals). `scripts/build_and_push.sh` builds and pushes the
image; everything below is console clicks.

## 1. Create the ECR repository

1. AWS Console -> **ECR** -> **Repositories** -> **Create repository**.
2. Visibility: **Private**.
3. Repository name: `pronunciation-scorer` (or whatever you set
   `ECR_REPO_NAME` to in `scripts/build_and_push.sh`).
4. Leave scan-on-push and encryption at their defaults. Create.
5. Note the repository URI shown (`<account-id>.dkr.ecr.ap-south-1.amazonaws.com/pronunciation-scorer`).

## 2. Build and push the image

Edit the placeholders at the top of `scripts/build_and_push.sh`
(`AWS_ACCOUNT_ID`, `AWS_REGION`, `ECR_REPO_NAME`), then, with Docker
running and AWS CLI credentials configured locally:

```
bash scripts/build_and_push.sh
```

This logs in to ECR, builds with `--platform linux/amd64` (App Runner
runs amd64; build from an Apple Silicon Mac would otherwise produce an
incompatible image), and pushes both a timestamped tag and `:latest`.

## 3. Create the App Runner service

AWS Console -> **App Runner** -> **Create service**.

- **Source**: Container registry -> **Amazon ECR** -> browse to the repo
  and the tag `build_and_push.sh` just pushed (or `latest`).
- **Deployment trigger**: **Manual**. Do not turn on automatic
  deployments -- every deploy should be a deliberate re-run of the push
  script followed by a manual "Deploy" click in the console, not a side
  effect of pushing to ECR.
- **ECR access role**: let the console create a new access role if
  prompted (`AppRunnerECRAccessRole` or similar).
- **Service settings**:
  - **CPU**: 1 vCPU
  - **Memory**: 4 GB
  - **Port**: `8080` (must match the Dockerfile's `EXPOSE 8080` and
    `app.py`'s `server_port=8080`)
  - **Environment variables**: none required to start. `USE_QUEUE=0` is
    the one you may add later -- see Section 5.
  - **Auto scaling**: create a custom configuration (or edit the
    default) with **max size = 1**. This is the cost cap: at 1 vCPU / 4
    GB there is no meaningful concurrency headroom in this container
    anyway (single-process Gradio, unbatched CPU inference per
    REPORT.md Section 3.3), and capping instances at 1 means a traffic
    spike degrades to slow responses instead of an unbounded bill.
  - **Health check**: HTTP, path `/`, defaults are fine.
- Review and create. First deploy takes a few minutes; the image is
  large (see image-size note below) so the initial pull is the slow part,
  not the app itself (the model is already baked in -- see Dockerfile).

## 4. Pausing, resuming, and logs

- **Pause**: App Runner console -> your service -> **Actions** ->
  **Pause**. Stops billing for compute while paused; the service keeps
  its configuration and ECR image reference. Use this between demo
  sessions -- there's no reason to pay for an idle 1 vCPU / 4 GB
  instance.
- **Resume**: same menu -> **Resume**. Takes roughly as long as the
  original cold start (container start + model load from the baked-in
  cache -- no network download, see Dockerfile).
- **Logs**: service page -> **Logs** tab, or CloudWatch Logs directly
  under the log group App Runner created for the service
  (`/aws/apprunner/<service-name>/<service-id>/application` for
  stdout/stderr, plus a `.../service` group for App Runner's own
  deployment/health-check events). `app.py` runs with
  `PYTHONUNBUFFERED=1` (set in the Dockerfile) specifically so `print()`
  output and tracebacks show up here promptly instead of being buffered.

## 5. Known risk: Gradio's queue and App Runner's request timeout

Gradio's queue (`demo.queue()`, on by default) holds a long-lived
Server-Sent-Events connection open per session to stream results back.
App Runner's request timeout is not guaranteed to tolerate a connection
held open that long, and if it doesn't, requests will appear to hang
rather than fail cleanly.

`app.py` reads `USE_QUEUE` from the environment (default on) and skips
`demo.queue()` entirely when it's falsy. If requests start hanging in
production: App Runner console -> your service -> **Configuration** ->
edit environment variables -> add `USE_QUEUE=0` -> deploy. No rebuild,
no image change -- just a redeploy with the new environment variable.
Without the queue, Gradio falls back to plain synchronous HTTP requests
per interaction, which trade streaming/progress UI for a connection
model App Runner definitely supports.

## 6. Fallback: EC2 + Cloudflare Tunnel

If App Runner turns out to be a bad fit for this workload (the queue
issue above doesn't resolve cleanly, or the per-request cost model
doesn't work out), the same image runs unchanged on a small EC2 instance
(e.g. `t3.medium`, 4 GB RAM) with `docker run -p 8080:8080 <image>`, and
a [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/)
(`cloudflared tunnel run`) exposes it over HTTPS on a subdomain without
opening any inbound security-group port or provisioning a load balancer
-- a reasonable fallback specifically because it sidesteps the SSE/queue
timeout question entirely (no managed reverse proxy in front imposing an
opinion on connection lifetime).

## Image size

Check with `docker images pronunciation-scorer` after building. If it's
past the 4 GB instance's comfort zone, `docker history <image>` breaks
down which layer is taking the space -- the CPU-only torch wheel and the
baked-in acoustic model weights are the two largest contributors by
design; everything in `.dockerignore` (datasets/, cached posteriors,
features.parquet) is what was cut to keep it there.

## Verified: memory under the 4 GB cap

Image size is disk footprint, not runtime memory -- the two aren't the
same number, so this was checked separately. Ran the built image capped
to the actual App Runner target (`docker run --memory=4g
--memory-swap=4g --cpus=1 ...`, swap disabled so a real overrun shows up
as an OOM kill rather than silently spilling to disk) and drove it with
`gradio_client` against all three `examples/` clips rather than just
polling `/`, to catch any inference-time spike:

- After model load, idle: **513 MB** (12.5% of the cap)
- Peak during inference (three back-to-back requests): **714 MB** (17.4%)
- Settled after requests: **670 MB** (16.4%)
- `docker inspect` confirmed `OOMKilled=false` throughout; no errors in
  `docker logs`.

CPU pegged at ~97-110% of the single allocated vCPU while scoring --
expected, since inference here is unbatched and single-threaded per
request (REPORT.md Section 3.3) and fully saturates the one vCPU it's
given rather than contending for it.

Headroom is large: even at peak, memory usage is under a fifth of the 4
GB budget, so a single request has no realistic path to an OOM kill on
this instance size. This wasn't load-tested for concurrent requests --
see the Auto scaling note in Section 3 (max size = 1) for why that's a
deliberate non-goal here, not an oversight.
