"""Watch Modal spend for this calendar month and stop everything at a limit.

Polls `modal billing report --for "this month"` and appends each reading to
runs/modal_credits.log. When the month's total reaches --limit (default
$25, leaving $5 of the $30 monthly credits), it stops every running app
in the workspace (training, rebuild and the website).

  python scripts/modal_budget.py              # watch, every 5 min
  python scripts/modal_budget.py --once       # print current spend and exit

Modal reports full intervals only, so the reading can lag real spend by up
to about an hour. Hence the margin in the default limit. As a second
backstop, also set a workspace spending limit in the Modal dashboard
(Settings -> Usage & Billing).
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "runs", "modal_credits.log")


def modal_cmd():
    exe = shutil.which("modal") or os.path.join(os.path.dirname(sys.executable), "modal")
    return [exe]


def run(args):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(modal_cmd() + args, capture_output=True, text=True, encoding="utf-8", env=env)


def month_spend():
    r = run(["billing", "report", "--for", "this month", "--json"])
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or r.stdout.strip())
    rows = json.loads(r.stdout)
    per_app = {}
    for row in rows:
        per_app[row.get("description", "?")] = per_app.get(row.get("description", "?"), 0.0) + float(row["cost"])
    return sum(per_app.values()), per_app


def running_apps():
    r = run(["app", "list", "--json"])
    if r.returncode != 0:
        return []
    return [a for a in json.loads(r.stdout) if str(a.get("State", a.get("state", ""))).lower()
            in ("deployed", "running", "ephemeral", "ephemeral (detached)", "detached")]


def stop_all():
    for a in running_apps():
        app_id = a.get("App ID") or a.get("app_id")
        print(f"stopping {a.get('Description') or a.get('description')} ({app_id})", flush=True)
        run(["app", "stop", "-y", app_id])


def log(line):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=float, default=25.0)
    ap.add_argument("--every", type=int, default=300, help="seconds between checks")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    while True:
        now = datetime.datetime.now().isoformat(timespec="seconds")
        try:
            total, per_app = month_spend()
            detail = ", ".join(f"{k}=${v:.2f}" for k, v in sorted(per_app.items()))
            log(f"{now} month spend ${total:.2f} / limit ${a.limit:.2f}  ({detail})")
            if total >= a.limit:
                log(f"{now} LIMIT REACHED: stopping all Modal apps")
                stop_all()
                return
        except Exception as e:  # keep watching through transient API errors
            log(f"{now} billing check failed: {e}")
        if a.once:
            return
        time.sleep(a.every)


if __name__ == "__main__":
    main()
