#!/usr/bin/env python3
"""Show, diff or publish the private RunPod template "fluxrt-realtime-stylization" from this repo.

    .venv/bin/python deploy/runpod_template.py show      # the live definition (secrets masked)
    .venv/bin/python deploy/runpod_template.py diff      # what publish would change
    .venv/bin/python deploy/runpod_template.py publish   # update the template in place

The start command embeds deploy/template_post_start.sh (written to /post_start.sh, which RunPod's
/start.sh runs after SSH is up); the readme is deploy/template_readme.md. Everything else the pod
does comes from the fork at run time (deploy/pod_start.sh), so only changes to the hook, env, disks
or readme need a publish.
"""
import base64
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runpod_acquire as ra  # noqa: E402  (REST helper with the browser User-Agent + API key)

TEMPLATE_ID = "6psyho3eiw"
SECRET_HINTS = ("TOKEN", "KEY", "SECRET", "PASSWORD")


def desired() -> dict:
    hook = base64.b64encode((HERE / "template_post_start.sh").read_bytes()).decode()
    return {
        "name": "fluxrt-realtime-stylization",
        "imageName": "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
        "dockerStartCmd": ["bash", "-c",
                           f"echo {hook} | base64 -d > /post_start.sh && chmod +x /post_start.sh && exec /start.sh"],
        # container disk = the install root (WS=/root/ws): a full install with every engine incl. the
        # 14B is ~115 GB, plus the uv cache and optional TensorRT engines (~21 GB)
        "containerDiskInGb": 200,
        "volumeInGb": 20,  # /workspace: logs only (it can be a network filesystem)
        "volumeMountPath": "/workspace",
        "ports": ["8000/http", "22/tcp"],
        # multi_all: all four engines resident (RTX PRO 6000 class); stop the pod after 30 idle minutes.
        # HF_TOKEN comes from the RunPod secret "hf_token" (authenticated downloads + the TensorRT engine
        # cache in TRT_CACHE_REPO, used when a deploy sets ENABLE_TRT=1)
        "env": {"BACKEND_CONFIG": "multi_all_config", "WS": "/root/ws", "IDLE_STOP_MIN": "30",
                "HF_TOKEN": "{{ RUNPOD_SECRET_hf_token }}", "TRT_CACHE_REPO": "alexcloak/fluxrt-trt-engines"},
        "readme": (HERE / "template_readme.md").read_text(),
    }


def masked(d: dict) -> dict:
    out = dict(d)
    if isinstance(out.get("env"), dict):
        out["env"] = {k: ("***" if any(h in k.upper() for h in SECRET_HINTS) else v) for k, v in out["env"].items()}
    if isinstance(out.get("dockerStartCmd"), list):
        out["dockerStartCmd"] = [s if len(s) < 120 else s[:60] + f"...({len(s)} chars)" for s in out["dockerStartCmd"]]
    return out


def live() -> dict:
    code, d = ra._request(f"{ra.REST}/templates/{TEMPLATE_ID}")
    if code != 200 or not isinstance(d, dict):
        sys.exit(f"cannot read template {TEMPLATE_ID}: HTTP {code} {d}")
    return d


def missing_secrets(env: dict) -> list[str]:
    """Names of RunPod secrets the env references ({{ RUNPOD_SECRET_<name> }}) that don't exist."""
    wanted = set(re.findall(r"\{\{\s*RUNPOD_SECRET_(\w+)\s*\}\}", json.dumps(env)))
    if not wanted:
        return []
    have = {s["name"] for s in ra.gql("{ myself { secrets { name } } }")["myself"]["secrets"]}
    return sorted(wanted - have)


def changes(cur: dict, want: dict) -> dict:
    return {k: v for k, v in want.items() if cur.get(k) != v}


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "show":
        print(json.dumps(masked(live()), indent=1))
    elif cmd in ("diff", "publish"):
        cur, want = live(), desired()
        diff = changes(cur, want)
        if not diff:
            print("template is up to date")
            return 0
        for k, v in diff.items():
            print(f"{k}: {json.dumps(masked({k: cur.get(k)})[k])[:200]}\n  -> {json.dumps(masked({k: v})[k])[:200]}")
        if cmd == "publish":
            gone = missing_secrets(want["env"])
            if gone:
                print(f"not published: create the RunPod secret(s) {gone} first (console: Secrets)")
                return 1
            code, d = ra._request(f"{ra.REST}/templates/{TEMPLATE_ID}", "PATCH", diff)
            print(f"PATCH -> HTTP {code}")
            if code not in (200, 201):
                print(json.dumps(d)[:500] if not isinstance(d, str) else d[:500])
                return 1
            left = changes(live(), want)
            print("verified: template matches the repo" if not left else f"still different: {sorted(left)}")
            return 0 if not left else 1
    else:
        sys.exit(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
