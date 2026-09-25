#!/usr/bin/env python3
"""Deploy the RunPod template as published, verify it end to end, terminate the pod.

    .venv/bin/python deploy/verify_template.py [--gpus PRO6000-S,PRO6000-W] [--config multi_config]
        [--idle-min 3] [--clip .cache/bench/clip.mp4] [--stream-s 90] [--max-min 60] [--env ENABLE_TRT=1]

Creates a pod from the template with the template's own env plus a random ACCESS_TOKEN and
IDLE_STOP_MIN=--idle-min (--config overrides BACKEND_CONFIG), trying the regions of regions.json
"any_regions" in order. Times the progress page and readiness, streams the clip from this machine
through RunPod's proxy as the stage while switching through every engine, records the pod's env and
disk layout over ssh, then waits for the server's idle auto-stop to stop the pod (--idle-min 0 skips
that) and terminates it. Results: .cache/runpod/verify_<pod>.json (+ _logs.txt). Costs one pod run.
"""
import argparse
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import runpod_acquire as ra  # noqa: E402
from runpod_template import TEMPLATE_ID  # noqa: E402

OUT = REPO / ".cache" / "runpod"


def say(msg: str) -> None:
    print(f"[verify {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def api(url: str, token: str, method: str = "GET", body: dict | None = None):
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"User-Agent": ra.UA, "X-Access-Token": token, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def create(args, env: dict) -> tuple[str, str, dict]:
    cfg = ra.load_config()
    gpus = {g["short"]: g["id"] for g in cfg["gpus"]}
    wanted = [gpus[s] for s in args.gpus.split(",")]
    deadline = time.time() + 20 * 60
    while time.time() < deadline:
        for dc in [r["dc"] for r in cfg["any_regions"]]:
            for gpu in wanted:
                body = {"name": "fluxrt-verify", "templateId": TEMPLATE_ID, "computeType": "GPU", "cloudType": "SECURE",
                        "gpuTypeIds": [gpu], "gpuCount": 1, "dataCenterIds": [dc], "supportPublicIp": True, "env": env}
                code, d = ra._request(f"{ra.REST}/pods", "POST", body)
                if code in (200, 201) and isinstance(d, dict) and d.get("id"):
                    return d["id"], dc, d
        say("no stock for the requested GPUs in any region; retrying in 2 min")
        time.sleep(120)
    sys.exit("no capacity within 20 min")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", default="PRO6000-S,PRO6000-W")
    ap.add_argument("--config", default=None, help="BACKEND_CONFIG override (default: the template's)")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="extra pod env, e.g. ENABLE_TRT=1")
    ap.add_argument("--idle-min", type=float, default=3.0)
    ap.add_argument("--clip", default=str(REPO / ".cache" / "bench" / "clip.mp4"))
    ap.add_argument("--stream-s", type=int, default=90)
    ap.add_argument("--max-min", type=float, default=60, help="hard cap on the pod's lifetime")
    args = ap.parse_args()

    floor = ra.load_config().get("balance_floor_usd", 3.0)
    bal = ra.balance()
    if bal < floor:
        sys.exit(f"balance ${bal:.2f} is below the ${floor} floor")
    code, tpl = ra._request(f"{ra.REST}/templates/{TEMPLATE_ID}")
    if code != 200:
        sys.exit(f"cannot read template: HTTP {code}")
    token = secrets.token_urlsafe(12)
    env = dict(tpl.get("env") or {}, ACCESS_TOKEN=token, IDLE_STOP_MIN=str(args.idle_min))
    if args.config:
        env["BACKEND_CONFIG"] = args.config
    env.update(kv.split("=", 1) for kv in args.env)
    say(f"balance ${bal:.2f}; template env {({k: v for k, v in env.items() if k != 'ACCESS_TOKEN'})}")

    pod, dc, created = create(args, env)
    t0 = time.time()
    cap = t0 + args.max_min * 60
    url = f"https://{pod}-8000.proxy.runpod.net"
    res = {"pod": pod, "dc": dc, "cost_per_hr": created.get("costPerHr"), "env": {k: v for k, v in env.items() if k != "ACCESS_TOKEN"},
           "container_disk_gb": created.get("containerDiskInGb"), "volume_gb": created.get("volumeInGb")}
    say(f"created {pod} in {dc} (${created.get('costPerHr')}/h, container {created.get('containerDiskInGb')} GB, "
        f"volume {created.get('volumeInGb')} GB)")
    ssh_ip = ssh_port = None
    try:
        # ── deploy -> progress page -> ready ─────────────────────────────────
        while time.time() < cap:
            try:
                st = api(url + "/api/state", token)
                if st.get("ready") and (not st.get("engines") or all(e["ready"] for e in st["engines"])):
                    res["ready_s"] = round(time.time() - t0)
                    res["engines"] = [e["name"] for e in st.get("engines") or []]
                    say(f"READY after {res['ready_s']}s: {st.get('backend')} engines={res['engines']}")
                    break
            except urllib.error.HTTPError as e:
                if e.code == 503 and "progress_page_s" not in res:
                    res["progress_page_s"] = round(time.time() - t0)
                    say(f"progress page up after {res['progress_page_s']}s")
                elif e.code == 401:
                    say("401: the access token was rejected")
                if e.code == 503:
                    try:
                        page = urllib.request.urlopen(urllib.request.Request(url + "/", headers={"User-Agent": ra.UA, "X-Access-Token": token}), timeout=20).read().decode()
                        if "Setup FAILED" in page:
                            res["setup_failed_s"] = round(time.time() - t0)
                            say("progress page reports SETUP FAILED")
                            break
                    except Exception:  # noqa: BLE001
                        pass
            except Exception:  # noqa: BLE001  (proxy not up yet, timeouts)
                pass
            time.sleep(10)

        code, d = ra._request(f"{ra.REST}/pods/{pod}")
        if isinstance(d, dict):
            ssh_ip, ssh_port = d.get("publicIp"), (d.get("portMappings") or {}).get("22")

        # ── stream from here through the proxy, switching through every engine ──
        if res.get("ready_s"):
            cenv = dict(os.environ, ACCESS_TOKEN=token)
            outdir = OUT / f"verify_{pod}_client"
            client = subprocess.Popen([sys.executable, str(REPO / "scripts" / "webrtc_client_test.py"), "--url", url,
                                       "--role", "stage", "--device", args.clip, "--seconds", str(args.stream_s),
                                       "--out", str(outdir)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=cenv)
            engines = res.get("engines") or []
            order = [e for e in engines if e != api(url + "/api/state", token).get("active_engine")] + engines[:1]
            gap = args.stream_s / (len(order) + 1) if order else 0
            res["switches"] = []
            for name in order:
                time.sleep(gap)
                try:
                    r = api(url + "/api/engine", token, "POST", {"name": name})
                    res["switches"].append({"engine": name, "ok": r.get("active_engine") == name})
                except urllib.error.HTTPError as e:
                    res["switches"].append({"engine": name, "ok": False, "http": e.code})
                say(f"switched to {name}: {res['switches'][-1]}")
            out, _ = client.communicate(timeout=args.stream_s + 120)
            res["stream"] = [ln for ln in out.splitlines() if ln.startswith(("connection", "first output", "saved")) or " recv=" in ln][-14:]
            say("stream: " + " | ".join(ln for ln in res["stream"] if ln.startswith(("connection", "first output"))))
            for ln in res["stream"]:
                if " recv=" in ln:
                    print("   ", ln[:150])
            res["stream_end_s"] = round(time.time() - t0)

        # ── the pod's view: env, runpodctl, layout, logs ─────────────────────
        if ssh_ip and ssh_port:
            r = ra.ssh(ssh_ip, int(ssh_port), (
                # the container's env (pid 1): ssh sessions don't inherit it; secrets reported as set/unset only
                "tr '\\0' '\\n' < /proc/1/environ | grep -E '^(WS|IDLE_STOP_MIN|IDLE_ACTION|BACKEND_CONFIG|ENABLE_TRT|TRT_CACHE_REPO)=' | sort; "
                "tr '\\0' '\\n' < /proc/1/environ | awk -F= '/^(HF_TOKEN|RUNPOD_API_KEY)=/{v=substr($0, length($1)+2); "
                "print $1 \"=\" (v ~ /[{][{]/ ? \"UNRESOLVED\" : (length(v) ? \"set\" : \"empty\"))}'; "
                "grep -h 'trt-cache' /workspace/logs/setup.log /workspace/logs/pod_start.log 2>/dev/null | tail -3; "
                "command -v runpodctl && runpodctl version 2>&1 | head -1; echo ---; df -hT / /workspace; echo ---; "
                "du -sh /root/ws/* /root/venvs 2>/dev/null | sort -h | tail -8; echo ---; cat /workspace/logs/netcheck.txt; "
                "tail -n 15 /workspace/logs/pod_start.log; echo ---; grep -E 'idle|runpodctl' /workspace/logs/server.log | tail -5"),
                timeout=120)
            (OUT / f"verify_{pod}_logs.txt").write_text(r.stdout)
            res["pod_view"] = r.stdout.split("---")[0].strip().splitlines()
            say("pod env: " + "; ".join(res["pod_view"]))

        # ── TensorRT engine cache: wait for pod_start's background push of newly built engines ──
        if env.get("ENABLE_TRT") == "1" and ssh_ip and ssh_port and res.get("ready_s"):
            until = min(cap - 60, time.time() + 25 * 60)
            say("waiting for the engine-cache upload (trt_cache_push.log)")
            while time.time() < until:
                r = ra.ssh(ssh_ip, int(ssh_port), "cat /workspace/logs/trt_cache_push.log 2>/dev/null; "
                           "pgrep -f 'trt_engine_cache.sh push' >/dev/null && echo RUNNING || echo IDLE", timeout=60)
                out = r.stdout.strip()
                if "pushed" in out or ("IDLE" in out and out != "IDLE"):
                    res["cache_push"] = out.replace("IDLE", "").strip()[-300:]
                    say(f"engine cache: {res['cache_push']}")
                    break
                if out == "IDLE" and time.time() - t0 > res["ready_s"] + 120:
                    res["cache_push"] = "no push started (no new engines?)"
                    say(res["cache_push"])
                    break
                time.sleep(30)
            else:
                say("engine-cache upload did not finish in time")

        # ── idle auto-stop: the server should stop the pod by itself ─────────
        if args.idle_min > 0 and res.get("ready_s"):
            wait_until = min(cap, time.time() + args.idle_min * 60 + 180)
            say(f"waiting up to {round((wait_until - time.time()) / 60, 1)} min for the idle auto-stop")
            while time.time() < wait_until:
                code, d = ra._request(f"{ra.REST}/pods/{pod}")
                status = d.get("desiredStatus") if isinstance(d, dict) else None
                if status in ("EXITED", "TERMINATED") or code == 404:
                    res["idle_stopped_after_stream_s"] = round(time.time() - t0 - res.get("stream_end_s", 0))
                    say(f"pod stopped itself ({status}) {res['idle_stopped_after_stream_s']}s after the stream ended")
                    break
                time.sleep(20)
            else:
                say("idle auto-stop did NOT happen in time")
    finally:
        ra.release(pod)
        res["billed_s"] = round(time.time() - t0)
        res["balance_after"] = round(ra.balance(), 2)
        (OUT / f"verify_{pod}.json").write_text(json.dumps(res, indent=1))
        say(f"terminated after {res['billed_s']}s; balance ${res['balance_after']}")
        print("RESULT " + json.dumps(res), flush=True)
    return 0 if res.get("ready_s") and all(s["ok"] for s in res.get("switches", [])) else 1


if __name__ == "__main__":
    sys.exit(main())
