#!/usr/bin/env python3
"""Acquire a healthy RunPod GPU pod in a region that holds our prepared network volume.

Recipe (see deploy/regions.json):
  1. Build the candidate list: regions in priority order (lowest latency from the
     studio first) x GPU types in preference order, each with a price ceiling.
  2. Each round: try every candidate in strict priority order. A create that fails with
     "no instances" costs nothing and is the only authoritative stock answer; the API's
     stock report is fuzzy (it flips between checks a minute apart), so it is logged but
     never allowed to reorder candidates, or a stale report would send us to a worse region.
  3. For a created pod: wait for SSH, then run acceptance checks
       - CUDA initialises (cuInit == 0)            [seen: broken passthrough, error 999]
       - GPU idle on arrival (util/mem/power low)   [seen: shared 5090 at 100% / 400 W]
       - large-file Hugging Face download speed    [seen: dead links; small files mislead]
       - GitHub reachability (reported; optional)  [seen: Iceland hosts without GitHub]
     A failed host is terminated immediately and the next candidate is tried.
  4. Money: never create below the balance floor, respect per-GPU price ceilings,
     terminate unaccepted pods on every exit path (errors, Ctrl-C, SIGTERM).
  5. Every attempt is appended to .cache/runpod/attempts.jsonl (our own stock history).

CLI:
  python deploy/runpod_acquire.py dry-run                  # show candidates + current stock, create nothing
  python deploy/runpod_acquire.py acquire [--timeout 30] [--release]
        prints the accepted pod as JSON and saves it to .cache/runpod/last_pod.json;
        --release terminates it right after acceptance (for testing the recipe)
  python deploy/runpod_acquire.py release [POD_ID]         # terminate a pod (default: last acquired)

Library:
  from runpod_acquire import load_config, acquire, release
  pod = acquire(load_config())      # dict with id, dc, gpu, ip, ssh_port, proxy_url, ...
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
STATE_DIR = REPO / ".cache" / "runpod"
ATTEMPTS = STATE_DIR / "attempts.jsonl"
LAST_POD = STATE_DIR / "last_pod.json"
REST = "https://rest.runpod.io/v1"
GQL = "https://api.runpod.io/graphql"
UA = "Mozilla/5.0 fluxrt-acquire"   # RunPod's API and proxy reject Python's default User-Agent
SSH_KEY = Path.home() / ".ssh" / "id_ed25519"


# ── API ──────────────────────────────────────────────────────────────────────
def _api_key() -> str:
    txt = (Path.home() / ".runpod" / "config.toml").read_text()
    return re.search(r'apikey\s*=\s*"([^"]+)"', txt).group(1)


def _request(url: str, method: str = "GET", body: dict | None = None) -> tuple[int, dict | list | str]:
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json",
                                          "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read().decode()
            return r.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw


def gql(query: str) -> dict:
    code, data = _request(GQL, "POST", {"query": query})
    if code != 200 or not isinstance(data, dict) or "data" not in data:
        raise RuntimeError(f"GraphQL error {code}: {str(data)[:200]}")
    return data["data"]


def balance() -> float:
    return float(gql("{ myself { clientBalance } }")["myself"]["clientBalance"])


def stock(dc: str, gpu_id: str) -> tuple[str | None, float | None]:
    q = (f'{{ gpuTypes(input:{{id:"{gpu_id}"}}) {{ lowestPrice(input:{{gpuCount:1, secureCloud:true, '
         f'dataCenterId:"{dc}"}}) {{ stockStatus uninterruptablePrice }} }} }}')
    try:
        lp = gql(q)["gpuTypes"][0]["lowestPrice"]
        return lp.get("stockStatus"), lp.get("uninterruptablePrice")
    except Exception:  # noqa: BLE001
        return None, None


def release(pod_id: str) -> bool:
    code, _ = _request(f"{REST}/pods/{pod_id}", "DELETE")
    return code in (200, 204, 404)


# ── helpers ──────────────────────────────────────────────────────────────────
def load_config(path: str | Path = HERE / "regions.json") -> dict:
    return json.loads(Path(path).read_text())


def candidates(cfg: dict, no_volume: bool = False, gpu_filter: list[str] | None = None) -> list[dict]:
    regions = cfg["any_regions"] if no_volume else cfg["regions"]
    gpus = [g for g in cfg["gpus"] if not gpu_filter or g.get("short") in gpu_filter or g["id"] in gpu_filter]
    if cfg.get("order", "region") == "gpu":
        pairs = [(r, g) for g in gpus for r in regions]
    else:
        pairs = [(r, g) for r in regions for g in gpus]
    return [{"dc": r["dc"], "volume": r.get("volume"), "latency_ms": r.get("latency_ms"), "gpu": g["id"],
             "short": g.get("short", g["id"]), "max_price": g.get("max_price")} for r, g in pairs]


def log_attempt(rec: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    rec = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **rec}
    with ATTEMPTS.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def say(msg: str) -> None:
    print(f"[acquire {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def ssh(ip: str, port: int, cmd: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", "-i", str(SSH_KEY), "-p", str(port), "-o", "StrictHostKeyChecking=accept-new",
                           "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", f"root@{ip}", cmd],
                          capture_output=True, text=True, timeout=timeout)


# ── one candidate ────────────────────────────────────────────────────────────
class _Pending:
    """Tracks the pod currently being evaluated so signal handlers can clean it up."""
    pod_id: str | None = None


def _create(cfg: dict, cand: dict) -> tuple[str | None, dict | str]:
    p = cfg["pod"]
    disk = int(cfg.get("_disk_gb") or 0)
    body = {"name": f"fluxrt-{cand['short'].lower()}-{cand['dc'].lower()}", "imageName": p["image"],
            "computeType": "GPU", "cloudType": p.get("cloud", "SECURE"), "gpuTypeIds": [cand["gpu"]], "gpuCount": 1,
            "dataCenterIds": [cand["dc"]], "containerDiskInGb": max(p.get("container_disk_gb", 40), 80 if disk else 0),
            "volumeInGb": 0 if cand.get("volume") else (disk or 150),
            "volumeMountPath": "/workspace", "ports": p["ports"],
            "supportPublicIp": True, "env": p.get("env", {})}
    if cand.get("volume"):
        body["networkVolumeId"] = cand["volume"]
    code, data = _request(f"{REST}/pods", "POST", body)
    if code in (200, 201) and isinstance(data, dict) and data.get("id"):
        return data["id"], data
    return None, data


def _wait_ssh(pod_id: str, deadline: float) -> tuple[str, int]:
    while time.time() < deadline:
        code, d = _request(f"{REST}/pods/{pod_id}")
        pm = (d.get("portMappings") or {}) if isinstance(d, dict) else {}
        if code == 200 and d.get("publicIp") and pm.get("22"):
            ip, port = d["publicIp"], int(pm["22"])
            while time.time() < deadline:
                try:
                    if ssh(ip, port, "true", timeout=25).returncode == 0:
                        return ip, port
                except subprocess.TimeoutExpired:
                    pass
                time.sleep(5)
        time.sleep(8)
    raise TimeoutError("ssh never became usable")


def _check_host(cfg: dict, ip: str, port: int) -> tuple[bool, dict]:
    c = cfg["checks"]
    probe = (
        "nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits; "
        "python3 -c \"import ctypes; print('cuinit', ctypes.CDLL('libcuda.so.1').cuInit(0))\" 2>/dev/null || echo 'cuinit fail'; "
        f"curl -sL -r 0-209715199 -o /dev/null -m 20 -w 'hf %{{speed_download}}\\n' '{c['hf_probe_url']}'; "
        "timeout 25 git ls-remote https://github.com/domprosys/FluxRT.git HEAD >/dev/null 2>&1 && echo 'github ok' || echo 'github fail'"
    )
    out = ssh(ip, port, probe, timeout=120).stdout
    r: dict = {"raw": out.strip()[-400:]}
    try:
        name, drv, util, mem, power = [x.strip() for x in out.splitlines()[0].split(",")]
        r.update(gpu_name=name, driver=drv, util_pct=float(util), mem_mib=float(mem), power_w=float(power))
    except Exception:  # noqa: BLE001
        return False, {**r, "reason": "nvidia-smi unreadable"}
    m = re.search(r"cuinit (\S+)", out)
    r["cuinit"] = m.group(1) if m else "?"
    m = re.search(r"hf ([0-9.]+)", out)
    r["hf_mbps"] = round(float(m.group(1)) / 1e6, 1) if m else 0.0
    r["github"] = "github ok" in out
    reasons = []
    if r["cuinit"] != "0":
        reasons.append(f"cuInit={r['cuinit']}")
    if r["util_pct"] > c["idle_max_util_pct"] or r["mem_mib"] > c["idle_max_mem_mib"] or r["power_w"] > c["idle_max_power_w"]:
        reasons.append(f"GPU busy on arrival ({r['util_pct']:.0f}% {r['mem_mib']:.0f}MiB {r['power_w']:.0f}W)")
    if r["hf_mbps"] < c["min_hf_mbps"]:
        reasons.append(f"slow HF download {r['hf_mbps']} MB/s")
    if c.get("require_github") and not r["github"]:
        reasons.append("GitHub unreachable")
    r["reason"] = "; ".join(reasons) or None
    return not reasons, r


# ── main loop ────────────────────────────────────────────────────────────────
def acquire(cfg: dict, timeout_min: float | None = None, dry_run: bool = False,
            no_volume: bool = False, gpu_filter: list[str] | None = None) -> dict | None:
    poll = cfg["poll"]
    deadline = time.time() + 60 * (timeout_min if timeout_min is not None else poll["timeout_min"])
    cands = candidates(cfg, no_volume=no_volume, gpu_filter=gpu_filter)
    rnd = 0
    while True:
        rnd += 1
        bal = balance()
        if bal < cfg["balance_floor_usd"]:
            say(f"balance ${bal:.2f} is below the floor ${cfg['balance_floor_usd']:.2f}; not creating pods")
            return None
        # stock report: informational only (logged); candidates are always tried in priority order
        for cnd in cands:
            cnd["stock"], cnd["price"] = stock(cnd["dc"], cnd["gpu"])
        ranked = cands
        if dry_run:
            say(f"balance ${bal:.2f}; candidates in priority order (stock is indicative only):")
            for i, cnd in enumerate(cands, 1):
                print(f"  {i}. {cnd['dc']:9} {cnd['short']:10} vol={cnd['volume'] or 'none':10} ceiling ${cnd['max_price']:.2f}  "
                      f"stock={cnd['stock'] or '-'} price={cnd['price'] or '-'}  latency~{cnd['latency_ms']} ms")
            return None
        say(f"round {rnd}: balance ${bal:.2f}; reported stock: "
            + (", ".join(f"{c['dc']}/{c['short']}" for c in ranked if c["stock"]) or "none"))
        for cnd in ranked:
            if time.time() > deadline:
                break
            if cnd["price"] and cnd["max_price"] and cnd["price"] > cnd["max_price"]:
                continue
            base = {"dc": cnd["dc"], "gpu": cnd["short"], "reported_stock": cnd["stock"], "round": rnd}
            pod_id, info = _create(cfg, cnd)
            if not pod_id:
                msg = info.get("error") if isinstance(info, dict) else str(info)
                log_attempt({**base, "outcome": "unavailable", "detail": str(msg)[:160]})
                continue
            _Pending.pod_id = pod_id
            cost = info.get("costPerHr")
            if cnd["max_price"] and cost and float(cost) > cnd["max_price"]:
                release(pod_id); _Pending.pod_id = None
                log_attempt({**base, "outcome": "over_price", "pod": pod_id, "price": cost})
                continue
            say(f"created {pod_id}: {cnd['dc']} {cnd['short']} ${cost}/hr, checking host...")
            t0 = time.time()
            try:
                ip, port = _wait_ssh(pod_id, min(deadline + 600, time.time() + poll["ssh_timeout_s"]))
                ok, checks = _check_host(cfg, ip, port)
            except Exception as e:  # noqa: BLE001
                ok, checks, ip, port = False, {"reason": f"{type(e).__name__}: {e}"}, None, None
            rec = {**base, "pod": pod_id, "price": cost, "ip": ip, "secs_to_checked": round(time.time() - t0),
                   **{k: v for k, v in checks.items() if k != "raw"}}
            if not ok:
                release(pod_id); _Pending.pod_id = None
                log_attempt({**rec, "outcome": "rejected"})
                say(f"  rejected: {checks.get('reason')}")
                continue
            log_attempt({**rec, "outcome": "accepted"})
            result = {"id": pod_id, "dc": cnd["dc"], "gpu": cnd["gpu"], "volume": cnd["volume"], "ip": ip,
                      "ssh_port": port, "proxy_url": f"https://{pod_id}-8000.proxy.runpod.net", "cost_per_hr": cost,
                      "latency_ms": cnd["latency_ms"], "github_ok": checks["github"], "checks": checks,
                      "acquired_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            LAST_POD.write_text(json.dumps(result, indent=2))
            _Pending.pod_id = None
            say(f"  ACCEPTED {pod_id} in {round(time.time() - t0)}s: {checks['gpu_name']}, "
                f"HF {checks['hf_mbps']} MB/s, github {'ok' if checks['github'] else 'unreachable'}")
            return result
        if time.time() > deadline:
            say("timed out without an acceptable pod")
            return None
        say(f"nothing acceptable this round; next round in {poll['interval_s']}s")
        time.sleep(poll["interval_s"])


def _cleanup_and_exit(signum=None, frame=None):
    if _Pending.pod_id:
        say(f"interrupted: terminating unaccepted pod {_Pending.pod_id}")
        release(_Pending.pod_id)
        log_attempt({"pod": _Pending.pod_id, "outcome": "aborted"})
    sys.exit(130 if signum else 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dry-run")
    a = sub.add_parser("acquire")
    a.add_argument("--timeout", type=float, default=None, help="minutes (default from config)")
    a.add_argument("--release", action="store_true", help="terminate right after acceptance (recipe test)")
    a.add_argument("--no-volume", action="store_true", help="any region, pod-local disk (cold setup)")
    a.add_argument("--gpus", default=None, help="comma-separated GPU short names to allow, e.g. PRO6000-S,PRO6000-W")
    a.add_argument("--disk-gb", type=int, default=0, help="pod-local /workspace size for --no-volume (default 150)")
    r = sub.add_parser("release")
    r.add_argument("pod_id", nargs="?")
    g = sub.add_parser("guard", help="terminate a pod after a hard time cap (run in the background)")
    g.add_argument("pod_id")
    g.add_argument("--max-hours", type=float, required=True)
    g.add_argument("--since", type=float, default=None, help="epoch seconds the cap counts from (default: now)")
    ap.add_argument("--config", default=str(HERE / "regions.json"))
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.cmd == "dry-run":
        acquire(cfg, dry_run=True)
        return 0
    if args.cmd == "guard":
        t0 = args.since or time.time(); cap = t0 + args.max_hours * 3600
        say(f"guard: {args.pod_id} will be terminated at {time.strftime('%H:%M:%S', time.localtime(cap))}")
        warned = False
        while time.time() < cap:
            code, d = _request(f"{REST}/pods/{args.pod_id}")
            if code == 404 or (isinstance(d, dict) and d.get("desiredStatus") in ("TERMINATED", "EXITED")):
                say("guard: pod already gone"); return 0
            if not warned and cap - time.time() < 1800:
                say(f"guard: 30 minutes left on {args.pod_id}"); warned = True
            time.sleep(60)
        say(f"guard: hard cap reached, terminating {args.pod_id}")
        release(args.pod_id); log_attempt({"pod": args.pod_id, "outcome": "guard_terminated"})
        return 0
    if args.cmd == "release":
        pid = args.pod_id or json.loads(LAST_POD.read_text())["id"]
        ok = release(pid)
        say(f"release {pid}: {'ok' if ok else 'failed'}")
        return 0 if ok else 1

    signal.signal(signal.SIGTERM, _cleanup_and_exit)
    signal.signal(signal.SIGINT, _cleanup_and_exit)
    try:
        if args.disk_gb:
            cfg["_disk_gb"] = args.disk_gb
        pod = acquire(cfg, timeout_min=args.timeout, no_volume=args.no_volume,
                      gpu_filter=args.gpus.split(",") if args.gpus else None)
    except Exception:
        _cleanup_and_exit()
        raise
    if not pod:
        return 2
    print(json.dumps(pod, indent=2))
    if args.release:
        say(f"--release: terminating {pod['id']}")
        release(pod["id"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
