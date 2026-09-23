#!/usr/bin/env python3
"""Deploy-to-ready timing + backend performance on one RunPod GPU type.

    python deploy/runpod_bench.py --gpu "NVIDIA GeForce RTX 5090" --dc EUR-IS-1 --volume <id> --tag 5090

Timeline recorded (seconds from pod creation): ssh port mapped, ssh usable,
bootstrap done (venvs restored or built, weights present), SD server ready
(via the RunPod proxy), FluxRT server ready. Then test_backend runs for the
sd, sd_controlnet and fluxrt configs and the per-frame time is parsed. The pod
is terminated at the end (also on failure). Results: .cache/bench/<tag>.json.

Needs: ~/.runpod/config.toml (API key), ~/.ssh/id_ed25519 registered on RunPod.
"""
import argparse, json, re, subprocess, sys, time, urllib.request
from pathlib import Path

REST = "https://rest.runpod.io/v1"
IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
UA = {"User-Agent": "Mozilla/5.0 fluxrt-bench"}


def api_key():
    m = re.search(r'apikey\s*=\s*"([^"]+)"', Path.home().joinpath(".runpod/config.toml").read_text())
    return m.group(1)


def rest(method, path, body=None):
    req = urllib.request.Request(f"{REST}{path}", method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
        return json.loads(raw) if raw else {}


class Pod:
    def __init__(self, gpu, dc, volume, name):
        self.t0 = time.time()
        self.info = rest("POST", "/pods", {
            "name": name, "imageName": IMAGE, "computeType": "GPU", "cloudType": "SECURE",
            "gpuTypeIds": [gpu], "gpuCount": 1, "dataCenterIds": [dc],
            "containerDiskInGb": 40, "volumeInGb": 0, "volumeMountPath": "/workspace",
            "networkVolumeId": volume, "ports": ["8000/http", "22/tcp"], "supportPublicIp": True,
            "env": {"HF_HOME": "/workspace/hf"},
        })
        self.id = self.info["id"]
        self.cost_hr = self.info.get("costPerHr")
        self.ip = self.port = None

    def elapsed(self):
        return round(time.time() - self.t0, 1)

    def wait_ssh(self, timeout=900):
        while time.time() - self.t0 < timeout:
            d = rest("GET", f"/pods/{self.id}")
            pm = d.get("portMappings") or {}
            if d.get("publicIp") and pm.get("22"):
                self.ip, self.port = d["publicIp"], int(pm["22"])
                t_mapped = self.elapsed()
                while time.time() - self.t0 < timeout:
                    if self.ssh("true", timeout=20).returncode == 0:
                        return t_mapped, self.elapsed()
                    time.sleep(5)
            time.sleep(10)
        raise TimeoutError("ssh never came up")

    def ssh(self, cmd, timeout=3600):
        return subprocess.run(
            ["ssh", "-i", str(Path.home() / ".ssh/id_ed25519"), "-p", str(self.port),
             "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30",
             f"root@{self.ip}", cmd], capture_output=True, text=True, timeout=timeout)

    def proxy_ready(self, timeout):
        url = f"https://{self.id}-8000.proxy.runpod.net/api/state"
        t = time.time()
        while time.time() - t < timeout:
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=15) as r:
                    d = json.load(r)
                if d.get("ready"):
                    return self.elapsed(), d
                if d.get("alive") is False:
                    raise RuntimeError("backend died")
            except RuntimeError:
                raise
            except Exception:
                pass
            time.sleep(5)
        raise TimeoutError("server never became ready")

    def terminate(self):
        try:
            rest("DELETE", f"/pods/{self.id}")
        except Exception as e:  # noqa: BLE001
            print("terminate failed:", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--dc", required=True)
    ap.add_argument("--volume", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--skip-fluxrt", action="store_true")
    a = ap.parse_args()
    out = Path(".cache/bench"); out.mkdir(parents=True, exist_ok=True)
    res = {"gpu": a.gpu, "dc": a.dc, "volume": a.volume, "timeline_s": {}, "perf": {}, "notes": []}

    def save():
        (out / f"{a.tag}.json").write_text(json.dumps(res, indent=2))

    pod = None

    def log(msg):
        print(f"[{a.tag} +{pod.elapsed() if pod else 0}s] {msg}", flush=True)
    try:
        pod = Pod(a.gpu, a.dc, a.volume, f"bench-{a.tag}")
        res["pod_id"], res["cost_per_hr"] = pod.id, pod.cost_hr
        log(f"pod {pod.id} created, ${pod.cost_hr}/hr")
        t_mapped, t_ssh = pod.wait_ssh()
        res["timeline_s"]["ssh_port_mapped"], res["timeline_s"]["ssh_usable"] = t_mapped, t_ssh
        log(f"ssh usable ({pod.ip}:{pod.port})")
        r = pod.ssh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; nproc; free -g | awk '/Mem/{print $2\"G RAM\"}'")
        res["machine"] = r.stdout.strip().replace("\n", " | ")
        log(res["machine"])

        cold = pod.ssh("test -f /workspace/venvs/venvs.tar && echo warm || echo cold").stdout.strip()
        res["volume_state"] = cold
        r = pod.ssh("curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; export PATH=$HOME/.local/bin:$PATH; "
                    "mkdir -p /workspace && cd /workspace && "
                    "([ -d fluxrt/.git ] || git clone -q --branch installation-controls https://github.com/domprosys/FluxRT.git fluxrt) && "
                    "cd fluxrt && git fetch -q origin 2>/dev/null; git fetch -q fork 2>/dev/null; "
                    "git reset -q --hard $(git rev-parse --verify -q fork/installation-controls || git rev-parse origin/installation-controls) && "
                    "bash deploy/runpod_setup.sh > /workspace/bench_setup.log 2>&1; echo exit=$?; tail -n 3 /workspace/bench_setup.log",
                    timeout=3600)
        res["timeline_s"]["bootstrap_done"] = pod.elapsed()
        res["bootstrap_tail"] = r.stdout.strip()[-400:]
        log(f"bootstrap done ({cold} volume): {r.stdout.strip().splitlines()[0] if r.stdout.strip() else r.stderr[-200:]}")
        if "exit=0" not in r.stdout:
            raise RuntimeError("bootstrap failed: " + r.stdout[-800:] + r.stderr[-400:])

        # SD server: deploy-to-ready
        pod.ssh("cd /workspace/fluxrt && HF_HOME=/workspace/hf nohup .venv/bin/python scripts/serve_web.py --config configs/sd_config.json --port 8000 > /workspace/serve_sd.log 2>&1 &")
        t, st = pod.proxy_ready(900)
        res["timeline_s"]["sd_server_ready"] = t
        res["perf"]["sd_server_state"] = {k: st.get(k) for k in ("model", "warmup_fps", "gpu_reserved_mb", "load_s")}
        log(f"SD server ready via proxy (warmup {st.get('warmup_fps')} fps)")
        pod.ssh("pkill -INT -f 'serve_web.p[y]'; sleep 3; pkill -9 -f 'sd_worke[r]'; true")

        if not a.skip_fluxrt:
            pod.ssh("cd /workspace/fluxrt && HF_HOME=/workspace/hf nohup .venv/bin/python scripts/serve_web.py --config configs/web_config.json --port 8000 > /workspace/serve_fluxrt.log 2>&1 &")
            t, st = pod.proxy_ready(1500)
            res["timeline_s"]["fluxrt_server_ready"] = t
            res["perf"]["fluxrt_server_state"] = {k: st.get(k) for k in ("gpu_reserved_mb",)}
            log("FluxRT server ready via proxy")
            pod.ssh("pkill -INT -f 'serve_web.p[y]'; sleep 5; pkill -9 -f 'spawn_mai[n]'; true")

        # per-backend throughput with synthetic input
        for cfg, secs in (("sd_config", 15), ("sd_controlnet_config", 15)) + (() if a.skip_fluxrt else (("web_config", 20),)):
            r = pod.ssh(f"cd /workspace/fluxrt && HF_HOME=/workspace/hf timeout 1500 .venv/bin/python scripts/test_backend.py "
                        f"--config configs/{cfg}.json --device -1 --seconds {secs} --out /workspace/bench_{a.tag}_{cfg} 2>&1 | "
                        f"grep -E 'ready in|proc=|saved|died|Error' | tail -n 6; pkill -9 -f 'spawn_mai[n]'; true", timeout=1800)
            lines = r.stdout.strip().splitlines()
            procs = [float(m.group(1)) for l in lines for m in [re.search(r"proc=([0-9.]+)s", l)] if m]
            ready = next((float(m.group(1)) for l in lines for m in [re.search(r"ready in ([0-9.]+)s", l)] if m), None)
            gpu = next((int(m.group(1)) for l in lines for m in [re.search(r"gpu=(\d+)MB", l)] if m), None)
            res["perf"][cfg] = {"ready_s": ready, "proc_ms": round(1000 * sum(procs) / len(procs), 1) if procs else None,
                                "fps": round(len(procs) / sum(procs), 1) if procs and sum(procs) > 0 else None,
                                "gpu_mb": gpu, "raw": lines[-3:]}
            log(f"{cfg}: ready {ready}s, {res['perf'][cfg]['proc_ms']} ms/frame, gpu {gpu} MB")
            save()
    except Exception as e:  # noqa: BLE001
        res["error"] = repr(e)
        print(f"[{a.tag}] ERROR: {e!r}", flush=True)
    finally:
        if pod is not None:
            res["timeline_s"]["terminated"] = pod.elapsed()
            res["cost_estimate_usd"] = round((pod.elapsed() / 3600) * float(pod.cost_hr or 0), 2)
            pod.terminate()
            print(f"[{a.tag}] pod terminated after {pod.elapsed()}s, est ${res['cost_estimate_usd']}", flush=True)
        save()
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
