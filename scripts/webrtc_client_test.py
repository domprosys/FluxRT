"""Headless WebRTC client for serve_web.py: sends webcam, synthetic or video-file frames as the
stage (or watches as a viewer), receives the styled output, reports fps/latency, saves samples.

    .venv/bin/python scripts/webrtc_client_test.py --url https://<pod>-8000.proxy.runpod.net \
        --role stage --device clip.mp4 --seconds 60 --out out/
ACCESS_TOKEN in the environment is sent as the X-Access-Token header.
"""
import argparse, asyncio, fractions, json, os, time, urllib.request
import cv2, numpy as np
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) fluxrt-webrtc-test"}  # RunPod's proxy rejects Python UAs
if os.environ.get("ACCESS_TOKEN"):
    UA["X-Access-Token"] = os.environ["ACCESS_TOKEN"]
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, VideoStreamTrack
from av import VideoFrame


class CamTrack(VideoStreamTrack):
    def __init__(self, device):
        super().__init__()
        src = int(device) if str(device).lstrip("-").isdigit() else device
        self.cap = None if src == -1 else cv2.VideoCapture(src)
        self.is_file = isinstance(src, str)
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.n = 0
        self.last = None

    async def recv(self):
        pts, tb = await self.next_timestamp()
        ok, img = (self.cap.read() if self.cap is not None else (False, None))
        if not ok and self.cap is not None and getattr(self, "is_file", False):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ok, img = self.cap.read()
        if not ok:
            img = np.zeros((720, 1280, 3), np.uint8)
            cv2.circle(img, (640 + int(300 * np.sin(self.n / 15)), 360), 120, (40, 200, 255), -1)
            cv2.putText(img, f"synthetic {self.n}", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        self.n += 1
        self.last = img
        f = VideoFrame.from_ndarray(img, format="bgr24")
        f.pts, f.time_base = pts, tb
        return f


async def main(a):
    base = a.url.rstrip("/")
    ice = json.load(urllib.request.urlopen(urllib.request.Request(f"{base}/api/ice-servers", headers=UA)))["iceServers"]
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[RTCIceServer(**s) for s in ice]))
    got = {"n": 0, "t0": None, "frames": []}

    pc.addTransceiver("video", direction="recvonly")
    cam = CamTrack(a.device)
    pc.addTransceiver(cam, direction="sendonly")

    @pc.on("track")
    def on_track(track):
        async def pull():
            while True:
                try:
                    fr = await track.recv()
                except Exception:
                    return
                if got["t0"] is None:
                    got["t0"] = time.monotonic()
                    print(f"first output frame after {time.monotonic() - t_start:.2f}s", flush=True)
                got["n"] += 1
                if got["n"] % 60 == 0:
                    got["frames"].append(fr.to_ndarray(format="bgr24"))
        asyncio.ensure_future(pull())

    @pc.on("connectionstatechange")
    def on_cs():
        print("connection:", pc.connectionState, flush=True)

    t_start = time.monotonic()
    await pc.setLocalDescription(await pc.createOffer())
    req = urllib.request.Request(f"{base}/api/offer", data=json.dumps({
        "sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "role": a.role, "force": True,
    }).encode(), headers={"content-type": "application/json", **UA})
    ans = json.load(urllib.request.urlopen(req))
    await pc.setRemoteDescription(RTCSessionDescription(sdp=ans["sdp"], type=ans["type"]))
    print("answer ok, id", ans["id"][:8], flush=True)

    for i in range(a.seconds):
        await asyncio.sleep(1)
        st = json.load(urllib.request.urlopen(urllib.request.Request(f"{base}/api/state", headers=UA)))
        el = (time.monotonic() - got["t0"]) if got["t0"] else 0
        print(f"t={i+1:2d}s recv={got['n']:4d} ({got['n']/el if el else 0:.1f} fps) | server: ready={st['ready']} "
              f"in={st['input_fps']} gen={st['base_fps']} fps proc={st['proc_time_s']*1000:.0f}ms gpu={st['gpu_reserved_mb']}MB "
              f"prompt#{st['prompt_index']}", flush=True)

    for k, fr in enumerate(got["frames"][-2:]):
        cv2.imwrite(f"{a.out}/out_{k}.png", fr)
    if cam.last is not None:
        cv2.imwrite(f"{a.out}/in_last.png", cam.last)
    print(f"saved {min(2, len(got['frames']))} output samples to {a.out}", flush=True)
    await pc.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--role", default="stage")
    ap.add_argument("--device", default="0", help="camera index, -1 for synthetic, or a video file path (looped)")
    ap.add_argument("--seconds", type=int, default=20)
    ap.add_argument("--out", default=".")
    asyncio.run(main(ap.parse_args()))
