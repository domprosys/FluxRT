"""LivePortrait face reenactment on the generated frame.

The generated (stylized) frame is the LivePortrait *source*; the latest webcam frame is the
*driving* signal. Per generated frame we detect the face in both, extract the implicit 3D
keypoints (motion extractor, batch of 2), move the chosen expression keypoints of the source
towards the driving face, re-render the source face crop (warping module + SPADE decoder) and
paste it back with LivePortrait's feathered mask. The head pose of the generated frame is kept
(FluxRT already follows the webcam pose) unless region="all" with relative=False.

Options (the "lip_transfer" config block; see LivePortraitPostProcessor.from_config):
  models_dir    dir with base_models/, retargeting_models/ (the HF "liveportrait/" folder);
                insightface/models/buffalo_l/ must sit next to it. Relative paths are tried
                against the CWD, then the repo root.
  region        "lip" (default) | "eyes" | "lip_eyes" | "exp" (lips+eyes+brows, all expression
                keypoints) | "all" (exp + head pose; only differs from "exp" when relative=False)
  relative      False (default): the driving expression replaces the source one on the chosen
                keypoints (LivePortrait's video-to-video formula). Right for FluxRT, whose output
                already carries the webcam expression: it re-syncs the mouth to the latest frame.
                True: the original FluxRT formula (LivePortrait's image-driven one), source
                expression + (driving expression - neutral template). It adds the driving motion
                on top of what the generated face already shows, so an open mouth in the generated
                frame gets opened twice (verified on stills).
  source_crop   "detect" (default): detect the face in the generated frame too, skip the frame
                when there is none (stylized beyond recognition). "driving": reuse the webcam
                face crop for the generated frame (one detection per frame, always applies).
  det_thresh    face detector threshold (default 0.1 = LivePortrait's cropper default)
  multiplier    motion multiplier (default 1.0)
  device        torch device for the networks, "cuda" (default) or "cpu"
  detector_device  "cuda" | "cpu" for the ONNX face detector/landmarks (default: = device)
  half          fp16 autocast for the networks on CUDA (default True)
  compile       torch.compile the warping module + decoder (default False)
  compile_mode  torch.compile mode when compile=True (default "default"; "max-autotune" is
                what LivePortrait uses, with CUDA graphs and minutes of autotuning)
  warmup        run every network once at load time (default True)
  log_interval_s  print per-frame timing/face stats this often (default 10; 0 = off)
"""
import os.path as osp
import time
import traceback
from collections import Counter
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from fluxrt import LIVEPORTRAIT_AVAILABLE
from fluxrt.stream_processor.postprocessors.base import BasePostProcessor

# Implicit-keypoint indices, from LivePortrait's live_portrait_pipeline.py (regional control).
_LIP_INDICES = [6, 12, 14, 17, 19, 20]
_EYE_INDICES = [11, 13, 15, 16, 18]
_EXP_INDICES = [1, 2, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
REGIONS = ("lip", "eyes", "lip_eyes", "exp", "all")
SOURCE_CROPS = ("detect", "driving")

_REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", "..", ".."))
_REQUIRED = [
    "base_models/appearance_feature_extractor.pth",
    "base_models/motion_extractor.pth",
    "base_models/spade_generator.pth",
    "base_models/warping_module.pth",
    "retargeting_models/stitching_retargeting_module.pth",
    "../insightface/models/buffalo_l/det_10g.onnx",
    "../insightface/models/buffalo_l/2d106det.onnx",
]
# config keys consumed by the caller, not by this class
_ORCHESTRATION_KEYS = {"enable", "start_active"}

_LP = None


def _import_liveportrait() -> SimpleNamespace:
    """Import the LivePortrait code lazily, so a missing/broken add-on install can never break
    `import fluxrt` for configs that don't use lip transfer."""
    global _LP
    if _LP is None:
        if not LIVEPORTRAIT_AVAILABLE:
            raise RuntimeError(
                "LivePortrait not installed: expected LivePortrait-code/ in the repo root "
                "(run deploy/setup_liveportrait.sh)"
            )
        # torch is imported first (module top) on purpose: onnxruntime-gpu's CUDA EP then
        # reuses the CUDA 12 / cuDNN 9 libraries torch already loaded from the nvidia wheels.
        import onnxruntime  # noqa: F401  (fail early with a clear ImportError)
        from liveportrait.config.inference_config import InferenceConfig
        from liveportrait.live_portrait_wrapper import LivePortraitWrapper
        from liveportrait.utils.camera import get_rotation_matrix
        from liveportrait.utils.crop import crop_image, prepare_paste_back, paste_back
        from liveportrait.utils.face_analysis_diy import FaceAnalysisDIY
        from liveportrait.utils.dependencies.insightface.app.common import Face
        import liveportrait.live_portrait_wrapper as _lpw

        _lpw.log = lambda *a, **kw: None  # per-checkpoint rich log lines; our "ready" line says it all

        _LP = SimpleNamespace(
            InferenceConfig=InferenceConfig,
            LivePortraitWrapper=LivePortraitWrapper,
            get_rotation_matrix=get_rotation_matrix,
            crop_image=crop_image,
            prepare_paste_back=prepare_paste_back,
            paste_back=paste_back,
            FaceAnalysisDIY=FaceAnalysisDIY,
            Face=Face,
            onnxruntime=onnxruntime,
        )
    return _LP


def resolve_models_dir(path: str) -> str:
    p = osp.expanduser(osp.expandvars(path))
    if osp.isabs(p):
        return p
    if osp.isdir(p):
        return osp.abspath(p)
    alt = osp.join(_REPO_ROOT, p)
    return alt if osp.isdir(alt) else osp.abspath(p)


class LivePortraitPostProcessor(BasePostProcessor):
    def __init__(
        self,
        models_dir: str,
        region: str = "lip",
        relative: bool = False,
        source_crop: str = "detect",
        det_thresh: float = 0.1,
        multiplier: float = 1.0,
        device: str = "cuda",
        detector_device: str | None = None,
        half: bool = True,
        compile: bool = False,
        compile_mode: str = "default",
        warmup: bool = True,
        log_interval_s: float = 10.0,
        **unknown,
    ):
        unknown = {k: v for k, v in unknown.items() if k not in _ORCHESTRATION_KEYS and not k.startswith("_")}
        if unknown:
            print(f"[LivePortrait] ignoring unknown lip_transfer options: {sorted(unknown)}", flush=True)
        if region not in REGIONS:
            raise ValueError(f"lip_transfer.region must be one of {REGIONS}, got {region!r}")
        if source_crop not in SOURCE_CROPS:
            raise ValueError(f"lip_transfer.source_crop must be one of {SOURCE_CROPS}, got {source_crop!r}")
        lp = _import_liveportrait()
        self._lp = lp

        models_dir = resolve_models_dir(models_dir)
        missing = [f for f in _REQUIRED if not osp.isfile(osp.normpath(osp.join(models_dir, f)))]
        if missing:
            raise FileNotFoundError(
                f"LivePortrait weights missing under {models_dir}: {missing} "
                "(run deploy/setup_liveportrait.sh)"
            )
        insightface_dir = osp.join(osp.dirname(models_dir), "insightface")

        self.region = region
        self.relative = bool(relative)
        self.source_crop = source_crop
        self.multiplier = float(multiplier)
        self.log_interval_s = float(log_interval_s or 0)

        use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
        if str(device).startswith("cuda") and not use_cuda:
            print("[LivePortrait] CUDA not available, running on CPU (slow)", flush=True)
        dev_id = (torch.device(device).index or 0) if use_cuda else 0
        det_dev = detector_device or ("cuda" if use_cuda else "cpu")
        det_cuda = str(det_dev).startswith("cuda") and use_cuda

        inf_cfg = lp.InferenceConfig(
            checkpoint_F=osp.join(models_dir, "base_models", "appearance_feature_extractor.pth"),
            checkpoint_M=osp.join(models_dir, "base_models", "motion_extractor.pth"),
            checkpoint_G=osp.join(models_dir, "base_models", "spade_generator.pth"),
            checkpoint_W=osp.join(models_dir, "base_models", "warping_module.pth"),
            checkpoint_S=osp.join(models_dir, "retargeting_models", "stitching_retargeting_module.pth"),
            flag_use_half_precision=bool(half) and use_cuda,
            flag_force_cpu=not use_cuda,
            device_id=dev_id,
            flag_pasteback=True,
            flag_do_crop=True,
            flag_stitching=True,
            flag_relative_motion=self.relative,
            flag_do_torch_compile=False,  # compiled below without the wrapper's global dynamo side effects
            driving_multiplier=self.multiplier,
        )
        t0 = time.perf_counter()
        self.wrapper = lp.LivePortraitWrapper(inf_cfg)
        self.inf_cfg = inf_cfg
        self.device = self.wrapper.device
        if compile:
            self.wrapper.warping_module = torch.compile(self.wrapper.warping_module, mode=compile_mode)
            self.wrapper.spade_generator = torch.compile(self.wrapper.spade_generator, mode=compile_mode)
            # warp_decode() calls cudagraph_mark_step_begin() when this is set (needed by cudagraph modes)
            self.wrapper.compile = compile_mode in ("reduce-overhead", "max-autotune")

        # Face detector (SCRFD det_10g) + 106-point landmarks: LivePortrait's cropper minus its
        # 203-point landmark.onnx refinement, whose output only feeds the retargeting ratios.
        if det_cuda:
            providers = [("CUDAExecutionProvider", {"device_id": dev_id}), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.face_analysis = lp.FaceAnalysisDIY(
            name="buffalo_l",
            root=insightface_dir,
            allowed_modules=["detection", "landmark_2d_106"],
            providers=providers,
        )
        self.face_analysis.prepare(ctx_id=dev_id if det_cuda else -1, det_size=(512, 512), det_thresh=float(det_thresh))
        self.det_model = self.face_analysis.det_model
        self.lmk_model = self.face_analysis.models["landmark_2d_106"]
        self.detector_providers = self.det_model.session.get_providers()
        if det_cuda and "CUDAExecutionProvider" not in self.detector_providers:
            print(
                "[LivePortrait] WARNING: onnxruntime could not use CUDA for the face detector "
                f"(providers {self.detector_providers}); it runs on the CPU. Check that "
                "onnxruntime-gpu==1.26.0 (CUDA 12 build) is installed and torch is cu12x.",
                flush=True,
            )

        self.lip_array = torch.from_numpy(inf_cfg.lip_array).to(dtype=torch.float32, device=self.device)
        self.mask_crop = inf_cfg.mask_crop
        self.crop_kw = dict(dsize=512, scale=2.3, vy_ratio=-0.125, flag_do_rot=True)  # CropConfig defaults

        if warmup:
            self._warmup()
        self.load_s = time.perf_counter() - t0
        self._stats_reset()
        self._stats_t0 = time.perf_counter()
        self._errors = 0
        self.last_status = None
        print(
            f"[LivePortrait] ready in {self.load_s:.1f}s: region={self.region} relative={self.relative} "
            f"source_crop={self.source_crop} device={self.device} half={inf_cfg.flag_use_half_precision} "
            f"compile={bool(compile)} detector={self.detector_providers[0]} models={models_dir}",
            flush=True,
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "LivePortraitPostProcessor":
        """Build from the config's "lip_transfer" block (ignores enable/start_active and "_" comment keys)."""
        return cls(**{k: v for k, v in cfg.items() if k not in _ORCHESTRATION_KEYS and not k.startswith("_")})

    # ── internals ────────────────────────────────────────────────────────────
    def _warmup(self) -> None:
        """First calls pay for cuDNN autotuning / ORT arena setup / (Blackwell) PTX JIT of the
        onnxruntime kernels; do them at load time instead of on the first live face."""
        lp = self._lp
        img = np.zeros((368, 640, 3), np.uint8)
        self.det_model.detect(img, max_num=0, metric="default")
        face = lp.Face(bbox=np.array([220, 84, 420, 284], np.float32), kps=None, det_score=1.0)
        self.lmk_model.get(img, face)
        with torch.no_grad():
            for bs in (1, 2):
                x = torch.zeros(bs, 3, 256, 256, device=self.device)
                kp = self.wrapper.get_kp_info(x)
            x_s_info = {k: v[:1] for k, v in kp.items()}
            f_s = self.wrapper.extract_feature_3d(x[:1])
            x_s = self.wrapper.transform_keypoint(x_s_info)
            x_d = self.wrapper.stitching(x_s, x_s.clone())
            out = self.wrapper.warp_decode(f_s, x_s, x_d)
            self.wrapper.parse_output(out["out"])

    def _detect(self, img_rgb: np.ndarray, near=None):
        """Return the 106-point landmarks of one face, or None. Picks the largest face, or the
        face closest to `near` = (cx, cy, max_dist) when given."""
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        bboxes, kpss = self.det_model.detect(img_bgr, max_num=0, metric="default")
        if bboxes.shape[0] == 0:
            return None, None
        if near is None:
            area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            i = int(np.argmax(area))
        else:
            cx = (bboxes[:, 0] + bboxes[:, 2]) / 2 - near[0]
            cy = (bboxes[:, 1] + bboxes[:, 3]) / 2 - near[1]
            dist = np.hypot(cx, cy)
            i = int(np.argmin(dist))
            if dist[i] > near[2]:
                return None, None
        face = self._lp.Face(bbox=bboxes[i, 0:4], kps=None if kpss is None else kpss[i], det_score=bboxes[i, 4])
        self.lmk_model.get(img_bgr, face)
        return face.landmark_2d_106, bboxes[i, 0:4]

    def _crop(self, img_rgb: np.ndarray, lmk: np.ndarray) -> dict:
        crop = self._lp.crop_image(img_rgb, lmk, **self.crop_kw)
        crop["img_crop_256x256"] = cv2.resize(crop["img_crop"], (256, 256), interpolation=cv2.INTER_AREA)
        return crop

    def _compose(self, x_s_info: dict, x_d_info: dict, R_s: torch.Tensor):
        """New expression / rotation / translation for the source, per region and mode
        (mirrors live_portrait_pipeline.py: image-driven relative, source-video absolute)."""
        exp_s, exp_d = x_s_info["exp"], x_d_info["exp"]
        delta = exp_s.clone()
        R_new, t_new = R_s, x_s_info["t"].clone()
        lip = self.region in ("lip", "lip_eyes")
        eyes = self.region in ("eyes", "lip_eyes")
        if self.relative:
            if self.region in ("exp", "all"):
                delta = exp_s + (exp_d - self.lip_array)
            if lip:
                delta[:, _LIP_INDICES, :] = (exp_s + (exp_d - self.lip_array))[:, _LIP_INDICES, :]
            if eyes:
                delta[:, _EYE_INDICES, :] = (exp_s + exp_d)[:, _EYE_INDICES, :]
            # image-driven relative motion has no reference driving frame, so pose stays the source's
        else:
            if self.region in ("exp", "all"):
                delta[:, _EXP_INDICES, :] = exp_d[:, _EXP_INDICES, :]
                delta[:, 3:5, 1] = exp_d[:, 3:5, 1]
                delta[:, 5, 2] = exp_d[:, 5, 2]
                delta[:, 8, 2] = exp_d[:, 8, 2]
                delta[:, 9, 1:] = exp_d[:, 9, 1:]
            if lip:
                delta[:, _LIP_INDICES, :] = exp_d[:, _LIP_INDICES, :]
            if eyes:
                delta[:, _EYE_INDICES, :] = exp_d[:, _EYE_INDICES, :]
            if self.region == "all":
                R_new = self._lp.get_rotation_matrix(x_d_info["pitch"], x_d_info["yaw"], x_d_info["roll"])
                t_new = x_d_info["t"].clone()
        t_new[..., 2] = 0  # zero tz
        return delta, R_new, t_new

    def _process(self, source_rgb: np.ndarray, driving_rgb: np.ndarray, tm: dict):
        lp = self._lp
        t = time.perf_counter()
        lmk_d, box_d = self._detect(driving_rgb)
        if lmk_d is None:
            tm["detect"] = time.perf_counter() - t
            return source_rgb, "no_driving_face"
        sx = source_rgb.shape[1] / driving_rgb.shape[1]
        sy = source_rgb.shape[0] / driving_rgb.shape[0]
        if self.source_crop == "driving":
            lmk_s = lmk_d * np.array([sx, sy], dtype=lmk_d.dtype)
        else:
            w = (box_d[2] - box_d[0]) * sx
            near = ((box_d[0] + box_d[2]) / 2 * sx, (box_d[1] + box_d[3]) / 2 * sy, max(w, 32.0))
            lmk_s, _ = self._detect(source_rgb, near=near)
            if lmk_s is None:
                tm["detect"] = time.perf_counter() - t
                return source_rgb, "no_source_face"
        crop_s = self._crop(source_rgb, lmk_s)
        crop_d = self._crop(driving_rgb, lmk_d)
        tm["detect"] = time.perf_counter() - t

        t = time.perf_counter()
        with torch.no_grad():
            both = np.stack([crop_s["img_crop_256x256"], crop_d["img_crop_256x256"]])
            x = torch.from_numpy(both).to(self.device).permute(0, 3, 1, 2).float().div_(255.0)
            kp = self.wrapper.get_kp_info(x)
            x_s_info = {k: v[0:1] for k, v in kp.items()}
            x_d_info = {k: v[1:2] for k, v in kp.items()}
            R_s = lp.get_rotation_matrix(x_s_info["pitch"], x_s_info["yaw"], x_s_info["roll"])
            f_s = self.wrapper.extract_feature_3d(x[0:1])
            x_s = self.wrapper.transform_keypoint(x_s_info)

            delta, R_new, t_new = self._compose(x_s_info, x_d_info, R_s)
            x_d_new = x_s_info["scale"] * (x_s_info["kp"] @ R_new + delta) + t_new
            x_d_new = self.wrapper.stitching(x_s, x_d_new)
            x_d_new = x_s + (x_d_new - x_s) * self.multiplier
            out = self.wrapper.warp_decode(f_s, x_s, x_d_new)
            I_p = self.wrapper.parse_output(out["out"])[0]  # 512x512 RGB uint8 (syncs the GPU)
        tm["net"] = time.perf_counter() - t

        t = time.perf_counter()
        dsize = (source_rgb.shape[1], source_rgb.shape[0])
        mask_ori = lp.prepare_paste_back(self.mask_crop, crop_s["M_c2o"], dsize=dsize)
        result = lp.paste_back(I_p, crop_s["M_c2o"], source_rgb, mask_ori)
        tm["paste"] = time.perf_counter() - t
        return result, "applied"

    def _stats_reset(self) -> None:
        self._st = Counter()
        self._st_ms = []
        self._st_parts = Counter()

    def _maybe_log(self) -> None:
        if not self.log_interval_s:
            return
        now = time.perf_counter()
        if now - self._stats_t0 < self.log_interval_s or not self._st_ms:
            return
        ms = sorted(self._st_ms)
        n = len(ms)
        parts = " ".join(f"{k} {1000 * v / n:.1f}" for k, v in sorted(self._st_parts.items()))
        counts = " ".join(f"{k}={v}" for k, v in sorted(self._st.items()))
        print(
            f"[LivePortrait] {now - self._stats_t0:.0f}s: {n} frames ({counts}) | "
            f"mean {sum(ms) / n:.1f} ms p50 {ms[n // 2]:.1f} p95 {ms[min(n - 1, int(0.95 * n))]:.1f} | "
            f"per-frame avg ms: {parts}",
            flush=True,
        )
        self._stats_reset()
        self._stats_t0 = now

    # ── public ───────────────────────────────────────────────────────────────
    def process(self, source_rgb: np.ndarray, driving_rgb: np.ndarray) -> np.ndarray:
        """source_rgb: generated frame, driving_rgb: latest input frame; uint8 RGB HxWx3.
        Never raises: on any per-frame error the generated frame is returned unchanged, so a
        LivePortrait problem cannot freeze the stream."""
        t0 = time.perf_counter()
        tm = {}
        try:
            out, status = self._process(source_rgb, driving_rgb, tm)
        except Exception as exc:  # noqa: BLE001
            self._errors += 1
            if self._errors <= 3 or self._errors % 100 == 0:
                print(f"[LivePortrait] frame error #{self._errors}: {type(exc).__name__}: {exc}", flush=True)
                if self._errors == 1:
                    traceback.print_exc()
            out, status = source_rgb, "error"
        self.last_status = status
        self._st[status] += 1
        self._st_ms.append(1000 * (time.perf_counter() - t0))
        self._st_parts.update(tm)
        self._maybe_log()
        return out
