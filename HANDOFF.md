# FluxRT — low-VRAM webcam demo: Windows (4070 Ti, 12 GB) setup

This fork adds a **CPU text-encoder offload** so FLUX.2-klein-4B fits a **12 GB GPU**, plus a
**pre-encoded prompt-cycle demo** with transform filters and dreamy input smoothing.

Target box: **Windows + NVIDIA RTX 4070 Ti (12 GB)**, run **natively on Windows** (not WSL —
WSL2 webcam passthrough is unreliable; the GUI uses the Windows DSHOW/MSMF camera backends).

## What changed vs upstream `tensorforger/FluxRT`
- `src/fluxrt/stream_processor/model_inference_subprocess.py` — `text_encoder_device: "cpu"`
  offload (encoder loaded in **bf16 on CPU**, ~7.5 GB RAM; encodes only on prompt change and
  moves the small embeds to GPU), explicit module placement (don't `pipe.to()` the encoder —
  that was a 7.5/15 GB transient leak), `cuda_memory_fraction` cap knob, and prompt-cycle
  pre-encoding (`precompute_prompt_cycle`, `_apply_prompt_index`).
- `pipeline.py` — derive the run device from the transformer, not `_execution_device` (which
  resolves to CPU when the encoder is offloaded).
- `stream_processor.py` — `set_prompt_index()` for instant cached-prompt switching.
- `scripts/run_gui.py` — prompt cycling on a timer, output-only **fullscreen (F11)**, and
  **input temporal smoothing** (`input_smoothing_alpha`, EMA over recent frames → ghost-trails).
- `configs/` — `lowvram_config.json` (512×288), `lowvram_640_config.json` (640×368),
  `cycle_demo_config.json` (single-person transform filters), `cycle_demo_multi_config.json`
  (2+ people), plus 448 variants.
- `scripts/validate_lowvram.py`, `enc_bench.py` — measurement/benchmark helpers.

## Prerequisites (Windows)
1. **NVIDIA driver** with CUDA 12.8+ support (recent Game Ready / Studio driver).
2. **git** + **git-lfs** (`git lfs install`).
3. **uv** — `winget install astral-sh.uv` (or the install script).
4. *(For `compile_models: true`)* **Visual Studio 2022 Build Tools** → "Desktop development
   with C++". TorchInductor needs `cl.exe` on Windows. If you skip this, run a `*_nocompile`
   config (eager mode — slower first frames but no compiler needed).

## Setup
```powershell
git clone -b lowvram-cpu-offload https://github.com/domprosys/FluxRT.git
cd FluxRT

uv venv --python 3.12
.venv\Scripts\activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt    # installs triton-windows + SpoutGL via platform markers
uv pip install -e .

# verify GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
(Reference versions validated on Linux: Python 3.12, torch 2.11.0+cu128.)

## Download models (~62 GB, public, into the repo root)
```powershell
git lfs install
git clone https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
git clone https://huggingface.co/aydin99/FLUX.2-klein-4B-int8
git clone https://huggingface.co/TensorForger/RIFE-safetensors
```
The CPU-offload path uses the **bf16 base** text encoder (from `FLUX.2-klein-4B`), the **int8
transformer** (from `FLUX.2-klein-4B-int8`), and `FLUX.2-klein-4B`'s scheduler + VAE. RIFE =
frame interpolation.

## Run the demo
```powershell
python scripts/run_gui.py --int8 --config configs/cycle_demo_config.json
```
- Cycles transform filters (dragon, monster, superhero, knight, cyborg, golem), 15 s each,
  switches are instant (prompts pre-encoded at startup — first load ~1–2 min on bf16).
- **F11 / F** = output-only fullscreen, **Esc** = exit.
- Expected: **~6 GB VRAM** at 512×288 (fits 12 GB with headroom). Bump to
  `configs/lowvram_640_config.json` for 640×368 (~6.8 GB).

## Tunables (in the config JSON)
- `text_encoder_cpu_dtype`: `"bfloat16"` (default, 7.5 GB RAM) or `"float32"` (~2× faster
  encode but ~15 GB RAM — only if the box has ≥32 GB RAM).
- `input_smoothing_alpha`: `1.0` = off; lower (0.25–0.4) = dreamier ghost-trails on motion.
- `prompt_cycle_interval_s`: seconds per filter.
- `prompt_cycle`: the list of transform prompts (edit-instruction style — see notes below).
- `compile_models`: set `false` if you don't have MSVC Build Tools (eager fallback).

## First-bring-up tip
If `torch.compile` errors on Windows (missing `cl.exe`), test first with a no-compile config to
confirm the pipeline + webcam work, then install Build Tools and re-enable compile for speed:
```powershell
python scripts/run_gui.py --int8 --config configs/lowvram_448_nocompile.json
```

## Prompt-writing notes (learned this is image EDITING, not generation)
- Lead with an imperative targeting the subject: **"Transform/Turn the person into …"**, then the
  specific feature changes, then **"Keep the same pose, head position, and framing."**, then
  background + style. Anchor to the **in-frame head AND body** (face/skin + shoulders/chest).
- **Single vs multiple people:** singular ("the person") for one subject; plural ("everyone in
  the frame") duplicates a lone subject — use `cycle_demo_multi_config.json` only when 2+ people
  are actually present.
