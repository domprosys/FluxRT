# ============================================================================
# FluxRT kiosk bootstrap — rebuild the runtime from scratch after a teardown
# (or on a fresh Windows box). Re-downloads the trimmed model weights and
# rebuilds the Python venv, then recreates the desktop shortcut.
#
# Prerequisites: a recent NVIDIA driver (CUDA 12.8+) and winget. git optional.
# Run via setup_kiosk.bat (double-click) or:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_kiosk.ps1
# ============================================================================
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
Write-Host "FluxRT kiosk setup in: $repo`n"

# --- 1. uv (creates/manages the venv) ---------------------------------------
function Find-Uv {
    foreach ($p in @("uv", "$env:LOCALAPPDATA\Microsoft\WinGet\Links\uv.exe")) {
        $c = Get-Command $p -ErrorAction SilentlyContinue
        if ($c) { return $c.Source }
    }
    return $null
}
$uv = Find-Uv
if (-not $uv) {
    Write-Host "Installing uv (winget)..."
    winget install -e --id astral-sh.uv --accept-source-agreements --accept-package-agreements --disable-interactivity
    $uv = Find-Uv
    if (-not $uv) { throw "uv installed but not yet on PATH. Open a NEW terminal and re-run this script." }
}
Write-Host "uv: $uv"

# --- 2. VS 2022 Build Tools (C++) — needed for torch.compile (compile_models) -
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$haveVC = $false
if (Test-Path $vswhere) {
    if (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath) { $haveVC = $true }
}
if (-not $haveVC) {
    Write-Host "Installing VS 2022 Build Tools (Desktop development with C++)..."
    winget install -e --id Microsoft.VisualStudio.2022.BuildTools --accept-source-agreements --accept-package-agreements --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended" --disable-interactivity
} else {
    Write-Host "VS Build Tools: present"
}

# --- 3. Python venv + dependencies ------------------------------------------
Write-Host "`nCreating venv + installing dependencies (a few minutes)..."
& $uv venv --python 3.12
& $uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
& $uv pip install -r requirements.txt
& $uv pip install -e .
$hf = Join-Path $repo ".venv\Scripts\hf.exe"
if (-not (Test-Path $hf)) { throw "Dependency install failed (hf not found in .venv)." }

# --- 4. Trimmed model weights (matches the original install, ~12 GB) ---------
# Only what the int8 + CPU-offload path needs: base text-encoder/vae/scheduler/
# tokenizer (NO base transformer), int8 transformer, RIFE, TAEF2.
# NOTE: hf's multi-pattern --include reliably drops its FIRST pattern, so each
# base/int8 download lists the small sacrificial file first, then re-fetches it.
Write-Host "`nDownloading model weights (~12 GB)..."
& $hf download black-forest-labs/FLUX.2-klein-4B --local-dir FLUX.2-klein-4B --include "scheduler/*" "text_encoder/*" "tokenizer/*" "vae/*" "model_index.json"
& $hf download black-forest-labs/FLUX.2-klein-4B scheduler/scheduler_config.json --local-dir FLUX.2-klein-4B
& $hf download aydin99/FLUX.2-klein-4B-int8 --local-dir FLUX.2-klein-4B-int8 --include "config.json" "diffusion_pytorch_model.safetensors" "quanto_qmap.json" "quantized_flux2.py"
& $hf download aydin99/FLUX.2-klein-4B-int8 config.json --local-dir FLUX.2-klein-4B-int8
& $hf download TensorForger/RIFE-safetensors --local-dir RIFE-safetensors
& $hf download madebyollin/taef2 taef2.safetensors --local-dir taef2

# --- 5. Verify the critical files are present --------------------------------
$required = @(
    "FLUX.2-klein-4B\text_encoder\model-00001-of-00002.safetensors",
    "FLUX.2-klein-4B\text_encoder\model-00002-of-00002.safetensors",
    "FLUX.2-klein-4B\scheduler\scheduler_config.json",
    "FLUX.2-klein-4B\vae\diffusion_pytorch_model.safetensors",
    "FLUX.2-klein-4B\tokenizer\tokenizer.json",
    "FLUX.2-klein-4B-int8\diffusion_pytorch_model.safetensors",
    "FLUX.2-klein-4B-int8\config.json",
    "FLUX.2-klein-4B-int8\quanto_qmap.json",
    "RIFE-safetensors\flownet.safetensors",
    "taef2\taef2.safetensors"
)
$missing = $required | Where-Object { -not (Test-Path $_) }
if ($missing) {
    Write-Host "`nERROR: missing after download:"; $missing | ForEach-Object { Write-Host "  $_" }
    throw "Weight download incomplete."
}
Write-Host "All required weights present."

# --- 6. (Re)create the desktop shortcut -------------------------------------
$lnk = Join-Path ([Environment]::GetFolderPath('Desktop')) "FluxRT.lnk"
$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut($lnk)
$sc.TargetPath = Join-Path $repo "scripts\launch_production.bat"
$sc.WorkingDirectory = $repo
$sc.IconLocation = (Join-Path $repo ".venv\Scripts\python.exe") + ",0"
$sc.Description = "Start FluxRT real-time video (kiosk)"
$sc.Save()
Write-Host "Desktop shortcut created."

Write-Host "`n=== Setup complete. Double-click the FluxRT desktop shortcut to launch. ==="
Write-Host "First launch warms up (compile + prompt encode), ~1-2 min; then it's cached/fast."
