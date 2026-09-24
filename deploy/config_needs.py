#!/usr/bin/env python3
"""Print the install flags a config needs, as shell `export` lines (stdlib only).

    eval "$(python3 deploy/config_needs.py configs/multi_config.json)"

Handles single-engine configs and multi-engine configs ("backend": "multi", whose
"engines" point at other configs). Output variables are the knobs of runpod_setup.sh:
SKIP_SD, SKIP_SDV2, SKIP_FLUXRT_WEIGHTS, WITH_BF16, WITH_LIVEPORTRAIT, WITH_FACEID,
EXTRA_HF_MODELS (space-separated "org/repo" or "org/repo::glob1,glob2" specs).
"""
import json
import shlex
import sys
from pathlib import Path


def engine_configs(path: Path) -> list[dict]:
    cfg = json.loads(path.read_text())
    if cfg.get("backend") != "multi":
        return [cfg]
    out = []
    for spec in cfg["engines"].values():
        spec = {"config": spec} if isinstance(spec, str) else spec
        name = spec["config"] if spec["config"].endswith(".json") else spec["config"] + ".json"
        sub = json.loads((path.parent / name).read_text())
        sub.update(spec.get("overrides") or {})
        out.append(sub)
    return out


def needs(path: Path) -> dict:
    cfgs = engine_configs(path)
    kinds = [c.get("backend", "fluxrt") for c in cfgs]
    flux = [c for c in cfgs if c.get("backend", "fluxrt") == "fluxrt"]
    sd = [c for c in cfgs if c.get("backend") == "sd"]
    models: list[str] = []
    for c in cfgs:
        for m in c.get("hf_models") or []:
            if m not in models:
                models.append(m)
    return {
        "SKIP_SD": "0" if "sd" in kinds else "1",
        "SKIP_SDV2": "0" if "sdv2" in kinds else "1",
        "SKIP_FLUXRT_WEIGHTS": "0" if flux else "1",
        "WITH_BF16": "1" if any(not c.get("enable_int8_quantization", True) for c in flux) else "0",
        "WITH_LIVEPORTRAIT": "1" if any((c.get("lip_transfer") or {}).get("enable") for c in flux) else "0",
        "WITH_FACEID": "1" if any((c.get("worker") or {}).get("use_ipadapter") for c in sd) else "0",
        "WITH_TRT": "1" if any((c.get("worker") or {}).get("acceleration") == "tensorrt" for c in sd) else "0",
        "SDV2_14B": "1" if any((c.get("worker") or {}).get("model_size", "").lower() == "14b"
                               for c in cfgs if c.get("backend") == "sdv2") else "0",
        "EXTRA_HF_MODELS": " ".join(models),
    }


if __name__ == "__main__":
    for k, v in needs(Path(sys.argv[1])).items():
        print(f"export {k}={shlex.quote(v)}")
