"""Rockchip NPU detection and inference backend selection.

Usage (self-report what this host offers):
    docker compose exec immich-pet-tagger python /app/npu.py

Rockchip boards pair weak CPU cores with an NPU that PyTorch cannot address at
all: models have to be converted ahead of time into Rockchip's own .rknn format
and run through the RKNPU runtime. That makes the NPU a third inference backend
next to CPU and CUDA rather than just another torch device, so the choice is
resolved once here instead of at every `torch.cuda.is_available()` call site.

Detection is passive. The SoC is read from the device tree, and the NPU only
counts as usable if the RKNPU runtime wheel is installed, which is only true in
the -rknn image variant. A plain arm64 CPU install on the same board therefore
keeps running exactly as before.
"""

import importlib.util
import logging
import os
from pathlib import Path

import torch

log = logging.getLogger("npu")

# Pin the backend explicitly ("cpu", "cuda" or "rknn") to override detection,
# e.g. to keep a scan on the CPU cores while the NPU is busy with something else.
BACKEND = os.environ.get("BACKEND", "auto").lower()

# SoCs whose NPU rknn-toolkit2 can build models for. The rk3588s is the cheaper
# single-package rk3588 and some board device trees report it under its own name,
# but the NPU is identical and the toolkit only accepts "rk3588" as the platform.
SUPPORTED_SOCS = {"rk3562", "rk3566", "rk3568", "rk3576", "rk3588"}
_SOC_ALIASES = {"rk3588s": "rk3588"}

_DEVICE_TREE = Path("/proc/device-tree/compatible")


def soc() -> str | None:
    """Name of this host's Rockchip SoC, or None if it isn't a supported one.

    The device tree's `compatible` property is a NUL-separated list of
    "vendor,model" strings, most specific first, e.g.
    "radxa,rock-5b\\0rockchip,rk3588\\0". Every entry is checked rather than just
    the last one, since board vendors are free to order and extend that list."""
    try:
        raw = _DEVICE_TREE.read_text()
    except OSError:
        return None  # not an ARM SBC, or /sys is not mounted in this container
    for entry in raw.strip("\x00").split("\x00"):
        name = entry.split(",")[-1].strip().lower()
        name = _SOC_ALIASES.get(name, name)
        if name in SUPPORTED_SOCS:
            return name
    return None


def _has_runtime() -> bool:
    """Whether the RKNPU runtime wheel is installed. Only the -rknn image has it,
    so this is what keeps the plain arm64 CPU image on the CPU."""
    return importlib.util.find_spec("rknnlite") is not None


def backend() -> str:
    """The inference backend to use: "cuda", "rknn" or "cpu"."""
    if BACKEND in ("cpu", "cuda", "rknn"):
        return BACKEND
    if torch.cuda.is_available():
        return "cuda"
    if soc() is not None and _has_runtime():
        return "rknn"
    return "cpu"


def describe() -> str:
    """The backend plus, when there is an NPU that isn't being used, why not.

    "Why is my NPU idle" is the one question this feature is guaranteed to
    generate, so the startup log answers it up front instead of leaving a bare
    "Device: cpu" on a board that clearly has an NPU."""
    chosen = backend()
    name = soc()
    if name is None:
        return chosen
    if chosen == "rknn":
        return f"rknn ({name})"
    if not _has_runtime():
        return f"{chosen} ({name} NPU found, but this image has no RKNPU runtime: use the -rknn variant)"
    return f"{chosen} ({name} NPU available, not selected by BACKEND={BACKEND})"


if __name__ == "__main__":
    try:
        raw = repr(_DEVICE_TREE.read_text())
    except OSError as e:
        raw = f"unavailable ({e})"
    print(f"Device tree: {raw}")
    print(f"SoC:         {soc() or 'not a supported Rockchip SoC'}")
    print(f"Runtime:     {'rknn-toolkit-lite2 installed' if _has_runtime() else 'not installed'}")
    print(f"Backend:     {describe()}")
