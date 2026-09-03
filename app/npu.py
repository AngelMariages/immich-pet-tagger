"""Rockchip NPU detection and inference backend selection.

Usage (self-report what this host offers):
    docker compose exec immich-pet-tagger python /app/npu.py

Rockchip boards pair weak CPU cores with an NPU that PyTorch cannot address at
all: models have to be converted ahead of time into Rockchip's own .rknn format
and run through the RKNPU runtime. That makes the NPU a third inference backend
next to CPU and CUDA rather than just another torch device, so the choice is
resolved once here instead of at every `torch.cuda.is_available()` call site.

Detection is passive. The RKNPU runtime wheel is the opt-in signal, and it is
only installed in the -rknn image variant, so a plain arm64 CPU install on the
very same board keeps running exactly as before. Whether the NPU is actually
reachable is left to the runtime to report when a model is loaded, the same way
a failed YOLO or CLIP load is surfaced today.
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

# Where converted .rknn models live, on the /data volume so they survive an image
# upgrade: they are built per SoC and per configured model, not shipped in the image.
MODEL_DIR = Path(os.environ.get("DATA_DIR", "/data")) / "rknn"

_DEVICE_TREE = Path("/proc/device-tree/compatible")

# The SoC name is only published in the device tree, and a container usually
# cannot read it: /proc/device-tree is a symlink into /sys/firmware, which Docker
# masks by default, so even bind-mounting /sys leaves it empty unless the
# container runs with --security-opt systempaths=unconfined. RKNN_SOC covers the
# case where it is unreadable, but only for choosing which prebuilt model to load.
# It is not an alternative to unmasking: the RKNPU runtime reads the device tree
# itself when it starts, and refuses to run without it.
RKNN_SOC = os.environ.get("RKNN_SOC", "").strip().lower()


def soc() -> str | None:
    """Name of this host's Rockchip SoC, or None if it can't be determined.

    The device tree's `compatible` property is a NUL-separated list of
    "vendor,model" strings, most specific first, e.g.
    "radxa,rock-5b\\0rockchip,rk3588\\0". Every entry is checked rather than just
    the last one, since board vendors are free to order and extend that list."""
    if RKNN_SOC:
        return _SOC_ALIASES.get(RKNN_SOC, RKNN_SOC)
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
    # The wheel, not the SoC, is what decides. Pulling the -rknn image is already
    # a deliberate choice, and the SoC often isn't readable from inside a
    # container anyway (see RKNN_SOC), so gating on it would strand real NPUs.
    if _has_runtime():
        return "rknn"
    return "cpu"


def describe() -> str:
    """The backend plus, when there is an NPU that isn't being used, why not.

    "Why is my NPU idle" is the one question this feature is guaranteed to
    generate, so the startup log answers it up front instead of leaving a bare
    "Device: cpu" on a board that clearly has an NPU."""
    chosen = backend()
    name = soc()
    if chosen == "rknn":
        return f"rknn ({name or 'SoC unknown'})"
    if name is None:
        return chosen
    if not _has_runtime():
        return f"{chosen} ({name} NPU found, but this image has no RKNPU runtime: use the -rknn variant)"
    return f"{chosen} ({name} NPU available, not selected by BACKEND={BACKEND})"


if __name__ == "__main__":
    try:
        print(f"Device tree: {_DEVICE_TREE.read_text()!r}")
    except OSError as e:
        print(f"Device tree: unavailable ({e})")
        print("             /sys/firmware is masked unless the container runs with")
        print("             --security-opt systempaths=unconfined, which the RKNPU")
        print("             runtime needs too. RKNN_SOC only names the model to load.")
    print(f"SoC:         {soc() or 'not a supported Rockchip SoC'}")
    print(f"Runtime:     {'rknn-toolkit-lite2 installed' if _has_runtime() else 'not installed'}")
    print(f"Backend:     {describe()}")
