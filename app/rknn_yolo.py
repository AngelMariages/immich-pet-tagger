"""YOLO detection on a Rockchip NPU.

Usage:
    docker compose exec immich-pet-tagger python /app/rknn_yolo.py export

Unlike CLIP, nothing here runs the model: Ultralytics has its own RKNN backend,
so once a .rknn exists next to the metadata Ultralytics writes at export time,
`YOLO(path)` loads it and its box decoding and NMS carry on unchanged. All this
module does is produce that pair of files and tell the detector where they are.

The build itself is the same converter CLIP uses (tools/rknn), driven by a
sidecar written here. Ultralytics normalizes with mean 0 and std 255, i.e. it
just scales pixels into [0,1], which in the sidecar's [0,1] image units is a
mean of 0 and a std of 1.
"""

import json
import logging
import sys
from pathlib import Path

import npu

log = logging.getLogger("rknn_yolo")

MODEL_DIR = npu.MODEL_DIR


def model_dir(model_name: str) -> Path:
    """One directory per YOLO model, holding its ONNX, its .rknn builds and the
    metadata.yaml Ultralytics reads from alongside the model it loads."""
    return MODEL_DIR / f"{Path(model_name).stem}_rknn_model"


def _sidecar(model_name: str) -> Path:
    return model_dir(model_name) / f"{Path(model_name).stem}.json"


def model_path(model_name: str, input_size: int, soc: str | None = None) -> Path:
    """The built model for this host, or a RuntimeError explaining how to build it."""
    soc = soc or npu.soc()
    if soc is None:
        raise RuntimeError(
            "Could not determine the Rockchip SoC (the device tree is masked in most "
            "containers). Set RKNN_SOC, e.g. RKNN_SOC=rk3588."
        )
    path = model_dir(model_name) / f"{Path(model_name).stem}_{soc}.rknn"
    if not path.exists():
        raise RuntimeError(
            f"{path} not found. Export the ONNX with `python /app/rknn_yolo.py export`, "
            "then build it for this SoC with the converter in tools/rknn."
        )

    sidecar = _sidecar(model_name)
    if sidecar.exists():
        built_for = json.loads(sidecar.read_text()).get("input_size")
        if built_for and built_for != input_size:
            raise RuntimeError(
                f"{path.name} was built for {built_for}px but YOLO_INPUT_SIZE is {input_size}. "
                "An .rknn has its input shape fixed at build time: re-export and rebuild, "
                f"or set YOLO_INPUT_SIZE={built_for}."
            )
    return path


def export(model_name: str, input_size: int) -> Path:
    """Export the configured YOLO model to ONNX, next to the metadata Ultralytics
    needs when it loads the built model back.

    Batch 1 and opset 19 are what rknn-toolkit2 accepts; simplify is off so the
    export needs nothing beyond the onnx package. Ultralytics embeds its metadata
    in the ONNX itself, so it is copied out to the metadata.yaml its RKNN backend
    looks for beside the model."""
    from ultralytics import YOLO
    from ultralytics.nn.backends.base import BaseBackend
    from ultralytics.utils import YAML

    target = model_dir(model_name)
    target.mkdir(parents=True, exist_ok=True)

    log.info(f"Exporting {model_name} at {input_size}px...")
    exported = Path(YOLO(model_name).export(
        format="onnx", imgsz=input_size, batch=1, opset=19, simplify=False,
    ))
    onnx_path = target / exported.name
    exported.replace(onnx_path)

    YAML.save(target / "metadata.yaml", BaseBackend.read_metadata(onnx_path))
    _sidecar(model_name).write_text(json.dumps({
        "model": model_name,
        "input_size": input_size,
        "input_name": "images",
        # Ultralytics feeds the NPU raw pixels and expects the model to scale them
        # into [0,1]: mean 0, std 1 in the sidecar's image units.
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
    }, indent=2) + "\n")

    print(f"Wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.0f} MB) and metadata.yaml")
    print("\nNext, build it for this board (from the repo root):")
    print("  docker build -t rknn-convert tools/rknn")
    print(f"  docker run --rm -v ./data:/data rknn-convert {onnx_path} rk3588")
    return onnx_path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    from detector import YOLO_INPUT_SIZE, YOLO_MODEL_NAME

    if sys.argv[1:2] == ["export"]:
        export(YOLO_MODEL_NAME, YOLO_INPUT_SIZE)
        return 0
    print(__doc__.split("\n\n")[1])
    return 1


if __name__ == "__main__":
    sys.exit(main())
