"""Build a .rknn model for one Rockchip SoC from an exported ONNX file.

Usage:
    python convert.py <model.onnx> <soc> [--quantize]

The sidecar written next to the ONNX by `rknn_clip.py export` supplies the input
size and the mean/std to compile into the model, so nothing about a particular
CLIP model is hardcoded here. Folding normalization into the build is what lets
the runtime hand the NPU raw uint8 pixels instead of paying for the conversion
on the CPU for every crop.

fp16 is the default. Quantizing to int8 needs a calibration set and costs
embedding accuracy, which is the one thing that has to survive here for a
classifier trained on CPU embeddings to keep working.
"""

import json
import sys
from pathlib import Path

from rknn.api import RKNN

SUPPORTED_SOCS = ("rk3562", "rk3566", "rk3568", "rk3576", "rk3588")


def convert(onnx_path: Path, soc: str, quantize: bool = False) -> Path:
    meta_path = onnx_path.with_suffix(".json")
    if not meta_path.exists():
        sys.exit(f"{meta_path} not found; it is written alongside the ONNX by `rknn_clip.py export`.")
    meta = json.loads(meta_path.read_text())

    size = int(meta["input_size"])
    # rknn.config takes mean/std on the 0-255 scale it will see at runtime, while
    # CLIP's are for [0,1] floats, hence the x255.
    mean = [[v * 255 for v in meta["mean"]]]
    std = [[v * 255 for v in meta["std"]]]
    out_path = onnx_path.with_name(f"{onnx_path.stem}_{soc}.rknn")

    rknn = RKNN(verbose=False)
    rknn.config(target_platform=soc, mean_values=mean, std_values=std)

    if rknn.load_onnx(model=str(onnx_path), inputs=["pixel_values"], input_size_list=[[1, 3, size, size]]) != 0:
        sys.exit("Loading the ONNX model failed.")
    if rknn.build(do_quantization=quantize) != 0:
        sys.exit("Building the RKNN model failed.")
    if rknn.export_rknn(str(out_path)) != 0:
        sys.exit("Exporting the RKNN model failed.")
    rknn.release()

    print(f"\nWrote {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)")
    return out_path


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        sys.exit(__doc__)
    onnx_path, soc = Path(args[0]), args[1].lower()
    if not onnx_path.exists():
        sys.exit(f"{onnx_path} not found.")
    if soc not in SUPPORTED_SOCS:
        sys.exit(f"{soc} is not one of {', '.join(SUPPORTED_SOCS)}.")
    convert(onnx_path, soc, quantize="--quantize" in sys.argv)


if __name__ == "__main__":
    main()
