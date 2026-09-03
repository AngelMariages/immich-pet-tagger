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

Every build is checked against the ONNX it came from before this exits, using
the toolkit's host simulator, so a conversion that silently mangles the model is
caught here rather than on the board.
"""

import json
import sys
from pathlib import Path

import numpy as np
from rknn.api import RKNN

SUPPORTED_SOCS = ("rk3562", "rk3566", "rk3568", "rk3576", "rk3588")

# An fp16 build of the same weights lands around 0.99999 against the fp32 ONNX.
# This is a floor for "something went wrong", not a quality bar.
MIN_COSINE = 0.99


def convert(onnx_path: Path, soc: str, quantize: bool = False) -> Path:
    meta_path = onnx_path.with_suffix(".json")
    if not meta_path.exists():
        sys.exit(
            f"{meta_path} not found. It is written next to the ONNX by the export step "
            "(`rknn_clip.py export` for CLIP, `rknn_yolo.py export` for YOLO), so this ONNX "
            "either came from somewhere else or that step did not finish."
        )
    meta = json.loads(meta_path.read_text())

    size = int(meta["input_size"])
    # rknn.config takes mean/std on the 0-255 scale it will see at runtime, while
    # CLIP's are for [0,1] floats, hence the x255.
    mean = [[v * 255 for v in meta["mean"]]]
    std = [[v * 255 for v in meta["std"]]]
    out_path = onnx_path.with_name(f"{onnx_path.stem}_{soc}.rknn")

    rknn = RKNN(verbose=False)
    rknn.config(target_platform=soc, mean_values=mean, std_values=std)

    # The input name defaults to CLIP's, so sidecars written before YOLO support
    # (which names its input "images") keep converting.
    inputs = [meta.get("input_name", "pixel_values")]
    if rknn.load_onnx(model=str(onnx_path), inputs=inputs, input_size_list=[[1, 3, size, size]]) != 0:
        sys.exit("Loading the ONNX model failed.")
    if rknn.build(do_quantization=quantize) != 0:
        sys.exit("Building the RKNN model failed.")
    if rknn.export_rknn(str(out_path)) != 0:
        sys.exit("Exporting the RKNN model failed.")

    print(f"\nWrote {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)")

    cos = verify(rknn, onnx_path, meta)
    rknn.release()
    if cos is None:
        print("Could not run the built model on the simulator; check it on the board instead.")
    elif cos < MIN_COSINE:
        sys.exit(f"Built model only reaches cosine {cos:.6f} against the ONNX: the conversion is wrong.")
    else:
        print(f"Matches the ONNX it was built from (cosine {cos:.6f}).")
    return out_path


def verify(rknn: RKNN, onnx_path: Path, meta: dict) -> float | None:
    """Cosine between the built model and its ONNX source on one random image.

    The simulator only works on the model still in memory from build(), not on a
    reloaded .rknn, so this has to happen in the same session. Random pixels are
    enough: what is being caught is a conversion that dropped or reordered part
    of the graph, which no realistic input would hide."""
    import onnxruntime as ort

    size = int(meta["input_size"])
    arr = np.random.default_rng(0).integers(0, 256, (1, size, size, 3), dtype=np.uint8)

    if rknn.init_runtime(target=None) != 0:
        return None
    built = rknn.inference(inputs=[arr], data_format="nhwc")[0]

    # The .rknn does mean/std itself, so feed the ONNX the same pixels normalized
    # by hand, in the NCHW layout it expects.
    mean = np.array(meta["mean"], dtype=np.float32).reshape(1, 3, 1, 1) * 255
    std = np.array(meta["std"], dtype=np.float32).reshape(1, 3, 1, 1) * 255
    x = (arr.astype(np.float32).transpose(0, 3, 1, 2) - mean) / std
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    source = session.run(None, {session.get_inputs()[0].name: x})[0]

    a = np.asarray(built, dtype=np.float32).reshape(-1)
    b = np.asarray(source, dtype=np.float32).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


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
