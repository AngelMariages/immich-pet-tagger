"""CLIP image embeddings on a Rockchip NPU.

Usage:
    docker compose exec immich-pet-tagger python /app/rknn_clip.py export
    docker compose exec immich-pet-tagger python /app/rknn_clip.py check photo.jpg [...]
    docker compose exec immich-pet-tagger python /app/rknn_clip.py check /some/folder
    docker compose exec immich-pet-tagger python /app/rknn_clip.py bench /some/folder

The NPU cannot run a PyTorch model, so the configured CLIP image encoder is
exported to ONNX once and then built into a .rknn for one specific SoC. The build
step deliberately lives elsewhere (tools/rknn): rknn-toolkit2 pins numpy<=1.26.4
and torch<=2.4.0, which this image cannot satisfy, so it gets its own container.

Two things are worth knowing when reading this:

- Mean/std normalization is baked into the .rknn at build time, so the NPU is fed
  raw uint8 pixels and does that arithmetic itself. preprocess() therefore stops
  one step short of what open_clip's own transform does.
- The model is built at batch 1. Batch size is fixed when the .rknn is built, and
  an NPU has no batching win to give anyway, so the batching queue that exists to
  keep a GPU busy is pure overhead here.
"""

import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

import npu

log = logging.getLogger("rknn_clip")

MODEL_DIR = npu.MODEL_DIR

# Below this, the NPU is not reproducing the PyTorch embeddings closely enough for
# a classifier trained on one to be trusted on the other. Chosen as a smoke-test
# floor, not a quality target: an fp16 build of the same weights should land far
# above it, and anything near it means the conversion did something wrong.
MIN_COSINE = 0.99

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def stem(model_name: str, pretrained: str) -> str:
    """Filename stem for a CLIP model, e.g. "clip_ViT_B_16_openai"."""
    return re.sub(r"[^A-Za-z0-9]+", "_", f"clip_{model_name}_{pretrained}").strip("_")


def preprocess(img: Image.Image, size: int) -> np.ndarray:
    """Resize the short side, center crop, and return 1xHxWx3 uint8 RGB.

    Mirrors torchvision's Resize+CenterCrop (the first half of open_clip's
    transform) and stops there: the float conversion and mean/std normalization
    that follow are compiled into the .rknn model, so handing the NPU anything
    other than raw uint8 pixels would apply them twice."""
    img = img.convert("RGB")
    w, h = img.size
    if w < h:
        new_w, new_h = size, int(size * h / w)
    else:
        new_w, new_h = int(size * w / h), size
    img = img.resize((new_w, new_h), Image.BICUBIC)
    left, top = (new_w - size) // 2, (new_h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    return np.asarray(img, dtype=np.uint8)[None]


def _meta_path(model_name: str, pretrained: str) -> Path:
    return MODEL_DIR / f"{stem(model_name, pretrained)}.json"


class Encoder:
    """CLIP image embeddings from a prebuilt .rknn model.

    RKNNLite keeps its own copy of the weights and one instance cannot be called
    from several threads at once, so each worker thread builds its own Encoder.
    Which of the NPU's cores runs the work is left to the driver (NPU_CORE_AUTO),
    which spreads concurrent Encoders across the RK3588's three cores without
    anything being pinned by hand."""

    def __init__(self, model_name: str, pretrained: str, soc: str | None = None):
        soc = soc or npu.soc()
        if soc is None:
            raise RuntimeError(
                "Could not determine the Rockchip SoC (the device tree is masked in most "
                "containers). Set RKNN_SOC, e.g. RKNN_SOC=rk3588."
            )
        path = MODEL_DIR / f"{stem(model_name, pretrained)}_{soc}.rknn"
        if not path.exists():
            raise RuntimeError(
                f"{path} not found. Export the ONNX with `python /app/rknn_clip.py export`, "
                "then build it for this SoC with the converter in tools/rknn."
            )

        meta_file = _meta_path(model_name, pretrained)
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        self.size = int(meta.get("input_size", 224))
        if not meta:
            log.warning(f"{meta_file} missing, assuming a {self.size}px input.")

        from rknnlite.api import RKNNLite

        self._rknn = RKNNLite()
        if self._rknn.load_rknn(str(path)) != 0:
            raise RuntimeError(f"Could not load {path}")
        if self._rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
            raise RuntimeError(
                "Could not start the RKNPU runtime. Check that the container has /dev/dri "
                "passed through and that librknnrt.so matches the host's rknpu driver."
            )
        log.info(f"CLIP running on the NPU from {path.name}")

    def preprocess(self, img: Image.Image) -> np.ndarray:
        """Kept separate from run() so callers can do it in their own thread, the
        way the PyTorch path already preprocesses outside the worker. Resizing a
        photo costs more CPU than the NPU spends on the embedding, so doing it in
        the worker would serialize the expensive half behind the cheap one."""
        return preprocess(img, self.size)

    def run(self, arr: np.ndarray) -> np.ndarray | None:
        """One L2-normalized embedding from an already preprocessed image.

        The .rknn graph stops at the image features, exactly like encode_image
        does; normalizing here keeps that step in numpy where it is free, rather
        than asking the converter to map a ReduceL2 onto the NPU."""
        out = self._rknn.inference(inputs=[arr], data_format="nhwc")
        if not out:
            return None
        vec = np.asarray(out[0], dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else None

    def encode(self, img: Image.Image) -> np.ndarray | None:
        """Both halves, for callers that have nowhere better to preprocess."""
        return self.run(self.preprocess(img))

    def close(self) -> None:
        self._rknn.release()


# ---------------------------------------------------------------------------
# CLI: export the ONNX, and check a built .rknn against the PyTorch model
# ---------------------------------------------------------------------------

def _configured() -> tuple[str, str]:
    """The CLIP model this install is set up for. Imported late so that embedder
    stays the single owner of those env vars without an import cycle."""
    from embedder import CLIP_MODEL_NAME, CLIP_PRETRAINED
    return CLIP_MODEL_NAME, CLIP_PRETRAINED


def export(model_name: str, pretrained: str, opset: int = 18) -> Path:
    """Export the CLIP image encoder to ONNX plus a sidecar describing the build.

    The sidecar carries what the converter needs (input size, mean/std) and what
    the runtime needs (input size again), so neither has to hardcode constants
    for a model the user is free to change."""
    import open_clip
    import torch

    log.info(f"Loading {model_name}/{pretrained}...")
    model, _, transform = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval()

    image_size = model.visual.image_size
    size = int(image_size[0] if isinstance(image_size, (tuple, list)) else image_size)
    norm = transform.transforms[-1]  # torchvision Normalize, the last step

    class _ImageEncoder(torch.nn.Module):
        """Only the image tower: the text tower and logit scale never run here."""

        def __init__(self, clip):
            super().__init__()
            self.clip = clip

        def forward(self, pixel_values):
            return self.clip.encode_image(pixel_values)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = MODEL_DIR / f"{stem(model_name, pretrained)}.onnx"
    dummy = torch.zeros(1, 3, size, size)

    log.info(f"Exporting to {onnx_path} (opset {opset})...")
    torch.onnx.export(
        _ImageEncoder(model),
        dummy,
        str(onnx_path),
        input_names=["pixel_values"],
        output_names=["image_features"],
        opset_version=opset,
        dynamo=False,
    )

    with torch.no_grad():
        embed_dim = int(model.encode_image(dummy).shape[-1])

    meta = {
        "model": model_name,
        "pretrained": pretrained,
        "input_size": size,
        "embed_dim": embed_dim,
        "opset": opset,
        "mean": [float(v) for v in norm.mean],
        "std": [float(v) for v in norm.std],
    }
    _meta_path(model_name, pretrained).write_text(json.dumps(meta, indent=2) + "\n")

    mb = onnx_path.stat().st_size / 1e6
    print(f"Wrote {onnx_path} ({mb:.0f} MB) and {_meta_path(model_name, pretrained).name}")
    print(f"Input {size}x{size}, {embed_dim}-dim embeddings")
    print("\nNext, build it for this board (from the repo root):")
    print("  docker build -t rknn-convert tools/rknn")
    print(f"  docker run --rm -v $PWD/data:/data rknn-convert {onnx_path} rk3588")
    return onnx_path


def check(paths: list[str], model_name: str, pretrained: str, runs: int = 5) -> int:
    """Compare NPU embeddings against PyTorch ones on real photos, and time both.

    This is the measurement the whole backend hinges on: an fp16 NPU build is not
    bit-identical to fp32 on the CPU, and the classifier only transfers between
    them if the two embeddings point essentially the same way."""
    import open_clip
    import torch

    model, _, transform = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval()
    encoder = Encoder(model_name, pretrained)

    def torch_vec(img):
        with torch.no_grad():
            feats = model.encode_image(transform(img).unsqueeze(0))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].numpy()

    def timed(fn, img):
        fn(img)  # warm up, so the first call's lazy setup isn't in the median
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            vec = fn(img)
            times.append((time.perf_counter() - t0) * 1000)
            if vec is None:
                return None, 0.0
        return vec, sorted(times)[len(times) // 2]

    print(f"{'photo':30s} {'cosine':>8s} {'cpu ms':>9s} {'npu ms':>9s} {'speedup':>8s}")
    cosines = []
    for path in paths:
        img = Image.open(path)
        cpu_vec, cpu_ms = timed(torch_vec, img)
        npu_vec, npu_ms = timed(encoder.encode, img)
        if npu_vec is None:
            print(f"{Path(path).name:30s} {'no output from the NPU':>36s}")
            continue
        if npu_vec.shape != cpu_vec.shape:
            print(f"{Path(path).name:30s} shape {npu_vec.shape} != torch {cpu_vec.shape}")
            return 1
        cos = float(np.dot(cpu_vec, npu_vec))
        cosines.append(cos)
        print(f"{Path(path).name:30s} {cos:8.5f} {cpu_ms:9.1f} {npu_ms:9.1f} {cpu_ms / npu_ms:7.1f}x")

    encoder.close()
    if not cosines:
        print("\nNothing was embedded.")
        return 1

    worst = min(cosines)
    print(f"\nMean cosine {sum(cosines) / len(cosines):.5f}, worst {worst:.5f}")
    if worst < MIN_COSINE:
        print(f"Below {MIN_COSINE}: the NPU is not reproducing the PyTorch embeddings.")
        return 1
    print("The NPU reproduces the PyTorch embeddings.")
    return 0


def image_paths(args: list[str]) -> list[str]:
    """Expand directory arguments into the image files inside them.

    A glob like /photos/*.jpg is expanded by the shell that types it, which for
    `docker run` is the host's, against a path that only exists in the container.
    Accepting the directory itself avoids handing anyone that trap."""
    paths = []
    for arg in args:
        path = Path(arg)
        if path.is_dir():
            paths.extend(sorted(str(f) for f in path.iterdir() if f.suffix.lower() in _IMAGE_SUFFIXES))
        else:
            paths.append(arg)
    return paths


def bench(paths: list[str], runs: int = 3) -> int:
    """Throughput of the real embedding pipeline, driven the way a scan drives it.

    Runs through embedder rather than calling the Encoder directly, so what gets
    measured is the arrangement that actually ships: SCAN_WORKERS threads doing
    preprocessing in parallel, feeding GPU_WORKERS workers. Comparing backends is
    a matter of running it again with BACKEND=cpu.

    Photos are decoded and scaled down to preview size first, outside the timed
    section, because a scan embeds Immich thumbnails and YOLO crops rather than
    full resolution originals; timing a 12 MP decode would drown the difference
    the backend makes."""
    from concurrent.futures import ThreadPoolExecutor

    import embedder as emb

    print(f"Loading {len(paths)} photos...")
    images = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        img.thumbnail((1440, 1440))  # roughly what Immich serves as a preview
        images.append(img)

    emb.start_workers()
    emb.wait_for_ready()

    work = images * runs
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=emb.SCAN_WORKERS) as pool:
        vecs = list(pool.map(emb.embed_image, work))
    elapsed = time.perf_counter() - started
    emb.stop_workers()

    done = sum(v is not None for v in vecs)
    print(f"\nBackend {npu.describe()}, {emb.GPU_WORKERS} workers, {emb.SCAN_WORKERS} scan threads")
    print(f"{done}/{len(work)} embeddings in {elapsed:.1f}s: {done / elapsed:.2f} photos/s, {elapsed / done * 1000:.0f} ms each")
    return 0 if done == len(work) else 1


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = sys.argv[1:]
    model_name, pretrained = _configured()

    if args and args[0] == "export":
        opset = int(args[1]) if len(args) > 1 else 18
        export(model_name, pretrained, opset)
        return 0
    if args and args[0] in ("check", "bench") and len(args) > 1:
        paths = image_paths(args[1:])
        if not paths:
            print(f"No images found in {' '.join(args[1:])}")
            return 1
        if args[0] == "bench":
            return bench(paths)
        return check(paths, model_name, pretrained)

    print(__doc__.split("\n\n")[1])  # the usage block
    return 1


if __name__ == "__main__":
    sys.exit(main())
