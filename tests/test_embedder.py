"""Tests for embedder's worker selection, in particular the NPU path."""
import sys
import types

import numpy as np
import pytest
from PIL import Image

import embedder as emb


class _FakeEncoder:
    """Stands in for rknn_clip.Encoder without an NPU behind it."""

    instances: list["_FakeEncoder"] = []

    def __init__(self, model_name, pretrained, soc=None):
        self.model_name = model_name
        self.pretrained = pretrained
        self.seen: list[np.ndarray] = []
        self.closed = False
        _FakeEncoder.instances.append(self)

    def preprocess(self, img):
        return np.full((1, 224, 224, 3), 7, dtype=np.uint8)

    def run(self, arr):
        self.seen.append(arr)
        return np.array([0.6, 0.8], dtype=np.float32)

    def close(self):
        self.closed = True


@pytest.fixture
def npu_backend(monkeypatch):
    """Point embedder at a fake NPU and tear its workers down afterwards."""
    _FakeEncoder.instances.clear()
    fake = types.ModuleType("rknn_clip")
    fake.Encoder = _FakeEncoder
    monkeypatch.setitem(sys.modules, "rknn_clip", fake)
    monkeypatch.setattr(emb.npu, "backend", lambda: "rknn")
    yield _FakeEncoder
    emb.stop_workers()


def test_embed_image_goes_through_the_npu_encoder(npu_backend):
    vec = emb.embed_image(Image.new("RGB", (300, 200)))
    assert vec.tolist() == pytest.approx([0.6, 0.8])
    assert any(e.seen for e in npu_backend.instances)


def test_each_worker_gets_its_own_encoder(npu_backend, monkeypatch):
    """RKNNLite cannot be shared between threads, so one Encoder per worker."""
    monkeypatch.setattr(emb, "GPU_WORKERS", 3)
    emb.start_workers()
    emb.wait_for_ready(timeout=10)
    assert len(npu_backend.instances) == 3


def test_preprocessing_happens_in_the_caller_not_the_worker(npu_backend):
    """The NPU worker must receive an already preprocessed array, so that resizing
    stays parallel across scan threads instead of serializing behind inference."""
    emb.embed_image(Image.new("RGB", (300, 200)))
    arr = next(a for e in npu_backend.instances for a in e.seen)
    assert arr.shape == (1, 224, 224, 3)
    assert arr.dtype == np.uint8


def test_workers_release_the_npu_on_shutdown(npu_backend):
    emb.embed_image(Image.new("RGB", (300, 200)))
    emb.stop_workers()
    assert all(e.closed for e in npu_backend.instances)


def test_a_failing_encoder_is_reported_not_swallowed(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("clip_ViT_B_16_openai_rk3588.rknn not found")

    fake = types.ModuleType("rknn_clip")
    fake.Encoder = boom
    monkeypatch.setitem(sys.modules, "rknn_clip", fake)
    monkeypatch.setattr(emb.npu, "backend", lambda: "rknn")
    emb.start_workers()
    with pytest.raises(RuntimeError, match="rknn"):
        emb.wait_for_ready(timeout=10)
    emb.stop_workers()
