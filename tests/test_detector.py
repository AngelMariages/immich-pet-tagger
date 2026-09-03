"""Tests for how the YOLO worker loads its model and sizes its batches."""
import sys
import threading
import types

import numpy as np
import pytest
import torch

import detector as det


class _FakeResult:
    boxes: list = []


class _FakeYOLO:
    """Records how it was constructed and how large each batch it ran was."""

    built: list = []
    batches: list = []

    def __init__(self, model, task=None):
        self.model = model
        self.task = task
        _FakeYOLO.built.append((str(model), task))

    def to(self, device):
        self.device = device
        return self

    def __call__(self, stacked, **kwargs):
        _FakeYOLO.batches.append(len(stacked))
        return [_FakeResult() for _ in range(len(stacked))]


@pytest.fixture
def fake_yolo(monkeypatch):
    _FakeYOLO.built.clear()
    _FakeYOLO.batches.clear()
    monkeypatch.setattr(sys.modules["ultralytics"], "YOLO", _FakeYOLO)
    monkeypatch.setattr(torch, "stack", lambda tensors, **kw: list(tensors))
    yield _FakeYOLO
    det.stop_workers()


def _run_one(image_count=1):
    for _ in range(image_count):
        det.detect_animals(_FakeImage())


class _FakeImage:
    size = (640, 640)

    def resize(self, size, resample=None):
        return self

    def __array__(self, dtype=None):
        return np.zeros((640, 640, 3), dtype=dtype or np.float32)


def test_cpu_path_loads_the_configured_weights(fake_yolo, monkeypatch):
    monkeypatch.setattr(det.npu, "backend", lambda: "cpu")
    _run_one()
    assert set(fake_yolo.built) == {("yolov8n.pt", None)}
    assert len(fake_yolo.built) == det.YOLO_WORKERS  # one model per worker


def test_npu_path_loads_the_built_rknn_model(fake_yolo, monkeypatch):
    monkeypatch.setattr(det.npu, "backend", lambda: "rknn")
    fake = types.ModuleType("rknn_yolo")
    fake.model_path = lambda name, size, soc=None: f"/data/rknn/{name}_{size}.rknn"
    monkeypatch.setitem(sys.modules, "rknn_yolo", fake)
    _run_one()
    assert set(fake_yolo.built) == {("/data/rknn/yolov8n.pt_640.rknn", "detect")}


def test_npu_path_runs_one_image_at_a_time(fake_yolo, monkeypatch):
    """The .rknn model's batch size is fixed at build time, so batching would fail."""
    monkeypatch.setattr(det.npu, "backend", lambda: "rknn")
    monkeypatch.setattr(det, "YOLO_WORKERS", 1)
    fake = types.ModuleType("rknn_yolo")
    fake.model_path = lambda name, size, soc=None: "/data/model.rknn"
    monkeypatch.setitem(sys.modules, "rknn_yolo", fake)

    threads = [threading.Thread(target=_run_one) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert fake_yolo.batches and max(fake_yolo.batches) == 1
