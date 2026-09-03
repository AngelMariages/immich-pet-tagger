"""Tests for locating the built YOLO model and reporting when it isn't usable."""
import json

import pytest

import rknn_yolo


@pytest.fixture
def model_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(rknn_yolo, "MODEL_DIR", tmp_path)
    return tmp_path / "yolov8n_rknn_model"


def _build(model_dir, soc="rk3588", input_size=640):
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / f"yolov8n_{soc}.rknn").write_bytes(b"fake")
    (model_dir / "yolov8n.json").write_text(json.dumps({"input_size": input_size}))


def test_model_dir_is_named_the_way_ultralytics_names_its_exports(monkeypatch, tmp_path):
    monkeypatch.setattr(rknn_yolo, "MODEL_DIR", tmp_path)
    assert rknn_yolo.model_dir("yolov8n.pt").name == "yolov8n_rknn_model"


def test_model_path_returns_the_build_for_this_soc(model_dir):
    _build(model_dir)
    assert rknn_yolo.model_path("yolov8n.pt", 640, soc="rk3588").name == "yolov8n_rk3588.rknn"


def test_model_path_reports_a_missing_build_with_the_command_that_makes_it(model_dir):
    with pytest.raises(RuntimeError, match="rknn_yolo.py export"):
        rknn_yolo.model_path("yolov8n.pt", 640, soc="rk3588")


def test_model_path_reports_an_unknown_soc(model_dir, monkeypatch):
    monkeypatch.setattr(rknn_yolo.npu, "soc", lambda: None)
    with pytest.raises(RuntimeError, match="RKNN_SOC"):
        rknn_yolo.model_path("yolov8n.pt", 640)


def test_model_path_catches_an_input_size_it_was_not_built_for(model_dir):
    """An .rknn has its input shape baked in, so YOLO_INPUT_SIZE changing after a
    build has to be an error rather than a confusing failure inside the runtime."""
    _build(model_dir, input_size=640)
    with pytest.raises(RuntimeError, match="built for 640px"):
        rknn_yolo.model_path("yolov8n.pt", 1280, soc="rk3588")


def test_model_path_accepts_a_build_with_no_sidecar(model_dir):
    """A model built by hand, or by Ultralytics' own exporter, has no sidecar."""
    _build(model_dir)
    (model_dir / "yolov8n.json").unlink()
    assert rknn_yolo.model_path("yolov8n.pt", 640, soc="rk3588").exists()
