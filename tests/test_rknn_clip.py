"""Tests for CLIP-on-NPU preprocessing, model file naming, and Encoder setup."""
import json
import sys
import types

import numpy as np
import pytest
from PIL import Image

import rknn_clip


def test_stem_sanitizes_model_names_into_a_filename():
    assert rknn_clip.stem("ViT-B-16", "openai") == "clip_ViT_B_16_openai"


def test_stem_survives_a_pretrained_tag_with_slashes():
    assert rknn_clip.stem("ViT-B-32", "laion2b_s34b_b79k") == "clip_ViT_B_32_laion2b_s34b_b79k"


# ---------------------------------------------------------------------------
# preprocess: the NPU is fed raw uint8 NHWC, normalization happens in the model
# ---------------------------------------------------------------------------

def test_preprocess_returns_a_single_uint8_nhwc_batch():
    arr = rknn_clip.preprocess(Image.new("RGB", (640, 480)), 224)
    assert arr.shape == (1, 224, 224, 3)
    assert arr.dtype == np.uint8


def test_preprocess_scales_the_short_side_and_crops_the_long_one():
    """A portrait photo keeps its full width and loses the top and bottom."""
    img = Image.new("RGB", (100, 400), "black")
    img.paste(Image.new("RGB", (100, 40), "white"), (0, 180))  # centered band
    arr = rknn_clip.preprocess(img, 224)[0]
    assert arr.shape == (224, 224, 3)
    assert arr[112, 112].tolist() == [255, 255, 255]  # the middle survives
    assert arr[0, 112].tolist() == [0, 0, 0]          # the far ends are cropped away
    assert arr[223, 112].tolist() == [0, 0, 0]


def test_preprocess_converts_non_rgb_images():
    arr = rknn_clip.preprocess(Image.new("L", (300, 300), 128), 224)
    assert arr.shape == (1, 224, 224, 3)
    assert arr[0, 0, 0].tolist() == [128, 128, 128]


def test_preprocess_leaves_values_untouched():
    """No /255 and no mean/std: those are compiled into the .rknn model, and
    doing them here too would apply them twice."""
    arr = rknn_clip.preprocess(Image.new("RGB", (224, 224), (10, 20, 30)), 224)
    assert arr[0, 100, 100].tolist() == [10, 20, 30]


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class _FakeRKNNLite:
    """Stands in for rknnlite.api.RKNNLite, recording what it was asked to do."""

    NPU_CORE_AUTO = 0
    instances: list["_FakeRKNNLite"] = []

    def __init__(self):
        self.loaded = None
        self.core_mask = None
        self.released = False
        self.output = [np.array([[3.0, 4.0]], dtype=np.float32)]
        _FakeRKNNLite.instances.append(self)

    def load_rknn(self, path):
        self.loaded = path
        return 0

    def init_runtime(self, core_mask=None):
        self.core_mask = core_mask
        return 0

    def inference(self, inputs, data_format=None):
        self.inputs = inputs
        self.data_format = data_format
        return self.output

    def release(self):
        self.released = True


@pytest.fixture
def fake_rknnlite(monkeypatch):
    _FakeRKNNLite.instances.clear()
    api = types.ModuleType("rknnlite.api")
    api.RKNNLite = _FakeRKNNLite
    pkg = types.ModuleType("rknnlite")
    pkg.api = api
    monkeypatch.setitem(sys.modules, "rknnlite", pkg)
    monkeypatch.setitem(sys.modules, "rknnlite.api", api)
    return _FakeRKNNLite


@pytest.fixture
def model_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(rknn_clip, "MODEL_DIR", tmp_path)
    return tmp_path


def _write_model(model_dir, soc="rk3588", **meta):
    (model_dir / f"clip_ViT_B_16_openai_{soc}.rknn").write_bytes(b"fake")
    (model_dir / "clip_ViT_B_16_openai.json").write_text(json.dumps({"input_size": 224, **meta}))


def test_encoder_reports_a_missing_model_with_the_command_that_builds_it(model_dir):
    with pytest.raises(RuntimeError, match="rknn_clip.py export"):
        rknn_clip.Encoder("ViT-B-16", "openai", soc="rk3588")


def test_encoder_reports_an_unknown_soc(model_dir, monkeypatch):
    monkeypatch.setattr(rknn_clip.npu, "soc", lambda: None)
    with pytest.raises(RuntimeError, match="RKNN_SOC"):
        rknn_clip.Encoder("ViT-B-16", "openai")


def test_encoder_loads_the_model_for_the_detected_soc(model_dir, fake_rknnlite, monkeypatch):
    monkeypatch.setattr(rknn_clip.npu, "soc", lambda: "rk3588")
    _write_model(model_dir)
    rknn_clip.Encoder("ViT-B-16", "openai")
    assert fake_rknnlite.instances[0].loaded.endswith("clip_ViT_B_16_openai_rk3588.rknn")
    assert fake_rknnlite.instances[0].core_mask == _FakeRKNNLite.NPU_CORE_AUTO


def test_encoder_takes_the_input_size_from_the_sidecar(model_dir, fake_rknnlite):
    _write_model(model_dir, input_size=336)
    encoder = rknn_clip.Encoder("ViT-B-16", "openai", soc="rk3588")
    encoder.encode(Image.new("RGB", (400, 400)))
    assert fake_rknnlite.instances[0].inputs[0].shape == (1, 336, 336, 3)


def test_encode_returns_a_unit_vector(model_dir, fake_rknnlite):
    encoder = _write_model(model_dir) or rknn_clip.Encoder("ViT-B-16", "openai", soc="rk3588")
    vec = encoder.encode(Image.new("RGB", (300, 300)))
    assert vec.tolist() == pytest.approx([0.6, 0.8])  # 3,4 normalized
    assert float(np.linalg.norm(vec)) == pytest.approx(1.0)


def test_encode_feeds_the_npu_uint8_nhwc(model_dir, fake_rknnlite):
    _write_model(model_dir)
    encoder = rknn_clip.Encoder("ViT-B-16", "openai", soc="rk3588")
    encoder.encode(Image.new("RGB", (300, 200)))
    fake = fake_rknnlite.instances[0]
    assert fake.data_format == "nhwc"
    assert fake.inputs[0].dtype == np.uint8


def test_encode_returns_none_when_the_npu_produces_nothing(model_dir, fake_rknnlite):
    _write_model(model_dir)
    encoder = rknn_clip.Encoder("ViT-B-16", "openai", soc="rk3588")
    fake_rknnlite.instances[0].output = []
    assert encoder.encode(Image.new("RGB", (300, 300))) is None


def test_encode_rejects_a_zero_vector_instead_of_dividing_by_zero(model_dir, fake_rknnlite):
    _write_model(model_dir)
    encoder = rknn_clip.Encoder("ViT-B-16", "openai", soc="rk3588")
    fake_rknnlite.instances[0].output = [np.zeros((1, 2), dtype=np.float32)]
    assert encoder.encode(Image.new("RGB", (300, 300))) is None


# ---------------------------------------------------------------------------
# CLI argument handling
# ---------------------------------------------------------------------------

def test_image_paths_expands_a_directory(tmp_path):
    for name in ("b.jpg", "a.png", "notes.txt"):
        (tmp_path / name).touch()
    assert rknn_clip.image_paths([str(tmp_path)]) == [str(tmp_path / "a.png"), str(tmp_path / "b.jpg")]


def test_image_paths_passes_files_through(tmp_path):
    photo = tmp_path / "one.jpg"
    photo.touch()
    assert rknn_clip.image_paths([str(photo)]) == [str(photo)]


def test_image_paths_of_an_empty_directory_is_empty(tmp_path):
    assert rknn_clip.image_paths([str(tmp_path)]) == []
