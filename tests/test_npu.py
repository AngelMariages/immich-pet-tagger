"""Tests for Rockchip SoC detection and inference backend selection."""
import pytest

import npu


def _device_tree(tmp_path, *entries: str):
    """Write a device-tree `compatible` file: NUL-separated, NUL-terminated."""
    path = tmp_path / "compatible"
    path.write_text("\x00".join(entries) + "\x00")
    return path


@pytest.fixture(autouse=True)
def default_host(monkeypatch, tmp_path):
    """No NPU, no CUDA, no runtime wheel, no env overrides."""
    monkeypatch.setattr(npu, "BACKEND", "auto")
    monkeypatch.setattr(npu, "RKNN_SOC", "")
    monkeypatch.setattr(npu, "_DEVICE_TREE", tmp_path / "missing")
    monkeypatch.setattr(npu, "_has_runtime", lambda: False)
    monkeypatch.setattr(npu.torch.cuda, "is_available", lambda: False)


def test_soc_reads_last_device_tree_entry(monkeypatch, tmp_path):
    monkeypatch.setattr(npu, "_DEVICE_TREE", _device_tree(tmp_path, "radxa,rock-5b", "rockchip,rk3588"))
    assert npu.soc() == "rk3588"


def test_soc_scans_every_entry_not_just_the_last(monkeypatch, tmp_path):
    monkeypatch.setattr(npu, "_DEVICE_TREE", _device_tree(tmp_path, "rockchip,rk3566", "vendor,some-board"))
    assert npu.soc() == "rk3566"


def test_soc_maps_rk3588s_onto_its_toolkit_platform(monkeypatch, tmp_path):
    monkeypatch.setattr(npu, "_DEVICE_TREE", _device_tree(tmp_path, "orangepi,opi5", "rockchip,rk3588s"))
    assert npu.soc() == "rk3588"


def test_soc_ignores_unsupported_socs(monkeypatch, tmp_path):
    monkeypatch.setattr(npu, "_DEVICE_TREE", _device_tree(tmp_path, "raspberrypi,4-model-b", "brcm,bcm2711"))
    assert npu.soc() is None


def test_soc_without_a_device_tree():
    """The usual case in a container: /sys/firmware is masked by Docker."""
    assert npu.soc() is None


def test_soc_env_override_when_the_device_tree_is_masked(monkeypatch, tmp_path):
    monkeypatch.setattr(npu, "RKNN_SOC", "rk3588")
    assert npu.soc() == "rk3588"


def test_soc_env_override_is_aliased_too(monkeypatch):
    monkeypatch.setattr(npu, "RKNN_SOC", "rk3588s")
    assert npu.soc() == "rk3588"


def test_backend_defaults_to_cpu():
    assert npu.backend() == "cpu"


def test_backend_prefers_cuda_when_available(monkeypatch):
    monkeypatch.setattr(npu.torch.cuda, "is_available", lambda: True)
    assert npu.backend() == "cuda"


def test_backend_picks_rknn_when_the_runtime_wheel_is_present(monkeypatch, tmp_path):
    monkeypatch.setattr(npu, "_DEVICE_TREE", _device_tree(tmp_path, "rockchip,rk3588"))
    monkeypatch.setattr(npu, "_has_runtime", lambda: True)
    assert npu.backend() == "rknn"


def test_backend_picks_rknn_even_with_an_unreadable_device_tree(monkeypatch):
    """A container almost never sees the device tree, so the NPU cannot depend on it."""
    monkeypatch.setattr(npu, "_has_runtime", lambda: True)
    assert npu.soc() is None
    assert npu.backend() == "rknn"


def test_backend_stays_on_cpu_without_the_runtime_wheel(monkeypatch, tmp_path):
    """The plain arm64 CPU image runs on the same boards and must not change behavior."""
    monkeypatch.setattr(npu, "_DEVICE_TREE", _device_tree(tmp_path, "rockchip,rk3588"))
    assert npu.backend() == "cpu"


@pytest.mark.parametrize("pinned", ["cpu", "cuda", "rknn"])
def test_backend_env_override_wins_over_detection(monkeypatch, tmp_path, pinned):
    monkeypatch.setattr(npu, "BACKEND", pinned)
    monkeypatch.setattr(npu, "_DEVICE_TREE", _device_tree(tmp_path, "rockchip,rk3588"))
    monkeypatch.setattr(npu, "_has_runtime", lambda: True)
    assert npu.backend() == pinned


def test_describe_is_just_the_backend_without_an_npu():
    assert npu.describe() == "cpu"


def test_describe_names_the_soc_when_the_npu_is_in_use(monkeypatch, tmp_path):
    monkeypatch.setattr(npu, "_DEVICE_TREE", _device_tree(tmp_path, "rockchip,rk3588"))
    monkeypatch.setattr(npu, "_has_runtime", lambda: True)
    assert npu.describe() == "rknn (rk3588)"


def test_describe_admits_when_the_soc_is_unknown(monkeypatch):
    monkeypatch.setattr(npu, "_has_runtime", lambda: True)
    assert npu.describe() == "rknn (SoC unknown)"


def test_describe_explains_a_missing_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(npu, "_DEVICE_TREE", _device_tree(tmp_path, "rockchip,rk3588"))
    assert "no RKNPU runtime" in npu.describe()


def test_describe_explains_an_npu_skipped_by_config(monkeypatch, tmp_path):
    monkeypatch.setattr(npu, "BACKEND", "cpu")
    monkeypatch.setattr(npu, "_DEVICE_TREE", _device_tree(tmp_path, "rockchip,rk3588"))
    monkeypatch.setattr(npu, "_has_runtime", lambda: True)
    assert "not selected by BACKEND=cpu" in npu.describe()
