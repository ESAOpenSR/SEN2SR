"""Tests for frame.api.services.model -- cached model loading.

Uses injected fake loader callables throughout, so these tests never touch
the network or download the real SEN2SRLite weights (Phase 7 requirement:
API unit tests must not require the real model unless explicitly marked
integration). The real `mlstac`-based loaders are exercised only by
frame/tests/test_api_integration.py.
"""

from pathlib import Path

from frame.api.services import model as model_service


def setup_function() -> None:
    model_service.clear_cache()


def teardown_function() -> None:
    model_service.clear_cache()


def test_get_model_calls_ensure_weights_and_load_compiled_model():
    calls = {"ensure_weights": 0, "load": 0}

    def fake_ensure_weights(cache_dir: Path) -> Path:
        calls["ensure_weights"] += 1
        return cache_dir

    def fake_load(weight_dir: Path, device: str):
        calls["load"] += 1
        return f"fake-model-on-{device}"

    result = model_service.get_model("cpu", ensure_weights=fake_ensure_weights, load_compiled_model=fake_load)
    assert result == "fake-model-on-cpu"
    assert calls == {"ensure_weights": 1, "load": 1}


def test_get_model_is_cached_and_not_reloaded_on_second_call():
    calls = {"load": 0}

    def fake_ensure_weights(cache_dir: Path) -> Path:
        return cache_dir

    def fake_load(weight_dir: Path, device: str):
        calls["load"] += 1
        return object()

    first = model_service.get_model("cpu", ensure_weights=fake_ensure_weights, load_compiled_model=fake_load)
    second = model_service.get_model("cpu", ensure_weights=fake_ensure_weights, load_compiled_model=fake_load)
    assert first is second
    assert calls["load"] == 1


def test_different_devices_are_cached_independently():
    def fake_ensure_weights(cache_dir: Path) -> Path:
        return cache_dir

    def fake_load(weight_dir: Path, device: str):
        return f"model-{device}"

    cpu_model = model_service.get_model("cpu", ensure_weights=fake_ensure_weights, load_compiled_model=fake_load)
    cuda_model = model_service.get_model("cuda", ensure_weights=fake_ensure_weights, load_compiled_model=fake_load)
    assert cpu_model == "model-cpu"
    assert cuda_model == "model-cuda"


def test_clear_cache_forces_a_reload():
    calls = {"load": 0}

    def fake_ensure_weights(cache_dir: Path) -> Path:
        return cache_dir

    def fake_load(weight_dir: Path, device: str):
        calls["load"] += 1
        return object()

    model_service.get_model("cpu", ensure_weights=fake_ensure_weights, load_compiled_model=fake_load)
    model_service.clear_cache()
    model_service.get_model("cpu", ensure_weights=fake_ensure_weights, load_compiled_model=fake_load)
    assert calls["load"] == 2


def test_resolve_device_returns_explicit_device_unchanged():
    assert model_service.resolve_device("cpu") == "cpu"
    assert model_service.resolve_device("cuda") == "cuda"


def test_resolve_device_auto_resolves_to_cpu_or_cuda():
    resolved = model_service.resolve_device("auto")
    assert resolved in ("cpu", "cuda")
