import json
from dataclasses import asdict

import pytest

from autosub.app import MainWindow
from autosub.domain import Project, TimingSettings
from autosub.model_cache import clear_model, get_model
from autosub.preferences import load_preferences, preferences_path
from autosub.runtime import check_runtime


def test_preferences_round_trip_project_precedence_and_missing_device(qtbot):
    first = MainWindow()
    qtbot.addWidget(first)
    first.volume.setValue(37)
    first.language.setCurrentIndex(first.language.findData("ko"))
    first.model.setCurrentText("tiny")
    first.glossary.setPlainText("ARK, エーペックス")
    first.timing_inputs["tail_ms"].setValue(120)
    first.save_user_preferences()
    raw, warning = load_preferences()
    assert not warning and raw["volume"] == 37
    raw["output_device"] = b"missing-device".hex()
    preferences_path().write_text(json.dumps(raw), encoding="utf-8")
    second = MainWindow()
    qtbot.addWidget(second)
    assert second.volume.value() == 37 and second.audio_output.volume() == pytest.approx(.37)
    assert second.language.currentData() == "ko" and second.model.currentText() == "tiny"
    assert second.timing_inputs["tail_ms"].value() == 120
    assert second.output_devices.currentData() is None and "不存在" in second.status.text()
    second.project = Project(language="ja", model="small", settings=TimingSettings(tail_ms=250))
    second.load_controls()
    assert second.language.currentData() == "ja" and second.model.currentText() == "small"
    assert second.timing_inputs["tail_ms"].value() == 250
    assert second.volume.value() == 37  # project settings do not override audio preferences


def test_invalid_preferences_stay_untouched_until_explicit_save(qtbot):
    path = preferences_path()
    path.write_text("broken settings", encoding="utf-8")
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.volume.value() == 80 and "原檔保留" in window.status.text()
    assert path.read_text(encoding="utf-8") == "broken settings"
    window.save_user_preferences()
    assert path.with_suffix(".json.invalid.bak").read_text(encoding="utf-8") == "broken settings"
    assert load_preferences()[0]["schema_version"] == 1


def test_output_device_refresh_clears_disconnected_binding(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.output_devices.addItem("removed output", b"disconnected")
    window.output_devices.setCurrentIndex(window.output_devices.count() - 1)
    assert window.audio._device_id == b"disconnected"
    window.refresh_output_devices()
    assert window.output_devices.currentData() is None
    assert window.audio._device_id is None


def test_runtime_no_gpu_and_unsupported_precision_provide_actionable_error(monkeypatch):
    import ctranslate2
    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 0)
    with pytest.raises(RuntimeError, match="CPU 繼續"):
        check_runtime("cuda")
    monkeypatch.setattr(ctranslate2, "get_supported_compute_types", lambda _: {"float32"})
    with pytest.raises(RuntimeError, match="int8"):
        check_runtime("cpu")


def test_cpu_retry_preserves_saved_chunks_words_and_manual_cues(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    window.project = Project(device="cuda", status="failed", completed_chunks=[0, 1])
    original = asdict(window.project)
    started = []
    monkeypatch.setattr(window, "start", lambda: started.append(asdict(window.project)))
    window.retry_on_cpu()
    assert len(started) == 1
    event = started[0]["review_events"][-1]
    assert event["action"] == "device_fallback" and event["preserved_chunks"] == [0, 1]
    assert started[0] == {**original, "device": "cpu", "review_events": [event]}
    assert window.device.currentText() == "cpu"
    window.retry_on_cpu()
    assert len(started) == 1


def test_single_model_cache_reuses_and_invalidates_changed_local_weights(tmp_path):
    clear_model()
    weights = tmp_path / "model.bin"
    weights.write_bytes(b"original")
    calls = []
    def factory(*args, **kwargs):
        calls.append((args, kwargs))
        return object()
    one, reused = get_model(factory, str(tmp_path), "cpu", "int8")
    assert not reused
    two, reused = get_model(factory, str(tmp_path), "cpu", "int8")
    assert reused and one is two and len(calls) == 1
    weights.write_bytes(b"new-weights")
    three, reused = get_model(factory, str(tmp_path), "cpu", "int8")
    assert not reused and three is not one and len(calls) == 2
    clear_model()


def test_failed_model_load_is_not_cached():
    clear_model()
    count = []
    def factory(*args, **kwargs):
        count.append(1)
        if len(count) == 1:
            raise RuntimeError("load failed")
        return object()
    with pytest.raises(RuntimeError, match="load failed"):
        get_model(factory, "tiny", "cpu", "int8")
    _, reused = get_model(factory, "tiny", "cpu", "int8")
    assert not reused and len(count) == 2
    clear_model()
