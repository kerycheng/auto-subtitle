import hashlib
import json

import pytest

from autosub.domain import Cue, Word
from autosub.language_validation import normalized_text, run_language_validation


def fixture(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.bin").write_bytes(b"placeholder")
    audio = tmp_path / "sample.audio"
    audio.write_bytes(b"test audio")
    samples = [{"id": f"{language}-{index}", "language": language, "dataset_id": index,
        "reference_text": "Dataset reference.", "audio_path": str(audio),
        "audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest()}
        for language in ("en", "zh", "ja", "ko") for index in range(2)]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    return manifest, model, samples


def test_normalization_keeps_script_differences_explicit():
    assert normalized_text("ＡＢＣ，한국어 テスト。你好！") == "abc한국어テスト你好"
    assert normalized_text("漢") != normalized_text("汉")


def test_four_languages_export_and_reference_is_not_given_to_pipeline(tmp_path, monkeypatch):
    import autosub.language_validation as validation
    manifest, model, _ = fixture(tmp_path)
    languages = []
    def pipeline(project, path, cancel, progress):
        languages.append(project.language)
        assert project.glossary == "" and project.words == [] and project.cues == []
        project.duration_ms = 1000
        project.words = [Word(100, 700, "テスト")]
        project.cues = [Cue("a", 80, 900, "テスト", 100, 700)]
        return project
    monkeypatch.setattr(validation, "run_pipeline", pipeline)
    output = tmp_path / "results"
    report = run_language_validation(manifest, model, output)
    assert report["status"] == "passed" and len(report["samples"]) == 8
    assert set(languages) == {"en", "zh", "ja", "ko"}
    assert all(row["mechanical_cer"] > 0 for row in report["samples"])  # passing is not an accuracy claim
    assert len(list(output.glob("*.ass"))) == len(list(output.glob("*.srt"))) == 8
    assert report["acoustic_timing_accuracy"] is None


@pytest.mark.parametrize("problem", ["missing_language", "changed_audio", "duplicate", "empty_reference"])
def test_invalid_manifest_rejected_before_any_asr_or_output(tmp_path, monkeypatch, problem):
    import autosub.language_validation as validation
    manifest, model, samples = fixture(tmp_path)
    if problem == "missing_language":
        samples = samples[:-2]
    elif problem == "changed_audio":
        samples[0]["audio_sha256"] = "different"
    elif problem == "duplicate":
        samples[1]["id"] = samples[0]["id"]
    else:
        samples[0]["reference_text"] = "..."
    manifest.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    monkeypatch.setattr(validation, "run_pipeline", lambda *_: pytest.fail("ASR must not be called"))
    output = tmp_path / "results"
    with pytest.raises(ValueError):
        run_language_validation(manifest, model, output)
    assert not output.exists()
