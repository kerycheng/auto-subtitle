import copy
import json
from threading import Event
from types import SimpleNamespace as NS

import numpy as np
import pytest

from autosub.domain import Project, Word, Cue, word_key
from autosub.media import Cancelled
from autosub.native_pipeline import source_clips
from autosub.pipeline import run_pipeline
from autosub.timing import apply_timing


def test_source_clips_keep_short_pauses_and_original_clock():
    speech = [[5000, 8000], [12000, 15000], [21000, 24000]]
    assert source_clips(speech, 30000) == [4., 16., 20., 25.]
    assert source_clips(speech, 30000, 13000) == [13., 16., 20., 25.]


def setup(tmp_path, monkeypatch, fail=False):
    import autosub.pipeline as pipeline
    import autosub.native_pipeline as native
    import faster_whisper
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    monkeypatch.setattr(pipeline, "decode_audio", lambda *_: None)
    monkeypatch.setattr(pipeline, "load_audio", lambda _: np.zeros(70000 * 16, dtype=np.float32))
    monkeypatch.setattr(native, "check_runtime", lambda _: {"compute_type": "int8"})
    calls = []
    def segment(seek, start, end, text):
        return NS(seek=seek, text=text, words=[NS(start=start, end=end, word=text, probability=.9)])
    all_segments = [segment(100, 1, 3, "前の文。"), segment(100, 4, 8, "東京駅です。"),
                    segment(2600, 26, 28, "で次の文。"), segment(2600, 30, 34, "最後です。")]
    class Model:
        frames_per_second = 100
        def __init__(self, *args, **kwargs): pass
        def transcribe(self, audio, **kwargs):
            calls.append((len(audio), kwargs))
            def output():
                for s in all_segments:
                    if s.seek / 100 >= kwargs["clip_timestamps"][0]:
                        yield s
                if fail and len(calls) == 1:
                    raise RuntimeError("simulated GPU failure")
            return output(), None
    monkeypatch.setattr(faster_whisper, "WhisperModel", Model)
    return Project(media_path=str(source), speech=[[1000, 65000]]), tmp_path / "new.autosub.json", calls


def test_default_ja_preserves_native_groups_and_retimes_after_reload(tmp_path, monkeypatch):
    p, path, calls = setup(tmp_path, monkeypatch)
    run_pipeline(p, path, Event(), lambda *_: None)
    assert p.asr_strategy == "native-v1" and p.native_complete and p.native_resume_ms == 70000
    assert p.native_groups == [[0], [1], [2], [3]] and p.completed_chunks == []
    assert len(p.cues) == 4 and p.cues[2].text == "で次の文。"
    assert calls[0][0] == 70000*16 and calls[0][1]["clip_timestamps"] == [0., 66.]
    assert not calls[0][1]["vad_filter"] and calls[0][1]["word_timestamps"]
    loaded = Project.load(path)
    assert apply_timing(loaded) == p.cues
    run_pipeline(loaded, path, Event(), lambda *_: None)
    assert len(calls) == 1  # no model inference when recognition is complete


def test_cancel_resume_keeps_whole_batch_once_and_source_clock(tmp_path, monkeypatch):
    p, path, calls = setup(tmp_path, monkeypatch)
    cancel = Event()
    def stop(_percent, message):
        if "已保存至 26.0" in message:
            cancel.set()
    with pytest.raises(Cancelled):
        run_pipeline(p, path, cancel, stop)
    saved = Project.load(path)
    assert saved.status == "cancelled" and saved.native_resume_ms == 26000
    assert saved.native_groups == [[0], [1]] and not saved.native_complete
    prefix = copy.deepcopy(saved.words)
    saved.device = "cpu"
    run_pipeline(saved, path, Event(), lambda *_: None)
    assert saved.words[:2] == prefix and len(saved.words) == 4
    assert calls[1][1]["clip_timestamps"] == [26., 66.]
    assert saved.words[2].start_ms == 26000


def test_failed_partial_batch_is_discarded_and_cpu_retry_keeps_prefix(tmp_path, monkeypatch):
    p, path, calls = setup(tmp_path, monkeypatch, fail=True)
    p.device = "cuda"
    with pytest.raises(RuntimeError, match="CPU"):
        run_pipeline(p, path, Event(), lambda *_: None)
    saved = Project.load(path)
    assert saved.native_groups == [[0], [1]] and len(saved.words) == 2
    assert saved.native_resume_ms == 26000 and saved.status == "failed"
    saved.device = "cpu"
    run_pipeline(saved, path, Event(), lambda *_: None)
    assert len(saved.words) == 4 and saved.native_complete


def test_default_native_retiming_preserves_manual_lock(tmp_path):
    words = [Word(100,700,"最初。"), Word(900,1600,"手動。"), Word(1900,2500,"最後。")]
    p = Project(duration_ms=4000, words=words, native_groups=[[0,1,2]], asr_strategy="native-v1")
    p.cues = [Cue("manual",880,1700,"人工修正",900,1600,locked=True,source_tokens=[word_key(words[1])])]
    p.save(tmp_path / "locked.json")
    result = apply_timing(Project.load(tmp_path / "locked.json"))
    assert [c.text for c in result] == ["最初。", "人工修正", "最後。"]
    assert result[1] == p.cues[0]


def test_schema3_migrates_without_changing_legacy_boundaries_or_original_bytes(tmp_path):
    path = tmp_path / "old.autosub.json"
    Project().save(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["schema_version"] = 3
    for key in ("native_groups", "native_resume_ms", "native_complete"):
        raw.pop(key)
    original = json.dumps(raw, ensure_ascii=False, indent=3).encode("utf-8")
    path.write_bytes(original)
    p = Project.load(path)
    assert p.schema_version == 4 and p.asr_strategy == "fixed30-v1"
    assert path.read_bytes() == original
    p.save(path)
    assert path.with_suffix(".json.v3.bak").read_bytes() == original


@pytest.mark.parametrize("change", [
    {"native_groups": [[0,0]]}, {"native_groups": [[True]]}, {"native_groups": []},
    {"native_resume_ms": -1}, {"native_resume_ms": True}, {"native_complete": True},
    {"native_groups": [[1]]}, {"native_groups": [[0], []]}, {"language": "en"},
    {"completed_chunks": [0]},
])
def test_bad_native_provenance_rejected(change):
    p = Project(duration_ms=1000, words=[Word(100,500,"文")], asr_strategy="native-v1", native_groups=[[0]])
    for key, value in change.items():
        setattr(p, key, value)
    with pytest.raises(ValueError):
        p.validate()
