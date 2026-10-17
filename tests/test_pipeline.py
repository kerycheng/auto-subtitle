from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from autosub.domain import Project, Word
from autosub.media import Cancelled, decode_audio, fingerprint, load_audio
from autosub.pipeline import run_pipeline


def test_gpu_library_failure_saves_progress_and_explains_cpu_recovery(tmp_path, monkeypatch):
    import autosub.pipeline as pipeline
    import faster_whisper
    from autosub.domain import Cue
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    monkeypatch.setattr(pipeline, "decode_audio", lambda *_: None)
    monkeypatch.setattr(pipeline, "load_audio", lambda _: np.zeros(61 * 16000, dtype=np.float32))
    monkeypatch.setattr(pipeline, "check_runtime", lambda _: {"compute_type": "float16"})
    def broken_model(*args, **kwargs):
        raise RuntimeError("cublas64_12.dll not found")
    monkeypatch.setattr(faster_whisper, "WhisperModel", broken_model)
    p = Project(media_path=str(source), duration_ms=61000, device="cuda", speech=[[0, 61000]],
        completed_chunks=[0], words=[Word(100, 800, "保存。")],
        cues=[Cue("manual", 80, 1000, "人工修正", 100, 800, locked=True)])
    path = tmp_path / "gpu.autosub.json"
    with pytest.raises(RuntimeError, match="CPU 繼續") as error:
        run_pipeline(p, path, Event(), lambda *_: None)
    assert "cublas64_12.dll" in str(error.value)
    saved = Project.load(path)
    assert saved.status == "failed" and saved.completed_chunks == [0]
    assert saved.cues[0].locked and saved.cues[0].text == "人工修正"
    assert saved.words == [Word(100, 800, "保存。")]


def make_wav(path: Path, seconds=1):
    import wave
    samples = (np.sin(np.arange(seconds * 16000) * 2 * np.pi * 440 / 16000) * 5000).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(samples.tobytes())


def test_real_ffmpeg_round_trip_unicode_and_spaces(tmp_path):
    source = tmp_path / "日本語 空白.wav"
    output = tmp_path / "cache" / "decoded.wav"
    make_wav(source)
    decode_audio(source, output, 0, Event())
    assert len(load_audio(output)) == 16000


def test_changed_source_rejects_stale_resume(tmp_path):
    source = tmp_path / "test.wav"
    make_wav(source)
    p = Project(media_path=str(source), source_fingerprint={"size": -1})
    with pytest.raises(ValueError, match="來源檔案已改變"):
        run_pipeline(p, tmp_path / "test.autosub.json", Event(), lambda *_: None)


def test_cancel_checkpoint_can_be_loaded(tmp_path):
    source = tmp_path / "test.wav"
    make_wav(source)
    p = Project(media_path=str(source))
    cancel = Event()
    cancel.set()
    path = tmp_path / "test.autosub.json"
    with pytest.raises(Cancelled):
        run_pipeline(p, path, cancel, lambda *_: None)
    assert Project.load(path).status == "cancelled"


def test_completed_chunks_do_not_load_whisper_again(tmp_path, monkeypatch):
    import faster_whisper
    source = tmp_path / "test.wav"
    make_wav(source)
    p = Project(media_path=str(source), duration_ms=1000, status="transcribing",
                speech=[[100, 800]], words=[Word(100, 800, "テスト。")], completed_chunks=[0])
    def forbidden(*args, **kwargs):
        raise AssertionError("Completed chunks must not reload the model")
    monkeypatch.setattr(faster_whisper, "WhisperModel", forbidden)
    result = run_pipeline(p, tmp_path / "resume.autosub.json", Event(), lambda *_: None)
    assert result.status == "complete"
    assert len(result.words) == len(result.cues) == 1


def test_silence_uses_real_silero_without_hallucinated_captions(tmp_path):
    import wave
    source = tmp_path / "silence.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(bytes(32000))
    p = run_pipeline(Project(media_path=str(source)), tmp_path / "silent.autosub.json", Event(), lambda *_: None)
    assert p.status == "complete"
    assert not p.cues


def test_chunk_context_maps_words_to_original_clock_once(tmp_path, monkeypatch):
    import faster_whisper
    import faster_whisper.vad
    import autosub.pipeline as pipeline
    source = tmp_path / "dummy.wav"
    source.write_bytes(b"test-source")
    monkeypatch.setattr(pipeline, "decode_audio", lambda *_: None)
    monkeypatch.setattr(pipeline, "load_audio", lambda _: np.zeros(61 * 16000, dtype=np.float32))
    monkeypatch.setattr(faster_whisper.vad, "get_speech_timestamps",
                        lambda *args, **kwargs: [{"start": 0, "end": 61 * 16000}])
    batches = [
        [(29, 29.5, " first"), (30.5, 31, " second")],
        [(1, 1.5, " first"), (2.5, 3, " second"), (31.5, 32, " third")],
        [(1.5, 2, " third"), (2.2, 2.8, " fourth")],
    ]
    calls = []
    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass
        def transcribe(self, audio, **kwargs):
            calls.append(kwargs)
            batch = batches[len(calls) - 1]
            words = [SimpleNamespace(start=s, end=e, word=t, probability=0.9) for s, e, t in batch]
            return iter([SimpleNamespace(words=words)]), None
    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)
    p = run_pipeline(Project(media_path=str(source), language="ja", asr_strategy="fixed30-v1"), tmp_path / "chunk.autosub.json",
                     Event(), lambda *_: None)
    assert [(w.start_ms, w.text.strip()) for w in p.words] == [
        (29000, "first"), (30500, "second"), (59500, "third"), (60200, "fourth")]
    assert p.completed_chunks == [0, 1, 2]
    assert all(not call["multilingual"] and call["language"] == "ja" for call in calls)


@pytest.mark.parametrize('resumed',[False,True])
@pytest.mark.parametrize('language',['en','ja'])
def test_quiet_retry_reaches_asr_with_raw_audio_and_external_clips(tmp_path,monkeypatch,resumed,language):
    import faster_whisper
    import faster_whisper.vad
    import autosub.pipeline as pipeline
    source=tmp_path/'quiet.wav';source.write_bytes(b'quiet fixture')
    raw=np.full(16000,.002,dtype=np.float32); before=raw.copy()
    monkeypatch.setattr(pipeline,'decode_audio',lambda *a:None)
    monkeypatch.setattr(pipeline,'load_audio',lambda *a:raw)
    peaks=[]
    def vad(audio,*a,**k):
        assert not resumed
        peaks.append(float(np.max(np.abs(audio))))
        return [] if len(peaks)==1 else [{'start':1600,'end':9600}]
    monkeypatch.setattr(faster_whisper.vad,'get_speech_timestamps',vad)
    class Model:
        frames_per_second=100
        def __init__(self,*a,**k):pass
        def transcribe(self,audio,**kwargs):
            assert np.array_equal(audio,before)
            assert kwargs['vad_filter'] is False
            assert kwargs['clip_timestamps']==([0,1.0] if language=='ja' else [0,.8])
            w=SimpleNamespace(start=.1,end=.6,word=' quiet',probability=.9)
            return iter([SimpleNamespace(words=[w],text=' quiet',seek=0)]),None
    monkeypatch.setattr(faster_whisper,'WhisperModel',Model)
    path=tmp_path/'quiet.json'
    p=Project(media_path=str(source),language=language)
    if resumed:
        p.speech=[[100,600]];p.status='transcribing'
        if language=='ja': p.asr_strategy='native-v1'
        p.review_events=[{'action':'vad_quiet_retry','policy':'quiet-empty-v1','retry_regions':1}]
        p.save(path);p=Project.load(path)
    result=run_pipeline(p,path,Event(),lambda *a:None)
    assert result.words[0].start_ms==100 and result.words[0].end_ms==600
    assert (not peaks if resumed else peaks[1]>peaks[0]) and np.array_equal(raw,before)
    assert any(e['action']=='vad_quiet_retry' for e in Project.load(path).review_events)


def test_completed_empty_project_is_not_silently_reinterpreted(tmp_path,monkeypatch):
    import faster_whisper.vad
    import autosub.pipeline as pipeline
    source=tmp_path/'quiet.wav';source.write_bytes(b'quiet fixture')
    monkeypatch.setattr(pipeline,'decode_audio',lambda *a:None)
    monkeypatch.setattr(pipeline,'load_audio',lambda *a:np.full(16000,.002,dtype=np.float32))
    calls=[]
    def vad(*a,**k): calls.append(1);return []
    monkeypatch.setattr(faster_whisper.vad,'get_speech_timestamps',vad)
    p=Project(media_path=str(source),status='complete')
    result=run_pipeline(p,tmp_path/'old.json',Event(),lambda *a:None)
    assert not result.words and len(calls)==1 and not result.review_events
