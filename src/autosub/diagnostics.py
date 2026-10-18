"""Offline verification for a packaged application; never downloads models."""
from __future__ import annotations

import json
import os
import platform
import traceback
import wave
from pathlib import Path
from threading import Event


def self_test(directory: str, model_path: str | None = None, audio_path: str | None = None,
              playback_project: str | None = None) -> None:
    target = Path(directory).resolve()
    target.mkdir(parents=True, exist_ok=True)
    os.environ["AUTOSUB_SETTINGS_PATH"] = str(target / "preferences.json")
    os.environ["AUTOSUB_ENABLE_REPAIR_UI"] = "1"  # archived functionality regression only
    os.environ.pop("AUTOSUB_ENABLE_PROJECT_TOOLS", None)
    from . import __version__
    report = {"version": __version__, "platform": platform.platform(), "status": "failed"}
    try:
        from .domain import Cue, Project, Word
        from .export import export_subtitles
        from .pipeline import run_pipeline
        from .media import ffmpeg_path
        report["ffmpeg"] = ffmpeg_path()
        # Packaged storage smoke test: synthetic Windows errors only, no real user data.
        from unittest.mock import patch
        from . import file_io
        snapshot_path = target / "storage.autosub.json"
        old_snapshot = Project(glossary="before")
        old_snapshot.save(snapshot_path)
        before_storage = snapshot_path.read_bytes()
        new_snapshot = Project(glossary="after")
        real_replace = file_io._replace
        replace_attempts = []
        def transient_denial(source_path, destination_path):
            replace_attempts.append(1)
            if len(replace_attempts) < 3:
                failure = PermissionError("synthetic Windows denial")
                failure.winerror = 5
                raise failure
            return real_replace(source_path, destination_path)
        with patch.object(file_io, "_replace", transient_denial):
            new_snapshot.save(snapshot_path)
        assert len(replace_attempts) == 3 and Project.load(snapshot_path) == new_snapshot
        assert not snapshot_path.with_suffix(snapshot_path.suffix + ".tmp").exists()
        def persistent_denial(*_):
            failure = PermissionError("synthetic persistent denial")
            failure.winerror = 5
            raise failure
        with patch.object(file_io, "_replace", persistent_denial):
            try:
                old_snapshot.save(snapshot_path)
            except PermissionError:
                pass
            else:
                raise AssertionError("Persistent denial must not report success")
        assert Project.load(snapshot_path) == new_snapshot
        assert snapshot_path.with_suffix(snapshot_path.suffix + ".tmp").read_bytes() == before_storage
        report["checkpoint_storage"] = "passed; synthetic bounded retry and both snapshots preserved"
        source = target / "silence.wav"
        with wave.open(str(source), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(bytes(32000))
        p = run_pipeline(Project(media_path=str(source)), target / "silence.autosub.json", Event(), lambda *_: None)
        assert p.status == "complete" and not p.cues
        report["silero_silence"] = "passed"
        p.cues = [Cue("test", 0, 800, "字幕測試 · テスト · Test · 테스트", 0, 600)]
        for suffix in [".ass", ".srt"]:
            export_subtitles(p, target / ("export" + suffix))
        report["export"] = "passed"
        from .native_segmentation import segment_native
        first = [Word(100, 400, "よかったよ"), Word(400, 500, "な")]
        second = [Word(500, 1000, "いや"), Word(1000, 2000, "今日は休み")]
        guard = Project(duration_ms=3000, words=first+second, speech=[[100, 2000]])
        assert [c.text for c in segment_native(guard, [first, second])] == ["よかったよな", "いや今日は休み"]
        report["native_partial_token_boundary_guard"] = "passed"
        import numpy as np
        from .speech_detection import detect_speech, external_vad_clips
        quiet = np.full(16000, .002, dtype=np.float32)
        before = quiet.copy()
        attempts = []
        def probe_vad(samples, *args, **kwargs):
            attempts.append(float(np.max(np.abs(samples))))
            return [] if len(attempts) == 1 else [{"start": 1600, "end": 9600}]
        spans, analysis = detect_speech(quiet, None, probe_vad)
        assert spans and analysis["analysis_gain_db"] <= 24.000001
        assert np.array_equal(quiet, before) and attempts[1] > attempts[0]
        assert external_vad_clips([[100,600]], 0, 1000) == [0, .8]
        report["quiet_vad_retry_rule"] = "passed; synthetic detector, original ASR samples preserved"
        if model_path and audio_path:
            recognized = run_pipeline(Project(media_path=str(Path(audio_path).resolve()),
                                               model=str(Path(model_path).resolve()), language="en"),
                                      target / "recognized.autosub.json", Event(), lambda *_: None)
            assert recognized.words and recognized.cues
            export_subtitles(recognized, target / "recognized.ass")
            report["whisper_local_model"] = {"words": len(recognized.words), "cues": len(recognized.cues)}
            traced = Project.load(target / "recognized.autosub.json")
            contexts = [e for e in traced.review_events if e.get("action") == "asr_trace_context"]
            checkpoints = [e for e in traced.review_events if e.get("action") == "asr_trace_checkpoint"]
            assert contexts and checkpoints
            assert all(e["app_version"] == __version__ and e["resolved_options"]
                       and e["packages"]["faster-whisper"] for e in contexts)
            temperatures = [s["temperature"] for e in checkpoints for s in e["segments"]]
            assert temperatures and all(t is not None for t in temperatures)
            assert sum(e["accepted_word_count"] for e in checkpoints) == len(traced.words)
            report["recognition_trace"] = {"status": "passed; persisted actual emitted-segment evidence",
                                           "temperatures": temperatures,
                                           "rejected_fallback_attempts": "unobserved"}
            from .alignment import LocalWhisperAligner, analyze_cue, apply_candidate, restore_candidate
            backend = LocalWhisperAligner(recognized.model, recognized.device)
            candidate = analyze_cue(recognized, 0, target / "alignment-cache", Event(), backend=backend)
            assert candidate["status"] in {"aligned", "partial"}, candidate
            original = (recognized.cues[0].start_ms, recognized.cues[0].end_ms, recognized.cues[0].text)
            recognized.cues[0].candidate = candidate
            recognized.save(target / "alignment.autosub.json")
            apply_candidate(recognized, 0)
            recognized.validate()
            restore_candidate(recognized, 0)
            assert (recognized.cues[0].start_ms, recognized.cues[0].end_ms, recognized.cues[0].text) == original
            report["local_alignment"] = {"status": candidate["status"], "apply_restore": "passed"}
            from faster_whisper.tokenizer import Tokenizer
            # Tokenizer/API compatibility only; English audio is not four-language acoustic validation.
            report["tokenizer_languages"] = {}
            for code, text in [("ja", "テスト。"), ("zh", "測試。"), ("en", "Test."), ("ko", "테스트.")]:
                tokenizer = Tokenizer(backend.model.hf_tokenizer, backend.model.model.is_multilingual,
                                      task="transcribe", language=code)
                assert tokenizer.decode(tokenizer.encode(text)) == text
                report["tokenizer_languages"][code] = "passed; acoustic accuracy not tested"
        from .alignment import analyze_cue
        class MissingBackend:
            identity = {"backend": "unavailable-test"}
            def align(self, *args):
                raise ImportError("backend unavailable")
        fallback = analyze_cue(p, 0, target / "fallback-cache", Event(), backend=MissingBackend())
        assert fallback["status"] == "fallback"
        report["alignment_failure_fallback"] = "passed"
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from .app import MainWindow
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        window.project = p
        window.project_path = target / "ui.autosub.json"
        window.refresh()
        from .editing import split_cue
        window.project.cues = split_cue(window.project, 0, 300, "字幕測試", "テスト · Test · 테스트")
        window.refresh()
        window.table.selectRow(0)
        window.merge_selected()
        assert len(window.project.cues) == 1 and window.project.cues[0].locked
        window.wave.duration_ms = p.duration_ms
        window.wave.focus_region()
        window.wave.zoom(2)
        assert 0 <= window.wave.bounds()[0] < window.wave.bounds()[1] <= p.duration_ms
        report["manual_review"] = "passed"
        window.confirm_and_next()
        assert window.project.cues[0].reviewed
        assert "已複核 1／1" in window.selection_label.text()
        report["basic_review_confirmation"] = "passed; synthetic cue only"
        from .review_ui import ComparisonDialog
        from .app import Waveform
        from .comparison import export_report, summary
        reference = target / "synthetic-reference.srt"
        reference.write_text("1\n00:00:00,000 --> 00:00:00,800\n介面自測參考（非人工品質標準）\n", encoding="utf-8")
        comparison = ComparisonDialog(window.project, target / "comparison.autosub.json", Waveform, window)
        comparison.error = lambda message: report.update(comparison_error=str(message))
        comparison.reference.setText(str(reference))
        comparison.start_compare()
        import time
        deadline = time.monotonic() + 15
        while comparison.thread is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert comparison.thread is None and comparison.session is not None, report
        assert summary(comparison.session)["human_confirmed_usable_rate"] is None
        export_report(comparison.session, target / "comparison.html")
        comparison.show()
        app.processEvents()
        assert comparison.grab().save(str(target / "packaged-comparison-ui.png"))
        comparison.close()
        report["finished_reference_review"] = "passed; synthetic UI test, no human quality claim"
        from .project_tools import relink_media
        moved = target / "moved-source.wav"
        import shutil
        shutil.copy2(source, moved)
        relinked = relink_media(p, target / "relink.autosub.json", moved, Event())
        assert relinked.media_path == str(moved.resolve())
        assert relinked.cues == p.cues
        report["verified_media_relink"] = "passed"
        # Exercise the real modal preview, durable store and existing Undo path.
        # Synthetic ownership/VAD only; not a recognition or semantic benchmark.
        from .domain import Word, word_key
        from .media import fingerprint
        from PySide6.QtCore import QTimer
        words = [Word(100,400,"声も入る"),Word(400,800,"こちら")]
        native = Project(media_path=str(source),duration_ms=1000,status="complete",words=words,
            speech=[[100,800]],asr_strategy="native-v1",native_groups=[[0],[1]],
            native_complete=True,native_resume_ms=1000,source_fingerprint=fingerprint(source),
            cues=[Cue("n1",80,400,words[0].text,100,400,source_tokens=[word_key(words[0])]),
                  Cue("n2",400,1000,words[1].text,400,800,source_tokens=[word_key(words[1])])])
        window.autosave.stop()
        window.project = native
        window.project_path = target / "merge-roundtrip.autosub.json"
        window.load_controls()
        window.refresh()
        native.save(window.project_path)
        import copy
        before_cues = copy.deepcopy(native.cues)
        window.table.selectRow(0)
        def accept_merge():
            dialog = app.activeModalWidget()
            assert dialog is not None and dialog.windowTitle() == "短句合併預覽"
            assert dialog.grab().save(str(target / "packaged-merge-preview.png"))
            dialog.accept()
        QTimer.singleShot(0,accept_merge)
        window.merge_selected()
        assert len(window.project.cues) == 1
        window.project = Project.load(window.project_path)
        window.reset_history()
        window.load_controls()
        window.refresh()
        window.undo()
        assert window.project.cues == before_cues
        assert Project.load(window.project_path).cues == before_cues
        report["sentence_merge_preview_restore"] = "passed; synthetic GUI, not human adoption"
        window.project = p
        window.project_path = target / "ui.autosub.json"
        window.reset_history()
        window.dirty = False
        window.load_controls()
        window.refresh()
        if playback_project or (model_path and audio_path):
            # Decode/seek-only checks miss silent output routes. Exercise the actual
            # PCM sink and its clock, using an explicitly supplied project if present.
            audition = Project.load(Path(playback_project)) if playback_project else recognized
            window.autosave.stop()
            window.project_path = None
            window.dirty = False
            window.project = audition
            window.load_controls()
            window.refresh()
            window.load_playback()
            window.volume.setValue(25)
            window.table.selectRow(0)
            window.play_cue()
            assert (window.audio.start_ms, window.audio.end_ms) == (
                max(0, audition.cues[0].start_ms), min(audition.duration_ms, audition.cues[0].end_ms))
            report["selected_cue_bounds"] = "passed; exact display range without surrounding padding"
            start = window.audio.start_ms
            deadline = time.monotonic() + 8
            while window.audio.position() < start + 250 and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(.01)
            assert window.audio.sink is not None, window.status.text()
            assert window.audio.sink.error().value == 0, window.status.text()
            assert window.audio.position() >= start + 250, window.status.text()
            window.audio.pause()
            paused = window.audio.position()
            sink = window.audio.sink
            until = time.monotonic() + .2
            while time.monotonic() < until:
                app.processEvents()
                time.sleep(.01)
            assert window.audio.position() == paused
            assert window.audio.resume() and window.audio.sink is sink
            deadline = time.monotonic() + 5
            while window.audio.position() < paused + 150 and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(.01)
            assert window.audio.position() >= paused + 150, window.status.text()
            report["pause_resume"] = {"paused_ms": paused, "resumed_ms": window.audio.position(),
                "same_output_sink": window.audio.sink is sink, "status": "passed"}
            from .playback import pcm_for_output
            fmt = window.audio_output.device().preferredFormat()
            _, peak = pcm_for_output(audition.audio_path, start, window.audio.end_ms,
                                     fmt.sampleRate(), fmt.channelCount(), fmt.sampleFormat())
            assert peak > 0, "test excerpt must contain a non-silent signal"
            report["pcm_audition"] = {"status": "passed", "output": window.audio_output.device().description(),
                "clock_progress_ms": window.audio.position() - start, "decoded_peak": peak,
                "sink_error": str(window.audio.sink.error()), "note": "device/clock test; physical audibility requires listener"}
            window.audio.stop()
            for row in range(min(5, len(audition.cues))):
                window.table.selectRow(row)
                window.play_cue()
                assert (window.audio.start_ms, window.audio.end_ms) == (
                    max(0, audition.cues[row].start_ms), min(audition.duration_ms, audition.cues[row].end_ms))
                app.processEvents()
                assert window.audio.sink is not None and window.audio.sink.error().value == 0
                window.audio.pause()
            window.audio.stop()
            report["repeated_selection_audition"] = "passed; up to five ranges"
            export_subtitles(audition, target / "user-single-line.ass")
            report["ass_default_wrap"] = "disabled; intentional manual line breaks preserved"
        window.show()
        app.processEvents()
        assert window.grab().save(str(target / "packaged-ui.png"))
        window.close()
        restored = MainWindow()
        assert restored.volume.value() == window.volume.value()
        assert restored.wrap_export.isChecked() == window.wrap_export.isChecked()
        restored.close()
        report["user_preferences_round_trip"] = "passed; isolated self-test settings file"
        os.environ.pop("AUTOSUB_ENABLE_REPAIR_UI",None)
        public = MainWindow()
        assert public.tools_button.isHidden()
        assert __version__ in public.windowTitle()
        from .vocabulary_ui import VocabularyDialog
        from .vocabulary import load_library
        vocabulary = VocabularyDialog(public)
        vocabulary.name.setText("Example terms")
        vocabulary.new_word.setText("Star Token")
        vocabulary.add_word()
        vocabulary.save()
        assert load_library()["lists"][-1]["text"] == "Star Token"
        public.refresh_vocabulary()
        public.vocabulary_combo.setCurrentIndex(public.vocabulary_combo.findData(vocabulary.saved_id))
        assert public.glossary.toPlainText() == "Star Token"
        cancelled_vocabulary = VocabularyDialog(public)
        cancelled_vocabulary.name.setText("Not saved")
        cancelled_vocabulary.reject()
        assert len(load_library()["lists"]) == 1
        report["vocabulary_library"] = "passed; synthetic save/select/cancel, isolated user storage"
        public.project = audition if playback_project or (model_path and audio_path) else p
        public.project_path = None
        public.autosave.stop()
        public.load_controls()
        public.refresh()
        public.advanced.setChecked(True)
        assert not public.repair_enabled and public.edit_panel.isHidden() and public.candidate_panel.isHidden()
        assert public.only_flags.isHidden() and public.table.isColumnHidden(3)
        assert public.table.editTriggers() == public.table.EditTrigger.NoEditTriggers
        from .appearance import NumberBox, DecimalBox
        from PySide6.QtWidgets import QStyle, QStyleOptionSpinBox
        public.show()
        app.processEvents()
        boxes = public.findChildren(NumberBox) + public.findChildren(DecimalBox)
        assert boxes
        for box in boxes:
            # Grabbing a hidden widget can finish layout and change its geometry.
            rendered = box.grab().toImage()
            option = QStyleOptionSpinBox()
            box.initStyleOption(option)
            for control in (QStyle.SubControl.SC_SpinBoxUp, QStyle.SubControl.SC_SpinBoxDown):
                rect = box.style().subControlRect(QStyle.ComplexControl.CC_SpinBox, option, control, box)
                assert rect.width() >= 20
                center = rect.center()
                ratio = rendered.devicePixelRatio()
                assert rendered.pixelColor(round(center.x() * ratio), round(center.y() * ratio)).name() in (
                    "#52657e", "#a6b0bd")
        assert public.timing_group.grab().save(str(target / "numeric-controls.png"))
        report["numeric_controls"] = "passed; visible step arrows in actual integer and decimal fields"
        public.advanced.setChecked(False)
        public.show()
        app.processEvents()
        assert public.grab().save(str(target / "public-ui.png"))
        public.dirty = False
        public.close()
        report["export_first_ui"] = "passed; read-only preview, repair tools hidden by default"
        from .runtime import check_runtime
        report["cpu_runtime"] = check_runtime("cpu")
        report["qt_ui"] = "passed"
        report["status"] = "passed"
    except Exception:
        report["traceback"] = traceback.format_exc()
    (target / "self-test.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if report["status"] != "passed":
        raise SystemExit(1)
