"""Isolated native decoding ablation; never reads reference text or changes app defaults."""
import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path
from threading import Event

from autosub.alignment_lab import file_hash
from autosub.asr_comparison import model_identity
from autosub.domain import Project
from autosub.export import export_subtitles
from autosub.media import load_audio
from autosub.model_cache import clear_model
from autosub.native_pipeline import transcribe_native
from autosub.timing import apply_timing


def prepare_probe(source):
    source.validate()
    if not source.native_complete or source.language != 'ja' or source.glossary:
        raise ValueError('Expected completed Japanese native project without vocabulary prompt')
    result = copy.deepcopy(source)
    result.words, result.cues, result.native_groups = [], [], []
    result.native_resume_ms, result.native_complete = 0, False
    result.review_events, result.completed_chunks = [], []
    result.status = 'vad_complete'
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--mode', choices=['default', 'zero'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output exists')
    source = Project.load(args.source)
    project = prepare_probe(source)
    audio = load_audio(project.audio_path)
    if round(len(audio)/16) != project.duration_ms:
        raise ValueError('Cached audio duration differs')
    project_hash, audio_hash = file_hash(args.source), file_hash(project.audio_path)
    from faster_whisper import WhisperModel
    records, options = [], {}

    class ProbeModel:
        def __init__(self, *a, **kw):
            self.model = WhisperModel(*a, **kw)
            self.frames_per_second = self.model.frames_per_second

        def transcribe(self, audio, **kw):
            if args.mode == 'zero':
                kw['temperature'] = 0.0
            options.update(kw)
            segments, info = self.model.transcribe(audio, **kw)

            def record():
                for segment in segments:
                    records.append(asdict(segment))
                    yield segment
            return record(), info

    args.output.mkdir(parents=True)
    report = {'status':'running', 'mode':args.mode, 'source_project_sha256':project_hash,
        'source_audio_sha256':audio_hash, 'model_identity':model_identity(Path(project.model)),
        'reference_text_provided':False, 'decoding_options':options, 'segments':records,
        'scope':'same full audio, cached VAD, native-v1; isolated decoding ablation, not app default'}
    path = args.output/'native.autosub.json'
    clear_model()
    try:
        transcribe_native(project,audio,path,Event(),lambda n,m:print(n,m,flush=True),ProbeModel)
        project.cues = apply_timing(project)
        project.status = 'complete'
        project.save(path)
        for suffix in ('.ass','.srt'):
            export_subtitles(project,args.output/('native'+suffix))
        if file_hash(args.source) != project_hash or file_hash(project.audio_path) != audio_hash:
            raise ValueError('Source changed during experiment')
        report['status'] = 'complete'
        report['output_project_sha256'] = file_hash(path)
    except Exception as exc:
        report['status'], report['error'] = 'failed', str(exc)
        raise
    finally:
        (args.output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        clear_model()


if __name__ == '__main__':
    main()
