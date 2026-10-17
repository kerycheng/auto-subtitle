"""Compare old/new production pipelines using identical recorded decoder output.

No ASR or reference text is supplied. Research records must have exactly one
decoder call; multi-call/older records are explicitly skipped, never flattened.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
from threading import Event
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import faster_whisper
import autosub.native_pipeline as native
import autosub.pipeline as pipeline
from autosub.domain import Project
from autosub.media import load_audio


def baseline_module(ref, file):
    source = subprocess.check_output(['git', 'show', f'{ref}:src/autosub/{file}.py'])
    module = ModuleType('autosub._baseline_' + file)
    module.__package__ = 'autosub'
    exec(compile(source, f'{ref}/{file}.py', 'exec'), module.__dict__)
    return module, hashlib.sha256(source).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-ref', required=True)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output directory')
    args.output.mkdir(parents=True)
    previous, pipeline_hash = baseline_module(args.baseline_ref, 'pipeline')
    old_native, native_hash = baseline_module(args.baseline_ref, 'native_pipeline')
    report = {'baseline_ref': args.baseline_ref, 'baseline_pipeline_sha256': pipeline_hash,
              'baseline_native_sha256': native_hash, 'scope': 'same decoder replay, no new ASR or accuracy evaluation',
              'records_sha256': hashlib.sha256(args.records.read_bytes()).hexdigest(),
              'cases': [], 'skipped': []}
    cases = json.loads(args.records.read_text(encoding='utf-8'))['cases']
    for case in cases:
        if len(case.get('decoder_calls') or []) != 1 or case.get('decoder_segments') is None:
            report['skipped'].append(case['id'])
            continue
        original_path = args.records.parent / case['id'] / 'generated.autosub.json'
        original_hash = hashlib.sha256(original_path.read_bytes()).hexdigest()
        original = Project.load(original_path)
        audio = load_audio(Path(original.audio_path))
        call = case['decoder_calls'][0]
        segments = [SimpleNamespace(**{**s, 'words': [SimpleNamespace(**w) for w in s['words']]})
                    for s in case['decoder_segments']]
        calls = []
        class ReplayModel:
            frames_per_second = 100
            def __init__(self, *a, **k): pass
            def transcribe(self, samples, **options):
                assert hashlib.sha256(samples.tobytes()).hexdigest() == call['asr_samples_sha256']
                assert options == call['options']
                calls.append(options)
                return iter(segments), None
        runtime = lambda _: {'compute_type': 'int8'}
        results = []
        with patch.object(faster_whisper, 'WhisperModel', ReplayModel), \
             patch.object(previous, 'decode_audio', lambda *a: None), \
             patch.object(pipeline, 'decode_audio', lambda *a: None), \
             patch.object(previous, 'load_audio', lambda *a: audio), \
             patch.object(pipeline, 'load_audio', lambda *a: audio), \
             patch.object(previous, 'check_runtime', runtime), \
             patch.object(pipeline, 'check_runtime', runtime), \
             patch.object(old_native, 'check_runtime', runtime), \
             patch.object(native, 'check_runtime', runtime):
            for name, worker, native_worker in [('before', previous.run_pipeline, old_native.transcribe_native),
                                                 ('after', pipeline.run_pipeline, native.transcribe_native)]:
                project = Project(media_path=original.media_path, language=original.language,
                                  model=original.model, device='cpu', speech=original.speech,
                                  asr_strategy=original.asr_strategy, status='vad_complete')
                with patch.object(native, 'transcribe_native', native_worker):
                    worker(project, args.output / case['id'] / (name + '.json'), Event(), lambda *a: None)
                results.append(project)
        before, after = map(asdict, results)
        trace = after.pop('review_events')
        assert before.pop('review_events') == []
        before.pop('audio_path'); after.pop('audio_path')
        assert before == after
        assert results[0].words == original.words
        assert len(calls) == 2
        contexts = [e for e in trace if e['action'] == 'asr_trace_context']
        checkpoints = [e for e in trace if e['action'] == 'asr_trace_checkpoint']
        assert len(contexts) == 1
        assert contexts[0]['input']['samples_sha256'] == call['asr_samples_sha256']
        observed = [s['temperature'] for e in checkpoints for s in e['segments']]
        assert observed == [s['temperature'] for s in case['decoder_segments']]
        assert hashlib.sha256(original_path.read_bytes()).hexdigest() == original_hash
        report['cases'].append({'id': case['id'], 'words': len(original.words), 'calls_unchanged': True,
                                'all_project_fields_except_cache_path_and_trace_unchanged': True,
                                'observed_temperatures': observed})
    assert {row['id'][:2] for row in report['cases']} == {'en', 'zh', 'ja', 'ko'}
    report['status'] = 'passed'
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Passed {len(report['cases'])} four-language recorded decoder replays; skipped {len(report['skipped'])} unavailable/multi-call records")


if __name__ == '__main__':
    main()
