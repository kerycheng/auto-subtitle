"""Run frozen, reference-free vocabulary prompt diagnostics on local audio."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from autosub.alignment_lab import file_hash
from autosub.asr_comparison import model_identity
from autosub.media import load_audio
from autosub.prompt_observation import observe_prompts
from autosub.runtime import check_runtime


def validate(plan):
    if set(plan) != {'text_provided', 'vocabulary', 'model_path', 'model_identity', 'implementation_sha256', 'jobs'}:
        raise ValueError('Unexpected plan fields')
    if plan['text_provided'] is not False or not isinstance(plan['vocabulary'], str) or not plan['vocabulary'].strip():
        raise ValueError('Reference-free vocabulary diagnostics required')
    ids = set()
    for job in plan['jobs']:
        if set(job) != {'id', 'audio_path', 'audio_sha256', 'input_frames', 'asr_samples_sha256', 'options'} or job['id'] in ids:
            raise ValueError('Unexpected, duplicated or text-bearing job')
        ids.add(job['id'])
        a, b = job['input_frames']
        if any(not isinstance(x, int) or isinstance(x, bool) for x in [a, b]) or not 0 <= a < b or b-a > 960000:
            raise ValueError('Invalid source frame clock')
        options = job['options']
        expected = {'language', 'multilingual', 'word_timestamps', 'vad_filter', 'clip_timestamps',
                    'initial_prompt', 'hotwords', 'condition_on_previous_text', 'beam_size'}
        if set(options) not in (expected, expected | {'temperature'}):
            raise ValueError('Unexpected decoder settings')
        if (options['condition_on_previous_text'] is not False or options['vad_filter'] is not False
                or options['language'] not in {'en', 'zh', 'ja', 'ko'}):
            raise ValueError('Expected explicit language and reset decoder context')
        hints = [options['initial_prompt'], options['hotwords']]
        if hints.count(None) != 1 or next(h for h in hints if h is not None) != plan['vocabulary']:
            raise ValueError('Exactly one fixed vocabulary channel is required')
        clips = options['clip_timestamps']
        if (not clips or len(clips) % 2 or any(not 0 <= c < d <= (b-a)/16000
                for c, d in zip(clips[::2], clips[1::2]))
                or any(c < d for c, d in zip(clips[2::2], clips[1::2]))):
            raise ValueError('Invalid relative clip clocks')
    if not ids:
        raise ValueError('No jobs')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf-8')); validate(plan)
    if args.output.exists():
        raise ValueError('Choose a fresh directory; no automatic result selection or retries')
    for path, signature in plan['implementation_sha256'].items():
        if file_hash(path) != signature:
            raise ValueError('Frozen worker changed')
    if model_identity(plan['model_path']) != plan['model_identity']:
        raise ValueError('Model changed')
    for job in plan['jobs']:
        if file_hash(job['audio_path']) != job['audio_sha256']:
            raise ValueError('Source audio changed')
    args.output.mkdir(parents=True)
    path = args.output / 'result.json'
    result = {'status': 'running', 'plan_sha256': file_hash(args.plan), 'model_identity': plan['model_identity'],
              'reference_text_provided': False, 'results': []}
    def save():
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8'); temp.replace(path)
    save()
    try:
        result['runtime'] = check_runtime('cuda')
        from faster_whisper import WhisperModel
        model = WhisperModel(plan['model_path'], device='cuda', compute_type='float16', local_files_only=True)
        for job in plan['jobs']:
            audio = load_audio(Path(job['audio_path'])); a, b = job['input_frames']; samples = audio[a:b]
            if b > len(audio) or hashlib.sha256(samples.tobytes()).hexdigest() != job['asr_samples_sha256']:
                raise ValueError('Input samples changed')
            row = {'id': job['id'], 'status': 'running', 'options': job['options'], 'samples': len(samples),
                   'asr_samples_sha256': job['asr_samples_sha256'], 'segments': [], 'prompt_observations': []}
            result['results'].append(row); save(); started = time.perf_counter()
            try:
                with observe_prompts(model, plan['vocabulary']) as observations:
                    row['prompt_observations'] = observations
                    segments, info = model.transcribe(samples, **job['options'])
                    for segment in segments:
                        row['segments'].append(asdict(segment))
                    row['resolved_options'] = asdict(info.transcription_options)
                row.update(status='complete', text=''.join(s['text'] for s in row['segments']))
            except Exception as exc:
                row.update(status='failed', error_type=type(exc).__name__, error=str(exc))
            row['elapsed_seconds'] = time.perf_counter() - started
            save(); print(row['id'], row['status'], len(row['segments']), flush=True)
        result['status'] = 'complete' if all(r['status']=='complete' for r in result['results']) else 'complete-with-failures'
        save()
    except BaseException as exc:
        result.update(status='interrupted-or-failed', error_type=type(exc).__name__, error=str(exc)); save(); raise


if __name__ == '__main__':
    main()
