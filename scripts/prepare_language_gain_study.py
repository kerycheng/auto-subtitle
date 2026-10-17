"""Download only frozen language-gated rows and freeze complete-source controls before ASR."""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from threading import Event
import urllib.parse
import wave

import numpy as np
from faster_whisper.vad import VadOptions, get_speech_timestamps

from fetch_language_samples import fetch
from autosub.alignment_lab import file_hash
from autosub.asr_comparison import model_identity
from autosub.compensation_evaluation import _generated_controls
from autosub.level_compensation_lab import compensate, union_regions
from autosub.media import decode_audio, load_audio


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def samples_hash(audio):
    return hashlib.sha256(audio.tobytes()).hexdigest()


def wav(path, samples):
    with wave.open(str(path), 'wb') as f:
        f.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        f.writeframes(np.rint(samples * 32768).astype('<i2').tobytes())


def fetch_sources(plan_path, output):
    plan = json.loads(plan_path.read_text(encoding='utf-8'))['dataset']
    path = output / 'manifest.json'
    manifest = {'plan_sha256': file_hash(plan_path), 'status': 'running',
                'dataset': plan['name'], 'split': plan['split'], 'license': plan['license'],
                'selection_role': 'independent fixed evaluation, not for retuning', 'samples': []}
    if path.exists():
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if manifest['plan_sha256'] != file_hash(plan_path):
            raise ValueError('Frozen plan changed')
    output.mkdir(parents=True, exist_ok=True)
    expected = {f'{lang}-{p}' for lang in plan['languages'] for p in plan['zero_based_positions']}
    done = set()
    for row in manifest['samples']:
        if row['id'] in done or row['id'] not in expected or file_hash(row['audio_path']) != row['audio_sha256']:
            raise ValueError('Cached row or source changed')
        done.add(row['id'])
    for lang in plan['languages']:
        for position in plan['zero_based_positions']:
            name = f'{lang}-{position}'
            if name in done:
                continue
            query = urllib.parse.urlencode({'dataset': plan['name'], 'config': plan['configs'][lang],
                                           'split': plan['split'], 'offset': position, 'length': 1})
            endpoint = 'https://datasets-server.huggingface.co/rows?' + query
            metadata_path = output / (name + '-metadata.json')
            metadata = metadata_path.read_bytes() if metadata_path.exists() else fetch(endpoint, 4*1024*1024)
            metadata_path.write_bytes(metadata)
            item = json.loads(metadata)['rows'][0]
            if item['row_idx'] != position:
                raise ValueError('Dataset row position changed')
            row = item['row']; audio = row['audio'][0]
            data = fetch(audio['src'], 20*1024*1024)
            source = output / (name + '.audio'); source.write_bytes(data)
            manifest['samples'].append({'id': name, 'language': lang, 'position': position,
                'config': plan['configs'][lang], 'dataset_id': row['id'], 'audio_path': str(source.resolve()),
                'audio_sha256': file_hash(source), 'reference_text': row['raw_transcription'],
                'metadata_endpoint': endpoint, 'metadata_sha256': file_hash(metadata_path)})
            save(path, manifest)
            print('downloaded', name, flush=True)
    manifest['status'] = 'complete'; save(path, manifest)
    (output/'ATTRIBUTION.txt').write_text('Google FLEURS · CC-BY-4.0\nhttps://huggingface.co/datasets/google/fleurs\n'
        'Fixed validation rows 12–15 in en_us, cmn_hans_cn, ko_kr. No reference enters ASR.\n', encoding='utf-8')


def prepare(plan_path, manifest_path, model_path, output):
    if output.exists():
        raise ValueError('Use a new study directory; never overwrite frozen inputs')
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['status'] != 'complete' or manifest['plan_sha256'] != file_hash(plan_path):
        raise ValueError('Incomplete or wrong sources')
    expected = {f'{lang}-{p}' for lang in plan['dataset']['languages'] for p in plan['dataset']['zero_based_positions']}
    sources = {s['id']: s for s in manifest['samples']}
    if len(sources) != len(manifest['samples']) or set(sources) != expected:
        raise ValueError('Source selection changed')
    output.mkdir(parents=True); pcm_dir = output/'source-pcm'; pcm_dir.mkdir()
    pcm = {}; source_labels = {}
    for name, source in sources.items():
        if (file_hash(source['audio_path']) != source['audio_sha256']
                or source['id'] != f"{source['language']}-{source['position']}"
                or source['config'] != plan['dataset']['configs'][source['language']]):
            raise ValueError('Original source changed')
        path = pcm_dir/(name+'.wav')
        decode_audio(Path(source['audio_path']), path, 0, Event())
        pcm[name] = load_audio(path)
        source_labels[name] = {'sample_id': name, 'source_path': str(path.resolve()),
            'source_sha256': file_hash(path), 'reference_text': source['reference_text'], 'attenuation_db': 0}
    cases = []; labels = []; analyses = []
    options = VadOptions(threshold=.5, min_speech_duration_ms=100, min_silence_duration_ms=100, speech_pad_ms=0)

    def add(name, lang, kind, audio, regions, extra=None):
        before = audio.copy(); processed, gain = compensate(audio)
        if not np.array_equal(audio, before) or len(processed) != len(audio) or np.max(np.abs(processed)) > 1:
            raise ValueError('Gain violated source or clipping constraints')
        path = output/(name+'.wav'); wav(path, audio)
        if not np.array_equal(load_audio(path), audio):
            raise ValueError('PCM conversion changed frozen samples')
        row = {'id': name, 'language': lang, 'kind': kind, 'audio_path': str(path.resolve()),
               'audio_sha256': file_hash(path), 'duration_ms': round(len(audio)/16)}
        cases.append(row); labels.append({**row, 'regions': regions, **(extra or {})})
        raw_vad = get_speech_timestamps(audio, options, sampling_rate=16000)
        boosted = get_speech_timestamps(processed, options, sampling_rate=16000)
        np.save(output/(name+'.analysis.npy'), processed)
        analyses.append({'id': name, 'original_samples_sha256': samples_hash(audio),
            'processed_samples_sha256': samples_hash(processed), 'gain_trace': gain,
            'raw_vad': raw_vad, 'boosted_vad': boosted, 'union_vad': union_regions(raw_vad, boosted)})

    controls = _generated_controls()
    for lang in plan['dataset']['languages']:
        for position in plan['dataset']['zero_based_positions']:
            name = f'{lang}-{position}'; audio = pcm[name]
            add(name+'-standalone', lang, 'standalone', audio,
                [{**source_labels[name], 'sample_frames': [0, len(audio)]}])
        for left, right in plan['dataset']['pairs']:
            first, second = f'{lang}-{left}', f'{lang}-{right}'
            for db in plan['cases']['full_source_pairs']['second_source_db']:
                for gap in plan['cases']['full_source_pairs']['gap_ms']:
                    a, b = pcm[first], pcm[second]
                    low = np.rint(b*32768*10**(db/20)).astype('<i2').astype(np.float32)/32768
                    start = len(a)+gap*16; joined = np.concatenate((a, np.zeros(gap*16,dtype=np.float32), low))
                    add(f'{lang}-{left}-{right}-db{abs(db)}-gap{gap}', lang, 'speech-pair', joined,
                        [{**source_labels[first], 'sample_frames': [0,len(a)]},
                         {**source_labels[second], 'attenuation_db': db, 'sample_frames': [start,len(joined)]}])
        name = f"{lang}-{plan['cases']['known_generated_noise_tails']['source_position']}"; a = pcm[name]
        for mode in plan['cases']['known_generated_noise_tails']['modes']:
            generator = 'white-noise' if mode == 'white' else mode
            add(name+'-noise-'+mode, lang, 'non-speech-tail', np.concatenate((a,controls[generator])),
                [{**source_labels[name], 'sample_frames': [0,len(a)]}],
                {'generator': generator, 'non_speech_frames': [len(a),len(a)+160000]})
    if len(cases) != plan['cases']['total_cases']:
        raise ValueError('Planned case count changed')
    identity = model_identity(model_path)
    protocol = {'plan_sha256': file_hash(plan_path), 'source_dataset_path': str(manifest_path.resolve()),
                'dataset_sha256': file_hash(manifest_path), 'model_identity': identity,
                'model_path': str(model_path.resolve()), 'scope': 'language-gated frozen independent public controls'}
    save(output/'cases.json', {'text_provided': False, 'protocol': protocol, 'cases': cases})
    save(output/'labels.json', {'protocol': protocol, 'cases': labels})
    save(output/'analyses.json', {'protocol': protocol, 'cases': analyses})
    save(output/'pre-asr-freeze.json', {**protocol, 'cases_sha256': file_hash(output/'cases.json'),
         'labels_sha256': file_hash(output/'labels.json'), 'analyses_sha256': file_hash(output/'analyses.json'),
         'implementation_sha256': {str(p.resolve()): file_hash(p) for p in [
             Path(__file__), *[Path(__file__).resolve().parents[1]/'src'/'autosub'/(name+'.py') for name in
                ('pipeline','level_compensation_lab','speech_detection','recognition_trace','language_gain_study')]]},
         'packages': {name:version(name) for name in ('faster-whisper','ctranslate2','numpy','onnxruntime')},
         'engineering_notes': ['Frozen white generator label maps to unchanged fixed white-noise generator, seed 2300.',
                               'Pairs are not cropped to 30s; keep existing chunk ownership and source-clock contexts.'],
         'asr_started': False})
    print('prepared and froze', len(cases), 'cases', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=['fetch','prepare'])
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--model', type=Path)
    args = parser.parse_args()
    if args.action == 'fetch': fetch_sources(args.plan,args.output)
    else: prepare(args.plan,args.manifest,args.model,args.output)
