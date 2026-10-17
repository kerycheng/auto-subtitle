"""Fetch a frozen development selection; reserved positions remain untouched."""
import argparse
import hashlib
import json
import os
import time
import urllib.parse
from pathlib import Path

from fetch_language_samples import fetch
from autosub.alignment_lab import file_hash
from autosub.asr_experiments import validate_selection_plan


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    validate_selection_plan(plan)
    manifest_path = args.output / 'manifest.json'
    if args.output.exists() and not args.resume:
        raise ValueError('Output exists')
    manifest = {'dataset': 'https://huggingface.co/datasets/' + plan['dataset'], 'license': 'CC-BY-4.0',
        'split': plan['split'], 'selection_role': 'development', 'plan_sha256': file_hash(args.plan),
        'purpose': 'fixed read-speech development samples; not livestream or held-out performance',
        'status': 'running', 'samples': []}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('plan_sha256') != file_hash(args.plan):
            raise ValueError('Selection plan changed')
    args.output.mkdir(parents=True, exist_ok=True)
    def save():
        temporary = manifest_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temporary, manifest_path)
    done = set()
    for row in manifest['samples']:
        if (row['id'] in done or row['position'] not in plan['development_positions']
                or row['language'] not in plan['configs'] or row['id'] != f"{row['language']}-{row['position']}"
                or file_hash(row['audio_path']) != row['audio_sha256']):
            raise ValueError('Cached selection or audio changed')
        done.add(row['id'])
    save()
    for language, config in plan['configs'].items():
        for position in plan['development_positions']:
            name = f'{language}-{position}'
            if name in done:
                continue
            query = urllib.parse.urlencode({'dataset': plan['dataset'], 'config': config,
                'split': plan['split'], 'offset': position, 'length': 1})
            endpoint = 'https://datasets-server.huggingface.co/rows?' + query
            metadata = fetch(endpoint, 4 * 1024 * 1024)
            time.sleep(1.5)
            item = json.loads(metadata)['rows'][0]
            if item['row_idx'] != position:
                raise ValueError('Dataset server returned a different position')
            row = item['row']
            audio = row['audio'][0]
            data = fetch(audio['src'], 20 * 1024 * 1024)
            path = args.output / (name + '.audio')
            path.write_bytes(data)
            (args.output / (name + '-metadata.json')).write_bytes(metadata)
            manifest['samples'].append({'id': name, 'language': language, 'position': position,
                'config': config, 'dataset_id': row['id'], 'audio_path': str(path.resolve()),
                'audio_sha256': hashlib.sha256(data).hexdigest(), 'reference_text': row['raw_transcription'],
                'metadata_endpoint': endpoint, 'mime_type': audio['type']})
            save()
            print(name + ': downloaded', flush=True)
            time.sleep(1.5)
    manifest['status'] = 'complete'
    save()
    (args.output / 'ATTRIBUTION.txt').write_text('Google FLEURS, CC-BY-4.0\n' + manifest['dataset'] +
        '\nFixed development positions from the recorded plan. Reserved positions were not fetched.\n'
        'Unmodified downloaded audio; analysis converts to mono 16kHz PCM16.\n', encoding='utf-8')


if __name__ == '__main__':
    main()
