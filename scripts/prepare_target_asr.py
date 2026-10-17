"""Fixed 30s audio windows; user reference text stays outside ASR manifests."""
import argparse
import json
from pathlib import Path

from autosub.alignment_lab import file_hash
from autosub.domain import Project
from autosub.reference_text import read_srt_events, reconstruct_overlap_bridges


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('project', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output exists')
    p = Project.load(args.project)
    if not p.native_complete or p.language != 'ja':
        raise ValueError('Expected a completed Japanese native project')
    events, trace = reconstruct_overlap_bridges(read_srt_events(args.reference))
    if any(e['end_ms'] > p.duration_ms for e in events):
        raise ValueError('Reference exceeds source duration')
    windows, evidence, selected = [], [], set()
    for start in range(0, p.duration_ms, 30000):
        end = min(start+30000, p.duration_ms)
        rows = [e for e in events if e['start_ms'] >= start+500 and e['end_ms'] <= end-500]
        if not rows:
            continue
        name = f'window-{start//30000}'
        windows.append({'id': name, 'window_ms': [start, end], 'owner_ms': [start, end], 'language': 'ja'})
        evidence.append({'id': name, 'reference_text': ''.join(e['text'] for e in rows),
            'source_rows': [e['source_rows'] for e in rows], 'reference_span_ms': [rows[0]['start_ms'],rows[-1]['end_ms']]})
        selected.update(e['id'] for e in rows)
    if not windows:
        raise ValueError('No complete reference events inside fixed windows')
    args.output.mkdir(parents=True)
    manifest = {'schema': 'asr-lab-v1', 'language': 'ja', 'audio_path': p.audio_path,
        'audio_sha256': file_hash(p.audio_path), 'duration_ms': p.duration_ms, 'text_provided': False,
        'windows': windows, 'scope': 'post-baseline diagnostics; same fixed audio windows, not independent generalization'}
    labels = {'reference_sha256': file_hash(args.reference), 'project_sha256': file_hash(args.project),
        'windows': evidence, 'overlap_reconstruction': trace,
        'unscored_reference_ids': [e['id'] for e in events if e['id'] not in selected],
        'scope': 'only complete reference display events 500ms inside fixed 30s windows; no transcript prompt'}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (args.output/'evidence.json').write_text(json.dumps(labels,ensure_ascii=False,indent=2),encoding='utf-8')
    proof = {'manifest_sha256': file_hash(args.output/'manifest.json'),
        'evidence_sha256': file_hash(args.output/'evidence.json'),
        'reference_path': str(args.reference.resolve()), 'project_path': str(args.project.resolve())}
    (args.output/'provenance.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
    print(len(windows),'fixed audio windows')


if __name__ == '__main__':
    main()
