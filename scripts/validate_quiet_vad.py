"""Run ALL prepared empty-VAD cases through the actual generation pipeline."""
import argparse
import json
import wave
from collections import Counter
from pathlib import Path
from threading import Event

import numpy as np
from faster_whisper.vad import get_speech_timestamps, VadOptions

from autosub.alignment_lab import file_hash
from autosub.asr_comparison import character_alignment, model_identity, validate_input
from autosub.comparison import read_reference
from autosub.domain import Project, word_key
from autosub.export import export_subtitles
from autosub.media import load_audio
from autosub.pipeline import run_pipeline


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('labels',type=Path)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.output.exists() and not args.resume: raise ValueError('Output exists')
    data=json.loads(args.manifest.read_text(encoding='utf-8'))
    validate_input(data)
    if file_hash(data['audio_path'])!=data['audio_sha256']: raise ValueError('Input changed')
    labels=json.loads(args.labels.read_text(encoding='utf-8'))
    by_id={r['id']:r for r in labels}
    if len(by_id)!=len(labels) or set(by_id)!={w['id'] for w in data['windows']}:
        raise ValueError('Reference cases differ')
    audio=load_audio(Path(data['audio_path']))
    identity=model_identity(args.model)
    core={name:file_hash(Path('src/autosub')/name) for name in ('pipeline.py','speech_detection.py')}
    report={'status':'running','manifest_sha256':file_hash(args.manifest),'labels_sha256':file_hash(args.labels),
        'model_identity':identity,'core_sources':core,'cases':[],
        'scope':'all raw-empty controlled development cases, original low-amplitude ASR waveform; not livestream accuracy'}
    if args.resume:
        previous=json.loads((args.output/'report.json').read_text(encoding='utf-8'))
        if any(previous.get(k)!=report[k] for k in ('manifest_sha256','labels_sha256','model_identity','core_sources')):
            raise ValueError('Resume input/model/core source changed')
    args.output.mkdir(parents=True,exist_ok=args.resume)
    for window in data['windows']:
        case=args.output/window['id']; case.mkdir(exist_ok=args.resume)
        a,b=window['window_ms']; samples=audio[a*16:b*16]
        options=VadOptions(threshold=.5,min_speech_duration_ms=100,min_silence_duration_ms=100,speech_pad_ms=0)
        if get_speech_timestamps(samples,options,sampling_rate=16000):
            raise ValueError('Prepared case no longer has an empty raw VAD result')
        source=case/'input.wav'
        if source.exists():
            if not args.resume or not np.array_equal(load_audio(source),samples):
                raise ValueError('Existing case source differs')
        else:
            with wave.open(str(source),'wb') as stream:
                stream.setparams((1,2,16000,0,'NONE','not compressed'))
                stream.writeframes(np.rint(samples*32768).astype('<i2').tobytes())
        path=case/'generated.autosub.json'
        if path.exists():
            if not args.resume: raise ValueError('Project exists')
            p=Project.load(path)
            if (p.status!='complete' or p.media_path!=str(source.resolve()) or p.language!=window['language']
                    or p.model!=str(args.model.resolve())):
                raise ValueError('Saved project is incomplete or mismatched')
        else:
            p=Project(media_path=str(source.resolve()),language=window['language'],
                model=str(args.model.resolve()),device='cuda')
            run_pipeline(p,path,Event(),lambda *a:None)
        if not np.array_equal(load_audio(Path(p.audio_path)),samples):
            raise ValueError('ASR cache waveform changed')
        if Counter(k for c in p.cues for k in c.source_tokens or [])!=Counter(word_key(w) for w in p.words):
            raise ValueError('Word ownership changed during segmentation')
        for suffix in (('.ass','.srt') if p.cues else ()):
            exported=case/('generated'+suffix)
            if not exported.exists(): export_subtitles(p,exported)
            parsed,_=read_reference(exported)
            if len(parsed)!=len(p.cues): raise ValueError('Export event count changed')
        truth=by_id[window['id']]['reference_text']
        score=character_alignment(truth,''.join(w.text for w in p.words))
        report['cases'].append({'id':window['id'],'language':p.language,'words':len(p.words),'cues':len(p.cues),
            'source_audio_sha256':file_hash(source),'project_sha256':file_hash(path),
            'vad_analysis':p.review_events,'baseline_character_edits':score['reference_characters'],
            'score':score,'reference':truth,'hypothesis':''.join(w.text for w in p.words)})
        (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(len(report['cases']),len(data['windows']),window['id'],len(p.words),'words',flush=True)
    report['status']='complete'
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
