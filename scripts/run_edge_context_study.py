"""Serial local ASR on the frozen edge study. No human reference is loaded."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from autosub.alignment_lab import file_hash
from autosub.asr_comparison import model_identity
from autosub.media import load_audio
from autosub.runtime import check_runtime


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('plan',type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();plan=json.loads(args.plan.read_text(encoding='utf-8'))
    if args.output.exists():raise ValueError('Choose a new output; no automatic rerun selection')
    if plan['text_provided'] is not False or plan['policy']!='edge-context-lab-v1':raise ValueError('Wrong plan')
    for path,signature in plan['implementation_sha256'].items():
        if file_hash(path)!=signature:raise ValueError('Frozen implementation changed')
    if model_identity(plan['model_path'])!=plan['model_identity']:raise ValueError('Model changed')
    for p in plan['source_projects']:
        if file_hash(p['path'])!=p['sha256']:raise ValueError('Source project changed')
    for c in plan['cases']:
        if file_hash(c['audio_path'])!=c['audio_sha256']:raise ValueError('Source audio changed')
    args.output.mkdir(parents=True);path=args.output/'result.json'
    result={'status':'running','plan_sha256':file_hash(args.plan),'model_identity':plan['model_identity'],
            'reference_text_provided':False,'results':[]}
    def save():
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)
    save();check_runtime('cuda')
    from faster_whisper import WhisperModel
    model=WhisperModel(plan['model_path'],device='cuda',compute_type='float16',local_files_only=True)
    try:
        for case in plan['cases']:
            audio=load_audio(Path(case['audio_path']));a,b=case['geometry']['input_frames'];samples=audio[a:b]
            signature=hashlib.sha256(samples.tobytes()).hexdigest()
            if signature!=case['asr_samples_sha256']:raise ValueError('Source samples changed')
            for mode in plan['strategies']:
                started=time.perf_counter();options=dict(plan['decoding'])
                options['clip_timestamps']=case['geometry']['original_relative_clips' if mode=='original-clips' else 'expanded_relative_clips']
                row={'id':case['id'],'mode':mode,'status':'running','options':options,
                     'asr_samples_sha256':signature,'samples':len(samples),'segments':[]}
                result['results'].append(row);save()
                try:
                    segments,info=model.transcribe(samples,**options)
                    for segment in segments:row['segments'].append(asdict(segment))
                    row.update(status='complete',resolved_options=asdict(info.transcription_options),
                               text=''.join(s['text'] for s in row['segments']))
                except Exception as exc:
                    row.update(status='failed',error_type=type(exc).__name__,error=str(exc))
                row['elapsed_seconds']=time.perf_counter()-started;save()
                print(case['id'],mode,row['status'],row.get('text',''),flush=True)
        result['status']='complete' if all(r['status']=='complete' for r in result['results']) else 'complete-with-failures'
        save()
    except BaseException as exc:
        result.update(status='interrupted-or-failed',error_type=type(exc).__name__,error=str(exc));save();raise


if __name__=='__main__':main()
