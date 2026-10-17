"""Frozen local VAD-context/vocabulary diagnostic, without reading reference text."""
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


def relative_clips(source_clips, start_ms, end_ms):
    if not 0 <= start_ms < end_ms or len(source_clips)%2:
        raise ValueError('Invalid source window/clips')
    result=[]
    for a,b in zip(source_clips[::2],source_clips[1::2]):
        if not 0 <= a < b:
            raise ValueError('Invalid clip clock')
        start,end=max(a*1000,start_ms),min(b*1000,end_ms)
        if start<end:
            result.extend([(start-start_ms)/1000,(end-start_ms)/1000])
    return result


def validate(plan):
    if plan.get('text_provided') is not False or not plan.get('jobs'):
        raise ValueError('Reference-free jobs required')
    ids=set()
    for job in plan['jobs']:
        if set(job)!={'id','window_ms','clip_timestamps','initial_prompt'} or job['id'] in ids:
            raise ValueError('Duplicate or text-bearing job')
        ids.add(job['id']);a,b=job['window_ms']
        clips=job['clip_timestamps']
        if (not 0<=a<b<=plan['duration_ms'] or b-a>30000 or not clips or len(clips)%2
                or any(not 0<=c<d<=(b-a)/1000 for c,d in zip(clips[::2],clips[1::2]))):
            raise ValueError('Invalid source-clock clips')
        if job['initial_prompt'] is not None and not 0<len(job['initial_prompt'])<=500:
            raise ValueError('Invalid fixed vocabulary')


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('plan',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text(encoding='utf-8'));validate(plan)
    if file_hash(plan['audio_path'])!=plan['audio_sha256']:
        raise ValueError('Source waveform changed')
    identity=model_identity(plan['model_path'])
    if identity!=plan['model_identity']:
        raise ValueError('Local model changed')
    if args.output.exists() and not args.resume:
        raise ValueError('Choose a new output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'result.json'
    signature=file_hash(args.plan)
    report={'status':'running','plan_sha256':signature,'model_identity':identity,'results':[],
            'worker_sha256':file_hash(__file__),'scope':'feedback-informed local development, not general quality',
            'reference_text_provided':False,'temperature':0.0}
    if path.exists():
        report=json.loads(path.read_text(encoding='utf-8'))
        if (report['plan_sha256']!=signature or report['model_identity']!=identity
                or report['worker_sha256']!=file_hash(__file__)):
            raise ValueError('Resume settings/code changed')
        done={r['id'] for r in report['results']}
        if len(done)!=len(report['results']) or not done<={j['id'] for j in plan['jobs']}:
            raise ValueError('Resume jobs changed')
    else:done=set()
    def save():
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)
    if len(done)==len(plan['jobs']):
        return
    check_runtime('cuda')
    from faster_whisper import WhisperModel
    model=WhisperModel(plan['model_path'],device='cuda',compute_type='float16',local_files_only=True)
    audio=load_audio(Path(plan['audio_path']))
    try:
        for job in plan['jobs']:
            if job['id'] in done:continue
            a,b=job['window_ms'];samples=audio[a*16:b*16]
            options={'language':'ja','multilingual':False,'word_timestamps':True,'vad_filter':False,
                     'clip_timestamps':job['clip_timestamps'],'initial_prompt':job['initial_prompt'],
                     'condition_on_previous_text':False,'beam_size':5,'temperature':0.0}
            started=time.perf_counter()
            segments,info=model.transcribe(samples,**options)
            rows=[asdict(s) for s in segments]
            report['results'].append({'id':job['id'],'window_ms':job['window_ms'],'samples':len(samples),
                'asr_samples_sha256':hashlib.sha256(samples.tobytes()).hexdigest(),'options':options,
                'resolved_options':asdict(info.transcription_options),'segments':rows,
                'text':''.join(s['text'] for s in rows),'elapsed_seconds':time.perf_counter()-started})
            save();print(job['id'],report['results'][-1]['text'],flush=True)
        report['status']='complete';save()
    except BaseException as exc:
        report.update(status='interrupted-or-failed',error_type=type(exc).__name__,error=str(exc));save();raise


if __name__=='__main__':main()
