"""Isolated VAD/ASR gain factors; no reference text is read by this worker."""
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from threading import Event

import numpy as np
from faster_whisper.vad import get_speech_timestamps,VadOptions

from autosub.alignment_lab import file_hash
from autosub.asr_comparison import model_identity
from autosub.domain import Project
from autosub.export import export_subtitles
from autosub.level_compensation_lab import compensate,union_regions
from autosub.media import load_audio
from autosub.pipeline import run_pipeline
from autosub.speech_detection import external_vad_clips


def sample_hash(audio):return hashlib.sha256(audio.tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('cases',type=Path)
    parser.add_argument('--mode',choices=['raw','vad-gain','both-gain'],required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reuse-run',type=Path)
    parser.add_argument('--reuse-cases',type=Path)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Output exists')
    data=json.loads(args.cases.read_text(encoding='utf-8'))
    if data['text_provided'] is not False or data['protocol']['plan_sha256']!=file_hash(args.plan):
        raise ValueError('Wrong plan or text-bearing input')
    allowed={'id','language','audio_path','audio_sha256','duration_ms','kind'}
    if (not data['cases'] or len({c['id'] for c in data['cases']})!=len(data['cases'])
            or any(set(c)!=allowed or c['duration_ms']>30000 for c in data['cases'])):
        raise ValueError('Invalid reference-free cases')
    identity=model_identity(args.model);reusable={}
    if args.reuse_run or args.reuse_cases:
        if args.mode!='raw' or not (args.reuse_run and args.reuse_cases):raise ValueError('Reuse requires both raw inputs')
        previous=json.loads((args.reuse_run/'report.json').read_text(encoding='utf-8'))
        old=json.loads(args.reuse_cases.read_text(encoding='utf-8'))
        if (previous['status']!='complete' or previous['cases_sha256']!=file_hash(args.reuse_cases)
                or previous['model_identity']!=identity):raise ValueError('Reuse identity changed')
        old_inputs={c['id']:c for c in old['cases']}
        for row in previous['cases']:
            old_input=old_inputs[row['id']];path=args.reuse_run/row['id']/'generated.autosub.json'
            if file_hash(path)!=row['project_sha256'] or file_hash(old_input['audio_path'])!=old_input['audio_sha256']:
                raise ValueError('Old result/source changed')
            reusable[row['id']]=(old_input,path)
    import faster_whisper
    original=faster_whisper.WhisperModel;active={}
    class ResearchModel:
        def __init__(self,*a,**kw):
            self.model=original(*a,**kw);self.frames_per_second=self.model.frames_per_second
        def transcribe(self,audio,**kw):
            item=active['item']
            if not np.array_equal(audio,active['raw']):raise ValueError('Expected a complete original source-clock window')
            selected=audio
            if args.mode!='raw':
                kw.pop('vad_parameters',None);kw['vad_filter']=False
                if 'clip_timestamps' not in kw:
                    kw['clip_timestamps']=external_vad_clips(active['speech'],0,len(audio)/16)
                if args.mode=='both-gain':selected=active['processed']
            item['decoder_calls'].append({'options':json.loads(json.dumps(kw)),
                'asr_samples_sha256':sample_hash(selected),'samples':len(selected)})
            segments,info=self.model.transcribe(selected,**kw)
            def recorded():
                for segment in segments:
                    item['decoder_segments'].append(asdict(segment));yield segment
            return recorded(),info
    report={'status':'running','mode':args.mode,'cases_sha256':file_hash(args.cases),
        'plan_sha256':file_hash(args.plan),'model_identity':identity,'reference_text_provided':False,'cases':[]}
    args.output.mkdir(parents=True)
    faster_whisper.WhisperModel=ResearchModel
    try:
        for case in data['cases']:
            if file_hash(case['audio_path'])!=case['audio_sha256']:raise ValueError('Source changed')
            raw=load_audio(Path(case['audio_path']))
            item={'id':case['id'],'kind':case['kind'],'source_sha256':case['audio_sha256'],
                'original_samples_sha256':sample_hash(raw),'decoder_calls':[],'decoder_segments':[],'reused':False}
            old=reusable.get(case['id'])
            if old:
                old_input,path=old
                if any(old_input[k]!=case[k] for k in ('language','audio_sha256','duration_ms')):
                    raise ValueError('Reused source differs')
                p=Project.load(path)
                if p.status!='complete' or p.media_path!=old_input['audio_path']:raise ValueError('Wrong old project')
                item.update(reused=True,decoder_segments=None,decoder_metadata_note='the reused run did not record actual temperatures')
            else:
                folder=args.output/case['id'];folder.mkdir();path=folder/'generated.autosub.json'
                processed,trace=compensate(raw)
                options=VadOptions(threshold=.5,min_speech_duration_ms=100,min_silence_duration_ms=100,speech_pad_ms=0)
                plain=get_speech_timestamps(raw,options,sampling_rate=16000)
                boosted=get_speech_timestamps(processed,options,sampling_rate=16000)
                union=union_regions(plain,boosted)
                item.update(gain_trace=trace,processed_samples_sha256=sample_hash(processed),raw_vad=plain,boosted_vad=boosted,union_vad=union)
                if args.mode!='raw':np.save(folder/'processed-asr-or-analysis.npy',processed)
                p=Project(media_path=case['audio_path'],language=case['language'],model=str(args.model.resolve()),device='cuda')
                if args.mode!='raw':
                    p.speech=[[round(s['start']/16),round(s['end']/16)] for s in union]
                    if not p.speech:raise ValueError('Fixed normal-first-source study expected nonempty union')
                active.update(raw=raw,processed=processed,speech=p.speech,item=item)
                run_pipeline(p,path,Event(),lambda *a:None)
                if not np.array_equal(load_audio(Path(p.audio_path)),raw):raise ValueError('Original analysis cache changed')
                for suffix in (('.ass','.srt') if p.cues else ()):
                    export_subtitles(p,folder/('generated'+suffix))
            item.update(project_path=str(path.resolve()),project_sha256=file_hash(path),words=len(p.words),cues=len(p.cues))
            report['cases'].append(item)
            (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            print(args.mode,len(report['cases']),len(data['cases']),case['id'],len(p.words),'words',flush=True)
        report['status']='complete'
        (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    finally:
        faster_whisper.WhisperModel=original


if __name__=='__main__':main()
