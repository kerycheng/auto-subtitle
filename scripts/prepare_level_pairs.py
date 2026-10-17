"""Fixed sub-30s complete-source pairs with a quieter second utterance."""
import argparse
import json
import wave
from pathlib import Path

import numpy as np
from faster_whisper.vad import get_speech_timestamps, VadOptions

from autosub.alignment_lab import file_hash
from autosub.asr_comparison import validate_input
from autosub.media import load_audio
from autosub.speech_detection import detect_speech


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('dataset',type=Path)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--include-control',action='store_true',help='Add unattenuated controls in a separate study output')
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('Output exists')
    data=json.loads(args.dataset.read_text(encoding='utf-8'))
    m=json.loads(args.manifest.read_text(encoding='utf-8'));validate_input(m)
    if m['source_manifest_sha256']!=file_hash(args.dataset) or file_hash(m['audio_path'])!=m['audio_sha256']:
        raise ValueError('Source pairing changed')
    samples={s['id']:s for s in data['samples']}
    windows={w['id']:w for w in m['windows']}
    combined=load_audio(Path(m['audio_path']))
    prepared=[]
    for language in ('en','zh','ja','ko'):
        parts=[]
        for position in (8,9):
            name=f'{language}-{position}';sample=samples[name];window=windows[name]
            path=args.manifest.parent/(name+'.wav');audio=load_audio(path)
            a,b=window['window_ms']
            if (file_hash(sample['audio_path'])!=sample['audio_sha256']
                    or not np.array_equal(np.pad(audio,(0,window['padding_samples'])),combined[a*16:b*16])):
                raise ValueError('Complete source PCM changed')
            parts.append((sample,path,audio))
        if (sum(len(p[2]) for p in parts)/16+800)>30000:
            raise ValueError('Pair exceeds fixed single-window protocol')
        prepared.append((language,parts))
    args.output.mkdir(parents=True)
    labels=[];inputs=[]
    options=VadOptions(threshold=.5,min_speech_duration_ms=100,min_silence_duration_ms=100,speech_pad_ms=0)
    levels=(0,-18,-30) if args.include_control else (-18,-30)
    for language,parts in prepared:
        for db in levels:
            for gap in (0,800):
                name=f'{language}-8-9-db{abs(db)}-gap{gap}'
                left=parts[0][2];right=parts[1][2]
                low=np.rint(right*32768*10**(db/20)).astype('<i2').astype(np.float32)/32768
                start=len(left)+gap*16
                joined=np.concatenate((left,np.zeros(gap*16,dtype=np.float32),low))
                audio=args.output/(name+'.wav')
                with wave.open(str(audio),'wb') as target:
                    target.setparams((1,2,16000,0,'NONE','not compressed'))
                    target.writeframes(np.rint(joined*32768).astype('<i2').tobytes())
                raw=[]
                def detector(a,*args,**kw):
                    result=get_speech_timestamps(a,*args,**kw);raw.append(result);return result
                final,trace=detect_speech(joined,options,detector)
                clean_right=get_speech_timestamps(right,options,sampling_rate=16000)
                proxy_total=sum(s['end']-s['start'] for s in clean_right)
                overlap=sum(max(0,min(v['end'],s['end']+start)-max(v['start'],s['start']+start))
                    for v in raw[0] for s in clean_right)
                regions=[];cursor=0
                for number,(sample,path,source) in enumerate(parts):
                    if number: cursor=start
                    regions.append({'sample_id':sample['id'],'source_path':str(path.resolve()),
                        'source_sha256':file_hash(path),'sample_frames':[cursor,cursor+len(source)],
                        'reference_text':sample['reference_text'],'attenuation_db':db if number else 0})
                entry={'id':name,'language':language,'audio_path':str(audio.resolve()),'audio_sha256':file_hash(audio),
                    'duration_ms':round(len(joined)/16)}
                inputs.append(entry)
                labels.append({**entry,'regions':regions,'raw_vad':raw[0],'final_vad':final,'analysis':trace,
                    'second_clean_vad_proxy_coverage':overlap/proxy_total if proxy_total else None})
    protocol={'pairs':[[f'{lang}-8',f'{lang}-9'] for lang in ('en','zh','ja','ko')],
        'second_gain_db':list(levels),'inserted_gaps_ms':[0,800],'maximum_pair_ms':30000,
        'dataset_sha256':file_hash(args.dataset),'source_manifest_sha256':file_hash(args.manifest),
        'scope':'fixed development complete-source pairs; clean VAD is a coverage proxy, not acoustic gold; no heldouts'}
    (args.output/'cases.json').write_text(json.dumps({'protocol':protocol,'text_provided':False,'cases':inputs},indent=2),encoding='utf-8')
    (args.output/'labels.json').write_text(json.dumps({'protocol':protocol,
        'source_dataset_path':str(args.dataset.resolve()),'cases':labels},ensure_ascii=False,indent=2),encoding='utf-8')
    print(len(inputs),'fixed cases prepared')


if __name__=='__main__':
    main()
