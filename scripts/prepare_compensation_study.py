"""Bind fixed compensation cases and generated non-speech controls before ASR."""
import argparse
import json
import wave
from pathlib import Path

import numpy as np

from autosub.alignment_lab import file_hash
from autosub.media import load_audio


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('inputs',type=Path)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Output exists')
    plan=json.loads(args.plan.read_text(encoding='utf-8'))
    old=json.loads((args.inputs/'cases.json').read_text(encoding='utf-8'))
    labels=json.loads((args.inputs/'labels.json').read_text(encoding='utf-8'))
    if old['protocol']['second_gain_db']!=[0,-18,-30]: raise ValueError('Missing unattenuated controls')
    args.output.mkdir(parents=True)
    rows=[{**c,'kind':'speech-pair'} for c in old['cases']]
    evidence=[{**c,'kind':'speech-pair'} for c in labels['cases']]
    t=np.arange(160000)/16000;rng=np.random.default_rng(2300)
    controls={'silence':np.zeros(len(t)),'white-noise':rng.normal(0,.003,len(t)),
        'hum':.02*np.sin(2*np.pi*220*t),'am-hum':.02*np.sin(2*np.pi*220*t)*(.6+.4*np.sin(2*np.pi*4*t)),
        'chord':.006*(np.sin(2*np.pi*220*t)+np.sin(2*np.pi*330*t)+np.sin(2*np.pi*440*t)),
        'clicks':np.where(np.arange(len(t))%8000==0,.02,0)}
    for language in plan['controls']['initial_languages']:
        original=next(c for c in labels['cases'] if c['language']==language)['regions'][0]
        source=load_audio(Path(original['source_path']))
        if file_hash(original['source_path'])!=original['source_sha256']:raise ValueError('Source changed')
        for name,data in controls.items():
            noise=np.rint(data*32768).clip(-32768,32767).astype('<i2').astype(np.float32)/32768
            joined=np.concatenate((source,noise));identifier=f'{language}-8-noise-{name}'
            path=args.output/(identifier+'.wav')
            with wave.open(str(path),'wb') as out:
                out.setparams((1,2,16000,0,'NONE','not compressed'));out.writeframes(np.rint(joined*32768).astype('<i2').tobytes())
            row={'id':identifier,'language':language,'audio_path':str(path.resolve()),'audio_sha256':file_hash(path),
                'duration_ms':round(len(joined)/16),'kind':'non-speech-tail'}
            rows.append(row)
            evidence.append({**row,'regions':[{**original,'sample_frames':[0,len(source)]}],
                'non_speech_frames':[len(source),len(joined)],'generator':name})
    protocol={'plan_sha256':file_hash(args.plan),'parent_cases_sha256':file_hash(args.inputs/'cases.json'),
        'parent_labels_sha256':file_hash(args.inputs/'labels.json'),'source_dataset_path':labels['source_dataset_path'],
        'dataset_sha256':labels['protocol']['dataset_sha256'],'scope':'fixed development gain study; original files read-only'}
    (args.output/'cases.json').write_text(json.dumps({'text_provided':False,'protocol':protocol,'cases':rows},indent=2),encoding='utf-8')
    (args.output/'labels.json').write_text(json.dumps({'protocol':protocol,'cases':evidence},ensure_ascii=False,indent=2),encoding='utf-8')
    print(len(rows),'cases:',len(old['cases']),'speech pairs and',len(rows)-len(old['cases']),'non-speech tails')


if __name__=='__main__':main()
