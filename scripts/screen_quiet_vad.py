"""Reproducible quiet-VAD screen on an existing reference-free development manifest."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from faster_whisper.vad import get_speech_timestamps, VadOptions

from autosub.alignment_lab import file_hash
from autosub.asr_comparison import validate_input
from autosub.media import load_audio
from autosub.speech_detection import detect_speech


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('Output exists')
    m=json.loads(args.manifest.read_text(encoding='utf-8'));validate_input(m)
    if file_hash(m['audio_path'])!=m['audio_sha256']: raise ValueError('Audio changed')
    original=load_audio(Path(m['audio_path']))
    options=VadOptions(threshold=.5,min_speech_duration_ms=100,min_silence_duration_ms=100,speech_pad_ms=0)
    report={'manifest_sha256':file_hash(args.manifest),'source_audio_sha256':m['audio_sha256'],
        'helper_sha256':file_hash(Path('src/autosub/speech_detection.py')),'cases':[],'negative_controls':[],
        'scope':'fixed development audio x 0/-18/-30dB PCM16, clean VAD not acoustic gold; synthetic negatives not real BGM'}
    def measured(samples):
        raw=[]
        def detector(audio,*a,**kw):
            stamps=get_speech_timestamps(audio,*a,**kw)
            raw.append(stamps)
            return stamps
        stamps,trace=detect_speech(samples,options,detector)
        if raw[0] and stamps!=raw[0]: raise ValueError('Existing VAD spans changed')
        return {'raw_vad':raw[0],'final_vad':stamps,'analysis':trace,
            'pcm_sha256':hashlib.sha256(np.rint(samples*32768).astype('<i2').tobytes()).hexdigest()}
    for row in m['windows']:
        a,b=row['window_ms'];source=original[a*16:b*16]
        for db in (0,-18,-30):
            samples=np.rint(source*32768*10**(db/20)).clip(-32768,32767).astype('<i2').astype(np.float32)/32768
            report['cases'].append({'source_id':row['id'],'attenuation_db':db,**measured(samples)})
    rng=np.random.default_rng(2300);t=np.arange(160000)/16000
    controls={'silence':np.zeros(len(t)),'white-noise':rng.normal(0,.003,len(t)),
        'hum':.02*np.sin(2*np.pi*220*t),'am-hum':.02*np.sin(2*np.pi*220*t)*(.6+.4*np.sin(2*np.pi*4*t)),
        'chord':.006*(np.sin(2*np.pi*220*t)+np.sin(2*np.pi*330*t)+np.sin(2*np.pi*440*t)),
        'clicks':np.where(np.arange(len(t))%8000==0,.02,0)}
    for name,data in controls.items():
        samples=np.rint(data*32768).clip(-32768,32767).astype('<i2').astype(np.float32)/32768
        report['negative_controls'].append({'name':name,**measured(samples)})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Cases',len(report['cases']),'raw empty',sum(not r['raw_vad'] for r in report['cases']),
        'final empty',sum(not r['final_vad'] for r in report['cases']))


if __name__=='__main__':
    main()
