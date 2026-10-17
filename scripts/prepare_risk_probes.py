"""Select mechanical audition risks without supplying any text to either ASR."""
import argparse
import json
import wave
from pathlib import Path

from autosub.alignment_lab import file_hash
from autosub.domain import Project
from autosub.segmentation_audit import uncovered_speech
from autosub.timing import visible_length


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('suite',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Please select a fresh output directory')
    suite=json.loads((args.suite/'validation.json').read_text(encoding='utf-8'))
    if suite['status']!='passed':
        raise ValueError('Expected a completed development suite')
    selected=[]
    for sample in suite['samples']:
        p=Project.load(args.suite/sample['id']/'native.autosub.json')
        risks=[(a,b,'uncovered-vad') for a,b in uncovered_speech(p)]
        # Keep ambiguous short native fragments; normal short replies are retained.
        risks.extend((c.start_ms,c.end_ms,'short-low-confidence-native') for c in p.cues
                     if visible_length(c.text)<=4 and c.end_ms-c.start_ms<1200 and
                     '辨識信心偏低' in c.flags)
        for a,b,reason in risks:
            lo,hi=max(0,a-3000),min(p.duration_ms,b+3000)
            selected.append({'id':f'{sample["id"]}-{len(selected)}','sample':sample['id'],
                'reason':reason,'audio_path':p.audio_path,'audio_sha256':file_hash(p.audio_path),
                'context_ms':[lo,hi],'target_ms':[a,b],
                'primary_context_text':' '.join(c.text for c in p.cues if c.start_ms<hi and c.end_ms>lo)})
    args.output.mkdir(parents=True)
    wav_path=args.output/'risk-contexts.wav'
    cursor,windows=0,[]
    with wave.open(str(wav_path),'wb') as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        for case in selected:
            lo,hi=case['context_ms']
            with wave.open(case['audio_path'],'rb') as source:
                assert (source.getframerate(),source.getnchannels(),source.getsampwidth())==(16000,1,2)
                source.setpos(lo*16)
                data=source.readframes((hi-lo)*16)
                assert len(data)==(hi-lo)*32
                target.writeframes(data)
            windows.append({'id':case['id'],'window_ms':[cursor,cursor+hi-lo],
                'owner_ms':[cursor,cursor+hi-lo], 'language':'ja'})
            cursor+=hi-lo
    # Baseline/context text is in a separate evidence file. Probe inputs are audio only.
    (args.output/'cases.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest={'schema':'asr-lab-v1','strategy':'risk-review-only','audio_path':str(wav_path.resolve()),
              'audio_sha256':file_hash(wav_path),'duration_ms':cursor,'language':'ja',
              'text_provided':False,'windows':windows}
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'cases':len(windows),'audio_ms':cursor}))


if __name__=='__main__':
    main()
