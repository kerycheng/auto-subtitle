"""Read-only listening evidence for unlabelled natural clips; never text repair."""
import html
import json
from pathlib import Path
import wave

import numpy as np

from .alignment_lab import file_hash
from .domain import Project
from .media import load_audio
from .segmentation_audit import uncovered_speech


def targets(project):
    contexts={e['trace_id']:e for e in project.review_events if e.get('action')=='asr_trace_context'}
    result=[]
    for checkpoint in project.review_events:
        if checkpoint.get('action')!='asr_trace_checkpoint':
            continue
        context=contexts.get(checkpoint['trace_id'])
        if not context:
            raise ValueError('Missing decoder trace context')
        for segment in checkpoint['segments']:
            temperature=segment.get('temperature')
            if temperature is None or temperature<=0:
                continue
            if segment.get('start') is None or segment.get('end') is None:
                continue  # unknown clocks cannot be guessed into audio clips
            offset=context['input']['start_ms']
            start=round(offset+segment['start']*1000);end=round(offset+segment['end']*1000)
            if not 0<=start<=end<=project.duration_ms:
                raise ValueError('Observed segment is outside source clock')
            result.append({'kind':'observed-positive-temperature','source_ms':[start,end],
                           'temperature':temperature,'independent_review':'unreviewed',
                           'note':'Selected emitted-segment temperature; rejected attempts unknown, not a confirmed error.'})
    result.extend({'kind':'uncovered-vad-interval','source_ms':span,'independent_review':'unreviewed',
                   'note':'VAD not covered by positive-duration word times; noise or timing drift also possible.'}
                  for span in uncovered_speech(project))
    return sorted(result,key=lambda r:(r['source_ms'][0],r['source_ms'][1],r['kind']))


def build_review(project_path,output):
    project_path,output=Path(project_path),Path(output)
    if output.exists():
        raise ValueError('Use a new review directory; never overwrite listening decisions')
    before=file_hash(project_path);project=Project.load(project_path)
    if project.status!='complete':
        raise ValueError('Wait for a complete baseline before freezing listening targets')
    if output.resolve().is_relative_to(Path(project.media_path).resolve().parent):
        raise ValueError('Keep review derivatives outside original media directory')
    audio=load_audio(Path(project.audio_path))
    if round(len(audio)/16)!=project.duration_ms:
        raise ValueError('Cached audio clock differs')
    output.mkdir(parents=True)
    rows=targets(project)
    for index,row in enumerate(rows,1):
        start,end=row['source_ms']
        a,b=max(0,start-1500)*16,min(len(audio),(end+1500)*16)
        row.update(id=f'listen-{index:03}',context_ms=[a/16,b/16],
                   nearby_cues=[{'start_ms':c.start_ms,'end_ms':c.end_ms,'text':c.text}
                                for c in project.cues if c.start_ms<b/16 and c.end_ms>a/16])
        path=output/(row['id']+'.wav')
        # Byte-equivalent PCM16 samples from the unchanged ASR/playback cache.
        pcm=np.rint(audio[a:b]*32768).astype('<i2')
        with wave.open(str(path),'wb') as f:
            f.setparams((1,2,16000,0,'NONE','not compressed'));f.writeframes(pcm.tobytes())
        if not np.array_equal(load_audio(path),audio[a:b]):
            raise ValueError('Listening excerpt changed source samples')
        row.update(audio_file=path.name,audio_sha256=file_hash(path))
    result={'project_path':str(project_path.resolve()),'project_sha256':before,
            'source_path':project.media_path,'cache_audio_sha256':file_hash(project.audio_path),
            'scope':'unlabelled listening targets, no error verdict or automatic correction',
            'human_completion_rate':None,'targets':rows}
    (output/'targets.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    esc=lambda value:html.escape(str(value),quote=True)
    body=['<!doctype html><meta charset="utf-8"><title>辨識結果優先試聽</title>',
          '<style>body{font:16px system-ui;margin:28px;max-width:1100px}section{border-top:1px solid #bbb;padding:16px 0}audio{width:min(100%,600px)}</style>',
          '<h1>未校對基準：優先試聽位置</h1><p>正溫度與 VAD 空檔只是定位線索，不代表錯字、漏詞或說話者判定。'
          '每份原音訊副本包含目標前後各1.5秒；原詞、原軸與原片均不修改。</p>']
    for row in rows:
        a,b=row['source_ms'];kind='輸出片段溫度 '+str(row['temperature']) if 'temperature' in row else 'VAD／詞時間空檔'
        body.append(f'<section><h2>{esc(row["id"])} · {a/1000:.2f}–{b/1000:.2f}s · {esc(kind)}</h2>'
                    f'<audio controls preload="none" src="{esc(row["audio_file"])}"></audio>'
                    f'<p>{esc(" ／ ".join(c["text"] for c in row["nearby_cues"]))}</p></section>')
    (output/'index.html').write_text(''.join(body),encoding='utf-8')
    if file_hash(project_path)!=before:
        raise ValueError('Listening preparation mutated baseline')
    return result
