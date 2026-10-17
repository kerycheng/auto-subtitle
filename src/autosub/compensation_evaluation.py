"""Read-only evaluation of the frozen local-gain study, never transcript repair."""
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np

from .alignment_lab import file_hash
from .asr_comparison import character_alignment
from .comparison import read_reference
from .domain import Project,word_key
from .language_validation import normalized_text
from .level_compensation_lab import compensate
from .media import load_audio
from .sentence_benchmark import assess_pair,matched_positions


def source_matches(regions,hypothesis):
    references=[normalized_text(r['reference_text']) for r in regions]
    _,matches=matched_positions(''.join(references),normalized_text(hypothesis))
    rows=[];cursor=0
    for region,reference in zip(regions,references):
        rows.append({'sample_id':region['sample_id'],'reference_characters':len(reference),
            'exact_matched_characters':sum(cursor<=p<cursor+len(reference) for p in matches.values())})
        cursor+=len(reference)
    return {'regions':rows,'scope':'minimum-edit exact-character matches; repeated text can be ambiguous, not a source-specific error rate'}


def noise_candidates(project,frames):
    start,end=[f/16 for f in frames]
    return [{'text':w.text,'raw_word_ms':[w.start_ms,w.end_ms],
        'relation':'starts-in-generated-nonspeech' if w.start_ms>=start else 'straddles-generated-nonspeech',
        'independently_reviewed':False}
        for w in project.words if w.start_ms<end and (w.end_ms>start or w.start_ms>=start)]


def _sample_hash(audio):return hashlib.sha256(audio.tobytes()).hexdigest()


def _generated_controls():
    t=np.arange(160000)/16000;rng=np.random.default_rng(2300)
    raw={'silence':np.zeros(len(t)),'white-noise':rng.normal(0,.003,len(t)),
        'hum':.02*np.sin(2*np.pi*220*t),'am-hum':.02*np.sin(2*np.pi*220*t)*(.6+.4*np.sin(2*np.pi*4*t)),
        'chord':.006*(np.sin(2*np.pi*220*t)+np.sin(2*np.pi*330*t)+np.sin(2*np.pi*440*t)),
        'clicks':np.where(np.arange(len(t))%8000==0,.02,0)}
    return {k:np.rint(v*32768).clip(-32768,32767).astype('<i2').astype(np.float32)/32768 for k,v in raw.items()}


def _validate_label(case,label,dataset,controls):
    if any(label[k]!=case[k] for k in case):raise ValueError('Case/evidence pairing changed')
    audio=load_audio(Path(case['audio_path']))
    sources=[]
    for region in label['regions']:
        original=dataset[region['sample_id']]
        if (region['reference_text']!=original['reference_text'] or original['language']!=case['language']
                or file_hash(region['source_path'])!=region['source_sha256']):
            raise ValueError('Original reference/source changed')
        sources.append(load_audio(Path(region['source_path'])))
    first=label['regions'][0]
    if first['sample_frames']!=[0,len(sources[0])] or not np.array_equal(audio[:len(sources[0])],sources[0]):
        raise ValueError('First complete source changed')
    if case['kind']=='speech-pair':
        second=label['regions'][1];start,end=second['sample_frames'];db=second['attenuation_db']
        if db not in (0,-18,-30) or start-len(sources[0]) not in (0,12800) or end!=len(audio):
            raise ValueError('Invalid source regions/attenuation')
        expected=np.rint(sources[1]*32768*10**(db/20)).astype('<i2').astype(np.float32)/32768
        if (end-start!=len(sources[1]) or np.any(audio[len(sources[0]):start])
                or not np.array_equal(audio[start:end],expected)):
            raise ValueError('Second complete source changed')
    else:
        if (label['non_speech_frames']!=[len(sources[0]),len(audio)]
                or not np.array_equal(audio[len(sources[0]):],controls[label['generator']])):
            raise ValueError('Generated non-speech no longer matches frozen control')
    return audio


def evaluate_study(inputs,run_dirs,plan_path):
    inputs,plan_path=Path(inputs),Path(plan_path)
    cases_path=inputs/'cases.json';labels_path=inputs/'labels.json'
    cases=json.loads(cases_path.read_text(encoding='utf-8'));labels=json.loads(labels_path.read_text(encoding='utf-8'))
    if cases['text_provided'] is not False or cases['protocol']!=labels['protocol']:
        raise ValueError('Study protocol/evidence pairing changed')
    protocol=cases['protocol']
    if (protocol['plan_sha256']!=file_hash(plan_path)
            or file_hash(protocol['source_dataset_path'])!=protocol['dataset_sha256']):
        raise ValueError('Frozen plan/reference dataset changed')
    dataset={r['id']:r for r in json.loads(Path(protocol['source_dataset_path']).read_text(encoding='utf-8'))['samples']}
    expected={c['id']:c for c in cases['cases']};by_id={c['id']:c for c in labels['cases']}
    if len(expected)!=len(cases['cases']) or len(by_id)!=len(labels['cases']) or set(expected)!=set(by_id):
        raise ValueError('Duplicate or mismatched study IDs')
    controls=_generated_controls();audio={}
    for identifier,case in expected.items():
        if file_hash(case['audio_path'])!=case['audio_sha256']:raise ValueError('Original input changed')
        audio[identifier]=_validate_label(case,by_id[identifier],dataset,controls)
    result={'plan_sha256':file_hash(plan_path),'cases_sha256':file_hash(cases_path),'labels_sha256':file_hash(labels_path),
        'runs':[],'scope':'frozen development diagnostics; one trial per strategy, not general livestream quality or acoustic gold'}
    identity=None;modes=set()
    for folder in map(Path,run_dirs):
        path=folder/'report.json';run=json.loads(path.read_text(encoding='utf-8'));mode=run['mode']
        if (run['status']!='complete' or run['cases_sha256']!=file_hash(cases_path)
                or run['plan_sha256']!=file_hash(plan_path) or run['reference_text_provided'] is not False
                or len(run['cases'])!=len(expected) or {r['id'] for r in run['cases']}!=set(expected)
                or mode not in {'raw','vad-gain','both-gain'} or mode in modes):
            raise ValueError('Incomplete, duplicated or changed run')
        if identity is not None and identity!=run['model_identity']:raise ValueError('Models differ')
        identity=run['model_identity'];modes.add(mode)
        measured={'mode':mode,'run_sha256':file_hash(path),'cases':[],'summary':{}}
        totals=defaultdict(lambda:{'character_edits':0,'reference_characters':0,'cases':0})
        for row in run['cases']:
            identifier=row['id'];case=expected[identifier];label=by_id[identifier];raw=audio[identifier]
            project_path=Path(row['project_path'])
            if (row['source_sha256']!=case['audio_sha256'] or row['original_samples_sha256']!=_sample_hash(raw)
                    or file_hash(project_path)!=row['project_sha256']):raise ValueError('Result/input changed')
            p=Project.load(project_path)
            if (p.status!='complete' or p.language!=case['language']
                    or file_hash(p.media_path)!=case['audio_sha256'] or not np.array_equal(load_audio(Path(p.audio_path)),raw)):
                raise ValueError('Saved source-clock project changed')
            if row['reused']:
                if mode!='raw' or row['decoder_segments'] is not None:raise ValueError('Invalid reused metadata')
                temperatures=None
            else:
                processed,trace=compensate(raw)
                if row['processed_samples_sha256']!=_sample_hash(processed) or trace!=row['gain_trace']:
                    raise ValueError('Gain analysis differs from frozen algorithm')
                selected=processed if mode=='both-gain' else raw
                if not row['decoder_calls']:raise ValueError('Missing decoder evidence')
                for call in row['decoder_calls']:
                    if (call['asr_samples_sha256']!=_sample_hash(selected) or call['samples']!=len(raw)
                            or call['options'].get('initial_prompt') not in (None,'')):
                        raise ValueError('ASR samples or no-prompt evidence changed')
                if mode!='raw' and not np.array_equal(np.load(folder/identifier/'processed-asr-or-analysis.npy',allow_pickle=False),processed):
                    raise ValueError('Processed waveform changed')
                temperatures=dict(Counter(str(s['temperature']) for s in row['decoder_segments']))
            keys=[word_key(w) for w in p.words];actual=[k for c in p.cues for k in c.source_tokens or []]
            hypothesis=''.join(w.text for w in p.words);reference=''.join(r['reference_text'] for r in label['regions'])
            if actual!=keys or normalized_text(hypothesis)!=normalized_text(''.join(c.text for c in p.cues)):
                raise ValueError('Words/ownership/order/display text changed')
            for suffix in ('.ass','.srt'):
                if p.cues and len(read_reference(project_path.parent/('generated'+suffix))[0])!=len(p.cues):
                    raise ValueError('Export event count differs')
            score=character_alignment(reference,hypothesis)
            category='noise' if case['kind']=='non-speech-tail' else ('original-level' if label['regions'][1]['attenuation_db']==0 else 'attenuated')
            key=case['language']+'/'+category
            totals[key]['character_edits']+=score['character_edits'];totals[key]['reference_characters']+=score['reference_characters'];totals[key]['cases']+=1
            measured['cases'].append({'id':identifier,'language':p.language,'category':category,'reference':reference,
                'hypothesis':hypothesis,'score':score,'source_match_evidence':source_matches(label['regions'],hypothesis),
                'ownership':assess_pair(p,label) if case['kind']=='speech-pair' else None,
                'generated_noise_timestamp_candidates':noise_candidates(p,label['non_speech_frames']) if category=='noise' else [],
                'decoder_temperatures':temperatures,'reused':row['reused']})
        measured['summary']=dict(totals);result['runs'].append(measured)
    if modes!={'raw','vad-gain','both-gain'}:raise ValueError('All three strategies required')
    baseline={c['id']:c for r in result['runs'] if r['mode']=='raw' for c in r['cases']}
    for run in result['runs']:
        run['regressions_vs_raw']=[c['id'] for c in run['cases'] if c['score']['character_edits']>baseline[c['id']]['score']['character_edits']]
    result['production_decision']='Keep 0.9.3; local gain remains research-only. Do not infer broad benefit from pooled scores.'
    return result
