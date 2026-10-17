"""Validate every paired outcome. Mechanical evidence is not a speech verdict."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from .alignment_lab import file_hash
from .asr_comparison import character_alignment
from .compensation_evaluation import _generated_controls
from .edge_context_lab import prefix_word_candidates
from .language_validation import normalized_text
from .media import load_audio


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def evaluate(plan_path,run_path,confirmed_reference=None,*,confirmed_case_id=None):
    plan,run=read(plan_path),read(run_path)
    if (plan['text_provided'] is not False or run['reference_text_provided'] is not False
            or run['status']!='complete' or run['plan_sha256']!=file_hash(plan_path)
            or run['model_identity']!=plan['model_identity']):
        raise ValueError('Incomplete or changed frozen run')
    cases={c['id']:c for c in plan['cases']}
    if confirmed_reference is not None and confirmed_case_id not in cases:
        raise ValueError('An explicit matching case ID is required for a confirmed reference')
    keys={(c['id'],mode) for c in plan['cases'] for mode in plan['strategies']}
    rows={(r['id'],r['mode']):r for r in run['results']}
    if len(cases)!=len(plan['cases']) or len(rows)!=len(run['results']) or set(rows)!=keys:
        raise ValueError('Missing or duplicated strategy/case')
    for p in plan['source_projects']:
        if file_hash(p['path'])!=p['sha256']:raise ValueError('Original project changed')
    pairs=[];controls=_generated_controls()
    for identifier,c in cases.items():
        if file_hash(c['audio_path'])!=c['audio_sha256']:raise ValueError('Original audio changed')
        audio=load_audio(Path(c['audio_path']));geometry=c['geometry'];a,b=geometry['input_frames']
        samples=audio[a:b];signature=hashlib.sha256(samples.tobytes()).hexdigest()
        if (signature!=c['asr_samples_sha256'] or len(samples)!=b-a
                or [a/16,b/16]!=geometry['source_window_ms']):
            raise ValueError('Source samples or clock changed')
        if c['kind']=='generated-prefix-control':
            meta=c['metadata'];source=load_audio(Path(meta['source_path']))
            if (file_hash(meta['source_path'])!=meta['source_sha256'] or meta['generated_prefix_frames']!=[0,80000]
                    or meta['seed']!=2300 or not np.array_equal(audio[:80000],controls[meta['generator']][:80000])
                    or not np.array_equal(audio[80000:],source)):
                raise ValueError('Known generated prefix or full source changed')
        measured={}
        for mode in plan['strategies']:
            row=rows[identifier,mode]
            expected_options={**plan['decoding'],'clip_timestamps':geometry[
                'original_relative_clips' if mode=='original-clips' else 'expanded_relative_clips']}
            if (row['status']!='complete' or row['options']!=expected_options
                    or row['asr_samples_sha256']!=signature or row['samples']!=len(samples)
                    or row['text']!=''.join(s['text'] for s in row['segments'])
                    or row['resolved_options']['initial_prompt'] is not None):
                raise ValueError('Actual decoder evidence differs from frozen policy')
            temperatures=dict(Counter(str(s['temperature']) for s in row['segments']))
            prefix=prefix_word_candidates(row['segments'],a/16,geometry['original_clip_ms'][0])
            measured[mode]={'text':row['text'],'decoder_temperatures':temperatures,
                'prefix_word_candidates':prefix,'scope':'timestamps in newly exposed region, unassessed unless generated control'}
            if identifier==confirmed_case_id and confirmed_reference is not None:
                measured[mode]['confirmed_case_text_score']=character_alignment(confirmed_reference,row['text'])
        pairs.append({'id':identifier,'kind':c['kind'],'source_window_ms':geometry['source_window_ms'],
            'same_input_samples_both_modes':True,
            'normalized_text_unchanged':normalized_text(measured['original-clips']['text'])==normalized_text(measured['expanded-context']['text']),
            'modes':measured})
    negative=[p for p in pairs if p['kind']=='generated-prefix-control']
    return {'plan_sha256':file_hash(plan_path),'run_sha256':file_hash(run_path),'pairs':pairs,
        'complete_outputs':len(rows),'negative_prefix_candidates':{
            mode:sum(len(p['modes'][mode]['prefix_word_candidates']) for p in negative) for mode in plan['strategies']},
        'natural_pairs_changed':sum(not p['normalized_text_unchanged'] for p in pairs if p['kind']!='generated-prefix-control'),
        'generated_pairs_changed':sum(not p['normalized_text_unchanged'] for p in negative),
        'reference_scope':'Only explicitly identified confirmed text is scored; no full-clip acoustic or text gold is implied',
        'production_adoption':False,'human_completion_rate':None,
        'note':'Zero generated-prefix candidates does not establish natural-noise safety or unchanged speech recognition. No words deleted or projects rewritten.'}
