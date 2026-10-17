"""Research-only frozen language-gated evaluation; never imported by the GUI."""
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from threading import Event

import numpy as np

from .alignment_lab import file_hash
from .asr_comparison import character_alignment, model_identity
from .compensation_evaluation import _validate_label, _generated_controls, noise_candidates, source_matches
from .comparison import read_reference
from .domain import Project, word_key
from .export import export_subtitles
from .language_validation import normalized_text
from .level_compensation_lab import compensate, union_regions
from .media import load_audio
from .pipeline import run_pipeline
from .recognition_trace import safe, words_digest
from .sentence_benchmark import assess_pair
from .speech_detection import external_vad_clips


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, data):
    path = Path(path); temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def sample_hash(audio):
    return hashlib.sha256(audio.tobytes()).hexdigest()


def expected_cases(plan):
    result = {}
    for lang in plan['dataset']['languages']:
        for p in plan['dataset']['zero_based_positions']:
            result[f'{lang}-{p}-standalone'] = (lang, 'standalone')
        for left, right in plan['dataset']['pairs']:
            for db in plan['cases']['full_source_pairs']['second_source_db']:
                for gap in plan['cases']['full_source_pairs']['gap_ms']:
                    result[f'{lang}-{left}-{right}-db{abs(db)}-gap{gap}'] = (lang, 'speech-pair')
        source = plan['cases']['known_generated_noise_tails']['source_position']
        for mode in plan['cases']['known_generated_noise_tails']['modes']:
            result[f'{lang}-{source}-noise-{mode}'] = (lang, 'non-speech-tail')
    if len(result) != plan['cases']['total_cases']:
        raise ValueError('Plan case counts do not agree')
    return result


def validate_bundle(inputs, plan_path):
    """Reference-free worker validation; labels are not parsed or sent to ASR."""
    inputs = Path(inputs); plan = read(plan_path); frozen = read(inputs/'pre-asr-freeze.json')
    if not set(plan['dataset']['languages']) <= {'en','zh','ko'}:
        raise ValueError('Candidate is not enabled for Japanese, auto or mixed languages')
    for path, signature in frozen['implementation_sha256'].items():
        if file_hash(path) != signature:
            raise ValueError('Pre-ASR implementation changed')
    for key, path in [('plan_sha256', plan_path), ('cases_sha256', inputs/'cases.json'),
                      ('labels_sha256', inputs/'labels.json'), ('analyses_sha256', inputs/'analyses.json')]:
        if frozen[key] != file_hash(path):
            raise ValueError('Frozen bundle changed: '+key)
    data = read(inputs/'cases.json'); analyses = read(inputs/'analyses.json')
    if data['text_provided'] is not False or data['protocol'] != analyses['protocol']:
        raise ValueError('Reference-bearing or mismatched inputs')
    protocol = data['protocol']
    if (protocol['plan_sha256'] != file_hash(plan_path)
            or protocol['dataset_sha256'] != file_hash(protocol['source_dataset_path'])
            or protocol['model_identity'] != frozen['model_identity']):
        raise ValueError('Frozen source/model identity changed')
    expected = expected_cases(plan)
    cases = {c['id']: c for c in data['cases']}; analysis = {c['id']: c for c in analyses['cases']}
    if (set(cases) != set(expected) or set(analysis) != set(expected)
            or len(cases) != len(data['cases']) or len(analysis) != len(analyses['cases'])):
        raise ValueError('Missing or duplicated fixed case')
    for identifier, c in cases.items():
        if (set(c) != {'id','language','kind','audio_path','audio_sha256','duration_ms'}
                or (c['language'], c['kind']) != expected[identifier]
                or file_hash(c['audio_path']) != c['audio_sha256']):
            raise ValueError('Case/source changed')
        raw = load_audio(Path(c['audio_path'])); a = analysis[identifier]
        processed, gain = compensate(raw)
        if (not len(raw) or round(len(raw)/16) != c['duration_ms']
                or sample_hash(raw) != a['original_samples_sha256']
                or sample_hash(processed) != a['processed_samples_sha256'] or gain != a['gain_trace']
                or not np.array_equal(np.load(inputs/(identifier+'.analysis.npy'),allow_pickle=False),processed)
                or a['union_vad'] != union_regions(a['raw_vad'],a['boosted_vad'])):
            raise ValueError('Frozen analysis/sample clock changed')
    return data, analysis, frozen


def run_study(inputs, plan_path, mode, output, resume=False):
    if mode not in {'production-0.9.4','vad-local-gain-union-original-asr'}:
        raise ValueError('Unknown frozen strategy')
    inputs, output = Path(inputs), Path(output)
    data, analyses, frozen = validate_bundle(inputs, plan_path)
    identity = model_identity(frozen['model_path'])
    if identity != frozen['model_identity']:
        raise ValueError('Local model changed')
    signature = file_hash(inputs/'pre-asr-freeze.json')
    path = output/'report.json'
    report = {'status':'running','mode':mode,'freeze_sha256':signature,
              'model_identity':identity,'reference_text_provided':False,'cases':[]}
    if output.exists():
        if not resume or not path.exists():
            raise ValueError('Use a new directory or explicit resume')
        report = read(path)
        if (report['mode'] != mode or report['freeze_sha256'] != signature
                or report['model_identity'] != identity):
            raise ValueError('Resume settings changed')
        if any(c['status'] == 'running' for c in report['cases']):
            raise ValueError('Interrupted active case needs infrastructure audit; will not silently rerun')
        for row in report['cases']:
            if file_hash(row['project_path']) != row['project_sha256']:
                raise ValueError('Completed/failed artifact changed')
    else:
        output.mkdir(parents=True)
    done = {r['id'] for r in report['cases']}
    if len(done) != len(report['cases']) or not done <= {c['id'] for c in data['cases']}:
        raise ValueError('Invalid resume rows')
    import faster_whisper
    original = faster_whisper.WhisperModel; active = {}

    class ObservedModel:
        def __init__(self,*args,**kwargs):
            self._engine = original(*args,**kwargs)
            self.model = self._engine.model
            self.frames_per_second = self._engine.frames_per_second

        def transcribe(self,audio,**kwargs):
            raw = active['raw']; base = audio
            while isinstance(base.base,np.ndarray):
                base = base.base
            start = (audio.ctypes.data-base.ctypes.data)//audio.dtype.itemsize
            if (not np.array_equal(base,raw) or start < 0
                    or not np.array_equal(audio,raw[start:start+len(audio)])):
                raise ValueError('ASR is not an original source-clock window')
            requested = safe(kwargs)
            if mode != 'production-0.9.4':
                kwargs.pop('vad_parameters',None)
                kwargs['vad_filter'] = False
                kwargs['clip_timestamps'] = external_vad_clips(active['speech'],start/16,(start+len(audio))/16)
                if not kwargs['clip_timestamps']:
                    raise ValueError('No external speech inside scheduled ASR window')
            call = {'start_sample':start,'samples':len(audio),'asr_samples_sha256':sample_hash(audio),
                    'application_requested_options':requested,'actual_options':safe(kwargs),'segments':[]}
            active['row']['decoder_calls'].append(call)
            segments, info = self._engine.transcribe(audio,**kwargs)
            call['info'] = safe(info)
            def recorded():
                try:
                    for segment in segments:
                        call['segments'].append(safe(asdict(segment)))
                        yield segment
                finally:
                    save(path,report)
            return recorded(),info

    faster_whisper.WhisperModel = ObservedModel
    try:
        for case in data['cases']:
            if case['id'] in done:
                continue
            folder = output/case['id']; folder.mkdir()
            project_path = folder/'generated.autosub.json'
            raw = load_audio(Path(case['audio_path']))
            project = Project(media_path=case['audio_path'],language=case['language'],device='cuda',
                              model=frozen['model_path'])
            if mode != 'production-0.9.4':
                project.speech = [[round(s['start']/16),round(s['end']/16)] for s in analyses[case['id']]['union_vad']]
                project.status = 'vad_complete'
            row = {'id':case['id'],'status':'running','decoder_calls':[],'project_path':str(project_path.resolve())}
            report['cases'].append(row); save(path,report)
            active.update(raw=raw,speech=project.speech,row=row)
            try:
                run_pipeline(project,project_path,Event(),lambda *a:None)
                if not np.array_equal(load_audio(Path(project.audio_path)),raw):
                    raise ValueError('Original analysis cache changed')
                for suffix in ('.ass','.srt'):
                    if project.cues:
                        export_subtitles(project,folder/('generated'+suffix))
                row.update(status='complete',words=len(project.words),cues=len(project.cues))
            except Exception as exc:
                row.update(status='failed',error_type=type(exc).__name__,error=str(exc))
                if not project_path.exists():
                    project.status='failed';project.save(project_path)
            row['project_sha256'] = file_hash(project_path)
            save(path,report)
            print(mode,len(report['cases']),len(data['cases']),case['id'],row['status'],flush=True)
        report['status'] = 'complete' if all(r['status']=='complete' for r in report['cases']) else 'complete-with-failures'
        save(path,report)
    finally:
        faster_whisper.WhisperModel = original
    return report
