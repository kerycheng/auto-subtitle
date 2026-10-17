import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('prompt_probe', Path(__file__).resolve().parents[1] / 'scripts/probe_prompt_persistence.py')
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)


def plan():
    return {'text_provided': False, 'vocabulary': 'example', 'model_path': 'model', 'model_identity': {},
            'implementation_sha256': {}, 'jobs': [{'id': 'synthetic', 'audio_path': 'audio.wav', 'audio_sha256': 'hash',
            'input_frames': [0, 32000], 'asr_samples_sha256': 'hash', 'options': {
            'language': 'en', 'multilingual': False, 'word_timestamps': True, 'vad_filter': False,
            'clip_timestamps': [0.,1.,1.,2.], 'initial_prompt': None, 'hotwords': 'example',
            'condition_on_previous_text': False, 'beam_size': 5}}]}


def test_reference_free_frozen_jobs_accept_both_hint_channels():
    p = plan(); worker.validate(p)
    p['jobs'][0]['options'].update(initial_prompt='example', hotwords=None, temperature=0.)
    worker.validate(p)


@pytest.mark.parametrize('change', ['reference', 'duplicate', 'overlap', 'clock', 'both-hints', 'full-answer'])
def test_probe_rejects_changed_or_text_bearing_jobs(change):
    p = plan(); job = p['jobs'][0]
    if change == 'reference': p['reference_text'] = 'not for decoder'
    elif change == 'duplicate': p['jobs'] *= 2
    elif change == 'overlap': job['options']['clip_timestamps'] = [0.,1.5,1.,2.]
    elif change == 'clock': job['input_frames'] = [False, 32000]
    elif change == 'both-hints': job['options']['initial_prompt'] = 'example'
    else: job['options']['hotwords'] = 'complete answer'
    with pytest.raises(ValueError): worker.validate(p)
