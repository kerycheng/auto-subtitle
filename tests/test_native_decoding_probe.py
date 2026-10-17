import copy
import runpy
from pathlib import Path

import pytest

from autosub.domain import Project, Word


prepare_probe = runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/probe_native_decoding.py'))['prepare_probe']


def completed_source():
    return Project(duration_ms=1000,asr_strategy='native-v1',native_complete=True,
        native_resume_ms=1000,words=[Word(100,500,'原文')],native_groups=[[0]],
        audio_path='cached.wav',speech=[[100,500]],status='complete')


def test_decoding_probe_resets_only_derived_recognition_without_mutating_source():
    source=completed_source()
    before=copy.deepcopy(source)
    result=prepare_probe(source)
    assert source==before and not result.native_complete
    assert not result.words and not result.native_groups and not result.cues
    assert result.audio_path==source.audio_path and result.speech==source.speech
    assert result.model==source.model and result.settings==source.settings
    result.speech[0][0]=0
    assert source==before


@pytest.mark.parametrize('problem',['unfinished','language','prompt'])
def test_probe_rejects_inapplicable_or_text_prompted_sources(problem):
    source=completed_source()
    if problem=='unfinished':
        source.native_complete=False
    elif problem=='language':
        source.language='en'
    else:
        source.glossary='人工字幕'
    with pytest.raises(ValueError):
        prepare_probe(source)
