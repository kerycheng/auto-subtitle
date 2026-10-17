import importlib.util
from pathlib import Path
import pytest


def worker():
    path=Path(__file__).resolve().parents[1]/'scripts'/'probe_clip_context.py'
    spec=importlib.util.spec_from_file_location('probe_clip_context',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_relative_clips_keep_source_silence_gaps_and_original_window_clock():
    assert worker().relative_clips([4.,8.,14.,17.,21.,31.],0,30000)==[4.,8.,14.,17.,21.,30.]
    assert worker().relative_clips([21.,31.,39.,46.],30000,60000)==[0,1.,9.,16.]


@pytest.mark.parametrize('change',['extra-reference','duplicate','bad-clock'])
def test_probe_rejects_text_bearing_duplicate_and_invalid_jobs(change):
    plan={'text_provided':False,'duration_ms':60000,'jobs':[
        {'id':'a','window_ms':[30000,60000],'clip_timestamps':[0,30],'initial_prompt':None}]}
    if change=='extra-reference':plan['jobs'][0]['reference_text']='not for ASR'
    elif change=='duplicate':plan['jobs']*=2
    else:plan['jobs'][0]['clip_timestamps']=[29,31]
    with pytest.raises(ValueError):worker().validate(plan)
