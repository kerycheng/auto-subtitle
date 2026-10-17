"""Bounded edge-context experiment. No production imports or automatic adoption."""
import math


def edge_window(previous_end_ms, clip_start_ms, clip_end_ms, duration_ms,
                pre_roll_ms=5000, right_context_ms=8000):
    values=(previous_end_ms,clip_start_ms,clip_end_ms,duration_ms,pre_roll_ms,right_context_ms)
    if (not all(isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v)
                and abs(v-round(v))<1e-6 for v in values)
            or not 0<=previous_end_ms<clip_start_ms<clip_end_ms<=duration_ms
            or pre_roll_ms<=0 or right_context_ms<=0):
        raise ValueError('Expected source-clock millisecond bounds and a preceding excluded gap')
    previous,start,end,duration,pre,right=map(round,values)
    a=max(previous,start-pre);b=min(end,start+right)
    return {'input_frames':[a*16,b*16],'source_window_ms':[a,b],
            'original_clip_ms':[start,b],'excluded_prefix_ms':[a,start],
            'original_relative_clips':[(start-a)/1000,(b-a)/1000],
            'expanded_relative_clips':[0.,(b-a)/1000],
            'policy':'edge-context-lab-v1','automatic_adoption':False}


def prefix_word_candidates(segments, input_start_ms, prefix_end_ms):
    """Word timestamps in a known prefix; cannot decide hallucination vs drift."""
    result=[]
    for segment in segments:
        for word in segment.get('words') or []:
            a=input_start_ms+word['start']*1000;b=input_start_ms+word['end']*1000
            if a<prefix_end_ms and b>=input_start_ms:
                result.append({'text':word['word'],'source_ms':[a,b],
                    'relation':'wholly-before-source-speech' if b<=prefix_end_ms else 'straddles-source-speech-start',
                    'independent_review':'unreviewed','note':'Could be timestamp drift or extra words; never delete automatically.'})
    return result
