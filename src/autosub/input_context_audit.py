"""Read-only checks on saved native ASR input ranges, never acoustic verdicts."""
import math


def native_input_ranges(project):
    """Union of saved requested clips, not proof that skipped/silent windows decoded."""
    ranges=[]
    for event in project.review_events:
        if event.get('action')!='asr_trace_context' or event.get('strategy')!='native-v1':
            continue
        options=event.get('requested_options') or {}
        clips=options.get('clip_timestamps');offset=(event.get('input') or {}).get('start_ms')
        if (options.get('vad_filter') is not False or not isinstance(clips,list)
                or not clips or len(clips)%2 or not isinstance(offset,(float,int))):
            continue
        local=[]
        for a,b in zip(clips[::2],clips[1::2]):
            if (not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in (a,b,offset))
                    or not 0<=a<b or not 0<=offset+a*1000<offset+b*1000<=project.duration_ms):
                local=[];break
            local.append([offset+a*1000,offset+b*1000])
        ranges.extend(local)
    if not ranges:
        return None  # old projects remain unknown, never recompute guessed historical clips
    merged=[]
    for a,b in sorted(ranges):
        if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],b)
        else:merged.append([a,b])
    return merged


def point_coverage(project,time_ms):
    if not isinstance(time_ms,(int,float)) or not math.isfinite(time_ms) or not 0<=time_ms<=project.duration_ms:
        raise ValueError('Target point must be on the source clock')
    ranges=native_input_ranges(project)
    return {'source_ms':time_ms,'requested_input_covered':None if ranges is None else any(a<=time_ms<b for a,b in ranges),
            'preceding_clip_end_ms':None if ranges is None else max((b for a,b in ranges if b<=time_ms),default=None),
            'following_clip_start_ms':None if ranges is None else min((a for a,b in ranges if a>time_ms),default=None),
            'scope':'saved requested native input; skipped windows and acoustic correctness unobserved'}


def clip_edge_targets(project,tolerance_ms=50):
    if not isinstance(tolerance_ms,int) or isinstance(tolerance_ms,bool) or tolerance_ms<0:
        raise ValueError('Invalid edge tolerance')
    ranges=native_input_ranges(project)
    if ranges is None:
        return {'status':'unknown-input-metadata','targets':[]}
    result=[];previous_end=0
    for start,end in ranges:
        candidates=[(i,w) for i,w in enumerate(project.words) if start<=w.start_ms<end and w.end_ms>w.start_ms]
        if start>previous_end and candidates:
            index,first=min(candidates,key=lambda pair:(pair[1].start_ms,pair[0]))
            if first.start_ms-start<=tolerance_ms:
                result.append({'kind':'word-near-requested-clip-start','word_index':index,'text':first.text,
                    'word_ms':[first.start_ms,first.end_ms],'requested_clip_ms':[start,end],
                    'preceding_excluded_ms':[previous_end,start],
                    'distance_from_input_start_ms':first.start_ms-start,
                    'confirmed_missing_text':False,'independent_review':'unreviewed'})
        previous_end=end
    return {'status':'observed-requested-input','targets':result,
            'scope':'edge proximity can flag missing context; absence does not establish complete recognition'}
