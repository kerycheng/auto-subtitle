"""Retry an empty VAD result on quiet analysis audio; retain the ASR waveform."""
import math

import numpy as np


def detect_speech(audio, options, detector, *, retry_quiet=True):
    stamps = detector(audio, options, sampling_rate=16000)
    if stamps or not retry_quiet or not len(audio):
        return stamps, None
    peak = float(np.max(np.abs(audio)))
    if not math.isfinite(peak):
        raise ValueError('分析音訊包含無效數值')
    if not 0 < peak < 0.1:
        return stamps, None
    gain = min(10 ** (24 / 20), 0.5 / peak)
    # Same sample count and source clock. Never normalize the shared array in
    # place: recognition, playback and the on-disk WAV use the original samples.
    analyzed = audio * np.float32(gain)
    retry = detector(analyzed, options, sampling_rate=16000)
    return retry, {'action': 'vad_quiet_retry', 'policy': 'quiet-empty-v1',
        'input_peak': peak, 'analysis_gain': gain, 'analysis_gain_db': 20 * math.log10(gain),
        'original_regions': 0, 'retry_regions': len(retry), 'asr_audio_changed': False}


def quiet_retry_applied(project):
    return any(e.get('action') == 'vad_quiet_retry' and e.get('policy') == 'quiet-empty-v1'
               and e.get('retry_regions', 0) > 0 for e in project.review_events)


def external_vad_clips(speech, start_ms, end_ms):
    """Padded original-clock VAD ranges, relative to the unchanged ASR window."""
    clips = []
    for a, b in speech:
        lo, hi = max(start_ms, a-200), min(end_ms, b+200)
        if lo >= hi:
            continue
        if clips and lo <= clips[-1][1]:
            clips[-1][1] = max(clips[-1][1], hi)
        else:
            clips.append([lo, hi])
    return [(v-start_ms)/1000 for clip in clips for v in clip]
