"""Research-only bounded local gain. Not imported by the application pipeline."""
import numpy as np


def compensate(audio, sample_rate=16000):
    if (audio.ndim != 1 or audio.dtype != np.float32 or sample_rate != 16000
            or not np.all(np.isfinite(audio)) or np.any(np.abs(audio) > 1)):
        raise ValueError('Expected original finite mono16k float32 waveform in [-1,1]')
    blocks=[]
    for start in range(0,len(audio),sample_rate):
        end=min(start+sample_rate,len(audio))
        peak=float(np.max(np.abs(audio[start:end])))
        gain=min(10**(24/20),.5/peak) if 0<peak<.1 else 1.
        blocks.append({'sample_frames':[start,end],'peak':peak,'gain':gain})
    output=np.empty_like(audio)
    for i,block in enumerate(blocks):
        start,end=block['sample_frames'];gain=block['gain']
        left=min(gain,blocks[i-1]['gain']) if i else gain
        right=min(gain,blocks[i+1]['gain']) if i+1<len(blocks) else gain
        # Boundary safety knots keep the envelope <= this block's gain. A
        # center-only interpolation can otherwise amplify a neighboring loud
        # block and clip it. Strong blocks and zero blocks stay at unity gain.
        if end-start==1:
            envelope=np.array([min(left,gain,right)],dtype=np.float32)
        else:
            envelope=np.interp(np.arange(start,end),[start,(start+end-1)/2,end-1],
                               [left,gain,right]).astype(np.float32)
        output[start:end]=audio[start:end]*envelope
        block['left_gain'],block['right_gain']=left,right
    return output,{'policy':'local-gain-lab-v1.1','sample_rate':sample_rate,'block_ms':1000,
        'gain_cap_db':24,'blocks':blocks,'source_frames':len(audio),'time_shift_samples':0,
        'research_only':True}


def union_regions(*sources):
    result=[]
    for span in sorted((s for source in sources for s in source),key=lambda s:s['start']):
        if span['start']<0 or span['end']<=span['start']:
            raise ValueError('Invalid VAD sample range')
        if result and span['start']<=result[-1]['end']:
            result[-1]['end']=max(result[-1]['end'],span['end'])
        else:
            result.append({'start':span['start'],'end':span['end']})
    return result
