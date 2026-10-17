"""Deterministic synthetic tonal background, not a claim about real game BGM."""
import argparse
import json
import wave
from pathlib import Path

import numpy as np
from autosub.alignment_lab import file_hash


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('labels', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output exists')
    source = json.loads(args.labels.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True)
    cases = []
    for case in source['cases']:
        if file_hash(case['audio_path']) != case['audio_sha256']:
            raise ValueError('Source changed')
        with wave.open(case['audio_path'], 'rb') as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
                raise ValueError('Expected PCM16 mono16k')
            samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float64) / 32768
        t = np.arange(len(samples)) / 16000
        # No speech, lyrics or downloaded music. Same phases/frequencies in every case.
        background = (np.sin(2*np.pi*220*t) + .6*np.sin(2*np.pi*330*t) + .4*np.sin(2*np.pi*440*t)) * (
            .6 + .4*np.sin(2*np.pi*.7*t)**2)
        voice_rms = np.sqrt(np.mean(samples*samples))
        bg_rms = np.sqrt(np.mean(background*background))
        if not voice_rms or not bg_rms:
            raise ValueError('Silent source')
        for snr in (20, 10):
            mixed = samples + background * voice_rms / (bg_rms * 10**(snr/20))
            scale = min(1.0, .99 / np.max(np.abs(mixed)))
            pcm = np.round(mixed * scale * 32767).astype('<i2').tobytes()
            name = case['id'] + f'-tonal-snr{snr}'
            audio = args.output / (name + '.wav')
            with wave.open(str(audio), 'wb') as target:
                target.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                target.writeframes(pcm)
            cases.append({**case, 'id': name, 'audio_path': str(audio.resolve()), 'audio_sha256': file_hash(audio),
                'derived_from_sha256': case['audio_sha256'], 'tonal_background_snr_db': snr,
                'common_gain': scale, 'source_labels_unchanged': True})
    result = {**source, 'scope': 'deterministic tonal interference on synthetic utterance pairs; not realistic game music or natural dialogue',
        'source_labels_sha256': file_hash(args.labels), 'cases': cases}
    (args.output / 'labels.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(len(cases), 'deterministic stress cases')


if __name__ == '__main__':
    main()
