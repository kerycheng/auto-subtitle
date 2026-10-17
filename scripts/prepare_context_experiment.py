"""Prepare alternate acoustic context; text evidence is separate from ASR input."""
import argparse
from pathlib import Path

from autosub.asr_experiments import prepare_contexts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('cases', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--padding-ms', type=int, required=True)
    args = parser.parse_args()
    result = prepare_contexts(args.cases, args.output, args.padding_ms)
    print(f"{len(result['windows'])} cases, {result['duration_ms']} ms, no transcript supplied")
