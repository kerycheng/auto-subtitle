"""Run local research jobs serially; each probe validates its own checkpoint identity."""
import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('plan', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    jobs = json.loads(args.plan.read_text(encoding='utf-8'))['jobs']
    for job in jobs:
        interpreter = root / ('.cache/envs/alignment/Scripts/python.exe' if job['backend'] == 'qwen'
                              else '.venv/Scripts/python.exe')
        command = [str(interpreter), 'scripts/probe_asr_backends.py', job['manifest'], '--backend', job['backend'],
            '--model', job['model'], '--output', job['output'], '--resume']
        if job.get('temperature_zero'):
            command.append('--temperature-zero')
        if job.get('prompt_file'):
            command.extend(['--prompt-file', job['prompt_file']])
        subprocess.run(command, cwd=root, check=True)


if __name__ == '__main__':
    main()
