"""Check staged repository content before committing or sharing source code."""
import argparse
from pathlib import Path, PurePosixPath
import re
import subprocess

ROOT_FILES = {
    '.editorconfig', '.gitattributes', '.gitignore', 'LICENSE', 'README.md', 'CONTRIBUTING.md', 'AGENTS.md',
    'Start-AutoSubtitle.cmd', 'pyproject.toml', 'requirements-windows-py312.lock',
    'requirements-alignment-windows-py312.lock',
}
PUBLIC_DOCS = {
    'docs/windows-quickstart.txt', 'docs/timing-rules.md', 'docs/release-notes-v0.10.0.md',
    'docs/images/main-window.svg', 'docs/images/vocabulary-selection.svg',
    'docs/images/timing-settings.svg',
}
SENSITIVE_PATTERNS = {
    'personal absolute path': re.compile(r"(?i)(?<![\w])(?:[a-z]:[\\/](?!Windows(?:[\\/]|[\"']))[^\s\"']+|/(?:Users|home)/[^/\s]+/)"),
    'private key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'AWS access key': re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'GitHub token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b'),
    'embedded URL credentials': re.compile(r'https?://[^\s/]+:[^\s/@]+@'),
}


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.PIPE)


def allowed_path(name):
    path = PurePosixPath(name)
    return (name in ROOT_FILES or name in PUBLIC_DOCS or (
        len(path.parts) >= 2 and path.parts[0] in {'src', 'scripts', 'tests'}
        and path.suffix == '.py' and '__pycache__' not in path.parts))


def audit(root):
    """Read index blobs, not working copies; return names/reasons, never secret values."""
    problems = []
    entries = git(root, 'ls-files', '--stage', '-z').split(b'\0')
    for entry in filter(None, entries):
        metadata, raw_name = entry.split(b'\t', 1)
        mode, _, stage = metadata.split()
        name = raw_name.decode('utf-8')
        if mode not in {b'100644', b'100755'} or stage != b'0':
            problems.append((name, 'link, submodule or unresolved index entry'))
            continue
        if not allowed_path(name):
            problems.append((name, 'outside public source/documentation allowlist'))
            continue
        ignored = subprocess.run(['git', '-C', str(root), 'check-ignore', '--no-index', '-q', '--', name],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if ignored.returncode == 0:
            problems.append((name, 'ignored local-only file was forced into the index'))
            continue
        if ignored.returncode != 1:
            raise RuntimeError('Could not evaluate ignore rules')
        data = git(root, 'show', ':' + name)
        if len(data) > 1_000_000 or b'\0' in data:
            problems.append((name, 'binary or oversized file'))
            continue
        try:
            text = data.decode('utf-8-sig')
        except UnicodeDecodeError:
            problems.append((name, 'non UTF-8 content'))
            continue
        for label, pattern in SENSITIVE_PATTERNS.items():
            if pattern.search(text):
                problems.append((name, label))
    return problems


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    problems = audit(args.root)
    for name, reason in problems:
        print(f'{name}: {reason}')
    if problems:
        raise SystemExit(1)
    print('Staged source audit passed. Review the staged diff before committing.')


if __name__ == '__main__':
    main()
