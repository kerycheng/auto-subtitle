"""Synthetic staged repositories exercise the publication guard without private fixtures."""
import importlib.util
from pathlib import Path
import subprocess

import pytest

spec = importlib.util.spec_from_file_location('repository_check', Path(__file__).resolve().parents[1] / 'scripts/check_repository.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def repository(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / '.gitignore').write_text('/local/\n', encoding='utf-8')
    return tmp_path


def stage(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
    subprocess.run(['git', '-C', str(root), 'add', '-f', '--', name], check=True)
    return path


def test_checks_index_even_when_working_copy_was_cleaned(tmp_path):
    root = repository(tmp_path)
    path = stage(root, 'src/demo.py', 'value = ' + repr('Z:' + chr(92) + 'Users' + chr(92) + 'example' + chr(92) + 'clip.mp4'))
    # Use a literal path, not a Python-escaped representation, to exercise the matcher.
    stage(root, 'src/demo.py', '# ' + 'Z:' + chr(92) + 'Users' + chr(92) + 'example' + chr(92) + 'clip.mp4')
    path.write_text('# cleaned working copy', encoding='utf-8')
    assert checker.audit(root) == [('src/demo.py', 'personal absolute path')]


@pytest.mark.parametrize('name', ['docs/study.json', 'test-artifacts/result.json', 'media/sample.mp4', 'src/captions.srt'])
def test_rejects_forced_non_source_files(tmp_path, name):
    root = repository(tmp_path)
    stage(root, name, 'synthetic data')
    assert checker.audit(root)[0][1] == 'outside public source/documentation allowlist'


def test_accepts_public_source_and_system_font_path(tmp_path):
    root = repository(tmp_path)
    stage(root, 'src/demo.py', 'font = "' + 'C:' + '/Windows/Fonts/example.ttf"')
    stage(root, 'README.md', '# Public usage')
    assert checker.audit(root) == []


def test_rejects_token_without_printing_secret(tmp_path):
    root = repository(tmp_path)
    token = 'gh' + 'p_' + 'a' * 36
    stage(root, 'src/demo.py', token)
    problems = checker.audit(root)
    assert problems == [('src/demo.py', 'GitHub token')]
    assert token not in str(problems)


def test_only_named_public_text_illustrations_are_allowed(tmp_path):
    root = repository(tmp_path)
    for name in ('main-window', 'vocabulary-selection', 'timing-settings'):
        stage(root, f'docs/images/{name}.svg', '<svg xmlns="http://www.w3.org/2000/svg"><text>Example</text></svg>')
    assert checker.audit(root) == []
    assert not checker.allowed_path('docs/images/personal.svg')
    assert not checker.allowed_path('docs/images/main-window.png')


def test_public_illustration_text_is_still_checked_for_private_paths(tmp_path):
    root = repository(tmp_path)
    path = 'Z:' + chr(47) + 'Users' + chr(47) + 'example' + chr(47) + 'private.mov'
    stage(root, 'docs/images/main-window.svg', '<svg><text>' + path + '</text></svg>')
    assert checker.audit(root) == [('docs/images/main-window.svg', 'personal absolute path')]
