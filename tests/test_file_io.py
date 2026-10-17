"""Synthetic snapshots exercise failure/retry semantics without ASR or private input."""
import errno
import json
from dataclasses import asdict

import pytest

from autosub.domain import Project
from autosub import file_io


def windows_error(code):
    error = PermissionError(errno.EACCES, 'synthetic access failure')
    error.winerror = code
    return error


@pytest.mark.parametrize('code', [5, 32, 33])
def test_transient_windows_denial_retries_same_complete_snapshot(tmp_path, monkeypatch, code):
    path = tmp_path / 'synthetic.autosub.json'
    old = Project(glossary='before'); old.save(path)
    before = path.read_bytes(); new = Project(glossary='after')
    original = file_io._replace; calls = []; delays = []
    def replace(source, target):
        calls.append((source, target))
        assert Project.load(source) == new and path.read_bytes() == before
        if len(calls) < 3:
            raise windows_error(code)
        original(source, target)
    monkeypatch.setattr(file_io, '_replace', replace)
    monkeypatch.setattr(file_io, '_sleep', delays.append)
    new.save(path)
    assert Project.load(path) == new
    assert len(calls) == 3 and len(set(calls)) == 1 and delays == [.05, .1]
    assert not path.with_suffix(path.suffix + '.tmp').exists()


def test_permanent_windows_denial_is_bounded_and_keeps_both_versions(tmp_path, monkeypatch):
    path = tmp_path / 'synthetic.autosub.json'; old = Project(glossary='before'); old.save(path)
    before = path.read_bytes(); new = Project(glossary='after'); calls = []; delays = []
    failure = windows_error(5)
    def fail(*args):
        calls.append(args); raise failure
    monkeypatch.setattr(file_io, '_replace', fail)
    monkeypatch.setattr(file_io, '_sleep', delays.append)
    with pytest.raises(PermissionError) as caught:
        new.save(path)
    assert caught.value is failure and len(calls) == 5 and sum(delays) == .75
    assert path.read_bytes() == before and Project.load(path) == old
    assert Project.load(path.with_suffix(path.suffix + '.tmp')) == new


@pytest.mark.parametrize('failure', [PermissionError(errno.EACCES, 'generic denied'),
                                    OSError(errno.ENOSPC, 'full'), FileNotFoundError('missing')])
def test_other_io_errors_fail_immediately_without_discarding_snapshot(tmp_path, monkeypatch, failure):
    path = tmp_path / 'synthetic.autosub.json'; old = Project(glossary='before'); old.save(path)
    before = path.read_bytes(); new = Project(glossary='after'); calls = []
    def fail(*args):
        calls.append(args); raise failure
    monkeypatch.setattr(file_io, '_replace', fail)
    monkeypatch.setattr(file_io, '_sleep', lambda *_: pytest.fail('must not retry unrelated errors'))
    with pytest.raises(type(failure)) as caught:
        new.save(path)
    assert caught.value is failure and len(calls) == 1 and path.read_bytes() == before
    assert Project.load(path.with_suffix(path.suffix + '.tmp')) == new


def test_migration_backup_remains_exact_when_replacement_fails(tmp_path, monkeypatch):
    path = tmp_path / 'synthetic.autosub.json'
    raw = asdict(Project(asr_strategy='fixed30-v1')); raw['schema_version'] = 3
    path.write_text(json.dumps(raw, indent=4), encoding='utf-8'); before = path.read_bytes()
    migrated = Project.load(path)
    def fail(*_): raise windows_error(5)
    monkeypatch.setattr(file_io, '_replace', fail); monkeypatch.setattr(file_io, '_sleep', lambda *_: None)
    with pytest.raises(PermissionError): migrated.save(path)
    assert path.read_bytes() == before
    assert path.with_suffix(path.suffix + '.v3.bak').read_bytes() == before
    assert Project.load(path.with_suffix(path.suffix + '.tmp')) == migrated


def test_bad_serialization_cannot_truncate_a_previous_pending_snapshot(tmp_path):
    path = tmp_path / 'synthetic.autosub.json'; Project().save(path)
    before = path.read_bytes(); temporary = path.with_suffix(path.suffix + '.tmp')
    pending = Project(glossary='pending'); temporary.write_text(json.dumps(asdict(pending)), encoding='utf-8')
    checkpoint = temporary.read_bytes()
    invalid = Project(glossary=object())
    with pytest.raises(TypeError): invalid.save(path)
    assert path.read_bytes() == before and temporary.read_bytes() == checkpoint
