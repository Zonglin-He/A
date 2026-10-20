"""CPU-only isolation checks with synthetic files; never open project labels."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_annotation_guard_checks_symlink_target_and_bytes(tmp_path):
    from stvg.runtime import guard

    labels = tmp_path / 'annotations' / 'private.json'
    labels.parent.mkdir()
    labels.write_text('{}')
    alias = tmp_path / 'innocent.json'
    alias.symlink_to(labels)
    for path in (labels, alias, bytes(alias)):
        with pytest.raises(PermissionError):
            guard('open', (path, 'r', 0))
    guard('open', (tmp_path / 'video.mp4', 'r', 0))
    guard('open', (3, 'r', 0))  # OS descriptors are not interpreted as paths.


@pytest.mark.parametrize('extra', ['boxes', 'interval', 'subject'])
def test_request_metadata_rejected_before_decode_or_setup(tmp_path, monkeypatch, extra):
    from stvg import __main__ as cli
    from stvg import runtime

    def forbidden():
        pytest.fail('Inference setup must not run for rejected input')

    monkeypatch.setattr(runtime, 'setup', forbidden)
    request = tmp_path / 'request.json'
    request.write_text(json.dumps({'dataset': 'hc2', 'metadata': {
        'caption': 'A person walks.', 'index': 0, 'width': 4, 'height': 4,
        extra: 'synthetic forbidden field',
    }}))
    with pytest.raises(ValueError, match='Unsupported input fields'):
        cli.predict(SimpleNamespace(request=str(request), output=str(tmp_path / 'out.npz')))


def sealed_fixture(tmp_path):
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text('\n'.join(json.dumps({'id': k}) for k in ('one', 'two')) + '\n')
    root = tmp_path / 'predictions'
    root.mkdir()
    entries = {}
    for key in ('one', 'two'):
        p = root / (key + '.npz')
        p.write_bytes(b'synthetic saved bytes')
        entries[key] = {'path': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    seal = {'queries': 2, 'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(), 'files': entries}
    (root / 'sealed.json').write_text(json.dumps(seal))
    return manifest, root


@pytest.mark.parametrize('invalid', ['missing_seal', 'bad_manifest', 'later_prediction', 'outside_path'])
def test_score_rejects_invalid_seal_before_reading_labels(tmp_path, monkeypatch, invalid):
    from stvg import evaluation

    manifest, root = sealed_fixture(tmp_path)
    sealpath = root / 'sealed.json'
    seal = json.loads(sealpath.read_text())
    if invalid == 'missing_seal':
        sealpath.unlink()
    elif invalid == 'bad_manifest':
        seal['manifest_sha256'] = 'wrong'
        sealpath.write_text(json.dumps(seal))
    elif invalid == 'later_prediction':
        (root / 'two.npz').write_bytes(b'changed after sealing')
    else:
        outside = tmp_path / 'outside.npz'
        outside.write_bytes(b'synthetic saved bytes')
        seal['files']['two']['path'] = '../outside.npz'
        sealpath.write_text(json.dumps(seal))
    original = evaluation.read_rows
    labels = tmp_path / 'never_read_labels.jsonl'

    def monitored(path):
        assert Path(path) != labels, 'Labels were read before seal validation completed'
        return original(path)

    monkeypatch.setattr(evaluation, 'read_rows', monitored)
    with pytest.raises((FileNotFoundError, ValueError)):
        evaluation.score(manifest, labels, root, tmp_path / 'metrics.json')


def test_score_checks_all_predictions_before_labels(tmp_path, monkeypatch):
    from stvg import evaluation

    manifest, root = sealed_fixture(tmp_path)
    original_sha = evaluation.sha
    checked = []
    labels = tmp_path / 'synthetic_labels.jsonl'
    original_read = evaluation.read_rows

    def digest(path):
        if Path(path).suffix == '.npz':
            checked.append(Path(path).name)
        return original_sha(path)

    def monitored(path):
        if Path(path) == labels:
            assert set(checked) == {'one.npz', 'two.npz'}
            raise RuntimeError('synthetic label boundary reached after complete validation')
        return original_read(path)

    monkeypatch.setattr(evaluation, 'sha', digest)
    monkeypatch.setattr(evaluation, 'read_rows', monitored)
    with pytest.raises(RuntimeError, match='after complete validation'):
        evaluation.score(manifest, labels, root, tmp_path / 'metrics.json')
