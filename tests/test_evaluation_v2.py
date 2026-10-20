import hashlib
import json
from types import SimpleNamespace
import numpy as np
import pytest
from stvg.evaluation_v2 import score


def test_seconds_strict_thresholds_and_seal(tmp_path):
    ids = list(range(8))
    record = dict(id='synthetic', dataset='hc2', key='synthetic',
        metadata=dict(caption='Synthetic query.', index=0, width=4, height=4, fps=10., frame_ids=ids))
    manifest = tmp_path/'input.jsonl'
    labels = tmp_path/'labels.jsonl'
    manifest.write_text(json.dumps(record)+'\n')
    labels.write_text(json.dumps(dict(id='synthetic', dataset='hc2', parent='synthetic-parent',
        interval=[0, 3], boxes={str(i): [0, 0, 4, 4] for i in range(3)}))+'\n')
    pred = tmp_path/'pred'
    pred.mkdir()
    box = np.tile([.5, .5, 1., 1.], (8, 1))
    np.savez(pred/'one.npz', Source=box, Live=box, frame_ids=ids, positions8=ids,
             Native_interval=[0, 3], Strong_interval=[1, 4])
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    (pred/'sealed.json').write_text(json.dumps(dict(manifest_sha256=sha(manifest), queries=1,
        files={'synthetic': dict(path='one.npz', sha256=sha(pred/'one.npz'))})))
    result = score(manifest, labels, pred, tmp_path/'metrics.json')
    full = result['datasets']['hc2']['query_means']['Full']
    assert full['m_vIoU'] == .5 and full['vIoU@0.5'] == 0
    assert full['tIoU_ge_0_5'] == 1 and full['boundary_MAE_seconds'] == .1
    assert full['observed_actual_sIoU'] == 1 and full['unobserved_actual_sIoU'] is None
    # Hash rejection must happen even when no labels file is available.
    (pred/'one.npz').write_bytes(b'tampered')
    labels.unlink()
    with pytest.raises(ValueError, match='hash mismatch'):
        score(manifest, labels, pred, tmp_path/'bad.json')


def test_tube_request_rejects_labels_before_setup(tmp_path, monkeypatch):
    from stvg import tube_runner, runtime
    request = tmp_path/'request.json'
    request.write_text(json.dumps(dict(dataset='hc2', key='x', metadata=dict(
        caption='Synthetic.', index=0, width=4, height=4, interval=[0, 1]))))
    monkeypatch.setattr(runtime, 'setup', lambda: pytest.fail('Setup ran before schema rejection'))
    with pytest.raises(ValueError, match='Unsupported input fields'):
        tube_runner.predict(SimpleNamespace(request=request, output=tmp_path/'out.npz'))
