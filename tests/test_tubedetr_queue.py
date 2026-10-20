"""Resume a finished direction without rewriting its sealed artifacts."""
import json

import pytest

from tools import complete_tubedetr_main as queue


def test_complete_cached_direction_reused_and_tampering_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, 'ROOT', tmp_path)
    data = tmp_path/'data/evaluation'
    data.mkdir(parents=True)
    manifest, labels = data/'vidstg.jsonl', data/'vidstg_labels.jsonl'
    manifest.write_text(json.dumps(dict(id='one'))+'\n')
    labels.write_text('{}\n')
    out = tmp_path/'outputs/tubedetr/hc2_to_vid'
    (out/'qualitative').mkdir(parents=True)
    (out/'one.npz').write_bytes(b'fixed prediction')
    (out/'one.json').write_text('{}')
    queue.write(out/'sealed.json', dict(queries=1, manifest_sha256=queue.sha(manifest),
        files={'one':dict(path='one.npz', sha256=queue.sha(out/'one.npz'),
                          receipt_path='one.json', receipt_sha256=queue.sha(out/'one.json'))}))
    metrics = dict(status='scored_sealed_predictions', manifest_sha256=queue.sha(manifest),
        labels_sha256=queue.sha(labels), datasets={'vidstg':dict(queries=1,
        query_means={'Source':{},'Full':{}})}, rows=[dict(id='one')])
    queue.write(out/'metrics.json', metrics)
    queue.write(out/'efficiency.json', dict(status='complete_population_efficiency',
        queries=1, manifest_sha256=queue.sha(manifest)))
    queue.write(out/'qualitative/selection.json', dict(selections={}))
    paths = list(out.rglob('*'))
    before = {p:(p.read_bytes(),p.stat().st_mtime_ns) for p in paths if p.is_file()}
    monkeypatch.setattr(queue, 'run', lambda *a: pytest.fail('Completed scoring must not rerun'))
    queue.score('vidstg', 'hc2_to_vid')
    assert all((p.read_bytes(),p.stat().st_mtime_ns) == old for p,old in before.items())
    metrics['rows'][0]['changed'] = True
    queue.write(out/'metrics.json', metrics)
    with pytest.raises(ValueError, match='offline result hashes changed'):
        queue.score('vidstg', 'hc2_to_vid')
