"""Inference validation may bind inputs/code, never open the GT label files."""
import json
from tools import tubedetr_boxfixed_development as run


def test_checked_never_opens_labels_even_for_hashing(tmp_path,monkeypatch):
    import hashlib
    h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    cfg=tmp_path/'configs/tubedetr_boxfixed.json';cfg.parent.mkdir();cfg.write_text('{}')
    manifest=tmp_path/'manifest.jsonl';manifest.write_text('{}\n')
    p=tmp_path/'plan.json';p.write_text(json.dumps(dict(configuration_sha256=h(cfg),code_sha256={},datasets={'hc2':dict(manifest='manifest.jsonl',manifest_sha256=h(manifest),full_manifest='manifest.jsonl',full_manifest_sha256=h(manifest),labels='MUST_NOT_OPEN_labels.jsonl',labels_sha256='never',full_labels='MUST_NOT_OPEN_full_labels.jsonl',full_labels_sha256='never')})))
    monkeypatch.setattr(run,'ROOT',tmp_path);monkeypatch.setattr(run,'PLAN',p)
    seen=[];original=run.sha
    def checked_hash(path):
        assert 'labels'not in str(path);seen.append(str(path));return original(path)
    monkeypatch.setattr(run,'sha',checked_hash)
    assert run.checked()['datasets']['hc2']['labels_sha256']=='never'
    assert len(seen)==3
