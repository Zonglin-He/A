import json
from vg_tta import desta3d_v3_oracle_io as accounting

def test_failed_and_completed_receipts_each_count_once(tmp_path,monkeypatch):
    monkeypatch.setattr(accounting,'ROOT',tmp_path)
    for folder,name,seconds in [('external_privileged_opd_v1','failed',3.),('external_privileged_opd_v2','smoke',4.),('latent_oracle_v1','oracle001',144.82667824599775)]:
        p=tmp_path/'artifacts/desta3d_v3'/folder/name;p.mkdir(parents=True)
        (p/'RECEIPT.json').write_text(json.dumps({'seconds':seconds}))
    assert accounting.total_prior()==39977.51307785203+3+4+144.82667824599775

def test_resume_changes_accounting_and_pins_entry_only(monkeypatch):
    from scripts import desta3d_v3_external_resume as entry
    captured={}
    def fake(name,config,rows,paths):captured.update(name=name,config=config,rows=rows,paths=paths);return 'registered'
    monkeypatch.setattr(entry.io,'register_base',fake);monkeypatch.setattr(entry.io,'prior_seconds',lambda:0)
    entry.install_accounting();assert entry.io.prior_seconds is entry.total_prior
    rows=[{'source':'s'}];cfg={'alpha':.25,'seed':20260928};paths=[]
    assert entry.io.register_base('x',cfg,rows,paths)=='registered'
    assert captured['config']['alpha']==.25 and captured['config']['seed']==20260928
    assert captured['rows'] is rows and cfg=={'alpha':.25,'seed':20260928} and paths==[]
    assert {p.name for p in captured['paths']}=={'desta3d_v3_external_resume.py','desta3d_v3_external_resume_accounting_v1.md'}
