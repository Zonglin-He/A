import numpy as np
import torch
from tools.llava_st.backbone import parse_output
from tools.llava_st.evidence import SourceEvidence
from tools.boxfixed_tube.evidence import SourceEvidence as Original

def test_native_tokens_map_to_actual_frames_without_GT():
    ids=np.arange(100)*3+40;s='{<TEMP-010><TEMP-080>} <TEMP-010>:[<WIDTH-010><HEIGHT-020><WIDTH-050><HEIGHT-060>] <TEMP-080>:[<WIDTH-020><HEIGHT-030><WIDTH-060><HEIGHT-070>]'
    r=parse_output(s,ids);assert r['valid']and r['Native_interval']==[70,281]and r['Source'].shape==(100,4);assert np.isclose(r['Source'][10,2],40/99)

def test_invalid_native_output_stays_invalid_not_GT_repaired():
    assert not parse_output('no answer',range(100))['valid'];assert not parse_output('{<TEMP-080><TEMP-010>}',range(100))['valid']

def test_duplicate_short_clip_grid_merged_without_faking_new_frames():
    ids=np.rint(np.linspace(0,8,100)).astype(int);r=parse_output('{<TEMP-000><TEMP-099>} <TEMP-000>:[<WIDTH-010><HEIGHT-020><WIDTH-050><HEIGHT-060>]',ids)
    assert r['valid']and len(r['frame_ids'])==9 and r['Native_interval']==[0,9]

def test_generic_channel_evidence_regresses_original256_exact():
    torch.manual_seed(7);f=torch.randn(8,4,256);v=torch.ones(8,4,dtype=torch.bool);s=torch.rand(8,2,2)
    a=Original(f,v,list(range(8)),list(range(8)),s);b=SourceEvidence(f,v,list(range(8)),list(range(8)),s)
    for key in ['features','valid','seeds','observed','adjacent_forward_indices','adjacent_forward_weights']:assert torch.equal(getattr(a,key),getattr(b,key))
    c=SourceEvidence(torch.randn(8,4,3584),v,list(range(8)),list(range(8)),s);assert c.features.shape[-1]==3584

def test_geometry_dimension_adapter_regresses_original256_and_runs3584():
    from tools.llava_st.spatial import build_geometry
    from tools.boxfixed_tube.spatial import build_geometry as old_geometry,evidence_dp
    torch.manual_seed(8);v=torch.ones(12,4,dtype=torch.bool);s=torch.rand(8,2,2);E=torch.tensor([[.5,.5,.3,.3]]*12,dtype=torch.float64);positions=[0,1,3,4,6,8,10,11]
    ev=SourceEvidence(torch.randn(12,4,256),v,range(12),positions,s);a=old_geometry(ev,E,E[positions]);b=build_geometry(ev,E,E[positions])
    for name,x in a.__dict__.items():
        if torch.is_tensor(x):assert torch.equal(x,getattr(b,name)),name
    ra=evidence_dp(a.for_interval([0,12]));rb=evidence_dp(b.for_interval([0,12]));assert torch.equal(ra['final'],rb['final'])and ra['score']==rb['score']
    ev2=SourceEvidence(torch.randn(12,4,3584),v,range(12),positions,s);g=build_geometry(ev2,E,E[positions]);assert g.features.shape==(12,3,3584)

def test_checked_only_hashes_inputs_and_code_not_labels(tmp_path,monkeypatch):
    import json
    import tools.llava_st.development as run
    from stvg.runtime import sha
    m=tmp_path/'input.jsonl';m.write_text('{}\n');p=tmp_path/'plan.json';p.write_text(json.dumps(dict(code_sha256={},upstream_code_sha256={},datasets={'hc2':dict(manifest=m.name,manifest_sha256=sha(m),full_manifest=m.name,full_manifest_sha256=sha(m),labels='MUST_NOT_READ_labels.jsonl',labels_sha256='never')})))
    monkeypatch.setattr(run,'ROOT',tmp_path);monkeypatch.setattr(run,'PLAN',p);old=run.sha
    def safe(f):assert 'labels'not in str(f);return old(f)
    monkeypatch.setattr(run,'sha',safe);assert run.checked()['datasets']['hc2']['labels_sha256']=='never'

def test_invalid_track_keeps_valid_native_time_in_offline_score(tmp_path,monkeypatch):
    import json
    import tools.llava_st.development as run
    from stvg.runtime import sha
    from stvg.tube_runner import write
    params=dict(gamma=0.,admission=.1,guard=.1,link=0.);plan=tmp_path/'plan.json';plan.write_text('{}');out=tmp_path/'out';root=out/'hc2/profiles'/run.profile_id(params);root.mkdir(parents=True)
    c=dict(Source=np.zeros((10,4),np.float32),frame_ids=list(range(10)),Native_interval=[2,8],valid=False)
    f=root/'synthetic.npz';run.save_prediction(f,c,c['Source'],np.zeros(10,dtype=np.int64));rp=f.with_suffix('.json');rp.write_text('{}')
    labels=tmp_path/'synthetic_labels.jsonl';labels.write_text(json.dumps(dict(id='synthetic',parent='synthetic',interval=[0,10],boxes={'0':[0,0,2,2]}))+'\n')
    s=dict(status='complete',queries=1,plan_sha256=sha(plan),files={'synthetic':dict(path=f.name,sha256=sha(f),receipt_path=rp.name,receipt_sha256=sha(rp))});write(root/'sealed.json',s)
    request=tmp_path/'request.json';request.write_text(json.dumps(params));p=dict(datasets={'hc2':dict(labels=labels.name,labels_sha256=sha(labels))});rows=[dict(id='synthetic',metadata={'width':10,'height':10})]
    monkeypatch.setattr(run,'ROOT',tmp_path);monkeypatch.setattr(run,'OUT',out);monkeypatch.setattr(run,'PLAN',plan);monkeypatch.setattr(run,'rows_for',lambda *_:(p,rows));run.score('hc2',request)
    result=json.loads((root/'metrics.json').read_text());assert result['queries']==1 and result['invalid_native_outputs_retained']==1
    assert result['means']['Source']==dict(v=0.,s=0.,t=60.)and result['means']['Ours']==result['means']['Source']
    # Corrupted prediction must fail before even hashing a label file.
    f.write_bytes(b'corrupt');old=run.sha
    def safe(path):assert Path(path)!=labels;return old(path)
    from pathlib import Path
    monkeypatch.setattr(run,'sha',safe)
    import pytest
    with pytest.raises(AssertionError):run.score('hc2',request)
