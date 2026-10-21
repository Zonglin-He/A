"""Development population and delayed-execution/prefix contracts."""
import json
import numpy as np
import pytest
import torch

from tools import tubedetr_vid_development as dev
from tools import complete_tubedetr_vid_development as queue
from stvg.policy import fit


def test_input_hash_selection_is_parent_balanced_and_independent_of_outcomes():
    rows=[dict(id=f'{p}-{q}',metadata=dict(original_video_id=f'parent{p}'),outcome=p*q)
          for p in range(20)for q in range(3)]
    selected=dev.select_inputs(rows,count=8)
    assert len(selected)==len({r['metadata']['original_video_id']for r in selected})==8
    for r in rows:r['outcome']=-999
    assert [r['id']for r in selected]==[r['id']for r in dev.select_inputs(rows[::-1],count=8)]


def test_no_parameter_evaluation_before_original_main_and_ablation_complete(tmp_path,monkeypatch):
    (tmp_path/'runtime').mkdir()
    monkeypatch.setattr(queue,'ROOT',tmp_path);monkeypatch.setattr(dev,'OUT',tmp_path/'out')
    monkeypatch.setattr(dev,'PUBLIC',tmp_path/'results')
    monkeypatch.setattr(dev,'prepare',lambda:{})
    monkeypatch.setattr(queue,'require_original_completion',lambda:False)
    monkeypatch.setattr(dev,'initialization_candidates',lambda *a:pytest.fail('Premature parameter evaluation'))
    monkeypatch.setattr(queue,'launch_profiles',lambda *a,**kw:pytest.fail('Premature GPU work'))
    queue.main(False)
    state=json.loads((dev.OUT/'status.json').read_text())
    assert state['tuning_candidates_evaluated']==0 and state['new_GPU_work_started'] is False


def test_saved_prefix_matches_shorter_real_opd_run_and_resets_state():
    class Replay:
        def __init__(self):
            self.state=torch.zeros(8,256,dtype=torch.float64,requires_grad=True)
            self.E=torch.tensor([[.5,.5,.2,.3]]*8,dtype=torch.float64)
            self.positions=list(range(8));self.decoder=torch.nn.Identity();self.backwards=0
        def check(self):pass
        def values(self):
            z=self.state[:,:4]
            b=torch.cat((self.E[:,:2]+.05*z[:,:2],self.E[:,2:]*torch.exp(.05*z[:,2:])),dim=1)
            return dict(native=b,boxes=b)
    cfg=dict(scales=[.05,.25],samples=32,lr=.01,tau=.5,steps=6)
    a=Replay();target=a.E.numpy().copy();target[:,:2]+=.01
    long=fit(a,target,'synthetic','Live',41,cfg)
    b=Replay();short=fit(b,target,'synthetic','Live',41,{**cfg,'steps':3})
    assert torch.equal(long['path'][3]['boxes'],short['final'])
    assert a.state.count_nonzero()==b.state.count_nonzero()==0


def test_full_cache_seed_binds_full_manifest_without_resealing_subset(tmp_path,monkeypatch):
    monkeypatch.setattr(dev,'PLAN',tmp_path/'plan.json');dev.PLAN.write_text('{}')
    monkeypatch.setattr(dev,'OLD',tmp_path/'old');dev.OLD.mkdir();(dev.OLD/'sealed.json').write_text('{}')
    record=dict(id='one',dataset='vidstg',key='one',metadata={})
    subset=tmp_path/'subset.jsonl';subset.write_text(json.dumps(record)+'\n')
    full=tmp_path/'full.jsonl';full.write_text(subset.read_text()+json.dumps({**record,'id':'two'})+'\n')
    source=tmp_path/'candidate';p=source/'predictions/one.npz';p.parent.mkdir(parents=True)
    boxes=np.tile([.5,.5,.2,.3],(8,1));value={k:boxes for k in ['Source','E_Ref','Live','Live_raw']}
    value.update(frame_ids=np.arange(8),positions8=np.arange(8),Native_interval=[0,8],Strong_interval=[0,8])
    np.savez(p,**value)
    dev.write(p.with_suffix('.json'),dict(prediction_sha256=dev.sha(p),cost={},checkpoint={},initialization={}))
    before={x:x.read_bytes()for x in p.parent.iterdir()}
    plan=dict(baseline_configuration={},implementation_sha256='fixed',baseline_prediction_seal_sha256='sealed')
    out=tmp_path/'full_selected';params=dev.BASE
    queue.bind_seeded_view(subset,out,params,plan,source=source,scope='full_selected_profile',
                           identity_manifest=full,seal_when_complete=False)
    ident=dev.identity(full,params,'full_selected_profile',plan)
    assert dev.saved(out,record,ident)
    assert not (out/'sealed.json').exists()
    assert all(x.read_bytes()==v for x,v in before.items())
