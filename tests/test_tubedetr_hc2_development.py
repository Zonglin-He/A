"""HC initialization scaling, input panel and ordered experiment execution."""
import json
import numpy as np
import pytest
from tools import tubedetr_hc2_development as dev
from tools import complete_tubedetr_hc2_development as queue
from stvg.consensus import apply
from stvg.predictor import output_writeback


def test_hc_half_strength_baseline_is_not_scaled_twice():
    source=np.tile([.45,.55,.2,.3],(9,1)).astype(np.float32)
    delta=np.array([.08,-.03,.2,-.15])
    raw=apply(source,delta);old=output_writeback(source,raw,.5)
    record={'Source':source,'E_Ref':old}
    receipt={'initialization':{'delta':delta.tolist()}}
    plan={'baseline_configuration':{'initialization_gamma':.5}}
    restored=dev.unscaled_initialization(record,receipt,plan)
    assert np.array_equal(restored,raw)
    assert np.array_equal(output_writeback(source,restored,.5),old)
    assert not np.array_equal(output_writeback(source,old,.5),old)
    record['E_Ref']=old+1e-6
    with pytest.raises(ValueError,match='does not reproduce'):
        dev.unscaled_initialization(record,receipt,plan)


def test_hc_input_panel_is_parent_balanced_and_separate_from_vid():
    from tools.tubedetr_vid_development import select_inputs as vid_select
    rows=[dict(id=f'{p}-{q}',metadata=dict(original_video_id=f'{q}_parent{p}.mkv'))
          for p in range(200)for q in range(3)]
    panel=dev.select_inputs(rows)
    assert len(panel)==len({dev.parent_identity(x)for x in panel})==128
    assert panel==dev.select_inputs(rows[::-1])
    assert dev.parent_identity({'metadata':{'original_video_id':'50_parent_with_underscores.mkv'}})=='parent_with_underscores'
    with pytest.raises(ValueError,match='numeric clip prefix'):
        dev.parent_identity({'metadata':{'original_video_id':'bad.mkv'}})
    assert panel!=vid_select(rows)


def test_hc_never_evaluates_candidates_while_vid_full_is_pending(tmp_path,monkeypatch):
    (tmp_path/'runtime').mkdir()
    monkeypatch.setattr(queue,'ROOT',tmp_path)
    monkeypatch.setattr(dev,'OUT',tmp_path/'out');monkeypatch.setattr(dev,'PUBLIC',tmp_path/'results')
    monkeypatch.setattr(dev,'prepare',lambda:{})
    monkeypatch.setattr(queue,'require_original_completion',lambda:False)
    monkeypatch.setattr(dev,'initialization_candidates',lambda *a:pytest.fail('Premature HC candidate'))
    monkeypatch.setattr(queue,'launch_profiles',lambda *a,**kw:pytest.fail('Premature HC GPU work'))
    queue.main(False)
    state=json.loads((dev.OUT/'status.json').read_text())
    assert state['tuning_candidates_evaluated']==0 and state['new_GPU_work_started'] is False


def test_hc_gate_checks_selected_vid_final_hash_and_process_exit(tmp_path,monkeypatch):
    monkeypatch.setattr(queue,'ROOT',tmp_path)
    for name in ['results/tubedetr/final_seal.json','results/tubedetr/ablation_seal.json']:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(dict(status='complete',files={})))
    assert queue.require_original_completion() is False
    vid=tmp_path/'results/tubedetr_vid_development/final_seal.json';vid.parent.mkdir(parents=True)
    payload=tmp_path/'vid_result.json';payload.write_text('{}')
    vid.write_text(json.dumps(dict(status='complete',files={'vid_result.json':dev.sha(payload)})))
    reg=tmp_path/'runtime/tubedetr_active_jobs.json';reg.parent.mkdir()
    reg.write_text(json.dumps(dict(jobs={})))
    assert queue.require_original_completion() is True
    payload.write_text('changed')
    with pytest.raises(ValueError,match='differs from its seal'):
        queue.require_original_completion()


def test_hc_saved_prefix_matches_independent_40_step_and_resets():
    import torch
    from stvg.policy import fit
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
    cfg=dict(scales=[.05,.25],samples=32,lr=.014,tau=.14,steps=80)
    a=Replay();target=a.E.numpy().copy();target[:,:2]+=.01
    long=fit(a,target,'synthetic-hc','Live',41,cfg)
    b=Replay();short=fit(b,target,'synthetic-hc','Live',41,{**cfg,'steps':40})
    assert torch.equal(long['path'][40]['boxes'],short['final'])
    assert a.state.count_nonzero()==b.state.count_nonzero()==0
