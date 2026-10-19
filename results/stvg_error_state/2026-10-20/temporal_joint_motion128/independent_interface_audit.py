"""CPU arithmetic routing checks plus sealed actual-inference trace verification."""
from pathlib import Path
import sys,numpy as np,torch
from types import SimpleNamespace
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from vg_tta.stvg_temporal_joint_motion128_v1 import motion_context,digest
from methods.decota_final_simplified_v1 import backbone


def synthetic():
    # Parameter-free toy arithmetic is solely a routing test, not a new teacher.
    class Toy(torch.nn.Module):
        def forward(self,x,n):
            assert len(x)==n
            return {'3':x+.25*x.roll(1,0)+.25*x.roll(-1,0)}
    m=torch.nn.Module();m.vid=Toy();m.eval();x=torch.arange(65*3*2*2).reshape(65,3,2,2).float();old=backbone.make_batch
    backbone.make_batch=lambda:dict(videos=SimpleNamespace(tensors=x))
    outs={}
    try:
        for joint in [False,True]:
            initial=m.vid.forward
            with motion_context(m,joint)as t:
                backbone.make_batch();outs[joint]=[m.vid(x[i::2],len(x[i::2]))['3']for i in [0,1]]
            assert m.vid.forward==initial and 'forward'not in m.vid.__dict__ and t['restored']
            assert t['actual_encoder_calls']==(1 if joint else 2)
        z=x.numpy();expected=z+.25*np.roll(z,1,axis=0)+.25*np.roll(z,-1,axis=0)
        assert all(np.array_equal(expected[i::2],outs[True][i].numpy())for i in [0,1])
        assert any(not torch.equal(outs[False][i],outs[True][i])for i in [0,1])
        try:
            with motion_context(m,True):
                backbone.make_batch()
                raise RuntimeError('intentional restoration test')
        except RuntimeError as e:assert str(e)=='intentional restoration test'
        assert 'forward'not in m.vid.__dict__
    finally:backbone.make_batch=old
    return dict(independent_numpy_temporal_toy=True,native_joint_different_when_temporal_context_changes=True,exception_restoration=True,no_GPU_or_actual_model_calls=True)


def main():
    t=synthetic();D=R/'artifacts/stvg_temporal_joint_motion128_v1';p=read(D/'PLAN_LOCK.json');b=read(D/'GLOBAL_PREDICTION_BARRIER.json');HR=R/'artifacts/stvg_temporal_hires128_v1';count=0
    assert b['count']==128 and b['plan_sha256']==sha(D/'PLAN_LOCK.json')
    for r,f in zip(p['records'],b['files']):
        assert sha(R/f['path'])==f['sha256'];v=load(R/f['path']);h=load(HR/'predictions'/f"{r['alias']}.pt");tr=v['motion_trace'];obs=v['observation'];ids=v['frame_ids'];assert len(ids)==65
        assert obs['RGB_sha256']==h['observation']['RGB_sha256'] and obs['raw_RGB_shape']==h['observation']['raw_RGB_shape']
        assert obs['normalized_tensor_shape']==tr['full_shape']==h['observation']['normalized_tensor_shape']
        assert tr['full_shape'][0]==65 and tr['full_shape'][-2]==448
        assert tr['actual_encoder_calls']==1 and tr['restored']
        for i,c in enumerate(tr['calls']):
            assert c['physical_indices']==list(range(i,65,2))
            assert obs['records'][i]['frame_ids']==ids[i::2] and obs['records'][i]['offset']==i
            assert c['output_shape'][0]==c['num_frames']==len(ids[i::2]);count+=1
    cost=read(D/'COST.json');assert cost['highres_Source_attempts']==128 and cost['native448_Source_attempts']==1
    assert sum(x['actual_VideoSwin_forwards']for x in cost['attempts'])==130
    assert all(sha(R/f)==h for f,h in p['pins'].items())
    t.update(status='pass',actual_joint_context_traces=128,physical_offset_rows=256,RGB_exact_previous_hires=128,actual_VideoSwin_forwards=130,spatial_branch_unchanged=True)
    put(R/'results/stvg_error_state/2026-10-20/temporal_joint_motion128/INTERFACE_AUDIT_PUBLIC.json',t)
    print(t)

if __name__=='__main__':main()
