"""Serialize all saved update scalars and nonconstant-trajectory diagnostics."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load
from scripts.run_stvg_hc_anchor_content16_v1 import D,P,verify


def main():
    import torch
    torch.set_num_threads(2);plan=verify(True)
    assert read(D/'GLOBAL_PREDICTION_BARRIER.json')['count']==16
    steps=[];rows=[]
    for r in plan['records']:
        z=load(D/'predictions'/(r['alias']+'.pt'))
        for arm,f in z['fits'].items():
            for rr in f['rounds']:
                before=f['path'][rr['step']];after=f['path'][rr['step']+1];q=rr['rollout']
                steps.append(dict(alias=r['alias'],arm=arm,step=rr['step'],updated=rr['updated'],
                    anchor_cost_before=before['anchor_loss'],anchor_cost_after=after['anchor_loss'],
                    fitted_loss=rr['loss'],state_norm_before=float(before['state'].norm()),state_norm_after=float(after['state'].norm()),
                    state_step_L2=float(rr['content_step'].norm()),gradient_L2=None if rr['content_gradient']is None else float(rr['content_gradient'].norm()),
                    mean_gradient_max_error=rr['analytic_error'],double_reference_max_error=rr['double_error'],
                    policy_seed=None if q is None else q['seed'],
                    sampling_policy='none'if q is None else'current actual native mean'if arm=='Live'else'fixed initial native mean',
                    teacher_cost_mean=None if q is None else float(q['cost'].mean()),
                    teacher_KL_from_uniform=None if q is None else float((q['weights']*(q['weights'].clamp_min(1e-30).log()+np.log(32))).sum(-1).mean())))
            e=np.asarray(z['E']);b=f['final'].numpy();delta=np.concatenate((b[:,:2]-e[:,:2],np.log(b[:,2:]/e[:,2:])),1)
            nonconstant=float(np.sqrt(np.square(delta-delta.mean(0)).sum(1).mean()))
            rows.append(dict(alias=r['alias'],arm=arm,steps=f['steps'],parameters=f['parameters'],
                final_state_norms_by_anchor=f['path'][-1]['state'].norm(dim=-1).tolist(),
                final_chart_change_RMS=float(np.sqrt(np.square(delta).sum(1).mean())),
                nonconstant_chart_change_RMS=nonconstant,nonconstant=nonconstant>1e-10))
    assert len(steps)==1920 and not torch.cuda.is_initialized()
    put(P/'STEP_DIAGNOSTICS_PUBLIC.json',dict(status='saved_scalar_readback',count=16,steps=steps,rows=rows,
        actual_samples_weights_states_and_outputs_private=True,no_new_model_or_GT_scoring=True))
    print('serialized',len(steps),'updates; nonconstant trajectories',sum(r['nonconstant']for r in rows),'/',len(rows),flush=True)


if __name__=='__main__':main()
