"""Independent full-support MBR, token mask and saved-array closure checks."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load
from scripts.run_stvg_shared_text_query32_v1 import D,P,PRE,verify
import numpy as np
import torch
from vg_tta.stvg_posterior_decision_tree_v1 import brute_utility,expected_iou_fast

def main():
    torch.set_num_threads(4);p=verify();assert read(D/'GLOBAL_PREDICTION_BARRIER.json')['all_outputs_before_GT'];checks=collections.Counter();temporal={};maxerr=0
    for r in p['records']:
        a=r['alias'];o=load(D/'predictions'/(a+'.pt'));pre=load(D/'prefix'/(a+'.pt'));f=o['fit'];ds=r['dataset'];g=temporal.setdefault(ds,dict(actual_updates=0,noops=0,changed_logits=0,changed_intervals=0,TV=[],logit_change=[],loss_improved=0,loss_worse=0,selected_teacher_positions=0,safely_mapped=0))
        assert o['token_spec']==pre['spec']==p['target_noun_tokens'][a];g['safely_mapped']+=bool(o['token_spec']['indices']);g['selected_teacher_positions']+=len(f['positions'])
        for v in pre['views']:
            n=v['info']['fea_map_size'][0]*v['info']['fea_map_size'][1];L=v['prefix'].shape[0];assert L-2*n==o['token_spec']['token_count']
            selected=np.array([n+i for i in o['token_spec']['indices']],int);assert all(n<=i<L-n for i in selected);checks['actual_shared_text_slice_mapping']+=1
        for key,M in o['distributions'].items():
            ids=o['frame_ids'];i,j=np.triu_indices(len(ids),1);u=brute_utility(ids,M);fast=expected_iou_fast(ids,M)[i,j];err=float(abs(u-fast).max());maxerr=max(err,maxerr);assert err<1e-10
            best=int(u.argmax());iv=[int(ids[i[best]]),int(ids[j[best]])+1];arm='E_MBR'if key=='original'else'Joint';assert iv==o['intervals'][arm];checks['brute_all_legal_interval_decisions']+=len(u)
        g['actual_updates']+=sum(t['updated']for t in f['updates']);g['noops']+=sum(not t['updated']for t in f['updates']);g['changed_intervals']+=o['intervals']['Joint']!=o['intervals']['E_MBR']
        change=max(float((x-y).abs().max())for x,y in zip(f['path'][0]['logits'],f['path'][-1]['logits']));g['changed_logits']+=change>0;g['logit_change'].append(change);g['TV'].append(float(abs(o['distributions']['original']-o['distributions']['adapted']).sum()/2))
        before,after=f['path'][0]['spatial_objective'],f['path'][-1]['spatial_objective'];g['loss_improved']+=after<before;g['loss_worse']+=after>before
        for k,t in enumerate(f['updates']):
            if not t['updated']:assert torch.equal(f['path'][k]['delta'],f['path'][k+1]['delta']);checks['strict_noop']+=1
            else:assert torch.isfinite(t['gradient']).all()and t['gradient'].count_nonzero()>0;checks['finite_nonzero_spatial_gradient']+=1
    for ds,g in temporal.items():
        for k in ['TV','logit_change']:g[k]=dict(mean=float(np.mean(g[k])),max=float(np.max(g[k])),min=float(np.min(g[k])))
    verify(full=True);put(P/'ROOT_FINAL_READBACK.json',dict(status='pass',count=32,checks=dict(checks),max_brute_MBR_utility_error=maxerr,temporal_and_updates=temporal,
        private_full_tensors_kept_private=True,all_outputs_before_GT=True));print(temporal,flush=True)
if __name__=='__main__':main()
