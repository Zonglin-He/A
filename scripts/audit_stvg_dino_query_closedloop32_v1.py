"""Saved-array/matched-branch/gradient and independent official-frame audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_dino_query_closedloop32_v1 import D,P,PRE,verify

def main():
    import torch
    p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['all_outputs_before_GT']and bar['count']==32
    for v in bar['files']:assert sha(ROOT/v['path'])==v['sha256']
    from scripts.score_stvg_opd_p1_v1 import truths
    allgt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};torch.set_num_threads(4)
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in allgt};results=read(P/'RESULTS_PUBLIC.json');rows={r['alias']:r for r in results['rows']};checks=collections.Counter()
    for r in p['records']:
        a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];o=load(D/'predictions'/(a+'.pt'));old=load(PRE/'predictions'/(a+'.pt'));e=o['outputs']['E_MBR'];pos=o['planned_positions'];s=load(PRE/'source'/(a+'.pt'))
        assert np.array_equal(e,old['outputs']['E_MBR'])and np.array_equal(o['outputs']['Center_full_local'],old['outputs']['Center_full_local']);checks['exact_two_baselines']+=2
        f,g=o['fits']['Fixed_sample_KL'],o['fits']['On_policy_KL']
        assert torch.equal(f['path'][1]['boxes'],g['path'][1]['boxes'])and torch.equal(f['rounds'][0]['QR_step'],g['rounds'][0]['QR_step']);checks['matched_first_round']+=1
        for arm,fit in o['fits'].items():
            assert torch.equal(fit['path'][0]['boxes'],torch.tensor(e))and torch.count_nonzero(fit['path'][0]['state']['spatial.query_residual'])==0
            assert np.array_equal(fit['final'].numpy(),o['outputs'][arm])and fit['selected_step']==2 and fit['full_two_pass_exact'];checks['zero_E_final2_fullpass']+=1
            for k,t in enumerate(fit['rounds']):
                mu=t['mean'].numpy();tar=t['targets'].numpy();m=t['mask'].numpy();sc=np.array(t['scale']);expected=((mu[:,None]-tar)/sc**2*m[:,:,None]).sum(1)/16
                assert abs(expected-t['mean_gradient'].numpy()).max()<1e-10;checks['analytic_output_mean_gradients']+=1
                assert abs(.5*(((mu[:,None]-tar)/sc)**2).sum(-1).__mul__(m).sum()/16-t['loss'])<1e-9;checks['independent_KL_losses']+=1
                for name,val in fit['path'][k]['state'].items():
                    if name!='spatial.query_residual':assert torch.equal(val,fit['path'][k+1]['state'][name]);checks['frozen_LN_state_arrays']+=1
                assert torch.equal(fit['path'][k+1]['state']['spatial.query_residual']-fit['path'][k]['state']['spatial.query_residual'],t['QR_step'])
                if not t['updated']:assert torch.count_nonzero(t['QR_step'])==0;checks['no_feedback_exact_noop']+=1
            checks['scheduled_QR_rounds_recorded']+=2;checks['actual_Adam_steps']+=sum(v['updated']for v in fit['rounds'])
        # Reconstruct the live second-round center shift from unchanged original draws.
        if o['active']:
            from vg_tta.stvg_error_state_correction_v1 import encode,decode
            initE=load(PRE/'E'/(a+'.pt'));sc=np.asarray(old['scale']);initial=np.stack([decode(encode(s['Source'].double().numpy())+aa*sc)for aa in old['actions'][4:8]])
            shift=encode(g['path'][1]['boxes'].numpy())-encode(e);expected=np.stack([decode(encode(b)+shift)for b in initial]);assert np.array_equal(expected,g['rounds'][1]['live_reference_boxes']);checks['live_current_center_refs']+=1
            fixed=f['rounds'][1];assert np.array_equal(fixed['live_reference_boxes'],np.stack([decode(encode(b))for b in initial]));checks['matched_second_round_layout_fixed_refs']+=1
            for i,tr in g['rounds'][1]['live_traces'].items():
                for j,k in enumerate(tr['candidate_indices']):
                    h=tr['inverse'][j];t=tr['trace'];fb=g['rounds'][1]['feedback'][k]['anchors'][i]
                    assert np.array_equal(t['initial_reference'][h],expected[k,i].astype(np.float32))and np.array_equal(t['boxes'][-1,h],fb['box'])
                    assert fb['accepted']==(float(t['scores'][-1,h])>=.35)and int(t['content_slots'][h])==int(t['mask_slots'][h]);checks['actual_live_decoder_feedback']+=1
        # Independent official metric scalars, no reuse of scorer geometry helpers.
        row=plans[ds][q];gt,span=allgt[ds][0][q],allgt[ds][1][q];ids=np.asarray(o['frame_ids']);truthids=np.array(sorted(gt));truth=np.array([gt[int(i)]for i in truthids]);inside=(truthids>=max(span[0],o['interval'][0]))&(truthids<min(span[1],o['interval'][1]));den=max(max(span[1],o['interval'][1])-min(span[0],o['interval'][0]),1)
        for arm,bb in o['outputs'].items():
            xy=np.c_[bb[:,:2]-bb[:,2:]/2,bb[:,:2]+bb[:,2:]/2]*np.array([row['input']['width'],row['input']['height']]*2)
            if ds=='hc2':xy=np.maximum(xy,0)
            pred=np.column_stack([np.interp(truthids,ids,xy[:,j])for j in range(4)]);inter=np.maximum(0,np.minimum(pred[:,2:],truth[:,2:])-np.maximum(pred[:,:2],truth[:,:2])).prod(1)
            union=np.maximum(0,pred[:,2:]-pred[:,:2]).prod(1)+np.maximum(0,truth[:,2:]-truth[:,:2]).prod(1)-inter;iou=np.divide(inter,union,out=np.zeros_like(inter),where=union>0);iou[(truthids<ids[0])|(truthids>ids[-1])]=0
            tests=dict(v=float(iou[inside].sum()/den),s=float(iou.mean()))
            for key,mask in [('observed_actual_s',np.isin(truthids,ids[pos])),('unobserved_actual_s',np.isin(truthids,ids)&~np.isin(truthids,ids[pos])),('benchmark_interpolated_s',~np.isin(truthids,ids))]:tests[key]=float(iou[mask].mean())if mask.any()else None
            for k,v in tests.items():
                got=rows[a]['values'][arm][k];assert got is None if v is None else abs(got-v)<2e-10;checks['independent_metric_scalars']+=1
    verify(full=True);put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=32,checks=dict(checks),all32_before_GT=True,only_QR256=True,MBR_fixed=True,initial_E_exact=True,first_round_matched=True,second_round_actual_teacher_refresh=True))
    put(D/'ROOT_AUDIT_COMPLETION.json',dict(status='pass',count=32));print(dict(checks),flush=True)
if __name__=='__main__':main()
