"""Independent sealed state/Adam/acceptance and metric readback."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_query_e_response64_v1 import D,P,OLD,verify


def same(a,b):
    import torch
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return set(a)==set(b)and all(same(a[k],b[k])for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b)and all(same(x,y)for x,y in zip(a,b))
    return a==b


def main():
    lock=read(D/'ANALYSIS_IMPLEMENTATION_LOCK.json')
    for f,h in lock['pins'].items():assert sha(ROOT/f)==h,f
    import torch
    from scripts.score_stvg_opd_p1_v1 import truths
    from vg_tta.tastvg_oracle_event5_v1 import official,box_iou
    torch.set_num_threads(4);p=verify();result=read(P/'RESULTS.json');assert sha(P/'RESULTS.json')==read(D/'CPU_COMPLETION.json')['result_sha256']
    targets={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']}
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in targets}
    counts=collections.Counter();maxloss=0.
    for rec,res in zip(p['records'],result['rows']):
        assert rec['alias']==res['alias'];ds=rec['dataset'];q=rec['query_ordinal'];row=plans[ds][q];gt,span=targets[ds][0][q],targets[ds][1][q]
        z=load(D/'predictions'/(rec['alias']+'.pt'));old=load(OLD/'predictions'/(rec['alias']+'.pt'));teacher=old['E_training_teacher_float32'].double().numpy()
        ids=np.array(z['frame_ids']);umask=np.array([int(v)in gt for v in ids])&~np.isin(np.arange(len(ids)),z['planned_positions'])
        def check(box,reported):
            b=box.double().numpy();m=official(b,row,gt,span,z['interval'],ds)
            for k in ['v','t','s']:assert abs(m[k]-reported[k])<2e-10;counts['official_metrics']+=1
            xy=np.c_[b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2];ty=np.c_[teacher[:,:2]-teacher[:,2:]/2,teacher[:,:2]+teacher[:,2:]/2]
            assert abs(float(box_iou(xy,ty).mean())-reported['E_IoU_all'])<1e-12;counts['independent_E_geometry']+=1
            if umask.any():
                physical=xy*np.array([row['input']['width'],row['input']['height']]*2)
                if ds=='hc2':physical=np.maximum(physical,0)
                uu=box_iou(physical[umask],np.array([gt[int(t)]for t in ids[umask]])).mean()
                assert abs(uu-reported['unobserved_actual_s'])<1e-12;counts['unobserved_exact_GT_support']+=1
            else:assert reported['unobserved_actual_s']is None;counts['unknown_unobserved']+=1
        for l,r in zip(z['local'],res['local']):
            check(l['before']['boxes'],r['before'])
            assert l['round']in p['local_rounds'][ds]if l['primary']else rec['alias']=='H008'
            for name,v in l['variants'].items():
                check(v['boxes'],r['variants'][name]['values']);counts['local_variants']+=1
            if l['updated']:
                norms={n:float(v['delta'].norm())for n,v in l['variants'].items()};ref=norms['detach_history_full']
                for n,v in norms.items():assert abs(v-ref*(.5 if n.endswith('half')else 1))<1e-6*(1+ref);counts['matched_QR_norms']+=1
                f=old['fits'][l['arm']];k=l['round'];assert torch.equal(l['variants']['detach_history_full']['boxes'],f['path'][k+1]['boxes']);counts['original_next_output_exact']+=1
                assert torch.equal(l['gradients'][False],f['path'][k]['update']['gradient']);counts['original_gradient_exact']+=1
        for arm,f in z['fits'].items():
            rr=res['fits'][arm]
            assert len(f['rounds'])==p['configs'][ds]['steps'];initial=old['fits']['Dense_OPD']['initial']
            for st,m in zip(f['path'],rr['states']):
                check(st['boxes'],m)
                for n,v in st['state'].items():
                    if n!='spatial.query_residual':assert torch.equal(v,initial[n]);counts['frozen_LN_tensors']+=1
            for k,r in enumerate(f['rounds']):
                a,b=f['path'][k],f['path'][k+1];u0=a['state']['spatial.query_residual'];u1=b['state']['spatial.query_residual']
                assert len(r['trials'])<=4
                for j,tr in enumerate(r['trials']):
                    assert tr['scale']==[1.,.5,.25,.125][j]
                    assert tr['accepted']==(tr['loss']<=r['loss_before']+tr['tolerance'])
                    assert tr['tolerance']==1e-7*(1+abs(r['loss_before']));counts['trial_acceptance']+=1
                    if j<len(r['trials'])-1:assert not tr['accepted']
                if not r['updated']:
                    assert torch.equal(u0,u1)
                    assert torch.equal(a['boxes'],b['boxes'])
                    if r.get('rejected_state_restored'):
                        assert same(r['optimizer_before'],r['optimizer_after']);counts['full_rejection_Adam_QR_restore']+=1
                else:
                    scale=r['accepted_scale'];delta=a['proposal']['raw'];err=float((u1-u0-scale*delta).abs().max());assert err<2e-7
                    assert r['trials'][-1]['accepted']and scale==r['trials'][-1]['scale'];counts['first_accepted_and_QR_step']+=1
                    # Independent NumPy Adam moment recurrence from saved gradient.
                    g=a['proposal']['gradient'].numpy();prev=r['optimizer_before']['state'];aft=r['optimizer_after']['state'][0]
                    pm=np.zeros(256,np.float32)if not prev else prev[0]['exp_avg'].numpy();pv=np.zeros(256,np.float32)if not prev else prev[0]['exp_avg_sq'].numpy()
                    assert np.allclose(aft['exp_avg'].numpy(),.9*pm+.1*g,rtol=2e-5,atol=2e-7)
                    assert np.allclose(aft['exp_avg_sq'].numpy(),.999*pv+.001*g*g,rtol=2e-5,atol=2e-7);counts['independent_Adam_moments']+=1
                    support=f['support'];boxes=b['boxes'][support]
                    if ((boxes>0)&(boxes<1)).all():
                        mu=torch.logit(boxes).double().numpy();roll=r['rollout'];origin=roll['mean'].double().numpy();samples=roll['samples'].double().numpy();weights=roll['weights'].double().numpy();sigma=p['configs'][ds]['sigma']
                        loss=(((mu-origin)**2).sum(-1)/(2*sigma*sigma)-((weights-1/32)*(-((samples-mu[:,None])**2).sum(-1)/(2*sigma*sigma)-2*np.log(2*np.pi*sigma*sigma))).sum(-1)).mean()
                        err=abs(float(loss)-r['trials'][-1]['loss']);assert err<2e-5*(1+abs(loss));maxloss=max(maxloss,err);counts['accepted_actual_next_decoder_surrogate']+=1
                    else:counts['endpoint_surrogate_independent_unknown']+=1
    for ds,g in result['summary'].items():
        rr=[r for r in result['rows']if r['dataset']==ds];assert len(rr)==32
        for a,vals in g['means'].items():
            for ep,v in vals.items():
                x=[r['values'][a][ep]for r in rr if r['values'][a][ep]is not None]
                assert v is None if not x else abs(sum(x)/len(x)-v)<1e-12;counts['domain_means']+=1
        for name,vals in g['paired'].items():
            a,b=name.split('-')
            for ep,v in vals.items():
                x=[100*(r['values'][a][ep]-r['values'][b][ep])for r in rr if r['values'][a][ep]is not None and r['values'][b][ep]is not None]
                assert len(x)==v['n'];assert not x or abs(sum(x)/len(x)-v['mean_pp'])<1e-12;counts['paired_parent_means']+=1
    assert counts['local_variants']==194*8;assert not torch.cuda.is_initialized();verify()
    put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=64,checks=dict(counts),max_accepted_surrogate_difference=maxloss,
        rejected_trial_scalar_checks_not_independent_decoder_reruns=True,GT_used_only_post_seal=True,no_GT_selected_steps=True,
        original_predictions_source_and_vendor_code_pins_unchanged=True))
    put(D/'ROOT_READBACK_COMPLETION.json',dict(status='pass',count=64))
    print(dict(status='pass',checks=dict(counts)))


if __name__=='__main__':main()
