"""Independent assignment-cost enumeration, cache reconstruction and metrics audit.

Does not call the estimator or output-correction functions. This is a separate
mathematical implementation, not an independent human semantic annotation.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,itertools,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_joint_robust_correction32_v1 import D,P,OLD,ERROR,CRITIC,verify


def chart(b):
    b=np.asarray(b,float);return np.concatenate((b[:,:2],np.log(b[:,2:])),axis=1)


def brute(pools,scale,cutoff):
    # Independent global minimum of UNTRUNCATED conditional-assignment LS +
    # explicit lambda for every rejected frame, rather than rescoring proposals.
    best=float('inf');count=0
    for assign in itertools.product(*[range(-1,len(p))for p in pools]):
        active=np.array([p[k]for p,k in zip(pools,assign)if k>=0]).reshape(-1,4)
        d=active.mean(0)if len(active)else np.zeros(4)
        loss=float(np.sum(((active-d)/scale)**2)+(len(pools)-len(active))*cutoff)
        best=min(best,loss);count+=1
    return best,count


def close(x,y,tol=2e-10):
    if x is None or y is None:assert x is y;return
    assert np.max(np.abs(np.asarray(x)-np.asarray(y)),initial=0)<tol,(x,y)


def main():
    import torch
    from scripts.score_stvg_opd_p1_v1 import truths
    from vg_tta.tastvg_oracle_event5_v1 import DenseTube
    torch.set_num_threads(4);p=verify(inputs=True);cal=read(D/'SOURCE_CALIBRATION.json')
    bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');result=read(P/'RESULTS.json')
    assert len(result['rows'])==bar['count']==32 and bar['all_target_outputs_before_new_target_GT']
    assert sha(D/'SOURCE_CALIBRATION.json')==bar['source_calibration_sha256']
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    oldbind=read(ROOT/'artifacts/stvg_error_state_hc_diagnosis32_v1/RUNTIME_LOCK.json')
    reviewpath='artifacts/stvg_dino_error_attribution32_v1/VISUAL_REVIEW.json'
    assert sha(ROOT/reviewpath)==oldbind['inputs'][reviewpath]
    errors=[];sourcechecks=0;calculated={ds:[]for ds in ['hc2','vidstg']}
    for r in p['source']:
        z=load(CRITIC/'source'/(r['alias']+'.pt'));y=load(ROOT/'artifacts/stvg_joint_posterior_v1/source_supervision'/(r['alias']+'.pt'))
        pos=[i for i in z['expert']['positions4']if z['frame_ids'][i]in y['dense_source_GT']]
        if len(pos)<2:continue
        xy=np.array([y['dense_source_GT'][z['frame_ids'][i]]for i in pos],float)
        xy/=np.array([z['input']['width'],z['input']['height']]*2)
        b=np.concatenate(((xy[:,:2]+xy[:,2:])*.5,xy[:,2:]-xy[:,:2]),axis=1)
        res=chart(b)-chart(np.asarray(z['boxes'][0],float))[pos];calculated[r['dataset']].append(res-res.mean(0));sourcechecks+=1
    for ds,centered in calculated.items():
        scale=np.maximum(np.sqrt(sum((v*v).mean(0)for v in centered)/len(centered)),1e-8)
        v=np.concatenate([np.sum((r/scale)**2,axis=1)for r in centered]);w=np.concatenate([np.full(len(r),1/len(r))for r in centered])
        order=np.argsort(v,kind='stable');v=v[order];w=w[order]
        index=next(i for i,t in enumerate(np.cumsum(w))if t>=.95*w.sum())
        close(scale,cal['source_domains'][ds]['scale']);close(v[index],cal['source_domains'][ds]['truncation']);sourcechecks+=2
    truth={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']}
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in truth}
    rows={r['alias']:r for r in result['rows']};games=0;assignments=0;outputchecks=0;metricchecks=0;loochecks=0;scalarchecks=0;maxerr=0.
    for r in p['records']:
        z=load(D/'predictions'/(r['alias']+'.pt'));original=load(OLD/'predictions'/(r['alias']+'.pt'));old=load(ERROR/'dino_full'/(r['alias']+'.pt'))
        sd=cal['source_domains'][cal['target_to_source'][r['dataset']]];sc=np.array(sd['scale']);tau=sd['truncation'];saved=rows[r['alias']]
        assert len(z['frames'])==len(original['expert']['anchors']['single4'])
        for f,a in zip(z['frames'],original['expert']['anchors']['single4']):
            op=original['expert']['observations'][('original',a['position'])]['probe']
            assert f['position']==a['position']and np.array_equal(f['boxes'],np.asarray(op['boxes']))
            assert np.array_equal(f['boxes'][f['top1']],a['box'])
        row=plans[r['dataset']][r['query_ordinal']];gt,span=truth[r['dataset']][0][r['query_ordinal']],truth[r['dataset']][1][r['query_ordinal']]
        ids=np.array(z['frame_ids']);known=np.array([int(i)in gt for i in ids]);unobs=known&~np.isin(np.arange(len(ids)),z['planned_positions'])
        scale=np.array([row['input']['width'],row['input']['height']]*2)
        for base,data in z['bases'].items():
            b=data['Base'];pools=[chart(f['boxes'])-chart(b)[f['position']]for f in z['frames']]
            top=[v[[f['top1']]]for v,f in zip(pools,z['frames'])]
            for name,pool,est in [('B_robust_top1',top,data['estimate_B']),('C_joint',pools,data['estimate_C'])]:
                cost,num=brute(pool,sc,tau);close(cost,est['objective'],1e-8);games+=1;assignments+=num
                active=[v[k]for v,k in zip(pool,est['selected'])if k>=0]
                delta=np.mean(active,axis=0)if active else np.zeros(4);close(delta,est['delta']);games+=1
                assert len(active)==est['support_frames']
                for i in range(len(pool)):
                    # Check returned leave-one solution has the exact independent
                    # minimum; its recorded distance is independently reconstructed
                    # with a direct finite candidate scan, not by calling estimate.
                    sub=pool[:i]+pool[i+1:];minimum,_=brute(sub,sc,tau)
                    proposals=[]
                    for assignment in itertools.product(*[range(-1,len(v))for v in sub]):
                        sel=[v[k]for v,k in zip(sub,assignment)if k>=0];dd=np.mean(sel,axis=0)if sel else np.zeros(4)
                        dists=[np.square((v-dd)/sc).sum(1)for v in sub]
                        choice=tuple(int(v.argmin())if len(v)and v.min()<tau else -1 for v in dists)
                        obj=sum(min(tau,float(v.min()))if len(v)else tau for v in dists)
                        proposals.append((obj,choice,assignment,dd))
                    best=min(proposals,key=lambda v:(v[0],v[1],v[2]));close(best[0],minimum,1e-8)
                    close(np.linalg.norm((best[3]-est['delta'])/sc),est['leave_one_delta_distances'][i],1e-7);loochecks+=2
            for arm in ['A_mean','B_robust_top1','C_joint','D_C_center']:
                delta=data['delta_A']if arm=='A_mean'else data['estimate_B']['delta']if arm=='B_robust_top1'else data['estimate_C']['delta']if arm=='C_joint'else data['delta_D']
                c=chart(b)+delta;out=np.concatenate((c[:,:2],np.exp(c[:,2:])),axis=1)
                if np.array_equal(delta,np.zeros(4)):out=b.copy()
                if arm=='D_C_center':out[:,2:]=b[:,2:]
                err=float(np.max(abs(out-data['outputs'][arm]),initial=0));maxerr=max(maxerr,err);assert err<1e-12;outputchecks+=1
            assert np.array_equal(data['outputs']['A_mean'],old['corrections'][base]['global_error'])
            assert np.array_equal(data['outputs']['D_C_center'][:,:2],data['outputs']['C_joint'][:,:2])
            assert np.array_equal(data['outputs']['D_C_center'][:,2:],b[:,2:]);outputchecks+=3
            for arm,boxes in dict(Base=b,**data['outputs']).items():
                tube=DenseTube(boxes,row,gt,span,clip=r['dataset']=='hc2');met=tube.score(z['interval'])
                for k in ['v','t','s']:close(met[k],saved['values'][base][arm]['metrics'][k]);metricchecks+=1
                xy=np.concatenate((boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2),axis=1)*scale
                if r['dataset']=='hc2':xy=xy.clip(min=0)
                vals=[]
                for i in np.flatnonzero(unobs):
                    a=xy[i];g=np.asarray(gt[int(ids[i])],float)
                    wh=np.maximum(np.minimum(a[2:],g[2:])-np.maximum(a[:2],g[:2]),0);inter=float(wh.prod())
                    aa=float(np.maximum(a[2:]-a[:2],0).prod());ga=float(np.maximum(g[2:]-g[:2],0).prod())
                    vals.append(inter/(aa+ga-inter)if aa+ga-inter else 0.)
                close(np.mean(vals)if vals else None,saved['values'][base][arm]['supports']['unobserved']['s']);metricchecks+=1
        for base,gg in saved['geometry'].items():
            for item in gg:
                if not item['exact_GT_available']:assert item['candidate_IoUs']is None;continue
                vv=item['candidate_IoUs'];k=item['selected_index'];close(max(vv),item['best_retained_IoU']);close(vv[item['original_Top1_index']],item['original_Top1_IoU'])
                close(None if k<0 else vv[k],item['selected_IoU']);scalarchecks+=3
    # Re-evaluate saved-score bootstrap/CI, signs and leave-one means independently.
    for ds,bb in result['groups'].items():
        rr=[r for r in result['rows']if r['dataset']==ds]
        for base,group in bb.items():
            for contrast,eps in group['paired'].items():
                a,b=contrast.split('_minus_')
                for ep,pair in eps.items():
                    vals=[]
                    for r in rr:
                        x=r['values'][base][a];y=r['values'][base][b]
                        v=x['metrics'][ep]if ep in ['v','s']else x['supports'][ep]['s'];w=y['metrics'][ep]if ep in ['v','s']else y['supports'][ep]['s']
                        if v is not None and w is not None:vals.append((v-w)*100)
                    v=np.array(vals);close(v,pair['per_parent_pp']);close(v.mean(),pair['mean'])
                    rng=np.random.default_rng(20261018);idx=rng.integers(len(v),size=(10000,len(v)));close(np.quantile(v[idx].mean(1),[.025,.975]),pair['CI95'])
                    loo=(v.sum()-v)/(len(v)-1);close([loo.min(),loo.max()],pair['leave_one_saved_mean_range']);scalarchecks+=4
    verify(inputs=True);assert not torch.cuda.is_initialized()
    out=dict(status='pass',count=32,source_calibration_independent_checks=sourcechecks,
        independent_assignment_games=games,conditional_assignments_enumerated=assignments,
        leave_one_checks=loochecks,output_reconstructions_and_identity_checks=outputchecks,
        maximum_box_error=maxerr,independent_metric_and_actual_frame_IoU_checks=metricchecks,
        aggregate_CI_quality_checks=scalarchecks,all_original_pins_and_registries_unchanged=True,
        original_Top1_pool_exact=True,old_review_hash_matches_prior_binding=True,
        new_semantic_labels=0,independent_mathematical_implementation_not_human_labeling=True)
    put(P/'ROOT_INDEPENDENT_AUDIT.json',out);put(D/'ROOT_COMPLETION.json',out);print(out,flush=True)


if __name__=='__main__':main()
