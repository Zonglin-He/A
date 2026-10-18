"""Independent reconstruction of cached local updates and interpolation accounting."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_dino_local_residual32_v1 import D,P,PRE,verify,ARMS

def main():
    p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert len(bar['files'])==bar['count']==32 and bar['all_outputs_before_GT']
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    counts=collections.Counter();maxerror=0.;rr=read(P/'RESULTS_PUBLIC.json');rows={r['alias']:r for r in rr['rows']}
    from scripts.score_stvg_opd_p1_v1 import truths
    import torch
    allgt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};torch.set_num_threads(4)
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in allgt}
    # Independent z transforms, no use of construct() or scoring metrics().
    def z(b):
        b=np.asarray(b,float);return np.concatenate([b[...,:2],np.log(b[...,2:])],axis=-1)
    def b(z):return np.concatenate([z[...,:2],np.exp(z[...,2:])],axis=-1)
    for r in p['records']:
        a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];row=plans[ds][q]
        old=load(PRE/'predictions'/(a+'.pt'));out=load(D/'predictions'/(a+'.pt'));e=old['outputs']['E_MBR'];pos=old['planned_positions']
        unobs=~np.isin(np.arange(len(e)),pos);local=e.copy();sums=np.zeros((4,4));n=np.zeros(4,int)
        if old['active']:
            assert len(old['actions'])==len(old['matched']['B_aligned']['details'])==24
            for action,detail in zip(old['actions'],old['matched']['B_aligned']['details']):
                sampled=b(z(old['outputs']['Source_MBR'])+action*old['scale'])
                for j,i in enumerate(pos):
                    v=detail['anchors'].get(i)
                    if v and v['accepted']:
                        assert v['score']>=.35 and detail['legal'];sums[j]+=z(v['box'])-z(sampled[i]);n[j]+=1;counts['accepted_residuals']+=1
                counts['sampled_states']+=1
            for j,i in enumerate(pos):
                if np.any(sums[j]!=0):local[i]=b(z(e[i])+sums[j]/24)
        shared=e.copy();shared[pos]=old['outputs']['B_aligned_once'][pos]
        expected=[e,old['outputs']['B_aligned_once'],shared,local]
        for arm,want in zip(ARMS,expected):
            err=float(np.max(abs(want-out['outputs'][arm])));maxerror=max(maxerror,err);assert err<1e-12
            counts['complete_arrays']+=1
        assert np.array_equal(out['residual'],sums/24) and np.array_equal(out['accepted_counts'],n)
        assert out['interval']==old['interval'] and out['frame_ids']==old['frame_ids']
        for arm in ARMS[2:]:assert np.array_equal(out['outputs'][arm][unobs],e[unobs]);counts['exact_unobserved_arrays']+=1
        gt,span=allgt[ds][0][q],allgt[ds][1][q];truthids=np.array(sorted(gt));truth=np.asarray([gt[int(i)]for i in truthids]);ids=np.asarray(out['frame_ids'])
        observed=np.isin(truthids,ids[pos]);actual=np.isin(truthids,ids);inside=(truthids>=max(span[0],out['interval'][0]))&(truthids<min(span[1],out['interval'][1]))
        den=max(max(span[1],out['interval'][1])-min(span[0],out['interval'][0]),1);base=None
        for arm,boxes in out['outputs'].items():
            xy=np.column_stack([boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2])*np.array([row['input']['width'],row['input']['height']]*2)
            if ds=='hc2':xy=np.maximum(xy,0)
            pred=np.column_stack([np.interp(truthids,ids,xy[:,j])for j in range(4)])
            inter=np.maximum(0,np.minimum(pred[:,2:],truth[:,2:])-np.maximum(pred[:,:2],truth[:,:2])).prod(1)
            union=np.maximum(0,pred[:,2:]-pred[:,:2]).prod(1)+np.maximum(0,truth[:,2:]-truth[:,:2]).prod(1)-inter
            ii=np.divide(inter,union,out=np.zeros_like(inter),where=union>0);ii[(truthids<ids[0])|(truthids>ids[-1])]=0
            val=rows[a]['values'][arm]
            tests={'v':float(ii[inside].sum()/den),'s':float(ii.mean()),'benchmark_interpolated_s':float(ii[~actual].mean())if (~actual).any()else None}
            for name,mask in [('observed_actual_s',observed),('unobserved_actual_s',actual&~observed)]:
                tests[name]=float(ii[mask].mean())if mask.any()else None
            for k,v in tests.items():
                if v is None:assert val[k]is None
                else:assert abs(val[k]-v)<2e-10,(a,arm,k,val[k],v)
                counts['independent_metric_scalars']+=1
            if base is None:base=ii
            for scope,mask in [('observed_actual',observed),('unobserved_actual',actual&~observed),('benchmark_interpolated',~actual)]:
                change=100*(ii-base)[mask&inside].sum()/den
                assert abs(change-rows[a]['interpolation_effects'][arm][scope]['v_contribution_pp'])<1e-10;counts['v_contribution_scalars']+=1
    from scripts.score_stvg_joint_robust_correction32_v1 import paired
    for ds,g in rr['groups'].items():
        dsrows=[r for r in rr['rows']if r['dataset']==ds];assert len(dsrows)==16
        for contrast,stats in g['paired'].items():
            a,b=contrast.split('_minus_')
            for ep,value in stats.items():
                v=[None if r['values'][a][ep]is None or r['values'][b][ep]is None else 100*(r['values'][a][ep]-r['values'][b][ep])for r in dsrows]
                assert value==paired(v);counts['paired_summaries']+=1
    verify();put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=32,checks=dict(counts),maximum_array_error=maxerror,
        exact_prior_global_output=True,independent_residuals=True,full_denominator24=True,unknown_GT_not_failure=True,
        all32_sealed_before_new_GT=True,no_new_model_expert_features_training=True,no_dynamic_OPD=True))
    print(dict(counts),flush=True)

if __name__=='__main__':main()
