"""Independent roster, feedback, outputs and official-frame metric verification."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,collections,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_dino_endpoint_confirm32_v1 import D,P,OLD,EXCLUSIONS,verify
from vg_tta.stvg_dino_endpoint_confirm32_v1 import ARMS

def main():
    p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and bar['all_outputs_before_GT']
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    from scripts.score_stvg_opd_p1_v1 import truths
    import torch
    allgt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};torch.set_num_threads(4)
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in allgt}
    rr=read(P/'RESULTS_PUBLIC.json');rows={r['alias']:r for r in rr['rows']};counts=collections.Counter();maxerror=0.
    excluded={ds:set()for ds in plans}
    for f in EXCLUSIONS:
        doc=read(ROOT/f)
        for r in doc.get('records',doc.get('target',[])):
            ds='hc2'if r.get('dataset')in ['hc2','hcstvg2']else r.get('dataset')
            if ds in plans:excluded[ds].add(plans[ds][r['query_ordinal']]['input']['source'])
    design=read(OLD/'DESIGN_LOCK.json')
    for ds in plans:excluded[ds].update(plans[ds][q]['input']['source']for q in design['datasets'][ds]['development_query_ordinals'])
    def key(s):return hashlib.sha256(('endpoint_confirm32_v1|20261018|'+s).encode()).hexdigest()
    for ds,plan in plans.items():
        groups=collections.defaultdict(list)
        for q,row in enumerate(plan):groups[row['input']['source']].append(q)
        selected=sorted(set(groups)-excluded[ds],key=lambda par:key(ds+'|parent|'+par))[:16]
        rs=[r for r in p['records']if r['dataset']==ds];assert [r['parent']for r in rs]==selected
        assert not(set(selected)&excluded[ds])
        for r in rs:assert r['query_ordinal']==min(groups[r['parent']],key=lambda q:key(ds+'|query|'+r['parent']+'|'+str(q)))
        counts['roster_membership_and_query_selection']+=len(rs)
    def z(b):
        b=np.asarray(b,float);return np.concatenate([b[...,:2],np.log(b[...,2:])],axis=-1)
    def box(z):return np.concatenate([z[...,:2],np.exp(z[...,2:])],axis=-1)
    for r in p['records']:
        a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];row=plans[ds][q];out=load(D/'predictions'/(a+'.pt'));base=load(D/'E'/(a+'.pt'));source=load(D/'source'/(a+'.pt'))
        raw=torch.load(ROOT/r['prior_input'],map_location='cpu',weights_only=False)
        assert torch.equal(source['Source'],raw['native_boxes'])and source['Native_interval']==raw['interval'];counts['Source_exact_original_cache']+=1
        e=base['E'];pos=out['planned_positions'];expected={k:e.copy()for k in ARMS};sums=np.zeros((4,4));n=np.zeros(4,int);center=np.zeros(4,bool)
        if out['active']:
            scale=np.array(p['scale'][ds]);mu=np.array(base['estimate']['delta'])/scale
            rng=np.random.default_rng(p['seed']+int(hashlib.sha256(a.encode()).hexdigest()[:8],16));assert np.array_equal(out['actions'],mu+rng.standard_normal((24,4)));counts['exact_sampling_arrays']+=1
            candidates=np.stack([box(z(base['Source'])+v*scale)for v in out['actions']]);legal=np.isfinite(candidates[:,pos]).all((1,2))&((candidates[:,pos]>0)&(candidates[:,pos]<1)).all((1,2))
            for k,d in enumerate(out['sample_feedback']['details']):
                assert d['legal']==bool(legal[k]);counts['whole_sample_legality']+=1
                for j,i in enumerate(pos):
                    v=d['anchors'].get(i)
                    if v:
                        assert v['accepted']==bool(v['score']>=.35)and v['slot']==v['mask_slot']==out['native'][i]['high_slot'];counts['sample_admission_and_slot']+=1
                        if v['accepted']:sums[j]+=z(v['box'])-z(e[i]);n[j]+=1
            assert out['center_feedback']['legal']==bool(np.isfinite(e[pos]).all()and ((e[pos]>0)&(e[pos]<1)).all())
            for j,i in enumerate(pos):
                v=out['center_feedback']['anchors'].get(i)
                if v:
                    assert v['accepted']==bool(v['score']>=.35)and v['slot']==v['mask_slot']==out['native'][i]['high_slot'];counts['center_admission_and_slot']+=1
                    assert np.array_equal(out['traces']['center'][i]['initial_reference'],e[i:i+1].astype(np.float32))
                    if v['accepted']:
                        center[j]=True;expected['Center_full_local'][i]=np.asarray(v['box'],float)
                        if n[j]>0:expected['Center_p_local'][i]=box(z(e[i])+n[j]/24*(z(v['box'])-z(e[i])))
                if np.any(sums[j]!=0):expected['Samples24_endpoint'][i]=box(z(e[i])+sums[j]/24)
        assert np.array_equal(n,out['accepted_counts'])and np.array_equal(center,out['center_accepted'])and np.array_equal(out['p'],n/24)
        assert np.array_equal(out['common_admitted'],center&(n>0));counts['matched_p_and_common_mask']+=1
        unobs=~np.isin(np.arange(len(e)),pos)
        for arm,b in expected.items():
            err=float(abs(b-out['outputs'][arm]).max());maxerror=max(maxerror,err);assert err<1e-12
            assert np.array_equal(out['outputs'][arm][unobs],e[unobs]);counts['independent_output_and_unobserved_arrays']+=1
        ids=np.asarray(out['frame_ids'])
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
            common_ids=ids[np.asarray(out['planned_positions'])[np.asarray(out['common_admitted'])]]
            common_mask=np.isin(truthids,common_ids)
            tests['common_admitted_s']=float(ii[common_mask].mean())if common_mask.any()else None
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
        rs=[r for r in rr['rows']if r['dataset']==ds]
        for contrast,stats in g['paired'].items():
            a,b=contrast.split('_minus_')
            for ep,value in stats.items():
                v=[None if r['values'][a][ep]is None or r['values'][b][ep]is None else 100*(r['values'][a][ep]-r['values'][b][ep])for r in rs]
                assert paired(v)==value;counts['paired_summaries']+=1
    assert bar['GPU_total_allocation_seconds']<p['GPU_seconds_cap'];assert sum(g['cost']['encoding_frames']for g in rr['groups'].values())<=128
    for ds in ['hc2','vidstg']:
        quals=[v for r in p['records']if r['dataset']==ds for v in load(D/'predictions'/(r['alias']+'.pt'))['qualifications']]
        if any(r['active']for r in rr['rows']if r['dataset']==ds):assert len(quals)==4 and all(v['status']=='pass'for v in quals)
        counts['first_active_domain_qualifications']+=len(quals)
    verify(full=True)
    put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=32,checks=dict(counts),maximum_array_error=maxerror,
        roster_no_known_selection_parent_overlap=True,all32_sealed_before_GT=True,admissions_and_p_exact=True,
        pretrained_weights_frozen=True,no_parameter_search=True,no_dynamic_OPD=True))
    put(D/'ROOT_AUDIT_COMPLETION.json',dict(status='pass',count=32))
    print(dict(counts),flush=True)

if __name__=='__main__':main()
