"""Independently reconstruct finite-set accounting, using saved scores only."""
from pathlib import Path
import os,sys,collections
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.analyze_stvg_dino_local_hc_diagnosis32_v1 import D,P,PRE,LOCAL,verify

def main():
    plan=verify();x=read(P/'RESULTS_PUBLIC.json');rr={r['alias']:r for r in x['rows']};checks=collections.Counter();largest=0.;extra=[]
    private={r['alias']:r['rows']for r in load(PRE/'SCORED_FEEDBACK_PRIVATE.pt')}
    for rec in plan['records']:
        a=rec['alias'];old=load(PRE/'predictions'/(a+'.pt'));local=load(LOCAL/'predictions'/(a+'.pt'));stats=rr[a]
        source=np.asarray(old['outputs']['Source_MBR'],float);e=np.asarray(old['outputs']['E_MBR'],float);scale=np.asarray(old['scale'])
        sz=np.c_[source[:,:2],np.log(source[:,2:])];ez=np.c_[e[:,:2],np.log(e[:,2:])];variances=[]
        for j,i in enumerate(old['planned_positions']):
            ns=stats['nodes'][j];raw=[];inds=[];fback=[]
            for k,dd in enumerate(old.get('matched',{}).get('B_aligned',{}).get('details',[])):
                f=dd['anchors'].get(i)
                if f and f['accepted']:
                    bb=np.asarray(f['box'],float);raw.append(np.r_[bb[:2],np.log(bb[2:])]);inds.append(k);fback.append(f)
            assert ns['accepted']==len(inds)==local['accepted_counts'][j];checks['node_supports']+=1
            if not inds:
                assert np.array_equal(local['residual'][j],np.zeros(4));checks['exact_zero_feedback_nodes']+=1;continue
            t=(np.array(raw)-ez[i])/scale;n=(sz[i]+old['actions'][inds]*scale-ez[i])/scale;c=t-n
            actual=c.sum(0)/24
            err=float(np.max(abs(actual-local['residual'][j]/scale)));largest=max(largest,err);assert err<2e-13
            checks['source_scaled_residual_vectors']+=1
            alln=(sz[i]+old['actions']*scale-ez[i])/scale;sm=len(inds)/24*alln.mean(0);gate=n.sum(0)/24-sm
            vals=dict(applied_source_scaled_norm=np.linalg.norm(actual),teacher_term_source_scaled_norm=np.linalg.norm(t.sum(0)/24),
                input_term_source_scaled_norm=np.linalg.norm(n.sum(0)/24),all24_sampling_mean_component_norm=np.linalg.norm(sm),
                admission_contrast_component_norm=np.linalg.norm(gate),correction_coherence=np.linalg.norm(c.sum(0))/np.linalg.norm(c,axis=1).sum())
            for k,v in vals.items():assert abs(float(v)-ns[k])<2e-12;checks['independent_geometry_scalars']+=1
            s=[v for v in private[a]if v['mode']=='B_aligned'and v['position']==i]
            if s:
                assert len(s)==24
                for name,k1,k2 in [('teacher_minus_input_gated_pp','teacher','input'),('teacher_minus_E_gated_pp','teacher','E'),('E_minus_input_gated_pp','E','input')]:
                    value=sum(100*(v[k1]-v[k2])for v in s if v['accepted'])/24
                    assert abs(value-ns[name])<1e-12;checks['saved_GT_only_quality_scalars']+=1
            if len(inds)>=2:
                tc=t-t.mean(0);nc=n-n.mean(0);cc=c-c.mean(0);nenergy=float((nc**2).sum())
                variances.append(dict(slot=j,accepted=len(inds),teacher_input_dispersion_ratio=float(np.sqrt((tc**2).sum()/nenergy))if nenergy>0 else None,
                    correction_on_input_slope=float((cc*nc).sum()/nenergy)if nenergy>0 else None))
            updated=local['outputs']['Independent4_observed'][i];delta=np.r_[updated[:2]-e[i,:2],np.log(updated[2:])-np.log(e[i,2:])]
            assert abs(float(np.linalg.norm(delta[:2]/e[i,2:]))-ns['actual_center_relative_E_size'])<1e-14
            assert abs(float(np.linalg.norm(delta[2:]))-ns['actual_log_size_norm'])<1e-14;checks['independent_applied_amplitude']+=2
        extra.append(dict(alias=a,dataset=rec['dataset'],nodes=variances))
    for ds,g in x['groups'].items():
        rs=[r for r in x['rows']if r['dataset']==ds];nodes=[n for r in rs for n in r['nodes']if n['accepted']]
        assert sum(n['accepted']for n in nodes)==g['accepted_total'];checks['domain_coverage']+=1
        assert len(nodes)==g['nodes_with_feedback'];checks['domain_coverage']+=1
        from scripts.score_stvg_joint_robust_correction32_v1 import paired
        for key,s in g['quality'].items():assert paired([r['quality'][key]for r in rs])==s;checks['query_bootstrap']+=1
        for key,s in g['outcomes'].items():assert paired([r['outcome'][key]for r in rs])==s;checks['query_bootstrap']+=1
    summary={}
    for ds in ['hc2','vidstg']:
        vals=[n for r in extra if r['dataset']==ds for n in r['nodes']];ratio=[n['teacher_input_dispersion_ratio']for n in vals if n['teacher_input_dispersion_ratio']is not None]
        slope=[n['correction_on_input_slope']for n in vals if n['correction_on_input_slope']is not None]
        rs=[r for r in x['rows']if r['dataset']==ds];nodes=[n for r in rs for n in r['nodes']];fb=[n for n in nodes if n['accepted']]
        summary[ds]=dict(supported_nodes_with_multiple_samples=len(vals),median_teacher_input_dispersion_ratio=float(np.median(ratio)),
            mean_teacher_input_dispersion_ratio=float(np.mean(ratio)),median_correction_on_input_slope=float(np.median(slope)),
            median_accepted_per_supported_node=float(np.median([n['accepted']for n in fb])),
            mean_accepted_per_supported_node=float(np.mean([n['accepted']for n in fb])),
            mean_center_relative_E_size_all64=float(np.mean([n['actual_center_relative_E_size']for n in nodes])),
            max_center_relative_E_size=max(n['actual_center_relative_E_size']for n in nodes),max_log_size_norm=max(n['actual_log_size_norm']for n in nodes))
    verify();put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=32,checks=dict(checks),maximum_source_scaled_vector_error=largest,
        saved_scores_only=True,new_GT_reads=0,new_scoring=0,new_predictions=0,new_models=0,inputs_unchanged=True))
    put(P/'TEACHER_RESPONSE_DISPERSION.json',dict(status='complete',summary=summary,rows=extra,
        scope='within accepted same-node source-scaled coordinate dispersion, no GT or identity inference'))
    print(dict(checks=checks,summary=summary),flush=True)

if __name__=='__main__':main()
