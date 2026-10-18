"""Saved scalar/tensor accounting only: no model, media, annotations or scoring."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.score_stvg_joint_robust_correction32_v1 import paired
D=ROOT/'artifacts/stvg_dino_local_hc_diagnosis32_v1'
P=ROOT/'results/stvg_error_state/2026-10-18/dino_local_hc_diagnosis32'
PRE=ROOT/'artifacts/stvg_dino_mask_align32_v1'
LOCAL=ROOT/'artifacts/stvg_dino_local_residual32_v1'

def mean(v):
    v=[float(x)for x in v if x is not None];return float(np.mean(v))if v else None

def public(x):
    if isinstance(x,dict):return {k:public(v)for k,v in x.items()}
    if isinstance(x,list):return [public(v)for v in x]
    return x.item()if isinstance(x,np.generic)else x

def norm(x):return float(np.linalg.norm(x))
def cos(a,b):return float(np.dot(a,b)/(norm(a)*norm(b)))if norm(a)>1e-14 and norm(b)>1e-14 else None
def z(b):
    b=np.asarray(b,float);return np.concatenate([b[...,:2],np.log(b[...,2:])],axis=-1)
def box(v):return np.concatenate([v[...,:2],np.exp(v[...,2:])],axis=-1)

def prepare():
    old=read(LOCAL/'PLAN_LOCK.json');assert read(LOCAL/'FINAL_COMPLETION.json')['status']=='complete'
    files=['scripts/analyze_stvg_dino_local_hc_diagnosis32_v1.py','protocols/stvg_dino_local_hc_diagnosis32_v1.md',
        'methods/CURRENT_METHOD.json',str((PRE/'SCORED_FEEDBACK_PRIVATE.pt').relative_to(ROOT)),
        'results/stvg_error_state/2026-10-18/dino_mask_align32/RESULTS_PUBLIC.json',
        'results/stvg_error_state/2026-10-18/dino_local_residual32/RESULTS_PUBLIC.json']
    for r in old['records']:
        files += [str((p/'predictions'/(r['alias']+'.pt')).relative_to(ROOT))for p in [PRE,LOCAL]]
    put(D/'PLAN_LOCK.json',dict(status='locked',records=old['records'],count=32,per_domain=16,
        fixed_denominator=24,no_new_models_GT_scoring_or_predictions=True,no_amplitude_search=True,
        cohort_already_exposed=True,questions=['teacher_vs_E','accepted_sample_correction_aggregation','applied_displacement'],
        private_original_inputs_preserved=True))
    put(D/'RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f)for f in files},plan_sha256=sha(D/'PLAN_LOCK.json')))

def verify():
    x=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==x['plan_sha256']
    for f,h in x['pins'].items():assert sha(ROOT/f)==h,f
    return read(D/'PLAN_LOCK.json')

def run():
    p=verify();start=time.perf_counter();rows=[];checks=collections.Counter();error=0.
    scored={r['alias']:r['rows']for r in load(PRE/'SCORED_FEEDBACK_PRIVATE.pt')}
    result=read(ROOT/'results/stvg_error_state/2026-10-18/dino_local_residual32/RESULTS_PUBLIC.json')
    results={r['alias']:r for r in result['rows']}
    prior={r['alias']:r for r in read(ROOT/'results/stvg_error_state/2026-10-18/dino_mask_align32/RESULTS_PUBLIC.json')['rows']}
    for rec in p['records']:
        a=rec['alias'];old=load(PRE/'predictions'/(a+'.pt'));local=load(LOCAL/'predictions'/(a+'.pt'));src=z(old['outputs']['Source_MBR']);ez=z(old['outputs']['E_MBR']);scale=old['scale']
        saved=[r for r in scored[a]if r['mode']=='B_aligned'];nodes=[];quality=[]
        for j,pos in enumerate(old['planned_positions']):
            qq=[v for v in saved if v['position']==pos];known=bool(qq)
            if known:assert len(qq)==24;checks['known_node_24_score_sets']+=1
            accepted=[];teacher=[];noise=[];corrections=[]
            allnoise=np.array([z(box(src+ac*scale))[pos]-ez[pos]for ac in old['actions']])if old['active']else np.empty((0,4))
            for k,d in enumerate(old.get('matched',{}).get('B_aligned',{}).get('details',[])):
                v=d['anchors'].get(pos)
                if v and v['accepted']:
                    assert v['score']>=.35 and d['legal'];t=z(v['box'])-ez[pos];n=allnoise[k]
                    teacher.append(t);noise.append(n);corrections.append(t-n);accepted.append(k)
                    if known:assert qq[k]['accepted'] and qq[k]['sample']==k;checks['accepted_saved_GT_links']+=1
            n=len(accepted);coverage=n/24;tt=np.sum(teacher,axis=0)/24 if n else np.zeros(4);nn=np.sum(noise,axis=0)/24 if n else np.zeros(4)
            rr=np.sum(corrections,axis=0)/24 if n else np.zeros(4)
            expected=local['residual'][j];err=max(float(abs(rr-expected).max()),float(abs(rr-(tt-nn)).max()));error=max(error,err);assert err<2e-14
            if n:
                sampling=coverage*allnoise.mean(0);selection=nn-sampling
                cs=np.asarray(corrections)/scale;unit_r=rr/scale;unit_t=tt/scale;unit_n=nn/scale
                loo=unit_r[None]-cs/24
                stability=dict(max_leave_one_change_norm=float(np.linalg.norm(cs/24,axis=1).max()),
                    max_leave_one_change_over_actual=norm(cs[np.argmax(np.linalg.norm(cs,axis=1))]/24)/norm(unit_r)if norm(unit_r)>1e-14 else None,
                    min_leave_one_cosine=min([v for v in [cos(v,unit_r)for v in loo]if v is not None],default=None),
                    leave_one_opposite_direction=sum(norm(v)>1e-14 and np.dot(v,unit_r)<-1e-12*norm(v)*norm(unit_r) for v in loo))
                coherence=norm(cs.sum(0))/np.linalg.norm(cs,axis=1).sum()if np.linalg.norm(cs,axis=1).sum()>0 else None
                checks['accepted_residuals']+=n
            else:
                sampling=selection=np.zeros(4);unit_r=unit_t=unit_n=np.zeros(4);stability={};coherence=None
            io=dict(teacher_minus_input_gated_pp=None,teacher_minus_E_gated_pp=None,E_minus_input_gated_pp=None,
                accepted_teacher_better_than_E=0,accepted_teacher_worse_than_E=0,accepted_teacher_equal_E=0)
            if known:
                def score_delta(k1,k2):return float(np.mean([100*(v[k1]-v[k2])if v['accepted']else 0. for v in qq]))
                io.update(teacher_minus_input_gated_pp=score_delta('teacher','input'),teacher_minus_E_gated_pp=score_delta('teacher','E'),E_minus_input_gated_pp=score_delta('E','input'))
                v=[x['teacher']-x['E']for x in qq if x['accepted']]
                io.update(accepted_teacher_better_than_E=sum(x>1e-10 for x in v),accepted_teacher_worse_than_E=sum(x< -1e-10 for x in v),accepted_teacher_equal_E=sum(abs(x)<=1e-10 for x in v))
                assert abs(io['teacher_minus_input_gated_pp']-io['teacher_minus_E_gated_pp']-io['E_minus_input_gated_pp'])<1e-12
                quality.append(io);checks['teacher_scalar_identity']+=1
            applied=z(local['outputs']['Independent4_observed'][pos])-ez[pos]
            assert np.max(abs(applied-rr))<1e-14;checks['applied_residual_exact']+=1
            nodes.append(dict(slot=j,saved_GT_quality_available=known,accepted=n,coverage=coverage,**io,
                applied_source_scaled_norm=norm(unit_r),teacher_term_source_scaled_norm=norm(unit_t),input_term_source_scaled_norm=norm(unit_n),
                all24_sampling_mean_component_norm=norm(sampling/scale),admission_contrast_component_norm=norm(selection/scale),
                input_term_larger_than_teacher=norm(unit_n)>norm(unit_t)+1e-14,
                applied_teacher_term_cosine=cos(unit_r,unit_t),correction_coherence=coherence,
                actual_center_relative_E_size=norm(rr[:2]/old['outputs']['E_MBR'][pos,2:]),
                actual_center_image_norm=norm(rr[:2]),actual_log_size_norm=norm(rr[2:]),
                normalized_center_squared_fraction=float((unit_r[:2]**2).sum()/max((unit_r**2).sum(),1e-40)),
                accepted_input_center_mean_norm=norm(nn[:2]),accepted_input_log_size_mean_norm=norm(nn[2:]),
                **stability))
        r=results[a];q={k:mean([v[k]for v in quality])for k in ['teacher_minus_input_gated_pp','teacher_minus_E_gated_pp','E_minus_input_gated_pp']}
        for key,oldkey in [('teacher_minus_input_gated_pp','gated_teacher_minus_sample_pp'),('teacher_minus_E_gated_pp','gated_teacher_minus_E_pp')]:
            if q[key]is not None:assert abs(q[key]-prior[a]['accounting']['B_aligned'][oldkey])<1e-12;checks['prior_query_quality_exact']+=1
        outcome=dict(delta_v_pp=100*(r['values']['Independent4_observed']['v']-r['values']['E_MBR']['v']),
            delta_observed_s_pp=None if r['values']['Independent4_observed']['observed_actual_s']is None else 100*(r['values']['Independent4_observed']['observed_actual_s']-r['values']['E_MBR']['observed_actual_s']),
            interpolation_v_contribution_pp=r['interpolation_effects']['Independent4_observed']['benchmark_interpolated']['v_contribution_pp'],
            observed_v_contribution_pp=r['interpolation_effects']['Independent4_observed']['observed_actual']['v_contribution_pp'])
        feedback=[v for v in nodes if v['accepted']]
        rows.append(dict(alias=a,dataset=rec['dataset'],active=old['active'],nodes=nodes,quality=q,outcome=outcome,
            finite_set=dict(nodes_with_feedback=len(feedback),known_nodes_with_feedback=sum(v['saved_GT_quality_available']for v in feedback),
                mean_source_scaled_applied_norm=mean([v['applied_source_scaled_norm']for v in nodes]),
                mean_coherence=mean([v['correction_coherence']for v in feedback]),
                mean_input_term_norm=mean([v['input_term_source_scaled_norm']for v in nodes]),
                mean_teacher_term_norm=mean([v['teacher_term_source_scaled_norm']for v in nodes]))))
    groups={}
    for ds in ['hc2','vidstg']:
        rs=[r for r in rows if r['dataset']==ds];ns=[n for r in rs for n in r['nodes']];fb=[n for n in ns if n['accepted']];known=[n for n in fb if n['saved_GT_quality_available']]
        groups[ds]=dict(queries=16,planned_nodes=64,nodes_with_feedback=len(fb),known_nodes_with_feedback=len(known),
           unknown_nodes_with_feedback=len(fb)-len(known),accepted_total=sum(n['accepted']for n in ns),
           accepted_known_GT=sum(n['accepted']for n in known),accepted_teacher_better_E=sum(n['accepted_teacher_better_than_E']for n in ns),
           accepted_teacher_worse_E=sum(n['accepted_teacher_worse_than_E']for n in ns),accepted_teacher_equal_E=sum(n['accepted_teacher_equal_E']for n in ns),
           support_histogram={str(k):sum(n['accepted']==k for n in ns)for k in sorted(set(n['accepted']for n in ns))},
           input_norm_larger_than_teacher=sum(n['input_term_larger_than_teacher']for n in fb),
           applied_opposite_teacher=sum(n['applied_teacher_term_cosine']is not None and n['applied_teacher_term_cosine']<0 for n in fb),
           admission_contrast_larger_than_all24_mean=sum(n['admission_contrast_component_norm']>n['all24_sampling_mean_component_norm']for n in fb),
           mean_coherence_query_mean=mean([r['finite_set']['mean_coherence']for r in rs]),
           mean_input_term_norm_query_mean=mean([r['finite_set']['mean_input_term_norm']for r in rs]),
           mean_teacher_term_norm_query_mean=mean([r['finite_set']['mean_teacher_term_norm']for r in rs]),
           max_single_term_fraction_over_one=sum(n.get('max_leave_one_change_over_actual',0)is not None and n.get('max_leave_one_change_over_actual',0)>1+1e-12 for n in fb),
           leave_one_can_reverse_direction=sum(n.get('leave_one_opposite_direction',0)>0 for n in fb),
           quality={k:paired([r['quality'][k]for r in rs])for k in rs[0]['quality']},
           outcomes={k:paired([r['outcome'][k]for r in rs])for k in ['delta_v_pp','delta_observed_s_pp']})
    out=dict(status='complete',count=32,rows=rows,groups=groups,checks=dict(checks),max_residual_error=error,
        compute=dict(CPU_seconds=time.perf_counter()-start,GPU_seconds=0,new_model_calls=0,new_GT_reads=0,new_GT_scoring=0,new_predictions=0),
        causal_limits=['no amplitude intervention','no new GT derivative','quality versus geometry not identity','finite gate contrast not population causal bias'],
        no_dynamic_OPD=True,independent_confirmation=False)
    verify();put(P/'RESULTS_PUBLIC.json',public(out));put(D/'STATUS.json',dict(status='analysis_complete',count=32,GT_read=False),True)
    print({ds:g for ds,g in groups.items()},flush=True)

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:run()
