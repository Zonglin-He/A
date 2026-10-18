"""CPU-only existing tensor/scalar decomposition; never reads GT or scores new boxes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,collections,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.score_stvg_joint_robust_correction32_v1 import paired
D=ROOT/'artifacts/stvg_dino_endpoint_sampling_diagnosis32_v1'
P=ROOT/'results/stvg_error_state/2026-10-18/dino_sampling_diagnosis32'
PRE=ROOT/'artifacts/stvg_dino_endpoint_confirm32_v1'
OLD=ROOT/'results/stvg_error_state/2026-10-18/dino_endpoint_confirm32'
CODE=['scripts/analyze_stvg_dino_endpoint_sampling_diagnosis32_v1.py','scripts/audit_stvg_dino_endpoint_sampling_diagnosis32_v1.py','scripts/report_stvg_dino_endpoint_sampling_diagnosis32_v1.py','protocols/stvg_dino_endpoint_sampling_diagnosis32_v1.md']
ARMS=['E_MBR','Center_full_local','Samples24_endpoint','Center_p_local']
def mean(v):
    v=[float(x)for x in v if x is not None];return float(np.mean(v))if v else None
def norm(v):return float(np.linalg.norm(v))
def cosine(a,b):return float(np.dot(a,b)/(norm(a)*norm(b)))if norm(a)>1e-13 and norm(b)>1e-13 else None
def z(v):
    v=np.asarray(v,float);return np.concatenate([v[...,:2],np.log(v[...,2:])],axis=-1)
def iou(a,b):
    a=np.asarray(a);b=np.asarray(b);aa=np.r_[a[:2]-a[2:]/2,a[:2]+a[2:]/2];bb=np.r_[b[:2]-b[2:]/2,b[:2]+b[2:]/2]
    inter=np.maximum(0,np.minimum(aa[2:],bb[2:])-np.maximum(aa[:2],bb[:2])).prod();union=a[2:].prod()+b[2:].prod()-inter
    return float(inter/union)if union>0 else 0.
def public(v):
    if isinstance(v,dict):return {k:public(x)for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [public(x)for x in v]
    return v.item()if isinstance(v,np.generic)else v

def guard():
    # Runtime guard forbids data/model assets even if a helper were changed.
    def hook(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        name=os.fsdecode(args[0]);p=Path(name)
        if not p.is_absolute():p=Path.cwd()/p
        s=str(p)
        if any(v in s for v in ['/datasets/','/Downloads/','/weights/','/checkpoints/','/annotations/']) or p.suffix.lower()in ['.mp4','.avi','.jpg','.jpeg','.png','.webp','.safetensors','.pth']:
            raise PermissionError('cached scalar/tensor diagnosis forbids assets')
    sys.addaudithook(hook)

def prepare():
    assert read(PRE/'FINAL_COMPLETION.json')['status']=='complete';old=read(PRE/'PLAN_LOCK.json')
    files=CODE+['methods/CURRENT_METHOD.json',str((OLD/'RESULTS_PUBLIC.json').relative_to(ROOT)),str((PRE/'PLAN_LOCK.json').relative_to(ROOT)),str((PRE/'GLOBAL_PREDICTION_BARRIER.json').relative_to(ROOT))]
    files += [str((PRE/'predictions'/(r['alias']+'.pt')).relative_to(ROOT))for r in old['records']]
    put(D/'PLAN_LOCK.json',dict(status='locked',records=old['records'],count=32,per_domain=16,no_new_models_GT_media_weights_or_scoring=True,
        no_new_predictions=True,no_new_cohort=True,no_amplitude_or_threshold_search=True,no_dynamic_OPD=True,CPU_seconds_cap=600))
    put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins={f:sha(ROOT/f)for f in files}))
    put(D/'STATUS.json',dict(status='prepared',count=0,total=32),True)

def verify():
    p=read(D/'PLAN_LOCK.json');r=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==r['plan_sha256']
    for f,h in r['pins'].items():assert sha(ROOT/f)==h,f
    return p

def run():
    guard();p=verify();start=time.perf_counter();old=read(OLD/'RESULTS_PUBLIC.json');scores={r['alias']:r for r in old['rows']};rows=[];checks=collections.Counter();maxerr=0.
    for rec in p['records']:
        a=rec['alias'];o=load(PRE/'predictions'/(a+'.pt'));r=scores[a];ez=z(o['outputs']['E_MBR']);scale=np.asarray(o['scale']);ns=[]
        L=sum(v['legal']for v in o['sample_feedback']['details']);assert not o['active']or len(o['sample_feedback']['details'])==24
        for j,pos in enumerate(o['planned_positions']):
            E=o['outputs']['E_MBR'][pos];center=o['center_feedback']['anchors'].get(pos);M=bool(center and center['accepted']);accepted=[];inputs=[]
            tr=o['traces'].get('samples',{}).get(pos)
            for k,d in enumerate(o['sample_feedback']['details']):
                v=d['anchors'].get(pos)
                if v and v['accepted']:
                    accepted.append(z(v['box']));assert d['legal']and v['score']>=.35;checks['accepted_trace_endpoints']+=1
                    ids=list(tr['candidate_indices']);ii=ids.index(k);idx=tr['inverse'][ii];trace=tr['B_aligned']
                    assert np.array_equal(trace['boxes'][-1,idx],v['box']);assert float(trace['scores'][-1,idx])==v['score'];inputs.append(z(trace['initial_reference'][idx]));checks['trace_mappings']+=1
            n=len(accepted);pp=n/24;assert n==o['accepted_counts'][j]and pp==o['p'][j];checks['node_p_and_counts']+=1
            t=np.mean(accepted,axis=0)if n else ez[pos].copy();c=z(center['box'])if M else ez[pos].copy();uc=c-ez[pos];us=pp*(t-ez[pos]);up=pp*uc;amplitude=(pp-1)*uc;terminal=pp*(t-c)
            for arm,delta in [('Center_full_local',uc),('Samples24_endpoint',us),('Center_p_local',up)]:
                err=float(abs(z(o['outputs'][arm][pos])-ez[pos]-delta).max());maxerr=max(maxerr,err);assert err<1e-13;checks['saved_applied_vector_reconstruction']+=1
            err=float(abs(us-uc-amplitude-terminal).max());maxerr=max(maxerr,err);assert err<1e-13;checks['vector_decomposition']+=1
            assert not n or L>0
            condition=n/L if L else None;assert n==0 if L==0 else abs(pp-(L/24)*condition)<1e-15;checks['p_legal_times_admission']+=1
            case='common'if M and n else'center_only'if M else'sample_only'if n else'neither'
            dispersion=None
            if n>=2:
                tt=np.asarray(accepted)/scale;ii=np.asarray(inputs)/scale;te=tt-tt.mean(0);ie=ii-ii.mean(0);inputr=np.sqrt((ie**2).mean(0).sum());teacherr=np.sqrt((te**2).mean(0).sum())
                dispersion=dict(accepted=n,teacher_RMS_source_scaled=float(teacherr),input_RMS_source_scaled=float(inputr),teacher_over_input_RMS=float(teacherr/inputr)if inputr>1e-13 else None)
            geometry={arm:dict(source_scaled_norm=norm((z(o['outputs'][arm][pos])-ez[pos])/scale),center_norm=norm(o['outputs'][arm][pos,:2]-E[:2]),log_size_norm=norm(z(o['outputs'][arm][pos])[2:]-ez[pos,2:]))for arm in ARMS[1:]}
            bmean=np.r_[t[:2],np.exp(t[2:])]
            ns.append(dict(slot=j,support=case,center_accepted=M,sample_accepted=n,p=pp,legal_states=L,legal_fraction=L/24,
                accepted_given_legal=condition,center_score=float(center['score'])if center else None,
                endpoint_mean_center_IoU=iou(bmean,center['box'])if M and n else None,
                endpoint_mean_center_source_scaled_distance=norm((t-c)/scale)if M and n else None,
                endpoint_direction_cosine=cosine((t-ez[pos])/scale,uc/scale)if M and n else None,
                endpoint_unshrunk_norm_ratio=norm((t-ez[pos])/scale)/norm(uc/scale)if M and n and norm(uc/scale)>1e-13 else None,
                amplitude_term_norm=norm(amplitude/scale),endpoint_and_support_term_norm=norm(terminal/scale),
                term_cosine=cosine(amplitude/scale,terminal/scale),dispersion=dispersion,actual_displacement=geometry))
        outcomes={k:100*(r['values'][v]['v']-r['values'][w]['v'])for k,v,w in [
            ('total_gap','Samples24_endpoint','Center_full_local'),('p_scaling_path','Center_p_local','Center_full_local'),
            ('endpoint_and_support_path','Samples24_endpoint','Center_p_local'),('center_gain','Center_full_local','E_MBR'),('sample_gain','Samples24_endpoint','E_MBR')]}
        assert abs(outcomes['total_gap']-outcomes['p_scaling_path']-outcomes['endpoint_and_support_path'])<1e-12;checks['query_saved_score_path']+=1
        common={k:None if r['values'][v]['common_admitted_s']is None else 100*(r['values'][v]['common_admitted_s']-r['values'][w]['common_admitted_s'])for k,v,w in [
            ('total_gap','Samples24_endpoint','Center_full_local'),('p_scaling_path','Center_p_local','Center_full_local'),('endpoint_path','Samples24_endpoint','Center_p_local')]}
        rows.append(dict(alias=a,dataset=rec['dataset'],active=o['active'],legal_states=L,nodes=ns,outcome_pp=outcomes,common_saved_sIoU_pp=common,
            center_benefit_group='benefit'if outcomes['center_gain']>1e-8 else'harm'if outcomes['center_gain']< -1e-8 else'unchanged',
            has_exclusive_support=any(n['support']in ['center_only','sample_only']for n in ns)))
    groups={}
    for ds in ['hc2','vidstg']:
        rr=[r for r in rows if r['dataset']==ds];nodes=[n for r in rr for n in r['nodes']];common=[n for n in nodes if n['support']=='common'];sample=[n for n in nodes if n['sample_accepted']];center=[n for n in nodes if n['center_accepted']]
        g=dict(count=16,active=sum(r['active']for r in rr),support_counts=dict(collections.Counter(n['support']for n in nodes)),
            existing_complete_path={k:paired([r['outcome_pp'][k]for r in rr])for k in ['total_gap','p_scaling_path','endpoint_and_support_path','center_gain','sample_gain']},
            common_saved_sIoU_path={k:paired([r['common_saved_sIoU_pp'][k]for r in rr])for k in ['total_gap','p_scaling_path','endpoint_path']},
            p=dict(all64_mean=mean([n['p']for n in nodes]),sample_supported_mean=mean([n['p']for n in sample]),common_mean=mean([n['p']for n in common]),common_median=float(np.median([n['p']for n in common]))if common else None,
                active_legal_fraction=mean([r['legal_states']/24 for r in rr if r['active']]),accepted_given_legal_all_active=mean([n['accepted_given_legal']for n in nodes]),
                center_only_nodes=sum(n['support']=='center_only'for n in nodes),sample_only_nodes=sum(n['support']=='sample_only'for n in nodes)),
            common_endpoints={k:mean([n[k]for n in common])for k in ['endpoint_mean_center_IoU','endpoint_mean_center_source_scaled_distance','endpoint_direction_cosine','endpoint_unshrunk_norm_ratio','amplitude_term_norm','endpoint_and_support_term_norm']},
            dispersion=dict(nodes_with_two_or_more=sum(n['dispersion']is not None for n in nodes),mean_teacher_over_input_RMS=mean([n['dispersion']['teacher_over_input_RMS']for n in nodes if n['dispersion']]),median_teacher_over_input_RMS=float(np.median([n['dispersion']['teacher_over_input_RMS']for n in nodes if n['dispersion']and n['dispersion']['teacher_over_input_RMS']is not None]))),
            displacement={arm:{k:mean([n['actual_displacement'][arm][k]for n in nodes])for k in ['source_scaled_norm','center_norm','log_size_norm']}for arm in ARMS[1:]},
            common_displacement={arm:{k:mean([n['actual_displacement'][arm][k]for n in common])for k in ['source_scaled_norm','center_norm','log_size_norm']}for arm in ARMS[1:]})
        strata={}
        for label in ['benefit','harm','unchanged']:
            ss=[r for r in rr if r['center_benefit_group']==label];cn=[n for r in ss for n in r['nodes']if n['center_accepted']]
            strata[label]=dict(queries=len(ss),p_mean_center_nodes=mean([n['p']for n in cn]),
                path_contribution_to_full16_mean_pp={k:sum(r['outcome_pp'][k]for r in ss)/16 for k in ['total_gap','p_scaling_path','endpoint_and_support_path']})
        g['center_outcome_descriptive_strata']=strata
        ss=[r for r in rr if not r['has_exclusive_support']];g['queries_without_exclusive_support']=dict(count=len(ss),endpoint_path=paired([r['outcome_pp']['endpoint_and_support_path']for r in ss]))
        order=sorted(rr,key=lambda r:r['outcome_pp']['total_gap']);g['case_selection']=dict(lowest3=[r['alias']for r in order[:3]],highest3=[r['alias']for r in order[-3:]])
        groups[ds]=g
    verify();import torch;assert not torch.cuda.is_initialized()
    put(P/'RESULTS_PUBLIC.json',public(dict(status='complete',count=32,rows=rows,groups=groups,checks=dict(checks),maximum_vector_error=maxerr,
        new_GT_reads=0,new_scoring=0,new_predictions=0,new_models=0,new_media_reads=0,new_weights_reads=0,new_samples=0,GPU_seconds=0,CPU_seconds=time.perf_counter()-start,
        same_p_not_equal_norm=True,endpoint_raw_GT_quality_unmeasured=True,arithmetic_path_not_unique_causal_attribution=True,old_signed_unknowns_retained=True)))
    put(D/'CPU_COMPLETION.json',dict(status='complete',count=32,CPU_seconds=time.perf_counter()-start,GPU_initialized=False));print({ds:dict(p=g['p'],endpoint=g['common_endpoints'],dispersion=g['dispersion'])for ds,g in groups.items()},flush=True)

if __name__=='__main__':
    {'prepare':prepare,'run':run}[sys.argv[1]]()
