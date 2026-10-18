"""Existing-candidate offline diagnosis; never a deployment or TTA experiment."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,argparse,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
from scripts.run_stvg_cosine_dense_opd32_v1 import D as OLD,P as OLDP
D=ROOT/'artifacts/stvg_cosine_candidate_headroom32_v1';P=ROOT/'results/stvg_error_state/2026-10-18/cosine_candidate_headroom32'
CODE=['vg_tta/stvg_cosine_candidate_headroom32_v1.py','scripts/run_stvg_cosine_candidate_headroom32_v1.py','scripts/audit_stvg_cosine_candidate_headroom32_v1.py','protocols/stvg_cosine_candidate_headroom32_v1.md']

def verify():
    from main_method.runtime import PinVerifier,resource_guard
    p=read(D/'PLAN_LOCK.json');r=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==r['plan_sha256'];PinVerifier(r['pins']).verify();resource_guard();return p

def prepare():
    p=read(OLD/'PLAN_LOCK.json');assert read(OLD/'FINAL_COMPLETION.json')['status']=='complete'
    put(D/'PLAN_LOCK.json',dict(status='locked',records=p['records'],count=32,per_domain=16,source_commit='30746bcc741816f9978efb84dbfce08452817260',
        scope='offline existing-candidate headroom and E-relative ranking',new_candidates=0,new_updates=0,new_model_or_expert_calls=0,new_GPU_seconds=0,
        fixed=p['configuration'],GT_use='offline diagnostics explicitly, never method result',thresholds_IoU_pp=[1,5,10],CPU_cap_seconds=600,created_epoch=time.time()))
    files=CODE+['methods/CURRENT_METHOD.json',str((OLD/'GLOBAL_PREDICTION_BARRIER.json').relative_to(ROOT)),str((OLDP/'RESULTS_PUBLIC.json').relative_to(ROOT))]+[str((OLD/'predictions'/(r['alias']+'.pt')).relative_to(ROOT))for r in p['records']]
    put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins={f:sha(ROOT/f)for f in files}));print('locked32',flush=True)

def avg(v):
    v=[float(x)for x in v if x is not None];return float(np.mean(v))if len(v)else None

def analyze():
    from scripts.score_stvg_opd_p1_v1 import truths
    from scripts.run_stvg_dino_local_residual32_v1 import metrics
    from scripts.score_stvg_joint_robust_correction32_v1 import paired
    from vg_tta.stvg_cosine_candidate_headroom32_v1 import xy,iou,oracle,summarize_frame
    torch.set_num_threads(4);p=verify();start=time.perf_counter();bar=read(OLD/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};oldrows={r['alias']:r for r in read(OLDP/'RESULTS_PUBLIC.json')['rows']};rows=[];files=[]
    for r in p['records']:
        assert time.perf_counter()-start<p['CPU_cap_seconds'];a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];row=plans[ds][q];truth,span=gt[ds][0][q],gt[ds][1][q];o=load(OLD/'predictions'/(a+'.pt'));E=o['outputs']['E_MBR'];ids=np.array(o['frame_ids']);n=len(ids);known=np.array([int(f)in truth for f in ids]);unobs=~np.isin(np.arange(n),o['planned_positions']);mask=known&unobs;active=o['active'];iv=o['interval'];inside=(ids>=iv[0])&(ids<iv[1]);fb=o['direct']['feedback'];b=o['direct']['candidates'].numpy();valid=fb['valid'].numpy().copy();valid[:,0]=True
        quality=np.full(b.shape[:2],np.nan);base=np.full(n,np.nan);actual=np.full(n,np.nan);step1=np.full(n,np.nan);frame=[];node_oracle=E.copy();nodechoice=np.zeros(n,dtype=int)
        xx=xy(b,row,ds);ox=xy(o['outputs']['On_policy_dense'],row,ds);sx=xy(o['fits']['On_policy_dense']['path'][1]['boxes'].numpy(),row,ds)
        for t in np.flatnonzero(known):
            quality[t]=iou(xx[t],np.array(truth[int(ids[t])]))
            base[t]=quality[t,0];actual[t]=iou(ox[t],np.array(truth[int(ids[t])]))
            step1[t]=iou(sx[t],np.array(truth[int(ids[t])]))
        for t in np.flatnonzero(mask):
            record=summarize_frame(quality[t],fb['rewards'][t].numpy(),fb['prob'][t].numpy(),valid[t],base[t],active,actual[t],step1[t]);record.update(position=int(t),inside_MBR=bool(inside[t]));frame.append(record)
            scores=quality[t].copy();scores[~valid[t]]=-np.inf;k=int(scores.argmax());nodechoice[t]=k;node_oracle[t]=b[t,k]
        exact=oracle(b,fb['valid'].numpy(),E,ids,o['planned_positions'],row,ds,truth,span,iv)
        vals=metrics(dict(dataset=ds,frame_ids=ids,planned_positions=o['planned_positions'],interval=iv,outputs=dict(E_MBR=E,Actual_node_oracle=node_oracle,Finite_pool_full_v_oracle=exact['boxes'],On_policy_dense=o['outputs']['On_policy_dense'],Cosine_readout=o['outputs']['Cosine_readout'])),row,truth,span)
        for ep in ['v','t','s','unobserved_actual_s']:
            assert abs(vals['values']['E_MBR'][ep]-oldrows[a]['values']['E_MBR'][ep])<1e-12;assert abs(vals['values']['On_policy_dense'][ep]-oldrows[a]['values']['On_policy_dense'][ep])<1e-12
        assert abs(vals['values']['Finite_pool_full_v_oracle']['v']-exact['optimum_v'])<2e-10;assert exact['optimum_v']>=vals['values']['E_MBR']['v']-1e-12
        stat={support:{k:avg([f[k]for f in frame if support=='unobserved_known'or f['inside_MBR']])for k in frame[0]if k not in ['position','inside_MBR']}for support in ['unobserved_known','unobserved_known_inside_MBR']}if frame else {}
        public=dict(alias=a,dataset=ds,active_teacher=active,actual_center_changed=bool(np.any(o['outputs']['On_policy_dense']!=E)),GT_interval_overlap=bool(min(iv[1],span[1])>max(iv[0],span[0])),
            known_unobserved_nodes=int(mask.sum()),unknown_unobserved_nodes=int((~known&unobs).sum()),known_unobserved_inside_MBR=int((mask&inside).sum()),
            values=vals['values'],frame_means=stat,frame_counts=dict(meaningful_headroom=sum(f['has_gain_5pp']for f in frame),meaningful_headroom_and_teacher=sum(f['has_gain_5pp']and active for f in frame)),frames=frame)
        rows.append(public);files.append(save(D/'diagnostics'/(a+'.pt'),dict(alias=a,dataset=ds,quality=quality,base_quality=base,mask=mask,frame_rows=frame,nodechoice=nodechoice,node_oracle=node_oracle,exact_oracle=exact,GT_only_diagnostic=True)))
        print(a,len(rows),round(time.perf_counter()-start,3),flush=True)
    def group(rr):
        keys=list(rows[0]['frame_means']['unobserved_known'])
        return dict(count=len(rr),active_teacher_queries=sum(r['active_teacher']for r in rr),known_unobserved_nodes=sum(r['known_unobserved_nodes']for r in rr),unknown_unobserved_nodes=sum(r['unknown_unobserved_nodes']for r in rr),
            metrics={arm:{ep:avg([r['values'][arm][ep]for r in rr])for ep in ['v','t','unobserved_actual_s']}for arm in rr[0]['values']},
            gain_vs_E={arm:{ep:paired([100*(r['values'][arm][ep]-r['values']['E_MBR'][ep])for r in rr])for ep in ['v','unobserved_actual_s']}for arm in ['Actual_node_oracle','Finite_pool_full_v_oracle','On_policy_dense','Cosine_readout']},
            query_balanced={support:{k:paired([r['frame_means'].get(support,{}).get(k)for r in rr])for k in keys}for support in ['unobserved_known','unobserved_known_inside_MBR']},
            frame_counts=dict(meaningful_headroom=sum(r['frame_counts']['meaningful_headroom']for r in rr),meaningful_headroom_and_teacher=sum(r['frame_counts']['meaningful_headroom_and_teacher']for r in rr)),
            zero_v_cases=[dict(alias=r['alias'],no_teacher=not r['active_teacher'],no_temporal_overlap=not r['GT_interval_overlap'],actual_center_changed=r['actual_center_changed'])for r in rr if r['values']['E_MBR']['v']==0 and r['values']['On_policy_dense']['v']==0])
    groups={ds:group([r for r in rows if r['dataset']==ds])for ds in gt};teacher_groups={ds:group([r for r in rows if r['dataset']==ds and r['active_teacher']])for ds in gt};verify();assert not torch.cuda.is_initialized()
    put(P/'RESULTS_PUBLIC.json',dict(status='complete',count=32,groups=groups,teacher_available_groups=teacher_groups,rows=rows,GT_only_offline_diagnosis=True,no_new_predictions_or_updates=True,original_experiment_unchanged=True,independent_confirmation=False,CPU_seconds=time.perf_counter()-start))
    put(D/'DIAGNOSTIC_COMPLETION.json',dict(status='complete',count=32,files=files,CPU_seconds=time.perf_counter()-start,GT_only_offline_diagnosis=True,new_models_updates_candidates_GPU=0))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','analyze']);x=a.parse_args();globals()[x.stage]()
