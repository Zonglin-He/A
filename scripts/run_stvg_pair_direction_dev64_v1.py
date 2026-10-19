"""Cached Live symmetric-pair decomposition and offline GT direction diagnosis."""
from pathlib import Path
import os,sys,time,collections
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=ROOT/'artifacts/stvg_pair_direction_dev64_v1';P=ROOT/'results/stvg_error_state/2026-10-19/pair_direction_dev64'
CODE=['scripts/run_stvg_pair_direction_dev64_v1.py','protocols/stvg_pair_direction_dev64_v1.md']
EPS=1e-12
VER=None

def verify(full=False):
 global VER
 from main_method.runtime import PinVerifier,resource_guard
 p=read(D/'PLAN_LOCK.json');rt=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==rt['plan_sha256']
 if VER is None:VER=PinVerifier(rt['pins'])
 VER.verify(full=full);resource_guard();return p

def prepare():
 assert not D.exists();D.mkdir();P.mkdir();records=[]
 for panel,folder in [('original','stvg_fine_roi_scoped_opd32_v1'),('selection','stvg_fine_step_confirm32_v1')]:
  for r in read(ROOT/'artifacts'/folder/'PLAN_LOCK.json')['records']:
   ds=r['dataset'];arm=('Step_high'if ds=='hc2'else'Fine_live')if panel=='original'else('Live_step10'if ds=='hc2'else'Live_step05');source=('stvg_fine_roi_sensitivity32_v1'if ds=='hc2'else folder)if panel=='original'else folder
   records.append(dict(alias=r['alias'],dataset=ds,parent=r['parent'],query_ordinal=r['query_ordinal'],panel=panel,arm=arm,path=f'artifacts/{source}/predictions/{r["alias"]}.pt',step=1. if ds=='hc2'else .5,result_path=f'results/stvg_error_state/2026-10-19/'+{'stvg_fine_roi_sensitivity32_v1':'fine_roi_sensitivity32','stvg_fine_roi_scoped_opd32_v1':'fine_roi_scoped_opd32','stvg_fine_step_confirm32_v1':'fine_step_confirm32'}[source]+'/RESULTS_PUBLIC.json'))
 formal=read(ROOT/'artifacts/stvg_fine_dataset_formal64_v1/PLAN_LOCK.json')['records'];assert not set(r['parent']for r in records)&set(r['parent']for r in formal);assert len(records)==len({r['parent']for r in records})==64
 plan=dict(status='locked',count=64,records=records,configuration='configs/stvg_fine_center_dataset_candidate_v1.json',current_HCstep1_Vidstep05=True,development_only=True,panels=['original','selection'],rounds=2,pairs_per_node=8,numerical_tie_epsilon=EPS,CPU_cap_seconds=900,new_GPU_model_expert_DINO_training_OPD_or_reward=0,GT_offline_only=True,formal64_not_used=True,created_epoch=time.time())
 files=CODE+['configs/stvg_fine_center_dataset_candidate_v1.json','methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json','scripts/score_stvg_opd_p1_v1.py','scripts/score_stvg_dino_mask_align32_v1.py','vg_tta/tastvg_oracle_event5_v1.py','scripts/score_stvg_joint_robust_correction32_v1.py','results/stvg_error_state/2026-10-19/fine_dataset_formal64/RESULTS_PUBLIC.json','artifacts/stvg_fine_dataset_formal64_v1/GLOBAL_PREDICTION_BARRIER.json']
 files+=list({r['path']for r in records}|{r['result_path']for r in records});put(D/'PLAN_LOCK.json',plan);put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins={f:sha(ROOT/f)for f in files}));print('locked development64 only',flush=True)

def seal():
 import torch,numpy as np
 torch.set_num_threads(4);p=verify();start=time.perf_counter();files=[]
 for r in p['records']:
  verify();assert time.perf_counter()-start<900;x=load(ROOT/r['path']);fit=x['fits'][r['arm']];E=np.asarray(x['outputs']['E_MBR']);allowed=x['allowed'].numpy();steps=[]
  for j,t in enumerate(fit['rounds']):
   before=t['eta_before'].numpy();after=t['eta_after'].numpy();d=t['standardized_candidates'].numpy();ee=d-before[:,None];q=t['feedback']['prob'].numpy();sp=t['student_prob'].numpy();assert np.max(abs(ee[:,1:9]+ee[:,9:17]))<1e-12;assert np.max(abs(ee[:,0]))<1e-12;assert np.array_equal(t['sampling_center'],before)
   active=allowed&t['feedback']['informative'].numpy();pair=r['step']*((q[:,1:9]-sp[:,1:9])-(q[:,9:17]-sp[:,9:17]))[...,None]*ee[:,1:9];pair[~active]=0;res=(after-before)-pair.sum(1);assert np.max(abs(res))<1e-12;assert np.array_equal(t['candidates'][:,0],fit['path'][j]['boxes']);assert np.max(abs((after-before)+r['step']*t['gradient'].numpy()))<1e-12
   steps.append(dict(step=j+1,active=active,pair_contribution=pair,rounding_residual=res,current=fit['path'][j]['boxes'],after=fit['path'][j+1]['boxes'],candidates=t['candidates'],q=q,p=sp,valid=t['feedback']['valid'],rewards=t['feedback']['rewards'],delta=after-before))
  files.append(save(D/'decomposition'/(r['alias']+'.pt'),dict(alias=r['alias'],dataset=r['dataset'],panel=r['panel'],E=E,allowed=allowed,frame_ids=x['frame_ids'],planned_positions=x['planned_positions'],steps=steps,GT_read=False)));print(r['alias'],'decomposed',flush=True)
 verify(full=True);put(D/'DECOMPOSITION_BARRIER.json',dict(status='complete',count=64,files=files,no_GT_read=True,CPU_seconds=time.perf_counter()-start))

def ious(b,gt,row,ds):
 import numpy as np
 b=np.asarray(b,float);xy=np.concatenate((b[...,:2]-b[...,2:]/2,b[...,:2]+b[...,2:]/2),-1)*np.array([row['input']['width'],row['input']['height']]*2)
 if ds=='hc2':xy=np.maximum(xy,0)
 g=np.asarray(gt,float);inter=np.maximum(0,np.minimum(xy[...,2:],g[2:])-np.maximum(xy[...,:2],g[:2])).prod(-1);union=np.maximum(0,xy[...,2:]-xy[...,:2]).prod(-1)+np.maximum(0,g[2:]-g[:2]).prod()-inter
 return np.divide(inter,union,out=np.zeros_like(inter),where=union>0)

def diagnose():
 import torch,numpy as np
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 torch.set_num_threads(4);p=verify();bar=read(D/'DECOMPOSITION_BARRIER.json');assert bar['count']==64
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 start=time.perf_counter();gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};rows=[];nodes=[]
 for r in p['records']:
  verify();assert time.perf_counter()-start<900;x=load(D/'decomposition'/(r['alias']+'.pt'));ids=x['frame_ids'];ds=r['dataset'];qord=r['query_ordinal'];truth=gt[ds][0][qord];row=plans[ds][qord];known=np.array([int(f)in truth for f in ids]);unobs=~np.isin(np.arange(len(ids)),x['planned_positions']);den=int((known&unobs).sum());counts=collections.Counter();norms=collections.Counter();proj=collections.Counter();stepdelta=[];qualified=[];nodelist=[]
  for st in x['steps']:
   stepgain=0.
   for t,fid in enumerate(ids):
    if known[t]and unobs[t]:stepgain+=float(ious(st['after'][t],truth[int(fid)],row,ds)-ious(st['current'][t],truth[int(fid)],row,ds))/max(den,1)
    if not st['active'][t]:continue
    cc=st['pair_contribution'][t];delta=st['delta'][t];q=st['q'][t];sp=st['p'][t];valid=st['valid'][t].numpy();rew=st['rewards'][t].numpy();quality=ious(st['candidates'][t],truth[int(fid)],row,ds)if known[t]else None;current=float(ious(st['current'][t],truth[int(fid)],row,ds))if known[t]else None;after=float(ious(st['after'][t],truth[int(fid)],row,ds))if known[t]else None;classes=[];pairs=[];good=np.zeros(2);cn=collections.Counter();nm=collections.Counter();pr=collections.Counter();dn=float(np.linalg.norm(delta));unit=delta/dn if dn>EPS else np.zeros(2)
    for j in range(8):
     a,b=j+1,j+9;alpha=float((q[a]-sp[a])-(q[b]-sp[b]));both=bool(valid[a]and valid[b]);gap=float(quality[a]-quality[b])if quality is not None else None
     if not both:cl='support_limited'
     elif quality is None:cl='unknown_GT'
     elif abs(gap)<=EPS:cl='GT_tie'
     elif abs(alpha)<=EPS:cl='teacher_tie'
     else:cl='correct'if alpha*gap>0 else'wrong'
     cn[cl]+=1;nm[cl]+=float(np.linalg.norm(cc[j]));pr[cl]+=float(cc[j]@unit);classes.append(cl)
     if cl=='correct':good+=cc[j]
     if both and quality is not None:
      raw=float(rew[a]-rew[b]);cn['both_valid_known_pairs']+=1
      if abs(raw)>EPS and abs(gap)>EPS:cn['raw_direction_defined']+=1;cn['raw_direction_correct']+=int(raw*gap>0)
      if abs(q[a]-q[b])>EPS and abs(gap)>EPS:cn['posterior_direction_defined']+=1;cn['posterior_direction_correct']+=int((q[a]-q[b])*gap>0)
      improve=max(float(quality[a]),float(quality[b]))>current+EPS;cn['pair_has_improver']+=int(improve);cn['pair_no_improver']+=int(not improve);cn['pair_both_harm_current']+=int(max(float(quality[a]),float(quality[b]))<current-EPS)
      if cl in ['correct','wrong']:
       chosen=a if alpha>0 else b;cn['actual_preference_defined']+=1;cn['preferred_side_improves_current']+=int(quality[chosen]>current+EPS);cn['preferred_side_harms_current']+=int(quality[chosen]<current-EPS)
       if cl=='correct'and not improve:cn['correct_but_no_improver']+=1
     pairs.append(dict(pair=j,classification=cl,coefficient=alpha,contribution=cc[j],valid_both=both,IoU_gap=gap,IoU_plus=None if quality is None else float(quality[a]),IoU_minus=None if quality is None else float(quality[b])))
    tilt=None if quality is None else float((q-sp)@quality);gain=None if quality is None else after-current;gn=float(np.linalg.norm(good));cos=None if gn<=EPS or dn<=EPS else float((good@delta)/(gn*dn));l1=sum(nm.values());cancellation=dn/l1 if l1>EPS else None
    if quality is not None:
     cn['known_active_positions']+=1;cn['candidate_has_improver_positions']+=int(quality.max()>current+EPS);cn['candidate_headroom5pp_positions']+=int(quality.max()>current+.05)
     if tilt>EPS:cn['positive_tilt_positions']+=1;cn['positive_tilt_actual_harm_positions']+=int(gain<-EPS)
     if gain>EPS:cn['actual_improve_positions']+=1
     if gain<-EPS:cn['actual_harm_positions']+=1
     qualified.append(dict(tilt=tilt,gain=gain,cos=cos,cancellation=cancellation))
    if cos is not None:cn['correct_component_alignment_defined']+=1;cn['correct_component_opposed']+=int(cos<0)
    counts.update(cn);norms.update(nm);proj.update(pr);public=dict(alias=r['alias'],dataset=ds,panel=r['panel'],step=st['step'],node=t,known_GT=bool(known[t]),counts=dict(cn),norm_mass=dict(nm),signed_projection_onto_actual=dict(pr),expected_quality_tilt_pp=None if tilt is None else 100*tilt,actual_IoU_delta_pp=None if gain is None else 100*gain,correct_component_cosine=cos,cancellation_ratio=cancellation);nodes.append(public);nodelist.append(dict(**public,pairs=pairs,correct_vector=good,actual_vector=delta,candidate_quality=quality,current_quality=current,after_quality=after))
   stepdelta.append(100*stepgain if den else None)
  def ratio(a,b):return counts[a]/counts[b]if counts[b]else None
  allmass=sum(norms.values());accmass=norms['correct']+norms['wrong'];metrics=dict(raw_pair_accuracy=ratio('raw_direction_correct','raw_direction_defined'),posterior_pair_accuracy=ratio('posterior_direction_correct','posterior_direction_defined'),pair_has_improver_rate=ratio('pair_has_improver','both_valid_known_pairs'),pair_both_harm_current_rate=ratio('pair_both_harm_current','both_valid_known_pairs'),preferred_side_improves_current_rate=ratio('preferred_side_improves_current','actual_preference_defined'),correct_but_no_improver_fraction=ratio('correct_but_no_improver','correct'),wrong_norm_fraction_among_classified=norms['wrong']/accmass if accmass>EPS else None,positive_tilt_actual_harm_rate=ratio('positive_tilt_actual_harm_positions','positive_tilt_positions'),correct_vector_opposed_rate=ratio('correct_component_opposed','correct_component_alignment_defined'),candidate_has_improver_rate=ratio('candidate_has_improver_positions','known_active_positions'),candidate_headroom5pp_rate=ratio('candidate_headroom5pp_positions','known_active_positions'),actual_harm_rate=ratio('actual_harm_positions','known_active_positions'),actual_improve_rate=ratio('actual_improve_positions','known_active_positions'),expected_quality_tilt_pp=float(np.mean([100*v['tilt']for v in qualified]))if qualified else None,actual_active_IoU_delta_pp=float(np.mean([100*v['gain']for v in qualified]))if qualified else None,all_known_unobserved_final_delta_pp=sum(stepdelta)if den else None,cancellation_ratio=float(np.mean([v['cancellation']for v in qualified if v['cancellation']is not None]))if any(v['cancellation']is not None for v in qualified)else None)
  old=next(v for v in read(ROOT/r['result_path'])['rows']if v['alias']==r['alias']);arm=r['arm'];actual_saved=100*(old['values'][arm]['unobserved_actual_s']-old['values']['E_MBR']['unobserved_actual_s']);assert abs(metrics['all_known_unobserved_final_delta_pp']-actual_saved)<1e-10
  rec=dict(alias=r['alias'],dataset=ds,panel=r['panel'],step=r['step'],known_unobserved_nodes=den,counts=dict(counts),norm_mass=dict(norms),signed_projection_onto_actual=dict(proj),step_all_known_unobserved_delta_pp=stepdelta,metrics=metrics);rows.append(rec);save(D/'diagnostics'/(r['alias']+'.pt'),dict(summary=rec,nodes=nodelist,GT_offline_diagnostic=True,no_new_predictions=True));print(r['alias'],'diagnosed',flush=True)
 groups={}
 for panel in ['original','selection','combined']:
  groups[panel]={}
  for ds in ['hc2','vidstg']:
   rr=[r for r in rows if r['dataset']==ds and(panel=='combined'or r['panel']==panel)];groups[panel][ds]=dict(parents=len(rr),metrics={k:paired([r['metrics'][k]for r in rr])for k in rows[0]['metrics']},sum_counts=dict(sum((collections.Counter(r['counts'])for r in rr),collections.Counter())),norm_mass_total=dict(sum((collections.Counter(r['norm_mass'])for r in rr),collections.Counter())))
 verify(full=True);put(P/'RESULTS_PUBLIC.json',dict(status='complete',count=64,groups=groups,rows=rows,formal64_not_used=True,GT_offline_only=True,pair_accuracy_is_conditional_not_main_deployment_metric=True));put(P/'NODE_SUMMARIES.json',dict(status='complete',nodes=nodes,no_GT_or_candidate_coordinates=True));put(D/'DIAGNOSIS_COMPLETION.json',dict(status='complete',count=64,CPU_diagnosis_seconds=time.perf_counter()-start,new_GPU_model_expert_DINO_training_OPD=0));print('diagnosis complete',flush=True)
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','seal','diagnose']);a=ap.parse_args()
 try:globals()[a.stage]()
 except BaseException as e:
  D.mkdir(exist_ok=True,parents=True);put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(stage=a.stage,error_type=type(e).__name__,error=str(e),all_partials_preserved=True));raise
