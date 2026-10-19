"""Cache-only DINO residual-error/refresh diagnosis, no model or adaptation."""
import os,sys,time,collections
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
from vg_tta.stvg_dino_residual_advantage64_v1 import *
D=ROOT/'artifacts/stvg_dino_residual_advantage64_v1';P=ROOT/'results/stvg_error_state/2026-10-19/dino_residual_advantage64';VER=None
ARMS=['E_MBR','Retained_top1_admitted','Native900_top035','Conditional_center035']
STRATA=['all','bad','far','scale','scale_near_center','instance_mismatch']

def verify(full=False):
 global VER
 from main_method.runtime import PinVerifier,resource_guard
 p=read(D/'PLAN_LOCK.json');r=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==r['plan_sha256']
 pins=dict(r['pins'])
 if (D/'RECOVERY_LOCK.json').exists():
  fix=read(D/'RECOVERY_LOCK.json');assert sha(D/'RUNTIME_LOCK.json')==fix['original_runtime_sha256'];pins.update(fix['replacement_pins'])
 if VER is None:VER=PinVerifier(pins)
 VER.verify(full=full);resource_guard();return p

def prepare():
 assert not D.exists();D.mkdir();P.mkdir();rr=read(ROOT/'artifacts/stvg_pair_direction_dev64_v1/PLAN_LOCK.json')['records'];selectrows={r['alias']:r for r in read(ROOT/'artifacts/stvg_fine_step_confirm32_v1/PLAN_LOCK.json')['records']};files=[]
 for r in rr:
  a=r['alias'];r['source_path']=f'artifacts/stvg_dino_endpoint_confirm32_v1/source/{a}.pt'if r['panel']=='original'else selectrows[a]['source_path'];r['E_path']=f'artifacts/stvg_dino_endpoint_confirm32_v1/E/{a}.pt'if r['panel']=='original'else selectrows[a]['prediction_path'];r['conditional_path']=f'artifacts/stvg_dino_endpoint_confirm32_v1/predictions/{a}.pt'if r['panel']=='original'else None;files +=[r['path'],r['source_path'],r['E_path']]+([r['conditional_path']]if r['conditional_path']else[])
 assert len(rr)==len({r['parent']for r in rr})==64
 deps=['scripts/run_stvg_dino_residual_advantage64_v1.py','vg_tta/stvg_dino_residual_advantage64_v1.py','protocols/stvg_dino_residual_advantage64_v1.md','methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json','configs/stvg_fine_center_dataset_candidate_v1.json','scripts/score_stvg_opd_p1_v1.py','scripts/run_stvg_dino_local_residual32_v1.py','scripts/score_stvg_joint_robust_correction32_v1.py','scripts/prepare_vidstg_wrong_domain_support.py','results/stvg_error_state/2026-10-19/fine_dataset_formal64/RESULTS_PUBLIC.json']
 p=dict(status='locked_cache_diagnosis',count=64,records=rr,arms=ARMS,conditional_subset=32,thresholds=dict(bad_iou=.3,far_GT_size_units=1.,scale_factor=2.,near_center=.25,instance_target_iou=.1,instance_other_iou=.5,recovery_iou=.5,recovery_gain=.1,endpoint_increment=.05,score=.35,native_similarity=.95),new_GPU_models_DINO_training_candidates_adaptation=0,GT_offline_only=True,formal64_excluded=True,CPU_phase_cap_seconds=900,created_epoch=time.time(),identity_HC_unknown=True)
 put(D/'PLAN_LOCK.json',p);put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins={f:sha(ROOT/f)for f in sorted(set(files+deps))}));put(P/'CPU_PROPERTIES.json',properties());print('prelocked64 cache-only',flush=True)

def cache():
 import torch,gc
 torch.set_num_threads(4);p=verify(True);files=[];stats=collections.Counter();start=time.perf_counter()
 for r in p['records']:
  verify();assert time.perf_counter()-start<900;a=r['alias'];s=load(ROOT/r['source_path']);ep=load(ROOT/r['E_path']);old=load(ROOT/r['path']);E=np.asarray(ep['E']if 'E'in ep else ep['outputs']['E_DINO_support'],float);assert np.array_equal(E,old['outputs']['E_MBR']);assert s['frame_ids']==ep['frame_ids']==old['frame_ids'];positions=ep['planned_positions'];assert positions==s['expert']['positions4']==old['planned_positions'];c=load(ROOT/r['conditional_path'])if r['conditional_path']else None
  admitted=[]
  for pos in positions:
   obs=s['expert']['observations'].get(('original',pos))
   if obs is not None and obs['probe']['accepted']:admitted.append(pos)
  selected={pos:int(ix)for pos,ix in zip(admitted,ep['estimate']['selected'])};assert len(admitted)==len(ep['estimate']['selected'])
  outputs={arm:E.copy()for arm in ARMS};frames=[]
  for pos in positions:
   obs=s['expert']['observations'].get(('original',pos));pr=obs['probe']if obs is not None else dict(boxes=[],target_scores=[],candidate_ids=[],accepted=False);b=np.asarray(pr['boxes'],float).reshape(-1,4);sc=np.asarray(pr['target_scores'],float);assert len(b)==len(sc);f=dict(position=int(pos),frame_id=int(s['frame_ids'][pos]),retained_boxes=b,retained_scores=sc,retained_ids=list(pr['candidate_ids']),probe_available=obs is not None,frame_admitted=bool(pr['accepted']),E_selected=selected.get(pos,-1),native=None,center=None,samples=[])
   if pr['accepted']:
    assert len(b);outputs['Retained_top1_admitted'][pos]=b[sc.argmax()]
   if c is not None:
    assert np.array_equal(c['outputs']['E_MBR'],E)
    if pos in c['native']:
     nn=c['native'][pos];nb=np.asarray(nn['layer_boxes'][-1],float);ns=np.asarray(nn['layer_scores'][-1],float);assert nb.shape==(900,4)and ns.shape==(900,);ix=select(nb,ns);f['native']=dict(boxes=nb,scores=ns,top035=ix);stats['native900_frames']+=1
     if ix is not None:outputs['Native900_top035'][pos]=nb[ix]
    if pos in c['center_feedback']['anchors']:
     v=c['center_feedback']['anchors'][pos];f['center']=dict(box=np.asarray(v['box'],float),score=float(v['score']),accepted=bool(v['accepted']));assert bool(v['accepted'])==(float(v['score'])>=.35)
     if v['accepted']:outputs['Conditional_center035'][pos]=v['box']
    tr=c['traces'].get('samples',{}).get(pos)
    if tr is not None:
     trace=tr['B_aligned'];assert len(tr['candidate_indices'])==len(tr['inverse']);assert np.array_equal(trace['content_slots'],trace['mask_slots'])
     for j,ix in enumerate(tr['candidate_indices']):
      k=int(tr['inverse'][j]);detail=c['sample_feedback']['details'][int(ix)]['anchors'][pos];box=np.asarray(trace['boxes'][-1,k],float);score=float(trace['scores'][-1,k]);assert np.array_equal(box,detail['box'])and score==float(detail['score']);f['samples'].append(dict(index=int(ix),input=np.asarray(trace['initial_reference'][k],float),box=box,score=score,accepted=bool(detail['accepted'])));stats['sample_frame_states']+=1
   frames.append(f);stats['planned_frames']+=1;stats['original_observation_frames']+=obs is not None;stats['retained_candidate_boxes']+=len(b);stats['admitted_frames']+=bool(pr['accepted'])
  z=dict(alias=a,dataset=r['dataset'],panel=r['panel'],frame_ids=s['frame_ids'],planned_positions=positions,interval=ep['interval'],E=E,frames=frames,outputs=outputs,conditional_cache_available=c is not None,GT_read=False);files.append(save(D/'extracted002'/(a+'.pt'),z));print(a,'cache geometry sealed',flush=True);del s,ep,old,c,z;gc.collect()
 verify(True);put(D/'CACHE_BARRIER.json',dict(status='complete',count=64,files=files,all_geometry_readouts_before_this_GT=True,counts=dict(stats),CPU_seconds=time.perf_counter()-start,new_models_experts_GPU_training=0));put(P/'CACHE_COVERAGE_PUBLIC.json',dict(status='complete',parents=64,conditional_parent_cache=32,**dict(stats)));print('64 cache barrier complete',flush=True)

def mean(xs):
 a=[x for x in xs if x is not None];return float(np.mean(a))if a else None

def boot(xs):
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 return paired(xs)

def aggregate_frames(frames,records):
 fields=['retained_best_gain_pp','retained_top1_gain_pp','admitted_top1_gain_pp','native_all_best_gain_pp','native035_best_gain_pp','native_top035_gain_pp','conditional_center_raw_gain_pp','conditional_center_admitted_gain_pp','retained_recovery','native035_recovery','native_top035_recovery','center_recovery']
 groups={}
 for ds in ['hc2','vidstg']:
  groups[ds]={}
  for panel in ['original','selection','combined']:
   parent_aliases=[r['alias']for r in records if r['dataset']==ds and(panel=='combined'or r['panel']==panel)];d={}
   for st in STRATA:
    ff=[f for f in frames if f['alias']in parent_aliases and f['GT_known']and f['strata'].get(st,False)];aa=sorted({f['alias']for f in ff});d[st]=dict(eligible_parent_count=len(aa),known_frames=len(ff),parents_total=len(parent_aliases),fields={k:boot([mean([f.get(k)for f in ff if f['alias']==a])for a in aa])for k in fields},frame_recovery_counts={k:sum(bool(f.get(k))for f in ff)for k in ['retained_recovery','native035_recovery','native_top035_recovery','center_recovery']},teacher_supported_frames={k:sum(f.get(k)is not None for f in ff)for k in fields})
   groups[ds][panel]=d
 return groups

def aggregate_refresh(states,frames,records):
 fields=['teacher_minus_input_pp','teacher_minus_E_pp','teacher_minus_center_pp','output_center_IoU','new_admitted_correct','common_novel_better','same_fixed_native_candidate','lost_center_admission','sample_recovery','center_recovery']
 groups={}
 for ds in ['hc2','vidstg']:
  pp=[r for r in records if r['dataset']==ds and r['panel']=='original'];groups[ds]={}
  for st in STRATA:
   keys={(f['alias'],f['position'])for f in frames if f['dataset']==ds and f['panel']=='original'and f['GT_known']and f['strata'].get(st,False)};ss=[s for s in states if(s['alias'],s['position'])in keys];d={}
   for mode in ['all_evaluated','both_admitted']:
    sub=ss if mode=='all_evaluated'else[s for s in ss if s['accepted']and s['center_accepted']];aliases=sorted({s['alias']for s in sub})
    # Equal physical frames within query, then parent macro. Sample replicates
    # never become bootstrap units or reweight frames with more valid states.
    vals={k:[]for k in fields}
    for a in aliases:
     positions=sorted({s['position']for s in sub if s['alias']==a})
     for k in fields:vals[k].append(mean([mean([s.get(k)for s in sub if s['alias']==a and s['position']==pos])for pos in positions]))
    d[mode]=dict(parents=len(aliases),physical_frames=len({(s['alias'],s['position'])for s in sub}),sample_state_instances=len(sub),fields={k:boot(v)for k,v in vals.items()},counts={k:sum(bool(s.get(k))for s in sub)for k in ['new_admitted_correct','common_novel_better','lost_center_admission','sample_recovery']})
   groups[ds][st]=d
 return groups

def score():
 import torch,ijson
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.prepare_vidstg_wrong_domain_support import load_vidor_annotations
 from scripts.run_stvg_dino_local_residual32_v1 import metrics
 torch.set_num_threads(4);p=verify(True);bar=read(D/'CACHE_BARRIER.json');assert bar['count']==64 and bar['all_geometry_readouts_before_this_GT']
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 start=time.perf_counter();put(D/'OFFLINE_GT_ACCESS.json',dict(cache_barrier_sha256=sha(D/'CACHE_BARRIER.json'),purpose='offline residual error/instance evidence only',time=time.time(),no_GT_online_selection=True))
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};raw=load_vidor_annotations(ROOT/'downloads/vidor/validation-annotation.zip',None);keys={plans['vidstg'][r['query_ordinal']]['key']for r in p['records']if r['dataset']=='vidstg'}
 with (ROOT/'artifacts/stvg_fullscale_diagnostics_v1/labels_diagnostic_only.json').open('rb')as f:target_ids={k:int(v['official_annotation']['target_id'])for k,v in ijson.kvitems(f,'',use_float=True)if k in keys}
 frames=[];states=[];rows=[];private=[]
 for r in p['records']:
  verify();assert time.perf_counter()-start<900;a=r['alias'];ds=r['dataset'];x=load(D/'extracted002'/(a+'.pt'));row=plans[ds][r['query_ordinal']];truth=gt[ds][0][r['query_ordinal']];span=gt[ds][1][r['query_ordinal']];local=[]
  for f in x['frames']:
   pos=f['position'];fid=f['frame_id'];rec=dict(alias=a,dataset=ds,panel=r['panel'],position=pos,GT_known=fid in truth,probe_available=f['probe_available'],frame_admitted=f['frame_admitted'],E_selected=f['E_selected'],retained_candidates=len(f['retained_boxes']),native_available=f['native']is not None,center_available=f['center']is not None,identity_evidence_available=False,identity_status='unknown')
   if fid not in truth:frames.append(rec);continue
   tr=truth[fid];geo=geometry(x['E'][pos],tr,row,ds);rec.update(geo);e=geo['E_IoU'];bb=f['retained_boxes'];scores=f['retained_scores'];pv=ious(bb,tr,row,ds)if len(bb)else np.array([]);best=float(pv.max())if len(pv)else None;top=float(pv[scores.argmax()])if len(pv)else None
   rec.update(retained_best_gain_pp=100*(best-e)if best is not None else None,retained_top1_gain_pp=100*(top-e)if top is not None else None,admitted_top1_gain_pp=100*(top-e)if top is not None and f['frame_admitted']else None,retained_recovery=recovered(best,e)if best is not None else None,retained_best_iou=best,retained_top1_iou=top)
   if ds=='vidstg':
    rawframe=raw[row['source']]['trajectories'][fid];tid=target_ids[row['key']];other=[]
    for obj in rawframe:
     b=obj.get('bbox',{});ob=np.array([b.get('xmin',0),b.get('ymin',0),b.get('xmax',0),b.get('ymax',0)],float)
     if (ob[2:]-ob[:2]<=0).any():continue
     if int(obj['tid'])==tid:assert np.allclose(ob,tr,rtol=0,atol=1e-8)
     else:other.append(ob)
    oi=[float(ious(x['E'][pos:pos+1],ob,row,ds)[0])for ob in other];maxother=max(oi,default=0.);rec.update(identity_evidence_available=True,annotated_other_instances=len(other),max_other_instance_IoU=maxother,identity_status='other_instance_overlap'if e<=.1 and maxother>=.5 else'not_flagged');rec['strata']['instance_mismatch']=rec['identity_status']=='other_instance_overlap'
   native=f['native'];nb=None;nv=None;valid=None
   if native is not None:
    nb=native['boxes'];nv=ious(nb,tr,row,ds);valid=legal(nb)&(native['scores']>=.35);ni=native['top035'];b035=float(nv[valid].max())if valid.any()else None;top035=float(nv[ni])if ni is not None else None;rec.update(native_all_best_gain_pp=100*(float(nv.max())-e),native035_best_gain_pp=100*(b035-e)if b035 is not None else None,native_top035_gain_pp=100*(top035-e)if top035 is not None else None,native035_best_iou=b035,native_top035_iou=top035,native035_recovery=recovered(b035,e)if b035 is not None else None,native_top035_recovery=recovered(top035,e)if top035 is not None else None,native035_candidates=int(valid.sum()))
   center=f['center'];ci=float(ious(center['box'][None],tr,row,ds)[0])if center is not None else None;ca=bool(center['accepted'])if center is not None else False;rec.update(conditional_center_raw_gain_pp=100*(ci-e)if ci is not None else None,conditional_center_admitted_gain_pp=100*(ci-e)if ci is not None and ca else None,conditional_center_iou=ci,center_accepted=ca,center_recovery=recovered(ci,e)if ci is not None and ca else None)
   for st in f['samples']:
    si=float(ious(st['box'][None],tr,row,ds)[0]);ii=float(ious(st['input'][None],tr,row,ds)[0]);sr=recovered(si,e);cr=recovered(ci,e)if ci is not None else False;sim=max((mutual_iou(st['box'],b)for b in nb[valid]),default=0.)if nb is not None else None
    states.append(dict(alias=a,dataset=ds,position=pos,sample_index=st['index'],accepted=st['accepted'],center_accepted=ca,input_IoU=ii,teacher_IoU=si,E_IoU=e,center_IoU=ci,teacher_minus_input_pp=100*(si-ii),teacher_minus_E_pp=100*(si-e),teacher_minus_center_pp=100*(si-ci)if ci is not None else None,output_center_IoU=mutual_iou(st['box'],center['box'])if center is not None else None,sample_recovery=bool(st['accepted']and sr),center_recovery=bool(ca and cr),new_admitted_correct=bool(st['accepted']and sr and not(ca and cr)),common_novel_better=bool(st['accepted']and ca and ci is not None and si-ci>=.05 and si-e>=.1),same_fixed_native_candidate=bool(sim>=.95)if sim is not None else None,native_max_box_IoU=sim,lost_center_admission=bool(ca and not st['accepted']),admission_only_recovery=bool(st['accepted']and sr and cr and not ca)))
   frames.append(rec);local.append(rec)
  m=metrics(x,row,truth,span);old=next(z for z in read(ROOT/r['result_path'])['rows']if z['alias']==a)
  for k,v in m['values']['E_MBR'].items():
   vv=old['values']['E_MBR'][k];assert v is None and vv is None or abs(v-vv)<1e-10
  rows.append(dict(alias=a,dataset=ds,panel=r['panel'],conditional_cache_available=x['conditional_cache_available'],known_planned_frames=len(local),**m));print(a,'offline cache diagnosis',flush=True)
 groups=aggregate_frames(frames,p['records']);refresh=aggregate_refresh(states,frames,p['records']);full={}
 for ds in ['hc2','vidstg']:
  full[ds]={}
  for panel in ['original','selection','combined']:
   sub=[r for r in rows if r['dataset']==ds and(panel=='combined'or r['panel']==panel)];full[ds][panel]={a:{ep:boot([100*(r['values'][a][ep]-r['values']['E_MBR'][ep])if r['values'][a][ep]is not None and r['values']['E_MBR'][ep]is not None else None for r in sub])for ep in ['v','unobserved_actual_s','observed_actual_s']}for a in ARMS[1:]}
 result=dict(status='complete_cache_only_diagnosis',parents=64,frames=frames,groups=groups,refresh_groups=refresh,full_readout_groups=full,rows=rows,conditional_states=states,GT_only_offline=True,no_new_models_experts_GPU_training_candidates_or_adaptation=True,identity_HC_unresolved=True,initial24_states_not_actual_updated_policy_trajectory=True,development_exposed=True)
 put(P/'RESULTS_PUBLIC.json',result);put(P/'COST_BREAKDOWN.json',dict(cache_seconds=bar['CPU_seconds'],diagnosis_seconds=time.perf_counter()-start,total_Source_captures_completed=0,final_worker_DINO_encodings_completed=0,engineering_DINO_encodings_completed=0,new_GPU_seconds=0,new_teacher_queries=0,new_models_or_training=0,old_teacher_cache_cost_excluded=True));put(D/'SCORE_COMPLETION.json',dict(status='complete',count=64,CPU_seconds=time.perf_counter()-start));verify(True);print('diagnosis64 complete',flush=True)

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','cache','score']);a=ap.parse_args()
 try:globals()[a.stage]()
 except BaseException as e:
  D.mkdir(exist_ok=True);put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(stage=a.stage,type=type(e).__name__,error=str(e),preserved_all_partials=True));raise
