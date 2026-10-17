"""Independent geometry root, same-position DINO baseline, unchanged Student and sealed evaluation."""
from pathlib import Path
import copy,gc,inspect,json,sys,time,types
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_r1_geometry32_v2 import BASE,PAPER,read,sha,put,verify
ARMS=('DIRECT','HEADS','DINO')
def reference(box,j,role,digest):
 return dict(position=j['position'],physical_frame_id=j['physical_frame_id'],box_cxcywh=list(map(float,box)),weight=1.,candidate_id=role,teacher_output_sha256=digest,GT_read=False)
def r1root():
 import numpy as np,torch
 from scripts.stvg_r1_geometry32_v2 import parse_coords
 from scripts.stvg_r1_localization_readout_v1 import readout
 verify();bar=read(BASE/'R1_GLOBAL_BARRIER.json');assert bar['responses']==128 and read(BASE/'R1_QUALIFICATION_COMPLETION.json')['status']=='pass';jobs={j['job_id']:j for j in read(BASE/'INPUT_JOB_LOCK.json')['records']};refs={a:{r['probe_alias']:[] for r in read(BASE/'ROSTER_LOCK.json')['records']} for a in ['DIRECT','HEADS']};dec={a:[] for a in refs}
 for f in bar['files']:
  assert sha(ROOT/f['path'])==f['sha256'];d=read(ROOT/f['path']);j=jobs[d['job_id']];meta=d['input_meta'];assert sha(ROOT/d['attention_path'])==d['attention_sha256'];assert meta['grid_width']*28==meta['processed_width'] and meta['grid_height']*28==meta['processed_height'];assert d['physical_frame_id']==j['physical_frame_id']
  with np.load(ROOT/d['attention_path'],allow_pickle=False) as z:
   loc=readout(torch.from_numpy(z['anchor_attention']),meta['grid_height'],meta['grid_width'],meta['processed_width'],meta['processed_height']);assert np.array_equal(loc.pop('map'),z['combined_map']);assert loc==d['heads'] and (np.abs(z['full_row_sum']-1)<.01).all()
  assert d['direct']==parse_coords(d['raw_output'],meta['processed_width'],meta['processed_height'])
  for role,x in [('DIRECT',d['direct']),('HEADS',d['heads'])]:
   if x['box'] is not None:refs[role][j['case_id']].append(reference(x['box'],j,'R1_'+role,f['sha256']))
   dec[role].append(dict(job_id=j['job_id'],case_id=j['case_id'],slot=j['slot'],status=x['status'],truncated=d['truncated'],GT_read=False))
 for role in refs:put(BASE/(role+'_REFERENCE_LOCK.json'),dict(status='locked',count=32,records=[dict(case_id=k,anchors=v) for k,v in refs[role].items()],decisions=dec[role],GT_read=False))
 put(BASE/'ROOT_R1_GEOMETRY_READBACK.json',dict(status='pass',count=32,responses=128,exact_attention_to_grid_and_box_recomputed=True,all_key_softmax_preserved=True,R1_no_DINO_or_Student_geometry=True,GT_read=False))
def dino():
 import numpy as np,torch
 from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
 from methods.decota_final_simplified_v1.observations import SpatialExpert
 from methods.decota_final_simplified_v1.tensors import state_hash
 from scripts.run_stvg_reference_swap_teacher002 import _observe_locked_position,_install_gt_read_guard
 verify();assert read(BASE/'ROOT_R1_GEOMETRY_READBACK.json')['status']=='pass';_install_gt_read_guard();expert=SpatialExpert(ROOT/EXPERT_SNAPSHOT);before=state_hash(expert.model.state_dict());rr={r['probe_alias']:r for r in read(BASE/'ROSTER_LOCK.json')['records']};refs={cid:[] for cid in rr};files=[];calls=0
 for j in read(BASE/'INPUT_JOB_LOCK.json')['records']:
  verify();assert sha(ROOT/j['image']['path'])==j['image']['sha256'];rgb=np.load(ROOT/j['image']['path'],allow_pickle=False);rec=rr[j['case_id']];subject=read(ROOT/rec['subject_path']);assert sha(ROOT/rec['subject_path'])==rec['subject_sha256'] and subject['GT_read'] is False;out=_observe_locked_position(expert,rgb,j['position'],j['physical_frame_id'],subject['parses']);calls+=bool(out.get('raw_boxes_cxcywh'));p=BASE/'dino'/(j['job_id']+'.json');put(p,dict(status='complete',job_id=j['job_id'],case_id=j['case_id'],dataset=j['dataset'],position=j['position'],physical_frame_id=j['physical_frame_id'],original_pixel_sha256=j['original_pixel_sha256'],output=out,GT_read=False));f=dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size);files.append(f)
  if out['decision_status']=='accepted':refs[j['case_id']].append(reference(out['selected_box_cxcywh'],j,'DINO_original',f['sha256']))
  put(BASE/'STATUS.json',dict(status='running',stage='DINO_geometry',done=len(files),total=128,count=32,GT_read=False),replace=True)
 assert state_hash(expert.model.state_dict())==before;put(BASE/'DINO_REFERENCE_LOCK.json',dict(status='locked',count=32,records=[dict(case_id=k,anchors=v) for k,v in refs.items()],GT_read=False));put(BASE/'DINO_GLOBAL_BARRIER.json',dict(status='complete',count=32,responses=128,actual_DINO_calls=calls,files=files,frozen_state_sha256=before,GT_read=False));put(BASE/'ROOT_DINO_READBACK.json',dict(status='pass',count=32,responses=128,same_observation_pixels=True,original_parser_and_admission=True,GT_read=False))
def prepare():
 verify();assert read(BASE/'ROOT_DINO_READBACK.json')['status']=='pass';rr=read(BASE/'ROSTER_LOCK.json');source=read(BASE/'SOURCE_CAPTURE_BARRIER.json');template=read(PAPER/'arms/B/trials/G32/DESIGN_LOCK.json');sourcemap={r['case_id']:r for r in source['files']}
 for arm in ARMS:
  a=BASE/'arms'/arm;a.mkdir(parents=True,exist_ok=False);put(a/'ROSTER_LOCK.json',rr);put(a/'SOURCE_CAPTURE_BARRIER.json',source);put(a/'SCOPE_CONFIG.json',dict(recipe='lite_fullstep',label='R1-only fixed geometry '+arm,GT_read=False));put(a/'teacher/G32/ROOT_READBACK.json',dict(status='pass',count=32,reference_source_sha256=sha(BASE/(arm+'_REFERENCE_LOCK.json')),GT_read=False));records=[];refs={r['case_id']:r['anchors'] for r in read(BASE/(arm+'_REFERENCE_LOCK.json'))['records']}
  for rec in rr['records']:records.append(dict(case_id=rec['probe_alias'],dataset=rec['dataset'],source_case_sha256=sourcemap[rec['probe_alias']]['source_case_sha256'],anchors=refs[rec['probe_alias']],GT_read=False))
  t=a/'trials/G32';t.mkdir(parents=True);put(t/'REFERENCE_LOCK.json',dict(status='locked',count=32,records=records,GT_read=False));(t/'TEMPORAL_LOCK.json').write_bytes((BASE/'TEMPORAL_LOCK.json').read_bytes());d=copy.deepcopy(template);d.update(count=32,cohort='fixed exposed core first16 per domain',scope='R1 independent geometry32 '+arm,trial_id='G32',family_reuse_from={ds:{m:None for m in values} for ds,values in d['configs'].items()});put(t/'DESIGN_LOCK.json',d)
  paths=[Path(__file__),ROOT/'scripts/stvg_overnight_student_v1.py',ROOT/'scripts/run_stvg_instance_student32_v1.py',ROOT/'scripts/run_stvg_soft_ep_coordinate_trial003.py',ROOT/'scripts/stvg_soft_ep_coordinate_fit002.py',ROOT/'scripts/stvg_spatial_lite_fit_v1.py',ROOT/'scripts/stvg_soft_ep_response_v1.py',ROOT/'scripts/audit_stvg_spatial_lite_fit002.py',ROOT/'scripts/review_stvg_lite_complete002.py',BASE/'EXECUTION_RUNTIME002.json',a/'ROSTER_LOCK.json',a/'SOURCE_CAPTURE_BARRIER.json',a/'SCOPE_CONFIG.json',a/'teacher/G32/ROOT_READBACK.json',t/'REFERENCE_LOCK.json',t/'TEMPORAL_LOCK.json',t/'DESIGN_LOCK.json'];put(t/'RUNTIME.json',dict(status='locked',trial_id='G32',pins={str(p.relative_to(ROOT)):sha(p) for p in paths},protected={},GT_read=False))
 put(BASE/'STUDENT_PREPARATION_COMPLETION.json',dict(status='complete',count=32,arms=list(ARMS),GT_read=False))
def configure(arm):
 from scripts import stvg_overnight_student_v1 as s
 s.BASE=BASE/'arms'/arm;s.CORE=PAPER;s.N=32;s.CFG=read(s.BASE/'SCOPE_CONFIG.json');s.common=types.SimpleNamespace(BASE=BASE,verify=verify,read=read,write=put,sha=sha);return s
def student(mode,arm):
 s=configure(arm)
 if mode=='root':return s.root()
 from scripts import stvg_pipeline_speed_v1 as speed
 ns=dict(s.__dict__);ns['speed']=types.SimpleNamespace(BASE=speed.BASE,Engine=lambda label,enabled:speed.Engine('r1_geometry32_'+label,enabled));exec(compile(inspect.getsource(s.worker),__file__,'exec'),ns);return ns['worker']()
def score():
 import torch,numpy as np
 from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
 from scripts import review_stvg_sparse_r1_v2 as truth,score_stvg_native_instance_nativeclock005 as metric
 from scripts.score_stvg_temporal_t2_v1 import subset_aggregate
 verify();bars={arm:read(BASE/'arms'/arm/'trials/G32/GLOBAL_PREDICTION_BARRIER.json') for arm in ARMS};assert all(b['count']==32 and b['status']=='complete' for b in bars.values());put(BASE/'EXPERIMENT_GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,arms={arm:sha(BASE/'arms'/arm/'trials/G32/GLOBAL_PREDICTION_BARRIER.json') for arm in ARMS},GT_read=False));rr=read(BASE/'ROSTER_LOCK.json')['records'];cases={r['probe_alias']:load_case(r)[0] for r in rr};targets,bindings=truth._truths_after_seal(cases);off,ind,_=truth._metric_functions();baseline={r['case_id']:r for r in read(BASE/'BASELINE_BARRIER.json')['files']};maps={arm:{r['case_id']:r for r in bar['files']} for arm,bar in bars.items()};rows=[];usage=[];quality=[];jobs=read(BASE/'INPUT_JOB_LOCK.json')['records'];refmaps={arm:{r['case_id']:{a['position']:a for a in r['anchors']} for r in read(BASE/(arm+'_REFERENCE_LOCK.json'))['records']} for arm in ARMS}
 def overlap(a,b):
  v=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod();return float(v/max(1e-12,np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-v))
 for rec in rr:
  cid=rec['probe_alias'];case=cases[cid];f=baseline[cid];assert sha(ROOT/f['path'])==f['sha256'];old=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False);preds={'Frozen':old['arms']['F'],'T2':old['arms']['T'],'Current_WF':old['arms']['TS_M2']};support={j['position'] for j in jobs if j['case_id']==cid}
  for arm in ARMS:
   f=maps[arm][cid];assert sha(ROOT/f['path'])==f['sha256'];d=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False);preds[arm]=d['arms']['TS_M2'];fit=d['fits']['M2'];usage.append(dict(case_id=cid,dataset=rec['dataset'],arm=arm,reference_count=len(refmaps[arm][cid]),actual_fit=fit is not None,accepted_steps=fit['optimizer_steps'] if fit else 0,proposed_steps=fit['proposed_optimizer_steps'] if fit else 0))
  for arm,p in preds.items():
   v,_=metric._score_student_arm(p,case,targets[cid],support,off,ind);v.update(R_vIoU_at_0p3=float(v['m_vIoU']>=.3),R_vIoU_at_0p5=float(v['m_vIoU']>=.5));rows.append(dict(case_id=cid,dataset=rec['dataset'],arm=arm,metrics=v));assert p['interval']==preds['T2']['interval'] if arm!='Frozen' else True
  gt=metric._truth_map(targets[cid]);scale=np.array([case['input']['width'],case['input']['height']]*2,float)
  for j in [j for j in jobs if j['case_id']==cid]:
   fid,pos=j['physical_frame_id'],j['position'];q=dict(case_id=cid,dataset=rec['dataset'],slot=j['slot'],GT_known=fid in gt,reference={})
   for arm in ARMS:
    a=refmaps[arm][cid].get(pos);value=None
    if a is not None and fid in gt:
     z=np.asarray(a['box_cxcywh']);bb=np.r_[z[:2]-z[2:]/2,z[:2]+z[2:]/2]*scale;value=overlap(bb,np.asarray(gt[fid]))
    q['reference'][arm]=dict(available=a is not None,IoU=value,success_at_0p5=None if value is None else value>=.5)
   quality.append(q)
 summary,paired=subset_aggregate(rows);put(BASE/'POSTSEAL_RESULTS.json',dict(status='complete',count=32,rows=rows,summary=summary,paired=paired,usage=usage,reference_quality=quality,GT_bindings=bindings,GT_read=True,development_not_confirmation=True));put(BASE/'CPU_COMPLETION.json',dict(status='complete',count=32,result_sha256=sha(BASE/'POSTSEAL_RESULTS.json'),GT_read=True))
 # Resource decision only; it cannot promote a production method.
 candidates=[]
 for arm in ['DIRECT','HEADS']:
  ps=[p for p in paired if p['comparison']==arm+'_minus_Current_WF' and p['metric']=='m_vIoU'];assert len(ps)==2;means=[p['mean_delta_pp'] for p in ps];ok=sum(means)/2>=.1 and min(means)>=0 and any(p['paired_source_t95_interval_pp'][0]>0 for p in ps);candidates.append(dict(arm=arm,mean_increment_pp=sum(means)/2,practical_gate_pass=bool(ok)))
 put(BASE/'STAGE1_DECISION.json',dict(status='complete',count=32,candidates=candidates,second_stage_authorized=not any(x['practical_gate_pass'] for x in candidates),not_method_promotion=True,GT_read=True))
if __name__=='__main__':
 if sys.argv[1] in ['r1root','dino','prepare','score']:globals()[sys.argv[1]]()
 else:student(sys.argv[1],sys.argv[2])
