"""Frozen paper method: T2 observation clock, fixed B, original sparse OPD.

Independent paper entry. Historical CURRENT and experiment sources immutable.
"""
from pathlib import Path
import ast,copy,gc,hashlib,inspect,json,os,sys,time,types
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,sha,write,guard
BASE=ROOT/'artifacts/stvg_paper_t2_b_v1';CORE=ROOT/'artifacts/stvg_spatial_controls256_v1';T2=ROOT/'artifacts/stvg_temporal_t2_vs_t4_v1';NATIVE=ROOT/'artifacts/stvg_factorial_native256_captionfix002'
MANIFEST=ROOT/'methods/paper/STVG_T2_B_EPISODIC_V1.json'
def put(p,d,replace=False):return write(p,d,mutable=replace)
def verify():
 r=read(BASE/'EXECUTION_RUNTIME001.json');assert time.time()<r['deadline_epoch']
 for p,h in r['pins'].items():assert sha(ROOT/p)==h,p
 return r

def setup():
 from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
 from scripts import stvg_temporal_lite_fit_v1 as lite
 BASE.mkdir(exist_ok=False);rr=read(CORE/'ROSTER_LOCK.json');assert rr['count']==256
 for p in (T2/'GLOBAL_PREDICTION_BARRIER.json',NATIVE/'arms/B/trials/G32/GLOBAL_PREDICTION_BARRIER.json'):assert read(p)['status']=='complete' and read(p)['count']==256
 for name in ('ROSTER_LOCK.json','SOURCE_CAPTURE_BARRIER.json','SOURCE_GEOMETRY.json','EVIDENCE_MODIFICATION_LOCK.json'):(BASE/name).write_bytes((CORE/name).read_bytes())
 (BASE/'temporal').mkdir();(BASE/'temporal/requests').symlink_to(CORE/'temporal/requests',target_is_directory=True)
 for name in ('TEMPORAL_LOCK.json','ROOT_READBACK.json','GLOBAL_PREDICTION_BARRIER.json'):(BASE/'temporal'/name).write_bytes((T2/name).read_bytes())
 (BASE/'TEMPORAL_LOCK.json').write_bytes((T2/'TEMPORAL_LOCK.json').read_bytes())
 tt={r['case_id']:r['selected_interval'] for r in read(BASE/'TEMPORAL_LOCK.json')['records']};old={r['case_id']:r['selected_interval'] for r in read(CORE/'TEMPORAL_LOCK.json')['records']};nat={r['case_id']:r['selected_interval'] for r in read(NATIVE/'TEMPORAL_LOCK.json')['records']};jobs={b:{cid:[] for cid in tt} for b in ('T4','Native')}
 for label,b in [('T4',CORE),('Native',NATIVE)]:
  for j in read(b/'TEACHER_JOB_LOCK.json')['records']:jobs[label][j['case_id']].append(j)
 inventory=[];caps=[]
 candidates={r['case_id']:r for r in read(T2/'CANDIDATE_LOCK.json')['records']};bar=read(T2/'TEACHER_GLOBAL_BARRIER.json');responses={r['path']:r['sha256'] for r in bar['files']}
 for rec in rr['records']:
  cid=rec['probe_alias'];target=tt[cid];origin='T4' if target==old[cid] else ('Native' if target==nat[cid] else None);b=CORE if origin=='T4' else NATIVE;inventory.append(dict(case_id=cid,dataset=rec['dataset'],exact_interval_scope=origin,scope=str(b.relative_to(ROOT)) if origin else None,jobs=jobs[origin][cid] if origin else [],GT_read=False));cc=[]
  for c in candidates[cid]['candidates']:
   a,z=c['interval_frames'];over=max(0,min(z,target[1])-max(a,target[0]));union=max(z,target[1])-min(a,target[0]);cc.append((over/union,c['canonical_id']))
  key=sorted(cc,key=lambda x:(-x[0],x[1]))[0][1] if cc else None
  p=T2/'teacher'/cid/(key+'.json') if key else None
  if p is not None:assert responses[str(p.relative_to(ROOT))]==sha(p)
  if p is None or not p.exists() or not read(p)['raw_caption'].strip():
   cap=next(x for x in read(NATIVE/'CAPTION_LOCK.json')['records'] if x['case_id']==cid);p=ROOT/cap['path'];key='time_Native';assert cap['available'] and sha(p)==cap['sha256']
  assert read(p)['raw_caption'].strip();caps.append(dict(case_id=cid,candidate_id=key,path=str(p.relative_to(ROOT)),sha256=sha(p),available=True,selection='maximum interval overlap with adapted T2; original canonical-ID tie break; independently acquired Native fallback if missing',GT_read=False))
 put(BASE/'EXACT_INPUT_REUSE_INVENTORY.json',dict(status='locked',count=256,records=inventory,selection_uses_GT_or_scores=False,GT_read=False));put(BASE/'CAPTION_LOCK.json',dict(status='locked',count=256,records=caps,independent_acquisition_not_gated_by_old_T4_eligibility=True,GT_read=False))
 cfg=read(NATIVE/'arms/B/SCOPE_CONFIG.json');cfg.update(core_scope=str(BASE.relative_to(ROOT)),label='Frozen paper T2 clock plus fixed B plus original episodic sparse Gaussian OPD');b=BASE/'arms/B';b.mkdir(parents=True);put(b/'ROSTER_LOCK.json',rr);put(b/'SCOPE_CONFIG.json',cfg)
 design=read(CORE/'trials/B/DESIGN_LOCK.json');source=read(CORE/'SOURCE_RUNTIME.json');manifest=dict(status='frozen_design_pending_integrated_validation',version='STVG_T2_B_EPISODIC_V1',entry='scripts/run_stvg_paper_t2_b_v1.py',scientific_dispatch='scripts/stvg_paper_t2_b_v1.py',historical_CURRENT_unchanged_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'),source_checkpoints=source['source_checkpoints'],teacher_checkpoint='checkpoints/STVG-R1-7B',teacher_config_sha256=sha(ROOT/'checkpoints/STVG-R1-7B/config.json'),temporal=dict(candidate_count=2,rule=read(T2/'METHOD_LOCK.json')['Alternative_rule'],parameters=512,config=lite._lite_config(),decision='human simplicity preference; original T2/T4 noninferiority remains unestablished',final_interval='full original time-grid decode',clock='fractional_uniform_v2'),spatial=dict(policy='B',global_frames=4,global_visual_keep=64,local_frames=5,local_visual_keep=96,selector_code_sha256=sha(ROOT/'scripts/stvg_referent_abc_tokens002.py'),Caption_rule='original maximum overlap among T2 candidates, canonical ID ties, independent Native acquisition fallback',reference='actual WHO sparse candidate selection; source candidate forbidden; presence>=0.6',active_parameters=1792,configs=design['configs'],response_settings=design['response_settings'],recipe=design['recipe'],state='episodic Source plus fresh Adam each query; no state propagation',final_readout='complete trained decoder/LN and action gates; full native grid',fixed_sigma=True,original_reject_rollback_early_stop=True),not_adopted=['C','Global32','noCoverage','annealing','Tube completion','new Decoder-Aligned target optimizer','online LN writeback'],evaluation=dict(integration='core256 exposed development, HC128/Vid128',GT='only after complete prediction/root barriers; no midrun selection',next='official full roster and second Backbone only with this same frozen method; no tuning',existing_confirmation_remains_failed_and_exposed=True),implementation_pins={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'scripts/stvg_temporal_t2_v1.py',ROOT/'scripts/stvg_temporal_lite_fit_v1.py',ROOT/'scripts/stvg_referent_abc_tokens002.py',ROOT/'scripts/stvg_spatial_lite_fit_v1.py',ROOT/'scripts/stvg_soft_ep_response_v1.py',ROOT/'scripts/stvg_qwen_uniform_clock002.py']},GT_read=False)
 MANIFEST.parent.mkdir(exist_ok=True);put(MANIFEST,manifest);put(BASE/'METHOD_LOCK.json',manifest)
 paths=[MANIFEST,Path(__file__),ROOT/'scripts/run_stvg_paper_t2_b_v1.py',ROOT/'scripts/score_stvg_paper_t2_b_v1.py',BASE/'ROSTER_LOCK.json',BASE/'SOURCE_CAPTURE_BARRIER.json',BASE/'TEMPORAL_LOCK.json',BASE/'CAPTION_LOCK.json',BASE/'EXACT_INPUT_REUSE_INVENTORY.json',BASE/'METHOD_LOCK.json',ROOT/'methods/CURRENT_METHOD.json',T2/'GLOBAL_PREDICTION_BARRIER.json',NATIVE/'arms/B/trials/G32/GLOBAL_PREDICTION_BARRIER.json']+[ROOT/p for p in manifest['implementation_pins']]
 put(BASE/'EXECUTION_RUNTIME001.json',dict(status='locked',count=256,pins={str(p.relative_to(ROOT)):sha(p) for p in paths},deadline_epoch=time.time()+43200,WHO_cap=1024,qualification_generation_cap=6,Spatial_fit_cap=256,new_Temporal_fits=0,scientific_search=False,GT_read=False));put(BASE/'STATUS.json',dict(status='prepared',stage='T2_clock_spatial_input',count=256,GT_read=False));print('PAPER_T2_B256_FROZEN',sum(r['exact_interval_scope'] is not None for r in inventory),'same interval input candidates',flush=True)

def prepare():
 from scripts import stvg_instance_spatial32_v1 as original,stvg_factorial_native256_v1 as raw
 verify();inventory=read(BASE/'EXACT_INPUT_REUSE_INVENTORY.json')['records'];same={r['case_id']:r for r in inventory if r['exact_interval_scope']};reused_jobs=[];counts=[];prop=BASE/'spatial/E1/proposals';prop.mkdir(parents=True)
 for cid,r in same.items():
  source=ROOT/r['scope'];p=source/'spatial/E1/proposals'/(cid+'.json');assert all(w['interval_half_open']==next(t['selected_interval'] for t in read(BASE/'TEMPORAL_LOCK.json')['records'] if t['case_id']==cid) for w in read(p)['windows']);(prop/(cid+'.json')).symlink_to(p.resolve());reused_jobs+=r['jobs'];counts.append(dict(case_id=cid,status='ready',exact_interval_and_E1_input_reuse=True,source_scope=r['scope'],GT_read=False))
 def fresh_roster():return [r for r in read(BASE/'ROSTER_LOCK.json')['records'] if r['probe_alias'] not in same]
 current=[None]
 def set_current(case):current[0]=case
 def rgb(request,wanted):
  ns=dict(raw.__dict__);ns.update(BASE=BASE,CURRENT_CASE=current[0]);exec(compile(inspect.getsource(raw.native_rgb),__file__,'exec'),ns);return ns['native_rgb'](request,wanted)
 src=inspect.getsource(original.prepare);src=src.replace('from scripts.run_stvg_weightfree_expand_v1 import selected_rgb','from scripts.run_stvg_weightfree_expand_v1 import selected_rgb as unused_old_decoder');start=src.index('oldpath=LITE/');end=src.index('pool=builder.build_candidate_pool',start);src=src[:start]+'cache={};mcache={}\n        '+src[end:];src=src.replace("[('E0',windows),('E1',updated)]","[('E1',updated)]");src=src.replace("for rec in read(BASE/'ROSTER_LOCK.json')['records']:","for rec in fresh_roster():");needle="cid=rec['probe_alias'];case,cp=load_case(rec);";src=src.replace(needle,needle+'set_current(case);');src=src.replace('jobs=[];files={};counts=[]','jobs=list(reused_jobs);files={};counts=list(reused_counts)')
 class Count(ast.NodeTransformer):
  def visit_Constant(self,n):return ast.copy_location(ast.Constant(256),n) if type(n.value)is int and n.value==32 else n
 ns=dict(original.__dict__);ns.update(BASE=BASE,SP=BASE/'spatial',verify=verify,selected_rgb=rgb,set_current=set_current,fresh_roster=fresh_roster,reused_jobs=reused_jobs,reused_counts=counts);exec(compile(ast.fix_missing_locations(Count().visit(ast.parse(src))),__file__,'exec'),ns);ns['prepare']()

def input_root():
 import numpy as np
 verify();bar=read(BASE/'spatial/CANDIDATE_INPUT_BARRIER.json');jobs=read(BASE/'spatial/TEACHER_JOB_LOCK.json')['records'];assert bar['count']==256 and len(bar['case_records'])==256 and len(jobs)<=1024
 from collections import Counter
 cc=Counter(j['case_id'] for j in jobs);assert all(v==4 for v in cc.values())
 assert all(r['status']=='unknown_insufficient_centers' for r in bar['case_records'] if not cc[r['case_id']])
 times={r['case_id']:r['selected_interval'] for r in read(BASE/'TEMPORAL_LOCK.json')['records']}
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 for j in jobs:
  assert j['prompt_sha256']==hashlib.sha256(j['prompt'].encode()).hexdigest();pool=read(BASE/'spatial/E1/proposals'/(j['case_id']+'.json'));w=next(w for w in pool['windows'] if w['slot']==j['center_slot']);assert w['interval_half_open']==times[j['case_id']] and w['center_physical_frame_id']==j['center_physical_frame_id']
  for v in j['videos']:
   p=ROOT/v['path'];a=np.load(p,allow_pickle=False);assert sha(p)==v['sha256'] and hashlib.sha256(a.tobytes()).hexdigest()==v['tensor_sha256']
 put(BASE/'TEACHER_JOB_LOCK.json',dict(status='locked',count=256,job_count=len(jobs),records=jobs,GT_read=False));put(BASE/'SPATIAL_INPUT_HANDOFF.json',dict(status='complete_pending_native_evidence',count=256,job_count=len(jobs),GT_read=False));put(BASE/'ROOT_INPUT_READBACK.json',dict(status='pass',count=256,T2_actual_observation_intervals=True,all_source_pixels_job_bindings=True,job_count=len(jobs),no_job_sources=256-len(cc),GT_read=False))

def features():
 import numpy as np,torch
 from scripts import run_stvg_spatial_controls256_native_v1 as g
 verify();sys.addaudithook(guard);g.BASE=BASE;m,p,device,proof,seconds=g.load();inventory={r['case_id']:r for r in read(BASE/'EXACT_INPUT_REUSE_INVENTORY.json')['records']};files=[];fresh=reuse=0
 for j in read(BASE/'TEACHER_JOB_LOCK.json')['records']:
  r=inventory[j['case_id']];oldpath=ROOT/r['scope']/'native_cache'/j['job_id']/'native.npz' if r['scope'] else None;frozen=None
  if oldpath is not None and oldpath.exists():
   prior=read(oldpath.with_name('POLICIES.json'));assert prior['job_sha256']==hashlib.sha256(json.dumps(j,sort_keys=True).encode()).hexdigest() and sha(oldpath)==prior['native_sha256']
   with np.load(oldpath,allow_pickle=False) as a:frozen=a['features'].copy()
  b,ps,e,pos,f,runs,pol,v,ma,gr,q,c,available=g.prepare(m,p,device,j,frozen=frozen);assert available;g.cache_write(j,proof,ps,f,pol,v,ma,gr,q,c,available);reuse+=frozen is not None;fresh+=frozen is None;path=BASE/'native_cache'/j['job_id']/'POLICIES.json';files.append(dict(case_id=j['case_id'],path=str(path.relative_to(ROOT)),sha256=sha(path)));put(BASE/'FEATURE_STATUS.json',dict(status='running',done=len(files),total=len(read(BASE/'TEACHER_JOB_LOCK.json')['records']),count=256,GT_read=False),replace=True);del b,e,pos,f,runs,pol,frozen;gc.collect();torch.cuda.empty_cache()
 put(BASE/'FEATURE_BARRIER.json',dict(status='complete',count=256,files=files,new_encoder_calls=fresh,exact_visual_reuses=reuse,policies_recomputed_with_exact_T2_Caption=True,model_load_seconds=seconds,GT_read=False))

def configure(module):
 verify();scope=BASE/'arms/B';module.BASE=scope;module.CORE=BASE;module.N=256;module.CFG=read(scope/'SCOPE_CONFIG.json');module.common=types.SimpleNamespace(BASE=BASE,CORE=CORE,verify=verify,deadline=lambda:read(BASE/'EXECUTION_RUNTIME001.json')['deadline_epoch'],read=read,write=put,sha=sha);return module

def arm_stage(stage):
 if stage in ('jobs','qualification','teacher','teacher_root'):
  from scripts import stvg_overnight_global_v1 as g
  configure(g);return {'jobs':g.jobs,'qualification':g.qualify,'teacher':g.teacher,'teacher_root':g.teacher_root}[stage]()
 from scripts import stvg_overnight_student_v1 as s
 configure(s)
 if stage=='student_worker':
  from scripts import stvg_pipeline_speed_v1 as speed
  ns=dict(s.__dict__);ns['speed']=types.SimpleNamespace(BASE=speed.BASE,Engine=lambda label,enabled:speed.Engine('paper_t2_b_'+label,enabled));exec(compile(inspect.getsource(s.worker),__file__,'exec'),ns);return ns['worker']()
 return {'student_prepare':s.prepare,'student_root':s.root}[stage]()
if __name__=='__main__':
 stage=sys.argv[1]
 if stage in ('setup','prepare','input_root','features'):globals()[stage]()
 else:arm_stage(stage)
