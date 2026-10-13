"""Fixed OPD factorial components; original science and sources remain intact."""
import collections,fcntl,gc,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha,load,save
from scripts.stvg_opd_LN_state_intervention_common_v1 import digest,noise
BASE=ROOT/'artifacts/stvg_opd_p7a_component_simplification_v1'
OUT=ROOT/'results/stvg_opd_p7a_component_simplification/2026-10-13'
TUNE=ROOT/'artifacts/decota_spatial_opd_tuning_v1'
HC=ROOT/'artifacts/stvg_opd_hc2_coordinate_v1'
Q='spatial.query_residual'
ARMS=['joint_carry','LN_carry','joint_reset','LN_reset']
BOOT=10000;SEED=20261013;EPS=1e-12
SCIENCE=['protocols/stvg_opd_p7a_component_simplification_v1.md',
 'scripts/stvg_opd_p7a_component_common_v1.py','scripts/run_stvg_opd_p7a_component_v1.py',
 'scripts/score_stvg_opd_p7a_component_v1.py','vg_tta/stvg_opd_p7a_component_state_audit_v1.py']

def clone(s):return {n:v.detach().cpu().clone() for n,v in s.items()}
def incoming(source,previous,arm):
 import torch
 assert arm in ARMS
 s=clone(source if previous is None or arm.endswith('reset') else previous)
 s[Q]=torch.zeros_like(s[Q]);return s
def commit(source,initial,final,arm,alpha):
 import torch
 assert arm in ARMS and 0<alpha<1
 if arm.endswith('reset'):return incoming(source,None,arm)
 return {n:torch.zeros_like(v) if n==Q else v+(final[n].cpu()-v)*alpha for n,v in initial.items()}
def config(ds,arm):
 c=dict(read(ROOT/'methods/decota_spatial_opd_v1/configs.json')['datasets'][ds]['config'])
 if arm.endswith('reset'):c['writeback']=0.
 return c
def verify():
 r=read(BASE/'RUNTIME.json')
 assert sha(BASE/'DESIGN_LOCK.json')==r['design_sha256'] and sha(BASE/'CPU_CONTRACTS.json')==r['contracts_sha256']
 for n,h in {**r['pins'],**r['bindings'],**r['protected']}.items():assert sha(ROOT/n)==h,n
 return r
def standby():
 from scripts.pause_stvg_opd_for_geometry_v1 import process
 for e in read(BASE/'BASELINE_STANDBY_IDENTITY.json')['processes']:
  a=process(e['pid']);assert a['state']=='T' and a['start_ticks']==e['start_ticks'] and a['command']==e['command']
def gpu():
 import torch
 from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
 from scripts.decota_paper_common_v1 import guard
 verify();standby();install_clean_loader();sys.addaudithook(guard)
 torch.set_num_threads(2);torch.manual_seed(20260920);np.random.seed(20260920)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 f=(BASE/'priority_GPU.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f
def objective(f,ex):
 from vg_tta.stvg_opd_p3_scope_precision001 import audit
 from vg_tta.stvg_opd_paper_component_audit_v1 import audit as original
 return audit(f,ex,original)
def projection(ds,base,records):
 from methods.decota_final_simplified_v1.objectives import project
 from scripts.stvg_opd_p6_existing_common001 import config as temporal_config,check_projection
 c=temporal_config(ds);logits=base.zero['logits'];actions=base.zero['actions']
 e=project(logits,actions,records,c);err=check_projection(logits,actions,records,e,c)
 iv=[v['interval'] for v in e['offsets']]
 return [min(x[0] for x in iv),max(x[1] for x in iv)],dict(status='pass',max_error=err,config=c.to_dict())
def contracts():
 import torch
 from vg_tta.stvg_opd_p7a_component_state_audit_v1 import selftest
 assert not torch.cuda.is_initialized();detail=selftest()
 s={Q:torch.zeros(256),**{f'LN{i}':torch.ones(256)*(i+1) for i in range(6)}}
 f={n:v+.2 for n,v in s.items()}
 for a in ARMS:
  i=incoming(s,None,a);c=commit(s,i,f,a,.125)
  assert not torch.count_nonzero(c[Q])
  for n in s:
   if n!=Q:assert torch.equal(c[n],s[n] if a.endswith('reset') else s[n]+(f[n]-s[n])*.125)
  assert not torch.count_nonzero(incoming(s,c,a)[Q])
 return dict(status='pass',scope='synthetic scoped Adam/absolute LN source/carry/reset/Query zero; not actual GPU qualification',
  detail=detail,CPU_only=True,GT_read=False,time=time.time())
def prepare():
 import torch
 from scripts.stvg_opd_legacy_input_bridge003 import normalized
 from scripts.pause_stvg_opd_for_geometry_v1 import process
 torch.set_num_threads(2);assert not torch.cuda.is_initialized()
 assert not BASE.exists() and not OUT.exists();BASE.mkdir();OUT.mkdir(parents=True)
 design=read(TUNE/'DESIGN_LOCK.json');cfgs=read(ROOT/'methods/decota_spatial_opd_v1/configs.json')['datasets']
 records=[];stages={};bindings={};oldfitcount=0
 for ds in ['hc2','vidstg']:
  dd=design['datasets'][ds];qs=dd['development_parents'];orders=dd['refine_orders'];assert len(qs)==32
  folder=HC/'trials/cfg_e8735d3fe1dd4bc2' if ds=='hc2' else TUNE/'trials/vidstg/refine_10'
  trial=read(folder/'CONFIG.json');bar=read(folder/'PREDICTION_BARRIER.json')
  assert trial['parents']==qs and trial['orders']==orders and trial['config']==cfgs[ds]['config']
  assert bar['status']=='sealed' and bar['cells']==64 and not bar['GT_read']
  inputbase=ROOT/bar['input_base'] if 'input_base' in bar else (HC if ds=='hc2' else TUNE)
  assert not bar.get('exact_history_reused',False)
  parentids={q:len(records)+j for j,q in enumerate(qs)}
  byquery={};source_state=None;parent_sources=set()
  plan=read(ROOT/'artifacts/decota_paper_experiments_v1'/ds/'PLAN.json')
  for order,seq in orders.items():
   assert len(seq)==32 and set(seq)==set(qs)
   previous=None;prevhash=None
   for at,q in enumerate(seq):
    p=folder/order/f'{at:05}.pt';h=sha(p);z=load(p);rc=read(p.with_suffix('.json'))
    key=str(p.relative_to(ROOT if ds=='hc2' else TUNE))
    assert bar['files'][key]==h==rc['sha256'] and not rc['GT_read'] and not z['GT_read'] and not z['fit']['GT_used']
    assert z['parent']==q and z['config']==trial['config'] and z['previous_payload_sha256']==prevhash
    ip=inputbase/z['input']['path'];assert sha(ip)==z['input']['sha256'];inp=normalized(load(ip))
    assert inp['dataset']==ds and inp['frame_ids']==plan['rows'][q]['frame_ids']
    if source_state is None:source_state=clone(z['fit']['initial'])
    expect=incoming(source_state,previous,'joint_carry')
    assert all(torch.equal(v,z['fit']['initial'][n]) for n,v in expect.items())
    c=commit(source_state,expect,z['fit']['state'],'joint_carry',trial['config']['writeback'])
    assert all(torch.equal(v,z['committed'][n]) for n,v in c.items())
    b=dict(path=str(p.relative_to(ROOT)),sha256=h,input_path=str(ip.relative_to(ROOT)),input_sha256=sha(ip),arrival=at,order=order)
    if q not in byquery:byquery[q]=dict(index=parentids[q],dataset=ds,source=dd['source'],query_ordinal=q,
     parent_id=parentids[q],source_model_sha256=inp['source_model_state_sha256'],pixel_sha256=inp['pixel_sha256'],
     frame_ids_sha256=digest(inp['frame_ids']),evidence_sha256=digest(inp['expert']),originals={},input=b)
    rr=byquery[q];assert rr['pixel_sha256']==inp['pixel_sha256'] and rr['evidence_sha256']==digest(inp['expert'])
    rr['originals'][order]=b;bindings[b['path']]=h;bindings[b['input_path']]=b['input_sha256']
    parent_sources.add(plan['rows'][q]['source']);previous=c;prevhash=h;oldfitcount+=1
  assert len(parent_sources)==32
  records.extend(byquery[q] for q in qs)
  sf=BASE/'source_initial'/f'{ds}.pt';save(sf,source_state);bindings[str(sf.relative_to(ROOT))]=sha(sf)
  stages[ds]=dict(dataset=ds,source=dd['source'],parents=qs,orders=orders,
   configs={a:config(ds,a) for a in ARMS},source_initial_path=str(sf.relative_to(ROOT)),source_initial_sha256=sha(sf))
  for p in [folder/'CONFIG.json',folder/'PREDICTION_BARRIER.json']:bindings[str(p.relative_to(ROOT))]=sha(p)
 assert len(records)==64 and oldfitcount==128
 write(BASE/'DESIGN_LOCK.json',dict(status='locked_before_new_GPU_and_GT',arms=ARMS,records=records,stages=stages,
  development_parents=64,historically_exposed_development=True,confirmation_used=False,
  qualification_fits=64,qualification_captures=16,formal_fits=512,formal_captures=128,formal_rounds=12800,
  qualification_rounds=1600,primary='P7-A I1 parent-macro mvIoU',secondary='Native I0 mvIoU',
  bootstrap=BOOT,seed=SEED,new_DINO_calls=0,no_alternate_objective=True,no_parameter_search_this_stage=True,
  production_promoted=False,monitor_recreated=False,time=time.time()))
 write(BASE/'CPU_CONTRACTS.json',contracts())
 identities=[process(n) for n in [509176,3091846]];assert all(x['state']=='T' for x in identities)
 write(BASE/'BASELINE_STANDBY_IDENTITY.json',dict(status='pass',processes=identities,GT_read=False,time=time.time()))
 shared=['scripts/run_decota_paper_main_v1.py','scripts/stvg_opd_legacy_input_bridge003.py',
  'vg_tta/decota_spatial_opd_tunable_v1.py','vg_tta/stvg_opd_paper_ablations_v1.py','vg_tta/stvg_opd_p3_scope_precision001.py',
  'vg_tta/stvg_opd_common_state_geometry_arithmetic004.py','methods/decota_final_simplified_v1/objectives.py',
  'scripts/stvg_opd_p6_existing_common001.py','methods/decota_final_simplified_v1/config.py']
 shared += [str(p.relative_to(ROOT)) for p in (ROOT/'vg_tta').glob('stvg_opd_frozen_*0*.py')]
 protected={n:sha(ROOT/n) for n in ['methods/CURRENT_METHOD.json','methods/decota_spatial_opd_v1/configs.json',
  'artifacts/stvg_opd_temporal_guided_spatial_candidate_v1/P7_A/FINAL_COMPLETION.json',
  'artifacts/stvg_opd_supplement_v1/baselines/USER_CANCEL_REMAINING_FORMAL_RECEIPT.json']}
 write(BASE/'RUNTIME.json',dict(status='locked_before_new_GPU_GT',pins={n:sha(ROOT/n) for n in SCIENCE+shared},bindings=bindings,
  protected=protected,design_sha256=sha(BASE/'DESIGN_LOCK.json'),contracts_sha256=sha(BASE/'CPU_CONTRACTS.json'),GT_read=False,time=time.time()))
 write(BASE/'SOURCE_STATE_CUSTODY_READBACK.json',dict(status='pass',old_joint_carry_complete_fits=128,unique_inputs=64,
  parent_sources=64,original_development_and_orders_exact=True,source_state_coordinates=3584,CPU_only=True,GT_read=False,time=time.time()))
 status(BASE/'STATUS.json',dict(status='prepared_actual_qualification_then512_formal',done=0,total=512,GT_read=False,time=time.time()))
 verify();standby()
