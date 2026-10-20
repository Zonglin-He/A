"""Frozen five-row, two-direction 128-parent evaluation; label-free predictions.

HC reuses the exact previously frozen configuration/cohort outputs after an
end-to-end compatibility case. Reverse inference is new. No model selection.
"""
from pathlib import Path
import os,sys,json,time,gc,hashlib,collections,traceback
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
D=R/'artifacts/stvg_spatial_strong_bidirectional256_v1'
P=R/'results/stvg_error_state/2026-10-20/spatial_strong_bidirectional256'
OLD=R/'artifacts/stvg_temporal_native_stage128_v1'
PRE=R/'artifacts/stvg_referdino_condition_hc128_v1'
FIN=R/'artifacts/stvg_referdino_finalize_v1'
ROWS=['Source','Spatial_only','Temporal_only','Full','Without_spatial_adaptation']

def read(p):return json.loads(Path(p).read_text())
def put(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def load(p):
 import torch
 return torch.load(p,map_location='cpu',weights_only=False)
def save(p,x):
 import torch
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');torch.save(x,tmp);os.replace(tmp,p)
 return dict(path=str(p.relative_to(R)),sha256=sha(p),bytes=p.stat().st_size)
def prepare():
 assert not (D/'PLAN_LOCK.json').exists()
 cfg=read(R/'methods/stvg_referdino_opd_v1/config.json');cal=read(R/'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json')['source_domains']
 records=[];cohorts={};configs={}
 for ds in ['hc2','vidstg']:
  raw=read(R/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']
  if ds=='hc2':selected=[(r['alias'],r['query_ordinal'])for r in read(OLD/'PLAN_LOCK.json')['records']]
  else:
   groups=collections.defaultdict(list)
   for r in raw:groups[r['source']].append(r)
   hh=lambda s:hashlib.sha256(('spatial-strong-bidirectional256-v1|'+s).encode()).hexdigest()
   parents=sorted(groups,key=hh)[:128]
   selected=[(f'BV{i+1:03}',min(groups[s],key=lambda r:hh(str(r['key'])))['ordinal'])for i,s in enumerate(parents)]
  assert len(selected)==128 and len({raw[q]['source']for a,q in selected})==128
  for a,q in selected:
   sf=R/f'artifacts/decota_paper_experiments_v1/{ds}/subjects/{q:05}.json';r=raw[q]
   rec=dict(alias=a,dataset=ds,query_ordinal=q,parent=r['source'],subject_path=str(sf.relative_to(R)),subject_sha256=sha(sf),historically_exposed=True)
   if ds=='hc2':rec['reuse']={str(p.relative_to(R)):sha(p)for p in [PRE/'inputs'/f'{a}.pt',FIN/'Live/predictions'/f'{a}.pt',OLD/'predictions'/f'{a}.pt']}
   records.append(rec)
  source='vidstg'if ds=='hc2'else'hc2';cc=dict(cfg,dataset=ds,source_dataset=source,E_calibration=cal[source],source_checkpoint='checkpoints/TASTVG_VidSTG.pth'if ds=='hc2'else'checkpoints/TASTVG_HCSTVG2.pth');configs[ds]=cc
  cohorts[ds]=dict(parents=128,queries=128,full_queries=len(raw),full_parents=len({r['source']for r in raw}),historical_exposure='legacy manifests already exposed; HC128 also directly used for method/temperature selection; reverse is a locked panel, not certified unseen',manifest_sha256=sha(R/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json'))
 own=['scripts/run_stvg_spatial_strong_bidirectional256_v1.py','scripts/score_stvg_spatial_strong_bidirectional256_v1.py','protocols/stvg_spatial_strong_bidirectional256_v1.md']
 own +=[str(f.relative_to(R))for f in (R/'methods/stvg_referdino_opd_v1').glob('*.py')]+['methods/stvg_referdino_opd_v1/config.json','vg_tta/stvg_temporal_hires128_v1.py']
 p=dict(status='locked',records=records,cohorts=cohorts,configs=configs,seeds=[41,42,43],rows=ROWS,GT_read=False,selection='HC exact prior128; Vid parents SHA256 rank then one query SHA256 rank, no outcome filter',Strong='0.5 posterior448_final + 0.5 mean of native224 second-pass six layer posteriors; offset envelope before layer mixing',sampling_keys='HC original RC aliases; reverse BV aliases',maximum_queries=256,new_inference_max_wall_seconds=43200,created_epoch=time.time(),pins={f:sha(R/f)for f in own},checkpoints={cc['source_checkpoint']:sha(R/cc['source_checkpoint'])for cc in configs.values()},protected={f:sha(R/f)for f in ['methods/CURRENT_METHOD.json','methods/RESEARCH_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json']if(R/f).exists()})
 put(D/'PLAN_LOCK.json',p);put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','created_epoch']});print('LOCKED',256,flush=True)
def verify():
 p=read(D/'PLAN_LOCK.json')
 for f,h in {**p['pins'],**p['protected'],**p['checkpoints']}.items():assert sha(R/f)==h,f
 return p

def source_for(ds):
 import torch
 from methods.decota_final_simplified_v1._tastvg_load import official_imports,load_model_on_device
 official_imports();pipeline=sys.modules['models.pipeline'];old=pipeline.load_json;seen=[]
 def empty(path):
  assert str(path).endswith(('/annos/train.json','/annos/test.json'));seen.append(str(path));return {}
 pipeline.load_json=empty
 src='vidstg'if ds=='hc2'else'hcstvg2';runtime='vidstg'if ds=='hc2'else'hc-stvg2';ck='TASTVG_VidSTG.pth'if ds=='hc2'else'TASTVG_HCSTVG2.pth'
 try:m,_,_=load_model_on_device(src,R/'artifacts/tastvg_runtime'/runtime,checkpoint=R/'checkpoints'/ck,device='cuda',source_dataset=src)
 finally:pipeline.load_json=old
 assert len(seen)==2 and m.verb_label=={}and m.verb_label2=={}
 return m.eval().requires_grad_(False)

def capture(m,frames,ids,row,cfg,check):
 import torch,numpy as np
 from methods.stvg_referdino_opd_v1._source_forward import frozen_forward
 from methods.decota_final_simplified_v1.objectives import prediction
 from methods.decota_final_simplified_v1.tensors import detached
 from methods.stvg_referdino_opd_v1.temporal import probabilities,mass_table,final_mbr
 from vg_tta.stvg_temporal_hires128_v1 import resolution
 from methods.decota_final_simplified_v1.observations import SpatialExpert,observations
 from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
 from methods.stvg_referdino_opd_v1.initialization import encode,estimate,apply
 from methods.stvg_referdino_opd_v1.experts import positions8
 from methods.stvg_referdino_opd_v1.teacher_bridge import refer_native
 check();cost={};t=time.perf_counter();m.cuda();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
 batch,recs,base=frozen_forward(m,frames,row);native=prediction(base.zero['logits'],base.zero['boxes'],recs,ids);torch.cuda.synchronize();cost['Source224_seconds']=time.perf_counter()-tick
 inputs=detached(base.spatial_inputs,'cpu');src=native['boxes'].detach().cpu();logits=detached(base.zero['logits'],'cpu');tick=time.perf_counter()
 with torch.no_grad(),torch.autocast('cuda',enabled=False):ll=[m.temp_embed(h).detach().cpu()for h in base.temporal_inputs]
 assert all(torch.equal(x[-1],z)for x,z in zip(ll,logits)), 'aux final must recover original'
 U,q=mass_table(probabilities(recs,logits));assert U==ids
 layerq=[mass_table(probabilities(recs,[x[k]for x in ll]))[1]for k in range(6)];assert np.array_equal(layerq[-1],q)
 native_iv=final_mbr(ids,q)['interval'];cost['Aux_heads_posterior_seconds']=time.perf_counter()-tick
 del batch,base;gc.collect();torch.cuda.empty_cache();check();tick=time.perf_counter()
 with resolution(448):bb,rr,hr=frozen_forward(m,frames,row)
 hrl=detached(hr.zero['logits'],'cpu');hu,hq=mass_table(probabilities(rr,hrl));assert hu==ids
 strongq=.5*hq+.5*np.mean(layerq,axis=0);strong_iv=final_mbr(ids,strongq)['interval'];torch.cuda.synchronize();cost['Source448_Strong_seconds']=time.perf_counter()-tick
 del bb,hr;m.cpu();gc.collect();torch.cuda.empty_cache();check();tick=time.perf_counter()
 expert=SpatialExpert(R/EXPERT_SNAPSHOT);torch.cuda.synchronize();cost['DINO_load_seconds']=time.perf_counter()-tick;tick=time.perf_counter()
 ex=observations(expert,row['parses'],frames,ids,native['indices'],audit=True);torch.cuda.synchronize();cost['DINO_observation_seconds']=time.perf_counter()-tick;cost['DINO_image_forwards']=ex['new_DINO'];del expert;gc.collect();torch.cuda.empty_cache();tick=time.perf_counter()
 pool=[]
 for a in ex['anchors']['single4']:
  pos=a['position'];z=ex['observations'][('original',pos)]['probe'];assert z['accepted'];pool.append(dict(position=pos,boxes=np.asarray(z['boxes'],float).reshape(-1,4),scores=np.asarray(z['target_scores'],float)))
 cal=cfg['E_calibration'];pools=[encode(x['boxes'])-encode(src.double().numpy())[x['position']]for x in pool];est=estimate(pools,[x['scores']for x in pool],cal['scale'],cal['truncation'],1/16);E=apply(src.double().numpy(),est['delta']);p4=ex['positions4'];p8=positions8(ids,p4);cost['E_seconds']=time.perf_counter()-tick
 check();teacher=refer_native(frames[p8],row['input']['caption']);cost['ReferDINO']=teacher['cost'];cost['ReferDINO_load_seconds']=teacher['model_load_seconds'];cost['ReferDINO_process_wall_seconds']=teacher['process_wall_seconds'];cost['peak_GPU_bytes']=max(torch.cuda.max_memory_allocated(),teacher['peak_GPU_bytes']);cost['capture_seconds']=time.perf_counter()-t
 return dict(Source=src,E=E,frame_ids=ids,interval=native_iv,strong_interval=strong_iv,planned_positions=p8,original_positions4=p4,spatial_inputs=inputs,records=detached(recs,'cpu'),teacher_native=teacher,cost=cost,GT_read=False,posterior=dict(Native=q,Strong=strongq),logits=dict(native=logits,aux=ll,hires=hrl),expert=detached(ex,'cpu'))

def trim_fit(f):
 # Preserve full current-state -> samples -> feedback -> gradient chain, all40 steps.
 return f

def measure(ds,limit=None):
 import torch,numpy as np
 from methods.stvg_referdino_opd_v1.runtime import setup,lease
 from methods.stvg_referdino_opd_v1.predictor import ReferDINOOPDPredictor
 from methods.decota_final_simplified_v1.tensors import state_hash
 from scripts.run_decota_paper_main_v1 import frames_for
 from main_method.runtime import resource_guard
 setup();p=verify();lock=lease();beg=time.perf_counter();records=[r for r in p['records']if r['dataset']==ds];raw=read(R/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows'];times=[];m=None
 def check():
  resource_guard();assert time.perf_counter()-beg<p['new_inference_max_wall_seconds'];assert all(sha(R/f)==h for f,h in p['protected'].items())
 try:
  for rec in records[:limit]:
   check();a=rec['alias'];dest=D/'predictions'/ds/f'{a}.pt'
   if dest.exists():continue
   if ds=='hc2'and a!='RC001':
    assert (D/'HC_COMPATIBILITY.json').exists()
    for f,h in rec['reuse'].items():assert sha(R/f)==h
    inp=load(PRE/'inputs'/f'{a}.pt');sp=load(FIN/'Live/predictions'/f'{a}.pt');stage=load(OLD/'predictions'/f'{a}.pt');assert inp['frame_ids']==stage['frame_ids']==sp['frame_ids'];assert np.array_equal(inp['E'],sp['outputs']['E_MBR']);assert np.array_equal(inp['Source'].numpy(),sp['outputs']['Source_MBR'])
    cfg=p['configs'][ds]
    for s in p['seeds']:
     f=sp['fits'][f'T20_Live_{s}'];assert f['selected_step']==40 and f['actual_updates']==40
     for k in ['tau','lr','steps','samples','scales']:assert f['configuration'][k]==cfg[k]
     assert np.array_equal(stage['trajectories'][f'Live{s}'],sp['outputs'][f'T20_Live_{s}'])
    z=dict(alias=a,dataset=ds,query_ordinal=rec['query_ordinal'],frame_ids=inp['frame_ids'],planned_positions=inp['planned_positions'],Source=inp['Source'].double().numpy(),E=np.asarray(inp['E']),Live={s:np.asarray(sp['outputs'][f'T20_Live_{s}'])for s in p['seeds']},intervals={k:stage['outputs'][v]['interval']for k,v in [('Native','Native'),('Strong','HiResAuxMix')]},posterior={k:stage['outputs'][v]['q']for k,v in [('Native','Native'),('Strong','HiResAuxMix')]},reuse=rec['reuse'],actual_new_updates=0,reused_updates=120,cost={'mode':'verified_cache_reuse; not free end-to-end inference'},GT_read=False)
    save(dest,z);del inp,sp,stage,z;gc.collect();print('REUSED',ds,a,flush=True);continue
   row=dict(raw[rec['query_ordinal']]);sf=R/rec['subject_path'];assert sha(sf)==rec['subject_sha256'];sub=read(sf);assert not sub['GT_read'];row['parses']=sub['parses'];start=time.perf_counter();tick=time.perf_counter();frames,ids,_=frames_for(ds,row,collections.OrderedDict());decode_s=time.perf_counter()-tick
   loading=0.
   if m is None:
    tick=time.perf_counter();m=source_for(ds);loading=time.perf_counter()-tick;mh=state_hash(m.state_dict())
   cache=capture(m,frames,ids,row,p['configs'][ds],check);predictor=ReferDINOOPDPredictor(config=p['configs'][ds],source=m,check=check);tick=time.perf_counter();adapt=predictor.adapt(cache,key=a,arms=('Live',),seeds=p['seeds']);torch.cuda.synchronize();adapt_s=time.perf_counter()-tick
   assert all(not x.requires_grad and x.grad is None for x in m.parameters());assert state_hash(m.state_dict())==mh
   cost=dict(cache['cost'],decode_seconds=decode_s,Source_cold_load_seconds=loading,adapt_three_seed_wall_seconds=adapt_s,adapt=adapt['cost'],complete_three_seed_wall_seconds=time.perf_counter()-start,peak_GPU_bytes=max(cache['cost']['peak_GPU_bytes'],torch.cuda.max_memory_allocated()))
   # Each seed is a separate single-query run. Common stages must be counted once per seed.
   cost['single_seed_component_sum_seconds']={str(s):decode_s+cache['cost']['capture_seconds']+adapt['fits'][f'Live_{s}']['seconds']for s in p['seeds']}
   z=dict(alias=a,dataset=ds,query_ordinal=rec['query_ordinal'],frame_ids=ids,planned_positions=cache['planned_positions'],Source=cache['Source'].double().numpy(),E=np.asarray(cache['E']),Live={s:adapt['outputs'][f'Live_{s}']for s in p['seeds']},intervals={'Native':cache['interval'],'Strong':cache['strong_interval']},posterior=cache['posterior'],actual_new_updates=adapt['cost']['actual_updates'],reused_updates=0,cost=cost,GT_read=False)
   if ds=='hc2':
    inp=load(PRE/'inputs'/f'{a}.pt');oldsp=load(FIN/'Live/predictions'/f'{a}.pt');oldstage=load(OLD/'predictions'/f'{a}.pt');checks={}
    checks['Source_max_error']=float(np.abs(z['Source']-inp['Source'].numpy()).max());checks['E_max_error']=float(np.abs(z['E']-inp['E']).max());checks['Strong_posterior_error']=float(np.abs(z['posterior']['Strong']-oldstage['outputs']['HiResAuxMix']['q']).max());checks['Native_posterior_error']=float(np.abs(z['posterior']['Native']-oldstage['outputs']['Native']['q']).max());checks['Live_errors']={str(s):float(np.abs(z['Live'][s]-oldsp['outputs'][f'T20_Live_{s}']).max())for s in p['seeds']}
    put(D/'HC_COMPATIBILITY_CHECK_ACTUAL.json',checks)
    assert max(checks[k]for k in ['Source_max_error','E_max_error','Strong_posterior_error','Native_posterior_error'])<1e-10 and max(checks['Live_errors'].values())<1e-8,checks
    put(D/'HC_COMPATIBILITY.json',dict(status='pass',**checks,complete_inference_cost=cost))
   save(D/'traces'/ds/f'{a}.pt',dict(fits={k:trim_fit(v)for k,v in adapt['fits'].items()},logits=cache['logits'],records=cache['records'],teacher=cache['teacher_native'],expert=cache['expert'],GT_read=False));save(dest,z);times.append(cost);put(D/f'STATUS_{ds}.json',dict(done=len(list((D/'predictions'/ds).glob('*.pt'))),total=128,pid=os.getpid(),epoch=time.time(),last=a));print('PREDICTED',ds,a,round(time.perf_counter()-start,3),flush=True)
   del frames,cache,adapt,predictor,z;gc.collect();torch.cuda.empty_cache()
  if m is not None:assert state_hash(m.state_dict())==mh
  verify();files=sorted((D/'predictions'/ds).glob('*.pt'))
  if len(files)==128:put(D/f'BARRIER_{ds}.json',dict(status='sealed',count=128,files={str(f.relative_to(R)):sha(f)for f in files},plan_sha256=sha(D/'PLAN_LOCK.json'),GT_read=False,epoch=time.time()))
  if all((D/f'BARRIER_{d}.json').exists()for d in ['hc2','vidstg']):put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='sealed',parents=256,queries=256,barriers={d:sha(D/f'BARRIER_{d}.json')for d in ['hc2','vidstg']},all_before_GT=True,epoch=time.time()))
 finally:
  put(D/f'INVOCATION_COST_{ds}_{time.time_ns()}.json',dict(wall_seconds=time.perf_counter()-beg,new_cases=times,GT_read=False));lock.close()
if __name__=='__main__':
 try:
  if sys.argv[1]=='prepare':prepare()
  else:measure(sys.argv[2],int(sys.argv[3])if len(sys.argv)>3 else None)
 except BaseException as e:
  put(D/f'FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),GT_read=False));raise
