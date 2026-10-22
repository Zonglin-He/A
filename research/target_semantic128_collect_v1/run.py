"""256 parent-isolated source-only semantic acquisitions; strict cached expert reads."""
import sys,os,time,json,hashlib,traceback,gc
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));PROVIDER=R.parent/'stvg-opd'
from research.joint_event_v1.run import sha,read,put
D=R/'outputs/target_semantic128_collect_v1';P=R/'results/target_semantic128_collect_v1';L=R/'outputs/overnight_event_loop_20261023'
def who():return dict(pid=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
def verify(assets=False):
 p=read(D/'plan.json');assert read(R/'outputs/tastvg_paper_v1/state.json')['status']=='paused_by_user';assert read(L/'state.json').get('deadline_cancelled_by_user')
 for root,key in [(R,'code'),(R,'paper_code'),(PROVIDER,'provider_sources')]:
  for f,h in p[key].items():assert sha(root/f)==h,f
 if assets:assert all(sha(PROVIDER/f)==h for f,h in p['assets'].items())
 assert sha(L/'target_semantic128_input_selection.json')==p['selection_sha256'];return p

def prepare():
 assert not(D/'plan.json').exists();paper=read(R/'outputs/tastvg_paper_v1/plan.json');selection=read(L/'target_semantic128_input_selection.json');items=[]
 for ds,rows in selection['items'].items():
  assert len(rows)==128 and len({x['parent']for x in rows})==128
  assert sum(x['split']=='develop'for x in rows)==64 and sum(x['split']=='confirm'for x in rows)==64
  old=read(PROVIDER/'outputs/boxfixed_full_v1'/ds/'sealed.json')
  for x in rows:
   y=dict(x);y['reference']=old['files'][x['row']['id']];items.append(y)
 code={str(f.relative_to(R)):sha(f)for f in Path(__file__).parent.glob('*.py')}
 for f in ['research/native_event_target_semantic_v1/run.py','research/native_event_target_semantic_v1/core.py','research/native_event_semantic_prototype_v1/core.py','research/native_event_support_view_v1/core.py','research/native_event_spatial_text_v1/core.py','research/joint_event_v1/run.py']:code[f]=sha(R/f)
 p=dict(schema='target_semantic128_collect_v1',count=256,parents_per_direction=128,selection_sha256=sha(L/'target_semantic128_input_selection.json'),items=items,code=code,paper_code=paper['code'],provider_sources=paper['provider_sources'],assets=paper['assets'],provider_runtime_fingerprint=paper['provider_runtime_fingerprint'],seed=20260920,selection_seed=20261023,
 authorization='Latest user cancels07limit, accepts target semantic prototype direction, explicitly authorizes larger128data and finite tuning; one-round/shrinkage/weak-evidence comparison, space/expert unchanged.',
 exposure='Allofficialfull results historicallyexposed;64+64 per-direction parent split is prospective for NEWcandidate selection only, not pristine unexposed test. Selectwholeofficialparentpool227/732, inputSHAonly, no candidate scores/GT used.',
 acquisition='Recompute exact originalnative224 Source only where compact paper caches lack actualtime features. Same2offset/inputgrid/precision/seed/Fullcaption. Original cached expert evidence only, no _call allowed; exactRGB/context/hash verifies. Rebuild sameERef/geometry, compare Source/E/Full/positions/intervals to original sealed full bitwise before semantic reads. No writes to oldpaper/archive receipts.',
 semantic='Same learned wordhead on actual original temporal memory:Global/allkeysNoOp/Target1.5/reflected1.5/wrongtimecontext. Same8 expertpositions, fulltimecompleteBoxFixed auxiliary. Original global and noOphead bitwise; store p/u andrankmatchedrandom for CPU-only finite tuning. All source model weights frozen.',
 families=['OriginalLoop2','OriginalLoop1','OneRoundPrototypeShrink','OneRoundWeakEvidence'],kappa_grid=[1,4,16],beta_grid=[.25,.5,.75],no_stacked_shrink_beta=True,split='64develop+64confirm per direction, parent disjoint; choose on development only then freeze before confirmation scoring.',GT_in_inference=False,new_expert=0,new_observations=0,new_training=0,new_other_backbone=0,new_highres=0,maximum_GPU_seconds=7200,no_deadline_fallback=True)
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();put(D/'plan.json',p);put(P/'protocol.json',{k:v for k,v in p.items()if k not in ['items','code','paper_code','provider_sources','assets']})
 for f in code:
  dest=D/'source_snapshot'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((R/f).read_bytes())
 put(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']));print(p['fingerprint'])

def collect():
 sys.path.insert(0,str(PROVIDER));import torch,numpy as np
 from stvg.runtime import setup,lease
 from stvg.predictor import Predictor
 from stvg.visual_cache import VisualCache
 from stvg.teacher_bridge import ResidentTeacher,evidence_key
 from scripts.boxfixed_tuning.cache import RGBCache,from_native
 from scripts.tube_fusion.core import head_logits
 from tools.tastvg_paper_full import load_panel
 from stvg.spatial import evidence_dp
 from research.native_event_target_semantic_v1.run import head_read
 from research.native_event_semantic_prototype_v1.core import matched_random,loop
 p=verify(assets=True);setup();torch.set_num_threads(4);gpu=lease();start=time.perf_counter();put(D/'state.json',dict(status='collecting_Source_GPU',fingerprint=p['fingerprint'],**who()));panel=load_panel();files={};counts=dict(Source_queries=0,RGB_decode_queries=0,ASA_head_forwards=0,new_expert=0,new_observations=0,new_training=0,all_old_outputs_bitwise=0);rgb=RGBCache();visual=VisualCache(4*2**30)
 class CacheOnlyTeacher(ResidentTeacher):
  def _call(self,*args,**kwargs):raise RuntimeError('No new ReferDINO forward authorized; preserve failure/cache miss')
  def __call__(self,frames,caption,*,positions,frame_ids,dataset):
   key,_=evidence_key(frames,caption,positions=positions,frame_ids=frame_ids,fingerprint=self.fingerprint,dataset=dataset)
   roots=[PROVIDER/'outputs/boxfixed_full_v1/teacher_cache',R/'outputs/tastvg_paper_v1/teacher_cache'];folder=next((f for f in roots if(f/(key+'.json')).exists()),None)
   if folder is None:raise RuntimeError('Bound previous expert evidence missing; no replacement call allowed: '+key)
   self.cache_dir=folder;out=super().__call__(frames,caption,positions=positions,frame_ids=frame_ids,dataset=dataset);assert out['cache_hit']and out['query_forwards']==0 and self.process is None;return out
 from stvg.global6 import posterior
 teacher=CacheOnlyTeacher(fingerprint=p['provider_runtime_fingerprint'])
 try:
  for ds in ['hc2','vidstg']:
   model=Predictor(ds);model.teacher_service=teacher
   for item in [x for x in p['items']if x['dataset']==ds]:
    assert time.perf_counter()-start<p['maximum_GPU_seconds'];row=item['row'];key=item['alias'];tick=time.perf_counter();ref=item['reference'];original=PROVIDER/'outputs/boxfixed_full_v1'/ds/ref['path'];assert sha(original)==ref['sha256'];frames,ids,rc=rgb.get(row['metadata'],ds);counts['RGB_decode_queries']+=int(not rc['hit']);torch.manual_seed(p['seed']);np.random.seed(p['seed'])
    # Original provider explicitly native224; never invoke its historical Strong default.
    with visual.context(model.source,frames,ids)if model.source is not None else __import__('contextlib').nullcontext():native=model.capture(frames,ids,row['metadata'],temporal_mode='native224',export_replay=True)
    counts['Source_queries']+=1;assert native['cost']['ReferDINO_query_forwards']==0 and native['cost']['ReferDINO_cache_hit'];head=model.source.temp_embed.cuda();logs,hcost=head_logits(head,[o['hidden']for o in native['temporal_cache']['offsets']],[o['native_logits']for o in native['temporal_cache']['offsets']]);head.cpu();cap=from_native(native,logs);arr,_=panel.derive(cap);q0=posterior(cap['records'],logs,ids)
    with np.load(original,allow_pickle=False)as z:
     assert np.array_equal(arr['Source'],z['Source'])and np.array_equal(arr['Full'],z['Final'])and np.array_equal(arr['SharedInit'],z['E_Ref'])and np.array_equal(arr['Full_interval'],z['Final_interval'])and np.array_equal(arr['Source_interval'],z['Source_interval'])and np.array_equal(arr['positions8'],z['positions8']),'Original Source/Full/E/interval/8evidence bitwise failed'
     source_iv=z['Source_interval'].tolist()
    counts['all_old_outputs_bitwise']+=1;aux=evidence_dp(native['geometry'].for_interval([ids[0],ids[-1]+1]))['final'];probs={};us={};stats={};mass={};model.source.t_spatial_clas.cuda()
    for kind in ['Global','AllKeysNoOp','TargetContext','ReflectedContext','WrongTimeContext']:
     probs[kind],us[kind],stats[kind],mass[kind]=head_read(model.source,native,aux,kind);counts['ASA_head_forwards']+=2
    model.source.t_spatial_clas.cpu();assert np.array_equal(probs['Global'],probs['AllKeysNoOp'])and np.array_equal(us['Global'],us['AllKeysNoOp']);randomp=matched_random(probs['TargetContext'],us['TargetContext']);parentq,parentsteps=loop(probs['TargetContext'],q0)
    cap.update(temporal_cache=native['temporal_cache'],planned_positions=native['planned_positions']);cf=D/'captures'/f'{key}.pt';cf.parent.mkdir(parents=True,exist_ok=True);torch.save(cap,cf);f=D/'semantic'/f'{key}.pt';f.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(q0=q0,semantic_p=probs,target_u=us['TargetContext'],random_p=randomp,original_loop2=parentq,original_loop1=parentsteps[0]['q'],Source=arr['Source'],Source_interval=source_iv,NativeG6=arr['Full'],NativeG6_interval=arr['Full_interval'].tolist(),positions8=arr['positions8'],statistics=stats,keep_mass=mass,GT_read=False),f);e=dict(path=str(f.relative_to(D)),sha256=sha(f),capture_path=str(cf.relative_to(D)),capture_sha256=sha(cf),old_prediction_sha256=ref['sha256'],original_Source_Full_E_interval_positions_bitwise=True,global_head_noop_bitwise=True,ReferDINO_query_forwards=0,expert_cache_key=native['teacher_native']['cache_key'],seconds=time.perf_counter()-tick,source_cost=native['cost'],semantic_ASA_head_forwards=10);files[key]=e;put(D/'receipts'/f'{key}.json',e);put(D/'progress.json',dict(completed=len(files),total=256,dataset=ds,alias=key,seconds=time.perf_counter()-start,counts=counts,recent_query_seconds=e['seconds'],visual=visual.stats(),**who()));print(json.dumps(dict(completed=len(files),alias=key,seconds=e['seconds'],expert_forwards=0)),flush=True);del native,cap,arr,frames,logs,aux,probs,us;gc.collect();torch.cuda.empty_cache()
   assert all(not x.requires_grad and x.grad is None for x in model.source.parameters());del model;gc.collect();torch.cuda.empty_cache();visual.clear()if hasattr(visual,'clear')else None
  verify();assert counts['Source_queries']==counts['all_old_outputs_bitwise']==256 and counts['ASA_head_forwards']==2560;put(D/'sealed.json',dict(count=256,files=files,plan_sha256=sha(D/'plan.json'),GT_read=False,counts=counts,seconds=time.perf_counter()-start));put(D/'state.json',dict(status='sealed_semantic256_no_new_GT',fingerprint=p['fingerprint']))
 finally:teacher.close();gpu.close()
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:put(D/f'failure-{time.time_ns()}.json',dict(error=traceback.format_exc(),partial_preserved=True,**who()));put(D/'state.json',dict(status='engineering_failed',partial_preserved=True));raise
