"""Frozen TA-STVG dual-domain paper package; one capture feeds every paper arm."""
import argparse,hashlib,json,os,sys,time,subprocess,traceback,gc,importlib.util,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PROVIDER=ROOT.parent/'stvg-opd'
RUN=ROOT/'outputs/tastvg_paper_v1';PUBLIC=ROOT/'results/tastvg_paper_v1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(q,p)
def identity(pid=None):
 pid=os.getpid()if pid is None else pid
 return dict(pid=pid,startticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]))
def alive(x):
 try:return identity(x['pid'])=={k:x[k]for k in ['pid','startticks']}
 except (FileNotFoundError,ProcessLookupError):return False

def load_panel():
 # Bind the maintained exact Global6 helper into the original immutable provider
 # namespace. Both helpers and all provider code are hashed by this new plan.
 for name,rel in [('stvg.global6','stvg/global6.py'),('paper_panel','stvg/paper_panel.py')]:
  spec=importlib.util.spec_from_file_location(name,ROOT/rel);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
 return sys.modules['paper_panel']

def check(assets=False):
 p=read(RUN/'plan.json');assert p['fingerprint']==hashlib.sha256(json.dumps({k:v for k,v in p.items()if k!='fingerprint'},sort_keys=True).encode()).hexdigest()
 assert all(sha(ROOT/f)==h for f,h in p['code'].items())
 assert all(sha(PROVIDER/f)==h for f,h in p['provider_sources'].items())
 for ds,j in p['jobs'].items():
  assert sha(PROVIDER/j['manifest'])==j['manifest_sha256']
  assert sha(PROVIDER/'outputs/boxfixed_full_v1'/ds/'sealed.json')==p['previous_seals'][ds]
 if assets:assert all(sha(PROVIDER/f)==h for f,h in p['assets'].items())
 assert sha(PROVIDER/'outputs/boxfixed_full_v1/capture_index.json')==p['capture_index_sha256']
 assert sha(RUN/'tta_audit.json')==p['tta_audit_sha256']
 return p

def prepare():
 if(RUN/'plan.json').exists():return check()
 old=read(PROVIDER/'outputs/boxfixed_full_v1/plan.json');assert read(PROVIDER/'outputs/boxfixed_full_v1/state.json')['status']=='complete_published'
 assert read(ROOT/'outputs/hc_temporal_full_v3_exploratory/state.json')['status']=='paused_by_user'
 assert not alive(dict(pid=2973918,startticks=222392809))and not alive(dict(pid=2974491,startticks=222393380))
 paths=[*sorted((ROOT/'stvg').glob('*.py')),*sorted((ROOT/'tools').glob('tastvg_paper_*.py')),ROOT/'configs/default.json']
 code={str(f.relative_to(ROOT)):sha(f)for f in paths}
 p=dict(schema='tastvg_paper_full_v1',authorization='Latest user: choose relatively best fixed parameters and run full; then TA-STVG parts of supplied paper task. This explicitly supersedes prior stop-until-both-Source-groups gate. Prior failures remain failures, not reclassified.',backbone='TA-STVG',directions=['vidstg_to_hc2','hc2_to_vidstg'],profile=dict(temperature=1.,observed_weight=.5,guard_weight=.1,link_weight=.25,temporal='original native224 twooffset6 posterior mean/original hull/physical expected-IoU MBR'),choice_reason='On same35 development recheck originalGlobal6 beats MBR3 and deep bracket in both fullv and totalt and has smaller long-event loss. Retain original stable configuration; no more temporal search. Complete previous Source/Full official outputs reused only after bitwise verification.',validation_passed=False,prior_Source_group_gate_failed_preserved=True,code=code,provider_sources=old['sources'],provider_runtime_fingerprint=old['runtime_fingerprint'],provider_plan_sha256=sha(PROVIDER/'outputs/boxfixed_full_v1/plan.json'),assets=old['assets'],jobs=old['jobs'],previous_seals={ds:sha(PROVIDER/'outputs/boxfixed_full_v1'/ds/'sealed.json')for ds in old['jobs']},capture_index_sha256=sha(PROVIDER/'outputs/boxfixed_full_v1/capture_index.json'),total_queries=12205,tta_audit_sha256=sha(RUN/'tta_audit.json'),expert='same ReferDINO,8 unchanged native positions and complete boxes/masks per query',GT_in_inference=False,main_arms=['Source','Interpolation','OursS','Full'],ablation_arms=['Full','NoLocal','Independent','OursS'],mechanism_arms=['Source','SharedInit','OursS'],NoLocal='ERef at unobserved nodes; direct native expert at observed nodes inside same Full event; unchanged ERef fallback outside event. No propagation support or links.',Independent='same Full interval/candidates/unary/legality/tie order, remove pair score and choose per-node argmax.',Interpolation='physical time center-linear/log-width-height interpolation between consecutive finite positive native boxes; beyond first/last valid anchor ERef; no GT.',extent='same maps/anchor/direction; archived strict>.5 own-peak 4-connected grid-edge component with largest map mass,row-major tie. Complete affine native extent versus mask-derived extent. Do not GT-select directions.',spatial_evaluation='actual sampled GT-supported nodes, common observed partition and native interval for Source/SharedInit/OursS; per-query mean then equal-query aggregate; empty groups excluded with counts',distance_bins_seconds=[0,.5,1,2,4,None],bootstrap='10000 parent-cluster resamples, equal-query estimand,seed20261022',cost='per direction16 SHA-selected input parents/1query each, fixed before labels; model weights/processes loaded once,warmup,3 repeated matched Source/Ours runs; no inter-query result/visual/RGB caches. Transfers and preprocessing retained; loads/startup separate; nested times not summed.',qualitative='private local continuous-frame sheets and plotting code; do not publish raw dataset media/query/GT coordinates/weights/features. Root reviews candidate categories after complete results.',new_training=False,new_OPD=False,new_expert_model=False,no_more_tuning=True,other_backbones_authorized=False,exposure='Existing full and development results already exposed; implementation collection and paper module analysis, not a new untouched-test claim.')
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();atomic(RUN/'plan.json',p)
 for rel in code:
  f=RUN/'source_snapshot'/rel;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes((ROOT/rel).read_bytes())
 cost={}
 for ds,j in old['jobs'].items():
  rows=[json.loads(s)for s in(PROVIDER/j['manifest']).read_text().splitlines()];groups={}
  for r in rows:groups.setdefault(str(r['metadata']['source']),[]).append(r)
  rank=lambda x:hashlib.sha256(('TA-paper-cost-20261022|'+str(x)).encode()).hexdigest()
  cost[ds]=[min(groups[k],key=lambda x:rank(x['id']))for k in sorted(groups,key=rank)[:16]]
 atomic(RUN/'cost_selection.json',cost);atomic(PUBLIC/'protocol.json',{k:v for k,v in p.items()if k not in ('assets','jobs')});atomic(PUBLIC/'cost_selection.json',dict(queries_per_domain=16,parents_per_domain=16,selection='SHA input-only fixed before new labels',private_manifest_sha256=sha(RUN/'cost_selection.json')))
 return check(assets=True)

def regression():
 sys.path.insert(0,str(PROVIDER));import numpy as np,torch
 panel=load_panel();p=prepare();ix=read(PROVIDER/'outputs/boxfixed_full_v1/capture_index.json')['files'];results={}
 from scripts.boxfixed_tuning.cache import binding
 from scripts.boxfixed_tuning.core import temporal,spatial,BASE
 from stvg.spatial import SpatialGeometry,evidence_dp
 for b,e in ix.items():
  f=PROVIDER/e['path'];assert sha(f)==e['sha256'];c=torch.load(f,map_location='cpu',weights_only=False);a,_=panel.derive(c);ref=temporal(c,1.);reference=spatial(c,ref['interval'],BASE)
  assert np.array_equal(a['Full_interval'],ref['interval'])and np.array_equal(a['Full'],reference['final'])
  g=SpatialGeometry.__new__(SpatialGeometry);g.__dict__.update(c['geometry']);bank=g.for_interval(a['Full_interval']);bank.score_pair=bank.score_pair*0;assert np.array_equal(evidence_dp(bank)['final'],a['Independent']);results[b]=a
 count=0
 for ds,j in p['jobs'].items():
  for r in [json.loads(s)for s in(PROVIDER/j['manifest']).read_text().splitlines()]:
   b=binding(r,p['provider_runtime_fingerprint'])
   if b not in results:continue
   a=results[b]
   with np.load(PROVIDER/'outputs/boxfixed_full_v1'/ds/'predictions'/(r['id']+'.npz'),allow_pickle=False)as z:
    assert np.array_equal(a['Source'],z['Source'])and np.array_equal(a['Full'],z['Final'])and np.array_equal(a['Full_interval'],z['Final_interval'])and np.array_equal(a['SharedInit'],z['E_Ref'])
   count+=1
 assert len(results)==len(ix)==192 and count==159
 atomic(RUN/'regression.json',dict(status='passed',indexed_math_captures=len(results),official_member_bitwise_aliases=count,nonmatching_index_entries=len(results)-count,Source_Full_ERef_interval_bitwise=True,Independent_zero_pair_DP_bitwise=True,new_GPU_calls=0,new_expert_calls=0,GT_read=False));print(json.dumps(read(RUN/'regression.json')))

def worker(ds):
 sys.path.insert(0,str(PROVIDER));import numpy as np,torch
 from stvg.runtime import setup,lease,load_source
 from stvg.predictor import Predictor
 from stvg.visual_cache import VisualCache
 from stvg.inputs import validate_input
 from stvg.worker import resource_guard
 from scripts.boxfixed_tuning.cache import RGBCache,ReusingTeacher,binding,from_native
 from scripts.tube_fusion.core import head_logits
 panel=load_panel();p=check();assert read(RUN/'regression.json')['status']=='passed';setup();gpu=lease();dest=RUN/ds
 rows=[json.loads(s)for s in(PROVIDER/p['jobs'][ds]['manifest']).read_text().splitlines()];ix=read(PROVIDER/'outputs/boxfixed_full_v1/capture_index.json')['files']
 oldseal=read(PROVIDER/'outputs/boxfixed_full_v1'/ds/'sealed.json');expert=ReusingTeacher(RUN/'teacher_cache',p['provider_runtime_fingerprint'],previous=[PROVIDER/'outputs/boxfixed_full_v1/teacher_cache']);model=Predictor(ds,check=resource_guard);model.teacher_service=expert;rgb=RGBCache();visual=VisualCache(4*2**30);files={};times=[];begin=time.perf_counter();atomic(dest/'worker.json',dict(**identity(),fingerprint=p['fingerprint'],dataset=ds))
 try:
  for i,r in enumerate(rows):
   resource_guard();key=r['id'];rp=dest/'receipts'/(key+'.json');npz=dest/'predictions'/(key+'.npz');cp=dest/'captures'/(key+'.pt');b=binding(r,p['provider_runtime_fingerprint'])
   if rp.exists():
    rec=read(rp);assert rec['fingerprint']==p['fingerprint']and rec['input_binding']==b and sha(npz)==rec['prediction_sha256']and sha(cp)==rec['capture_sha256'];files[key]=dict(path=str(npz.relative_to(dest)),sha256=sha(npz),receipt_path=str(rp.relative_to(dest)),receipt_sha256=sha(rp),capture_path=str(cp.relative_to(dest)),capture_sha256=sha(cp));continue
   assert not npz.exists()and not cp.exists(),'Preserve unbound output for recovery; do not overwrite'
   tick=time.perf_counter();cost={};reuse=False;before=visual.stats();rgb_before=(rgb.hits,rgb.misses)
   if b in ix:
    e=ix[b];f=PROVIDER/e['path'];assert sha(f)==e['sha256'];cap=torch.load(f,map_location='cpu',weights_only=False);reuse=True
   else:
    frames,ids,rc=rgb.get(r['metadata'],ds);validate_input(frames,ids,r['metadata'])
    if model.source is None:model.source=load_source(model.config)
    torch.manual_seed(20260920);np.random.seed(20260920)
    with visual.context(model.source,frames,ids):native=model.capture(frames,ids,r['metadata'],temporal_mode='native224',export_replay=True)
    tc=native['temporal_cache'];head=model.source.temp_embed.cuda();logits,hcost=head_logits(head,[v['hidden']for v in tc['offsets']],[v['native_logits']for v in tc['offsets']]);head.cpu();cap=from_native(native,logits);cost=dict(native['cost']);cost.update(head=hcost,RGB=rc);del native,tc,logits,frames
   arr,trace=panel.derive(cap)
   oldentry=oldseal['files'][key];oldpath=PROVIDER/'outputs/boxfixed_full_v1'/ds/oldentry['path'];assert sha(oldpath)==oldentry['sha256']
   with np.load(oldpath,allow_pickle=False)as z:
    assert np.array_equal(z['Source'],arr['Source'])and np.array_equal(z['E_Ref'],arr['SharedInit'])and np.array_equal(z['Source_interval'],arr['Source_interval'])and np.array_equal(z['Final'],arr['Full'])and np.array_equal(z['Final_interval'],arr['Full_interval'])and np.array_equal(z['positions8'],arr['positions8']),'Original Source/Full/evidence compatibility failed'
   for f in (npz,cp):f.parent.mkdir(parents=True,exist_ok=True)
   torch.save(cap,cp);np.savez_compressed(npz,**arr);after=visual.stats()
   receipt=dict(fingerprint=p['fingerprint'],input_binding=b,prediction_sha256=sha(npz),capture_sha256=sha(cp),complete_capture_reused=reuse,original_Source_and_Full_bitwise_alias=True,original_prediction_sha256=oldentry['sha256'],expert_forwards=cost.get('ReferDINO_query_forwards',0),capture_cost=cost,query_wall_seconds=time.perf_counter()-tick,visual_cumulative=after,visual_delta={k:after[k]-before[k]for k in ('hits','misses')},RGB_delta=dict(hits=rgb.hits-rgb_before[0],misses=rgb.misses-rgb_before[1]),worker=identity(),trace=trace,GT_read=False)
   atomic(rp,receipt);files[key]=dict(path=str(npz.relative_to(dest)),sha256=sha(npz),receipt_path=str(rp.relative_to(dest)),receipt_sha256=sha(rp),capture_path=str(cp.relative_to(dest)),capture_sha256=sha(cp));times.append(receipt['query_wall_seconds']);times=times[-100:]
   atomic(dest/'progress.json',dict(completed=i+1,total=len(rows),recent_seconds=float(np.mean(times)),ETA_seconds=float(np.mean(times))*(len(rows)-i-1),**identity(),visual=after,GT_read=False));print(json.dumps(dict(dataset=ds,completed=i+1,total=len(rows),seconds=receipt['query_wall_seconds'],capture_hit=reuse)),flush=True);del cap,arr;gc.collect();torch.cuda.empty_cache()
  check();atomic(dest/'sealed.json',dict(status='all_predictions_sealed',fingerprint=p['fingerprint'],queries=len(rows),files=files,GT_read=False));atomic(dest/'run.json',dict(status='completed_sealed',seconds=time.perf_counter()-begin,**identity()))
 finally:expert.close();gpu.close()

def controller(resume=False):
 p=prepare();lock=(RUN/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if(RUN/'state.json').exists():
  old=read(RUN/'state.json');assert resume and old['fingerprint']==p['fingerprint']and old['status']!='complete_published'and not any(alive(x)for x in old.get('children',[]))
 state=dict(status='running',fingerprint=p['fingerprint'],controller=identity(),children=[],jobs={});atomic(RUN/'state.json',state);pending=[]
 def launch(script,args,role,env=None):
  log=(RUN/(role+'.log')).open('ab');child=subprocess.Popen([sys.executable,'-B',str(ROOT/'tools'/script),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,**(env or {})));log.close();who={**identity(child.pid),'role':role};state['children'].append(who);atomic(RUN/'state.json',state);return child,who
 def finish(c,w):
  code=c.wait();state['children']=[x for x in state['children']if x!=w];atomic(RUN/'state.json',state);assert code==0,f'{w["role"]} exit{code}; preserve outputs and logs'
 try:
  assert read(RUN/'regression.json')['status']=='passed'
  for ds in ('hc2','vidstg'):
   if not(RUN/ds/'sealed.json').exists():
    state['jobs'][ds]='running';c,w=launch('tastvg_paper_full.py',['worker','--dataset',ds],ds+'-worker');finish(c,w)
   state['jobs'][ds]='sealed'
   if not(RUN/ds/'score_receipt.json').exists():pending.append(launch('tastvg_paper_score.py',['--dataset',ds],ds+'-scorer',{'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}))
  for c,w in pending:finish(c,w)
  state.update(status='cost_benchmark',jobs={'hc2':'scored','vidstg':'scored'});atomic(RUN/'state.json',state)
  if not(RUN/'cost/sealed.json').exists():c,w=launch('tastvg_paper_cost.py',[],'fair-cost');finish(c,w)
  if not(RUN/'qualitative/manifest.json').exists():c,w=launch('tastvg_paper_figures.py',[],'qualitative',{'CUDA_VISIBLE_DEVICES':''});finish(c,w)
  c,w=launch('tastvg_paper_report.py',[],'paper-report',{'CUDA_VISIBLE_DEVICES':''});finish(c,w)
  state['status']='awaiting_publication_and_qualitative_review';atomic(RUN/'state.json',state)
 except BaseException:
  state.update(status='failed',error=traceback.format_exc());atomic(RUN/'state.json',state);raise
 finally:lock.close()

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','regression','worker','run']);ap.add_argument('--dataset',choices=['hc2','vidstg']);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
 controller(a.resume)if a.command=='run'else worker(a.dataset)if a.command=='worker'else globals()[a.command]()
