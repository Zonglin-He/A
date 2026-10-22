"""Cached positive clauses: jointly select boundaries with narrative event order."""
import os,sys,time,hashlib,json,traceback
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.joint_event_v1.run import sha,read,put,records,B
D=R/'outputs/native_event_order_hc64_v1';P=R/'results/native_event_order_hc64_v1';PARENT=R/'outputs/native_event_composition_hc64_v1'
ARMS=['NativeG6','OrderedClauseFusion','DirectOrdered','RoleOnlyFusion','SwappedOrderFusion','SelfOrderFusion','ClauseFusion','SelfHullFusion','NoComposition']
def who():return dict(pid=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
def verify():
 p=read(D/'plan.json');assert read(R/'outputs/tastvg_paper_v1/state.json')['status']=='paused_by_user';assert sha(PARENT/'sealed.json')==p['parent_seal_sha256']
 for f,h in p['code'].items():assert sha(R/f)==h,f
 for f,h in p['paper_code'].items():assert sha(R/f)==h,f
 for f,h in p['provider_sources'].items():assert sha(R.parent/'stvg-opd'/f)==h,f
 assert time.time()<p['stop_epoch'];return p
def prepare():
 assert not(D/'plan.json').exists();parent=read(PARENT/'plan.json');code={str(f.relative_to(R)):sha(f)for f in Path(__file__).parent.glob('*.py')}
 for f in ['research/joint_event_v1/run.py','research/boundary_feedback_v1/score.py']:code[f]=sha(R/f)
 p=dict(schema='native_event_order_hc64_v1',primary='OrderedClauseFusion',arms=ARMS,count=64,code=code,paper_code=parent['paper_code'],provider_sources=parent['provider_sources'],captures=parent['captures'],parent_files=read(PARENT/'sealed.json')['files'],parent_seal_sha256=sha(PARENT/'sealed.json'),labels_sha256=parent['labels_sha256'],supported=parent['supported'],
 authorization='Same overnight user research loop; unchanged positive clause evidence, no expert, no model or spatial changes',rationale='Unordered clause hull gains2.4693pp time but short events lose5.7702pp. CasesRC014/RC054/RC026 show the final-action standalone posterior selecting an earlier occurrence; unordered hull mixes incompatible phases. Test fixed textual first/last narrative order to disambiguate occurrences jointly, not another fusion weight. Offline boundary audit does NOT show first-start intrinsically better; success must come from joint order constraints, not an assumed marginal advantage.',
 fixed=dict(primary='Using exact same first/last positive native posteriors, condition pairs on startFirst<=startLast AND endFirst<=endLast (overlap permitted), then output[startFirst,endLast]. Mix half with originalq0. Original MBR and complete-box DP; no thresholds/weight scan.',controls='DirectOrdered same information; RoleOnlyFusion first-start/last-end without inner order; SwappedOrderFusion exchanges semantic first/last roles; SelfOrderFusion uses two copies of unchanged original posterior; retain parent unordered ClauseFusion/SelfHullFusion; no-composition exact',scope='Same52/64 query-only inherited-subject main AND cases; all unsupported unchanged. Narrative AND need not always be chronological, a stated limitation; do not claim a guaranteed temporal relation. Valid-order probability logged for all, never an acceptance gate.',GT='All new predictions sealed before independent new scoring; old cases used only offline hypothesis development, not query-level online choice',space='Original complete-box geometry/expert8/physicalgrid and all preserved Source/Native outputs'),
 new_RGB=0,new_visual_encoder=0,new_text_encoder=0,new_decoder=0,new_expert=0,new_optimizer=0,GT_in_inference=False,all64_seal_before_new_GT=True,maximum_CPU_seconds=600,stop_epoch=1792708800,exposure=parent['exposure'],qualification=parent['qualification'],references=parent['references'])
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();put(D/'plan.json',p);put(P/'protocol.json',{k:v for k,v in p.items()if k not in ['paper_code','provider_sources','captures','parent_files']})
 for f in code:
  q=D/'source_snapshot'/f;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((R/f).read_bytes())
 put(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']));print(p['fingerprint'])
def predict():
 import torch,numpy as np
 from stvg.temporal import final_mbr
 from stvg.spatial import SpatialGeometry,evidence_dp
 from research.native_event_order_v1.core import ordered,roles
 p=verify();torch.set_num_threads(1);start=time.perf_counter();files={};put(D/'state.json',dict(status='predicting_CPU_cache',fingerprint=p['fingerprint'],**who()))
 for row in records():
  verify();assert time.perf_counter()-start<p['maximum_CPU_seconds'];k=row['id'];e=p['parent_files'][k];assert sha(PARENT/e['path'])==e['sha256']and sha(PARENT/e['trace_path'])==e['trace_sha256'];old=torch.load(PARENT/e['trace_path'],weights_only=False,map_location='cpu');cf=B/'captures'/f'{k}.pt';assert sha(cf)==p['captures'][k]['capture_sha256'];cap=torch.load(cf,weights_only=False,map_location='cpu');ids=cap['frame_ids'];q0=old['q']['NativeG6'];tick=time.perf_counter();qs={a:q0.copy()for a in ARMS};masses={}
  if e['supported']:
   first=old['intervened_q']['First'];last=old['intervened_q']['Last'];q,masses['real_order']=ordered(first,last);swap,masses['swapped_order']=ordered(last,first);selfq,masses['self_order']=ordered(q0,q0);qs.update(OrderedClauseFusion=.5*q0+.5*q,DirectOrdered=q,RoleOnlyFusion=.5*q0+.5*roles(first,last),SwappedOrderFusion=.5*q0+.5*swap,SelfOrderFusion=.5*q0+.5*selfq,ClauseFusion=old['q']['ClauseFusion'],SelfHullFusion=old['q']['SelfHullFusion'])
  iv={a:final_mbr(ids,q)['interval']for a,q in qs.items()};geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);arrays={a:evidence_dp(geom.for_interval(iv[a]))['final'].numpy()for a in ARMS}
  with np.load(PARENT/e['path'],allow_pickle=False)as z:
   for a in ['NativeG6','ClauseFusion','SelfHullFusion','NoComposition']:assert np.array_equal(arrays[a],z[a])and iv[a]==z[a+'_interval'].tolist()
   arrays['Source']=z['Source'].copy();iv['Source']=z['Source_interval'].tolist()
  score=np.zeros_like(q0);take=q0>0;score[take]=np.log(qs['OrderedClauseFusion'][take])-np.log(q0[take]);f=D/'predictions'/f'{k}.npz';assert not f.exists();f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,**arrays,**{a+'_interval':span for a,span in iv.items()},frame_ids=ids,positions8=cap['planned_positions']);tr=D/'traces'/f'{k}.pt';tr.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(q=qs,score=score,clauses=old['clauses'],order_mass=masses,GT_read=False,no_composition_exact=True),tr);files[k]=dict(path=str(f.relative_to(D)),sha256=sha(f),trace_path=str(tr.relative_to(D)),trace_sha256=sha(tr),seconds=time.perf_counter()-tick,supported=e['supported']);put(D/'progress.json',dict(completed=len(files),total=64,seconds=time.perf_counter()-start,**who()));print(k,e['supported'],iv['NativeG6'],iv['OrderedClauseFusion'],masses,flush=True)
 verify();assert not torch.cuda.is_initialized();put(D/'sealed.json',dict(count=64,files=files,plan_sha256=sha(D/'plan.json'),GT_read=False,counts=dict(new_RGB=0,new_visual_encoder=0,new_text_encoder=0,new_decoder=0,new_expert=0,new_optimizer=0),seconds=time.perf_counter()-start));put(D/'state.json',dict(status='sealed',fingerprint=p['fingerprint']))
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:put(D/f'failure-{time.time_ns()}.json',dict(error=traceback.format_exc(),partial_preserved=True,**who()));put(D/'state.json',dict(status='engineering_failed',partial_preserved=True));raise
