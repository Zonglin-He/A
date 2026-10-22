"""Positive clause location feedback with original grid-span duration prior."""
import os,sys,time,hashlib,json,traceback
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.joint_event_v1.run import sha,read,put,records,B
D=R/'outputs/native_event_conditional_hc64_v1';P=R/'results/native_event_conditional_hc64_v1';PARENT=R/'outputs/native_event_shared_query_hc64_v1'
ARMS=['NativeG6','ConditionalClause','SharedQueryClauseFusion','ConditionalSelfHull','SelfHullFusion','ConditionalReverse','ConditionalSharpen','NoCondition']
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
 p=dict(schema='native_event_conditional_hc64_v1',primary='ConditionalClause',arms=ARMS,count=64,code=code,paper_code=parent['paper_code'],provider_sources=parent['provider_sources'],captures=parent['captures'],parent_files=read(PARENT/'sealed.json')['files'],parent_seal_sha256=sha(PARENT/'sealed.json'),labels_sha256=parent['labels_sha256'],supported=parent['supported'],
 authorization='User overnight native event-information research, same positive clause evidence and frozen space; no additional expert/model forward',rationale='Actual native positive clause information has matched-duration near-native-prior ranking57.33%CI[53.41,61.42], vs native52.14%, but envelope shifts duration and harms short events. Old ReferDINO excess normalization lacked validated event ranking. Use measured native semantic evidence only to redistribute locations conditional on the SAME grid-span length; preserve native posterior span-length masses. Not a new information source, not simply repeating ReferDINO normalization.',
 fixed=dict(primary='For each legal node-span length l=j-i, preserve m0(l)=sum nativeq[i,i+l]. Redistribute this mass proportional to sealed parent SharedQueryClauseFusion restricted to original native-positive support at that l. No new half blend or weight; parent fixed half native/half clause already retained. Same physical MBR and BoxFixed.',length='Discrete sampled node-span length, NOT exact physical seconds. Report physical duration spread in each span group and actual final interval lengths; MBR may still change duration. GT event lengths never available online.',controls='Parent direct SharedQueryClauseFusion same information; ConditionalSelfHull and parentSelfHull; ConditionalReverse from same reversed-clause teacher; ConditionalSharpen from nativeq squared; exact native target no-op. Existing same52 syntactic support, twelve unsupported exactnative, no quality acceptance.',scope='No fitting/query update, thresholds, length bins or bandwidth tuning. Primary preserves all native feasible interval support and all native duration masses. Parent contains half native, so impossible teacher-support fallback cannot trigger in primary. No GT duration-specific branch.',space='Same original expert8/geometry/ERef/complete-box DP; all64 new predictions seal before independent official CPU scoring'),
 new_RGB=0,new_visual_encoder=0,new_text_encoder=0,new_decoder=0,new_expert=0,new_observations=0,new_optimizer=0,GT_in_inference=False,all64_seal_before_new_GT=True,maximum_CPU_seconds=600,stop_epoch=1792708800,exposure=parent['exposure'],qualification='Positive Source and NativeG6 time/fullv, improvements>harms, both Source short/long temporal groups positive; beat same-information nonsemantic controls and frozen dual-domain recheck before adoption. No native-duration preservation-only claim of correctness.',references=['https://arxiv.org/abs/2111.04321','https://arxiv.org/abs/2203.05243','https://github.com/HengLan/TA-STVG'],scope='Own frozen native-feedback conditional readout feasibility test. Debiasing papers use trained representations/interventions; this does not reproduce them or prove causal debiasing.')
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();put(D/'plan.json',p);put(P/'protocol.json',{k:v for k,v in p.items()if k not in ['paper_code','provider_sources','captures','parent_files']})
 for f in code:
  q=D/'source_snapshot'/f;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((R/f).read_bytes())
 put(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']));print(p['fingerprint'])
def predict():
 import torch,numpy as np
 from stvg.temporal import final_mbr
 from stvg.spatial import SpatialGeometry,evidence_dp
 from research.native_event_conditional_v1.core import duration_conditional
 from stvg.runtime import setup
 p=verify();setup();torch.set_num_threads(1);start=time.perf_counter();files={};put(D/'state.json',dict(status='predicting_CPU_cache',fingerprint=p['fingerprint'],**who()))
 for row in records():
  verify();assert time.perf_counter()-start<p['maximum_CPU_seconds'];k=row['id'];e=p['parent_files'][k];assert sha(PARENT/e['path'])==e['sha256']and sha(PARENT/e['trace_path'])==e['trace_sha256'];old=torch.load(PARENT/e['trace_path'],weights_only=False,map_location='cpu');cf=B/'captures'/f'{k}.pt';assert sha(cf)==p['captures'][k]['capture_sha256'];cap=torch.load(cf,weights_only=False,map_location='cpu');ids=cap['frame_ids'];q0=old['q']['NativeG6'];tick=time.perf_counter();qs={a:q0.copy()for a in ARMS};masses={}
  if e['supported']:
   real=old['q']['SharedQueryClauseFusion'];selfq=old['q']['SelfHullFusion'];reverse=old['q']['ReverseSharedClauses'];sharp=q0*q0;sharp/=sharp.sum();changed={}
   for arm,target in [('ConditionalClause',real),('ConditionalSelfHull',selfq),('ConditionalReverse',reverse),('ConditionalSharpen',sharp)]:changed[arm],masses[arm]=duration_conditional(q0,target)
   qs.update(**changed,SharedQueryClauseFusion=real,SelfHullFusion=selfq)
  noop,nooperr=duration_conditional(q0,q0.copy());assert np.array_equal(noop,q0)and nooperr==0
  iv={a:final_mbr(ids,q)['interval']for a,q in qs.items()};geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);arrays={a:evidence_dp(geom.for_interval(iv[a]))['final'].numpy()for a in ARMS}
  with np.load(PARENT/e['path'],allow_pickle=False)as z:
   for a in ['NativeG6','SharedQueryClauseFusion','SelfHullFusion']:assert np.array_equal(arrays[a],z[a])and iv[a]==z[a+'_interval'].tolist()
   arrays['Source']=z['Source'].copy();iv['Source']=z['Source_interval'].tolist()
  score=np.zeros_like(q0);take=q0>0;score[take]=np.log(qs['ConditionalClause'][take])-np.log(q0[take]);f=D/'predictions'/f'{k}.npz';assert not f.exists();f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,**arrays,**{a+'_interval':span for a,span in iv.items()},frame_ids=ids,positions8=cap['planned_positions']);tr=D/'traces'/f'{k}.pt';tr.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(q=qs,score=score,clauses=old['clauses'],span_mass_error=masses,physical_span_duration_spread_frames=max(max(ids[i+l]+1-ids[i]for i in range(len(ids)-l))-min(ids[i+l]+1-ids[i]for i in range(len(ids)-l))for l in range(1,len(ids))),GT_read=False,no_composition_exact=True),tr);files[k]=dict(path=str(f.relative_to(D)),sha256=sha(f),trace_path=str(tr.relative_to(D)),trace_sha256=sha(tr),seconds=time.perf_counter()-tick,supported=e['supported']);put(D/'progress.json',dict(completed=len(files),total=64,seconds=time.perf_counter()-start,**who()));print(k,e['supported'],iv['NativeG6'],iv['ConditionalClause'],masses,flush=True)
 verify();assert not torch.cuda.is_initialized();put(D/'sealed.json',dict(count=64,files=files,plan_sha256=sha(D/'plan.json'),GT_read=False,counts=dict(new_RGB=0,new_visual_encoder=0,new_text_encoder=0,new_decoder=0,new_expert=0,new_optimizer=0),seconds=time.perf_counter()-start));put(D/'state.json',dict(status='sealed',fingerprint=p['fingerprint']))
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:put(D/f'failure-{time.time_ns()}.json',dict(error=traceback.format_exc(),partial_preserved=True,**who()));put(D/'state.json',dict(status='engineering_failed',partial_preserved=True));raise
