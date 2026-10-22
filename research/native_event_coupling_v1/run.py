"""No new model: preserve true native event marginals while changing their coupling."""
import os,sys,time,hashlib,json,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.joint_event_v1.run import sha,read,put,records,B
D=R/'outputs/native_event_coupling_hc64_v1';P=R/'results/native_event_coupling_hc64_v1';PHASE=R/'outputs/native_event_composition_hc64_v1'
ARMS=['NativeG6','MonotoneClauseFusion','IndependentClauseFusion','AntimonotoneFusion','ReflectedLastFusion','SelfCoupledFusion','SelfIndependentFusion','DirectCoupled','NoComposition']
def who():return dict(pid=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
def verify():
 p=read(D/'plan.json');assert read(R/'outputs/tastvg_paper_v1/state.json')['status']=='paused_by_user';assert sha(PHASE/'sealed.json')==p['phase_seal_sha256']
 for f,h in p['code'].items():assert sha(R/f)==h,f
 for f,h in p['paper_code'].items():assert sha(R/f)==h,f
 for f,h in p['provider_sources'].items():assert sha(R.parent/'stvg-opd'/f)==h,f
 assert time.time()<p['stop_epoch'];return p

def prepare():
 assert not(D/'plan.json').exists();parent=read(PHASE/'plan.json');code={str(f.relative_to(R)):sha(f)for f in Path(__file__).parent.glob('*.py')}
 for f in ['research/joint_event_v1/run.py','research/boundary_feedback_v1/score.py','research/native_event_composition_v1/core.py']:code[f]=sha(R/f)
 p=dict(schema='native_event_coupling_hc64_v1',primary='MonotoneClauseFusion',arms=ARMS,count=64,code=code,paper_code=parent['paper_code'],provider_sources=parent['provider_sources'],assets=parent['assets'],captures=parent['captures'],phase_files=read(PHASE/'sealed.json')['files'],phase_seal_sha256=sha(PHASE/'sealed.json'),labels_sha256=parent['labels_sha256'],supported=52,
 authorization='User bounded overnight native-event lightweight research through07:00; same8 observations and complete spatial method',rationale='Independent first/last posterior draws pair unrelated occurrences and manufacture long envelopes. Source-centered shared actual context has short-event harm despite full true-event coverage. Test dependence rather than further window/ratio/weight changes. Unlike previous hard ordered assignment, exact common-quantile coupling retains BOTH complete clause interval marginals and never conditions on rare allowed-order mass.',fixed=dict(coupling='Sort all positive interval atoms by physical midpoint, start/end deterministic ties. Exact common-uniform-quantile coupling of ordered lists, minstart/maxend hull. No atom threshold, no discarded mass, marginal errors<1e-12. Not2D OT or established semantic occurrence matching.',primary='.5native+.5coupled hull, original physical MBR/BoxFixed.',controls='Same actual first/last evidence unblended DirectCoupled; exact archived IndependentClauseFusion; antimonotone rank pairing; time-reflected Last monotone pairing; native self-coupling; archived self-independent-hull; exact native/no-op/Source. Same52supported,12unsupported native, noGT/confidence/quality veto.',source='Frozen old native global6 outputs and genuine First/Last posteriors, no new model or labels during prediction. Space/geometry/eight teacher observations unchanged.'),new_RGB=0,new_visual_encoder=0,new_text_encoder=0,new_fusion=0,new_decoder=0,new_expert=0,new_observations=0,new_optimizer=0,weights_frozen=True,GT_in_inference=False,all64_seal_before_new_GT=True,stop_epoch=1792708800,maximum_CPU_seconds=600,exposure=parent['exposure'],qualification='Source/G6 actualt and fullv positive; improved>harmed; Source shortANDlongpositive; exceed independent/self/mismatch controls then fixed dual-domain recheck; no coupling weight/bin/cost scan or posthoc control promotion.',references=parent['references'],scope='Own1D occurrence-coupling structural feasibility; not trained event reasoning/2D transport.')
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();put(D/'plan.json',p);put(P/'protocol.json',{k:v for k,v in p.items()if k not in ['paper_code','provider_sources','assets','captures','phase_files']})
 for f in code:
  q=D/'source_snapshot'/f;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((R/f).read_bytes())
 put(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']));print(p['fingerprint'])
def predict():
 os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 import torch,numpy as np
 from stvg.temporal import final_mbr
 from stvg.spatial import SpatialGeometry,evidence_dp
 from research.native_event_coupling_v1.core import couple
 p=verify();torch.set_num_threads(1);start=time.perf_counter();files={};put(D/'state.json',dict(status='predicting_CPU',fingerprint=p['fingerprint'],**who()))
 for row in records():
  verify();assert time.perf_counter()-start<p['maximum_CPU_seconds'];k=row['id'];e=p['phase_files'][k];assert sha(PHASE/e['trace_path'])==e['trace_sha256']and sha(PHASE/e['path'])==e['sha256'];tr=torch.load(PHASE/e['trace_path'],weights_only=False,map_location='cpu');cf=B/'captures'/f'{k}.pt';assert sha(cf)==p['captures'][k]['capture_sha256'];cap=torch.load(cf,weights_only=False,map_location='cpu');ids=cap['frame_ids'];q0=tr['q']['NativeG6'];tick=time.perf_counter();qs={a:q0.copy()for a in ARMS};audits={}
  if tr['clauses']['supported']:
   first,last=tr['intervened_q']['First'],tr['intervened_q']['Last'];joint,audits['real']=couple(first,last,ids);anti,audits['anti']=couple(first,last,ids,True);refl,audits['reflected']=couple(first,last[::-1,::-1].T.copy(),ids);selfq,audits['self']=couple(q0,q0,ids);assert np.allclose(selfq,q0,atol=1e-12,rtol=0);qs.update(MonotoneClauseFusion=.5*q0+.5*joint,IndependentClauseFusion=tr['q']['ClauseFusion'],AntimonotoneFusion=.5*q0+.5*anti,ReflectedLastFusion=.5*q0+.5*refl,SelfCoupledFusion=.5*q0+.5*selfq,SelfIndependentFusion=tr['q']['SelfHullFusion'],DirectCoupled=joint)
  iv={a:final_mbr(ids,q)['interval']for a,q in qs.items()};geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);arrays={a:evidence_dp(geom.for_interval(iv[a]))['final'].numpy()for a in ARMS}
  with np.load(PHASE/e['path'],allow_pickle=False)as z:
   for a,alias in [('NativeG6','NativeG6'),('IndependentClauseFusion','ClauseFusion'),('SelfIndependentFusion','SelfHullFusion'),('NoComposition','NoComposition')]:assert np.array_equal(arrays[a],z[alias])and iv[a]==z[alias+'_interval'].tolist()
   arrays['Source']=z['Source'].copy();iv['Source']=z['Source_interval'].tolist()
  score=np.zeros_like(q0);take=q0>0;score[take]=np.log(qs['MonotoneClauseFusion'][take])-np.log(q0[take]);f=D/'predictions'/f'{k}.npz';assert not f.exists();f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,**arrays,**{a+'_interval':span for a,span in iv.items()},frame_ids=ids,positions8=cap['planned_positions']);tf=D/'traces'/f'{k}.pt';tf.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(q=qs,score=score,clauses=tr['clauses'],coupling=audits,GT_read=False),tf);files[k]=dict(path=str(f.relative_to(D)),sha256=sha(f),trace_path=str(tf.relative_to(D)),trace_sha256=sha(tf),seconds=time.perf_counter()-tick,supported=tr['clauses']['supported']);put(D/'progress.json',dict(completed=len(files),total=64,seconds=time.perf_counter()-start,**who()));print(k,iv['NativeG6'],iv['MonotoneClauseFusion'],flush=True)
 verify();assert not torch.cuda.is_initialized();put(D/'sealed.json',dict(count=64,files=files,plan_sha256=sha(D/'plan.json'),GT_read=False,counts=dict(new_decoder=0,new_fusion=0,new_text_encoder=0,new_visual_encoder=0,new_expert=0,new_observations=0,new_RGB=0,new_optimizer=0),seconds=time.perf_counter()-start));put(D/'state.json',dict(status='sealed',fingerprint=p['fingerprint']))
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:put(D/f'failure-{time.time_ns()}.json',dict(error=traceback.format_exc(),partial_preserved=True,**who()));put(D/'state.json',dict(status='engineering_failed',partial_preserved=True));raise
