"""Fixed HC64 event-vs-presence feedback trial. Inference never imports audit."""
import os,sys,time,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.joint_event_v1.run import sha,read,put,records,B
D=R/'outputs/event_presence_hc64_v2';P=R/'results/event_presence_hc64_v2';V1=R/'outputs/joint_event_hc64_v1';OLD=R/'outputs/event_query_hc64_v1'
ARMS=['NativeG6','Conditional','RawVerb','Ownership','Noun','ShiftConditional']
def verify():
 p=read(D/'plan.json');assert read(R/'outputs/tastvg_paper_v1/state.json')['status']=='paused_by_user';assert sha(R/'outputs/tastvg_paper_v1/plan.json')==p['paper_plan_sha256']
 for k,v in p['code'].items():assert sha(R/k)==v,k
 for k,v in p['paper_code'].items():assert sha(R/k)==v,k
 for k,v in p['provider_sources'].items():assert sha(R.parent/'stvg-opd'/k)==v,k
 assert sha(V1/'postseal_role_groups.json')==p['role_groups_sha256'];return p
def prepare():
 assert not(D/'plan.json').exists();old=read(V1/'plan.json');paper=read(R/'outputs/tastvg_paper_v1/plan.json');code={str(f.relative_to(R)):sha(f)for f in Path(__file__).parent.glob('*.py')}
 for k in ['research/event_presence_v1/core.py','research/joint_event_v1/core.py','research/joint_event_v1/run.py','research/joint_event_v2/core.py','research/boundary_feedback_v1/score.py']:code[k]=sha(R/k)
 p=dict(schema='event_presence_hc64_v2',queries=64,arms=ARMS,primary='Conditional',
  rationale='V1 between-slot nuisance calibration extrapolates49/64 beyond3sd. Replace ONLY nuisance estimator by leave-one-time-out same-target calibration; no score/guard/weight/readout change.',
  fixed=dict(calibration='per-anchor OLS verb on nominal logit in other7 times of SAME selected track, held-out time residual; no GT/event-negative assumption',projection='same ridge1 normalized six-layer hidden, both offsets, full8 plus8LOO for all arms; no CV veto',
   posterior='same calibrated occupancy masks and matched-uniform-null log softIoU, q0*exp(score), original physical MBR; NO membership mass preservation or no-update guard',
   shifted='roll calibrated conditional logit profiles by floor(T/4), restore native occupancy; same readout',
   space='unchanged complete-box geometry and DP per final interval; also fixed native-space evaluation'),
  bindings=old['bindings'],prior_files=read(V1/'sealed.json')['files'],role_groups_sha256=sha(V1/'postseal_role_groups.json'),
  paper_code=paper['code'],provider_sources=paper['provider_sources'],paper_plan_sha256=sha(R/'outputs/tastvg_paper_v1/plan.json'),code=code,labels_sha256=old['labels_sha256'],
  no_GT_in_inference=True,all64_seal_before_new_GT=True,new_model_forwards=0,new_expert=0,new_visual_decoder=0,query_steps=0,
  maximum_CPU_seconds=600,exposure='same repeatedly exposed64 parents, development only; preceding label-based teacher audit disclosed',automatic_resume=False,
  references=['https://github.com/iSEE-Laboratory/ReferDINO','https://seung-hun-lee.github.io/projects/TGL/'],
  mechanism_limit='regression cannot create missing event information; action-specific residual is a testable hypothesis, not calibrated event probability')
 p['v1_plan_sha256']=sha(R/'outputs/event_presence_hc64_v1/plan.json');p['v1_seal_sha256']=sha(R/'outputs/event_presence_hc64_v1/sealed.json');p['fingerprint']=hashlib.sha256(__import__('json').dumps(p,sort_keys=True).encode()).hexdigest();put(D/'plan.json',p);put(P/'protocol.json',{k:v for k,v in p.items()if k not in ['bindings','prior_files','paper_code','provider_sources']})
 for k in code:
  f=D/'source_snapshot'/k;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes((R/k).read_bytes())
 put(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']));print(p['fingerprint'])
def predict():
 os.environ['CUDA_VISIBLE_DEVICES']=''
 import torch,numpy as np
 from stvg.runtime import guard
 from stvg.spatial import SpatialGeometry,evidence_dp
 from stvg.temporal import final_mbr
 from research.event_presence_v2.core import conditional_event,word_logits,full_reweight
 from research.joint_event_v1.core import relative_event,projections,calibrated_masks,physical_membership
 from research.joint_event_v2.core import excess_scores
 p=verify();torch.set_num_threads(1);sys.addaudithook(guard);assert not(D/'sealed.json').exists()and not torch.cuda.is_initialized();roles=read(V1/'postseal_role_groups.json');files={};start=time.perf_counter();put(D/'state.json',dict(status='predicting_CPU',fingerprint=p['fingerprint']))
 for row in records():
  k=row['id'];bind=p['bindings'][k];prior=p['prior_files'][k];semfile=OLD/bind['semantic']['path'];capfile=B/'captures'/f'{k}.pt'
  assert sha(semfile)==bind['semantic']['sha256']and sha(capfile)==bind['capture_sha256'];assert sha(V1/prior['trace_path'])==prior['trace_sha256']and sha(V1/prior['path'])==prior['sha256']
  sem=torch.load(semfile,weights_only=False,map_location='cpu');cap=torch.load(capfile,weights_only=False,map_location='cpu');tr=torch.load(V1/prior['trace_path'],weights_only=False,map_location='cpu');ids=cap['frame_ids'];pos=cap['planned_positions'];pairs=tr['pairs'];q0=tr['q']['NativeG6'];base=final_mbr(ids,q0)
  v=roles[k]['predicate_groups'];n=roles[k]['nominal_groups'];assert v and n;tick=time.perf_counter();x=sem['premax'].numpy();j=sem['selected_query'];res,diag=conditional_event(x,v,n,j)
  labels=dict(Conditional=res,RawVerb=word_logits(x,v)[:,j],Ownership=relative_event(x,v,n,j),Noun=word_logits(x,n)[:,j]);iv={'NativeG6':base['interval']};qs={'NativeG6':q0};details={};saved={}
  overlap,width,length=physical_membership(ids,pairs);occupancy=float(q0[pairs[:,0],pairs[:,1]]@length/width.sum())
  for a,y in labels.items():
   models,skills=projections(cap['temporal_cache']['offsets'],ids,pos,y);masks=calibrated_masks(models,width,occupancy);combined=(masks[0][0]+masks[1][0])/2;score,_=excess_scores(combined,overlap,width,length);q=full_reweight(q0,pairs,score);d=final_mbr(ids,q);iv[a]=d['interval'];qs[a]=q;details[a]=dict(accept=d['interval']!=base['interval'],score_range=float(np.ptp(score)),projection_skill=skills);saved[a]=dict(labels=y,combined=combined,score=score)
   if a=='Conditional':
    shifted=[[np.roll(z,len(ids)//4,axis=0)for z in vv]for vv in models];mm=calibrated_masks(shifted,width,occupancy);ss,_=excess_scores((mm[0][0]+mm[1][0])/2,overlap,width,length);qq=full_reweight(q0,pairs,ss);dd=final_mbr(ids,qq);iv['ShiftConditional']=dd['interval'];qs['ShiftConditional']=qq;details['ShiftConditional']=dict(accept=dd['interval']!=base['interval'],score_range=float(np.ptp(ss)));saved['ShiftConditional']=dict(score=ss)
  feedback=time.perf_counter()-tick;geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);arrays={a:evidence_dp(geom.for_interval(iv[a]))['final'].numpy()for a in ARMS}
  with np.load(V1/prior['path'],allow_pickle=False)as z:
   assert np.array_equal(arrays['NativeG6'],z['NativeG6'])and np.array_equal(iv['NativeG6'],z['NativeG6_interval']);arrays['Source']=z['Source'].copy();iv['Source']=z['Source_interval'].tolist()
  out=D/'predictions'/f'{k}.npz';assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**arrays,**{a+'_interval':span for a,span in iv.items()},frame_ids=ids,positions8=pos)
  trace=D/'traces'/f'{k}.pt';trace.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(q=qs,semantic=saved,details=details,calibration=diag,pairs=pairs,GT_read=False),trace)
  files[k]=dict(path=str(out.relative_to(D)),sha256=sha(out),trace_path=str(trace.relative_to(D)),trace_sha256=sha(trace),feedback_seconds=feedback,seconds=time.perf_counter()-tick)
  put(D/'progress.json',dict(completed=len(files),total=64,seconds=time.perf_counter()-start));print(k,{a:details[a]['accept']for a in details},flush=True);assert time.perf_counter()-start<p['maximum_CPU_seconds']
 verify();put(D/'sealed.json',dict(count=64,files=files,plan_sha256=sha(D/'plan.json'),GT_read=False,CUDA_initialized=torch.cuda.is_initialized(),seconds=time.perf_counter()-start));put(D/'state.json',dict(status='sealed',fingerprint=p['fingerprint']))
if __name__=='__main__':globals()[sys.argv[1]]()
