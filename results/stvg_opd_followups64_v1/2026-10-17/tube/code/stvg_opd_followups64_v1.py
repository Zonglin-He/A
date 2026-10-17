"""Two separately locked follow-ups: annealed OPD first, Tube feasibility next.

No new Teacher, diffusion network, Temporal update, or production registration.
"""
from pathlib import Path
import collections,copy,gc,hashlib,inspect,json,math,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,sha,write,guard
CORE=ROOT/'artifacts/stvg_spatial_controls256_v1'
FIXED=ROOT/'artifacts/stvg_factorial_native256_captionfix002'
BASE=ROOT/'artifacts/stvg_opd_anneal_and_tube64_v1'
def put(p,d,replace=False):return write(p,d,mutable=replace)
def verify():
 d=read(BASE/'EXECUTION_RUNTIME001.json');assert time.time()<d['deadline_epoch']
 for p,h in d['pins'].items():assert sha(ROOT/p)==h,p
 return d
def schedules(sigma):
 return dict(constant=([sigma*math.sqrt(7/12)]*3,[0,1,2]),coarse_to_fine=([sigma,sigma/math.sqrt(2),sigma/2],[0,1,2]),reverse=([sigma/2,sigma/math.sqrt(2),sigma],[2,1,0]))

def complete_residual(source,positions,reference,seconds,drop=None):
 """Gaussian residual prior with Brownian increments on actual physical time.

 Loss: mean anchor residual error + mean residual magnitude + mean edge
 squared increments / (actual seconds / one second). Both coefficients=1.
 No reference or invalid logit chart gives an explicit whole-query no-op.
 """
 import numpy as np
 b=np.asarray(source,dtype=np.float64);t=np.asarray(seconds,dtype=np.float64);e=np.asarray(reference,dtype=np.float64).reshape(-1,4);pp=list(map(int,positions));n=len(b)
 if drop is not None:e=np.delete(e,drop,axis=0);pp.pop(drop)
 if not pp:return b.copy(),np.zeros_like(b),dict(status='exact_noop',reason='no_reference',used_references=0)
 if not(np.isfinite(b).all() and np.isfinite(e).all() and ((b>0)&(b<1)).all() and ((e>0)&(e<1)).all()):return b.copy(),np.zeros_like(b),dict(status='exact_noop',reason='noninvertible_logit_chart_no_clamping',used_references=len(pp))
 dt=np.diff(t);assert len(t)==n and n>1 and (dt>0).all()
 x=np.log(b)-np.log1p(-b);y=(np.log(e)-np.log1p(-e))-x[pp];k=len(pp)
 H=np.eye(n)/n;rhs=np.zeros((n,4))
 for i,p in enumerate(pp):H[p,p]+=1/k;rhs[p]+=y[i]/k
 for i,d in enumerate(dt):
  w=1/((n-1)*d);H[i,i]+=w;H[i+1,i+1]+=w;H[i,i+1]-=w;H[i+1,i]-=w
 from scipy.linalg import cho_factor,cho_solve
 r=cho_solve(cho_factor(H,lower=True,check_finite=True),rhs);relative=float(np.linalg.norm(H@r-rhs)/max(np.linalg.norm(rhs),1e-30));assert relative<1e-10
 out=__import__('scipy.special',fromlist=['expit']).expit(x+r);assert np.isfinite(out).all() and ((out>0)&(out<1)).all()
 return out,r,dict(status='completed',used_references=k,linear_residual_relative=relative,anchor_term=float(((r[pp]-y)**2).sum(1).mean()),magnitude_term=float((r*r).sum(1).mean()),time_term=float(((np.diff(r,axis=0)**2).sum(1)/dt).mean()),lambda0=1.,lambda_time_seconds=1.,GT_read=False)

def adapted_fitter(family,arm,original_positions=None,original_boxes=None):
 """Versioned source adapter; the existing pinned fitter is never edited."""
 from scripts import stvg_spatial_lite_fit_v1 as lite
 src=inspect.getsource(lite.fit);ns=dict(lite.__dict__)
 if family=='anneal':
  needle='        for step in range(cfg["steps"]):';assert src.count(needle)==1;src=src.replace(needle,needle+'\n            cfg = dict(cfg, sigma=float(SIGMAS[step]))')
  assert src.count('policy = rollout(mean, evidence, key, step, config=cfg)')==1;src=src.replace('policy = rollout(mean, evidence, key, step, config=cfg)','policy = rollout(mean, evidence, key, NOISE_SLOTS[step], config=cfg)')
  assert src.count('"step": step,')==1;src=src.replace('"step": step,','"step": step, "sigma": float(cfg["sigma"]), "noise_slot": int(NOISE_SLOTS[step]),')
  assert src.count('stopped = True')==1;src=src.replace('stopped = True','stopped = False  # reject this proposal, preserve rollback, try next fixed scale')
  ns.update(SIGMAS=arm[0],NOISE_SLOTS=arm[1])
 elif family=='tube':
  before='len(ids), active_positions, response_scope, event_support_positions)';assert src.count(before)==1;src=src.replace(before,'len(ids), ACCEPT_POSITIONS, response_scope, event_support_positions)')
  replacements={
   'aligned_iou(boxes_before[active_positions], evidence)':'aligned_iou(boxes_before[ACCEPT_POSITIONS], ACCEPT_BOXES.to(boxes_before))',
   'aligned_iou(candidate_boxes[active_positions], evidence)':'aligned_iou(candidate_boxes[ACCEPT_POSITIONS], ACCEPT_BOXES.to(candidate_boxes))',
   'reference_iou_before.to(candidate_iou), candidate_iou, anchor_weights)':'reference_iou_before.to(candidate_iou), candidate_iou, torch.ones_like(candidate_iou))',
  }
  for a,b in replacements.items():assert src.count(a)==1,a;src=src.replace(a,b)
  ns.update(ACCEPT_POSITIONS=list(original_positions),ACCEPT_BOXES=original_boxes)
 else:raise ValueError(family)
 exec(compile(src,__file__,'exec'),ns);return ns['fit']

def contracts():
 import numpy as np
 b=np.tile([.5,.5,.2,.2],(5,1));e=b[[0,4]].copy();t=[0,.1,.5,1.7,2.]
 out,r,d=complete_residual(b,[0,4],e,t);assert np.array_equal(out,b) and np.count_nonzero(r)==0
 e[0,0]=.6;out,r,d=complete_residual(b,[0,4],e,t);assert d['status']=='completed' and d['linear_residual_relative']<1e-10 and np.isfinite(out).all()
 out2,_,_=complete_residual(b,[0,4],e,[0,.5,1.,1.5,2.]);assert not np.array_equal(out,out2)
 z,rz,_=complete_residual(b,[],[],t);assert np.array_equal(z,b) and np.count_nonzero(rz)==0
 sch=schedules(.4);assert sch['coarse_to_fine'][0]==list(reversed(sch['reverse'][0]));assert abs(sum(x*x for x in sch['coarse_to_fine'][0])-sum(x*x for x in sch['constant'][0]))<1e-14
 adapted_fitter('anneal',sch['constant']);adapted_fitter('tube',None,[0,4],None)
 return dict(status='pass',count=7,CPU_only=True,synthetic_only=True,GT_read=False)

def setup():
 import torch,numpy as np
 from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
 from scripts import run_stvg_adjacent_motion_student_v1 as h
 sys.addaudithook(guard);BASE.mkdir(exist_ok=False);rs=read(CORE/'ROSTER_LOCK.json')['records'];chosen=[]
 for ds in ('hc2','vidstg'):chosen.extend(sorted([r for r in rs if r['dataset']==ds],key=lambda r:(r['parent_sha256'],r['probe_alias']))[:32])
 assert len({r['parent_sha256'] for r in chosen})==64
 oldbase=FIXED/'arms/B/trials/G32';bar=read(oldbase/'GLOBAL_PREDICTION_BARRIER.json');assert bar['status']=='complete' and bar['count']==256 and not bar['GT_read'];index={r['case_id']:r for r in bar['files']};rows=[];proposals=[]
 put(BASE/'ROSTER_LOCK.json',dict(status='locked',count=64,per_domain=32,records=chosen,selection='lowest existing parent SHA256 within each domain, then original alias tie break; no outcome/admission/GT screen',cohort='already exposed core256 development',GT_read=False))
 put(BASE/'CPU_CONTRACT_COMPLETION.json',contracts())
 for rec in chosen:
  cid=rec['probe_alias'];case,cp=load_case(rec);f=index[cid];assert sha(ROOT/f['path'])==f['sha256'];old=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False);ids=list(map(int,case['frame_ids']));anchors=h._anchors(old['reference_records'],ids,torch);positions=[a['position'] for a in anchors];ref=np.array([a['box'].tolist() for a in anchors]);src=case['zero']['boxes'].cpu().numpy();assert np.array_equal(h._xyxy(torch.as_tensor(src)),old['arms']['T']['boxes'])
  proposal,residual,proof=complete_residual(src,positions,ref,np.array(ids)/case['input']['fps']);out=copy.deepcopy(old['arms']['T']);out['boxes']=h._xyxy(torch.as_tensor(proposal));loo=None
  if len(positions)>=2:
   drop=int(rec['parent_sha256'][:16],16)%len(positions);lb,_,lp=complete_residual(src,positions,ref,np.array(ids)/case['input']['fps'],drop=drop);loo=dict(heldout_reference_index=drop,heldout_position=positions[drop],heldout_prediction_cxcywh=lb[positions[drop]].tolist(),heldout_pseudoreference_cxcywh=ref[drop].tolist(),not_GT=True,completion_status=lp['status'])
  p=BASE/'tube/proposals'/(cid+'.pt');p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();torch.save(dict(case_id=cid,source_cxcywh=src,proposal_cxcywh=proposal,residual_logit=residual,proposal=out,completion=proof,leave_one_reference=loo,GT_read=False),p);proposals.append(dict(case_id=cid,path=str(p.relative_to(ROOT)),sha256=sha(p)))
  rows.append(dict(case_id=cid,dataset=rec['dataset'],old_student_path=f['path'],old_student_sha256=f['sha256'],source_path=str(cp.relative_to(ROOT)),source_sha256=sha(cp),source_case_sha256=case['source_case_sha256'],reference_record=old['reference_records'],original_anchor_positions=positions,protection_positions=[i for i in range(len(ids)) if i not in set(positions)],native_frame_ids=ids,final_Temporal_interval=list(old['arms']['T']['interval']),proposal_path=str(p.relative_to(ROOT)),proposal_sha256=sha(p),GT_read=False));del old,case;gc.collect()
 put(BASE/'INPUT_LOCK.json',dict(status='locked',count=64,records=rows,reference_scope='caption-fixed Native-clock B, original complete E1 Teacher selection; final Temporal is existing T4 for all arms',GT_read=False))
 put(BASE/'tube/PROPOSAL_GLOBAL_BARRIER.json',dict(status='complete',count=64,files=proposals,GT_read=False))
 method=dict(status='locked',count=64,per_domain=32,execution_order=['anneal','tube'],new_Teacher_calls=0,new_Temporal_fits=0,diffusion_training=False,old_decoder_aligned22_scope_remains_paused=True,anneal=dict(arms=['constant','coarse_to_fine','reverse'],stages=3,scale_factors=[1.,1/math.sqrt(2),.5],constant_factor=math.sqrt(7/12),same_standard_noise_per_case_scale_slot=True,rejected_proposal='restore parameters AND Adam state; proceed to next stage in all3 arms',recipe='lite_fullstep',historical_original_OPD='HC5/Vid3 original early-stop only historical baseline, not matched primary',primary='coarse_to_fine minus reverse AND constant, actual Student outcomes and stage displacements',same_original_reference_acceptance_and_protection=True),tube=dict(arms=['P0_F','P0_O','P0_R','P0_R_to_S'],lambda0=1.,lambda_time_seconds=1.,time_operator='(r_next-r)/sqrt(actual physical delta seconds /1second)',reward_positions='every original native-grid position for valid completion, fixed before GT',protection_and_acceptance_positions='original sparse Teacher positions and original full-grid-minus-sparse protection, NOT recomputed from dense reward positions',updater='same fixed-sigma original Gaussian OPD/Adam/M2/fullstep/early-stop; no new Jacobian target optimizer',no_reference_or_invalid_logit='whole-query Source no-op kept in denominator',P0_R='external proposal diagnostic, not Student TTA'),evaluation='all64 predictions seal before GT; fixed original Teacher anchors, other native GT positions, off-grid interpolated GT positions; unknown preserved; paired source t95 separately by domain, no success or significance gate for feasibility',CURRENT_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'),GT_read=False)
 put(BASE/'METHOD_LOCK.json',method)
 paths=[Path(__file__),ROOT/'scripts/run_stvg_opd_followups64_v1.py',ROOT/'scripts/score_stvg_opd_followups64_v1.py',ROOT/'scripts/continue_stvg_opd_followups64_v1.py',ROOT/'scripts/stvg_spatial_lite_fit_v1.py',ROOT/'scripts/stvg_soft_ep_response_v1.py',ROOT/'scripts/stvg_pipeline_speed_v1.py',ROOT/'scripts/stvg_weightfree_coordinate_fit002.py',ROOT/'vg_tta/decota_spatial_opd_tunable_v1.py',ROOT/'methods/CURRENT_METHOD.json',BASE/'ROSTER_LOCK.json',BASE/'INPUT_LOCK.json',BASE/'METHOD_LOCK.json',oldbase/'GLOBAL_PREDICTION_BARRIER.json',oldbase/'REFERENCE_LOCK.json',oldbase/'DESIGN_LOCK.json']
 put(BASE/'EXECUTION_RUNTIME001.json',dict(status='locked',count=64,pins={str(p.relative_to(ROOT)):sha(p) for p in paths},deadline_epoch=time.time()+21600,maximum_new_fit_calls=256,maximum_extra_qualification_fit_calls=0,GT_read=False));put(BASE/'STATUS.json',dict(status='prepared',stage='anneal_first_then_tube',count=64,GT_read=False));print('ANNEAL3_THEN_TUBE64_LOCKED',flush=True)

if __name__=='__main__':setup()
