"""Independent CPU arithmetic/custody for actual fixed GPU qualification."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import math,collections,time
import numpy as np
from scripts.stvg_opd_observation_response_projection_common_v1 import *

def vector(state):return np.concatenate([v.detach().cpu().numpy().ravel() for v in state.values()]).astype(float)

def adam(w,cfg):
 g=w['gradient'].numpy().astype(float);before=w['Adam_before'];after=w['Adam_after'];expected=[];checked=0
 offset=0
 for i,n in enumerate(w['Adam_after']['param_groups'][0]['params']):
  aa=after['state'][n];ab=before['state'].get(n,{})
  gg=g[offset:offset+256];offset+=256
  m0=ab.get('exp_avg',np.zeros(256));v0=ab.get('exp_avg_sq',np.zeros(256))
  if hasattr(m0,'numpy'):m0=m0.numpy().astype(float);v0=v0.numpy().astype(float)
  m=.9*m0+.1*gg;v=.999*v0+.001*gg**2;t=float(aa['step'])
  assert t==float(ab.get('step',0))+1
  assert np.allclose(m,aa['exp_avg'].numpy(),rtol=2e-5,atol=4e-6)
  assert np.allclose(v,aa['exp_avg_sq'].numpy(),rtol=2e-5,atol=4e-6)
  delta=-cfg['lr']*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8)
  expected.extend(delta);checked+=256*3
 assert offset==1792 and np.max(abs(np.asarray(expected)-w['d_A'].numpy()))<4e-6
 return checked

def actual_checks(z,r,reference):
 w=z['witness'];h=z['entry'];counts=collections.Counter()
 if not w['updated']:
  assert w['no_information'] and w['complete_original_no_op'] and 'update' not in h
  counts['no_op_probes']+=1;return counts
 counts['independent_Adam_coordinates']+=adam(w,r['config'])
 if z['arm']=='R':return counts
 Jo=w['J_O'].numpy();da=w['d_A'].numpy();candidate=w['candidate'].numpy();raw=w['actual_candidate'].numpy()
 normA=np.linalg.norm(da);norm=np.linalg.norm(raw);v=Jo@da
 p=np.linalg.lstsq(Jo,v,rcond=1e-10)[0]
 if z['arm']=='P':assert np.linalg.norm(candidate-p)<1e-6
 else:
  Ju=w['J_U'].numpy();positions=w['unobserved_positions'];e=(h['boxes'].numpy()[positions]-reference['fit']['before'].numpy()[positions]).ravel().astype(float)
  # Convex KKT certificate: feasible response/norm and projected gradient
  # stationarity are sufficient for global optimality in this ball problem.
  uu,ss,vv=np.linalg.svd(Jo,full_matrices=False);Q=vv[ss>max(1e-12,1e-10*ss.max(initial=0))]
  n=candidate-p;projected=Ju-(Ju@Q.T)@Q;grad=projected.T@(e+Ju@candidate)/max(1,len(positions))
  lam=w['solver']['lagrange_multiplier']
  error=np.linalg.norm(grad+lam*n)
  assert error<1e-6*(1+np.linalg.norm(grad)),error
  assert np.linalg.norm(Jo@n)<2e-6 and np.linalg.norm(candidate)<=normA+1e-6
  before=np.square(e+Ju@da).sum()/max(1,len(positions));after=np.square(e+Ju@candidate).sum()/max(1,len(positions))
  assert after<=before+1e-7*max(1,before)
  assert abs(before-w['solver']['linear_A_objective'])<1e-6 and abs(after-w['solver']['linear_F_objective'])<1e-6
  counts['independent_convex_KKT_certificates']+=1
 assert np.linalg.norm(Jo@(candidate-da))<2e-6
 obs=np.linalg.norm(w['candidate_mean'].numpy().astype(float)-w['original_proposal_mean'].numpy().astype(float))
 response=np.linalg.norm(w['original_proposal_mean'].numpy().astype(float)-w['mean'].numpy().astype(float))
 tolerance=.1*response+32*np.finfo(np.float32).eps*math.sqrt(w['mean'].numel())
 lin=np.linalg.norm(Jo@(raw-da));lintol=2e-6+2e-5*np.linalg.norm(v)
 tests=dict(observed_response=obs<=tolerance,linear_response=lin<=lintol,parameter_norm=norm<=normA+4e-6)
 pos=w['unobserved_positions'];ref=reference['fit']['before'].numpy()[pos].astype(float)
 DA=np.square(w['original_proposal_boxes'].numpy()[pos].astype(float)-ref).sum()/(2*max(1,len(pos)))
 DC=np.square(w['candidate_boxes'].numpy()[pos].astype(float)-ref).sum()/(2*max(1,len(pos)))
 dtol=1e-10+64*np.finfo(np.float32).eps*DA
 if z['arm']=='F':tests.update(actual_functional_nonincrease=DC<=DA+dtol,actual_functional_improvement=DC<DA-dtol)
 assert tests==w['acceptance'] and w['fallback']==(not all(tests.values()))
 assert abs(obs-w['observed_error'])<1e-10 and abs(DA-w['D_A'])<1e-10 and abs(DC-w['D_candidate'])<1e-10
 if w['fallback']:assert np.array_equal(w['actual_raw'].numpy().astype(float),da)
 else:assert np.array_equal(w['actual_raw'].numpy().astype(float),raw)
 counts['actual_response_norm_functional_checks']+=1
 return counts

def root():
 import torch
 torch.set_num_threads(2);assert not torch.cuda.is_initialized();verify()
 from scripts.stvg_opd_design_ablation_common_v1 import activate_old,audit
 activate_old()
 from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
 from vg_tta.stvg_opd_common_state_geometry_arithmetic004 import bare,mathematical_readback
 from scripts.stvg_opd_legacy_input_bridge003 import normalized
 from scripts.run_decota_paper_main_v1 import unpack_expert
 b=read(BASE/'QUALIFICATION_BARRIER.json');assert b['status']=='sealed' and not b['GT_read'] and len(b['files'])==160
 lookup={r['index']:r for r in read(BASE/'DESIGN_LOCK.json')['records']}
 rows=[];counts=collections.Counter();comparisons=dict(tensors=0,tensor_coordinates=0,scalar_values=0);fd=[]
 for rel,hashed in b['files'].items():
  assert sha(ROOT/rel)==hashed;z=load(ROOT/rel);r=lookup[z['index']];reference=load(ROOT/r['A_path'])
  assert z['runtime_sha256']==sha(BASE/'RUNTIME.json') and not z['GT_read']
  if 'witness' not in z:
   inp=normalized(load(ROOT/r['originals'][0]['input_path']));ex=unpack_expert(inp['expert'])
   equal(bare(z['fit']),bare(reference['fit']),comparisons);equal(z['fit']['geometry'],reference['fit']['geometry'],comparisons)
   assert audit(z['fit'],ex)==z['audit']['objective'];m=mathematical_readback(z['fit']);assert m==z['audit']['Adam']
   counts['original_complete_replays']+=1;counts['original_Adam_coordinates']+=m['scalar_checks'];continue
  assert z['revision_runtime_bindings'] and all(sha(ROOT/n)==h for n,h in z['revision_runtime_bindings'].items())
  counts.update(actual_checks(z,r,reference));w=z['witness']
  if 'derivative_checks' in w:fd.append(w['derivative_checks']);counts['actual_complete_derivative_contexts']+=1
  row=dict(parent_id=z['index'],dataset=z['dataset'],arm=z['arm'],boundary=z['boundary'],repeat=z['repeat'],
   updated=w['updated'],empty=w.get('empty',False),fallback=w.get('fallback',False),
   accepted=w.get('accepted_projection'),seconds=w['seconds'],fallback_reasons=w.get('fallback_reasons',[]),
   observed_relative_error=w.get('observed_relative_error'),original_norm=w.get('original_norm'),candidate_norm=w.get('candidate_norm'),
   D_A=w.get('D_A'),D_candidate=w.get('D_candidate'),lambda_R=w.get('lambda_R'),
   rank=w.get('solver',{}).get('rank'),Jacobian_rows=w.get('Jacobian',{}).get('actual_scalar_VJP_rows'),
   Jacobian_seconds=w.get('Jacobian',{}).get('seconds'),FD_status=w.get('derivative_checks',{}).get('status'))
  rows.append(row);counts['local_proposals']+=1
  assert sha(ROOT/rel)==hashed
 assert counts['original_complete_replays']==16 and counts['local_proposals']==144 and len(rows)==144
 decisions={}
 for ds in ['hc2','vidstg']:
  probes=[r for r in rows if r['dataset']==ds and r['arm']=='F' and r['updated']]
  accepted=sum(r['accepted'] for r in probes);fraction=accepted/len(probes) if probes else None
  decisions[ds]=dict(informative_probes=len(probes),accepted=accepted,fallback=len(probes)-accepted,fraction=fraction,
   technical_response_condition=bool(probes and fraction>=.5))
 eligible=all(v['technical_response_condition'] for v in decisions.values()) and all(v['status']=='pass' for v in fd)
 write(BASE/'ROOT_QUALIFICATION.json',dict(status='pass',scope='full fixed GPU qualification custody/original complete replay/Adam/KKT/actual response and fallback scalar readback',
  counts=dict(counts),comparisons=comparisons,actual_derivative_contexts=len(fd),finite_difference_checks_all_pass=all(v['status']=='pass' for v in fd),
  formal_eligible=eligible,CPU_only=True,GT_read=False,time=time.time()))
 write(OUT/'QUALIFICATION_ROWS.json',rows)
 write(OUT/'QUALIFICATION_SUMMARY.json',dict(status='complete_actual_qualification',datasets=decisions,counts=dict(counts),formal_eligible=eligible,
  actual_derivative_checks=fd,GT_read=False,new_DINO_calls=0,historically_exposed=True))
 write(BASE/'TECHNICAL_DECISION.json',dict(status='eligible_fixed192_panel' if eligible else 'stop_current_projection_recipe_before_formal',
  scope='actual observed-response/functional viability at 8 fixed queries and three original A boundaries; not GT effectiveness',datasets=decisions,
  formal_eligible=eligible,formal_fits_executed=0,GT_read=False,CPU_only=True,time=time.time()))
 status(BASE/'STATUS.json',dict(status='qualified_pending_fixed_formal192' if eligible else 'qualification_complete_pending_actual_view_public_archive_final',
  done=144,total=144,CPU_only=True,GT_read=False,time=time.time()))
 verify()

if __name__=='__main__':root()
