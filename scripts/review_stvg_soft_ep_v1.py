"""CPU readback of actual direction filtering, acceptance, and full Student outputs."""
from pathlib import Path
import sys,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
BASE=ROOT/'artifacts/stvg_soft_ep_opd_v1'
def main():
 import torch,numpy as np
 from scripts import stvg_target_teacher_student_v1 as core
 from scripts.run_stvg_soft_ep_v1 import verify,QUAL
 from scripts.stvg_reference_recovery_common_v1 import load_case,PARENT
 from scripts.stvg_reference_recovery_fit_v1 import locked_config
 from scripts.run_stvg_table1_student004 import stable_fit_key
 verify();stage=sys.argv[1];assert stage in ['qualification','development'];bar=read(BASE/('QUALIFICATION_BARRIER.json'if stage=='qualification'else'STUDENT_GLOBAL_BARRIER.json'));assert bar['status']=='complete'and bar['count']==(4 if stage=='qualification'else 32)and not bar['GT_read']
 roster={r['probe_alias']:r for r in read(BASE/'ROSTER_LOCK.json')['records']};refs={r['case_id']:r for r in read(BASE/'REFERENCE_LOCK.json')['records']};cases=[];checks=accepted=proposed=0;qual=[];rhos=[];cos=[];residuals=[];normbetter=0;normcount=0;modecounts={m:dict(fits=0,accepted=0,proposals=0)for m in ['M1','M2']}
 for item in bar['files']:
  path=ROOT/item['path'];assert sha(path)==item['sha256'];d=torch.load(path,map_location='cpu',weights_only=False);cid=d['case_id'];rec=roster[cid];case,sp=load_case(rec);assert d['reference_records']==refs[cid]and d['source_restored']and not d['GT_read']and d['source_file_sha256']==sha(sp)and d['source_model_state_sha256']==case['source_model_state_sha256']and d['pixel_sha256']==case['pixel_sha256'];assert d['source_case_sha256']==case['source_case_sha256']and d['runtime_sha256']==sha(BASE/'RUNTIME.json')
  parent=torch.load(PARENT/'student'/(cid+'.pt'),map_location='cpu',weights_only=False);assert sha(PARENT/'student'/(cid+'.pt'))==d['parent_student_sha256']
  for a,p in [('F','F'),('T','T'),('S_M0','S_new_crop'),('TS_M0','TS_new_crop')]:assert np.array_equal(d['arms'][a]['boxes'],parent['arms'][p]['boxes'])and d['arms'][a]['interval']==parent['arms'][p]['interval']
  aa=refs[cid]['anchors'];pos=[x['position']for x in aa];empty=not aa
  if cid in QUAL and aa:
   baseline=d['M0_qualification_replay'];old=parent['fits']['new_crop'];assert core._tree_equal(baseline['final_state'],old['final_state'])and torch.equal(baseline['final_cached_decoder_boxes'],old['final_cached_decoder_boxes'])
  for mode in ['M1','M2']:
   f=d['fits'][mode];o=d['readouts'][mode]
   if empty:
    assert f is None and o is None and np.array_equal(d['arms']['S_'+mode]['boxes'],d['arms']['F']['boxes'])and np.array_equal(d['arms']['TS_'+mode]['boxes'],d['arms']['T']['boxes']);continue
   modecounts[mode]['fits']+=1;assert f['config']==locked_config(d['dataset'],'current5')and f['seed_key']==stable_fit_key(d['dataset'],rec['query_key_sha256'])and core._tree_equal(f['initial_state'],case['source_light_initial_state'])
   assert torch.equal(o['boxes'].cpu(),f['final_cached_decoder_boxes']);bb=o['boxes'].cpu().double().numpy();xy=np.concatenate((bb[:,:2]-bb[:,2:]/2,bb[:,:2]+bb[:,2:]/2),axis=1);assert np.array_equal(xy,d['arms']['S_'+mode]['boxes'])and np.array_equal(xy,d['arms']['TS_'+mode]['boxes'])
   for row in f['path']:
    if row.get('skipped'):continue
    proposed+=1;modecounts[mode]['proposals']+=1
    solver=row['solver'];dire=row['filtered_delta'].double();raw=row['original_adam_delta'].double();state=row['state_before'];after=row['state_after']
    if mode=='M2'and solver:
     normal=solver.get('normal_applied',solver['cg_rhs']);rr=solver['cg_rhs'].double()-normal.double();relative=float(rr.norm()/raw.norm().clamp_min(1e-30));assert abs(relative-solver['solver_residual_rel'])<=1e-5
     rho=float(dire.norm()/raw.norm());assert abs(rho-solver['rho'])<1e-5 and torch.isfinite(dire).all();rhos.append(rho);cos.append(solver['cos']);residuals.append(relative)
    if row['accepted']:
     accepted+=1;modecounts[mode]['accepted']+=1;c=row['candidates'][-1];assert c['accepted']and c['train_pass']and c['response_pass']and c['U_RMS']<=row['response_budget']+1.01e-7
     actual=torch.cat([(after[n]-state[n]).flatten()for n in state]).double();expected=dire*row['accepted_scale'];assert torch.max(abs(actual-expected))<=2e-7;assert actual.norm()>0
    else:assert core._tree_equal(state,after)and core._tree_equal(row['optimizer_before'],row['optimizer_after'])
    n=row.get('norm_control')
    if n:
     assert abs(float(n['matched_direction'].double().norm()-n['soft_direction'].double().norm()))<=1e-6;normcount+=1;normbetter+=n['soft_U_RMS']<n['matched_U_RMS']
    q=row.get('derivative_qualification')
    if q:
     assert q['adjoint_pass'];qual.extend([dict(case_id=cid,h=x['h'],FD_relative_error=x['relative_error'],FD_absolute_error=x['absolute_error'],gross_agreement=x['gross_agreement'])for x in q['finite_differences']]);assert all(x['gross_agreement']for x in q['finite_differences'])
    checks+=1
   assert f['optimizer_steps']==sum(bool(r.get('accepted'))for r in f['path'])
  cases.append(cid)
 assert len(set(cases))==bar['count']
 result=dict(status='pass',stage=stage,count=len(cases),mode_counts=modecounts,proposal_count=proposed,accepted_nonzero_count=accepted,actual_full_readout_count=sum(v['fits']for v in modecounts.values()),norm_matched_counterfactual_count=normcount,soft_lower_actual_response_at_equal_norm=normbetter,parameter_rho_range=[min(rhos),max(rhos)]if rhos else None,direction_cos_range=[min(cos),max(cos)]if cos else None,max_CG_relative_residual=max(residuals,default=0.),FD_diagnostics=qual,scope='saved source/reference/step/equal-norm process/actual full-model versus cached outputs; adjoint and finite-difference qualification with raw errors retained; no independent backward or universal accuracy claim',bindings=dict(runtime_sha256=sha(BASE/'RUNTIME.json'),barrier_sha256=sha(BASE/('QUALIFICATION_BARRIER.json'if stage=='qualification'else'STUDENT_GLOBAL_BARRIER.json'))),GT_read=False,CPU_only=True)
 write(BASE/('ROOT_QUALIFICATION.json'if stage=='qualification'else'ROOT_READBACK.json'),result)
 if stage=='development':
  pins=[]
  for x in bar['files']:
   pins.extend([x,dict(path=str((ROOT/x['path']).with_suffix('.receipt.json').relative_to(ROOT)),sha256=sha((ROOT/x['path']).with_suffix('.receipt.json')))])
  for n in ['DESIGN_LOCK.json','ROSTER_LOCK.json','REFERENCE_LOCK.json','RUNTIME.json','ROOT_READBACK.json','STUDENT_GLOBAL_BARRIER.json','QUALIFICATION_BARRIER.json','ROOT_QUALIFICATION.json']:pins.append(dict(path=str((BASE/n).relative_to(ROOT)),sha256=sha(BASE/n)))
  write(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=pins,student_barrier_sha256=sha(BASE/'STUDENT_GLOBAL_BARRIER.json'),root_readback_sha256=sha(BASE/'ROOT_READBACK.json'),scorer_sha256=sha(ROOT/'scripts/measure_stvg_soft_ep_v1.py'),GT_read=False))
 print(result)
if __name__=='__main__':main()
