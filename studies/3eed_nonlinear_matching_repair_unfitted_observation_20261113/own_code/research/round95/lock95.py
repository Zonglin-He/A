from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).resolve().parents[2];H=R/'research/round95'
for n in ['score','audit_scalar']:
 s=(R/f'research/round94/{n}94.py').read_text().replace('round94','round95').replace('PASS_R94','PASS_R95')
 if n=='score':
  s=s.replace("refs=['R91_frozen','R91_mean3','R59','binding_direct','binding_optimal_mixture','binding_bt1']","refs=['R91_frozen','R94_direct','R59','direct_FC','direct_four','lm_FC']").replace("['binding_bt1']","['lm_FC']")
 (H/f'{n}95.py').write_text(s)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
programs=[f'research/round95/{s}95.py' for s in ['core','run','audit','score','audit_scalar']]+['tools/manual_r95_guard.py']
inputs=['research/round95/experiment/DEV336_PRIVATE.json','crossplatform/round94/prediction336_attempt001/SEALED.json','crossplatform/round94/solver_repair336_attempt001/SEALED.json','crossplatform/round91/trained_source_attempt001/SELECTED_SOURCE_ONLY.json','crossplatform/round91/trained_source_attempt001/states_PRIVATE/R76_full_disc_e1.npz','research/round95/engineering_native_audit/AUDIT_RESULT.json']
p={'round':'R95','locked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'primary':'lm_FC','n':336,'arms':26,'source_training':False,'FINAL_access':False,'TTA_parameters':18496,'target':'same sealed R94 binding contrast distribution','loss':'centered log probability squared error','max_iterations':10,'max_damping_trials_per_iteration':6,'initial_damping_trace_factor':.01,'accept_gain_ratio_above':.1,'expand_above':.75,'contract_below':.25,'relative_loss_stop':.001,'absolute_loss_stop':1e-12,'damping_on_reject':10,'damping_on_good_accept':1/3,'damping_on_weak_accept':2,'weight_L2_cap_primary':None,'capped_control_per_step_fraction':.05,'unfitted_observation_mask_seed':76003,'unfitted_observation_used_only_after_fitting':True,'GT_for_acceptance_or_selection':False,'FP32_probability_tolerance':1e-6,'FP32_parameter_tolerance':1e-7,'max_native_top1_mismatches':0,'programs_sha256':{f:sha(R/f) for f in programs},'inputs_sha256':{f:sha(R/f) for f in inputs}}
out=H/'experiment/DESIGN_LOCK.json';assert not out.exists();out.write_text(json.dumps(p,indent=2)+'\n')
s=json.loads((H/'STATE.json').read_text());s.update(current_stage='locked_full_prediction_before_GT',design_lock_sha256=sha(out));(H/'STATE.json').write_text(json.dumps(s,indent=2)+'\n');print({'locked':sha(out),'programs':len(programs),'inputs':len(inputs)})
