"""Saved-only analysis of audited source interventions and previously sealed target controls."""
from pathlib import Path
import sys,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research/round76'))
from common76 import read,dump,sha
H=ROOT/'research/round77';A=H/'source_audit_score_attempt002'
assert read(H/'STATE.json')['CPU_scoring_authorized']
assert read(A/'AUDIT_RESULT.json')['state'].startswith('PASS_')
rows=read(A/'ALL_TASK_ROWS_PRIVATE.json');S=read(A/'SUMMARY.json');E=read(A/'ENDPOINT_SCALARS.json');G=[r for r in E if r['arm']=='gradient_diagnostic'];endpoints=[r for r in E if r['arm']!='gradient_diagnostic'];gate=read(A/'TARGET_GATE.json');names=list(S);mapping={r['row']:f'SOURCE128_{i+1:05d}' for i,r in enumerate(rows)}
public=[]
for r in rows:
 for a in names:public.append({'row':mapping[r['row']],'arm':a,'IoU':r['IoU'][a],'label_eligible':r['label_eligible'],'candidate_oracle_IoU':r['oracle_IoU'],'query_changed_vs_frozen':r['queries'][a]!=r['queries']['bidirectional_frozen'],'source_GT_control':a.startswith('ground_')})
tables={'SOURCE128_TASK_SCALARS':public,'SOURCE128_ENDPOINT_SCALARS':endpoints,'SOURCE128_GRADIENT_SCALARS':G,'SOURCE128_PARAMETER_GRADIENT_SCALARS':read(A/'PARAMETER_GRADIENT_SCALARS.json'),'SOURCE128_OPTIMIZER_STEPS':read(A/'STEP_SCALARS.json'),'SOURCE128_ORDERED_NEGATIVES':read(A/'ALL_ORDERED_NEGATIVES.json'),'SOURCE128_RECORDING_SCALARS':read(A/'RECORDING_SCALARS.json')}
for name,table in tables.items():
 if name!='SOURCE128_TASK_SCALARS':table=[{**r,**({'row':mapping[r['row']]} if 'row' in r else {})} for r in table]
 dump(H/(name+'.json'),table)
ep={(r['row'],r['arm']):r for r in endpoints};mechanism={'state':'R77_AUDITED_SOURCE_DIAGNOSTIC_NOT_DEPLOYABLE_ORACLE','n':128,'recordings':read(A/'AUDIT_RESULT.json')['source_recordings'],'candidate_oracle_mIoU_percent':float(np.mean([r['oracle_IoU'] for r in rows])*100),'label_eligible':sum(r['label_eligible'] for r in rows),'gradient_cosine_mean':float(np.mean([r['raw_gradient_cosine'] for r in G if r['raw_gradient_cosine'] is not None])),'gradient_cosine_median':float(np.median([r['raw_gradient_cosine'] for r in G if r['raw_gradient_cosine'] is not None])),'raw_gradient_positive_cosine_n':sum(r['raw_gradient_cosine'] is not None and r['raw_gradient_cosine']>0 for r in G),'Adam_direction_predicted_ground_loss_decrease_n':sum(r['ground_gradient_dot_aux_Adam_delta']<0 for r in G),'Adam_direction_predicted_ground_loss_increase_n':sum(r['ground_gradient_dot_aux_Adam_delta']>0 for r in G),'target_gate':gate,'arms':{}}
for a in names:
 if a in ('source','bidirectional_frozen'):continue
 ee=[r for r in endpoints if r['arm']==a];mechanism['arms'][a]={'ground_loss_mean':float(np.mean([r['ground_loss'] for r in ee])),'auxiliary_mean':float(np.mean([r['auxiliary'] for r in ee])),'ground_loss_decreased_n':sum(ep[(r['row'],a)]['ground_loss']<ep[(r['row'],'bidirectional_frozen')]['ground_loss']-1e-9 for r in rows),'ground_loss_increased_n':sum(ep[(r['row'],a)]['ground_loss']>ep[(r['row'],'bidirectional_frozen')]['ground_loss']+1e-9 for r in rows),'edit_L2_median':float(np.median([r['edited_scope_L2'] for r in ee])),'edit_L2_min':float(np.min([r['edited_scope_L2'] for r in ee])),'edit_L2_max':float(np.max([r['edited_scope_L2'] for r in ee])),'edited_parameters':ee[0]['edited_parameters'],'mean_representation_change_L2':float(np.mean([r['query_representation_delta_L2'] for r in ee]))}
mechanism['frozen_ground_loss_mean']=float(np.mean([ep[(r['row'],'bidirectional_frozen')]['ground_loss'] for r in rows]));mechanism['frozen_auxiliary_mean']=float(np.mean([ep[(r['row'],'bidirectional_frozen')]['auxiliary'] for r in rows]));dump(H/'SOURCE128_SUMMARY.json',S);dump(H/'MECHANISM_SUMMARY.json',mechanism)
# Source aliases and task values are rechecked independently of the GPU producer.
for a in names:
 v=[r['IoU'] for r in public if r['arm']==a];assert abs(np.mean(v)*100-S[a]['mIoU_percent'])<1e-12
assert len(public)==128*len(names) and len(tables['SOURCE128_ORDERED_NEGATIVES'])==read(A/'AUDIT_RESULT.json')['negative_comparisons']
prior=ROOT/'research/round76/score_dev336_attempt001';priorseal=read(prior/'SEALED.json')
for f,h in priorseal['files'].items():assert sha(prior/f)==h
oldstate=read(H/'experiment/INITIAL_RESEARCH_STATE_PRIVATE.json');newstate=read(ROOT/'research/STATE.json')
for key in ('avis','affordance'):
 if key in oldstate['directions']:assert newstate['directions'][key]==oldstate['directions'][key]
source=ROOT/'crossplatform/round6/waymo_source100/checkpoints/epoch100_model_attempt001.pth';assert sha(source)==read(H/'experiment/DESIGN_LOCK.json')['source_checkpoint_sha256']
result={'state':'PASS_R77_SAVED_SCALARS_RECOMPUTED_AND_PRIOR_CONTROLS_UNCHANGED','n':128,'task_values':len(public),'all_source_ordered_negatives':len(tables['SOURCE128_ORDERED_NEGATIVES']),'all_source_recording_arm_rows':len(tables['SOURCE128_RECORDING_SCALARS']),'source_checkpoint_unchanged':True,'other_paused_registries_unchanged':True,'target_context_values_reused':4032,'new_target_GT_scoring_calls':0,'source_prediction_seal_sha256':sha(ROOT/'crossplatform/round77/source128_attempt001/SEALED.json'),'source_native_audit_sha256':sha(A/'AUDIT_RESULT.json'),'prior_target_score_seal_sha256':sha(prior/'SEALED.json')};dump(H/'REPORT_NUMERIC_AUDIT.json',result)
print({'source_mIoU':{a:S[a]['mIoU_percent'] for a in names},'mechanism':{k:v for k,v in mechanism.items() if k!='arms'}})
