"""Separate structural repair and fixed original cases; no primary replacement."""
from structural_core87 import *
tasks=read(H87/'score_structural336_attempt001/ALL_TASK_ROWS_PRIVATE.json');su=read(H87/'score_structural336_attempt001/TASK_SUMMARY.json')['ALL336'];by={r['id_PRIVATE']:r for r in tasks};original=read(H87/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336'];methods=[];comp=[]
for arm in ARMS87:
 methods.append({'platform':'balanced','arm':arm,'mIoU_percent':su['balanced_mIoU_percent'][arm]})
 for pl in ['drone','quad']:methods.append({'platform':pl,'arm':arm,**su[pl][arm]})
for mode in MODES87:
 for readout in ['direct','half','tta1']:
  full='full_'+readout;arm=mode+'_'+readout;d=np.array([r['IoU'][full]-r['IoU'][arm] for r in tasks]);comp.append({'component_removed':mode,'readout':readout,'full_mIoU_percent':su['balanced_mIoU_percent'][full],'ablated_mIoU_percent':su['balanced_mIoU_percent'][arm],'component_net_pp':float(d.mean()*100),'full_better_n':int((d>1e-10).sum()),'full_worse_n':int((d< -1e-10).sum())})
follow=[];public=[]
for c in read(H87/'experiment/GOOD_BAD_CASES_PRIVATE.json'):
 n=dict(c);n['task']=by[c['row']['id']];follow.append(n);r={'case':c['case'],'platform':c['row']['platform']}
 for arm in ['full_direct','full_half','full_tta1','wrong_tta1','hard_alias_tta1','no_alias_tta1','independent_tta1','no_subject_tta1','original_full_tta1']:r[arm+'_IoU']=n['task']['IoU'][arm]
 public.append(r)
dump(H87/'experiment/STRUCTURAL_FOLLOWUP_CASES_PRIVATE.json',follow)
for name,obj in [('STRUCTURAL_METHOD_ROWS',methods),('STRUCTURAL_COMPONENT_ROWS',comp),('STRUCTURAL_CASE_FOLLOWUP_ROWS',public)]:dump(H87/(name+'.json'),obj)
v=su['balanced_mIoU_percent'];decision={'state':'R87_SINGLE_STRUCTURAL_REPAIR_DIAGNOSTIC_DEVELOPMENT_ONLY','original_primary_unchanged':True,'secondary_arm':'full_tta1','secondary_gains_pp':{r:v['full_tta1']-v[r] for r in ['frozen_bi','full_direct','full_half','wrong_tta1','original_full_tta1']},'original_source_preserving_half_mIoU_percent':v['original_full_half'],'corrected_full_direct_mIoU_percent':v['full_direct'],'corrected_full_half_mIoU_percent':v['full_half'],'corrected_full_TTA_mIoU_percent':v['full_tta1'],'robust_adoption':False,'after_original_DEV_exposure':True,'no_further_repair_or_tuning':True};dump(H87/'STRUCTURAL_REPAIR_DECISION.json',decision);print(decision)
