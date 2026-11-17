from common136 import *
import numpy as np
def main():
 check();assert rd(H136/'METHOD_AUDIT.json')['status']=='PASS';root=B136/'method';assert rd(root/'SEALED.json')['n']==1798;reg=rd(H136/'INPUT_REGISTRY_PRIVATE.json');rows=[];method=[];paired=[];scene=[];changes=[];arms=None
 for e in reg:
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;a=np.load(root/name/'samples_PRIVATE'/(sid+'.npz'));p=dict(zip(a['arm_names'],a['scores']));p['Frozen_B']=np.load(R136/'crossplatform/round135/formal'/name/'evidence_PRIVATE'/(sid+'.npz'))['pB'];arms=list(p);y=np.load(R136/'research/round131/scores_attempt001'/(name+'_'+sid+'_DIAGNOSTIC_IOUS_PRIVATE.npz'))['ious'];ev=np.load(B136/'evidence_PRIVATE'/name/(sid+'.npz'));rec=rd(root/name/'records_PRIVATE'/(sid+'.json'));q={k:int(v.argmax()) for k,v in p.items()};r={k:e[k] for k in ['source','target','sampling_weight','frame_PRIVATE','content_group_PRIVATE']}|{'id_PRIVATE':sid,'scene_PRIVATE':e['row']['scene'],'eligible':rec['eligible'],'effective_wrong':bool(ev['effective_wrong']),'IoUs':{k:float(y[j]) for k,j in q.items()},'queries_PRIVATE':q,'C_groups':int(ev['C'].sum()),'S_groups':int(ev['S'].sum()),'C_has_good':bool(((y>.25)&ev['C'][ev['assignment']]).any()),'A_top_in_C':bool(ev['C'][ev['assignment'][q['Frozen_A']]])};rows.append(r)
 for name in sorted({r['source']+'_to_'+r['target'] for r in rows}):
  rr=[r for r in rows if r['source']+'_to_'+r['target']==name];src,_,tgt=name.split('_')
  for weighted in [False,True]:
   ag='design_weighted' if weighted else 'raw';w=np.array([r['sampling_weight'] if weighted else 1. for r in rr])
   for arm in arms:
    yy=np.array([r['IoUs'][arm] for r in rr]);method.append({'source':src,'target':tgt,'aggregation':ag,'arm':arm,'n':len(rr),'mIoU':float(np.average(yy,weights=w)*100),'Acc25':float(np.average(yy>.25,weights=w)*100),'Acc50':float(np.average(yy>.5,weights=w)*100)})
    for sc in sorted({r['scene_PRIVATE'] for r in rr}):
     sub=[r for r in rr if r['scene_PRIVATE']==sc];ys=np.array([r['IoUs'][arm] for r in sub]);ws=np.array([r['sampling_weight'] if weighted else 1. for r in sub]);scene.append({'source':src,'target':tgt,'scene_PRIVATE':sc,'aggregation':ag,'arm':arm,'n':len(sub),'mIoU':float(np.average(ys,weights=ws)*100),'Acc25':float(np.average(ys>.25,weights=ws)*100),'Acc50':float(np.average(ys>.5,weights=ws)*100)})
   for ctrl in ['Frozen_A','Frozen_F','Frozen_B','AB_mean','direct_C_B_A','wrong_C_B_A','R135_A','R135_F']:
    delta=np.array([r['IoUs']['C_B_A']-r['IoUs'][ctrl] for r in rr]);paired.append({'source':src,'target':tgt,'aggregation':ag,'method':'C_B_A','control':ctrl,'gain_pp':float(np.average(delta,weights=w)*100),'better':int((delta>1e-12).sum()),'worse':int((delta< -1e-12).sum()),'same':int((abs(delta)<=1e-12).sum()),'query_changed':sum(r['queries_PRIVATE']['C_B_A']!=r['queries_PRIVATE'][ctrl] for r in rr),'largest_positive_contribution_pp':float(max(0.,(w*delta).max())/w.sum()*100/6),'largest_negative_contribution_pp':float(min(0.,(w*delta).min())/w.sum()*100/6)})
   for r in rr:
    if weighted:continue
    for ctrl in ['Frozen_A','Frozen_F','direct_C_B_A','wrong_C_B_A','R135_A']:
     delta=r['IoUs']['C_B_A']-r['IoUs'][ctrl]
     if abs(delta)>1e-12:changes.append({k:r[k] for k in ['source','target','id_PRIVATE','frame_PRIVATE','content_group_PRIVATE','scene_PRIVATE','sampling_weight','eligible','C_groups','C_has_good']}|{'control':ctrl,'delta_IoU':delta,'method_IoU':r['IoUs']['C_B_A'],'control_IoU':r['IoUs'][ctrl]})
 agg=[]
 for ag in ['raw','design_weighted']:
  for arm in arms:
   mm=[r for r in method if r['aggregation']==ag and r['arm']==arm];agg.append({'aggregation':ag,'arm':arm,**{k:float(np.mean([r[k] for r in mm])) for k in ['mIoU','Acc25','Acc50']}})
 controls=['Frozen_A','Frozen_F','Frozen_B','AB_mean','direct_C_B_A','wrong_C_B_A'];decision={'primary':'C_B_A','GT_diagnostic_is_not_a_method':True,'comparisons':{},'source_training':False,'FINAL_access':False,'target_GT_hyperparameter_or_sample_selection':False,'independent_validation':False}
 for ag in ['raw','design_weighted']:
  v={r['arm']:r['mIoU'] for r in agg if r['aggregation']==ag};decision['comparisons'][ag]={k:v['C_B_A']-v[k] for k in controls}
 decision['unlabeled_candidate_retained']=all(x>0 for rr in decision['comparisons'].values() for x in rr.values());decision['current_working_version']='R136_Cconditional_BtoA_development' if decision['unlabeled_candidate_retained'] else 'R135_intact_reader_baselines_retained_R136_recipe_not_adopted'
 for n,x in [('ALL_ROWS_PRIVATE',rows),('METHOD_ROWS',method),('SCENE_ROWS_PRIVATE',scene),('PAIRED_ROWS',paired),('AGGREGATE_ROWS',agg),('CHANGED_CASES_PRIVATE',changes)]:dump(H136/('METHOD_'+n+'.json'),x)
 dump(H136/'DECISION.json',decision);print(decision,flush=True)
if __name__=='__main__':main()
