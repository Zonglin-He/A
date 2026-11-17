"""Read cached fixed-box IoUs after all endpoint predictions sealed; steps never selected by GT."""
from common133 import *
import numpy as np
O133=H133/'scores_attempt001'
def metrics(rr,k,weighted):
 y=np.array([r['IoUs'][k] for r in rr]);w=np.array([r['sampling_weight'] if weighted else 1. for r in rr]);return {'n':len(rr),'mIoU_percent':float(np.average(y,weights=w)*100),'Acc25_percent':float(np.average(y>.25,weights=w)*100),'Acc50_percent':float(np.average(y>.5,weights=w)*100)}
def main():
 assert rd(H133/'NATIVE_AUDIT.json')['n']==1798;mat=rd(B133/'MATRIX_SEALED.json');reg=rd(H133/'INPUT_REGISTRY_PRIVATE.json');oldrows={(r['source'],r['target'],r['id_PRIVATE']):r for p in (R133/'research/round132/scores_attempt001').glob('*_ALL_ROWS_PRIVATE.json') for r in rd(p)};O133.mkdir(exist_ok=False);method=[];paired=[];scenes=[];allrows=[];compare=[('partial'+str(s)+'_F',c) for s in [1,3,5] for c in ['zero_F','direct_F','wrong'+str(s)+'_F']]+[('partial1_F','partial3_F'),('partial5_F','partial3_F')]
 for name in mat['seals']:
  src,_,tgt=name.split('_');D=B133/'formal'/name;rows=[]
  for e in [e for e in reg if e['source']==src and e['target']==tgt]:
   sid=e['row']['id'];a=np.load(D/'samples_PRIVATE'/(sid+'.npz'));ps=dict(zip(a['arm_names'],a['scores']));q={k:int(v.argmax()) for k,v in ps.items()};rec=rd(D/'records_PRIVATE'/(sid+'.json'));y=np.load(R133/'research/round131/scores_attempt001'/(name+'_'+sid+'_DIAGNOSTIC_IOUS_PRIVATE.npz'))['ious'];ious={k:float(y[j]) for k,j in q.items()};old=oldrows[(src,tgt,sid)]
   for k,oldk in [('zero_F','LOG_zero'),('direct_F','LOG_direct'),('partial3_F','LOG_partial'),('wrong3_F','LOG_wrong'),('R131_partial_F','partial_F')]:assert abs(ious[k]-old['IoUs'][oldk])<1e-14
   with np.load(R133/'crossplatform/round132/log_scopes_PRIVATE'/name/(sid+'.npz')) as f:ass=f['assignment'];S=f['S'];C=f['C'];qs=S[ass];qc=C[ass]
   q0=q['zero_F'];good=y>.25;diag={}
   for step in [1,3,5]:
    k='partial'+str(step)+'_F';qa=q[k];alternatives=np.flatnonzero(ass!=ass[q0]);j=int(alternatives[np.argmax(ps['zero_F'][alternatives])]) if len(alternatives) else int(np.argsort(ps['zero_F'])[-2]);margin0=float(np.log(ps['zero_F'][q0])-np.log(ps['zero_F'][j]));margin=float(np.log(ps[k][q0])-np.log(ps[k][j]));d=ious[k]-ious['zero_F'];diag[str(step)]={'query_changed':bool(qa!=q0),'geometric_group_changed':bool(ass[qa]!=ass[q0]),'same_group_query_switch':bool(qa!=q0 and ass[qa]==ass[q0]),'IoU_changed':bool(abs(d)>1e-12),'IoU_gain':d,'q0_in_S':bool(S[ass[q0]]),'q0_in_C':bool(C[ass[q0]]),'selected_in_S':bool(S[ass[qa]]),'selected_in_C':bool(C[ass[qa]]),'original_top_vs_best_other_group_margin_change':margin-margin0}
   rows.append({k:e[k] for k in ['source','target','frame_PRIVATE','content_group_PRIVATE','sampling_weight']}|{'id_PRIVATE':sid,'scene_PRIVATE':e['row']['scene'],'active':e['active'],'IoUs':ious,'diagnostics':diag,'seconds_new_joint_work':rec['seconds_new_joint_work'],'recoverable_zero_error':bool(good.any() and y[q0]<=.25),'S_has_good':bool((qs&good).any()),'C_has_good':bool((qc&good).any()),'zero_already_IoU_oracle':bool(y[q0]>=y.max()-1e-8),'q0_gt_Acc25':bool(y[q0]>.25)})
  dump(O133/(name+'_ALL_ROWS_PRIVATE.json'),rows);allrows+=rows
  for weighted in [False,True]:
   aggregation='design_weighted' if weighted else 'raw'
   for k in mat['arms']:
    method.append({'source':src,'target':tgt,'arm':k,'aggregation':aggregation,**metrics(rows,k,weighted)})
    for sc in sorted({r['scene_PRIVATE'] for r in rows}):scenes.append({'source':src,'target':tgt,'scene_PRIVATE':sc,'arm':k,'aggregation':aggregation,**metrics([r for r in rows if r['scene_PRIVATE']==sc],k,weighted)})
   for k,c in compare:
    d=np.array([r['IoUs'][k]-r['IoUs'][c] for r in rows]);w=np.array([r['sampling_weight'] if weighted else 1. for r in rows]);paired.append({'source':src,'target':tgt,'method':k,'control':c,'aggregation':aggregation,'n':len(rows),'gain_pp':float(np.average(d,weights=w)*100),'better':int((d>1e-12).sum()),'worse':int((d< -1e-12).sum()),'same':int((abs(d)<=1e-12).sum()),'largest_positive_six_direction_contribution_pp':float(max(0,(d*w).max())*100/w.sum()/6),'largest_negative_six_direction_contribution_pp':float(min(0,(d*w).min())*100/w.sum()/6)})
 aggregates=[]
 for aggregation in ['raw','design_weighted']:
  for group in ['all_six_equal','waymo','drone','quad']:
   for k in mat['arms']:
    rr=[r for r in method if r['aggregation']==aggregation and r['arm']==k and (group=='all_six_equal' or r['target']==group)];aggregates.append({'aggregation':aggregation,'group':group,'arm':k,**{m:float(np.mean([r[m] for r in rr])) for m in ['mIoU_percent','Acc25_percent','Acc50_percent']}})
 for name,x in [('ALL_METHOD_ROWS',method),('ALL_PAIRED_ROWS',paired),('ALL_SCENE_ROWS_PRIVATE',scenes),('AGGREGATE_ROWS',aggregates)]:dump(O133/(name+'.json'),x)
 decision={'primary_steps':3,'primary_unchanged_from_R132':True,'target_GT_selects_steps':False,'new_step_configuration_adopted':False,'no_parameter_selection_in_this_round':True,'sensitivity_steps':[1,5],'FINAL_access':False,'comparisons':{}}
 for aggregation in ['raw','design_weighted']:
  v={r['arm']:r['mIoU_percent'] for r in aggregates if r['aggregation']==aggregation and r['group']=='all_six_equal'};decision['comparisons'][aggregation]={str(step):{'mIoU_percent':v['partial'+str(step)+'_F'],'gain_vs_zero_pp':v['partial'+str(step)+'_F']-v['zero_F'],'gain_vs_direct_pp':v['partial'+str(step)+'_F']-v['direct_F'],'gain_vs_matched_wrong_pp':v['partial'+str(step)+'_F']-v['wrong'+str(step)+'_F'],'gain_vs_primary3_pp':v['partial'+str(step)+'_F']-v['partial3_F']} for step in [1,3,5]}
 dump(H133/'DECISION.json',decision);print(decision)
if __name__=='__main__':main()
