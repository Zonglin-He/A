"""Post-seal cached fixed-box task scores; no target-GT choice of configuration."""
from common134 import *
import numpy as np
O134=H134/'scores_attempt001'
def metrics(rr,k,weighted):
 y=np.array([r['IoUs'][k] for r in rr]);w=np.array([r['sampling_weight'] if weighted else 1. for r in rr]);return {'n':len(rr),'mIoU_percent':float(np.average(y,weights=w)*100),'Acc25_percent':float(np.average(y>.25,weights=w)*100),'Acc50_percent':float(np.average(y>.5,weights=w)*100)}
def main():
 assert rd(H134/'NATIVE_AUDIT.json')['n']==1798;mat=rd(B134/'MATRIX_SEALED.json');mat['arms']+=rd(B134/'POLICY_SEALED.json')['arms'];reg=rd(H134/'INPUT_REGISTRY_PRIVATE.json');oldrows={(r['source'],r['target'],r['id_PRIVATE']):r for p in (R134/'research/round133/scores_attempt001').glob('*_ALL_ROWS_PRIVATE.json') for r in rd(p)};O134.mkdir(exist_ok=False);method=[];paired=[];scenes=[];rows=[];compare=[(k,c) for k in ['partial_pair_F','pair_only_F','partial_wrongpair_F','tasklinked_pair_F','tasklinked_wrongpair_F','tasklinked_direct_F','tasklinked_partial3_direct_F','pair_direct_F','partial3_pair_direct_F','wrongpair_direct_F'] for c in ['zero_F','partial3_F','pair_direct_F','partial3_pair_direct_F','partial_wrongpair_F','tasklinked_direct_F','tasklinked_partial3_direct_F','tasklinked_wrongpair_F'] if k!=c]
 for name in mat['seals']:
  src,_,tgt=name.split('_');D=B134/'formal'/name;rr=[]
  for e in [e for e in reg if e['source']==src and e['target']==tgt]:
   sid=e['row']['id'];a=np.load(D/'samples_PRIVATE'/(sid+'.npz'));ps=dict(zip(a['arm_names'],a['scores']));pol=np.load(B134/'policy_PRIVATE'/name/(sid+'.npz'));ps.update(dict(zip(pol['arm_names'],pol['scores'])));q={k:int(p.argmax()) for k,p in ps.items()};ev=np.load(R134/e['evidence_path']);ass=ev['assignment'];groups=[np.flatnonzero(ass==g) for g in range(int(ass.max()+1))];y=np.load(R134/'research/round131/scores_attempt001'/(name+'_'+sid+'_DIAGNOSTIC_IOUS_PRIVATE.npz'))['ious'];ious={k:float(y[j]) for k,j in q.items()};old=oldrows[(src,tgt,sid)]
   for k,o in [('zero_F','zero_F'),('partial3_F','partial3_F'),('partial_direct_F','direct_F')]:assert abs(ious[k]-old['IoUs'][o])<1e-14
   representative=np.array([g[int(ps['zero_F'][g].argmax())] for g in groups]);quality=y[representative];w,l=ev['win'],ev['lose'];pairdiff=quality[w]-quality[l];q0=q['zero_F'];g0=ass[q0];top=(l==g0);good=y>.25;diag={}
   for k in ps:
    qa=q[k];gm=np.array([np.log(max(float(ps[k][g].max()),1e-300)) for g in groups]);diag[k]={'query_changed_vs_zero':bool(qa!=q0),'group_changed_vs_zero':bool(ass[qa]!=g0),'selected_in_S':bool(ev['S'][ass[qa]]),'selected_in_C':bool(ev['C'][ass[qa]]),'ordinal_loss_on_correct_pairs':float(np.maximum(ev['margin']-gm[w]+gm[l],0).mean()) if len(w) else 0.,'satisfied_correct_pairs':int((gm[w]-gm[l]>=ev['margin']-1e-8).sum())}
   rr.append({k:e[k] for k in ['source','target','frame_PRIVATE','content_group_PRIVATE','sampling_weight']}|{'id_PRIVATE':sid,'scene_PRIVATE':e['row']['scene'],'old_partial_active':e['active'],'pair_active':e['pair_active'],'pair_count':len(w),'old_top_challenged':bool(top.any()),'old_top_in_C':bool(ev['C'][g0]),'IoUs':ious,'diagnostics':diag,'pair_quality_better':int((pairdiff>1e-12).sum()),'pair_quality_worse':int((pairdiff< -1e-12).sum()),'pair_quality_tied':int((abs(pairdiff)<=1e-12).sum()),'top_pair_quality_better':int((pairdiff[top]>1e-12).sum()),'top_pair_quality_worse':int((pairdiff[top]< -1e-12).sum()),'recoverable_zero_error':bool(good.any() and y[q0]<=.25),'C_has_good':bool((good&ev['C'][ass]).any()),'zero_gt_Acc25':bool(y[q0]>.25),'zero_IoU_oracle':bool(y[q0]>=y.max()-1e-8)})
  dump(O134/(name+'_ALL_ROWS_PRIVATE.json'),rr);rows+=rr
  for weighted in [False,True]:
   aggregation='design_weighted' if weighted else 'raw'
   for k in mat['arms']:
    method.append({'source':src,'target':tgt,'arm':k,'aggregation':aggregation,**metrics(rr,k,weighted)})
    for sc in sorted({r['scene_PRIVATE'] for r in rr}):scenes.append({'source':src,'target':tgt,'scene_PRIVATE':sc,'arm':k,'aggregation':aggregation,**metrics([r for r in rr if r['scene_PRIVATE']==sc],k,weighted)})
   for k,c in compare:
    delta=np.array([r['IoUs'][k]-r['IoUs'][c] for r in rr]);weights=np.array([r['sampling_weight'] if weighted else 1. for r in rr]);paired.append({'source':src,'target':tgt,'method':k,'control':c,'aggregation':aggregation,'n':len(rr),'gain_pp':float(np.average(delta,weights=weights)*100),'better':int((delta>1e-12).sum()),'worse':int((delta< -1e-12).sum()),'same':int((abs(delta)<=1e-12).sum()),'largest_positive_contribution_pp':float(max(0.,(delta*weights).max())*100/weights.sum()/6),'largest_negative_contribution_pp':float(min(0.,(delta*weights).min())*100/weights.sum()/6)})
 agg=[]
 for aggregation in ['raw','design_weighted']:
  for group in ['all_six_equal','waymo','drone','quad']:
   for k in mat['arms']:
    pp=[r for r in method if r['aggregation']==aggregation and r['arm']==k and (group=='all_six_equal' or r['target']==group)];agg.append({'aggregation':aggregation,'group':group,'arm':k,**{m:float(np.mean([r[m] for r in pp])) for m in ['mIoU_percent','Acc25_percent','Acc50_percent']}})
 for name,obj in [('ALL_METHOD_ROWS',method),('ALL_PAIRED_ROWS',paired),('ALL_SCENE_ROWS_PRIVATE',scenes),('AGGREGATE_ROWS',agg)]:dump(O134/(name+'.json'),obj)
 decision={'primary':'partial_pair_F','target_GT_configuration_selection':False,'FINAL_access':False,'comparisons':{},'recipe_retained':None}
 for aggregation in ['raw','design_weighted']:
  v={r['arm']:r['mIoU_percent'] for r in agg if r['aggregation']==aggregation and r['group']=='all_six_equal'};decision['comparisons'][aggregation]={'mIoU_percent':v['partial_pair_F'],'gains_pp':{k:v['partial_pair_F']-v[k] for k in ['zero_F','partial3_F','pair_direct_F','partial3_pair_direct_F','partial_wrongpair_F']}}
 decision['recipe_retained']=all(g>0 for a in decision['comparisons'].values() for g in a['gains_pp'].values());decision['tasklinked_comparisons']={}
 for aggregation in ['raw','design_weighted']:
  v={r['arm']:r['mIoU_percent'] for r in agg if r['aggregation']==aggregation and r['group']=='all_six_equal'};decision['tasklinked_comparisons'][aggregation]={'mIoU_percent':v['tasklinked_pair_F'],'gains_pp':{k:v['tasklinked_pair_F']-v[k] for k in ['zero_F','partial3_F','tasklinked_direct_F','tasklinked_partial3_direct_F','tasklinked_wrongpair_F']}}
 decision['tasklinked_recipe_retained']=all(g>0 for a in decision['tasklinked_comparisons'].values() for g in a['gains_pp'].values());decision['robust_method_adopted']=False;decision['independent_validation']=False;decision['current_working_version']='R134_tasklinked_pair3_development_candidate' if decision['tasklinked_recipe_retained'] else ('R134_partial_pair3_development_candidate' if decision['recipe_retained'] else 'R132_originalB_LOG_partial3');dump(H134/'DECISION.json',decision);print(decision)
if __name__=='__main__':main()
