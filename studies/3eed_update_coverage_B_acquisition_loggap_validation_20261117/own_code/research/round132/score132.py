"""Post-seal cached A256 IoUs only; no new raw task labels/geometry/FINAL reads."""
from common132 import *
import numpy as np
O132=H132/'scores_attempt001'
def metrics(rr,arm,weighted):
 y=np.array([r['IoUs'][arm] for r in rr]);w=np.array([r['sampling_weight'] if weighted else 1. for r in rr]);return {'n':len(rr),'mIoU_percent':float(np.average(y,weights=w)*100),'Acc25_percent':float(np.average(y>.25,weights=w)*100),'Acc50_percent':float(np.average(y>.5,weights=w)*100),'weight_sum':float(w.sum())}
def main():
 assert rd(H132/'NATIVE_AUDIT.json')['n']==1798;mat=rd(B132/'MATRIX_SEALED.json');reg=rd(R132/'research/round131/INPUT_REGISTRY_PRIVATE.json');oldrows={(r['source'],r['target'],r['id_PRIVATE']):r for f in (R132/'research/round131/scores_attempt001').glob('*_ALL_ROWS_PRIVATE.json') for r in rd(f)};oldreasons={(r['source'],r['target'],r['id_PRIVATE']):r for r in rd(H132/'DIAGNOSIS_ROWS_PRIVATE.json')};O132.mkdir(exist_ok=False);method=[];paired=[];scenes=[];allrows=[]
 comparisons=[('partial_F',c) for c in ['zero_F','direct_partial_F','wrong_base_F']]+[(kind+'_partial',c) for kind in ['B3','LOG'] for c in [kind+'_zero',kind+'_direct',kind+'_wrong','partial_F']]+[('B3_zero','zero_F'),('B3_direct','direct_partial_F'),('LOG_direct','direct_partial_F')]
 for name in mat['seals']:
  src,_,tgt=name.split('_');rr=[];D=B132/'formal'/name
  for e in [e for e in reg if e['source']==src and e['target']==tgt]:
   sid=e['row']['id'];a=np.load(D/'samples_PRIVATE'/(sid+'.npz'));pp=dict(zip(a['arm_names'],a['scores']));q={k:int(v.argmax()) for k,v in pp.items()};rec=rd(D/'records_PRIVATE'/(sid+'.json'));old=oldrows[(src,tgt,sid)];y=np.load(R132/'research/round131/scores_attempt001'/(name+'_'+sid+'_DIAGNOSTIC_IOUS_PRIVATE.npz'))['ious'];values={k:float(y[j]) for k,j in q.items()}
   for k in ['zero_F','direct_partial_F','partial_F','wrong_base_F']:assert abs(values[k]-old['arms'][k]['IoU'])<1e-14
   row={k:e[k] for k in ['source','target','frame_PRIVATE','content_group_PRIVATE','sampling_weight']};row.update(id_PRIVATE=sid,scene_PRIVATE=e['row']['scene'],IoUs=values,old_skip_reason=oldreasons[(src,tgt,sid)]['reason'],old_skip_detail=oldreasons[(src,tgt,sid)]['detail'],seconds_new=rec['seconds']);diag={}
   oldroot=R132/'crossplatform/round131/formal'/name;oldrec=rd(oldroot/'records_PRIVATE'/(sid+'.json'))
   for kind in ['B3','LOG']:
    r=rec[kind];ref=r['reference_original'];f=np.load(oldroot/'features_PRIVATE'/(sid+'.npz')) if ref else np.load(D/'features_PRIVATE'/(sid+'_'+kind+'.npz'));ass=f['assignment'];S=f['S'];C=f['C'];W=f['wrong_C'];qs=S[ass];qc=C[ass];qw=W[ass];q0=q[kind+'_zero'];qp=q[kind+'_partial'];p0=pp[kind+'_zero'];trace=oldrec['traces']['partial'] if ref else r['traces']['partial'];active=oldrec['partial_eligible'] if ref else r['partial_eligible'];available=oldrec['B_available'] if ref else r['B_available'];good=y>.25;origq=q['zero_F'];C_size=int(C.sum());mass=float(p0[qc].sum()/p0[qs].sum()) if qs.any() else None;wrongmass=float(p0[qw].sum()/p0[qs].sum()) if qs.any() else None;logdelta=np.log(pp[kind+'_partial'])-np.log(p0);alts=np.flatnonzero(ass!=ass[q0]);j=int(alts[np.argmax(p0[alts])]) if len(alts) else int(np.argsort(p0)[-2]);m0=float(np.log(p0[q0])-np.log(p0[j]));m1=float(np.log(pp[kind+'_partial'][q0])-np.log(pp[kind+'_partial'][j]));effective=bool(active and not np.array_equal(C,W))
    if ref:parameter_norm=oldreasons[(src,tgt,sid)]['parameter_L2']
    else:
     with np.load(D/'states_PRIVATE'/(sid+'_'+kind+'.npz')) as state:parameter_norm=float(np.linalg.norm(state['partial']))
    diag[kind]={'parameter_L2':parameter_norm,'parameter_changed':parameter_norm>0,'reference_original':ref,'selected_B_rank':(r['selected_rank'] if kind=='B3' or available else None),'B_available':available,'active':active,'effective_wrong':effective,'S_groups':int(S.sum()),'C_groups':C_size,'S_has_Acc25':bool((good&qs).any()),'C_has_Acc25':bool((good&qc).any()),'old_zero_recoverable':bool(good.any() and y[origq]<=.25),'own_zero_recoverable':bool(good.any() and y[q0]<=.25),'all_good_removed_by_C_when_S_has_good':bool((good&qs).any() and not (good&qc).any()),'own_q0_in_S':bool(S[ass[q0]]),'own_q0_in_C':bool(C[ass[q0]]),'query_changed':q0!=qp,'geometry_group_changed':bool(ass[q0]!=ass[qp]),'iou_changed':bool(abs(y[q0]-y[qp])>1e-12),'IoU_gain':float(y[qp]-y[q0]),'accepted_steps':sum(t['accepted'] for t in trace),'shrunk_steps':sum(t['accepted'] and t['fraction']<1 for t in trace),'correct_retained_initial_mass':mass,'wrong_retained_initial_mass':wrongmass,'centered_log_score_change_L2':float(np.linalg.norm(logdelta-logdelta.mean())),'top_vs_other_group_margin_before':m0,'top_vs_other_group_margin_after':m1,'margin_change':m1-m0,'oracle_IoU':float(y.max()),'identity_not_certified_by_group_or_IoU':True}
   row['diagnostics']=diag;rr.append(row)
  dump(O132/(name+'_ALL_ROWS_PRIVATE.json'),rr);allrows+=rr
  for weighted in [False,True]:
   aggregation='design_weighted' if weighted else 'raw'
   for k in mat['arms']:
    method.append({'source':src,'target':tgt,'arm':k,'aggregation':aggregation,**metrics(rr,k,weighted)})
    for sc in sorted({r['scene_PRIVATE'] for r in rr}):scenes.append({'source':src,'target':tgt,'scene_PRIVATE':sc,'arm':k,'aggregation':aggregation,**metrics([r for r in rr if r['scene_PRIVATE']==sc],k,weighted)})
   for k,c in comparisons:
    dd=np.array([r['IoUs'][k]-r['IoUs'][c] for r in rr]);w=np.array([r['sampling_weight'] if weighted else 1. for r in rr]);paired.append({'source':src,'target':tgt,'method':k,'control':c,'aggregation':aggregation,'n':len(rr),'gain_pp':float(np.average(dd,weights=w)*100),'better':int((dd>1e-12).sum()),'worse':int((dd< -1e-12).sum()),'same':int((abs(dd)<=1e-12).sum()),'largest_positive_six_direction_contribution_pp':float(max(0,(dd*w).max())*100/w.sum()/6),'largest_negative_six_direction_contribution_pp':float(min(0,(dd*w).min())*100/w.sum()/6)})
 aggregate=[]
 for aggregation in ['raw','design_weighted']:
  for group in ['all_six_equal','waymo','drone','quad']:
   for arm in mat['arms']:
    rows=[r for r in method if r['aggregation']==aggregation and r['arm']==arm and (group=='all_six_equal' or r['target']==group)];aggregate.append({'aggregation':aggregation,'group':group,'arm':arm,**{k:float(np.mean([r[k] for r in rows])) for k in ['mIoU_percent','Acc25_percent','Acc50_percent']}})
 for name,x in [('ALL_METHOD_ROWS',method),('ALL_PAIRED_ROWS',paired),('ALL_SCENE_ROWS_PRIVATE',scenes),('AGGREGATE_ROWS',aggregate)]:dump(O132/(name+'.json'),x)
 decision={'no_combination':True,'cohort':1798,'source_training':False,'new_GT_geometry':False,'FINAL_access':False,'robust_method_adopted':False,'factors':{}}
 for kind in ['B3','LOG']:
  comparisons={}
  for aggregation in ['raw','design_weighted']:
   v={r['arm']:r['mIoU_percent'] for r in aggregate if r['aggregation']==aggregation and r['group']=='all_six_equal'};comparisons[aggregation]={'mIoU_percent':v[kind+'_partial'],'gains_pp':{c:v[kind+'_partial']-v[c] for c in [kind+'_zero',kind+'_direct',kind+'_wrong','partial_F']}}
  decision['factors'][kind]={'comparisons':comparisons,'retain_as_working_factor':all(v>0 for v in comparisons['design_weighted']['gains_pp'].values())}
 dump(H132/'DECISION.json',decision);print(json.dumps(decision,indent=2))
if __name__=='__main__':main()
