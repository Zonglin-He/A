"""Post-seal new-input scoring, metadata-only design weights fixed in advance."""
from pathlib import Path
import json,sys,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round131';B=R/'crossplatform/round131';O=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text());sys.path.insert(0,str(R/'research/round6/source_validation_scoring'));from batch_yaw_iou import many_iou
sys.path.insert(0,str(R/'tools'));from obb_iou import oriented_iou
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def metrics(rr,k,weighted=False):
 y=np.array([r['arms'][k]['IoU'] for r in rr]);w=np.array([r['sampling_weight'] if weighted else 1. for r in rr]);return {'n':len(rr),'mIoU_percent':float(np.average(y,weights=w)*100),'Acc25_percent':float(np.average(y>.25,weights=w)*100),'Acc50_percent':float(np.average(y>.5,weights=w)*100),'weight_sum':float(w.sum())}
def main():
 assert rd(H/'NATIVE_AUDIT.json')['n']==1798;O.mkdir(exist_ok=False);reg=rd(H/'INPUT_REGISTRY_PRIVATE.json');mat=rd(B/'MATRIX_SEALED.json');methods=[];paired=[];scene=[];rows=[];maxerr=0.;alias=0
 for name in mat['seals']:
  src,_,tgt=name.split('_');rr=[e for e in reg if e['source']==src and e['target']==tgt];D=B/'formal'/name;gtrows=rd(R/f'research/round96/experiment/OFFICIAL_{tgt.upper()}_VAL_ORDER_PRIVATE.json');old={r['id_PRIVATE']:r for r in rd(R/f'research/round119/analysis_attempt001/{name}_ALL_ROWS_PRIVATE.json')} if any(e['protected_alias_fallback'] for e in rr) else {};out=[]
  for e in rr:
   sid=e['row']['id'];a=np.load(D/'samples_PRIVATE'/(sid+'.npz'));rec=rd(D/'records_PRIVATE'/(sid+'.json'));ps=dict(zip(a['arm_names'],a['scores']));queries={k:int(p.argmax()) for k,p in ps.items()};arms={};diag={}
   if e['protected_alias_fallback']:
    alias+=1
    for k,q in queries.items():arms[k]={'IoU':old[sid]['values']['prepared_frozen'],'query_PRIVATE':q}
   else:
    boxes=np.c_[a['source_boxes'],np.zeros(256)].astype(float);boxes[:,3:6]=np.maximum(boxes[:,3:6],1e-6);gt=np.array(gtrows[e['row']['anno_index']]['source_GT7_float32']);ys=many_iou(boxes,gt);c,s=np.cos(gt[6]),np.sin(gt[6]);rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    for q in set(queries.values()):error=abs(oriented_iou(boxes[q,:3],boxes[q,3:6],np.eye(3),gt[:3],gt[3:6],rotation)-ys[q]);maxerr=max(maxerr,error);assert error<1e-6
    for k,q in queries.items():arms[str(k)]={'IoU':float(ys[q]),'query_PRIVATE':q}
    ff=np.load(D/'features_PRIVATE'/(sid+'.npz'));assignment=ff['assignment'];S=ff['S'][assignment];C=ff['C'][assignment];q0=queries['zero_F'];good=ys>.25;recoverable=bool(good.any() and ys[q0]<=.25);bestC=int(np.flatnonzero(C)[np.argmax(ys[C])]) if C.any() else None;T=ff['teacher_conditional'];qg=assignment[q0];qs_in=bool(ff['S'][qg]);qc_in=bool(ff['C'][qg]);diag={'oracle_IoU':float(ys.max()),'oracle_Acc25':bool(good.any()),'recoverable_zero_error_Acc25':recoverable,'S_has_Acc25':bool((S&good).any()),'C_has_Acc25':bool((C&good).any()),'q0_in_S':qs_in,'q0_in_C':qc_in,'S_has_better_IoU_than_zero':bool((S&(ys>ys[q0]+1e-8)).any()),'C_has_better_IoU_than_zero':bool((C&(ys>ys[q0]+1e-8)).any()),'q_partial_in_C':bool(C[queries['partial_F']]),'same_geometry_group_zero_partial':bool(assignment[q0]==assignment[queries['partial_F']]),'identity_not_certified_by_IoU':True}
    if bestC is not None:
     cg=assignment[bestC];diag.update(best_C_IoU=float(ys[bestC]),best_C_same_geometry_group_as_q0=bool(cg==qg),teacher_margin_best_C_minus_q0=float(T[cg]-T[qg]) if qc_in else None,best_C_rank_zero=int((ps['zero_F']>ps['zero_F'][bestC]).sum()+1),best_C_rank_partial=int((ps['partial_F']>ps['partial_F'][bestC]).sum()+1))
    if recoverable and (C&good).any() and qc_in:diag['within_C_teacher_already_prefers_better_Acc25_candidate']=bool(T[assignment[bestC]]>T[qg]+1e-12)
    np.savez_compressed(O/(name+'_'+sid+'_DIAGNOSTIC_IOUS_PRIVATE.npz'),ious=ys)
   out.append({'source':src,'target':tgt,'id_PRIVATE':sid,'frame_PRIVATE':e['frame_PRIVATE'],'content_group_PRIVATE':e['content_group_PRIVATE'],'scene_PRIVATE':e['row']['scene'],'sampling_weight':e['sampling_weight'],'protected_alias_fallback':e['protected_alias_fallback'],'B_available':rec.get('B_available',False),'scope_eligible':rec['scope_eligible'],'partial_eligible':rec['partial_eligible'],'effective_shuffle':rec['effective_shuffle'],'seconds':rec['seconds'],'diagnostics':diag,'arms':arms})
  dump(O/(name+'_ALL_ROWS_PRIVATE.json'),out);rows+=out
  for weighted in [False,True]:
   for k in mat['arms']:
    methods.append({'source':src,'target':tgt,'arm':k,'aggregation':'design_weighted' if weighted else 'raw',**metrics(out,k,weighted)})
    for sc in sorted({r['scene_PRIVATE'] for r in out}):scene.append({'source':src,'target':tgt,'scene_PRIVATE':sc,'arm':k,'aggregation':'design_weighted' if weighted else 'raw',**metrics([r for r in out if r['scene_PRIVATE']==sc],k,weighted)})
   for k,ct in [('partial_F','zero_F'),('partial_F','direct_partial_F'),('partial_F','wrong_base_F'),('direct_partial_F','zero_F'),('wrong_base_F','zero_F')]:
    dd=np.array([r['arms'][k]['IoU']-r['arms'][ct]['IoU'] for r in out]);w=np.array([r['sampling_weight'] if weighted else 1. for r in out]);paired.append({'source':src,'target':tgt,'method':k,'control':ct,'aggregation':'design_weighted' if weighted else 'raw','n':len(out),'gain_pp':float(np.average(dd,weights=w)*100),'better':int((dd>1e-12).sum()),'worse':int((dd< -1e-12).sum()),'same':int((abs(dd)<=1e-12).sum()),'largest_positive_six_direction_contribution_pp':float(max(0,(dd*w).max())*100/w.sum()/6),'largest_negative_six_direction_contribution_pp':float(min(0,(dd*w).min())*100/w.sum()/6)})
 aggregates=[]
 for weighted in ['raw','design_weighted']:
  for group in ['all_six_equal','waymo','drone','quad']:
   for k in mat['arms']:
    rr=[r for r in methods if r['aggregation']==weighted and r['arm']==k and (group=='all_six_equal' or r['target']==group)];aggregates.append({'aggregation':weighted,'group':group,'arm':k,**{m:float(np.mean([r[m] for r in rr])) for m in ['mIoU_percent','Acc25_percent','Acc50_percent']}})
 for name,x in [('ALL_METHOD_ROWS',methods),('ALL_PAIRED_ROWS',paired),('ALL_SCENE_ROWS_PRIVATE',scene),('AGGREGATE_ROWS',aggregates)]:dump(O/(name+'.json'),x)
 decision={'primary':'partial_F','primary_aggregation':'design_weighted_six_direction_macro','not_blind_or_new_scene':True,'no_parameter_or_output_selection':True,'FINAL_access':False,'comparisons':{}}
 for aggregation in ['raw','design_weighted']:
  a={r['arm']:r['mIoU_percent'] for r in aggregates if r['aggregation']==aggregation and r['group']=='all_six_equal'};decision['comparisons'][aggregation]={'mIoU_percent':a['partial_F'],'gains_pp':{c:a['partial_F']-a[c] for c in ['zero_F','direct_partial_F','wrong_base_F']}}
 decision['primary_positive_all_three_controls']=all(x>0 for x in decision['comparisons']['design_weighted']['gains_pp'].values());decision['robust_method_adopted']=False;dump(H/'DECISION.json',decision);dump(H/'GEOMETRY_AUDIT.json',{'n':len(rows),'protected_aliases_no_new_GT_indexing':alias,'independent_max_IoU_error':maxerr,'all_negative_and_unknown_inputs_retained':True});print(json.dumps(decision,indent=2))
if __name__=='__main__':main()
