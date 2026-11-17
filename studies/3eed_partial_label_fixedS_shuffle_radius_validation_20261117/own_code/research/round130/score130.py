"""R130 post-seal scoring; original-rho primary fixed, no endpoint selection."""
from pathlib import Path
import json,sys,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round130';B=R/'crossplatform/round130';O=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text())
sys.path.insert(0,str(R/'research/round6/source_validation_scoring'));from batch_yaw_iou import many_iou
sys.path.insert(0,str(R/'tools'));from obb_iou import oriented_iou
REF={'original_source': 'original_source', 'prepared_A': 'prepared_A', 'legacy_ABC_context': 'legacy_ABC_context', 'zero_F': 'zero_F', 'zero_C': 'zero_C', 'direct_partial_F': 'direct_partial_F', 'direct_partial_C': 'direct_partial_C', 'partial_F': 'partial_F', 'partial_C': 'partial_C', 'R129_guard_F': 'partial_guard_F', 'R129_guard_C': 'partial_guard_C'}
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def metrics(rr,k):
 y=np.array([r['arms'][k]['IoU'] for r in rr]);return {'n':len(rr),'mIoU_percent':float(y.mean()*100),'Acc25_percent':float((y>.25).mean()*100),'Acc50_percent':float((y>.5).mean()*100)}
def main():
 assert rd(H/'NATIVE_AUDIT.json')['n']==2304;mat=rd(B/'MATRIX_SEALED.json');O.mkdir(exist_ok=False);reg=rd(R/'research/round122/CACHE_REGISTRY_PRIVATE.json');methods=[];paired=[];scenes=[];allrows=[];maxerr=0;newreadouts=0
 for combo in mat['seals']:
  co,name=combo.split('/');src,_,tgt=name.split('_');D=B/'formal'/co/name;rr=[e for e in reg if e['cohort']==co and e['source']==src and e['target']==tgt];prior={r['id_PRIVATE']:r for r in rd(R/f'research/round129/scores_attempt001/{co}_{name}_ALL_ROWS_PRIVATE.json')};gtrows=rd(R/f'research/round96/experiment/OFFICIAL_{tgt.upper()}_VAL_ORDER_PRIVATE.json');rows=[]
  for e in rr:
   sid=e['row']['id'];p=np.load(D/'samples_PRIVATE'/(sid+'.npz'));rec=rd(D/'records_PRIVATE'/(sid+'.json'));ar={};diag={};prob=dict(zip(p['arm_names'],p['scores']));q=dict(zip(p['arm_names'],p['queries']));isalias=e['protected_alias_fallback'];scope=prior[sid]['scope_metadata']
   if not isalias:
    boxes=np.c_[p['source_boxes'],np.zeros(256)].astype(float);boxes[:,3:6]=np.maximum(boxes[:,3:6],1e-6);gt=np.array(gtrows[e['row']['anno_index']]['source_GT7_float32']);ys=many_iou(boxes,gt);c,s=np.cos(gt[6]),np.sin(gt[6]);rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    for j in set(q.values()):error=abs(oriented_iou(boxes[j,:3],boxes[j,3:6],np.eye(3),gt[:3],gt[3:6],rotation)-ys[j]);maxerr=max(maxerr,error);assert error<1e-6
    sp=np.load(R/'crossplatform/round129/scope_PRIVATE'/co/name/(sid+'.npz'));assignment=sp['assignment'];S=sp['S'][assignment];C=sp['C'][assignment];best=np.isclose(ys,ys.max(),rtol=0,atol=1e-8);has25=bool((ys>.25).any());Sgood=bool((ys[S]>.25).any()) if S.any() else False;Cgood=bool((ys[C]>.25).any()) if C.any() else False;diag={'oracle_IoU':float(ys.max()),'oracle_Acc25':has25,'S_has_Acc25_candidate':Sgood,'C_has_Acc25_candidate':Cgood,'S_contains_global_best_IoU':bool((S&best).any()),'C_contains_global_best_IoU':bool((C&best).any()),'C_retains_best_in_S':bool(C.any() and S.any() and ys[C].max()>=ys[S].max()-1e-8),'partial_eligible_recoverable_source_error':bool(scope['partial_eligible'] and has25 and ys[q['zero_F']]<=.25),'q0_in_S':bool(S[q['zero_F']]),'q0_in_C':bool(C[q['zero_F']]),'teacher_kept_zero_query':prior[sid]['diagnostics']['teacher_kept_zero_query'],'teacher_worse_than_zero':prior[sid]['diagnostics']['teacher_worse_than_zero'],'geometry_groups_not_GT_identity':True}
   for k,pp in prob.items():
    k=str(k);j=int(pp.argmax())
    if k in REF:ar[k]=dict(prior[sid]['arms'][REF[k]]);assert ar[k]['query_PRIVATE']==j
    elif isalias:ar[k]=dict(prior[sid]['arms']['zero_'+k.rsplit('_',1)[1]]);assert ar[k]['query_PRIVATE']==j
    else:ar[k]={'IoU':float(ys[j]),'query_PRIVATE':j};newreadouts+=1
   if not isalias:
    for k in ar:assert abs(ar[k]['IoU']-ys[ar[k]['query_PRIVATE']])<1e-6
    for k in ['proper_half_F','partial_F','proper_double_F','wrong_half_F','wrong_base_F','wrong_double_F','direct_partial_F','direct_wrong_F']:
     j=ar[k]['query_PRIVATE'];diag[k+'_same_geometry_group_as_zero']=bool(assignment[j]==assignment[q['zero_F']])
   rows.append({'cohort':co,'source':src,'target':tgt,'id_PRIVATE':sid,'scene_PRIVATE':e['row']['scene'],'protected_alias_fallback':isalias,'scope_eligible':scope['scope_eligible'],'partial_eligible':scope['partial_eligible'],'B_available':rec.get('B_available',scope.get('B_available')),'C_available':rec.get('C_available'),'seconds':rec['seconds'],'effective_shuffle':rec['control']['effective_shuffle'],'control_metadata':rec['control'],'scope_metadata':scope,'diagnostics':diag,'arms':ar})
  dump(O/(co+'_'+name+'_ALL_ROWS_PRIVATE.json'),rows);allrows+=rows
  for k in mat['arms']:
   methods.append({'cohort':co,'source':src,'target':tgt,'arm':k,**metrics(rows,k)})
   for sc in sorted({r['scene_PRIVATE'] for r in rows}):scenes.append({'cohort':co,'source':src,'target':tgt,'scene_PRIVATE':sc,'arm':k,**metrics([r for r in rows if r['scene_PRIVATE']==sc],k)})
  pairs=[('proper_half','wrong_half'),('partial','wrong_base'),('proper_double','wrong_double')]
  comps=[(k+'_'+v,c) for v in ['F','C'] for k,w in pairs for c in ['zero_'+v,'direct_partial_'+v,w+'_'+v,'direct_wrong_'+v]]+[(w+'_'+v,c) for v in ['F','C'] for k,w in pairs for c in ['zero_'+v,'direct_wrong_'+v]]
  for k,c in comps:
   dd=np.array([r['arms'][k]['IoU']-r['arms'][c]['IoU'] for r in rows]);paired.append({'cohort':co,'source':src,'target':tgt,'method':k,'control':c,'gain_pp':float(dd.mean()*100),'better':int((dd>1e-12).sum()),'worse':int((dd< -1e-12).sum()),'same':int((abs(dd)<=1e-12).sum()),'largest_positive_contribution_pp':float(dd.max()*100/len(dd)),'largest_negative_contribution_pp':float(dd.min()*100/len(dd))})
 agg=[{'cohort':co,'group':g,'arm':k,**{m:float(np.mean([r[m] for r in methods if r['cohort']==co and r['arm']==k and (g=='all_six_equal' or r['target']==g)])) for m in ['mIoU_percent','Acc25_percent','Acc50_percent']}} for co in ['DEV','EXP'] for g in ['all_six_equal','waymo','drone','quad'] for k in mat['arms']]
 post=[]
 for co in ['DEV','EXP']:
  for src in ['waymo','drone','quad']:
   for tgt in ['waymo','drone','quad']:
    if src==tgt:continue
    name=src+'_to_'+tgt;full=rd(R/f'research/round119/analysis_attempt001/{name}_ALL_ROWS_PRIVATE.json');rr=[r for r in allrows if r['cohort']==co and r['source']==src and r['target']==tgt];scs={r['scene_PRIVATE'] for r in rr};support=[r for r in full if r['scene_PRIVATE'] in scs]
    for k in mat['arms']:
     vals={m:0. for m in ['mIoU_percent','Acc25_percent','Acc50_percent']}
     for sc in scs:
      weight=sum(r['scene_PRIVATE']==sc for r in support)/len(support);mt=metrics([r for r in rr if r['scene_PRIVATE']==sc],k)
      for m in vals:vals[m]+=weight*mt[m]
     post.append({'cohort':co,'source':src,'target':tgt,'arm':k,'covered_full_mass':len(support)/len(full),**vals})
 postagg=[{'cohort':co,'group':g,'arm':k,'min_covered_full_mass':min(r['covered_full_mass'] for r in post if r['cohort']==co and r['arm']==k and (g=='all_six_equal' or r['target']==g)),**{m:float(np.mean([r[m] for r in post if r['cohort']==co and r['arm']==k and (g=='all_six_equal' or r['target']==g)])) for m in ['mIoU_percent','Acc25_percent','Acc50_percent']}} for co in ['DEV','EXP'] for g in ['all_six_equal','waymo','drone','quad'] for k in mat['arms']]
 for name,x in [('ALL_METHOD_ROWS',methods),('ALL_SCENE_ROWS_PRIVATE',scenes),('ALL_PAIRED_ROWS',paired),('AGGREGATE_ROWS',agg),('POSTSTRATIFIED_METHOD_ROWS',post),('POSTSTRATIFIED_AGGREGATE_ROWS',postagg)]:dump(O/(name+'.json'),x)
 decision={'primary':'partial_F','cohorts':{},'C_is_descriptive_only':True,'posthoc_threshold_or_primary_switch':False,'FINAL_access':False}
 for co in ['DEV','EXP']:
  a={r['arm']:r['mIoU_percent'] for r in agg if r['cohort']==co and r['group']=='all_six_equal'};p={r['arm']:r['mIoU_percent'] for r in postagg if r['cohort']==co and r['group']=='all_six_equal'};controls=['zero_F','direct_partial_F','wrong_base_F'];decision['cohorts'][co]={'mIoU_percent':a['partial_F'],'gains_pp':{c:a['partial_F']-a[c] for c in controls},'poststratified_gains_pp':{c:p['partial_F']-p[c] for c in controls},'passes_prelocked_controls':all(a['partial_F']>a[c] and p['partial_F']>p[c] for c in controls)}
 decision['passes_development_gate']=all(r['passes_prelocked_controls'] for r in decision['cohorts'].values());decision['robust_method_adopted']=False;dump(H/'DECISION.json',decision);dump(H/'GEOMETRY_AUDIT.json',{'n':2304,'new_readouts':newreadouts,'independent_max_IoU_error':maxerr,'protected_alias_no_GT_indexing':154});print(json.dumps(decision,indent=2),flush=True)
if __name__=='__main__':main()
