"""Independent DEV selection and all sealed chosen-box task scoring; CPU only."""
import sys,json,time,pickle,numpy as np,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'crossplatform/round32'),str(ROOT/'research/round5/audit')]
from common32 import read,dump,sha,verify,load_npz,rows,feature
from independent_core_v2 import candidate_ious,transform_gt
from audit32_independent import qr,ell
OUT=ROOT/'research/round32/independent_audit';EXP=ROOT/'research/round32/experiment';BASE=ROOT/'crossplatform/round32';ARMS=['scope_TTA','scope_memory','global_mean_TTA','global_norm_TTA','ungated_memory'];KEYS=['noun_interaction','full_literal_caption_interaction']
def close(a,b,tol=1e-10):assert abs(float(a)-float(b))<tol,(a,b)
def main():
 start=time.perf_counter();cache=pickle.load(open(OUT/'_readout_cache_PRIVATE.pkl','rb'));computed=cache['computed'];fb=read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json');allrows=[dict(r,split=s) for s in ['A208','DEV64','R64','S64','T128'] for r in rows(s)];lookup={r['id']:r for r in allrows};byalias={r['alias']:r for r in allrows};zmap={r['id']:feature(r) for r in allrows};counts=collections.Counter();feasraw=read(BASE/'feasibilityA_attempt001/PAIR_EXECUTION.json');feasidx={(x['support'],x['key'],x['k'],x['arm'],x['alias'],x['pair_alias']):x for x in feasraw};feas=[];feaswarnings=[]
 for support in ['row','endpoint']:
  for key in KEYS:
   for k in [1,4]:
    for arm in ARMS:
     for f in fb:
      r=lookup[f['id_PRIVATE']];s=computed[r['id']]['outputs'][support,key,k][arm];q=qr(r,s,zmap[r['id']]);pair=bool(s[f['positive_group_PRIVATE']].max()>s[f['negative_group_PRIVATE']].max());final=q in f['positive_group_PRIVATE'];ref=feasidx[support,key,k,arm,r['alias'],f['pair_alias']];row={'support':support,'key':key,'k':k,'arm':arm,'alias':r['alias'],'pair_alias':f['pair_alias'],'feedback':f['label'],'pair_satisfied':pair,'final_positive':final}
      if pair!=ref['pair_satisfied'] or final!=ref['final_positive']:feaswarnings.append(row)
      feas.append(row);counts['A_feasibility_pair_decisions_recomputed']+=1
 # Independent geometry score function bound to original metadata and frozen boxes.
 idx={x['id']:x for x in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};containers={}
 def score(r,qs):
  i=idx[r['id']];path=i['meta_path']
  if path not in containers:assert sha(path)==i['metadata_container_sha256'];containers[path]=read(path)
  gt=transform_gt(containers[path]['ground_info'][i['object_index']]['bbox_3d'],r['platform'],i.get('waymo_view_suffix'));b=zmap[r['id']]['boxes'][np.array(qs)].copy();b[:,3:]=np.maximum(b[:,3:],1e-6);return candidate_ious(b,gt)[0]
 devraw=read(EXP/'DEV_FAMILY_ROWS_PRIVATE.json');devindex={(x['support'],x['key'],x['k'],x['arm'],x['alias']):x for x in devraw};dev=[];devquerywarnings=[]
 for support in ['row','endpoint']:
  for key in KEYS:
   for k in [1,4]:
    for r in rows('DEV64'):
     qs=[qr(r,computed[r['id']]['outputs'][support,key,k][a],zmap[r['id']]) for a in ARMS];vv=score(r,qs)
     for arm,q,v in zip(ARMS,qs,vv):
      ref=devindex[support,key,k,arm,r['alias']]
      if q!=ref['q_PRIVATE']:devquerywarnings.append({'alias':r['alias'],'support':support,'key':key,'k':k,'arm':arm,'CPU_q_PRIVATE':q,'saved_q_PRIVATE':ref['q_PRIVATE'],'CPU_iou':float(v),'saved_iou':ref['iou']})
      dev.append({'support':support,'key':key,'k':k,'arm':arm,'alias':r['alias'],'platform':r['platform'],'iou':float(v),'query_PRIVATE':q});counts['DEV_family_task_values_recomputed']+=1
 table=[]
 for support in ['row','endpoint']:
  for key in KEYS:
   for k in [1,4]:
    for arm in ARMS:
     dd=[x for x in dev if x['support']==support and x['key']==key and x['k']==k and x['arm']==arm];pm={p:float(np.mean([x['iou'] for x in dd if x['platform']==p])*100) for p in ['drone','quad']};fr=[x for x in feas if x['support']==support and x['key']==key and x['k']==k and x['arm']==arm];B=[x for x in fr if x['feedback']=='B'];table.append({'support':support,'key':key,'k':k,'arm':arm,'platform_mIoU_percent':pm,'balanced_mIoU_percent':float(np.mean(list(pm.values()))),'pairs_satisfied':sum(x['pair_satisfied'] for x in fr),'B_pairs_satisfied':sum(x['pair_satisfied'] for x in B),'B_final_positive':sum(x['final_positive'] for x in B),'feasible':sum(x['pair_satisfied'] for x in B)>=5 and sum(x['final_positive'] for x in B)>=4})
 def best(a):return sorted(a,key=lambda x:(-x['balanced_mIoU_percent'],KEYS.index(x['key']),x['k'],ARMS.index(x['arm'])))[0]
 primary=best([x for x in table if x['support']=='endpoint' and x['arm']=='scope_TTA' and x['feasible']]);selected=read(EXP/'SELECTED_SCOPE_LOCK.json');selection={'primary':primary,'same_gate_memory':next(x for x in table if x['support']=='endpoint' and x['key']==primary['key'] and x['k']==primary['k'] and x['arm']=='scope_memory'),'best_DEV_memory_same_budget':best([x for x in table if x['support']=='endpoint' and x['arm']=='scope_memory']),'best_DEV_global_amount_control':best([x for x in table if x['support']=='endpoint' and x['arm'] in ['global_mean_TTA','global_norm_TTA']]),'best_DEV_row_support_ablation':best([x for x in table if x['support']=='row' and x['arm']=='scope_TTA' and x['feasible']]),'same_family_row_support_ablation':next(x for x in table if x['support']=='row' and x['key']==primary['key'] and x['k']==primary['k'] and x['arm']=='scope_TTA'),'best_DEV_ungated_memory':best([x for x in table if x['support']=='endpoint' and x['arm']=='ungated_memory'])};selectionwarnings=[]
 for name,row in selection.items():
  ref=selected[name]
  for field in ['support','key','k','arm','pairs_satisfied','B_pairs_satisfied','B_final_positive','feasible']:assert row[field]==ref[field],(name,field,row[field],ref[field])
  close(row['balanced_mIoU_percent'],ref['balanced_mIoU_percent'],1e-10)
 tablediff=[]
 originaltable={(x['support'],x['key'],x['k'],x['arm']):x for x in read(EXP/'DEV_FAMILY_TABLE.json')}
 for row in table:
  ref=originaltable[row['support'],row['key'],row['k'],row['arm']]
  if abs(row['balanced_mIoU_percent']-ref['balanced_mIoU_percent'])>1e-10:tablediff.append({'support':row['support'],'key':row['key'],'k':row['k'],'arm':row['arm'],'CPU_balanced_mIoU':row['balanced_mIoU_percent'],'saved_balanced_mIoU':ref['balanced_mIoU_percent']})
 # Independent exact task reconstruction of all main saved selected queries.
 records=read(EXP/'ALL528_TASK_ROWS_PRIVATE.json');assert len(records)==528;reconstructed=[]
 configs={'scoped_TTA':selected['primary'],'same_gate_QP_memory':selected['same_gate_memory'],'DEVbest_QP_memory':selected['best_DEV_memory_same_budget'],'global_mean_matched':dict(primary,arm='global_mean_TTA'),'global_norm_matched':dict(primary,arm='global_norm_TTA'),'DEVbest_global_amount':selected['best_DEV_global_amount_control'],'same_family_row_scope':selected['same_family_row_support_ablation'],'DEVbest_row_scope':selected['best_DEV_row_support_ablation'],'DEVbest_ungated_QP_memory':selected['best_DEV_ungated_memory']};selectedquerywarnings=[]
 for x in records:
  r=lookup[x['id_PRIVATE']];names=list(x['q']);qs=[x['q'][a] for a in names];vv=score(r,qs)
  for a,v in zip(names,vv):close(v,x['ious'][a],1e-12);counts['all_selected_frozen_box_task_values_recomputed']+=1
  if r['stage']=='T':
   p=load_npz(BASE/'selectedT128_attempt001/predictions_PRIVATE'/(r['id']+'.npz'));assert len(names)==13 and names==p['arm_names'].tolist();assert np.array_equal(np.array(qs),p['queries']) and np.array_equal(p['boxes'],zmap[r['id']]['boxes'][p['queries']]);counts['T128_all13_task_values_recomputed']+=13
  for a,spec in configs.items():
   expected=qr(r,computed[r['id']]['outputs'][spec['support'],spec['key'],spec['k']][spec['arm']],zmap[r['id']])
   if expected!=x['q'][a]:selectedquerywarnings.append({'alias':r['alias'],'arm':a,'CPU_q_PRIVATE':expected,'saved_q_PRIVATE':x['q'][a]})
  reconstructed.append(dict(x,ious=dict(zip(names,map(float,vv)))))
 # Whole-task summary independently reduced, including all negative comparisons.
 summary=read(EXP/'TASK_SUMMARY.json');negative=[]
 for split in ['A208','DEV64','R64','S64','T128']:
  for pl in ['drone','quad']:
   rs=[x for x in reconstructed if x['split']==split and x['platform']==pl]
   for arm in rs[0]['q']:
    v=np.array([x['ious'][arm] for x in rs]);d=v-np.array([x['ious']['source'] for x in rs]);ref=summary[split][pl][arm];assert ref['n']==len(rs);close(ref['mIoU_percent'],v.mean()*100);close(ref['Acc25_percent'],(v>.25).mean()*100);close(ref['Acc50_percent'],(v>.5).mean()*100);close(ref['delta_vs_source_pp'],d.mean()*100);assert ref['improved']==int((d>1e-10).sum()) and ref['worsened']==int((d< -1e-10).sum()) and ref['query_changes']==sum(x['q'][arm]!=x['q']['source'] for x in rs)
   for x in rs:
    for arm in x['q']:
     for ref in ['source','R30_ungated','same_gate_QP_memory','DEVbest_global_amount']:
      if x['ious'][arm]<x['ious'][ref]-1e-10:negative.append((x['alias'],split,pl,arm,ref,100*(x['ious'][arm]-x['ious'][ref])))
 actual=[(x['alias'],x['split'],x['platform'],x['arm'],x['reference'],x['delta_pp']) for x in read(EXP/'ALL_NEGATIVES.json')];assert sorted(actual)==sorted(negative)
 # Exactly identify the single-effect-cluster issue, not a valid populationCI.
 T=[x for x in reconstructed if x['split']=='T128'];clusters={p:[[r for r in T if r['platform']==p and r['scene_PRIVATE']==s] for s in sorted({r['scene_PRIVATE'] for r in T if r['platform']==p})] for p in ['drone','quad']};cr=read(EXP/'T128_SCENE_COMPARISONS.json');uncert=[]
 for ref in ['source','R30_ungated','same_gate_QP_memory','DEVbest_global_amount','global_norm_matched','R30_allA_zeroU_memory']:
  effect=sum(any(abs(r['ious']['scoped_TTA']-r['ious'][ref])>1e-12 for r in cc) for groups in clusters.values() for cc in groups);val=float(np.mean([np.mean([r['ious']['scoped_TTA']-r['ious'][ref] for r in T if r['platform']==p]) for p in ['drone','quad']])*100);o=cr['scoped_TTA_vs_'+ref];close(o['balanced_gain_pp'],val,1e-10);assert o.get('active_effect_clusters')==effect
  if effect<2:assert o.get('descriptive_scene_bootstrap95_pp') is None
  uncert.append({'reference':ref,'balanced_gain_pp':val,'active_effect_clusters':effect,'full_population_CI_not_estimable_with_one_Drone_scene':True,'degenerate_conditional_quantile_not_significance':effect<2})
 assert len(clusters['drone'])==1
 # Fullmodelclosure receipt is evidence of GPU execution, separately from thisCPUreplay.
 verify(BASE/'closureT128_attempt001');closure=read(BASE/'closureT128_attempt001/CLOSURE_AUDIT.json');assert sum(x['full_model_forwards'] for x in closure)==256 and sum(x['all_readout_argmax_exact'] for x in closure)==1664;assert all(x['box_maxerror']==0 and x['nonmatching_weights_unchanged'] and x['source_restored_bitexact'] for x in closure)
 assert sha(EXP/'SELECTED_SCOPE_LOCK.json')==read(BASE/'selectedT128_attempt001/SEALED.json')['selected_scope_sha256'];assert read(EXP/'SELECTED_SCOPE_LOCK.json')['selector_sha256']==sha(BASE/'select32.py')
 dump(OUT/'DEV_FAMILY_TABLE_INDEPENDENT.json',table);dump(OUT/'SELECTED_SCOPE_INDEPENDENT.json',selection);dump(OUT/'DEV_CPU_QUERY_WARNINGS_PRIVATE.json',devquerywarnings);dump(OUT/'FEASIBILITY_CPU_WARNINGS.json',feaswarnings);dump(OUT/'SELECTED_NEW_READOUT_CPU_WARNINGS_PRIVATE.json',selectedquerywarnings);dump(OUT/'SCENE_UNCERTAINTY_AUDIT.json',uncert)
 result={'state':'PRIMARY_AND_SELECTION_TASK_AUDIT_PASS_WITH_REPLAY_WARNINGS_IF_LISTED','counts':dict(counts),'primary_endpoint_noun_k1_reselected_from_DEV_only':True,'original_selection_rules_reduced_independently':True,'DEV_table_score_differences':tablediff,'DEV_query_warning_rows':len(devquerywarnings),'A_feasibility_warning_rows':len(feaswarnings),'selected_new_readout_warning_rows':len(selectedquerywarnings),'T128_selected13arm_task_scores_exact':True,'all_528_task_records_and_negative_comparisons_reconstructed':True,'negative_comparison_count':len(negative),'no_R_S_T_label_selection':True,'T128_only_one_Drone_scene_population_CI_unavailable':True,'GPUclosure_receipt_256forwards1664argmax_checked':True,'no_GPU_or_training_run_by_auditor':True,'seconds':time.perf_counter()-start};dump(OUT/'SELECTION_TASK_AUDIT_RESULT.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
