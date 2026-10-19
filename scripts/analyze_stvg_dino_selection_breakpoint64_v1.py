"""Postseal selection-breakpoint localization, no new prediction or adaptation."""
from pathlib import Path
import sys,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_dino_residual_advantage64_v1 import D,P,verify,read,put,load,sha
from vg_tta.stvg_dino_residual_advantage64_v1 import ious
import numpy as np

def main():
 from scripts.score_stvg_opd_p1_v1 import truths
 p=verify(True);bar=read(D/'CACHE_BARRIER.json');assert bar['count']==64;res=read(P/'RESULTS_PUBLIC.json');oldindex={(f['alias'],f['position']):f for f in res['frames']};gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};rows=[];checks=0
 for r in p['records']:
  x=load(D/'extracted002'/(r['alias']+'.pt'));refpos=[];refboxes=[]
  for f in x['frames']:
   j=f['E_selected'];pos=f['position'];prior=oldindex[r['alias'],pos]
   if j>=0:refpos.append(pos);refboxes.append(f['retained_boxes'][j])
   if not prior['GT_known']:continue
   tr=gt[r['dataset']][0][r['query_ordinal']][f['frame_id']];row=plans[r['dataset']][r['query_ordinal']];v=ious(f['retained_boxes'],tr,row,r['dataset']);selected=float(v[j])if j>=0 else None;best=int(v.argmax())if len(v)else None;top=int(f['retained_scores'].argmax())if len(v)else None;Ebad=prior['E_IoU']<=.3
   rows.append(dict(alias=r['alias'],dataset=r['dataset'],panel=r['panel'],position=pos,probe_available=f['probe_available'],frame_admitted=f['frame_admitted'],strata=prior['strata'],E_IoU=prior['E_IoU'],selected_reference_IoU=selected,top_reference_IoU=float(v[top])if top is not None else None,best_reference_IoU=float(v[best])if best is not None else None,selected_reference_bad=bool(selected is not None and selected<=.3),selected_reference_good_but_E_bad=bool(selected is not None and selected>=.5 and Ebad),good_pool_but_selected_bad=bool(best is not None and v[best]>=.5 and selected is not None and selected<=.3),good_pool_but_no_selected_reference=bool(best is not None and v[best]>=.5 and selected is None),top_good_selected_bad=bool(top is not None and v[top]>=.5 and selected is not None and selected<=.3),best_score=float(f['retained_scores'][best])if best is not None else None,selected_score=float(f['retained_scores'][j])if j>=0 else None,highest_score=float(f['retained_scores'][top])if top is not None else None))
  if r['panel']=='original':
   z=load(ROOT/f"artifacts/stvg_sparse_state_route32_v1/predictions/{r['alias']}.pt");assert z['positions']==refpos and np.array_equal(np.asarray(refboxes).reshape(-1,4),z['targets']);checks+=1
 groups={}
 for ds in ['hc2','vidstg']:
  groups[ds]={}
  for st in ['all','bad','far','scale','instance_mismatch']:
   rr=[r for r in rows if r['dataset']==ds and r['strata'].get(st,False)];groups[ds][st]=dict(known_frames=len(rr),parents=len({r['alias']for r in rr}),selected_known=sum(r['selected_reference_IoU']is not None for r in rr),counts={k:sum(r[k]for r in rr)for k in ['selected_reference_bad','selected_reference_good_but_E_bad','good_pool_but_selected_bad','good_pool_but_no_selected_reference','top_good_selected_bad']},example_aliases={k:sorted({r['alias']for r in rr if r[k]})for k in ['good_pool_but_selected_bad','top_good_selected_bad']})
 put(P/'SELECTION_BREAKPOINT_PUBLIC.json',dict(status='complete',posthoc_mechanism_decomposition=True,no_new_outputs_or_methods=True,groups=groups,rows=rows,old_sparse_state_label_replay_exact_queries=checks,old_sparse_state_teacher_is_E_selected_not_GT_or_native_top1=True,does_not_prove_all_old_failures_caused_by_wrong_labels=True));print('selection breakpoint',groups,'label checks',checks,flush=True)
if __name__=='__main__':main()
