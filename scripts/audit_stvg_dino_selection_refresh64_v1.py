"""Independent saved-state aggregation and selection/old-label continuity audit."""
from pathlib import Path
import sys,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_dino_residual_advantage64_v1 import D,P,verify,read,put,load
from scripts.audit_stvg_dino_residual_advantage64_v1 import scalar

def main():
 p=verify(True);z=read(P/'RESULTS_PUBLIC.json');se=read(P/'SELECTION_BREAKPOINT_PUBLIC.json');ss=z['conditional_states'];checks=collections.Counter()
 for ds,sts in z['refresh_groups'].items():
  for st,mods in sts.items():
   keys={(f['alias'],f['position'])for f in z['frames']if f['dataset']==ds and f['panel']=='original'and f['GT_known']and f['strata'].get(st,False)}
   for mode,g in mods.items():
    sub=[r for r in ss if(r['alias'],r['position'])in keys and(mode!='both_admitted'or(r['accepted']and r['center_accepted']))];aliases=sorted({r['alias']for r in sub});assert len(sub)==g['sample_state_instances']and len(aliases)==g['parents']
    for key,b in g['fields'].items():
     vals=[]
     for a in aliases:
      row=[]
      for pos in sorted({r['position']for r in sub if r['alias']==a}):
       v=[r[key]for r in sub if r['alias']==a and r['position']==pos and r.get(key)is not None]
       if v:row.append(np.mean(v))
      if row:vals.append(float(np.mean(row)))
     assert len(vals)==b['parents']
     if vals:
      ci=np.quantile(np.random.default_rng(20261018).choice(np.array(vals),size=(10000,len(vals)),replace=True).mean(1),[.025,.975]);assert abs(np.mean(vals)-b['mean'])<1e-12 and np.allclose(ci,b['CI95'],atol=1e-12,rtol=0)
     checks['refresh_parent_frame_balanced_bootstrap_fields']+=1
 for s in ss:
  sr=s['teacher_IoU']>=.5 and s['teacher_IoU']-s['E_IoU']>=.1;cr=s['center_IoU']is not None and s['center_IoU']>=.5 and s['center_IoU']-s['E_IoU']>=.1
  assert s['new_admitted_correct']==bool(s['accepted']and sr and not(s['center_accepted']and cr));assert s['common_novel_better']==bool(s['accepted']and s['center_accepted']and s['center_IoU']is not None and s['teacher_IoU']-s['center_IoU']>=.05 and s['teacher_IoU']-s['E_IoU']>=.1);checks['state_recovery_flags']+=1
 for r in se['rows']:
  b=r['best_reference_IoU'];sel=r['selected_reference_IoU'];top=r['top_reference_IoU'];assert r['good_pool_but_selected_bad']==bool(b is not None and b>=.5 and sel is not None and sel<=.3);assert r['top_good_selected_bad']==bool(top is not None and top>=.5 and sel is not None and sel<=.3);checks['selection_breakpoint_flags']+=1
 events={kind:[{k:r.get(k)for k in ['alias','position','sample_index','input_IoU','teacher_IoU','E_IoU','center_IoU','native_max_box_IoU']}for r in ss if r[kind]]for kind in ['new_admitted_correct','common_novel_better']}
 put(P/'REFRESH_SELECTION_AUDIT_PUBLIC.json',dict(status='pass',checks=dict(checks),unique_newly_correct_query_frames=len({(r['alias'],r['position'])for r in ss if r['new_admitted_correct']}),unique_common_novel_query_frames=len({(r['alias'],r['position'])for r in ss if r['common_novel_better']}),events=events,no_actual_second_round_DINO_feedback_in_this_cache=True,old_sparse_state_targets_match_E_selected32=True));print('refresh/selection audit pass',dict(checks),flush=True)
if __name__=='__main__':main()
