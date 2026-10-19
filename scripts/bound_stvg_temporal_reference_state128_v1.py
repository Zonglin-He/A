"""Recompute conditional-reference readout ceilings from ALREADY saved GT scalars.
No annotation access, source inference, new prediction or new task score.
"""
from pathlib import Path
import sys,json
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read
D=R/'artifacts/stvg_temporal_reference_state128_v1'


def main():
 q=read(D/'QUALIFICATION_PUBLIC.json');r=read(R/'results/stvg_error_state/2026-10-20/temporal_joint_risk128/RESULTS_PUBLIC.json');rr={x['alias']:x for x in r['rows']};affected=[a for a in q['rows']if any(k!=a['native']for k in a['all_candidate_unique_queries'])];saved=read(D/'ALL_SAVED_READOUT_CAPACITY_BOUNDS_PUBLIC.json');assert len(affected)==saved['affected_parents']==6;bounds={}
 for prefix in ['N','H','A']:
  for kind in ['Risk','Target']:
   arm=prefix+'_'+kind;extra=0
   for a in affected:
    z=rr[a['alias']];oracle=sum(z['values'][f'Live{s}_N_Time']['v']+z['diagnostics'][f'Live{s}']['GT_candidate_capacity_pp']/100 for s in [41,42,43])/3
    extra+=100*(oracle-z['values']['LiveMean_'+arm]['v'])/128
   old=r['paired'][arm+'_minus_N_Time']['v']['mean'];out=dict(old_delta=old,max_new_conditional_query_extra_pp=extra,max_total_pp=old+extra);assert all(abs(v-saved['bounds'][arm][k])<1e-12 for k,v in out.items());bounds[arm]=out
 print(json.dumps(dict(status='pass',bounds=bounds,GT_dependent_offline_capacity=True,no_raw_annotations_or_new_task_scoring=True,scope=saved['scope']),indent=2),flush=True)

if __name__=='__main__':main()
