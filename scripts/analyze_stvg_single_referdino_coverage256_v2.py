"""Meaningful parent coverage and signed contributions from saved results only."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_single_referdino_parameter_search256_common_v1 import P,read,put
from scripts.finalize_stvg_single_referdino_parameter_search256_v4 import result_sources

def main():
 selection=read(P/'SELECTION_PUBLIC.json');out={};lines=['# Parent coverage of parameter configurations','','Every denominator is all128 parents per target dataset. Changes are complete vIoU relative to the same initialization E_gamma+Strong. Any-positive uses1e−10pp numeric tolerance; meaningful improvement/decline is at least0.1pp in either direction. This fixed threshold reports meaningful coverage and is used by the declared offline selection policy; it is not an online gate or a searched parameter. All near-zero and zero-time-overlap parents remain.','', '| Dataset / setting | any improved / worse / tied | ≥0.1pp improved | ≤−0.1pp worse | positive / negative contribution pp | net pp | <−5pp |','|---|---:|---:|---:|---:|---:|---:|']
 for ds,sel in selection['datasets'].items():
  rows=[]
  for ns,phase,dp,pp,r in result_sources(ds):
   for arm,c in r['definitions'].items():
    ek='E_Strong'if ns=='main'else'E_G'+str(int(c['initialization_gamma']*100)).zfill(3);x=np.array([100*(v['values'][arm]['v']-v['values'][ek]['v'])for v in r['rows']]);rows.append(dict(namespace=ns,phase=phase,arm=arm,configuration=c,full_vIoU_percent=100*r['means'][arm]['v'],parents=128,any_improved=int((x>1e-10).sum()),any_worse=int((x< -1e-10).sum()),tied=int((abs(x)<=1e-10).sum()),meaningful_improved=int((x>=.1).sum()),meaningful_worse=int((x<=-.1).sum()),near_zero=int((abs(x)<.1).sum()),severe_loss=int((x< -5).sum()),positive_contribution_pp=float(np.maximum(x,0).mean()),negative_contribution_pp=float(np.minimum(x,0).mean()),net_adaptation_pp=float(x.mean())))
  def find(ns,ph,a):return next(x for x in rows if(x['namespace'],x['phase'],x['arm'])==(ns,ph,a))
  chosen=find(sel['namespace'],sel['phase'],sel['arm']);u=sel['unconstrained_mean_winner'];uncon=find(u['namespace'],u['phase'],u['arm']);old=find('main','coarse','C02_S'+('40'if ds=='hc2'else'30')+'_A100');eligible=[x for x in rows if x['severe_loss']<=sel['severe_loss_limit']];front=[]
  for a in eligible:
   dominated=any(b['full_vIoU_percent']>=a['full_vIoU_percent']and b['meaningful_improved']>=a['meaningful_improved']and b['meaningful_worse']<=a['meaningful_worse']and(b['full_vIoU_percent']>a['full_vIoU_percent']or b['meaningful_improved']>a['meaningful_improved']or b['meaningful_worse']<a['meaningful_worse'])for b in eligible)
   if not dominated:front.append(a)
  out[ds]=dict(recommended=chosen,original_baseline=old,unconstrained_mean_winner=uncon,protected_coverage_mean_Pareto_front=front,all_tested_configuration_coverage=rows,not_an_independent_confirmation=True)
  for name,x in [('original baseline',old),('recommended',chosen),('unconstrained mean winner',uncon)]:lines.append(f"| {ds} / {name} | {x['any_improved']} / {x['any_worse']} / {x['tied']} | {x['meaningful_improved']}/128 | {x['meaningful_worse']}/128 | {x['positive_contribution_pp']:+.6f} / {x['negative_contribution_pp']:+.6f} | {x['net_adaptation_pp']:+.6f} | {x['severe_loss']} |")
 put(P/'PARENT_COVERAGE_PUBLIC.json',dict(status='completed_saved_score_coverage_analysis',meaningful_threshold_pp=.1,datasets=out,new_GT_read=False,new_forward=False,per_video_configuration_switch=False));(P/'PARENT_COVERAGE.md').write_text('\n'.join(lines)+'\n');print('ALL_PARAMETER_PARENT_COVERAGE_COMPLETE')
if __name__=='__main__':main()
