"""Freeze old/new policy by a predeclared exposed-development ranking only."""
from pathlib import Path
import sys,json,numpy as np
from functools import cmp_to_key
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,sha
from scripts.run_stvg_referdino_dualscale_hc128_v1 import D,P,verify

def main():
 p=verify(True);assert read(P/'ROOT_INDEPENDENT_AUDIT.json')['status']=='pass';z=read(P/'RESULTS_PUBLIC.json');cost=read(P/'COST_BREAKDOWN.json');policies=list(p['policies']);rows=z['rows'];ranking=[]
 oldcost=read(ROOT/'results/stvg_error_state/2026-10-19/referdino_native_opd_hc128/COST_BREAKDOWN.json')
 for k in ['Old']+policies:
  vv=z['primary']['means'][k+'_Live_seedmean'];seconds=[v for r in (oldcost if k=='Old'else cost)['rows']for n,v in r['per_fit_seconds'].items()if n.startswith('Live_'if k=='Old'else k+'_Live_')];ranking.append(dict(policy=k,full_vIoU=vv['v'],outside8_sIoU=vv['unobserved_actual_s'],mean_fit_seconds=float(np.mean(seconds))))
 def compare(a,b):
  for field,descending in [('full_vIoU',True),('outside8_sIoU',True),('mean_fit_seconds',False)]:
   delta=a[field]-b[field]
   if abs(delta)>1e-9:return (-1 if delta>0 else 1)if descending else(1 if delta>0 else -1)
  return (a['policy']>b['policy'])-(a['policy']<b['policy'])
 ranking.sort(key=cmp_to_key(compare));selection=dict(status='frozen_after_development_only',selected_policy=ranking[0]['policy'],selected_scales=None if ranking[0]['policy']=='Old'else p['policies'][ranking[0]['policy']],ranking=ranking,seeds=p['seeds'],all_results_SHA256=sha(P/'RESULTS_PUBLIC.json'),confirmation_used_for_selection=False,no_further_structure_iteration=True)
 put(D/'SELECTION_LOCK.json',selection);put(P/'SELECTION_PUBLIC.json',selection)
 def fmt(x):return f"{x['mean']:+.4f} [{x['CI95'][0]:+.4f},{x['CI95'][1]:+.4f}]"
 txt=['# Target-size-normalized dual-scale output OPD, HC128 development','', 'Only policy coordinates/scales and the consistent TRUE mixture NLL changed. Frozen native ReferDINO selected8 labels, actual frozen spatial decoder, independent8×256 content states, E+finalMBR/physical frames/time/optimizer40steps remain. Seeds41/42/43, all reported. Candidates are output boxes, not decoded latent trajectory samples. Each actual update reruns complete decoder.','', 'Three predeclared (near,far) pairs: (.05,.25),(.05,.5),(.1,.5). Equal mixture in corrected(cx,cy,logw,logh); centers normalized by box(w,h); scale noise multiplicative.8 antithetic pairs per component,32 total. D/actions/teacher feedback stopped; exact logsumexp mixture NLL. No old single-Gaussian KL/control-variate. Thus old/new comparison changes the policy AND its fitting objective; not an isolated proof of wider coverage causality.','', 'Fresh Fixed retains initial mean/size covariance for all40 fresh paired batches. Live uses current actual mean/current detached size covariance. First round samples/feedback/update exact; no fixed-likelihood freezing. Boxes positive-sized but may cross image; no clipping/rejection or unreported truncated density.','', 'All2304 fits and128×19 new outputs sealed before GT. OldE/Direct/Fixed/Live reuse exact pinned prior results, no new baseline fit costs. Three-seed metric means are formed WITHIN parent then bootstrap128 parents,10000 resamples; not384 independent parents or box ensemble. Existing exposure and severe tails retained.','', '## Development primary comparisons (pp)','', '| Comparison | Full vIoU [95%CI] | Outside8 known actual sIoU [95%CI] |','|---|---:|---:|']
 for pair,v in z['primary']['paired'].items():txt.append(f"| {pair} | {fmt(v['v'])} | {fmt(v['unobserved_actual_s'])} |")
 for seed,g in z['by_seed'].items():
  txt+=['',f'## Seed {seed}','', '| Comparison | Full vIoU [95%CI] | Outside8 known actual sIoU [95%CI] |','|---|---:|---:|']
  for pair,v in g['paired'].items():txt.append(f"| {pair} | {fmt(v['v'])} | {fmt(v['unobserved_actual_s'])} |")
 tails={}
 for k in ['Old']+policies:
  arm=k+'_Live_seedmean';dd=sorted((100*(r['values'][arm]['v']-r['values']['E_MBR']['v']),r['alias'])for r in rows);vals=np.array([x for x,a in dd]);tails[k]=dict(positive=int(sum(vals>1e-12)),negative=int(sum(vals< -1e-12)),unchanged=int(sum(abs(vals)<=1e-12)),below_minus3pp=int(sum(vals< -3)),below_minus5pp=int(sum(vals< -5)),worst5=dd[:5],best5=dd[-5:])
 txt+=['','## Selection, tails and cost','',f"Frozen selected policy: **{selection['selected_policy']}**, scales{selection['selected_scales']}. Ranking uses full vIoU then outside8 sIoU then measured fit time, as predeclared. Selection is exposed-development tuning, not proof of significance. All configurations and every seed retained. Confirm only this frozen selection, no confirmation-driven switches.",'',f"New GPU allocation{cost['total_GPU_seconds']:.3f}s; updates{cost['actual_state_updates']}; full trajectory decoder{cost['trajectory_decoder_calls']}; offset{cost['offset_decoder_calls']}; sampled geometric feedback{cost['sampled_candidate_feedback']}. New development Source/ReferDINO/DINO calls0. Original caching cost not free. Actual per-fit cost and positive/negative tails below.",'',json.dumps(dict(ranking=ranking,tails=tails),ensure_ascii=False,indent=2),'','Independent saved-artifact audit reconstructs RNG/antithetic actions/current-or-initial covariance, TRUE mixture density via independent torch distributions, geometric weights/mean gradient/Adam, actual full/outside8 metrics and parent bootstrap. It is not a second full GPU rerun. No trained head or pretrained weight updates. No automatic production promotion or further structure search.']
 (P/'REPORT.md').write_text('\n'.join(txt)+'\n');put(P/'DECISION_PUBLIC.json',dict(status='development_complete_confirmation_pending',selection=selection,tails=tails,primary=z['primary'],by_seed=z['by_seed'],no_production_promotion=True,no_further_structure_iteration=True))
 print('FROZEN SELECTION',selection['selected_policy'],selection['selected_scales'],flush=True)
if __name__=='__main__':main()
