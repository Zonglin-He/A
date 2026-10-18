"""Anonymous complete three-feature qualification report, no OPD claim."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_fine_feature32_v1 import D,P,CODE,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put

def main():
 import numpy as np
 from vg_tta.stvg_fine_feature32_v1 import LAYERS
 z=read(P/'RESULTS_PUBLIC.json');p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');gate=read(P/'GATE.json');assert read(P/'ROOT_INDEPENDENT_AUDIT.json')['status']=='pass'
 put(P/'CONFIGURATION.json',{k:v for k,v in p.items()if k not in ['records','created_epoch','deadline_epoch']})
 provenance=read(ROOT/'results/stvg_error_state/2026-10-18/object_condition32/MODEL_PROVENANCE.json');provenance.pop('prefixes',None)
 provenance.update(prefix_reused=False,
  feature='actual checkpoint-frozen ResNet layer3,1024channels,stride16; original input_proj256 coarse,stride32',fine_source_forward_count=32,DINO='Original four-frame cached output only; no new expert call')
 put(P/'MODEL_PROVENANCE.json',provenance)
 cost=dict(total_Source_captures_completed=32,final_worker_DINO_encodings_completed=0,engineering_DINO_encodings_completed=0,
  new_training_runs=0,new_heads=0,new_pretrained_weights_updated=0,OPD_update_runs_this_qualification=0,
  GPU_allocation_seconds=bar['GPU_allocation_seconds'],actual_worker_GPU_seconds=bar['actual_worker_GPU_seconds'],peak_allocated_bytes=bar['peak_allocated_bytes'],
  matching_seconds=sum(r['compute']['matching_seconds']for r in z['rows']),capture_seconds=sum(r['compute']['capture_seconds']for r in z['rows']),
  old_Source_E_DINO_generation_cost_excluded=True,not_first_deployment_latency=True);put(P/'COST_BREAKDOWN.json',cost)
 def ci(t):return 'unknown'if t['mean']is None else f"{t['mean']:+.6f} [{t['CI95'][0]:+.6f},{t['CI95'][1]:+.6f}]"
 lines=['# Actual finer frozen feature qualification32','',
 'Same exposed NH16/NV16, one query per parent, unchanged E/final-MBR and original four-frame DINO/E references. No new expert, DINO call, projection/head training or pretrained parameter update. This run qualifies feedback; it is not an OPD result. Old query-reconstruction and D-FINE-head negative results stay unchanged.','',
 'Layer selection was locked before any new evaluation: ResNet layer3, stride16/1024channels. No layer sweep. Compare old input_proj coarse256, actual layer3, and adaptive-average-pooled raw layer3 on the exact old grid. The last two retain identical channels/layer and differ in grid support; coarse versus finer also differs in semantics/channels. A positive comparison would not isolate resolution alone.','',
 'Original two17 center-state candidates, all reference associations, affinity temperature0.07, region propagation/soft-IoU, Gaussian sampling prior, reward temperature0.1 and argmax readout are fixed. Planned anchor nodes, sizes and event intervals stay E. Genuine intermediate activations captured during one original Source forward per query; Source boxes/logits and recaptured old input_proj features match old arrays exactly. No old-map interpolation masquerading as new information.','',
 '| Output | HC vIoU % | Vid vIoU % | HC unobserved actual sIoU % | Vid unobserved actual sIoU % |','|---|---:|---:|---:|---:|']
 arms=['E_MBR']+LAYERS
 for a in arms:lines.append('| '+a+' | '+' | '.join(f"{100*z['groups'][ds]['metrics'][a][ep]:.6f}"for ds,ep in [('hc2','v'),('vidstg','v'),('hc2','unobserved_actual_s'),('vidstg','unobserved_actual_s')])+' |')
 for ds,g in z['groups'].items():
  lines+=['',f'## {ds} paired differences, pp','', '| Contrast | Complete vIoU [95% CI] | Unobserved actual sIoU [95% CI] | v positive/negative/neutral | v severe below -5pp |','|---|---|---|---|---|']
  for c,t in g['paired'].items():
   v=t['v'];lines.append(f"| {c} | {ci(v)} | {ci(t['unobserved_actual_s'])} | {v['positive']}/{v['negative']}/{v['neutral']} | {v['severe_loss_below_minus5pp']} |")
  lines+=[f"Active references {g['active']}/16; all16 stay in full-tube means. Missing actual-frame GT remains unknown."]
  lines+=['','Ranking diagnostics use query-balanced known unobserved actual nodes, not independent candidate pairs:']
  for layer in LAYERS:
   for key in ['raw_auc','posterior_auc','strong_pool_no_good_outranks_E_raw','strong_pool_no_good_outranks_E_posterior','raw_top_gain_pp','posterior_top_gain_pp']:
    lines.append(layer+' '+key+': '+str(g['ranking'][layer].get(key)))
 lines+=['','## All signed complete-tube changes versus E, pp','','| Alias | Target | Coarse | Fine layer3 | Fine downsampled |','|---|---|---:|---:|---:|']
 for r in z['rows']:lines.append('| '+r['alias']+' | '+r['dataset']+' | '+' | '.join(f"{100*(r['values'][a]['v']-r['values']['E_MBR']['v']):+.6f}"for a in LAYERS)+' |')
 lines+=['','## Prelocked decision',str(gate),'',
 'Gate uses both-domain positive full-v and unobserved actual-s differences from E, plus Fine-minus-Coarse full-v positive. Confidence intervals are reported as uncertainty. Passing is resource eligibility for a separately locked actual OPD experiment, not proof that OPD works. Failure stops this exact layer3 cosine feedback interface; no scanning layer/temperature/step count or adding new training/expert.','',
 'All32 outputs and feedback sealed before NEW GT evaluation. Historical target development exposure is retained. GT never enters candidate generation, anchor acceptance, feedback, readout or feature-layer choice. AUC, ranking on a perturbed pool, or visually smooth heatmaps cannot substitute for better predictions than E. Official interpolation is not new independent visual evidence. All no-reference and severe-loss cases retained; parent bootstrap10000 seed20261018.','',
 'CPU independent audit rechecks actual affinity/ROI maps on fixed non-observed frames, every candidate soft-IoU/probability/readout, baseline identity, frozen sizes/anchors/time, full official interpolation metrics, paired aggregates and bootstrap. No GPU or model rerun for that audit.','',
 'Preflight correction before first model/GT: use actual pixel padding masks for the finer grid rather than upsampling the coarse valid mask; remove unrelated old OPD fields from the reused ranking helper. Original code/runtime retained, experiment arms/formulas/roster/deadline unchanged.','',
 'Costs (old E/DINO excluded): '+str(cost),
 '', 'No production promotion. Only current TA-STVG checkpoint interface tested, not cross-backbone generality. The single GPT-6.1 Sol review (replacing the initial Luna static review) is advisory; root decides subsequent bounded rounds before the absolute overnight09:30 deadline.']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n')
 import matplotlib
 matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 for ax,ds in zip(axes,['hc2','vidstg']):
  ts=[z['groups'][ds]['paired'][a+'_minus_E_MBR']['v']for a in LAYERS];m=np.array([t['mean']for t in ts]);c=np.array([t['CI95']for t in ts]);ax.errorbar(range(3),m,yerr=[m-c[:,0],c[:,1]-m],fmt='o',capsize=4);ax.axhline(0,color='k',lw=.7);ax.set_xticks(range(3),['Coarse','True layer3','Layer3→old grid']);ax.set(title=ds,ylabel='Complete vIoU difference vs E (pp)')
 fig.tight_layout();fig.savefig(P/'COMPARISON.svg');fig.savefig(D/'COMPARISON.png',dpi=140);plt.close(fig)
 put(D/'PUBLIC_FILES_FINAL.json',dict(files=CODE+['scripts/audit_stvg_fine_feature32_v1.py','scripts/report_stvg_fine_feature32_v1.py']+[str(f.relative_to(ROOT))for f in P.iterdir()if f.is_file()]));put(D/'REPORT_COMPLETION.json',dict(status='complete',count=32))

if __name__=='__main__':main()
