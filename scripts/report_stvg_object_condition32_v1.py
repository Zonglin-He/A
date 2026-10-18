"""Report anonymous complete readouts, signed paired contrasts and costs."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_object_condition32_v1 import D,P,CODE,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put,sha

def main():
 import numpy as np
 from vg_tta.stvg_object_condition32_v1 import ARMS
 z=read(P/'RESULTS_PUBLIC.json');p=verify();assert read(P/'ROOT_INDEPENDENT_AUDIT.json')['status']=='pass'
 cfg={k:v for k,v in p.items()if k not in ['records','created_epoch','deadline_epoch']};put(P/'CONFIGURATION.json',cfg)
 cost=dict(total_Source_captures_completed=0,final_worker_DINO_encodings_completed=0,engineering_DINO_encodings_completed=0,
  new_training_runs=0,optimizer_steps=0,new_decoders_trained=0,prefixes_reused=32,
  original_Source_E_DINO_and_prefix_generation_cost_excluded=True,first_online_end_to_end_latency_measured=False,
  conditional_readouts_seconds=sum(r['compute']['readout_seconds']for r in z['rows']),**z['compute']);put(P/'COST_BREAKDOWN.json',cost)
 put(P/'MODEL_PROVENANCE.json',dict(checkpoints={f:sha(ROOT/f)for f in ['checkpoints/TASTVG_VidSTG.pth','checkpoints/TASTVG_HCSTVG2.pth']},
  target_hc2_uses='TASTVG_VidSTG.pth',target_vidstg_uses='TASTVG_HCSTVG2.pth',all_pretrained_weights_frozen=True,
  DINO='Original tiny expert cached output only, no DINO model instantiated',prefixes='Exact original frozen full fused prefix32 from shared_text_query32; no new Source forward'))
 lines=['# Frozen object-query reconstruction32','',
 'Completed exposed NH16/NV16 development experiment, one query per parent. No new localization head, source training, optimizer, OPD, DINO calls or video feature extraction. The stopped D-FINE-inspired source qualification remains unchanged. No deployment promotion.','',
 'Five outputs share the exact E trajectory, original four planned expert locations and pretrained TA-STVG source checkpoints. E versus DINO spatial conditions use the same E-selected reference positions and original associations. The learned native aggregation and trained decoder are reused. Regions constrain fractional grid support, each frame retains native attention mass, total query coefficients match the native mean, and the final initial query matches the native L2 norm. Full memory/context remains.','',
 'Only final-pass initial isq/itq changes. The original first pass and routing remain exact. Motion conditions read object regions in ALL actual frames, outside as well as inside the initial interval. This is not DSE-VTG replication and does not remove all internal actionness. Motion changes include full-frame support versus native chosen-frame pooling, so gains cannot be attributed exclusively to instance disambiguation.','',
 'E-retaining output: center(E)+(center(newNative)-center(Source)); size(E)*exp(log(size(newNative))-log(size(Source))). Zero condition is exactly E. No second application/refit of E. No-reference cases remain exact E in every arm.','',
 '| Target | Output | vIoU % | tIoU % | Unobserved actual sIoU % | Changed interval /16 |',
 '|---|---|---:|---:|---:|---:|']
 for ds,g in z['groups'].items():
  for a in ARMS:
   m=g['metrics'][a];changed=0 if a=='E_MBR'else g['boundary_changes'][a]['changed']
   lines.append(f"| {ds} | {a} | {100*m['v']:.6f} | {100*m['t']:.6f} | {100*m['unobserved_actual_s']:.6f} | {changed} |")
 lines+=['','## Paired differences from E (percentage points)','', '| Target | Output | vIoU delta [95% CI] | tIoU delta [95% CI] | Unobserved actual sIoU delta [95% CI] |','|---|---|---|---|---|']
 for ds,g in z['groups'].items():
  for a in ARMS[1:]:
   ps=g['paired'][a+'_minus_E_MBR'];parts=[]
   for k in ['v','t','unobserved_actual_s']:
    t=ps[k];parts.append(f"{t['mean']:+.6f} [{t['CI95'][0]:+.6f}, {t['CI95'][1]:+.6f}]")
   lines.append('| '+ds+' | '+a+' | '+' | '.join(parts)+' |')
 lines+=['','All32 outputs sealed before new GT evaluation. Official complete-query denominators remain16/domain. GT is only offline scoring; missing annotations remain null. Parent=query; paired bootstrap10000 seed20261018. All signed per-query results, positive/negative/neutral counts, severe losses below -5pp, leave-one sensitivity and near/far actual-node statistics are in RESULTS_PUBLIC.json. Near/far uses prelocked nearest planned-anchor distance/full sampled physical span <=1/8 versus >1/8; interpolation is not additional independent supervision.','']
 for ds,g in z['groups'].items():
  lines +=[f'## {ds}: matched inputs and changes',f"Active {g['active']}/16; E-selected references {g['references']}. No-ref {16-g['active']}/16 remains in every mean."]
  for a in ARMS[1:]:
   pp=g['paired'][a+'_minus_E_MBR'];lines.append(a+': '+str(dict(full_v_tail=pp['v'],near_actual=pp['near_actual_s'],far_actual=pp['far_actual_s'],boundary=g['boundary_changes'][a])))
  lines.append('DINO_region_spatial minus E_region_spatial (same support): '+str(g['paired']['DINO_region_spatial_minus_E_region_spatial']))
  lines.append('Spatial_then_temporal minus DINO_region_spatial (same boxes): '+str(g['paired']['Spatial_then_temporal_minus_DINO_region_spatial']))
 lines +=['','## Scope and decision',
 'This run tests one fixed no-training region-conditioned query construction. Changes in isq/itq or logits establish interface connectivity, not correct identity or event understanding. A useful dense teacher and OPD advantage require later matched evidence; neither is claimed here. No automatic OPD, extra rounds, threshold search, backbone change or wider cohort follows this experiment.',
 'All five outputs are published regardless of sign. Negative means or intervals crossing zero do not establish that every possible way of injecting DINO evidence is ineffective. They constrain this specific mass-and-norm-preserving, second-pass, region-gated initial-query construction.',
 '', f"GPU allocation including model load, cache replay, checks and conservative initialization allowance: {cost['GPU_allocation_seconds']:.6f}s; four conditional-readout sum {cost['conditional_readouts_seconds']:.6f}s; peak allocated {cost['peak_allocated_bytes']/2**30:.6f}GiB. Zero new Source captures/DINO encodings/training. These exclude original E and prefix acquisition; they are not first deployment end-to-end latency.",
 '', 'Independent audit checks checkpoint-derived CPU query reconstruction, actual fractional ROI support, preserved query mass/norm, unchanged native routing, all32 E/Source exactness, final physical-distribution MBR, unknown support and independent official interpolation/metric arithmetic. See ROOT_INDEPENDENT_AUDIT.json.',
 '', 'Conceptual comparison: [DSE-VTG query conditioning](https://arxiv.org/html/2609.08850v1); actual implementation follows the existing TA-STVG aggregation/decoder rather than its offset/similarity objective.']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n')
 import matplotlib
 matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,2,figsize=(12,7))
 for col,ds in enumerate(['hc2','vidstg']):
  g=z['groups'][ds]
  for row,k in enumerate(['v','unobserved_actual_s']):
   ax=axes[row,col];ps=[g['paired'][a+'_minus_E_MBR'][k]for a in ARMS[1:]];mu=[t['mean']for t in ps];ci=np.array([t['CI95']for t in ps]);ax.errorbar(range(4),mu,yerr=[np.array(mu)-ci[:,0],ci[:,1]-np.array(mu)],fmt='o',capsize=4)
   ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(4),['E spatial','DINO spatial','Object time','Spatial+time'],rotation=15);ax.set(title=ds+' '+('full vIoU'if k=='v'else'unobserved actual sIoU'),ylabel='Difference from E (pp)')
 fig.tight_layout();fig.savefig(P/'COMPARISON.svg');fig.savefig(D/'COMPARISON.png',dpi=140);plt.close(fig)
 files=CODE+['scripts/audit_stvg_object_condition32_v1.py','scripts/report_stvg_object_condition32_v1.py']+[str(f.relative_to(ROOT))for f in P.iterdir()if f.is_file()]
 put(D/'PUBLIC_FILES_FINAL.json',dict(files=sorted(files)));put(D/'REPORT_COMPLETION.json',dict(status='complete',count=32))

if __name__=='__main__':main()
