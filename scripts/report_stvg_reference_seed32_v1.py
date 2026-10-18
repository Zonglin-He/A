"""Finite reference seed capability report, no OPD effectiveness claim."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_reference_seed32_v1 import D,P,CODE,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put

def main():
 import numpy as np
 from vg_tta.stvg_reference_seed32_v1 import ARMS
 z=read(P/'RESULTS_PUBLIC.json');p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');gate=read(P/'GATE.json');assert read(P/'ROOT_INDEPENDENT_AUDIT.json')['status']=='pass'
 put(P/'CONFIGURATION.json',{k:v for k,v in p.items()if k not in ['records','created_epoch','deadline_epoch']})
 pr=read(ROOT/'results/stvg_error_state/2026-10-18/object_condition32/MODEL_PROVENANCE.json');pr.update(prefix_reused=True,feature='original full fused cached prefix and native spatial content; second-pass initial reference only',new_Source_captures=0,new_DINO_calls=0);put(P/'MODEL_PROVENANCE.json',pr)
 cost=dict(total_Source_captures_completed=0,final_worker_DINO_encodings_completed=0,engineering_DINO_encodings_completed=0,new_training_runs=0,new_heads=0,new_OPD_updates=0,GPU_allocation_seconds=bar['GPU_allocation_seconds'],actual_worker_GPU_seconds=bar['actual_worker_GPU_seconds'],peak_allocated_bytes=bar['peak_allocated_bytes'],readout_and_equal_control_seconds=sum(r['compute']['new_GPU_readout_seconds']for r in z['rows']),old_Source_E_DINO_prefix_cost_excluded=True,not_first_online_latency=True);put(P/'COST_BREAKDOWN.json',cost)
 def ci(t):return 'unknown'if t['mean']is None else f"{t['mean']:+.6f} [{t['CI95'][0]:+.6f},{t['CI95'][1]:+.6f}]"
 lines=['# Frozen spatial decoder reference geometry qualification32','',
 'Same exposed NH16/NV16, one query per parent, original opposite-source EMA TA-STVG weights, E/final-MBR and four DINO cached references. No new model/head training, Source capture, expert, DINO, QR/LN weight update or OPD. Old Fine and coarse bidirectional negative results remain unchanged.','',
 'Only SECOND-pass PosDecoder initial pred_boxes is replaced with the complete Source trajectory S or E, correctly split by original two offsets. Original native content, first pass/routing, full memory and fixed temporal output remain. Each subsequent layer predicts an absolute sigmoid box and detaches it for its next reference; this is not an iterative residual bbox head, nor spatial ROI resampling. No clamp/blend/layer selection.','',
 'Source-reference and E-reference outputs are raw absolute decoder outputs. The old E-retaining transform is deliberately absent: applying E+[D(E)-S] could double the E correction for an identity-like decoder. Native initial-reference control reproduces Source exactly; same S-reference clone gives exact output. D(S)=S is not required: S was the final prediction, not the original initial reference. No-E-support queries retain E in BOTH reseeded reported arms; raw outputs are saved privately for engineering diagnostics. Same support rule, all16 remain in means.','',
 '| Output | HC complete vIoU % | Vid complete vIoU % | HC unobserved actual sIoU % | Vid unobserved actual sIoU % |','|---|---:|---:|---:|---:|']
 for a in ARMS:lines.append('| '+a+' | '+' | '.join(f"{100*z['groups'][ds]['metrics'][a][ep]:.6f}"for ds,ep in [('hc2','v'),('vidstg','v'),('hc2','unobserved_actual_s'),('vidstg','unobserved_actual_s')])+' |')
 for ds,g in z['groups'].items():
  lines+=['',f'## {ds}: paired parent differences, pp','', '| Contrast | Complete vIoU [95% CI] | Unobserved actual sIoU [95% CI] | v positive/negative/neutral | v below -5pp |','|---|---|---|---|---|']
  for c,t in g['paired'].items():
   v=t['v'];lines.append(f"| {c} | {ci(v)} | {ci(t['unobserved_actual_s'])} | {v['positive']}/{v['negative']}/{v['neutral']} | {v['severe_loss_below_minus5pp']} |")
  lines.append(f"Active support {g['active']}/16. All no-reference and missing-GT cases retained; absent actual GT stays unknown.")
 lines+=['','## Fixed eligibility',str(gate),'',
 'Only both-domain positive E-reference minus E and minus Source-reference complete-v AND unobserved-actual-s means qualify further work. Confidence intervals retain uncertainty; passing only a damaged reseeding control is insufficient. Failure stops this exact initial-reference reseeding without fusion weights, per-layer clamp or scans. This run tests existing decoder capability and is not an OPD effectiveness result.','',
 'All32 outputs/traces sealed before new GT scoring. This is historically exposed target development data, not independent confirmation. GT does not generate references, choose layers or deployment outputs. Temporal intervals unchanged across all four arms; tIoU identical. Official interpolation does not count as additional visual supervision.','',
 'Independent CPU audit rechecks actual Source/E reference tensors and offset order, identical native content, all six saved outputs and final raw reconstruction with no E transform, native-reference identity, no-reference no-op, independent official interpolation metrics and parent paired bootstrap. Frozen weights checked in actual model execution and registries pinned. No extra GPU/model rerun for audit.','',
 'D(E) versus D(S) complete-v differences are HC+.005643pp and Vid-.000290pp, intervals cross zero. D(S) versus original Source is only -.024681/+.001052pp, while E versus Source was +2.920415/+3.906945pp. Thus the large loss versus E mainly reflects not retaining the established E output correction, not a demonstrated large additional harm caused by choosing E rather than S as reference. Per-layer forward response is in REFERENCE_RESPONSE_PUBLIC.json; no new identity claims.','Initial independent CPU metric verifier formed xyxy in float32, unlike official float64 evaluation; verifier-only float64 input cast corrected, original verifier preserved, strict2e-10 tolerance retained. Predictions/scoring/model runs unchanged.','Costs (old Source/E/DINO/prefix generation excluded): '+str(cost),'',
 'This interface is specific to the inspected TA-STVG PosDecoder. No production promotion or backbone-general claim. Single GPT-6.1 Sol reviewer, root decision authority and original09:30 absolute deadline retained.']
 lines+=['','## Every signed full-tube change vs E, pp','','| Alias | Target | Source-reference | E-reference |','|---|---|---:|---:|']
 for r in z['rows']:lines.append('| '+r['alias']+' | '+r['dataset']+' | '+' | '.join(f"{100*(r['values'][a]['v']-r['values']['E_MBR']['v']):+.6f}"for a in ARMS[2:])+' |')
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n')
 import matplotlib
 matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(9,4))
 for ax,ds in zip(axes,['hc2','vidstg']):
  tt=[z['groups'][ds]['paired'][a+'_minus_E_MBR']['v']for a in ARMS[2:]];m=np.array([t['mean']for t in tt]);c=np.array([t['CI95']for t in tt]);ax.errorbar(range(2),m,yerr=[m-c[:,0],c[:,1]-m],fmt='o',capsize=4);ax.axhline(0,color='k',lw=.7);ax.set_xticks(range(2),['Source-reference','E-reference']);ax.set(title=ds,ylabel='Complete vIoU difference vs E (pp)')
 fig.tight_layout();fig.savefig(P/'COMPARISON.svg');fig.savefig(D/'COMPARISON.png',dpi=140);plt.close(fig)
 put(D/'PUBLIC_FILES_FINAL.json',dict(files=CODE+['scripts/audit_stvg_reference_seed32_v1.py','scripts/report_stvg_reference_seed32_v1.py','scripts/analyze_stvg_reference_seed32_v1.py']+[str(f.relative_to(ROOT))for f in P.iterdir()if f.is_file()]));put(D/'REPORT_COMPLETION.json',dict(status='complete',count=32))
if __name__=='__main__':main()
