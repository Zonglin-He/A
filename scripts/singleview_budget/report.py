"""Render all signed positive/negative outcomes without selecting new parameters."""
from pathlib import Path
import json,numpy as np
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def main():
 d=ROOT/'results/singleview_budget_v1';r=read(d/'report.json');lines=['# Fixed single320 screen: rejected','',r['exposure'],'','| Method | Full vIoU % | tIoU % | Mean interval s | Boundary MAE s |','|---|---:|---:|---:|---:|']
 for a,m in r['means'].items():lines.append(f"| {a} | {100*m['m_vIoU']:.4f} | {100*m['m_tIoU']:.4f} | {m['interval_length_seconds']:.4f} | {m['boundary_MAE_seconds']:.4f} |")
 lines+=['','| Comparison | Full vIoU pp,95%CI | tIoU pp,95%CI | Severe v loss >5pp |','|---|---:|---:|---:|']
 for a,v in r['paired'].items():
  v,t=v['v'],v['t'];fmt=lambda x:f"{x['mean']:+.4f} [{x['CI95'][0]:.4f},{x['CI95'][1]:.4f}]";lines.append(f"| {a} | {fmt(v)} | {fmt(t)} | {v['declined_over5pp']} |")
 lines+=['','## Actual paired extra-branch cost','',r['cost_scope'],'','| Actual branch | Mean s | Median s |','|---|---:|---:|']
 for a,v in r['cost'].items():lines.append(f"| {a} | {v['mean']:.4f} | {v['median']:.4f} |")
 g=r['gate'];lines+=['',f"Extra-branch reduction {100*g['extra_branch_cost_reduction']:.2f}%; Strong full-v loss {g['v_loss_vs_Strong_pp']:.4f}pp. Gate FAIL: only the speed criterion passes. No320 HC32/Vid32 run, no288/352/384 or weight scan; separately authorized448 recheck proceeds.",'','The candidate retains full-frame context, original offset0, same-view hull and .5Global6. Its failure is not explained by losing the hull or by crop-induced motion changes. A middle resolution is not guaranteed to interpolate accuracy between224 and448: its native features, discrete temporal selection and boundary posterior can change non-monotonically. This experiment establishes the fixed320 result, not the causal share of training-scale mismatch, visual detail and query changes.','', 'All63 outputs sealed before locked-label scoring. Complete actual trajectories and a fixed common Strong spatial trajectory are separately evaluated. Thresholds, observed/unobserved spatial metrics, interval length, boundary errors and all signed losses are in report.json/signed_queries.tsv. No deployment change or full resume.']
 lines+=['', 'The matched single448-to-single320 branch reduction is only16.68% (2.2484s to1.8733s), while complete vIoU loses1.4437pp and tIoU2.9801pp under the SAME offset0/self-hull/global fusion. The nominal51% input-pixel ratio does not imply51% runtime: temporal replay, text processing, transfers and module copies remain. These timings include required native implementation costs and are not whole-method speedups.']
 (d/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
 patterns={}
 for b in ['Strong','Global6','Single448_0']:
  rr=r['queries_anonymous'];patterns[b]=dict(rescues_half=[v['query']for v in rr if v['values'][b]['t']<.5<=v['values']['Single320']['t']],harms_half=[v['query']for v in rr if v['values']['Single320']['t']<.5<=v['values'][b]['t']],severe_full_v_loss=[v['query']for v in rr if v['values']['Single320']['v']-v['values'][b]['v']<-.05])
 write(d/'crossings.json',patterns)
 q=ROOT/'results/single448_recheck_v1'
 if not(q/'summary.json').exists():return
 lines=['# Single448 same-view hull: two-direction development recheck','','Both offsets fixed separately: offset0 primary,offset1 sensitivity. Existing32 parents/domain were historically exposed. This is a development recheck, not untouched confirmation, per-query branch selection, or a statistical equivalence claim. BoxFixed remains unchanged.320 failed its earlier screen and was not run on these panels.','']
 for ds in ['hc2','vidstg']:
  z=read(q/ds/'report.json');lines +=[f'## {ds}:32 parents','', '| Method | Full vIoU % | tIoU % | Unobserved sIoU % | Mean interval s |','|---|---:|---:|---:|---:|']
  for a,m in z['means'].items():
   u=m['m_sIoU_unobserved_actual'];u='—'if u is None else f'{100*u:.4f}';lines.append(f"| {a} | {100*m['m_vIoU']:.4f} | {100*m['m_tIoU']:.4f} | {u} | {m['interval_length_seconds']:.4f} |")
  lines +=['','| Comparison | vIoU pp,95%CI | tIoU pp,95%CI | Severe v loss >5pp |','|---|---:|---:|---:|']
  for name,p in z['paired'].items():
   v,t=p['v'],p['t'];lines.append(f"| {name} | {v['mean']:+.4f} [{v['CI95'][0]:.4f},{v['CI95'][1]:.4f}] | {t['mean']:+.4f} [{t['CI95'][0]:.4f},{t['CI95'][1]:.4f}] | {v['declined_over5pp']} |")
  lines+=['','### Actual complete deployment timing,4 fixed queries','', '| Method | Mean all4 s | Median all4 s | Mean excluding first query s (n=3) | Extra temporal branch s,excluding first |','|---|---:|---:|---:|---:|']
  for a,c in z['costs'].items():
   t=c['actual_end_to_end_seconds'];lines.append(f"| {a} | {t['mean']:.4f} | {t['median']:.4f} | {t['excluding_first_mean']:.4f} | {c['extra_temporal_branch_seconds']['excluding_first_mean']:.4f} |")
  lines+=['',z['cost_scope'],'', 'All original Source and Global6 outputs recover previous fixed32 bitwise. Every actual timed complete pipeline independently recomputes224, eight-frame ReferDINO and geometry and must reproduce the cached precision prediction bitwise. Encoder hooks only count calls, with no module-level timing/synchronization. Startup is visible; all4 and matched excluding-first-query means are both reported. Three warm queries/domain are a small cost sample, not a full-stream throughput guarantee.','']
 lines+=['## Decision boundary','','Do not choose an offset per query or after reading GT; retain the complete offset0/offset1 signed panel. No automatic production adoption, full resume or new resolution/weight scan. Precision changes on these exposed32 panels do not establish equivalence; the complete method must be assessed against Source and temporal differences against fixed Global6/Strong. Entire fixed-space effects, thresholds, severe tails, individual timings and bootstrap CIs are public.']
 (q/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
