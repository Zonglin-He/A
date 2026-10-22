"""Render the TA-STVG-only paper package from sealed report JSON, no inference."""
from tastvg_paper_full import RUN,PUBLIC,read,atomic,sha

def main():
 reports={ds:read(PUBLIC/ds/'report.json')for ds in ('hc2','vidstg')};lines=['# TA-STVG: frozen dual-domain paper experiments','','Source/Full are exact aliases of the previously completed official full evaluation. Auxiliary arms use the same frozen inputs/checkpoints/eight observations. No full-result parameter selection. Historical development/full exposure is disclosed.','',
 '| Method | Expert observations | HC2 m_vIoU | R@.3 | R@.5 | Vid m_vIoU | R@.3 | R@.5 |','|---|---:|---:|---:|---:|---:|---:|---:|']
 def f(x):return '—'if x is None else f'{100*x:.4f}'
 for a in ('Source','Interpolation','OursS','Full','HistoricalOPD'):
  vals=[]
  for ds,r in reports.items():
   m=r['means'].get(a,{});vals += [f(m.get(k))for k in ('m_vIoU','vIoU@0.3','vIoU@0.5')]
  if all(v=='—'for v in vals):continue
  name='Historical project OPD (not third-party generic TTA)'if a=='HistoricalOPD'else a
  lines.append('| '+name+' | '+('0'if a=='Source'else'8')+' | '+' | '.join(vals)+' |')
 lines+=['','Third-party generic TTA was not verified. The historical project OPD row is separately labeled; no new OPD run was performed and incomplete Vid8405 is not used.','',
 '## Four module ablations','','All rows retain shared initialization. Two spatial ablations keep Full time/anchors; Native temporal readout is exactly OursS and runs unchanged spatial inference on the native interval.','',
 '| Target | Configuration | Full m_vIoU% | Δ versus Full pp | Unobserved sIoU% | tIoU% |','|---|---|---:|---:|---:|---:|']
 for ds,r in reports.items():
  for a in ('Full','NoLocal','Independent','OursS'):
   m=r['means'][a];lines.append(f"| {ds} | {a if a!='OursS'else'Native temporal readout'} | {f(m['m_vIoU'])} | {100*(m['m_vIoU']-r['means']['Full']['m_vIoU']):+.4f} | {f(m['m_sIoU_unobserved_actual'])} | {f(m['m_tIoU'])} |")
 lines+=['','## Spatial stages on common support','','Actual sampled GT-supported nodes; common8 observed positions, independent of each predicted interval. Query means then equal-query means; effective query counts shown. Shared initialization is an intermediate result, not a deletion ablation.','',
 '| Target | Spatial output | Observed sIoU% | n | Unobserved sIoU% | n |','|---|---|---:|---:|---:|---:|']
 for ds,r in reports.items():
  for a in ('Source','SharedInit','OursS'):
   m=r['means'][a];n=r['valid_queries'][a];lines.append(f"| {ds} | {a} | {f(m['m_sIoU_observed_actual'])} | {n['observed_actual_sIoU']} | {f(m['m_sIoU_unobserved_actual'])} | {n['unobserved_actual_sIoU']} |")
 lines+=['','## Complete extent versus mask-derived extent','','Same anchor and direction; no GT-selected direction. All-slot IoU zero-fills missing candidates; jointly-available paired IoU and missing rate reported separately.','',
 '| Target | Candidate | All-slot unobserved IoU% | Both available IoU% | Missing% |','|---|---|---:|---:|---:|']
 for ds,r in reports.items():
  for a,m in r['extent']['mean_per_query'].items():lines.append(f"| {ds} | {a} | {f(m['iou_all'])} | {f(m['iou_both_available'])} | {f(m['missing'])} |")
 lines+=['','## Temporal module','','| Target | Readout | tIoU% | Boundary MAE s | Zero-overlap% | Mean interval s | Full vIoU% |','|---|---|---:|---:|---:|---:|---:|']
 for ds,r in reports.items():
  for a in ('OursS','Full'):
   m=r['means'][a];lines.append(f"| {ds} | {a} | {f(m['m_tIoU'])} | {m['boundary_MAE_seconds']:.4f} | {f(m['tIoU_eq_0'])} | {m['interval_length_seconds']:.4f} | {f(m['m_vIoU'])} |")
 lines+=['','All core paired increments and parent-cluster95% intervals, distance bins with node/query counts, natural cache costs segmented by worker, signed changes and every severe decline are in [HC2](hc2/report.json) and [VidSTG](vidstg/report.json).','']
 if(PUBLIC/'cost.json').exists():
  c=read(PUBLIC/'cost.json');lines+=['## Matched runtime','','16 parents per direction,3 repeats, warmup excluded. No inter-query RGB/visual/result reuse; resident checkpoint/processes, deployment CPU/GPU offload and transfers included. Not all modules simultaneously GPU resident. Nested stages are not summed.','', '| Target | Method | Mean total s/query | Median total s/query |','|---|---|---:|---:|']
  for ds,arms in c['means'].items():
   for a,v in arms.items():lines.append(f"| {ds} | {a} | {v['mean_total_seconds']:.4f} | {v['median_total_seconds']:.4f} |")
  lines+=['','[Full cost components, loads and scope](cost.json).','']
 lines+=['## Qualitative status','','Continuous sampled-frame sheets are private local assets, pending root visual review for the four requested categories. Raw dataset frames, captions, GT coordinates and features are not exported to this repository. See [status](qualitative_status.json).']
 (PUBLIC/'REPORT.md').write_text('\n'.join(lines)+'\n');atomic(RUN/'report_receipt.json',dict(status='rendered',report_sha256=sha(PUBLIC/'REPORT.md'),no_new_inference=True))

if __name__=='__main__':main()
