"""All fixed anonymous technical outcomes, including failed contracts."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,math,time
import numpy as np
from scripts.stvg_opd_observation_response_projection_common_v1 import *

def report():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.backends.backend_pdf import PdfPages
 s=read(OUT/'QUALIFICATION_SUMMARY.json');rows=read(OUT/'QUALIFICATION_ROWS.json')
 assert len(rows)==144 and read(BASE/'ROOT_QUALIFICATION.json')['status']=='pass'
 fd=s['actual_derivative_checks'];fdrows=[]
 for context in fd:
  for di,direction in enumerate(context['directions']):
   for f in direction['finite_differences']:
    fdrows.append(dict(parent_id=context['parent_id'],dataset=context['dataset'],boundary=context['boundary'],
      direction='fixed_random' if di==0 else 'original_Adam',JVP_error=direction['JVP_error'],JVP_tolerance=direction['JVP_tolerance'],
      VJP_error=context['VJP_error'],VJP_tolerance=context['VJP_tolerance'],**f))
 write(OUT/'FINITE_DIFFERENCE_ROWS.json',fdrows)
 metrics={}
 for ds in ['hc2','vidstg']:
  rr=[x for x in rows if x['dataset']==ds];ff=[x for x in fdrows if x['dataset']==ds]
  metrics[ds]=dict(local_proposals=len(rr),informative=sum(x['updated'] for x in rr),no_op=sum(not x['updated'] for x in rr),
   FD_checks=len(ff),FD_pass=sum(x['pass_check'] for x in ff),FD_contexts=sum(x['dataset']==ds for x in fd),
   FD_all_pass_contexts=sum(x['dataset']==ds and x['status']=='pass' for x in fd),
   FD_relative_error_median=float(np.median([x['relative_error'] for x in ff])),
   FD_relative_error_range=[float(min(x['relative_error'] for x in ff)),float(max(x['relative_error'] for x in ff))],
   VJP_all_pass=all(x['VJP_error']<=x['VJP_tolerance'] for x in ff),
   JVP_all_pass=all(x['JVP_error']<=x['JVP_tolerance'] for x in ff),
   repeat_bitwise_fail=sum(x['repeat_scientific_status']=='failed_bitwise' for x in rr),
   F_response_viability=s['datasets'][ds],
   repeat1_no_directional_FD_local_seconds={a:float(np.median([x['seconds'] for x in rr if x['arm']==a and x['repeat']==1 and x['updated']])) for a in ARMS})
 write(OUT/'TECHNICAL_STATISTICS.json',dict(datasets=metrics,formal_eligible=s['formal_eligible'],GT_read=False,
  actual_accepted_original_complete_replays=16,earlier_engineering_complete_replays=8,
  actual_total_complete_A_replays_including_engineering=24,accepted_fresh_captures=16,earlier_engineering_fresh_captures=8,
  complete_candidate_formal_fits=0,local_candidate_proposals=144,raw_qualification_repeats_are_not_independent_parents=True))
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'font.family':'DejaVu Sans'})
 figures=OUT/'figures';figures.mkdir(exist_ok=True)
 p=figures/'TECHNICAL_QUALIFICATION.pdf'
 with PdfPages(p) as pdf:
  fig,axs=plt.subplots(2,2,figsize=(12,8))
  for col,ds in enumerate(['hc2','vidstg']):
   ff=[x for x in fdrows if x['dataset']==ds]
   ratios=[x['absolute_error']/x['tolerance'] for x in ff]
   axs[0,col].scatter(range(len(ff)),ratios,c=['#207a58' if x['pass_check'] else '#bb3b3b' for x in ff],s=27)
   axs[0,col].axhline(1,color='black',ls='--',lw=1);axs[0,col].set_yscale('log')
   axs[0,col].set(xlabel='Fixed context / direction / h ordinal',ylabel='FD error / locked tolerance',title=f"{ds}: {metrics[ds]['FD_pass']}/{len(ff)} finite differences pass")
   rr=[x for x in rows if x['dataset']==ds and x['repeat']==1 and x['updated'] and x['arm'] in ['P','F']]
   for arm,color in [('P','#5065a8'),('F','#e08124')]:
    xx=[x for x in rr if x['arm']==arm]
    axs[1,col].plot(range(len(xx)),[x['observed_relative_error']*100 for x in xx],marker='o',ms=3,label=arm,color=color)
   axs[1,col].axhline(10,color='black',ls='--',lw=1)
   axs[1,col].set(xlabel='Fixed parent / boundary ordinal (repeat 1)',ylabel='Actual observed response mismatch (%)',title='Actual forward check; absolute tolerance also applies')
   axs[1,col].legend()
  fig.suptitle('Observed-response projection: actual decoder qualification',fontsize=16,y=.98)
  fig.text(.5,.015,'Autograd contractions agree; physical finite differences are a separate contract. No GT read or formal efficacy comparison.',ha='center',fontsize=10)
  fig.tight_layout(rect=(0,.045,1,.95));pdf.savefig(fig);plt.close(fig)
  fig,axs=plt.subplots(2,2,figsize=(12,8))
  for col,ds in enumerate(['hc2','vidstg']):
   rr=[x for x in rows if x['dataset']==ds and x['arm']=='F' and x['repeat']==1 and x['updated']]
   reduction=[(x['D_A']-x['D_candidate'])/max(1e-15,x['D_A'])*100 for x in rr]
   axs[0,col].bar(range(len(rr)),reduction,color=['#207a58' if x['accepted'] else '#bb3b3b' for x in rr])
   axs[0,col].axhline(0,color='black',lw=1)
   axs[0,col].set(xlabel='Fixed parent / boundary ordinal (repeat 1)',ylabel='Reduction of actual reference drift (%)',title=f'{ds}: F local counterfactuals')
   costs=metrics[ds]['repeat1_no_directional_FD_local_seconds']
   axs[1,col].bar(list(costs),list(costs.values()),color=['#5065a8','#e08124','#8b6cad'])
   axs[1,col].set(xlabel='Local proposal operator',ylabel='Measured median local probe seconds',title='Directional FD probes excluded (repeat 1)')
  fig.suptitle('Functional counterfactuals and actual local costs',fontsize=16,y=.98)
  fig.text(.5,.015,'Costs include Jacobian/solver/readouts and CPU audits; these are local probes at A states, not full candidate fitting times.',ha='center',fontsize=10)
  fig.tight_layout(rect=(0,.045,1,.95));pdf.savefig(fig);plt.close(fig)
 lines=['# Observation-response matched update: actual technical qualification','',
  '**Decision: stop this fixed recipe before the 192-fit panel.**' if not s['formal_eligible'] else '**Technical qualification supports entering the fixed 192-fit panel.**','',
  'Eight fixed historical queries (HC 4 / Vid 4), two fresh captures each, 16 accepted complete original-A replays, and 144 local P/F/R proposals at first/middle/last A states are complete. The fixed 64-query candidate panel and all candidate GT efficacy scoring have not run. These local probes are not complete candidate fits.','',
  'A/N source, inherited state, Omega0 observations, admitted Top1 evidence, pixels, original Native WHEN, noise and configurations remain fixed. P/F use 1792 active parameters; R uses the once-per-query entering-state curvature-trace scale.','',
  '| Dataset | F actual checks accepted / informative | FD tests pass / total | Entire FD contexts pass | AD JVP / VJP | Repeat mismatch |','|---|---:|---:|---:|---|---:|']
 for ds,m in metrics.items():
  v=m['F_response_viability'];lines.append(f"| {ds} | {v['accepted']}/{v['informative_probes']} | {m['FD_pass']}/{m['FD_checks']} | {m['FD_all_pass_contexts']}/{m['FD_contexts']} | {m['JVP_all_pass']} / {m['VJP_all_pass']} | {m['repeat_bitwise_fail']} |")
 lines+=['','## Interpretation','',
  'All matrix rows cover the complete original parameter interface and selected output grid. However, an automatic-differentiation Jacobian of the native backward graph is not automatically the derivative of the complete physical output replay. The unchanged PosDecoder sets each next reference box to `new_pred_boxes.detach()` (original query_decoder.py line 189). FD changes those forward reference values whereas the backward graph cuts their derivative. This is an identified distinction in the implementation; we do not claim this source line alone proves every individual FD discrepancy or exclude activation-boundary effects.','',
  'The complete fixed FD checks and repeat outcomes are reported, including failures. Actual F counterfactual response/drift checks can pass despite this derivative discrepancy. Such local feasibility does not establish a qualified full functional Jacobian, GT retention, error correction, or method performance. We do not change the decoder gradients, frames, precision, thresholds, or cohorts to obtain a pass.','',
  'The attachment identity g=J_O^T h is valid for the *native AD graph* used by OPD. Its J_O cannot simultaneously be assumed to be the FD-consistent full output-function Jacobian when stop-gradient reference paths differ. The promised first-order true functional guarantee therefore has not been established for this implementation.','',
  'A P repeat can differ at float32 last bits when the diagnostic graph includes U rows in one capture and only observed rows in the other. Those repeat failures are retained as technical negatives; they are not omitted or converted into an exact-repeat success. All original complete-A replays are bitwise exact.','',
  'The two earlier state/optimizer storage dispatch mistakes and the initial repeat-abort boundary were preserved in separate recovery namespaces. Exact proposal state reinsertion avoids float32 subtract/add round trips; deep-cloning saved Adam snapshots avoids shared CPU `step` tensor mutation. On-disk original fits, scientific configuration, source model and baseline processes were preserved. The final accepted prefix is exact-rechecked; execution included 8 additional original complete replays/captures before this final bounded qualification (24 actual complete A replays in total), disclosed separately from the 16 accepted ones.','',
  '## Costs and undefined paths','',
  '| Dataset | P median seconds | F median seconds | R median seconds | No-op local probes |','|---|---:|---:|---:|---:|']
 for ds,m in metrics.items():
  c=m['repeat1_no_directional_FD_local_seconds'];lines.append(f"| {ds} | {c['P']:.6f} | {c['F']:.6f} | {c['R']:.6f} | {m['no_op']} |")
 lines+=['','These are real repeat-1 local-proposal costs, excluding repeat-0 directional FD instrumentation. They include each operator\'s Jacobian, solver, counterfactual output checks and CPU arithmetic/recording audits. They are not deployment throughput, full adaptation latency, or a new DINO benchmark. No new DINO call or GT read occurred. Empty evidence keeps the original no-op and unknown denominators remain null.','',
  '## Independent physical derivative readback','',
  'A separate finite root pass made 8 additional fresh captures on the same fixed queries, with 21 original-A state contexts and no optimizer step or new candidate fit. It saved actual plus/minus outputs and live VJP contractions, then independently recomputed all 84 FD checks and 16859136 stored AD-Jacobian coordinates on CPU. Exactly 25/84 FD checks passed, reproducing the qualification failure values. Total actual fresh captures are 32: 16 accepted qualification captures, 8 earlier engineering rechecks, and 8 independent root captures. Total complete original-A replays remain 24; there are no candidate formal fits or GT scores.', '',
  '## Scope and remaining obligations','',
  'Only this fixed numerical/technical qualification is closed. Formal P/F/R 192 fits, candidate GT evaluation, Source/order2 replication, independent new-parent validation and a newly generated online stream are unexecuted. Their conditional gate was not reached. A different complete-function derivative construction would be a material new qualification decision, not a silent fix to this result. This negative result does not reject every projection or functional-protection method.','',
  'P7-A stays frozen; CURRENT_METHOD remains the old Spatial OPD; remaining formal baselines remain cancelled with the same stopped process/optimizer/lease identity. No paper, new teacher, gate, full main run or production promotion was performed. The existing monitor remains ACTIVE because the future method/main protocol is unresolved.','',
  'Primary context: [EATA](https://proceedings.mlr.press/v162/niu22a.html) and [NTK-overlap analysis](https://proceedings.mlr.press/v130/doan21a.html) already discuss parameter protection/projected interference; this qualification claims no novel general projection theory.','',
  'All anonymous local outcomes, FD directions, no-information cases and measured costs are included in QUALIFICATION_ROWS.json, FINITE_DIFFERENCE_ROWS.json and TECHNICAL_STATISTICS.json. Private Jacobians, vectors, boxes, input caches, media, annotations, weights and conversation records are excluded from publication.']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
 write(BASE/'FIGURE_EXPORT.json',dict(status='pass',scope='all actual fixed technical qualification, including failed FD/repeat/no-op/costs',PDF_files={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size,pages=2)},PDFs=1,pages=2,CPU_only=True,GT_read=False,time=time.time()))
 print('REPORT_AND_TWO_PAGES_EXPORTED')

if __name__=='__main__':report()
