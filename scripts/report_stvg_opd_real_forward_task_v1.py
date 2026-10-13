"""Complete-task contrast, protection/correction/control and all negative/unknown outcomes."""
import os,time,collections
os.environ['CUDA_VISIBLE_DEVICES']=''
from scripts.stvg_opd_real_forward_task_common_v1 import *
from scripts.score_stvg_opd_real_forward_task_v1 import rows_read,FIELDS

def fmt(e):
 if e['mean'] is None:return 'unknown'
 lo,hi=e['ci95'];return f"{100*e['mean']:+.4f} [{100*lo:+.4f},{100*hi:+.4f}]; n={e['defined_parents']}"

def report():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.backends.backend_pdf import PdfPages
 assert read(BASE/'ROOT_READBACK.json')['status']=='pass';verify()
 s=read(OUT/'SUMMARY.json');rr=rows_read(OUT/'ROWS.jsonl.gz');rd=rows_read(OUT/'ROUND_ROWS.jsonl.gz')
 lines=['# Real-Forward: complete committed-state task validation','',
 'Actual fixed64 historical queries/Inherited-order1/Omega0, four complete arms:64 original-A control replays and192 new SEL/HALF/LN fits,6400 saved rounds. Qualification separately32 complete fits/800 rounds/eight fresh captures. All256 formal predictions were sealed before GT scoring. Every A control exactly reproduces its old complete fit. No new DINO, teacher, hyperparameter search, main run or production promotion.','',
 'Every step recomputes OPD gradient and Adam from the actual state committed by that arm. SEL uses the unchanged A/LN/Query/Half, rho=.9, fixed tie order and original A for nonpositive G(A). HALF/LN are proposal masks with all1792 Adam moments retained, not optimizer-block removal. Admitted observation evidence and actual noise are common; sampled actions diverge after states diverge.','',
 'Primary contrast SEL-A complete mvIoU; source-equal32 parents per target, paired10000 bootstrap seed20261013. All intervals are conditional on these historically exposed states; secondary intervals are exploratory and unadjusted. This is episodic receiving-state adaptation, not a new prequential stream.','',
 '|Target|Arm|Full mvIoU gain versus entering (pp,95%CI)|Observed GT learning|Unvisited correct-frame IoU gain|Unvisited wrong-frame IoU gain|','|---|---|---|---|---|---|']
 for ds,v in s['datasets'].items():
  for a in ARMS:lines.append('|'+ds+'|'+a+'|'+'|'.join(fmt(v[f]['arms'][a]['estimate']) for f in ['dense_v_delta','observed_delta','correct_U_delta','wrong_U_delta'])+'|')
 lines+=['','|Target|Contrast|Full mvIoU (pp,95%CI)|Positive/negative/neutral/unknown parents|>5/>20pp harms|','|---|---|---|---|---|']
 for ds,v in s['datasets'].items():
  for key,x in v['dense_v_delta']['contrasts'].items():lines.append(f"|{ds}|{key}|{fmt(x['estimate'])}|{x['positive']}/{x['negative']}/{x['neutral']}/{x['unknown']}|{x['harm5pp']}/{x['harm20pp']}|")
 lines+=['','|Target|Arm|Retention of initially correct unvisited frames (%)|Correction of initially wrong unvisited frames (%)|Expert agreement gain pp|Mean instrumented fit seconds|','|---|---|---|---|---|---|']
 for ds,v in s['datasets'].items():
  for a in ARMS:lines.append('|'+ds+'|'+a+'|'+'|'.join(fmt(v[f]['arms'][a]['estimate']) for f in ['correct_U_retained','wrong_U_corrected','expert_delta'])+f"|{v['compute'][a]['instrumented_fit_seconds_mean']:.6f}|")
 lines+=['','## Evidence and decision boundaries','',
 'Initially correct means entering real-event-frame IoU>=.5; initially wrong means <.5. Masks use the shared entering state and are defined only offline after the global seal. Real sampled GT frames exclude visited Omega0 for retention/correction; unsupported/interpolated frames are not imputed into these partitions. The full native interval is fixed across arms, so complete tIoU differences are zero. Unknown frame partitions remain unknown.','',
 'Protection, correction and task quality must be assessed together. A smaller drift or higher expert agreement does not prove correctness. Positive historical effects can justify specifying an independent replication; they do not certify a deployed method. Equivalence to HALF/LN requires a separate predeclared equivalence margin, not a confidence interval crossing zero. Simple-control comparisons are retained without a requirement to beat Capped Direct before this test.','',
 'Actual cost is instrumented complete-fit time including original native likelihood/action/gradient recording and independent arithmetic; it excludes frozen backbone capture, old expert generation and serialization. SEL additionally performs four actual complete-grid candidate decoder forwards per informative step. This is not an end-to-end deployment benchmark and does not imply zero fresh expert cost. Every parent timing, selected/fallback round and negative case is published.','',
 'The old local screen remains 4/9 below its locked50% gate. This is a separately authorized full-task test, not a retrospective screen pass. High-dimensional P/F numerical failures, ordinary R without a task test, negative-evidence scale confounds, Q0 candidate/mean and P9 weak-predictor uncertainties remain in EVIDENCE_SCOPE_CORRECTION.md. No other experiment was reopened.','',
 'CURRENT unchanged; cancelled-baseline controller/worker remain their original T standby identities, with accepted prefix and optimizer custody preserved. The user-deleted monitor stays deleted. No automatic candidate expansion, parameter search, order2, new teacher, paper or full-main launch.']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
 folder=OUT/'figures';folder.mkdir(exist_ok=True);p=folder/'REAL_FORWARD_TASK_VALIDATION.pdf';assert not p.exists()
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans','pdf.fonttype':42})
 with PdfPages(p) as pdf:
  for page,fields,title in [(0,['dense_v_delta','own_U_delta'],'Complete task and actual unvisited-frame learning'),(1,['correct_U_delta','wrong_U_delta'],'Protection and correction at the common entering state')]:
   fig,axes=plt.subplots(2,2,figsize=(12,8))
   for j,ds in enumerate(['hc2','vidstg']):
    v=s['datasets'][ds]
    for i,fld in enumerate(fields):
     ax=axes[i,j]
     for x,key in enumerate(['SEL_minus_A','SEL_minus_HALF','SEL_minus_LN']):
      e=v[fld]['contrasts'][key]['estimate']
      if e['mean'] is not None:
       ax.errorbar(x,100*e['mean'],yerr=[[100*(e['mean']-e['ci95'][0])],[100*(e['ci95'][1]-e['mean'])]],fmt='o',capsize=5)
     ax.axhline(0,color='gray',lw=1);ax.set_xticks(range(3),['SEL - A','SEL - HALF','SEL - LN']);ax.set(title=ds+' / '+fld,ylabel='Paired parent difference (pp,95% CI)')
   fig.suptitle(title,fontsize=16);fig.text(.5,.018,'32 historical parents per target; paired bootstrap10000. Unknown masks excluded with denominators retained.',ha='center');fig.tight_layout(rect=(0,.045,1,.94));pdf.savefig(fig);plt.close(fig)
  fig,axes=plt.subplots(2,2,figsize=(12,8))
  for j,ds in enumerate(['hc2','vidstg']):
   rows=[r for r in rr if r['dataset']==ds];deltas=[100*(r['SEL_dense_v_delta']-r['A_dense_v_delta']) for r in rows]
   axes[0,j].bar(range(32),deltas,color=['#3b8c74' if v>=0 else '#b85243' for v in deltas]);axes[0,j].set(title=ds+' / every parent SEL-A',xlabel='Fixed anonymous parent ordinal',ylabel='Full mvIoU difference (pp)');axes[0,j].axhline(0,color='gray',lw=1)
   v=s['datasets'][ds];axes[1,j].bar(ARMS,[v['compute'][a]['instrumented_fit_seconds_mean'] for a in ARMS],color=['#8e969b','#397b93','#d58e35','#8c73a1']);axes[1,j].set(title=ds+' / actual instrumented complete fit',ylabel='Mean seconds, including audit instrumentation')
  fig.suptitle('All tails and actual complete-fit costs',fontsize=16);fig.text(.5,.018,'No parent removed. Capture/expert generation/serialization excluded; four SEL candidates are real decoder outputs.',ha='center');fig.tight_layout(rect=(0,.045,1,.94));pdf.savefig(fig);plt.close(fig)
 write(BASE/'FIGURE_EXPORT.json',dict(status='pass',PDF_files={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size,pages=3)},PDFs=1,pages=3,CPU_only=True,GT_read=True,time=time.time()))
 print('TASK_REPORT_EXPORTED_THREE_PAGES')
if __name__=='__main__':report()
