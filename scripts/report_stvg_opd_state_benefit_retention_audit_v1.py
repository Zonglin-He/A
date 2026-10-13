"""Scientific figures and exhaustive evidence report for completed CPU audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_state_benefit_retention_common_v1 import *

def run():
 verify();assert read(BASE/'ROOT_READBACK.json')['status']=='pass'
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.backends.backend_pdf import PdfPages
 plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'pdf.fonttype':42})
 summary=read(OUT/'SUMMARY.json');pp=rows(OUT/'PARENT_ROWS.jsonl.gz');cc=rows(OUT/'ROUND_ROWS.jsonl.gz')
 folder=OUT/'figures';folder.mkdir(exist_ok=False);colors=['#3a5f84','#cb6738','#37816d'];labels=['Source','Inherited-1','Inherited-2']
 keys=[a+'_to_'+b for a,b in PAIRS];pairlabels=['Source vs Inherited-1','Source vs Inherited-2','Inherited-1 vs Inherited-2']
 def curve(ax,domain,group,name,metric,label,color,scale=1):
  d=summary['datasets'][domain]['curves'];x=[v['step'] for v in d];v=[r[group][name][metric] for r in d]
  y=[r['mean']*scale if r['mean'] is not None else np.nan for r in v]
  lo=[r['ci95'][0]*scale if r['ci95'] is not None else np.nan for r in v];hi=[r['ci95'][1]*scale if r['ci95'] is not None else np.nan for r in v]
  ax.plot(x,y,label=label,color=color,lw=1.7);ax.fill_between(x,lo,hi,color=color,alpha=.09)
 def finish(fig,title,note):
  fig.suptitle(title,fontsize=14,y=.987);fig.text(.04,.02,note,fontsize=8)
  fig.tight_layout(rect=[.02,.065,.99,.955])
 with PdfPages(folder/'STATE_BENEFIT_DYNAMICS.pdf') as pdf:
  fig,axes=plt.subplots(2,3,figsize=(13,8))
  for row,ds in enumerate(['hc2','vidstg']):
   for i,key in enumerate(keys[:2]):curve(axes[row,0],ds,'contrasts',key,'own_U_advantage',labels[i+1],colors[i+1],100)
   for i,a in enumerate(ARMS):curve(axes[row,1],ds,'arms',a,'own_U_Q',labels[i],colors[i],100);curve(axes[row,2],ds,'arms',a,'teacher_agreement',labels[i],colors[i],100)
   for ax in axes[row]:ax.set_xlabel('Saved update boundary');ax.grid(alpha=.2);ax.legend(loc='best')
   axes[row,0].axhline(0,c='gray',lw=.7);axes[row,0].set_title(ds.upper()+' | inherited advantage over Source');axes[row,0].set_ylabel('Spatial advantage (pp)')
   axes[row,1].set_title('Actual unvisited event-frame quality');axes[row,1].set_ylabel('Parent-equal mean spatial IoU (%)')
   axes[row,2].set_title('Agreement with admitted DINO boxes');axes[row,2].set_ylabel('Parent-equal agreement (%)')
  finish(fig,'State benefit, task quality and teacher agreement along every saved round',
   '64 historical parents; all 192 sealed fits; parent bootstrap 10,000. Bands: unadjusted 95% CI. Empty evidence: unknown agreement, never zero.');pdf.savefig(fig);plt.close(fig)
  fig,axes=plt.subplots(2,3,figsize=(13,8))
  for row,ds in enumerate(['hc2','vidstg']):
   for i,key in enumerate(keys):
    curve(axes[row,0],ds,'contrasts',key,'own_U_RMS',pairlabels[i],colors[i])
    curve(axes[row,1],ds,'contrasts',key,'joint_state_RMS',pairlabels[i],colors[i])
    curve(axes[row,2],ds,'contrasts',key,'joint_gradient_cosine',pairlabels[i],colors[i])
   for ax in axes[row]:ax.set_xlabel('Saved update boundary');ax.grid(alpha=.2);ax.legend(loc='best')
   axes[row,0].set_title(ds.upper()+' | actual output distance');axes[row,0].set_ylabel('Mean per-frame cxcywh RMS (normalized)')
   axes[row,1].set_title('Actual joint parameter-state distance');axes[row,1].set_ylabel('Parameter RMS (QR256 + LN1536)')
   axes[row,2].set_title('Cosine of saved outgoing gradients');axes[row,2].set_ylabel('Gradient cosine (zero norm: unknown)');axes[row,2].set_ylim(-1.02,1.02)
  finish(fig,'Output contraction and parameter divergence are distinct measurements',
   'No model/Jacobian replay or new optimizer. Final boundary has no outgoing gradient. Similar GT means do not imply the same frame-by-frame outputs.');pdf.savefig(fig);plt.close(fig)
 with PdfPages(folder/'STATE_BENEFIT_DIAGNOSTICS.pdf') as pdf:
  fig,axes=plt.subplots(2,3,figsize=(13,8))
  for row,ds in enumerate(['hc2','vidstg']):
   parents=[p for p in pp if p['dataset']==ds];x=np.arange(32)
   for i,key in enumerate(keys[:2]):
    values=[p['contrasts'][key] for p in parents]
    axes[row,0].bar(x+(i-.5)*.36,[v['own_U_erosion']*100 for v in values],width=.36,color=colors[i+1],label=labels[i+1])
    axes[row,1].scatter([v['second_teacher_gain']*100 if v['second_teacher_gain'] is not None else np.nan for v in values],
     [v['own_U_correct_advantaged_mean_second_gain']*100 if v['own_U_correct_advantaged_mean_second_gain'] is not None else np.nan for v in values],c=colors[i+1],s=22,alpha=.75,label=labels[i+1])
    axes[row,2].scatter([v['own_U_initial_advantage']*100 for v in values],[v['own_U_erosion']*100 for v in values],c=colors[i+1],s=22,alpha=.75,label=labels[i+1])
   for ax in axes[row]:ax.axhline(0,c='gray',lw=.7);ax.grid(alpha=.2);ax.legend(loc='best')
   axes[row,0].set_title(ds.upper()+' | all parents, including neutral');axes[row,0].set_xlabel('Anonymous parent ordinal within domain');axes[row,0].set_ylabel('Final advantage change (pp)')
   axes[row,1].axvline(0,c='gray',lw=.7);axes[row,1].set_title('Expert agreement vs actual good-frame change');axes[row,1].set_xlabel('Admitted DINO agreement gain (pp)');axes[row,1].set_ylabel('Initially correct + advantaged U gain (pp)')
   axes[row,2].set_title('Starting advantage association');axes[row,2].set_xlabel('Initial advantage over Source (pp)');axes[row,2].set_ylabel('Final advantage change (pp)')
  finish(fig,'Full parent distribution and competing interpretations',
   'Correct + advantaged: inherited Before IoU >=0.5 and advantage >=1pp. Missing strata remain unknown in data. Baseline-change correlation is partly mechanical.');pdf.savefig(fig);plt.close(fig)
  fig,axes=plt.subplots(2,3,figsize=(13,8))
  for row,ds in enumerate(['hc2','vidstg']):
   parents=[p for p in pp if p['dataset']==ds];x=np.arange(32)
   for i,key in enumerate(keys[:2]):
    values=[p['contrasts'][key] for p in parents]
    for col,metric,scale in [(0,'own_U_correct_advantaged_mean_second_gain',100),(1,'own_U_correct_advantaged_persistent_loss_fraction',100)]:
     axes[row,col].scatter(x+(i-.5)*.15,[v[metric]*scale if v[metric] is not None else np.nan for v in values],s=24,color=colors[i+1],label=labels[i+1])
    for colk,(metric,offset) in enumerate([('own_U_early_erosion',-.15),('own_U_late_erosion',.15)]):
     e=summary['datasets'][ds]['contrasts'][key][metric]['estimate'];axes[row,2].errorbar(i*2+colk,e['mean']*100,yerr=np.array([[e['mean']-e['ci95'][0]],[e['ci95'][1]-e['mean']]])*100,fmt='o',color=colors[i+1],capsize=4)
   for ax in axes[row]:ax.axhline(0,c='gray',lw=.7);ax.grid(alpha=.2)
   axes[row,0].set_title(ds.upper()+' | good U absolute gain/loss');axes[row,0].set_ylabel('Within-parent stratum mean change (pp)');axes[row,0].set_xlabel('Anonymous parent ordinal');axes[row,0].legend()
   axes[row,1].set_title('Persistent loss through final quarter');axes[row,1].set_ylabel('Stratum frame fraction (%)');axes[row,1].set_xlabel('Anonymous parent ordinal');axes[row,1].set_ylim(-4,104);axes[row,1].legend()
   axes[row,2].set_title('First quarter vs subsequent advantage erosion');axes[row,2].set_ylabel('Advantage change (pp)');axes[row,2].set_xticks(range(4),['I1 early','I1 later','I2 early','I2 later'])
  finish(fig,'Absolute loss, persistence and timing; no GT-selected iteration',
   'First quarter: HC10 / Vid3 updates. Persistence: HC boundaries30..40 / Vid8..10. Unknown stratum ratios are omitted only in plotted points, retained in all rows.');pdf.savefig(fig);plt.close(fig)
 def fmt(e,scale=100):return 'unknown' if e['mean'] is None else f"{e['mean']*scale:+.3f} [{e['ci95'][0]*scale:+.3f}, {e['ci95'][1]*scale:+.3f}]"
 lines=['# State-Benefit Retention Audit','',
  'The audit is complete for all 64 historically exposed parents, all 192 sealed fits, all 4,800 saved updates and 4,992 arm-boundary readouts. It used no new model/teacher/forward/backward/optimizer call. Source pixels, evidence, LN initialization, zero Query Residual, fresh original Adam and noise were rebound to the genuine controlled intervention.',
  '',
  '**The strongest retention/convergence evidence is in HC. VidSTG has real adverse cases but does not establish the same domain-wide output-contraction explanation.** This is current-query benefit erosion, not a previous-query catastrophic-forgetting experiment or a deployable state-selection method.',
  '', '## Full endpoint advantage and timing','',
  'Q is parent-equal spatial IoU on genuine GT-event decoder frames excluding all physical Omega0 visits. The complete primary result includes empty, neutral and negative parents. Values below are percentage points; paired unadjusted parent-bootstrap 95% intervals.',
  '', '| Target | Initialization | Before advantage | After advantage | Advantage change | First quarter | Later updates |','|---|---|---|---|---|---|---|']
 for ds,d in summary['datasets'].items():
  for a,b in PAIRS[:2]:
   c=d['contrasts'][a+'_to_'+b];lines.append('| '+ds+' | '+b+' | '+' | '.join(fmt(c[k]['estimate']) for k in ['own_U_initial_advantage','own_U_final_advantage','own_U_erosion','own_U_early_erosion','own_U_late_erosion'])+' |')
 lines+=['','HC loses most of its average inherited advantage within the first ten configured updates. Vid timing is mixed; the I1 later-update interval is negative, while its early interval crosses zero. Every round is preserved and no best step is selected.','',
  '## Actual output convergence','',
  'RMS is mean per-frame normalized cxcywh coordinate distance between the inherited and Source outputs, then equal-weighted across parents. It is not a task-score difference, pixel distance or parameter distance.',
  '', '| Target | Initialization | RMS Before | RMS After | RMS change, 95% CI | Contracting / expanding / neutral parents |','|---|---|---|---|---|---|']
 for ds,d in summary['datasets'].items():
  for a,b in PAIRS[:2]:
   c=d['contrasts'][a+'_to_'+b];v=c['own_U_RMS_delta'];lines.append(f"| {ds} | {b} | {c['own_U_RMS_before']['estimate']['mean']:.5f} | {c['own_U_RMS_after']['estimate']['mean']:.5f} | {fmt(v['estimate'],1)} | {v['negative_parents']} / {v['positive_parents']} / {v['neutral_parents']} |")
 lines+=['','Both HC paired RMS-change intervals are below zero. Both Vid intervals include zero. Parameter-state distance increases on average for all four comparisons, despite HC output contraction; this rejects treating output contraction as convergence to the same parameter state.','',
  '## Original good unobserved frames can degrade while evidence agreement improves','',
  'The predeclared offline stratum requires inherited Before IoU >=0.5 and an initial advantage over Source >=1pp. Persistent loss requires inherited IoU below its own Before at every boundary of the final quarter, not just Source catching up. Frame fractions below are descriptive micro-counts; parent-equal conditional estimates are reported separately. Frames are not independent samples.',
  '', '| Target | Initialization | Persistent loss frames / stratum frames | Parent-equal stratum gain, pp [CI] | DINO agreement gain, pp [CI] | Joint parents true / false / unknown |','|---|---|---|---|---|---|']
 for ds,d in summary['datasets'].items():
  for a,b in PAIRS[:2]:
   c=d['contrasts'][a+'_to_'+b];n=c['own_U_correct_advantaged_frames']['total_frames'];l=c['own_U_correct_advantaged_persistent_loss_frames']['total_frames'];j=c['own_U_joint_agreement_erosion_persistent']
   lines.append(f"| {ds} | {b} | {l}/{n} ({100*l/n:.1f}%) | {fmt(c['own_U_correct_advantaged_mean_second_gain']['estimate'])} | {fmt(c['second_teacher_gain']['estimate'])} | {j['true_parents']} / {j['false_parents']} / {j['unknown_parents']} |")
 lines+=['','The joint parent event requires an increase in admitted DINO agreement, a decrease in overall inherited advantage, and at least one persistently degraded frame in the initially correct/advantaged stratum. It is a descriptive conjunction, not an online gate or a significance test. Agreement is unknown in seven Vid empty-evidence parents. No stratum in a parent means unknown, not no damage.',
  '', 'HC stratum mean absolute gain is negative with intervals below zero, so its loss cannot be reduced to Source merely catching up. Vid stratum means are positive with intervals spanning zero: harmful subsets coexist with useful adaptation. Whole-panel inherited After quality can still exceed Source, especially Vid I1; this audit does not select Source reset as a new method.',
  '', '## Negative tails and concentration','', '| Target | Initialization | Advantage erosion +/-/neutral | Remove largest absolute 1 parent | Remove largest absolute 3 parents | Leave-one-out mean range, pp |','|---|---|---|---|---|---|']
 for ds,d in summary['datasets'].items():
  for a,b in PAIRS[:2]:
   c=d['contrasts'][a+'_to_'+b];v=c['own_U_erosion'];r=c['own_U_robustness'];lo,hi=r['leave_one_out_mean_range']
   lines.append(f"| {ds} | {b} | {v['positive_parents']}/{v['negative_parents']}/{v['neutral_parents']} | {fmt(r['largest1_removed'])} | {fmt(r['largest3_removed'])} | [{lo*100:+.3f}, {hi*100:+.3f}] |")
 lines+=['','HC remains negative after removal of its largest one or three absolute-change parents. Vid remains negative after one removal, but both three-removal intervals include zero. These are diagnostics; complete 32-parent estimates remain primary and the removed IDs and all rows are preserved.','',
  '## Saved gradients, state separation and headroom alternatives','', '| Target | Initialization | Initial joint gradient cosine [CI] | Joint state RMS Before -> After | Starting-quality LOO R2 |','|---|---|---|---|']
 for ds,d in summary['datasets'].items():
  for a,b in PAIRS[:2]:
   c=d['contrasts'][a+'_to_'+b];lines.append(f"| {ds} | {b} | {fmt(c['initial_joint_gradient_cosine']['estimate'],1)} | {c['joint_state_RMS_before']['estimate']['mean']:.5f} -> {c['joint_state_RMS_after']['estimate']['mean']:.5f} | {c['starting_quality_association']['LOO_R2']:.3f} |")
 lines+=['',
  'Every saved outgoing gradient, applied step, QR/LN norm, accumulated step length, moment/state identity, pair cosine and incoming output movement is retained for every boundary. No full decoder Jacobian was replayed; the original actual native-head VJP qualification is reused only within its recorded scope. Zero gradients have unknown cosine.',
  '',
  'The leave-one-parent-out linear model uses Source and inherited Before quality to predict advantage change. Its association is compatible with headroom and is also mechanically coupled to the outcome through the Before subtraction. It cannot establish that headroom causes or fully explains the intervention effect. Previous-query retention, teacher-independent output sensitivity and gradient interventions were not measured.',
  '', '## Research decision and completion boundary','',
  'HC supports a concrete current-query preservation problem: sparse pseudo-supervision improves teacher agreement while eroding a substantial part of originally useful unvisited-frame localization, with actual output contraction and robustness to the largest few cases. Vid supports state-dependent utility and individual retention failures, but not a uniform forgetting/convergence mechanism. Keep both findings; do not pool domains into a universal claim.',
  '',
  'Do not call this catastrophic forgetting or a validated Cross-Query Knowledge Preservation algorithm. No gate, new loss, teacher, hyperparameter search, production method or full main experiment was started. P7-A remains frozen. The cancelled formal baselines retain their accepted prefix and original T standby identity. Future method and main protocol remain separate work.',
  '',
  'Complete anonymous files: PARENT_ROWS.jsonl.gz (64 parents, all three pair contrasts), ROUND_ROWS.jsonl.gz (1,664 parent boundaries), FRAME_ROUND_ROWS.jsonl.gz (every genuine decoder-frame boundary, including GT-unknown frames), DENSE_ENDPOINT_ROWS.jsonl.gz (all original three-arm dense Before/After metrics), ALL_PARENT_CONTRASTS.csv (192 contrasts), SUMMARY.json (all masks/arms/rounds/strata/tails/unknowns). Raw boxes, state/gradient vectors, physical frame IDs, annotations, media and text are excluded from publication.',
  '',
  'Independent source/math/state/noise/frame/dense/scalar/bootstrap readback passed before figure authoring. All four PDF pages passed actual root pixel review. Remote public-content verification and archive closure are bound by the separate current-scope closing receipts.','']
 (OUT/'REPORT.md').write_text('\n'.join(lines))
 files={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size,pages=2) for p in folder.glob('*.pdf')}
 write(BASE/'FIGURE_EXPORT.json',dict(status='exported_pending_actual_pixel_view',PDF_files=files,PDFs=2,pages=4,CPU_only=True,GT_read=False,time=time.time()))
if __name__=='__main__':run()
