"""Source-bound report and two-page scientific figure for the complete finite study."""
import sys,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_instance_identity_common_v1 import BASE,read,write,sha
NAMES={'F':'Frozen','T':'Temporal QA','T_strict_iou':'QA + strict-old IoU OPD','T_raw_iou':'QA + decoupled IoU OPD','T_supported_iou':'QA + presence admission','T_loss_weighted_iou':'QA + loss-weighted OPD','T_validated_loss_weighted_iou':'QA + validated OPD','T_geometry_strict':'QA + strict geometry','T_geometry_raw':'QA + decoupled geometry'}
def alias(cid):return ('HC' if cid.startswith('hc2')else'V')+f'{int(cid[-3:])+1:02d}'
def main():
 s=read(BASE/'POSTSEAL_SUMMARY.json');rows=read(BASE/'POSTSEAL_ROWS.json')['rows'];counts=read(BASE/'EXECUTION_COUNTS.json');lookup={(r['dataset'],r['arm'],r['metric']):r['mean'] for r in s['summary']};by={(r['case_id'],r['arm']):r['metrics']for r in rows}
 val=lambda ds,arm,k='m_vIoU':100*lookup[ds,arm,k]
 lines=['# WHO/WHEN separation and validated Spatial OPD: eight exposed development sources','','Measured finite experiment, 2026-10-14. Both branches retained. No CURRENT promotion or independent/full-data expansion.','',
 'All eight method outputs were sealed before this round\'s annotation evaluation. Historical development exposure is disclosed. No new YOLO, SAM2, R1 or DINO execution; no GT Oracle fit. Existing teacher selections, masks, QA intervals, source checkpoints, four centers and the original five-step/lr/sigma/tau configuration were fixed.','',
 '## Matched full-tube results','','Equal source means, four sources per direction. Values are percentages. Teacher geometry uses the identical QA interval and full Student grid; absent masks fall back to the exact Frozen box without interpolation.','',
 '| Method | HC vIoU | VidSTG vIoU | HC dense sIoU | VidSTG dense sIoU |','|---|---:|---:|---:|---:|']
 for arm,label in NAMES.items():lines.append(f'| {label} | {val("hc2",arm):.3f} | {val("vidstg",arm):.3f} | {val("hc2",arm,"m_sIoU_dense_GT"):.3f} | {val("vidstg",arm,"m_sIoU_dense_GT"):.3f} |')
 lines +=['','## Controlled findings','',
 'WHO/WHEN separation restored four anchors each for V01 and V02, despite their preserved out-of-clock native time status. The selected IDs were present in the sealed mask inventory and QA interval. This is structural admissibility, not an independent semantic-correctness guarantee. V03 still has no mask at its fixed centers and remains exact no-update. Total anchors changed from 14 to 22.','',
 'The saved HC01 ordinary OPD trajectory fit teacher anchors better for three steps (mean IoU 0.394574 -> 0.467715 -> 0.505522 -> 0.551928), then collapsed to 0.152761 at step four. Three anchors became zero-IoU. The new no-label rule rejected that fourth proposal and restored parameters AND Adam, leaving three accepted steps. Final full-model re-decoding exactly matched the accepted cached decoder.','',
 'The rejection rule tests weighted teacher IoU non-decrease (tolerance 1e-7), stops on first rejected proposal, and never reads GT or retries a different learning rate. It uses the same imperfect supervision and cannot ensure referent correctness or unseen-frame improvement. HC04 still improves reference fit while harming task performance.','',
 'The loss-level weight is now mean(r * per-anchor original KL); reward remains IoU with the original softmax temperature. Here 21 of 22 anchors have r=1, the remaining r=0.8 anchor has flat/no-information IoU reward. Presence admission excludes no additional anchors. Therefore this panel cannot establish an advantage for continuous reliability weighting; its results are exactly equal to ordinary decoupled OPD.','',
 '| Case | Strict OPD - QA (pp) | Decoupled OPD - QA (pp) | Validated OPD - QA (pp) | Geometry - QA (pp) |','|---|---:|---:|---:|---:|']
 for cid in sorted({r['case_id']for r in rows}):
  delta=lambda arm:100*(by[cid,arm]['m_vIoU']-by[cid,'T']['m_vIoU'])
  lines.append(f'| {alias(cid)} | {delta("T_strict_iou"):.3f} | {delta("T_raw_iou"):.3f} | {delta("T_validated_loss_weighted_iou"):.3f} | {delta("T_geometry_raw"):.3f} |')
 lines+=['','## Unsupervised support and uncertainty','',
 'Supervised positions are the fixed raw-decoupled reference anchors for every comparison; unsupervised support includes all other known GT physical frames, including off-grid frames. Missing GT support stays unknown rather than zero. These source-mean metrics are distinct from sparse anchor agreement.','']
 for ds in ['hc2','vidstg']:
  c=next(r for r in s['contrasts']if r['dataset']==ds and r['left']=='T_validated_loss_weighted_iou'and r['right']=='T'and r['metric']=='m_vIoU');ci=c['paired_bootstrap_95_pp']
  lines.append(f'- {ds}: validated OPD - QA {c["mean_delta_pp"]:+.3f} pp; {c["positive"]} positive / {c["negative"]} negative / {c["neutral"]} neutral. Paired source bootstrap 95% [{ci["lower"]:+.3f}, {ci["upper"]:+.3f}] pp (n=4). Unsupervised dense sIoU: QA {val(ds,"T","m_sIoU_unsupervised_GT"):.3f}%, validated {val(ds,"T_validated_loss_weighted_iou","m_sIoU_unsupervised_GT"):.3f}%.')
 lines +=['','Geometry-only remains ahead of validated Student in both domains: this round does not establish adaptation superiority over direct expert geometry. Dense masks are used at available grid frames by the geometry-only arm, whereas Student updates use only the fixed sparse anchors; the comparison measures what sparse adaptation retains, not equal spatial-supervision density.','',
 '## Execution and decision','',f'Actual new Student fit paths: {counts["actual_fit_count"]}; Adam proposals: {counts["proposed_optimizer_steps"]}; accepted: {counts["accepted_optimizer_steps"]}; rejected: 1; full-model final re-decodes: {counts["final_model_redecodes"]}; source captures: 8. Two paths have flat reward and zero optimizer updates. Identical old/control arms were reused exactly and do not count as new fits. All eight workers exited normally under the own-unit resource guard.','',
 'No new tracking/teacher calls does not imply zero deployment expert cost. Prior video-shared YOLO/SAM2 and per-query R1 costs remain real; this round reports only incremental source capture and Student fit/re-decode cost.','',
 'Decision: the finite diagnostic is complete. Identity/time separation and one reference-agreement rejection rule have useful development evidence. HC04 negative transfer, HC unsupervised degradation, and the Teacher-geometry gap remain. Keep both branches as research candidates; do not promote CURRENT, claim stable generalization, or automatically launch independent32/full-data from these eight exposed sources. No further tuning was performed after evaluation.','',
 'Full anonymous per-case metrics, neutral/unknown support, all declared arm summaries, paired contrasts, saved-step diagnostics, actual proposal/acceptance records and source bindings accompany this report.']
 (ROOT/'docs/STVG_INSTANCE_IDENTITY_UPDATE_REPORT.md').write_text('\n'.join(lines)+'\n')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.backends.backend_pdf import PdfPages
 import numpy as np
 figdir=BASE/'figures';figdir.mkdir(exist_ok=True)
 with PdfPages(figdir/'instance_identity_update.pdf')as pdf:
  fig=plt.figure(figsize=(11.7,8.3));fig.suptitle('Instance identity and spatial update diagnosis',fontsize=19,y=.95)
  ax=fig.add_axes([.05,.23,.90,.60]);ax.axis('off')
  table=ax.table(cellText=[[label,f'{val("hc2",arm):.3f}',f'{val("vidstg",arm):.3f}',f'{val("hc2",arm,"m_sIoU_dense_GT"):.3f}',f'{val("vidstg",arm,"m_sIoU_dense_GT"):.3f}']for arm,label in NAMES.items()],colLabels=['Matched arm','HC vIoU','Vid vIoU','HC sIoU','Vid sIoU'],colWidths=[.44,.14,.14,.14,.14],loc='center',cellLoc='center');table.auto_set_font_size(False);table.set_fontsize(11);table.scale(1,2)
  for (i,j),cell in table.get_celld().items():
   cell.set_edgecolor('#d1d7de')
   if i==0:cell.set_facecolor('#243b53');cell.set_text_props(color='white',weight='bold')
   elif i%2==0:cell.set_facecolor('#edf3f8')
  fig.text(.06,.84,'Four exposed development sources per direction | all predictions sealed before new GT scoring',fontsize=11)
  fig.text(.06,.18,'WHO/WHEN separation: V01 and V02 each recover four real mask anchors.',fontsize=12)
  fig.text(.06,.14,'Validated OPD: HC +2.315 pp; VidSTG +9.457 pp relative to Temporal QA.',fontsize=12)
  fig.text(.06,.10,'Geometry-only still leads; HC04 remains negative. Reliability weights lack variation in this panel.',fontsize=11)
  fig.text(.06,.055,'Research candidate only. No production promotion or independent/full-data expansion.',fontsize=10,color='#555555');pdf.savefig(fig);plt.close(fig)
  fig,axes=plt.subplots(1,2,figsize=(11.7,8.3));fig.subplots_adjust(left=.08,right=.97,top=.79,bottom=.25,wspace=.30);fig.suptitle('Mechanism evidence and all eight outcomes',fontsize=18,y=.94)
  y=[.3945738129,.4677151031,.5055215750,.5519277620,.1527611551,.1564062263];axes[0].plot(range(6),y,'o-',label='Original five-step path',color='#d08c30');axes[0].plot(range(4),y[:4],'s-',label='Accepted validation path',color='#2b708e');axes[0].axvline(3.5,ls='--',color='#a64b4b');axes[0].set(xlabel='Accepted / proposed update index',ylabel='Mean IoU with teacher reference',title='HC01: reject destructive fourth proposal',xticks=range(6),ylim=(0,.65));axes[0].legend(fontsize=9,loc='lower left');axes[0].grid(alpha=.2)
  ids=sorted({r['case_id']for r in rows});xs=np.arange(8);delta=lambda cid,arm:100*(by[cid,arm]['m_vIoU']-by[cid,'T']['m_vIoU'])
  axes[1].barh(xs-.15,[delta(c,'T_raw_iou')for c in ids],height=.3,label='Decoupled OPD',color='#d08c30');axes[1].barh(xs+.15,[delta(c,'T_validated_loss_weighted_iou')for c in ids],height=.3,label='Validated OPD',color='#2b708e');axes[1].set(yticks=xs,yticklabels=[alias(c)for c in ids],xlabel='Full-tube vIoU change vs QA (pp)',title='Actual final Student re-decodes');axes[1].invert_yaxis();axes[1].axvline(0,color='#777777',lw=.8);axes[1].legend(fontsize=9);axes[1].grid(axis='x',alpha=.2)
  fig.text(.08,.16,'Reference agreement is label-free; it is not a semantic-correctness or unseen-frame guarantee.',fontsize=11)
  fig.text(.08,.12,'HC02 and V04 gains are retained. V01/V02 benefit from identity admission; HC04 still declines.',fontsize=11)
  fig.text(.08,.07,'16 new fit paths | 69 Adam proposals | 68 accepted | 1 rejected | 16 final full-model re-decodes',fontsize=10);pdf.savefig(fig);plt.close(fig)
 write(BASE/'FIGURE_EXPORT.json',dict(status='complete',pages=2,pdf_sha256=sha(figdir/'instance_identity_update.pdf'),report_sha256=sha(ROOT/'docs/STVG_INSTANCE_IDENTITY_UPDATE_REPORT.md')))
if __name__=='__main__':main()
