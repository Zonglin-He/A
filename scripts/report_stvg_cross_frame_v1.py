"""Complete finite comparison: preserve all declared arms and exposed-development limits."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_cross_frame_common_v1 import BASE,read,write,sha
NAMES={'F':'Frozen','T':'Temporal QA','S_original':'Original Spatial','TS_original':'Temporal + Original Spatial','S_A':'Spatial A (train validation)','TS_A':'Temporal + A [frozen recipe]','S_B':'Spatial B (cross-frame)','TS_B':'Temporal + B','T_fast':'Two-candidate Temporal','TS_fast':'Two-candidate Temporal + A','T_geometry':'Temporal + Teacher Geometry','T_fast_geometry':'Two-candidate + Geometry'}
def alias(cid):return ('HC' if cid.startswith('hc2')else'V')+f'{int(cid[-3:])+1:02d}'
def main():
 s=read(BASE/'POSTSEAL_SUMMARY.json');rows=read(BASE/'POSTSEAL_ROWS.json')['rows'];lookup={(r['dataset'],r['arm'],r['metric']):r['mean']for r in s['summary']};by={(r['case_id'],r['arm']):r['metrics']for r in rows}
 val=lambda ds,arm,k='m_vIoU':100*lookup[ds,arm,k]
 lines=['# Final Cross-Frame Validation and recipe freeze','', '2026-10-15. Eight historically exposed development sources, four per direction. This finite mechanism comparison is distinct from the authorized full Table 1 evaluation.','',
 '## Selection fixed before new annotation scoring','',
 'A is the current train-reference validated Gaussian IoU OPD. B additionally rejects a proposal if agreement at fixed non-training SAM2 frames worsens. The original 22 training anchors, source checkpoints, five-step budget, reward, sampling and optimizer are unchanged. No new detector, tracker, R1 or GT-Oracle fitting was run.','',
 'Validation positions were locked before visibility inspection: at most eight integer-quantile source-grid positions inside the QA interval, excluding training frames. Of 55 candidates, 35 had visible selected-ID masks and 20 remained unknown. Five cases had at least two valid positions; three fell back to A. The active validation set contains 33 positions; the two single-mask cases are excluded by the predeclared fallback rule.','',
 'B was eligible only if both domains retained informative accepted updates and their paired mean non-training Teacher IoU was no worse than A (tolerance 1e-7). HC had two evaluable sources and B-A = -0.000159379795; Vid had three and +0.001591818458. The global rule therefore selected A before new GT was accessed. This tests propagation agreement with the same imperfect teacher, not correct semantic identity.','',
 '## Complete full-tube comparison','',
 'Equal source means; values in percent. Labels were read only after every new prediction and the A/B choice were sealed. Historical development exposure remains.','',
 '| Arm | HC vIoU | Vid vIoU | HC dense sIoU | Vid dense sIoU |','|---|---:|---:|---:|---:|']
 for arm,label in NAMES.items():lines.append(f'| {label} | {val("hc2",arm):.3f} | {val("vidstg",arm):.3f} | {val("hc2",arm,"m_sIoU_dense_GT"):.3f} | {val("vidstg",arm,"m_sIoU_dense_GT"):.3f} |')
 lines+=['','B has higher post-seal development vIoU than A (+0.674 pp HC, +0.609 pp Vid). That positive observation is retained, but does not retrospectively change the label-free selection rule. Teacher Geometry-only remains stronger than either Student version; dense teacher geometry and four-anchor adaptation have different supervision densities.','',
 '| Case | A - Temporal (pp) | B - Temporal (pp) | B - A (pp) | Fast - A (pp) |','|---|---:|---:|---:|---:|']
 ids=sorted({r['case_id']for r in rows})
 for cid in ids:
  d=lambda a,b:100*(by[cid,a]['m_vIoU']-by[cid,b]['m_vIoU'])
  lines.append(f'| {alias(cid)} | {d("TS_A","T"):+.3f} | {d("TS_B","T"):+.3f} | {d("TS_B","TS_A"):+.3f} | {d("TS_fast","TS_A"):+.3f} |')
 lines+=['','## One bounded speed variant','',
 'The only method-changing speed trial removed expanded/contracted temporal candidates, retaining Native and Projection. Eight full-head replays first reproduced the original four-candidate parameters and intervals exactly, then eight actual two-candidate CPU Adam fits were sealed. No new teacher generation was needed because the exact original responses were cached.','',
 'The predeclared allowed loss was 1 percentage point per domain. The complete two-branch result lost 6.357 pp HC and 5.612 pp Vid, so this deletion was rejected. Historical request-count/component-time savings are not a fresh end-to-end throughput measurement. No subsequent prompt, learning-rate, radius or temperature search is authorized by this result.','',
 '## Execution and verification','',
 'The actual B worker loaded the two source models once each, ran seven source captures, seven fits and seven final full-model re-decodes in 54.499 seconds. V03 reused its sealed no-information/no-update output and is not counted as a new capture or fit. All eight output receipts were independently checked for source, frames, pixel binding, bytes and hashes. The worker exited normally under its own RAM/GPU/disk guard. This incremental Student time excludes all earlier expert work and is not Table 1 cost.','',
 'Independent endpoint geometry, parameter/Adam rollback, cached-to-full readout, official/independent metric agreement and summary aggregation were checked. Saved mathematical readback does not claim an additional independent GPU backward. All 96 case-arm rows, 120 metric summaries, paired parent-bootstrap contrasts, negative outcomes, neutral outcomes and unknown support are exported.','',
 '## Frozen next-stage method and scope','',
 'Use one episodic recipe for both domains: original four-candidate Caption-Query Alignment with 512-parameter Adam, official frozen YOLOv12-X/SAM2.1/R1 identity evidence, WHO/WHEN separation, and A: 1792-parameter Gaussian IoU OPD with training-reference rejection and full parameter/Adam rollback. CURRENT is unchanged. No new algorithm search follows this finite comparison.','',
 'Next execution is engineering qualification of model residency, exact per-video cache reuse, recovery and disk growth, followed by 32 fixed non-current-development parents per direction and the authorized 13,785-query official evaluation. HC2 is validation, VidSTG is test. Those phases are not completed by this report. Every official query prediction must be sealed before new official-label evaluation; no official labels select the frozen recipe. Old stopped queues are not resumed.','',
 'STVG-R1 external training overlap with target evaluation remains incompletely established. Describe the method as frozen external-expert-guided TTA; do not claim no target-domain teacher prior.']
 (ROOT/'docs/STVG_CROSS_FRAME_VALIDATION_REPORT.md').write_text('\n'.join(lines)+'\n')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.backends.backend_pdf import PdfPages
 import numpy as np
 out=BASE/'figures';out.mkdir(exist_ok=True)
 with PdfPages(out/'cross_frame_validation.pdf')as pdf:
  fig=plt.figure(figsize=(11.7,8.3));fig.suptitle('Last mechanism comparison and frozen recipe',fontsize=18,y=.95)
  ax=fig.add_axes([.04,.20,.92,.67]);ax.axis('off')
  tab=ax.table(cellText=[[label,f'{val("hc2",arm):.3f}',f'{val("vidstg",arm):.3f}']for arm,label in NAMES.items()],colLabels=['Declared arm','HC vIoU (%)','Vid vIoU (%)'],colWidths=[.62,.19,.19],loc='center');tab.auto_set_font_size(False);tab.set_fontsize(10);tab.scale(1,1.85)
  for (i,j),cell in tab.get_celld().items():
   cell.set_edgecolor('#d1d7de')
   if i==0:cell.set_facecolor('#243b53');cell.set_text_props(color='white',weight='bold')
   elif i%2==0:cell.set_facecolor('#edf3f8')
  fig.text(.06,.15,'A selected before new GT: B failed HC teacher-reference non-degradation.',fontsize=11)
  fig.text(.06,.11,'B has higher development vIoU; selection is not changed after scoring.',fontsize=11)
  fig.text(.06,.06,'Eight exposed development sources. Full official Table 1 remains a separate execution phase.',fontsize=10);pdf.savefig(fig);plt.close(fig)
  fig,axs=plt.subplots(1,2,figsize=(11.7,8.3));fig.subplots_adjust(left=.08,right=.97,top=.82,bottom=.22,wspace=.3);fig.suptitle('All eight cases: gains and losses retained',fontsize=18,y=.95)
  x=np.arange(8)
  for a,c,off in [('TS_A','#2b708e',-.18),('TS_B','#d08c30',.18)]:axs[0].barh(x+off,[100*(by[i,a]['m_vIoU']-by[i,'T']['m_vIoU'])for i in ids],height=.36,label=a,color=c)
  axs[1].barh(x,[100*(by[i,'TS_fast']['m_vIoU']-by[i,'TS_A']['m_vIoU'])for i in ids],color='#b35151')
  for ax in axs:ax.set_yticks(x,[alias(i)for i in ids]);ax.invert_yaxis();ax.axvline(0,color='#777',lw=.8);ax.grid(axis='x',alpha=.2)
  axs[0].set_title('Spatial gain over Temporal');axs[0].set_xlabel('Full-tube delta vIoU (pp)');axs[0].legend()
  axs[1].set_title('Rejected two-candidate Temporal');axs[1].set_xlabel('Full-tube delta vs frozen A recipe (pp)')
  fig.text(.08,.14,'Cross-frame agreement does not certify semantic identity or generalization.',fontsize=11)
  fig.text(.08,.09,'Fast deletion loses 6.36 / 5.61 pp: retain four candidates; optimize exact execution instead.',fontsize=11);pdf.savefig(fig);plt.close(fig)
 write(BASE/'FIGURE_EXPORT.json',dict(status='complete',pages=2,pdf_sha256=sha(out/'cross_frame_validation.pdf'),report_sha256=sha(ROOT/'docs/STVG_CROSS_FRAME_VALIDATION_REPORT.md')))
if __name__=='__main__':main()
