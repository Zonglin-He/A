"""Complete finite phase1 readout; anonymous scalar exports and vector PDF."""
import json,hashlib,statistics
from pathlib import Path
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
R=Path(__file__).resolve().parents[1];B=R/'artifacts/stvg_spatial_causal_decomposition_v1';O=R/'results/stvg_spatial_causal_decomposition/2026-10-14';O.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads(p.read_text());d=read(B/'phase1/ROOT_AMPLITUDE_READBACK.json');pool=read(B/'POOL_TEACHER_DIAGNOSIS.json');oracle=read(B/'OFFLINE_ORACLE_CUSTODY_READBACK.json')
def alias(cid):return('HC'if cid.startswith('hc2')else'V')+f'{int(cid[-3:])+1:02d}'
def clean(v):
 if isinstance(v,dict):return{k:clean(x)for k,x in v.items()if k not in ['GT_source_bindings','old_temporal_output_path']}
 if isinstance(v,list):return[clean(x)for x in v]
 if isinstance(v,str):
  for cid in [f'{s}_{i:03d}'for s in ['hc2','vidstg']for i in range(8)]:v=v.replace(cid,alias(cid))
 return v
for name,value in [('phase1_all_metrics',d),('pool_teacher_all_diagnostics',pool),('offline_oracle_custody',oracle)]:
 (O/(name+'.json')).write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n')
rows=d['rows'];table=['| Case | 0.25 | 0.5 | 1 |','|---|---:|---:|---:|']
for cid in sorted({r['case_id']for r in rows}):
 vals=[next(r['baseline_delta_vIoU_pp']for r in rows if r['case_id']==cid and r['alpha']==a and r['branch']=='S')for a in [.25,.5,1]];table.append('| '+alias(cid)+' | '+' | '.join(f'{x:+.6f}'for x in vals)+' |')
groups=['| Cohort | Domain | Branch | alpha | mean vIoU % | delta pp |','|---|---|---|---:|---:|---:|']
for g in d['groups']:groups.append(f"| {g['cohort']} | {g['dataset']} | {g['branch']} | {g['alpha']} | {100*g['mean_vIoU']:.6f} | {g['mean_delta_vIoU_pp']:+.6f} |")
text='''# Spatial causal decomposition: fixed-reference amplitude diagnostic

Phase 1 measurements and independent readback are complete; the three-stage research program is not complete. Temporal and Spatial remain the intended dual-branch method. No component removal, CURRENT promotion, old queue recovery, or full-scale expansion.

The fixed cohort is 16 historically exposed development sources, eight per domain. HC04/V04 endpoints refer to Motion-Guided Instance OPD, not Adjacent Motion. References, positions, saved stochastic updates and 1792-parameter directions are fixed. Alpha=[0,1/4,1/2,1] scales the realized final parameter displacement, not learning rate or optimizer trajectories. Missing multipliers in the request were resolved by this prelocked finite default.

Three nonzero cases received 3 fresh source captures plus 9 additional full-model readouts, yielding 12 actual outputs. Ten locked no-update and three zero-displacement fits reuse exact source outputs. No new teacher call, fit, backward or optimizer step. Both endpoints reproduce sealed arrays exactly; 96 readouts are scored postseal with official and independent dense metrics (maximum difference 1.23e-15). Original-eight T is exactly reused; additional-eight T is unavailable.

## Fixed-direction amplitude results

All values below are full-Tube vIoU percentage-point differences from Frozen, not sparse anchor IoU.

'''+ '\n'.join(table)+'''

Reducing displacement reduces HC04 harm and retains much of V04 gain. HC04 and V02 remain negative at every nonzero tested alpha. This supports amplification by update magnitude; it does not establish that smaller updates fix a wrong reference. No per-case GT-selected alpha is adopted.

## Candidate support and teacher selection

Use only GT-known positions shared by Source and every prelocked displayed candidate. Of 16 sources, 6 have no common support and remain unknown. Of the other 10, 9 have one common position and HC04 has two. Pool Oracle delta is positive in 4 and neutral in 6; because Source is included, a nonnegative maximum alone is not evidence of quality. Locked teacher delta: 1 positive, 2 negative, 7 neutral, 6 unknown. All 78 displayed candidate IDs are included; five other scored candidates remain separately marked as not displayed. Per-candidate pairwise supports differ and are not pooled into the common-support Oracle. Sparse IoU deltas and full-Tube vIoU deltas are never subtracted from one another.

HC04: same-source, same-position, same-config and same-seed historical offline true-reference IoU-OPD improves full-Tube vIoU by 6.2075 pp. The current selected reference worsens sparse common-support quality and the original update loses 10.6579 pp. This is a useful supervision positive control. Other cases can differ in observation support; the eight offline Oracle results are not uniformly matched reference-only causal contrasts, nor deployment results. No new Oracle fit.

## Full paired domain readouts

S delta is versus Frozen; TS delta is versus the exact reused Temporal-only baseline. Original and additional cohorts are never pooled as independent confirmation.

'''+ '\n'.join(groups)+'''

## Remaining authorized stages

Phase 2 is a separate finite, prelocked short-window candidate/association experiment near the existing QA event, with unchanged student optimizer. Existing source caches contain no saved spatial feature maps; obtaining localized frozen features requires new capture with explicit frame/grid/resize binding. Short support and feature association require matched controls; a valid-track count is not a candidate-quality result. Phase 3 requires one fixed GT-free Spatial rule and verifiably unused sources (at least 32 per direction if available); it has not been launched. Final confirmation must include F/T/S/TS.

## Engineering and evidence boundaries

A concurrent agent added one required-pin constant after declaring source frozen. Both versions were retained; the byte-exact pinned original was restored. The active process had passed its old self-hash check before model loading; scientific functions, runtime, contract, inputs and measurements were unchanged. Root runtime already pinned the launcher. This is a disclosed engineering coordination error, not a scientific result. The own resource-guarded finite unit exited successfully; no observed OOM. The measurement wall time includes this finite run only, not complete TTA deployment cost. Full positive/negative/neutral/unknown anonymous scalars and source bindings accompany this report. No raw media, labels, query text, weights, parameter states or private caches are published.
'''
(R/'docs/STVG_SPATIAL_CAUSAL_DECOMPOSITION_V1.md').write_text(text)
plt.rcParams.update({'font.size':10,'pdf.fonttype':42});pdf=R/'output/pdf/stvg_spatial_causal_phase1_v1.pdf';pdf.parent.mkdir(parents=True,exist_ok=True)
with PdfPages(pdf)as pp:
 fig,ax=plt.subplots(figsize=(10,6));
 for cid,col in [('hc2_003','#b34a3c'),('vidstg_001','#b48936'),('vidstg_003','#26768d')]:
  rr=sorted([r for r in rows if r['case_id']==cid and r['branch']=='S'],key=lambda r:r['alpha']);ax.plot([r['alpha']for r in rr],[r['baseline_delta_vIoU_pp']for r in rr],marker='o',label=alias(cid),color=col)
 ax.axhline(0,c='gray',lw=.8);ax.set(xlabel='Fraction of the fixed saved parameter displacement',ylabel='Full-Tube vIoU change from Frozen (pp)',title='Smaller updates reduce harm, but do not correct the supervision');ax.set_xticks([0,.25,.5,1]);ax.legend();ax.grid(alpha=.15);fig.text(.1,.025,'3 nonzero historical cases; 13 exact unchanged cases retained. No new fit or GT-selected alpha.',fontsize=9);fig.tight_layout(rect=[0,.05,1,1]);pp.savefig(fig);plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,6));ax.axis('off');data=[]
 for c in pool['cases']:
  p=c['displayed_pool_oracle_on_common_support'];t=c['teacher_selection']['selected_reference_common_support'];data.append([alias(c['case_id']),c['common_observation_support']['support_count'],p['classification'],t['classification']])
 tab=ax.table(cellText=data,colLabels=['Source','Common annotated positions','Pool Oracle delta','Locked teacher delta'],loc='center',cellLoc='center');tab.auto_set_font_size(False);tab.set_fontsize(10);tab.scale(1,1.4);ax.set_title('Sparse candidate diagnostics: retain missing support as unknown',pad=15);fig.text(.07,.02,'Sparse IoU and full-Tube vIoU have different supports and cannot be numerically mixed.',fontsize=9);fig.tight_layout(rect=[0,.05,1,1]);pp.savefig(fig);plt.close(fig)
print(json.dumps(dict(report='docs/STVG_SPATIAL_CAUSAL_DECOMPOSITION_V1.md',pdf=str(pdf),pages=2)))
