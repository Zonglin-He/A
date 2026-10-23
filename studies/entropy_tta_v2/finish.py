"""Read back completed, sealed/scored paired experiment; no model or new labels."""
import json,hashlib,sys,csv,math
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
R=Path(__file__).resolve().parents[2];D=R/'outputs/entropy_tta_v2';P=R/'results/entropy_tta_v2'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert read(D/'completion.json')['status']=='completed'
plan=read(D/'plan.json');seal=read(D/'sealed.json');receipt=read(D/'score_receipt.json');assert sha(D/'sealed.json')==receipt['seal_sha256']
for f,h in receipt['files'].items():assert sha(R/f)==h,f
assert read(D/'predict_exit.json')['exit_code']==read(D/'score_exit.json')['exit_code']==0
assert sha(R/'studies/entropy_tta_v2/run.py')==plan['code_sha256']
for ds,files in seal['files'].items():
 assert len(files)==128
 for entry in files.values():
  for k,h in [('path','sha256'),('trace_path','trace_sha256'),('native_path','native_sha256')]:assert sha(D/entry[k])==entry[h]
reports={};case_audit={}
for ds in ['hc2','vidstg']:
 report=read(P/ds/'report.json');rows=list(csv.DictReader((P/ds/'paired.csv').open()))
 assert len(rows)==128
 x=np.array([float(r['loss_before'])-float(r['loss_after'])for r in rows]);y=100*np.array([float(r['viou_after'])-float(r['viou_before'])for r in rows]);n=int((x>0).sum());k=int(((x>0)&(y<0)).sum())
 assert (n,k)==(report['surrogate_improves'],report['grounding_degrades_given_surrogate_improves'])
 assert np.allclose(x,[float(r['delta_surrogate'])for r in rows],atol=0,rtol=0)
 assert np.allclose(y,[float(r['delta_vIoU_pp'])for r in rows],atol=0,rtol=0)
 # Verify every point falls inside the originally declared figure bounds.
 xs=max(float(abs(x).max()),.01)*1.12;ys=max(float(abs(y).max()),1)*1.15
 assert (x>=-.1*xs).all()and(x<=xs).all()and(abs(y)<=ys).all(),'Original plot clips observations; independent all-point renderer needed'
 rng=np.random.default_rng(20261024);idx=rng.integers(0,len(rows),size=(10000,len(rows)));mean_ci=np.quantile(y[idx].mean(1),[.025,.975]).tolist()
 private=json.loads((D/ds/'paired_private.json').read_text());candidate=np.flatnonzero((x>0)&(y<0));cases=[]
 for j in sorted(candidate,key=lambda j:y[j])[:3]:
  raw=private[int(j)];entry=seal['files'][ds][raw['query_id']]
  with np.load(D/entry['path'],allow_pickle=False)as z:
   assert np.array_equal(z['Source'],z['EntropyTTA'])
   cases.append(dict(query_alias=f'Q{j+1:03d}',loss_before=raw['loss_before'],loss_after=raw['loss_after'],v_before_percent=100*raw['viou_before'],v_after_percent=100*raw['viou_after'],delta_v_pp=float(y[j]),source_boxes_identical=True,interval_changed=not np.array_equal(z['Source_interval'],z['EntropyTTA_interval'])))
 for j,raw in enumerate(private):
  entry=seal['files'][ds][raw['query_id']]
  with np.load(D/entry['path'],allow_pickle=False)as z:
   assert np.array_equal(z['Source'],z['EntropyTTA'])
   if np.array_equal(z['Source_interval'],z['EntropyTTA_interval']):assert y[j]==0
 case_audit[ds]=dict(mean_delta_v_CI95_pp=mean_ci,harmful_update_examples=cases,all_spatial_predictions_identical=True,unchanged_interval_implies_zero_quality_change=True)
 reports[ds]=report
(P/'case_audit.json').write_text(json.dumps(case_audit,indent=2)+'\n')
lines=['# Does optimizing a TTA surrogate improve STVG?','',
 'This figure uses actual paired network-parameter updates, not fabricated samples, arrows, confidence ranks, or relabeled prototype feedback. All selected queries and signs are retained.','',
 '| Target domain | Queries | Surrogate loss improves | Grounding declines among these | Conditional fraction (95% CI) | Mean ΔvIoU (pp) | vIoU win/harm/tie | Severe vIoU decline >5pp |',
 '|---|---:|---:|---:|---:|---:|---:|---:|']
for ds,r in reports.items():
 lo,hi=r['conditional_rate_CI95'];name='HC-STVG2.0'if ds=='hc2'else'VidSTG'
 lines.append(f"| {name} | 128 | {r['surrogate_improves']} | {r['grounding_degrades_given_surrogate_improves']} | {100*r['conditional_rate']:.2f}% [{100*lo:.2f}, {100*hi:.2f}] | {r['mean_delta_vIoU_pp']:+.4f} | {r['win']}/{r['harm']}/{r['tie']} | {r['severe_v_drop_gt5pp']} |")
lines+=['','## Figure and data','',
 '- [HC real paired scatter](hc2/fig1.svg), [paired data](hc2/paired.csv), [report](hc2/report.json).',
 '- [Vid real paired scatter](vidstg/fig1.svg), [paired data](vidstg/paired.csv), [report](vidstg/report.json).',
 '- PDF/300-dpi PNG versions remain in local outputs/entropy_tta_v2/figures/.',
 '', '## Scope of the evidence','',
 'Each domain has 128 deterministic hash-selected parent videos, one query/video. These are sampled exposed official sets, not complete test domains or fresh unseen validation. Loss is the mean entropy of the four native endpoint distributions; the same loss defines the horizontal axis. The ordinate is the official complete vIoU change, in percentage points. Right-lower points are real episodes in which the optimized surrogate improves but grounding degrades. Ties remain in the conditional denominator.',
 '', 'The implementation adapts only 9,728 existing temporal-decoder LayerNorm affine scalars, using episodic Adam0.001/five fixed steps. Cached final-pass input context, first-pass gating, Source boxes and every other parameter are frozen. It is a temporal LN entropy-minimization adapter, not canonical BN-Tent or a fully re-evaluated adapted first pass. No hyperparameters/steps/cases are selected from outcomes. A single limited adapter does not establish failure of all TTA or all entropy-minimization recipes. Mean gains/losses and harmful fractions must be read together.',
 '', '**Suggested caption.** Paired effects of entropy-minimization adaptation on STVG. Each point is one parent-video/query episode; the horizontal axis is the reduction in the optimized entropy objective and the vertical axis is the change in official complete vIoU. The shaded quadrant marks episodes where the surrogate improves but grounding degrades. Five fixed updates adapt only the native temporal decoder LayerNorm affine parameters, with episode-wise reset and fixed Source spatial predictions. Fractions report all harmful episodes among surrogate-improving episodes, including unchanged-quality episodes in the denominator.',
 '', '## Verification','',
 'All256 predictions sealed before any new labels were read. Exact Source boxes/logits/prior/interval, frozen replay, parameter restoration and reset logits were checked per query. Source official metrics reproduce the prior full. Independent official-vs-dense evaluator checks and every raw prediction/trace/native-cache SHA are verified. No new expert, new observations or extra backbone was used. Main SimplePath production sources, parameters and full outcomes remain unchanged. V1 pre-optimization stride failure is retained, not counted as a scientific result.']
(P/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
proof=dict(status='all_outputs_read_back_verified',fingerprint=plan['fingerprint'],raw_sha_all_verified=True,public_pair_statistics_recomputed=True,all_plot_points_visible=True,source_and_noop_checks_per_query=True,official_metric_checks=read(P/'verification.json')['official_dense_checks'])
(P/'readback.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(reports,indent=2))
