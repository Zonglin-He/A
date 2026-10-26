"""Root descriptive reports after whole independent audit; never choose a gamma."""
from common import *
import argparse

def tails(rr,a,b):
 import numpy as np
 dt=np.asarray([100*(r['values'][a]['t']-r['values'][b]['t'])for r in rr]);dv=np.asarray([100*(r['values'][a]['v']-r['values'][b]['v'])for r in rr])
 return dict(severe_t=int((dt< -5).sum()),severe_v=int((dv< -5).sum()),severe_union_once=int(((dt< -5)|(dv< -5)).sum()),rescue_union_once=int(((dt>5)|(dv>5)).sum()),v_improve=int((dv>0).sum()),v_harm=int((dv<0).sum()),v_tie=int((dv==0).sum()))

def run(ds):
 import numpy as np,torch
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 torch.set_num_threads(1);validate();gt_guard();start=time.perf_counter();audit=read(D/ds/'reporting_receipt.json');assert audit['status']=='verified_complete_retained192_all_gamma_arrays_actual_ROI_repool_parent_SHA_and_means';assert audit['score_sha256']==sha(D/ds/'score_receipt.json');seal=read(D/ds/'sealed.json');assert len(seal['files'])==192
 for task in ['gamma','roi']:
  rr=read(D/ds/(task+'_metrics.json'))['private_rows'];rep=read(P/task/ds/'report.json');ps=pairs(task,ds);summary={a+'-'+b:tails(rr,a,b)for a,b in ps};out=P/task/ds
  for a,m in rep['means'].items():
   for k,v in m.items():
    zz=[r['values'][a][k]for r in rr if r['values'][a][k]is not None];assert(v is None and not zz)or abs(v-float(np.mean(zz)))<1e-12
  atomic(out/'root_all_paired_tails_v1.json',summary)
  text=f'# {ds}: standalone retained192 {task} review control\n\nExactly192queries, {rep["parents"]} parents. This pre-existing retained-raw cohort includes previously exposed development and confirmation sources; it is not the full1902/10303 domain, an unseen test, or a prefix accuracy estimate. All192 prediction configurations sealed before this independent official scoring. Separate actual ROI repooling/cold gamma chain, prediction/input/parent/receipt SHA and all-mean audits passed.\n\n|Arm|tIoU %|vIoU %|R.3 %|R.5 %|\n|---|---:|---:|---:|---:|\n'
  for a,m in rep['means'].items():text+=f'|{a}|{100*m["t"]:.6f}|{100*m["v"]:.6f}|{100*m["vIoU_ge_03"]:.6f}|{100*m["vIoU_ge_05"]:.6f}|\n'
  text+='\nParent-paired10000-bootstrap pointwise95% CIs, no familywise significance claim:\n\n'
  for k,v in rep['paired'].items():text+=f'- {k}: t{v["t"]["mean"]:+.6f}pp CI{v["t"]["CI95"]}; v{v["v"]["mean"]:+.6f}pp CI{v["v"]["CI95"]}; severe t/v/union{summary[k]["severe_t"]}/{summary[k]["severe_v"]}/{summary[k]["severe_union_once"]}, rescues{summary[k]["rescue_union_once"]}.\n'
  text+='\nShort<=5s/long ground-truth-duration strata are offline only:\n\n'
  for group,v in rep['timegroups'].items():
   for k,z in v['paired'].items():text+=f'- {group}, n{v["queries"]}, {k}: v{z["v"]["mean"]:+.6f}pp CI{z["v"]["CI95"]}; t{z["t"]["mean"]:+.6f} CI{z["t"]["CI95"]}.\n'
  if task=='gamma':
   text+='\nEvery gamma point reconstructs E, bidirectional geometry, actual native-grid E ROI features, prototype/posterior, own legal jointMAP interval and finalDP. Both spatial and temporal downstream effects change, so this is not an isolated appearance parameter. Current historical defaults remain HC0.5/Vid1; the complete five-point curve is descriptive sensitivity on exposed data and does not establish source-only calibration. No default selection, new fullGamma acquisition, old-owner restoration or main-method change.\n'
   fig,axes=plt.subplots(1,2,figsize=(10.8,3.6),constrained_layout=True);x=np.asarray([0,.25,.5,.75,1]);names=GAMMA[1:];default='Gamma05'if ds=='hc2'else'Gamma1'
   for metric,label in [('t','tIoU'),('v','vIoU')]:axes[0].plot(x,[100*rep['means'][n][metric]for n in names],marker='o',label=label)
   axes[0].set(xlabel='Shared correction strength gamma',ylabel='Query-mean IoU (%)',title=f'{ds}: fixed retained192');axes[0].legend();axes[0].grid(alpha=.25)
   y=[];lo=[];hi=[]
   for n in names:
    z=dict(mean=0.,CI95=[0.,0.])if n==default else rep['paired'][n+'-'+default]['v'];y.append(z['mean']);lo.append(z['mean']-z['CI95'][0]);hi.append(z['CI95'][1]-z['mean'])
   axes[1].errorbar(x,y,yerr=[lo,hi],marker='o',capsize=4);axes[1].axhline(0,color='grey',lw=1);axes[1].set(xlabel='Shared correction strength gamma',ylabel='vIoU difference vs fixed default (pp)',title='Pointwise parent-paired 95% CI');axes[1].grid(alpha=.25)
   fig.savefig(D/ds/'gamma_sensitivity_v1.png',dpi=160);fig.savefig(out/'gamma_sensitivity_v1.svg');plt.close(fig)
  else:
   fallback={a:sum(e['diagnostics']['fallback_frames'].get(a,0)for e in seal['files'].values())for a in ['CorrectedROI','SourceROI','ShiftedROI','RandomROI']};atomic(out/'ROI_fallback_counts_v1.json',dict(queries=192,frames=sum(len(__import__('numpy').load(D/e['outputs']['roi']['path'])['frame_ids'])for e in seal['files'].values()),fallback_frames=fallback,definition='Original area-zero ROI pooling falls back to valid global grid, unchanged and counted.'))
   text+='\nAll appearance arms retain identical q0/eight observations, exact MAIN spatial candidate bank and coefficients, centering/unit/prototype/kappa0/beta1. Only temporal pooling changes, each computes own posterior/interval/finalDP. GlobalAppearance pools all valid native grid cells. ShiftedROI translates the corrected trajectory by a predeclared SHA-selected horizontal quarter image; RandomROI fixes a SHA-derived querywise center with original per-frame widths/heights. Context1.5 and original out-of-grid global fallback remain, with counts reported. These counterfactual controls can alter coverage/temporal smoothness and are not semantic ground truth or pure action-feature attribution. Source is untouched native-time Source boxes, distinct from the SourceROI full chain. No control, scale or main method is selected using these scores.\n'
   text+=f'\nGlobal-fallback frame counts: {fallback}.\n'
  text+='\nAll signed positive/negative rows, severe tails/rescues and short/long CIs are retained. Conditional replay elapsed clocks include file SHA, resource checks and actual original-grid pooling; not process CPU or complete deployment latency. New model/Source/expert/optimizer calls are zero. Original acquired Source/eight-expert contexts and old gamma preprocessing are inherited nonzero costs, not free observations. Shared original Vid producer/TTA costs must not be added twice.\n'
  (out/'ROOT_FINDINGS_v1.md').write_text(text)
 rec=dict(status='root_both_tasks_all_signed_shortlong_tails_curve_cost_complete_pending_image_inspection_publication',dataset=ds,code_sha256=sha(Path(__file__)),audit_sha256=sha(D/ds/'reporting_receipt.json'),report_sha256={t:sha(P/t/ds/'report.json')for t in ['gamma','roi']},actual_root_worker=identity(),actual_root_elapsed_wall_seconds=time.perf_counter()-start,conditional_predict_wall=seal['elapsed_wall_seconds'],official_score=read(D/ds/'score_receipt.json'),independent_audit=audit,old_gamma_precompute=read(G/ds/'sealed_retained_subset.json')['seconds'],new_model=0,new_Source=0,new_expert=0,new_GT_scoring=0);atomic(D/ds/'root_export_receipt_v1.json',rec);atomic(P/'gamma'/ds/'root_cost_v1.json',rec)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['hc2','vidstg'],required=True);run(p.parse_args().dataset)
