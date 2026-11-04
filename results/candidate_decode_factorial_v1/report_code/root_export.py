"""ROOT readback and interpretation after a complete independent domain audit."""
from pathlib import Path
import os,sys,json,time,csv,hashlib
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
R=Path(__file__).resolve().parents[3];D=R/'outputs/candidate_decode_factorial_v1';P=R/'results/candidate_decode_factorial_v1';sys.path[:0]=[str(R/'tools'),str(R)]
from full_score import cluster
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
ARMS=['E','AnchorsIndependent','AnchorsDP','TransportIndependent','Full'];FIELDS=['v','t','event_unobserved_IoU','event_observed_IoU'];PAIRS=[(a,'E')for a in ARMS if a!='E']+[('AnchorsDP','AnchorsIndependent'),('TransportIndependent','AnchorsIndependent'),('Full','TransportIndependent'),('Full','AnchorsDP')]
LABELS={'E':'Shared correction E only','AnchorsIndependent':'E + expert anchors, independent','AnchorsDP':'E + expert anchors + DP','TransportIndependent':'E + bidirectional candidates, independent','Full':'Full spatial inference'}
def one(ds):
 start=time.perf_counter();folder=P/ds;report=read(folder/'report.json');receipt=read(D/ds/'reporting_receipt.json');audit=read(D/ds/'audit_receipt.json');rr=read(D/ds/'metrics.json')['private_rows'];assert receipt['report_sha256']==sha(folder/'report.json')and audit['metrics_sha256']==sha(D/ds/'metrics.json');assert report['queries']==len(rr)==({'hc2':1902,'vidstg':10303}[ds]);cost={}
 for s in ['predict','score','audit','report']:
  x=read(D/'exits'/(ds+'_'+s+'.json'));assert x['actual_subprocess_wait']and x['exit_code']==0;cost[s]=x
 checked=0;tails={}
 for group,qq,stats in [('all',rr,report['paired'])]+[(g,[r for r in rr if r['duration_group']==g],v)for g,v in report['timegroups'].items()]:
  for a,b in PAIRS:
   pair=a+'-'+b;fresh=cluster(qq,a,b,[(f,100)for f in FIELDS]);old=stats[pair]
   for f,v in fresh.items():
    assert(v['mean']is None)==(old[f]['mean']is None)
    if v['mean']is not None:assert abs(v['mean']-old[f]['mean'])<1e-10 and max(abs(np.array(v['CI95'])-old[f]['CI95']))<1e-10
    checked+=1
   dt=np.array([100*(r['values'][a]['t']-r['values'][b]['t'])for r in qq]);dv=np.array([100*(r['values'][a]['v']-r['values'][b]['v'])for r in qq]);tail=-dv[dv< -1e-10];tails[group+'/'+pair]=dict(queries=len(qq),improved=int((dv>1e-10).sum()),harmed=int((dv< -1e-10).sum()),unchanged=int((abs(dv)<=1e-10).sum()),severe_t_gt5pp=int((dt< -5).sum()),severe_v_gt5pp=int((dv< -5).sum()),severe_tv_union_once_gt5pp=int(((dt< -5)|(dv< -5)).sum()),rescue_tv_union_once_gt5pp=int(((dt>5)|(dv>5)).sum()),v_decline_ge10pp=int((dv<=-10).sum()),harmed_median_pp=float(np.median(tail))if len(tail)else None,harmed_p90_pp=float(np.quantile(tail,.9))if len(tail)else None)
 for r in rr:
  for f in ['t','event_GT_frames','event_unobserved_frames','event_observed_frames']:assert len({r['values'][a][f]for a in ARMS})==1
 # Same-interval E/Full must reproduce the earlier independent full recalculation.
 base=read(R/'results/shared_time_spatial_recalc_v2'/ds/'report.json')
 for a,b in [('E','SharedCorrection'),('Full','Full')]:
  for f,x in report['means'][a].items():
   y=base['means'][b][f];assert(x is None)==(y is None)
   if x is not None:assert abs(x-y)<1e-12
 for f in FIELDS:assert max(abs(np.array(report['paired']['Full-E'][f]['CI95'])-base['paired']['Full-SharedCorrection'][f]['CI95']))<1e-10
 out=dict(queries=len(rr),parents=report['parents'],means=report['means'],paired=report['paired'],transport_DP_interaction=report['transport_DP_interaction'],signed_tails=tails,timegroups=report['timegroups'],distance=report['distance'],ROOT_independent_paired_CI_checks=checked,actual_stage_cost=cost,cost_scope='Cached geometry/candidate-mask/DP replay plus separate evaluation/audit/statistics IO. Inherited Source/expert acquisition is nonzero; this is not deployment time; overlapping stage walls cannot be added as elapsed.',source_audit_sha256=sha(D/ds/'audit_receipt.json'),support=dict(primary_queries=len(rr),scorable_conditional_unobserved_queries=sum(r['values']['E']['event_unobserved_IoU']is not None for r in rr),conditional_dense_unobserved_frames=sum(r['values']['E']['event_unobserved_frames']for r in rr)))
 write(folder/'ROOT_report.json',out);lines=[f'# Fixed-time candidate generation × decoding: {ds}','',f'{len(rr)} complete original queries/{report["parents"]}parent clusters; same8expert inputs, current E, native feature/geometry sufficient-state and immutable Full interval. No hard anchor clamp, temporal update, new Source/expert/GPU/model/optimizer or parameter selection.','', '|Spatial strategy|Transport|DP|vIoU|Event-unobserved IoU|ΔvIoU vs E, pp [95% CI]|ΔUnobs vs E, pp [95% CI]|','|---|---|---|---:|---:|---|---|']
 for a in ARMS:
  m=report['means'][a];v=report['paired'].get(a+'-E');fmt=lambda z:'unscorable'if z['mean']is None else f'{z["mean"]:+.6f} [{z["CI95"][0]:+.6f}, {z["CI95"][1]:+.6f}]';lines.append('|'+LABELS[a]+'|'+('yes'if a in ['TransportIndependent','Full']else'no')+'|'+('yes'if a in ['AnchorsDP','Full']else'no')+f'|{100*m["v"]:.6f}|{100*m["event_unobserved_IoU"]:.6f}|'+(fmt(v['v'])if v else'0')+'|'+(fmt(v['event_unobserved_IoU'])if v else'0')+'|')
 lines+=['','The no-transport intervention changes only unobserved candidate legality, preserving original E/expert node scores, edge scores, geometric support and all normalization denominators. It tests the added value of transported candidate boxes under common scoring, not removal of every computation used to prepare these scores. Independent selection is node-wise argmax with original first-index ties, not interpolation.','', 'No-transport unobserved NATIVE outputs are exactly E. Its conditional DENSE unobserved IoU can still change because the official evaluator interpolates between sampled predictions, including changed expert-observation nodes. These effects do not indicate an unintended transported candidate.','', 'Conditional Unobserved IoU: same annotated dense frames within predicted∩GT event, excluding exact8expert physical times. Query means first, parent10000pointwiseCI next. Empty-support queries remain null only for conditional metrics, all queries including zeros remain in vIoU. This is distinct from the original128 all-video native-frame Unobs column.','', '|Key controlled comparison|ΔvIoU, pp [95% CI]|ΔUnobs, pp [95% CI]|','|---|---|---|']
 for pair in ['AnchorsIndependent-E','TransportIndependent-AnchorsIndependent','AnchorsDP-AnchorsIndependent','Full-TransportIndependent','Full-AnchorsDP']:
  v=report['paired'][pair];lines.append('|'+pair+'|'+fmt(v['v'])+'|'+fmt(v['event_unobserved_IoU'])+'|')
 lines+=['','|Nearest expert distance|Supported queries|Annotated dense unobserved frames|E IoU|Transport independent IoU|Full IoU|','|---|---:|---:|---:|---:|---:|']
 for b,z in report['distance'].items():lines.append('|'+b+f'|{z["scorable_queries"]}|{z["annotated_frames"]}|'+ '|'.join('unscorable'if z['means'][a]is None else f'{100*z["means"][a]:.6f}'for a in ['E','TransportIndependent','Full'])+'|')
 lines+=['', 'All signed losses/rescues/severe tails and short/long outcomes are in ROOT_report.json and all_signed_scalars.tsv. Interaction is (Full−TransportIndependent)−(AnchorsDP−AnchorsIndependent), with paired query/parent CI. Positive/null/negative outcomes retain the frozen parameters. Full target sets were previously exposed; not unknown confirmation or familywise inference. These controls do not replace or remove separately preserved interpolation comparisons. No manuscript/Fig1 or original method/scientific hash change.','']
 (folder/'ANALYSIS.md').write_text('\n'.join(lines));write(D/ds/'ROOT_receipt.json',dict(status='ROOT_interpreted_and_independent_CI_checked',queries=len(rr),report_sha256=sha(folder/'ROOT_report.json'),analysis_sha256=sha(folder/'ANALYSIS.md'),independent_CI_checks=checked,elapsed_seconds=time.perf_counter()-start,new_GT=0,new_model=0));print(ds,'ROOT completed',json.dumps({k:report['paired'][k]['v']for k in ['AnchorsIndependent-E','TransportIndependent-AnchorsIndependent','Full-TransportIndependent','Full-E']}),flush=True)
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--dataset',choices=['hc2','vidstg'],required=True);one(a.parse_args().dataset)
