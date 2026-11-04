from pathlib import Path
import sys,json,time,csv
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R/'studies/fixed_interval_spatial_w3_v1'))
from common import *
from full_score import cluster

def fmt(x):return 'unscorable'if x['mean']is None else f"{x['mean']:+.6f} [{x['CI95'][0]:+.6f}, {x['CI95'][1]:+.6f}]"
def main():
 validate();torch.set_num_threads(1);sys.addaudithook(old.deny_gpu_media);outer=read(D/'controller_actual_exit.json');assert outer['actual_tool_wait']and outer['exit_code']==0
 exits={f.stem:read(f)for f in(D/'exits').glob('*.json')};assert len(exits)==6 and all(z['actual_subprocess_wait']and z['exit_code']==0 for z in exits.values())
 for w in [read(D/'controller.json')]+[z['worker']for z in exits.values()]:
  f=Path('/proc')/str(w['pid'])/'stat';assert not f.exists()or int(f.read_text().split()[21])!=w['startticks']
 audits={};reports={};root_checks=0;allsevere={};costs={}
 for ds in COUNTS:
  audits[ds]=read(D/ds/'audit_receipt.json');a=audits[ds];assert a['metrics_sha256']==sha(D/ds/'metrics.json')and a['report_sha256']==sha(P/ds/'report.json')
  rr=read(D/ds/'metrics.json')['private_rows'];q=read(P/ds/'report.json');reports[ds]=q;assert len(rr)==COUNTS[ds]
  b={z['id']:z for z in read(TEMP/'temporal'/ds/'metrics.json')['private_rows']};c={z['id']:z for z in read(COMP/'components'/ds/'metrics.json')['private_rows']}
  for row in rr:assert row['parent']==str(b[row['id']]['parent'])==str(c[row['id']]['parent'])and len({v['t']for v in row['values'].values()})==1
  for group,items,rep in [('all',rr,q)]+[(k,[r for r in rr if r['duration_group']==k],v)for k,v in q['time_groups'].items()]:
   for aa,bb in PAIRS:
    fresh=cluster(items,aa,bb);assert fresh==rep['paired'][aa+'-'+bb];root_checks+=len(fresh)
  severe={}
  for aa,bb in PAIRS:
   dt=np.array([100*(r['values'][aa]['t']-r['values'][bb]['t'])for r in rr]);dv=np.array([100*(r['values'][aa]['v']-r['values'][bb]['v'])for r in rr]);neg=-dv[dv< -1e-10]
   severe[aa+'-'+bb]=dict(improved=int((dv>1e-10).sum()),harmed=int((dv< -1e-10).sum()),unchanged=int((abs(dv)<=1e-10).sum()),severe_t_over5pp=int((dt< -5).sum()),severe_v_over5pp=int((dv< -5).sum()),severe_union_once=int(((dt< -5)|(dv< -5)).sum()),rescues_union_once=int(((dt>5)|(dv>5)).sum()),v_loss_ge10pp=int((dv<=-10).sum()),harm_magnitude_median_pp=float(np.median(neg))if len(neg)else None,harm_magnitude_p90_pp=float(np.quantile(neg,.9))if len(neg)else None)
   assert severe[aa+'-'+bb]['severe_t_over5pp']==0
  allsevere[ds]=severe;atomic(P/ds/'signed_tail_summary.json',severe)
  seal=read(D/ds/'sealed.json');segments={}
  for rec in seal['files'].values():
   w=rec['worker'];key=f"{w['pid']}:{w['startticks']}";s=segments.setdefault(key,dict(queries=0,query_wall_sum_seconds=0));s['queries']+=1;s['query_wall_sum_seconds']+=rec['conditional_cached_query_wall_seconds']
  costs[ds]=dict(prediction_segments=segments,prediction_whole_wall=seal['elapsed_wall_seconds'],stages={k:v for k,v in exits.items()if k.startswith(ds)},scope='Cached geometry/NPZ readback, finalDP and experimental audit/scoring costs, not end-to-end or pure spatial deployment latency. Inherited Source/features/expert calls remain nonzero; nested clocks cannot be added; no model load this task. Outside actual measured scope unknown clocks null, not0.')
 atomic(P/'cached_CPU_costs.json',dict(domains=costs,new_Source=0,new_expert=0,new_GPU=0,new_model=0,new_optimizer=0,inherited_acquisition_nonzero=True,deployment_latency=False,nested_clocks_not_additive=True))
 text='# Full-list fixed-interval spatial comparison\n\nEvery arm uses the original unchanged Full interval and the exact same eight raw expert observations, physical native grid and query list inside each benchmark. HC:1902 queries/227 parent sources; VidSTG:10303/732. All per-query tIoU values are exactly equal across seven arms. Existing five-arm predictions/official scores are reused unchanged; only PastOnlyFixed and OneWayMatchFixed are new cached-geometry final-DP predictions.\n\n'
 text+='|Dataset|Spatial strategy|tIoU %|vIoU %|R.3 %|R.5 %|Unobserved IoU %|\n|---|---|---:|---:|---:|---:|---:|\n'
 for ds,q in reports.items():
  for arm in ARMS:
   m=q['means'][arm];text+=f'|{ds}|{arm}|'+ '|'.join('unscorable'if m[k]is None else f'{100*m[k]:.6f}'for k in ['t','v','vIoU_ge_03','vIoU_ge_05','unobserved_actual_sIoU'])+'|\n'
 text+='\nPureLog retains the original center/log-size interpolation; PureLinear is linear xyxy-corners. Both use only raw8boxes, keep raw expert boxes exactly at observations and nearest endpoint boxes outside the anchor span. Both were already audited before this task; no target-dependent choice or relabeling. SharedCorrection directly returns the exact shared-corrected Source trajectory E, without candidate transport or finalDP. Its corrected-ROI temporal interval is exactly original Full.\n\n'
 text+='|Dataset|Controlled difference|Delta vIoU pp [95% CI]|Delta unobserved IoU pp [95% CI]|\n|---|---|---|---|\n'
 for ds,q in reports.items():
  for aa,bb in PAIRS:
   r=q['paired'][aa+'-'+bb];text+=f'|{ds}|{aa}-{bb}|{fmt(r["v"])}|{fmt(r["unobserved_actual_sIoU"])}|\n'
 text+='\nFull−SharedCorrection measures the added candidate propagation and final path selection together. Full−Independent isolates final global path selection using the same candidate bank/validity/unary, with no temporal or candidate-generation difference. Independent uses unary argmax and no adjacent links; the difference includes adjacent-link scoring plus global selection, not a solver-only comparison at an identical objective.\n\nPastOnlyFixed removes all future-anchor support and recomputes unary scores in the saved geometry, then re-solves DP at Full interval. It retains the past/own-observed direction and E fallback; this is an offline anchor-availability control, not a causal online method. OneWayMatchFixed keeps both anchors but uses saved one-way feature-correspondence fits instead of reciprocal products, preserving original source features/seeds/fit constraints, with DP at the identical interval. These distinguish two different meanings of bidirectional inference. They are ablations, not new competitive learned models or a broad TTA ranking.\n\n'
 text+='|Dataset|Full minus control|v gains|ties|harms|severe v >5pp|rescues >5pp|loss ≥10pp|\n|---|---|---:|---:|---:|---:|---:|---:|\n'
 for ds,pairs in allsevere.items():
  for pair,v in pairs.items():text+=f'|{ds}|{pair}|'+ '|'.join(str(v[k])for k in ['improved','unchanged','harmed','severe_v_over5pp','rescues_union_once','v_loss_ge10pp'])+'|\n'
 text+='\n## Evidence boundaries\n\nSame expert INPUT does not mean fixed identical observed OUTPUT. Unchanged Full/E/Independent/directional ablations permit soft selection of E versus the raw expert, while pure interpolation hard-retains raw expert boxes. The previously identified intended hard-anchor contract remains unsatisfied and is not repaired here. Thus path-quality differences include observed-node selection; these tables do not isolate only unobserved propagation or certify the intended hard-anchor method. Both positive and negative full-dataset results are retained.\n\nAll GT support/clipping/half-open interval rules are the original official benchmark rules. Unobserved IoU is queryequal over native GT-supported non-observed frames, not dense vIoU and not independent-frame confidence. Empty support remains null. No source, expert, grid acquisition, RGB decoding, optimization or temporal posterior update is executed. Full method/profile/predictions and prior scores unchanged; no manuscript edits, target tuning or old queued jobs.\n\nTent/SAR remain the previous temporal-decoder LayerNorm adapters with very few interval changes; this spatial experiment does not prove superiority over broad test-time refinement strategies. Queryequal 10000 paired-parent bootstrap seed20261022; short<=5s/long and all signed severe/rescue tails retained. CIs are pointwise on already exposed full benchmarks, not familywise/equivalence/unseen confirmation. Cached CPU replay/scoring walls include IO/audit and are not deployment latency; earlier Source/expert costs remain nonzero.\n\n'
 text+=f'New official checks:{sum(read(D/ds/"score_receipt.json")["new_official_checks"]for ds in COUNTS)}; max official discrepancy:{max(read(D/ds/"score_receipt.json")["max_official_discrepancy"]for ds in COUNTS):.15g}. Independent cold NumPy DP array fields:{sum(a["all_array_checks"]for a in audits.values())}; normalized dense scalar checks:{sum(a["independent_scalar_checks"]for a in audits.values())}; max scalar discrepancy:{max(a["max_independent_metric_error"]for a in audits.values()):.15g}. ROOT independently recomputed {root_checks}paired/group metric-CIs. All6childstages and actual controller exited0. Private source geometry/feature/GT/interval/box/caption/media/weights remain private; public only source/protocol/scalars/hashes.\n'
 (P/'ANALYSIS.md').write_text(text)
 verdict={ds:{pair:{k:v[k]for k in ['v','unobserved_actual_sIoU']}for pair,v in q['paired'].items()}for ds,q in reports.items()};atomic(P/'effect_summary.json',verdict)
 rec=dict(status='complete_full_fixed_interval_spatial_controls_ROOT_analyzed_PENDING_publication',scientific_fingerprint=validate()['fingerprint'],queries=COUNTS,new_official_checks=24410,root_paired_group_CI_checks=root_checks,all6actualchildstages_exit0=True,controller_actual_wait=outer,audit_sha256={ds:sha(D/ds/'audit_receipt.json')for ds in COUNTS},analysis_sha256=sha(P/'ANALYSIS.md'),effect_summary_sha256=sha(P/'effect_summary.json'),new_Source=0,new_expert=0,new_GPU=0,new_model=0,new_optimizer=0,hard_anchor_contract_repaired=False,method_or_paper_changed=False,time=time.time());atomic(D/'ROOT_completion.json',rec);atomic(P/'completion_scalars.json',rec);print(json.dumps(verdict),flush=True)
if __name__=='__main__':main()
