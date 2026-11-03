"""Root-only interpretation/export after a complete domain's audited scores."""
from pathlib import Path
import sys,json,hashlib,csv,time,argparse
R=Path(__file__).resolve().parents[4];S=R/'studies/expert_subset_followups_v1';sys.path.insert(0,str(S))
from common_v1 import *
from full_score import cluster

def main(expert,ds):
 validate();guard();torch.set_num_threads(1);folder=D/expert/ds;out=P/expert/ds
 ack=read(folder/'reporting_receipt.json');assert ack['queries']==ack['parents']==128 and ack['metrics_sha256']==sha(folder/'metrics.json')and ack['report_sha256']==sha(out/'report.json')
 assert(folder/'root_export_receipt.json').exists();rep=read(out/'report.json');rr=read(folder/'metrics.json')['private_rows'];tails=read(out/'tails.json');seal=read(folder/'sealed.json');assert len(rr)==len({r['parent']for r in rr})==len(seal['files'])==128
 fields=[('v',100),('t',100),('observed_actual_sIoU',100),('unobserved_actual_sIoU',100),('boundary_MAE_seconds',1)]
 checks=0
 for group,part in [('all',rr)]+[(g,[r for r in rr if r['duration_group']==g])for g in rep['timegroups']]:
  saved=rep['paired']if group=='all'else rep['timegroups'][group]['paired']
  for aa,bb in PAIRS:
   computed=cluster(part,aa,bb,fields)
   for f,_ in fields:
    a=computed[f];b=saved[aa+'-'+bb][f]
    assert a==b,(group,aa,bb,f,a,b);checks+=1
 costs=[];empty=[]
 for i,(key,e)in enumerate(seal['files'].items(),1):
  assert sha(D/e['receipt_path'])==e['receipt_sha256']
  costs.append(dict(query_index=i,pid=e['worker']['pid'],startticks=e['worker']['startticks'],logical_expert_queries=e['new_expert_queries'],model_steps=e['actual_model_steps'],empty_mask_box_conversions=e['empty_mask_to_full_frame_boxes'],allocated_peak_bytes=e['peak_allocated_GPU_bytes'],reserved_peak_bytes=e['peak_reserved_GPU_bytes'],prediction_sha256=e['sha256'],receipt_sha256=e['receipt_sha256'],raw_expert_output_sha256=e['raw_expert_output']['sha256'],expert_payload_sha256=e['expert_payload']['sha256'],**e['cost']));empty.append(e['empty_mask_to_full_frame_boxes'])
 assert sum(r['model_steps']for r in costs)==128*(8 if expert=='onlinerefer_r50'else 1)
 with(out/'all128_actual_query_costs.tsv').open('w')as f:
  w=csv.DictWriter(f,fieldnames=list(costs[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(costs)
 load=read(D/expert/'model_load.json');sc=read(folder/'score_receipt.json');counts=dict(observations=1024,queries=128,parents=128,box_conversion_empty_observations=sum(empty),queries_any_empty_box_conversion=sum(x>0 for x in empty),queries_all8_empty_box_conversion=sum(x==8 for x in empty),native_empty_masks_not_measured_for_direct_boxhead=(expert=='onlinerefer_r50'),native_empty_mask_count=None if expert=='onlinerefer_r50'else sum(empty),all_queries_retained=True)
 provenance=dict(unique_model_load_record=load,model_load_same_PID_start_shared_across_domains_not_sum_twice=True,counts=counts,logical_expert_queries=128,model_steps=sum(r['model_steps']for r in costs),new_Source=0,new_original_ReferDINO=0,new_optimizer=0,nested_segment_sums={k:sum(x['cost'][k]for x in seal['files'].values())for k in next(iter(seal['files'].values()))['cost']},OnlineRefer_forward_window_contains_step_checkpoint_IO=expert=='onlinerefer_r50',query_uses_cached_Source_not_uncached_deployment=True,inherited_Source_and_old_expert_acquisition_nonzero=True,peak_allocated_bytes=max(x['allocated_peak_bytes']for x in costs),peak_reserved_bytes=max(x['reserved_peak_bytes']for x in costs),official_checks=sc['new_official_checks'],max_official_error=sc['max_error'])
 atomic(out/'ROOT_cost_and_empty_provenance_v1.json',provenance)
 text=f'# Root audited subset interpretation: {expert}, {ds}\n\nAll128queries/128distinctparents whole-sealed, separate input/raw/Source-grid/state-or-mask/actual E-ROI/geometry/prototype/posterior/finalDP cold-array audit, independent official score and separate allmean/file/parent/SHA audit. {sc["new_official_checks"]} new official checks; max discrepancy{sc["max_error"]:.12g}. Root independently rederived all{checks} paired/group scalar bootstrap entries exactly. Exposed fixed128 panel, not full-domain/unknown confirmation.\n\n'
 text+='|Arm|tIoU %|vIoU %|R.3 %|R.5 %|Unobserved IoU %|\n|---|---:|---:|---:|---:|---:|\n'
 for arm,m in rep['means'].items():text+='|'+arm+'|'+'|'.join('missing'if m[f]is None else f'{100*m[f]:.6f}'for f in ['t','v','vIoU_ge_03','vIoU_ge_05','unobserved_actual_sIoU'])+'|\n'
 text+='\nPureSource uses native Source event. PureOwn is the identical raw-eight/nearest-tail interpolated path with this expert Current event. Current-minus-PureSource is total temporal+spatial change. Current-minus-PureOwn fixes exactly the same event and isolates the as-implemented spatial strategy; tIoU differences are exactly zero. Current still permits E at observed frames, whereas pure retains raw expert boxes. No hard-anchor fix or intended-hard-anchor validation.\n\n'
 for group,part in [('all',rep['paired'])]+[(g,v['paired'])for g,v in rep['timegroups'].items()]:
  text+=f'## {group}\n\n|Pair|Metric|Mean difference|95% paired parent CI|Queries / parents|\n|---|---|---:|---|---:|\n'
  for aa,bb in PAIRS:
   for f,_ in fields:
    x=part[aa+'-'+bb][f];ci=x['CI95'];label='missing'if ci is None else f'[{ci[0]:+.6f}, {ci[1]:+.6f}]';mean='missing'if x['mean']is None else f'{x["mean"]:+.6f}';text+=f'|{aa}-{bb}|{f}|{mean}|{label}|{x["queries"]} / {x["parents"]}|\n'
 text+='\n## Signed harm and rescue\n\n|Pair|v gains / harms / ties|Severe t / v / union once (>5pp loss)|Rescue union once (>5pp gain)|\n|---|---|---|---:|\n'
 for pair,t in tails.items():text+=f'|{pair}|{t["v_improve"]} / {t["v_harm"]} / {t["v_tie"]}|{t["severe_t"]} / {t["severe_v"]} / {t["severe_tv_union_once"]}|{t["rescue_tv_union_once"]}|\n'
 text+='\nAll CIs are pointwise10000paired-parent bootstrap, not familywise/equivalence. Missing scored observations are not zero. Positive own-expert gain does not imply beating native Source; all Source-relative losses, short/long results and observed/unobserved/boundary negatives above remain part of the result. No target setting/expert selection.\n\n'
 if expert=='onlinerefer_r50':text+='OnlineRefer uses official native causal state over only these eight supplied frames. Native query slots contract from five to one after initial selection; actual per-step logits/masks/boxes/prestate and subsequent state chain are checkpointed and audited, with no intervening RGB frames. Its boxes are selected direct boxhead outputs; zero mask-to-box conversion is not proof of zero empty masks. Original import-only0model failure, compatibility guard fix and strict CPU checkpoint doctor are preserved separately.\n\n'
 else:text+='SgMg uses the official native global track selection and refined MSO masks. A read-only hook retains original all-query logits; output conversion follows official crop/bilinear original size/sigmoid>.5 and halfopen bbox. Empty masks map to a prespecified full-frame neutral box, not a Source/GT/detector fallback. Every empty query stays in the denominator.\n\n'
 text+=f'Actual model steps{provenance["model_steps"]} for128logicaljoint8expert queries,0newSource/originalReferDINO/optimizer. Model load{load["seconds"]:.9f}s belongs once to PID{load["worker"]["pid"]}/start{load["worker"]["startticks"]} shared across both datasets. Forward, checkpoint, postprocess, finish and query clocks nest and must not be added; query inputs use inherited cached Source. OnlineRefer forward window includes raw-step checkpoint IO. No uncached deployment or equal-compute claim. Both new experts FP32/noAMP/noTF32; SgMg official example AMP is explicitly disabled. Source/older acquisition nonzero; CPU doctor/download costs are separate. Actual128perquery costs and peaks retained.\n\n'
 text+=f'Box-conversion empty observations{sum(empty)}/1024; queries with any/all-eight conversions{counts["queries_any_empty_box_conversion"]}/{counts["queries_all8_empty_box_conversion"]}. All128queries retained. No claim that new experts constitute independent families/equal knowledge. No paper/production/current method/old frozen file or protected draft change.\n'
 (out/'ROOT_FINDINGS_v1.md').write_text(text);receipt=dict(status='ROOT_whole128_audited_scalar_CI_signed_duration_cost_interpretation_complete_pending_publication',expert=expert,dataset=ds,paired_group_scalar_checks=checks,audit_sha256=sha(folder/'reporting_receipt.json'),report_sha256=sha(out/'report.json'),findings_sha256=sha(out/'ROOT_FINDINGS_v1.md'),cost_sha256=sha(out/'all128_actual_query_costs.tsv'),provenance_sha256=sha(out/'ROOT_cost_and_empty_provenance_v1.json'),code_sha256=sha(Path(__file__)),new_Source=0,new_expert=0,new_model=0,new_prediction=0,new_GT=0,time=time.time());atomic(folder/'root_interpretation_v1.json',receipt);print(json.dumps(dict(expert=expert,dataset=ds,paired_group_scalar_checks=checks,Current_vIoU=100*rep['means']['Current']['v'],total_pair=rep['paired']['Current-PureSource']['v'],same_event_spatial_pair=rep['paired']['Current-PureOwn']['v'],empty=counts)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--expert',required=True,choices=EXPERTS);p.add_argument('--dataset',required=True,choices=['hc2','vidstg']);a=p.parse_args();main(a.expert,a.dataset)
