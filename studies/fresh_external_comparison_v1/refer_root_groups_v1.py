"""Report-only root interpretation of already audited short/long paired effects."""
from refer_runtime_v4 import *
import argparse
def run(ds):
 validate_v2();folder=V2/'small_v1'/ds;out=V2P/'small_v1'/ds;a=read(folder/'reporting_receipt.json');assert a['metrics_sha256']==sha(folder/'metrics.json');assert (folder/'root_cost_export_receipt_v1.json').exists();rep=read(out/'report.json');text=f'# {ds}: root short/long interpretation\n\nGT event duration groups were used only for offline reporting after complete cohort seal; no samples are excluded. Pointwise parent-cluster95% CIs below are not familywise-adjusted or causal duration effects.\n\n'
 for group,g in rep['timegroups'].items():
  text+=f"## {group}: {g['queries']} queries\n\n"
  for key,z in g['paired'].items():
   text+=f"- {key}: t{z['t']['mean']:+.6f}pp CI{z['t']['CI95']}; v{z['v']['mean']:+.6f}pp CI{z['v']['CI95']}; v improve/harm/tie{z['v']['improved']}/{z['v']['harmed']}/{z['v']['unchanged']}.\n"
 if ds=='hc2':text+='\nBoth experts have positive long-event structured-minus-simple mean t/v gains with CIs above zero; both short-event gain CIs cross zero. Alternative-minus-current complete pipeline v loss is clear in long events and uncertain in the smaller short group. The aggregate advantage over each simple fusion must not be presented as established for both duration groups.\n'
 else:text+='\nInterpret each duration-specific CI directly; positive aggregate gain does not establish gain in both strata. Differences in expert quality and temporal-event prediction remain part of pipeline comparisons, not isolated propagation effects.\n'
 text+='\nSingle external expert per branch does not equalize parameters, prior knowledge, input resolution, processed frame count or complete uncached compute. Original current masks are used only in the separate baseline, not the replacement inference. All negative short/long rows remain in the full report and signed TSVs.\n';(out/'ROOT_DURATION_FINDINGS_v1.md').write_text(text);atomic(folder/'root_group_export_receipt_v1.json',dict(status='root_all_short_long_paired_effects_interpreted',audit_sha256=sha(folder/'reporting_receipt.json'),report_sha256=sha(out/'report.json'),findings_sha256=sha(out/'ROOT_DURATION_FINDINGS_v1.md'),code_sha256=sha(Path(__file__)),new_GT_scoring=0,new_model=0,new_expert=0))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['hc2','vidstg'],required=True);run(p.parse_args().dataset)
