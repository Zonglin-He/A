"""Report the fixed position-preserving context test; do not choose controls."""
import sys
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.native_event_position_v1.run import D,P,read,put,sha
def main():
 r=read(P/'report.json');a=read(P/'information_audit.json');s=read(D/'sealed.json');lines=[]
 for arm,m in r['means'].items():
  t=r['paired'].get(arm+'-NativeG6',{}).get('t',{});lines.append(f"|{arm}|{100*m['v']:.4f}|{100*m['t']:.4f}|{t.get('improved',0)}/{t.get('harmed',0)}/{t.get('unchanged',64)}|{t.get('severe_below_minus5pp',0)}|")
 pair=r['paired']['GlobalPositionInnovation-NativeG6'];source=r['paired']['GlobalPositionInnovation-Source'];control=r['paired']['GlobalPositionInnovation-ContextInnovation'];g=r['groups_vs_G6']
 text='''# Original global-grid temporal positions in overlapping contexts

Fixed, exposed HC64 development. Paper full remains paused, space unchanged.
Previous ContextInnovation almost restores Global6 but does not improve it, while
short events still suffer. Architectural audit finds native query_decoder.time_embed(t)
starts at0 in every crop. Suffix nodes at original offset indices8..32/31 become
0..24/23; a content-context intervention therefore also changes temporal positions.

One new configuration retains full-grid positional embeddings at actual selected
offset indices, with a scoped hook in both original native decoder passes. Frozen
weights/state_dict/source are unchanged. Full caption and original projected feature
values remain fixed; prefix0..48/suffix16..64 and censored innovation qContext/bSameLogits
feedback are unchanged. Full and Prefix have identical native position values and
must restore previous outputs bitwise64. This is a mechanism feasibility test, not a
trained debiasing method or a claim that temporal shift alone causes errors.

All percent metrics, improved/harmed/tied time versusNativeG6:

|Arm|complete vIoU|tIoU|time +/−/=|time loss>5pp|
|---|---:|---:|---:|---:|
'''+ '\n'.join(lines)+'\n\n'
 text+=f"Primary−NativeG6: v{pair['v']['mean']:+.4f}pp CI{pair['v']['CI95']}, t{pair['t']['mean']:+.4f}pp CI{pair['t']['CI95']}. Primary−Source: v{source['v']['mean']:+.4f}pp, t{source['t']['mean']:+.4f}pp. Primary−rebased ContextInnovation: t{control['t']['mean']:+.4f}pp CI{control['t']['CI95']}. Shortt{g['short_le5s']['GlobalPositionInnovation']['t']:+.4f}pp, longt{g['long_gt5s']['GlobalPositionInnovation']['t']:+.4f}pp versusNativeG6.\n\n"
 text+=f"Matched same-duration/near-native-prior feedback accuracy{100*a['matched_candidate_pair_macro_accuracy']:.4f}% CI{a['CI95']}, native{100*a['native_matched_pair_macro_accuracy']:.4f}%;{a['matched_candidate_pairs']:,}pairs. This post-seal diagnostic does not choose inference.\n\n"
 passed=pair['t']['mean']>0 and pair['v']['mean']>0 and source['t']['mean']>0 and source['v']['mean']>0 and pair['t']['improved']>pair['t']['harmed'] and all(g[x]['GlobalPositionInnovation']['t']>0 for x in g)
 text+=('Basic development screen passes only; frozen cross-domain recheck and tails still required before adoption.'if passed else'Primary does not pass broad-improvement development screen. Do not promote or scan position offsets/feedback weights.')+'\n\n'
 text+=f"All64 Full hidden/posterior and Prefix posterior bitwise; reused Source/Native/parent controls exact. Official/dense checks{r['checks']}, maxerror{r['max_metric_error']:.3g}. Original48+434 sources intact. Worker{s['seconds']:.2f}s including source loading, calls{s['counts']}; no end-to-end deployment speed claim.0additional expert/observations.\n\n"
 text+='[Signed outcomes](signed.tsv), [severe tails](severe_declines.tsv), [information audit](information_audit.json), [protocol](protocol.json).\n\nReferences about possible temporal/location bias, not validation of this method: [Towards Debiasing TSG](https://arxiv.org/abs/2111.04321), [Absolute Position Embeddings](https://aclanthology.org/2022.findings-emnlp.326/). The latter concerns language models, not STVG.\n'
 (P/'ANALYSIS.md').write_text(text);put(P/'verification.json',dict(queries=64,all64_before_GT=True,full_hidden_posterior_bitwise=64,prefix_posterior_bitwise=64,parent_arms_bitwise=64,official_dense_checks=r['checks'],max_error=r['max_metric_error'],paper_sources_unchanged=48,provider_sources_unchanged=434,new_RGB=0,new_visual_encoder=0,new_text_encoder=0,new_expert=0,new_optimizer=0,promoted=False,basic_development_screen=passed,seal_sha256=sha(D/'sealed.json')))
if __name__=='__main__':main()
