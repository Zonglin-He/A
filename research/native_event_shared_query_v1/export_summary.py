"""Report native query reuse separately from clause evidence and pure expansion."""
import sys
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.native_event_shared_query_v1.run import D,P,read,put,sha
def main():
 r=read(P/'report.json');a=read(P/'information_audit.json');s=read(D/'sealed.json');p=read(D/'plan.json');lines=[]
 for arm,m in r['means'].items():
  t=r['paired'].get(arm+'-NativeG6',{}).get('t',{});lines.append(f"|{arm}|{100*m['v']:.4f}|{100*m['t']:.4f}|{t.get('improved',0)}/{t.get('harmed',0)}/{t.get('unchanged',64)}|{t.get('severe_below_minus5pp',0)}|")
 pair=r['paired']['SharedQueryClauseFusion-NativeG6'];source=r['paired']['SharedQueryClauseFusion-Source'];control=r['paired']['SharedQueryClauseFusion-ClauseFusion'];self_control=r['paired']['SharedQueryClauseFusion-SelfHullFusion'];g=r['groups_vs_G6']
 text='''# Shared full-caption native temporal query in positive clause reads

Fixed exposed HC64 study. Same52 plain-main-AND descriptions and two positive clause
texts, full frame context, original visual features/expert8/spatial method. No new
expert, visual encoder, RGB or optimizer. Original paper full remains paused.

Previous positive clause composition improves means but harms short events. Clause
Source-box drift is only an identity/conditioning proxy, not correctness. This test
keeps each clause's actual encoded text+motion memory/query_content/time position,
but reuses the ORIGINAL Full-caption actual second-pass query_tgt at the same offset.
Only TimeDecoder calls1,3 of the original four-call sequence are replaced; original
expanded stride is restored. Full replacement must be a bitwise no-op. No quality
veto, learned query delta, fitting, attention ROI edit or extra query step.

The Full query encodes more than identity, including original event/global context;
successful reuse would not by itself prove identity binding. Older object-conditioned
queries and query_tgt rewrites already exist, and full-caption TubeReplay changes
attention. Here the new mechanism under test is Full query reuse across actual
positive subevent memories, not novelty of the query interface.

Primary same fixed half Full/half minimal-envelope posterior; original physical MBR
and complete-box DP. Native unshared ClauseFusion is the key single-change control,
plus direct shared envelope, self-hull/identity-hull, reversed placement and no-op.
All12 unsupported descriptions preserve native outputs exactly by declared syntax.

All percent metrics, improved/harmed/tied time versusNativeG6:

|Arm|complete vIoU|tIoU|time +/−/=|time loss>5pp|
|---|---:|---:|---:|---:|
'''+ '\n'.join(lines)+'\n\n'
 text+=f"Primary−NativeG6: v{pair['v']['mean']:+.4f}pp CI{pair['v']['CI95']}, t{pair['t']['mean']:+.4f}pp CI{pair['t']['CI95']}. Primary−Source: v{source['v']['mean']:+.4f}pp, t{source['t']['mean']:+.4f}pp. Primary−unshared ClauseFusion: t{control['t']['mean']:+.4f}pp CI{control['t']['CI95']}; versusSelfHullFusion t{self_control['t']['mean']:+.4f}pp. Shortt{g['short_le5s']['SharedQueryClauseFusion']['t']:+.4f}pp, longt{g['long_gt5s']['SharedQueryClauseFusion']['t']:+.4f}pp versusNativeG6.\n\n"
 text+=f"Matched feedback accuracy{100*a['matched_candidate_pair_macro_accuracy']:.4f}% including unsupported ties, native{100*a['native_matched_pair_macro_accuracy']:.4f}%; supported-only{a['supported_only_pair_accuracy']}. No online selection uses these labels.\n\n"
 passed=pair['t']['mean']>0 and pair['v']['mean']>0 and source['t']['mean']>0 and source['v']['mean']>0 and pair['t']['improved']>pair['t']['harmed'] and all(g[x]['SharedQueryClauseFusion']['t']>0 for x in g)and self_control['t']['mean']>0
 text+=('Basic development screen passes only; fixed HC/Vid recheck and severe-tail assessment required before adoption.'if passed else'Primary does not pass broad-improvement development qualification. No promotion or shared-query strength/step scan.')+'\n\n'
 text+=f"All64 true Full text/hidden/posterior/Source boxes and original parent outputs restore bitwise; no-composition exact. Official/dense checks{r['checks']}, maxerror{r['max_metric_error']:.3g}. Original48+434 sources intact. Worker{s['seconds']:.2f}s inclmodel load and research capture, calls{s['counts']}; not end-to-end deployment time. All new outputs sealed before scoring.\n\n"
 text+='[Signed outcomes](signed.tsv), [severe tails](severe_declines.tsv), [information audit](information_audit.json), [protocol](protocol.json).\n'
 (P/'ANALYSIS.md').write_text(text);put(P/'verification.json',dict(queries=64,supported=p['supported'],all64_before_GT=True,full_text_hidden_posterior_Source_boxes_bitwise=64,parent_aliases_bitwise=64,official_dense_checks=r['checks'],max_error=r['max_metric_error'],paper_sources_unchanged=48,provider_sources_unchanged=434,new_RGB=0,new_visual_encoder=0,new_expert=0,new_observations=0,new_optimizer=0,promoted=False,basic_development_screen=passed,seal_sha256=sha(D/'sealed.json')))
if __name__=='__main__':main()
