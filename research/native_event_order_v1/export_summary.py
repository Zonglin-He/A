"""Keep semantic-order and matched self-order controls, including all harms."""
import sys
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.native_event_order_v1.run import D,P,read,put,sha
def main():
 r=read(P/'report.json');a=read(P/'information_audit.json');s=read(D/'sealed.json');p=read(D/'plan.json');lines=[]
 for arm,m in r['means'].items():
  t=r['paired'].get(arm+'-NativeG6',{}).get('t',{});lines.append(f"|{arm}|{100*m['v']:.4f}|{100*m['t']:.4f}|{t.get('improved',0)}/{t.get('harmed',0)}/{t.get('unchanged',64)}|{t.get('severe_below_minus5pp',0)}|")
 pair=r['paired']['OrderedClauseFusion-NativeG6'];source=r['paired']['OrderedClauseFusion-Source'];control=r['paired']['OrderedClauseFusion-SelfOrderFusion'];g=r['groups_vs_G6']
 text='''# Joint first/last subevent boundary decisions with narrative order

Fixed cache-only follow-up, exposed HC64. Same52 main-AND queries, unchanged native
positive clause outputs/full captions/space/expert8. No model/expert/optimizer call.
Prior unordered hull improves average time but short events lose5.7702pp. Concrete
RC014,RC054,RC026 show the standalone final action selecting an earlier occurrence,
which unordered min/max can incorrectly span. Marginal first-start error is NOT
inherently lower; this test depends on jointly resolving occurrences using order.

Condition interval pairs on startFirst<=startLast AND endFirst<=endLast, permitting
overlap. Sum all legal ordered pairs into[startFirst,endLast], retaining full probability
mass within the fixed relation. Primary mixes half original Full, half conditional
ordered posterior; original physical MBR/complete-box DP. Valid-order mass is logged,
never a rejection gate. Narrative AND is not guaranteed chronological; this is an
explicit semantic assumption tested here, not supplied by a new event expert.

DirectOrdered uses the same information. RoleOnly omits inner endpoint order;
SwappedOrder exchanges semantic first/last roles. SelfOrder uses two copies of the
original posterior and controls new structural/length effects. Prior unordered hull
and self-hull remain, as does exact no-composition. No weight or threshold sweep.

All percent metrics, improved/harmed/tied time versusNativeG6:

|Arm|complete vIoU|tIoU|time +/−/=|time loss>5pp|
|---|---:|---:|---:|---:|
'''+ '\n'.join(lines)+'\n\n'
 text+=f"Primary−NativeG6: v{pair['v']['mean']:+.4f}pp CI{pair['v']['CI95']}, t{pair['t']['mean']:+.4f}pp CI{pair['t']['CI95']}. Primary−Source: v{source['v']['mean']:+.4f}pp, t{source['t']['mean']:+.4f}pp. Primary−SelfOrderFusion: t{control['t']['mean']:+.4f}pp CI{control['t']['CI95']}. Shortt{g['short_le5s']['OrderedClauseFusion']['t']:+.4f}pp, longt{g['long_gt5s']['OrderedClauseFusion']['t']:+.4f}pp versusNativeG6.\n\n"
 text+=f"Support{p['supported']}/64, all remaining outputs exactNative. Matched feedback accuracy{100*a['matched_candidate_pair_macro_accuracy']:.4f}% including unsupported ties, native{100*a['native_matched_pair_macro_accuracy']:.4f}%; supported-only{a['supported_only_pair_accuracy']}.\n\n"
 passed=pair['t']['mean']>0 and pair['v']['mean']>0 and source['t']['mean']>0 and source['v']['mean']>0 and pair['t']['improved']>pair['t']['harmed'] and all(g[x]['OrderedClauseFusion']['t']>0 for x in g)and control['t']['mean']>0
 text+=('Basic development screen passes only; frozen HC/Vid recheck and tails still required before adoption.'if passed else'Primary does not pass the broad-improvement development screen. Preserve positive means and all short-event/severe harms; no promotion or order-strength scan.')+'\n\n'
 text+=f"All64 parent aliases/Source/NativeG6 restore bitwise, no-composition exact. Official/dense checks{r['checks']}, maxerror{r['max_metric_error']:.3g}. Original48+434 sources intact. Worker{s['seconds']:.2f}s CPU cached arithmetic, calls{s['counts']}; not end-to-end deployment time. All new outputs sealed before labels/scoring.\n\n"
 text+='[Signed outcomes](signed.tsv), [severe tails](severe_declines.tsv), [information audit](information_audit.json), [protocol](protocol.json).\n'
 (P/'ANALYSIS.md').write_text(text);put(P/'verification.json',dict(queries=64,supported=p['supported'],all64_before_GT=True,parent_Source_Native_aliases_bitwise=64,official_dense_checks=r['checks'],max_error=r['max_metric_error'],paper_sources_unchanged=48,provider_sources_unchanged=434,new_RGB=0,new_visual_encoder=0,new_text_encoder=0,new_decoder=0,new_expert=0,new_optimizer=0,promoted=False,basic_development_screen=passed,seal_sha256=sha(D/'sealed.json')))
if __name__=='__main__':main()
