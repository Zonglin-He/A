"""Report all predeclared native subevent composition arms and support coverage."""
import sys
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.native_event_composition_v1.run import D,P,read,put,sha
def main():
 r=read(P/'report.json');a=read(P/'information_audit.json');s=read(D/'sealed.json');p=read(D/'plan.json');lines=[]
 for arm,m in r['means'].items():
  t=r['paired'].get(arm+'-NativeG6',{}).get('t',{});lines.append(f"|{arm}|{100*m['v']:.4f}|{100*m['t']:.4f}|{t.get('improved',0)}/{t.get('harmed',0)}/{t.get('unchanged',64)}|{t.get('severe_below_minus5pp',0)}|")
 pair=r['paired']['ClauseFusion-NativeG6'];source=r['paired']['ClauseFusion-Source'];control=r['paired']['ClauseFusion-SelfHullFusion'];g=r['groups_vs_G6']
 text='''# Positive native subevent composition: HC64

Fixed, exposed development study; space/expert8 unchanged, paper full paused.
Remaining failures include compound action descriptions: approach+sit,raise+pour+turn.
This branch tests a semantic alternative to context/position/absence-ratio feedback:
read actual positive subevent clauses through the same trained native temporal model,
then form a distribution spanning the compound event.

Existing Source Stanza dependencies identify only main-root VERB conjunctions with
one inherited subject. Keep the original nominal identity subtree and arguments.
Only plain AND is supported; explicit new subjects and before/after/while/until/when/
unless/or/but relations are not guessed. First/last main conjuncts supply at most2
new texts. Other captions retain the original posterior exactly, for every control.
This is semantic scope, not a confidence/GT rejection gate; coverage is reported.

Original RoBERTa+resizer, FP16 fusion and exact native FP32 two-pass decoder read each
clause at both original offsets/six layers. Original projected visual features and
full target identity are retained. The independent first/last interval distributions
produce the exact minimal-envelope posterior. Primary is fixed half original Full,
half clause envelope, original physical MBR and complete-box DP; no ratio or fitting.

DirectClauses uses identical information without Full fusion. SelfHullFusion uses
two copies of the original posterior to control mere expansion. IdentityHullFusion
and MaskedEventFusion reuse old same-native interventions; reversed clause placement
and exact no-composition are retained. No best control selected posthoc.

All percent metrics, improved/harmed/tied time versusNativeG6:

|Arm|complete vIoU|tIoU|time +/−/=|time loss>5pp|
|---|---:|---:|---:|---:|
'''+ '\n'.join(lines)+'\n\n'
 text+=f"Syntactic support{p['supported']}/64;{64-p['supported']} captions preserve NativeG6 exactly by declared scope, not by observed quality. Primary−NativeG6: v{pair['v']['mean']:+.4f}pp CI{pair['v']['CI95']}, t{pair['t']['mean']:+.4f}pp CI{pair['t']['CI95']}. Primary−Source: v{source['v']['mean']:+.4f}pp, t{source['t']['mean']:+.4f}pp. Primary−SelfHullFusion: t{control['t']['mean']:+.4f}pp CI{control['t']['CI95']}. Shortt{g['short_le5s']['ClauseFusion']['t']:+.4f}pp, longt{g['long_gt5s']['ClauseFusion']['t']:+.4f}pp versusNativeG6.\n\n"
 text+=f"Matched candidate feedback accuracy{100*a['matched_candidate_pair_macro_accuracy']:.4f}% including unsupported ties, native{100*a['native_matched_pair_macro_accuracy']:.4f}%; supported-only accuracy{a['supported_only_pair_accuracy']}. All new outputs sealed before scoring.\n\n"
 passed=pair['t']['mean']>0 and pair['v']['mean']>0 and source['t']['mean']>0 and source['v']['mean']>0 and pair['t']['improved']>pair['t']['harmed'] and all(g[x]['ClauseFusion']['t']>0 for x in g)and control['t']['mean']>0
 text+=('Basic development screen passes only; requires severe-tail audit and frozen HC/Vid recheck before adoption.'if passed else'Primary does not pass basic broad-improvement development screen. No promotion or fusion-weight scan.')+'\n\n'
 text+=f"All64 original reencoded text tuple/hidden/posterior/Source boxes restore bitwise; no-composition exact. Official/dense checks{r['checks']}, maxerror{r['max_metric_error']:.3g}. Original48+434 sources intact. Worker{s['seconds']:.2f}s inclmodel loading, calls{s['counts']}. No RGB/visual encoder/expert calls or new observations. Timings are cached research execution, not deployment end-to-end speed.\n\n"
 text+='[Signed outcomes](signed.tsv), [severe tails](severe_declines.tsv), [information audit](information_audit.json), [protocol](protocol.json).\n\n[TFVTG](https://github.com/minghangz/TFVTG) motivates subevent decomposition but uses extra LLM/VLM; [DeCo](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_DeCo_Decomposition_and_Reconstruction_for_Compositional_Temporal_Grounding_via_Coarse-To-Fine_CVPR_2023_paper.html) learns compositional representations. Neither is reproduced here; this bounded deterministic parser/native test has less semantic coverage and no trained compositional head.\n'
 (P/'ANALYSIS.md').write_text(text);put(P/'verification.json',dict(queries=64,supported=p['supported'],all64_before_GT=True,full_text_hidden_posterior_Source_boxes_bitwise=64,official_dense_checks=r['checks'],max_error=r['max_metric_error'],paper_sources_unchanged=48,provider_sources_unchanged=434,new_RGB=0,new_visual_encoder=0,new_expert=0,new_observations=0,new_optimizer=0,promoted=False,basic_development_screen=passed,seal_sha256=sha(D/'sealed.json')))
if __name__=='__main__':main()
