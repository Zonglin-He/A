"""Publish all controls and outcomes without adopting a development winner."""
import sys
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.native_event_innovation_v1.run import D,P,read,put,sha
def main():
 r=read(P/'report.json');a=read(P/'information_audit.json');s=read(D/'sealed.json');lines=[]
 for arm,m in r['means'].items():
  t=r['paired'].get(arm+'-NativeG6',{}).get('t',{});lines.append(f"|{arm}|{100*m['v']:.4f}|{100*m['t']:.4f}|{t.get('improved',0)}/{t.get('harmed',0)}/{t.get('unchanged',64)}|{t.get('severe_below_minus5pp',0)}|")
 pair=r['paired']['ContextInnovation-NativeG6'];source=r['paired']['ContextInnovation-Source'];control=r['paired']['ContextInnovation-SameInformationContext'];g=r['groups_vs_G6']
 text='''# Context-induced information versus unchanged full-context logits

Isolated fixed follow-up on exposed HC64; spatial method unchanged and full paused.
The previous context rule lowered tIoU6.8349pp; native mean interval8.648s became
7.293s. Same-information full hull calibration was even worse. RC059 lost68.90pp
tIoU when its end moved320frames earlier; RC056 lost46.05pp with inward endpoints.
This motivates separating actual context changes from the confidence already in
original logits, rather than selecting another likelihood exponent.

For the same two fixed windows, take all six original full-context endpoint logits,
slice at exactly the same physical nodes/parity and construct the same native hull
posterior bW. This uses no model call. Compare actual cropped-context qW with bW,
using the same strictly-interior joint/one-boundary marginal censoring. Primary
q0*exp(mean log(qW/bW)) changes only on context-induced prediction differences.
If cropped-context logits equal the original sliced logits, feedback is exactly zero.

SameInformationContext uses bW/uW with the original structural null and isolates
reusing original confidence. Original context/sharpening/reverse/no-innovation
controls are retained. The main coefficient/windows/MBR/space are fixed, no tuning.
This ratio is not assumed calibrated for an event crossing a crop boundary.

All percent metrics, improved/harmed/tied time versusNativeG6:

|Arm|complete vIoU|tIoU|time +/−/=|time loss>5pp|
|---|---:|---:|---:|---:|
'''+ '\n'.join(lines)+'\n\n'
 text+=f"Primary−NativeG6: v{pair['v']['mean']:+.4f}pp CI{pair['v']['CI95']}, t{pair['t']['mean']:+.4f}pp CI{pair['t']['CI95']}. Primary−Source: v{source['v']['mean']:+.4f}pp, t{source['t']['mean']:+.4f}pp. Primary−SameInformationContext: t{control['t']['mean']:+.4f}pp. Shortt{g['short_le5s']['ContextInnovation']['t']:+.4f}pp, longt{g['long_gt5s']['ContextInnovation']['t']:+.4f}pp versusNativeG6.\n\n"
 text+=f"Matched same-duration/near-native-prior candidate feedback accuracy{100*a['matched_candidate_pair_macro_accuracy']:.4f}% CI{a['CI95']}, native{100*a['native_matched_pair_macro_accuracy']:.4f}%;{a['matched_candidate_pairs']:,}pairs. All predictions sealed before scoring; all signed outcomes and severe tails retained.\n\n"
 passed=pair['t']['mean']>0 and pair['v']['mean']>0 and source['t']['mean']>0 and source['v']['mean']>0 and pair['t']['improved']>pair['t']['harmed'] and all(g[x]['ContextInnovation']['t']>0 for x in g)
 text+=('Basic development screen passes, still requires frozen dual-domain recheck; no default promotion.'if passed else'Primary fails basic broad-improvement screen. Do not promote, scan strength or substitute a control.')+'\n\n'
 text+=f"All64 NativeG6/Source and reused parent arms bitwise; no-innovation exact no-op64. Official/dense checks{r['checks']}, maxerror{r['max_metric_error']:.3g}. Original48+434 sources intact. Worker{s['seconds']:.2f}s, entirely CPU cached outputs;0RGB/encoder/decoder/expert/optimizer calls.\n\n"
 text+='[Signed outcomes](signed.tsv), [severe tails](severe_declines.tsv), [information audit](information_audit.json), [protocol](protocol.json).\n'
 (P/'ANALYSIS.md').write_text(text);put(P/'verification.json',dict(queries=64,all64_before_GT=True,no_innovation_exact_noop=64,source_native_and_parent_arms_bitwise=64,official_dense_checks=r['checks'],max_error=r['max_metric_error'],paper_sources_unchanged=48,provider_sources_unchanged=434,new_RGB=0,new_visual_encoder=0,new_text_encoder=0,new_expert=0,new_decoder=0,promoted=False,basic_development_screen=passed,seal_sha256=sha(D/'sealed.json')))
if __name__=='__main__':main()
