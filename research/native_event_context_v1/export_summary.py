"""Export complete development evidence; never select a control as the method."""
import sys
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.native_event_context_v1.run import D,P,read,put,sha

def main():
 r=read(P/'report.json');a=read(P/'information_audit.json');plan=read(D/'plan.json');lines=[]
 for arm,m in r['means'].items():
  t=r['paired'].get(arm+'-NativeG6',{}).get('t',{});lines.append(f"|{arm}|{100*m['v']:.4f}|{100*m['t']:.4f}|{t.get('improved',0)}/{t.get('harmed',0)}/{t.get('unchanged',64)}|{t.get('severe_below_minus5pp',0)}|")
 pair=r['paired']['CensoredContext-NativeG6'];source=r['paired']['CensoredContext-Source'];control=r['paired']['CensoredContext-HullCalibratedFull'];groups=r['groups_vs_G6'];s=read(D/'sealed.json')
 text='''# Overlapping native temporal contexts: HC64

Fixed, exposed development experiment. Full remains paused; no method promotion
from one panel or selection of a favorable control. Full caption and all original
sampled feature values remain unchanged. Two contexts are prefix indices0..48 and
suffix16..64 on the existing65 sampled nodes, each retaining both original offsets.
No image decoding, visual/text encoder, expert call, extra observation or training.
Original crossmodal fusion and native two-pass temporal initialization/decoder run
for each context. This differs from failed full/static, masked-text and temporally
averaged-feature ratios: the evidence comes from actual contiguous feature sequences.

Each cropped posterior is calibrated against its own uninformative zero-endpoint
logit two-offset hull. A hull of uniform endpoint scores is NOT uniform over intervals.
Strict interior start/end endpoints use joint evidence if both visible; if only one
is visible, the other is marginalized. Crop-edge/unavailable endpoints are censored,
not clipped into a negative observation. Mean Prefix/Suffix log evidence reweights
the native posterior, then uses original physical MBR and unchanged complete-box DP.
This is a fixed inference feasibility test, not a calibrated independence claim.

Controls isolate ordinary sharpening, same-information full hull calibration,
reversed context placement and exact uninformative-window no-op. Single-context
arms are retained as diagnostics, not post-hoc candidates.

All metrics percent; improved/harmed/tied time outcomes relative toNativeG6:

|Arm|complete vIoU|tIoU|time +/−/=|time loss>5pp|
|---|---:|---:|---:|---:|
'''+ '\n'.join(lines)+'\n\n'
 text+=f"Primary−NativeG6: v{pair['v']['mean']:+.4f}pp CI{pair['v']['CI95']}, t{pair['t']['mean']:+.4f}pp CI{pair['t']['CI95']}. Primary−Source: v{source['v']['mean']:+.4f}pp, t{source['t']['mean']:+.4f}pp. Primary−same-information HullCalibratedFull: t{control['t']['mean']:+.4f}pp. Shortt{groups['short_le5s']['CensoredContext']['t']:+.4f}pp, longt{groups['long_gt5s']['CensoredContext']['t']:+.4f}pp versusNativeG6.\n\n"
 text+=f"Same-duration/near-native-prior matched candidate feedback macro accuracy{100*a['matched_candidate_pair_macro_accuracy']:.4f}% CI{a['CI95']};{a['matched_candidate_pairs']:,} pairs, native{100*a['native_matched_pair_macro_accuracy']:.4f}%. These are offline diagnostics after all predictions sealed, not online selections.\n\n"
 text+=f"All64 Full hidden/posterior and original Source/NativeG6 outputs restore bitwise; uninformative context is an exact no-op. Independent official/DenseTube checks{r['checks']}, maxerror{r['max_metric_error']:.3g}. All48+434 bound sources unchanged. Actual worker{s['seconds']:.2f}s including source loading, calls{s['counts']}; no end-to-end deployment speed claim.\n\n"
 passed=pair['t']['mean']>0 and pair['v']['mean']>0 and source['t']['mean']>0 and source['v']['mean']>0 and pair['t']['improved']>pair['t']['harmed'] and all(groups[g]['CensoredContext']['t']>0 for g in groups)
 text+=('Primary passes basic development screen only; inspect controls/tails and complete frozen dual-domain recheck before any adoption.'if passed else'Primary does not pass the predeclared broad improvement screen. Preserve all outcomes; do not scan context weights or substitute the best control as a new method.')+'\n\n'
 text+='[Signed outcomes](signed.tsv), [severe tails](severe_declines.tsv), [information audit](information_audit.json), [protocol](protocol.json).\n\nContext/boundary work motivates the question but uses learned models: [MS-2D-TAN](https://arxiv.org/abs/2012.02646), [CBP](https://ojs.aaai.org/index.php/AAAI/article/view/6897). Neither validates this frozen window likelihood rule.\n'
 (P/'ANALYSIS.md').write_text(text);put(P/'verification.json',dict(queries=64,full_prefusion_hidden_posterior_bitwise=64,uninformative_exact_noop=64,all64_before_GT=True,official_dense_checks=r['checks'],max_error=r['max_metric_error'],paper_sources_unchanged=48,provider_sources_unchanged=434,new_RGB=0,new_visual_encoder=0,new_text_encoder=0,new_expert=0,promoted=False,basic_development_screen=passed,seal_sha256=sha(D/'sealed.json')))
if __name__=='__main__':main()
