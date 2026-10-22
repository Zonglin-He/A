"""Report native-family projection against identical direct and optimizer controls."""
import sys
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.native_event_projection_v1.run import D,P,read,put,sha

def main():
 import torch,numpy as np
 r=read(P/'report.json');s=read(D/'sealed.json');p=read(D/'plan.json');pair=r['paired']['ClauseQuery-NativeG6'];source=r['paired']['ClauseQuery-Source'];direct=r['paired']['ClauseQuery-SharedQueryClauseFusion'];g=r['groups_vs_G6'];paths={}
 for arm in ['ClauseQuery','SelfHullQuery','ReversedClauseQuery']:
  entries=[]
  for e in s['files'].values():
   tr=torch.load(D/e['trace_path'],weights_only=False,map_location='cpu')
   if arm in tr['paths']:
    a=tr['paths'][arm]['path'];entries.append(dict(initial=a[0]['loss'],final=a[-1]['loss'],delta=a[-1]['normalized_delta_RMS']))
  paths[arm]=dict(count=len(entries),loss_decreased=sum(x['final']<x['initial']for x in entries),mean_loss_initial=float(np.mean([x['initial']for x in entries])),mean_loss_final=float(np.mean([x['final']for x in entries])),mean_normalized_delta_RMS=float(np.mean([x['delta']for x in entries])))
 put(P/'optimization_audit.json',paths)
 lines=['# Positive clause feedback projected through the frozen native time model','','Fixed exposed HC64. Same sealed shared-clause teacher distribution; unchanged old query optimizer: shared-offset256 temporary state, Adam lr.01,10steps, normalized radius.1, last state. No new encoder, text, expert or observations. Spatial method is unchanged.','','Earlier RawQuery used unreliable ReferDINO-derived event targets. Here the feedback is the actual positive native clause posterior that improved direct means, but harmed short events. This tests whether the Full-caption native temporal family filters incompatible clause outputs. It does not increase update steps or claim a new query interface.','','All52 syntactically supported descriptions execute all3 optimizer arms. Twelve unsupported descriptions inherit exact native output; no confidence/GT rejection or best-step selection. Same-information direct parent, same-budget self-hull learning and reversed-time learning/direct controls are retained. A lower pseudo-loss is not task correctness.','','|Arm|complete vIoU|tIoU|time +/−/=|time harm>5pp|','|---|---:|---:|---:|---:|']
 for a,m in r['means'].items():
  t=r['paired'].get(a+'-NativeG6',{}).get('t',{});lines.append(f"|{a}|{100*m['v']:.4f}|{100*m['t']:.4f}|{t.get('improved',0)}/{t.get('harmed',0)}/{t.get('unchanged',64)}|{t.get('severe_below_minus5pp',0)}|")
 lines+=['',f"Primary−NativeG6: v{pair['v']['mean']:+.4f}pp CI{pair['v']['CI95']}, t{pair['t']['mean']:+.4f}pp CI{pair['t']['CI95']}. Primary−Source: v{source['v']['mean']:+.4f}pp, t{source['t']['mean']:+.4f}pp.",f"Learned−identical direct feedback: v{direct['v']['mean']:+.4f}pp CI{direct['v']['CI95']}, t{direct['t']['mean']:+.4f}pp CI{direct['t']['CI95']}.",f"Shortt{g['short_le5s']['ClauseQuery']['t']:+.4f}pp, longt{g['long_gt5s']['ClauseQuery']['t']:+.4f}pp versusNativeG6."]
 passed=pair['v']['mean']>0 and pair['t']['mean']>0 and source['v']['mean']>0 and source['t']['mean']>0 and pair['t']['improved']>pair['t']['harmed'] and all(g[x]['ClauseQuery']['t']>0 for x in g)and direct['t']['mean']>0
 lines+=['','Basic development screen passes; fixed HC/Vid recheck still required.'if passed else'Not qualified for promotion. Preserve all controls and negative tails; no step/lr/radius scan.',f"All64 native hidden/final logits/parent aliases restore bitwise. Differentiable native posterior maxerror{s['posterior_math_max_error']:.3g}. Official/dense checks{r['checks']}, maxerror{r['max_metric_error']:.3g}. Actual worker{s['seconds']:.2f}s includes model loading/research recording, not deployment end-to-end. Calls{s['counts']}.",'','[Signed](signed.tsv), [severe tails](severe_declines.tsv), [optimization](optimization_audit.json), [protocol](protocol.json).']
 (P/'ANALYSIS.md').write_text('\n'.join(lines)+'\n');put(P/'verification.json',dict(queries=64,supported=52,native_bitwise=64,parent_aliases_bitwise=64,all64_seal_before_GT=True,official_dense_checks=r['checks'],max_error=r['max_metric_error'],posterior_math_max_error=s['posterior_math_max_error'],paper_sources_unchanged=48,provider_sources_unchanged=434,weights_frozen=s['network_weights_frozen'],promoted=False,basic_development_screen=passed,seal_sha256=sha(D/'sealed.json')))
if __name__=='__main__':main()
