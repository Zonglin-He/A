"""Independent saved-scalar audit and full cached64 component report."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,collections,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_projection_components64_v1 import D,P,OLD,ARMS,GEOMETRIES,SUPPORTS,read,put,load,sha,avg

def run():
 private=read(D/'POSTSEAL_SCORED_PRIVATE.json')['rows'];checks=0;results={ds:read(P/f'RESULTS_{ds}.json')for ds in ['hc2','vidstg']}
 for ds,z in results.items():
  assert len({r['alias']for r in z['rows']})==len({r['parent_alias']for r in z['rows']})==32
  for r in z['rows']:
   assert not any(k in m for m in r['values'].values()for k in ['pred_to_GT_physical_length_ratio','pred_to_GT_metric_span_ratio'])
  for geo in GEOMETRIES:
   for sup in SUPPORTS:
    rr=[r for r in private if(r['dataset'],r['geometry'],r['support'])==(ds,geo,sup)];s=z['groups'][geo][sup]
    for a in ARMS:
     for k,v in s['overall'][a]['query_mean'].items():
      vv=[r['values'][a][k]for r in rr if r['values'][a][k]is not None];expected=sum(vv)/len(vv)if vv else None;assert v==expected or abs(v-expected)<1e-12;checks+=1
    parents=sorted({r['parent_alias']for r in rr});idx=np.random.default_rng(20261018).integers(len(parents),size=(10000,len(parents)))
    for pair,metrics in s['paired'].items():
     a,b=pair.split('__minus__')
     for k,v in metrics.items():
      y=np.array([sum(r['values'][a][k]-r['values'][b][k]for r in rr if r['parent_alias']==p)/sum(r['parent_alias']==p for r in rr)for p in parents]);assert abs(y.mean()-v['mean'])<1e-12;assert np.max(abs(np.quantile(y[idx].mean(1),[.025,.975])-v['CI95']))<1e-12;checks+=3
 files=read(D/'GLOBAL_READOUT_BARRIER.json')['files'];aud=[];configs={}
 for f in files:
  z=load(ROOT/f['path']);configs[z['dataset']]={k:v for k,v in z['actual_config'].items()if k in ['checkpoint_sha256','source_dataset','prior_weight','center_fraction','epsilon']};aud.extend(z['audit'])
 b=read(D/'GLOBAL_READOUT_BARRIER.json');put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=64,independent_saved_mean_paired_and_CI_checks=checks,direct_enumerated_checks=b['enumerated_checks'],maximum_decomposition_error=b['maximum_decomposition_error'],all_full_projection_and_MBR_decisions_exact=True,source_inputs_and_registries_unchanged=True,private_exact_GT_ratio_removed=True));put(P/'ACTUAL_CONFIG.json',dict(domains=configs,ARMS=ARMS,primary_geometry='After',bootstrap_parent_draws=10000,seed=20261018,physical_cells_vs_endpoint_intervals_distinct=True,no_new_model_training_or_parameter_update=True))
 lines=['# Exact cached64 projection length and fluctuation decomposition','','Same exposed32 parent queries/domain. Original Native time, same cached two offsets and original physical envelope. Fixed geometry is crossed with each temporal readout: **After is primary**, Source and inherited Before are conditional attribution controls. CPU-only, no new model/expert/Student fitting, training or new online stream.','','## Algebra and decision rules','','For the original standardized actionness u and normalized physical-cell weights omega, bar_u=sum omega*u. Every legal sampled-index interval I has L=bar_u*sum_I omega and F=sum_I omega*(u-bar_u). Original score is C + lambda*lp0 + L + F, where C=sum omega*logsigmoid(-u) is candidate-independent and lambda=.1. Each direct sliced-sum enumeration agrees with the original score and prefix decomposition (maximum error '+str(b['maximum_decomposition_error'])+'). Original full projection uses the original implementation and target exactly; Native preserves its original float32 MAP/ties. Component decisions use original float64 legal indices s<e and first argmax. MBR is copied from old sealed decisions, with no algorithm rerun.','','Omega interval mass uses Voronoi midpoint cells; physical output duration uses [fid_s,fid_e+1). They differ, including at endpoint cells. Removing the zero-mean fluctuation removes location-varying actionness but retains the Native prior; removing L preserves that prior and centered fluctuations. bar_u may be negative: length pressure is not assumed always expansive.','','## Complete fused readout metrics','','All metric columns are percentages. R03/R05 use strict vIoU>.3/>.5. sIoU is measured on all known GT frames and is invariant to temporal-only decisions.','','| Domain | Fixed geometry | Readout | vIoU | tIoU | sIoU | R03 | R05 |','|---|---|---|---:|---:|---:|---:|---:|']
 for ds,z in results.items():
  for geo in GEOMETRIES:
   for a in ARMS:
    m=z['groups'][geo]['fused']['overall'][a]['query_mean'];lines.append(f'| {ds} | {geo} | {a} | '+' | '.join(f'{100*m[k]:.6f}'for k in ['v','t','s','R03','R05'])+' |')
 lines+=['','## Primary After paired parent intervals','','Differences in percentage points. Parent bootstrap10000 fixed draws; every parent/query remains, positive/negative/neutral are counted before aggregation. Intervals are descriptive on historically exposed sources and multiple contrasts, not a new independent confirmation.','','| Domain | Contrast | Δv pp [CI95] | Δt pp [CI95] | Positive / negative / neutral parents (v) |','|---|---|---:|---:|---:|']
 for ds,z in results.items():
  for name,v in z['groups']['After']['fused']['paired'].items():
   f=lambda k:f'{100*v[k]["mean"]:+.6f} [{100*v[k]["CI95"][0]:+.6f}, {100*v[k]["CI95"][1]:+.6f}]';c=v['v'];lines.append(f'| {ds} | {name} | {f("v")} | {f("t")} | {c["positive"]}/{c["negative"]}/{c["neutral"]} |')
 lines+=['','These changes are separate argmax decisions. Full−Native is not the sum of (length-only−Native) and (fluctuation-only−Native); the discrete argmax and fused-envelope operation are nonlinear. Full−one-component is conditional evidence at the same prior, not a causal percentage or a decomposition of task gain. No component winner is selected for deployment.','','## Event strata and expansion support','','Event-length/start bins reuse the full13785 pooled boundaries after all readouts are sealed; they are GT-derived offline descriptions, not online routing. Ratios below use corrected physical GT event duration; legacy metric endpoints remain unchanged. Exact per-query GT durations and ratios are private; public rows retain only ratio bins, signed metrics and known/unknown support counts.','','| Domain | Readout (After fused) | Pred / physical-GT duration | Pred / metric-span duration | Added known frames/query | IoU on added-known support | Queries with known additions | Added unknown frames/query |','|---|---|---:|---:|---:|---:|---:|---:|']
 for ds,z in results.items():
  for a in ARMS:
   o=z['groups']['After']['fused']['overall'][a];m=o['query_mean'];v='unknown'if m['added_known_IoU']is None else f'{100*m["added_known_IoU"]:.3f}%';lines.append(f'| {ds} | {a} | {m["pred_to_GT_physical_length_ratio"]:.6f} | {m["pred_to_GT_metric_span_ratio"]:.6f} | {m["added_known_frames"]:.3f} | {v} | {o["known_queries"]["added_known_IoU"]} | {m["added_unknown_frames"]:.3f} |')
 lines+=['','Added-frame IoU averages queries that actually have known added frames; missing additions are unknown, not zero-filled. Counts and lost-frame IoU/unknowns remain in every domain result. Temporal expansion may include missing/unannotated support; no GT geometry is invented there.','','| Domain | Offline stratum | Bin | Queries / parents | Native v | Full v | Length v | Fluctuation v |','|---|---|---|---:|---:|---:|---:|---:|']
 for ds,z in results.items():
  for key,ss in z['groups']['After']['fused']['strata'].items():
   for label,v in ss.items():
    n=v['Native'];lines.append(f'| {ds} | {key} | {label} | {n["queries"]}/{n["parents"]} | '+' | '.join(f'{100*v[a]["query_mean"]["v"]:.6f}'for a in ARMS[:4])+' |')
 lines+=['','Full per-offset and fused metrics, Source/Before/After, every signed per-query effect, parent means and bootstrap intervals, R03/R05, bins, unknowns and legacy MBR are in RESULTS_hc2.json and RESULTS_vidstg.json. The original official span semantics and clipping are reused, independently checked against DenseTube. Original Native/projection/MBR metrics match the prior temporal64 audit exactly.','','## Scope and limitations','','This is a four-readout zero-parameter diagnostic on64 exposed cached queries, not full13785 MBR or new temporal learning. The actionness source, standardization, Native prior and spatial states are unchanged; any common calibration issue remains. Fixed geometry has not been recomputed after changing time. Length-only still uses actionness-derived bar_u and is not Actionness-free. Fluctuations are not certified semantic event evidence. No training, parameter sweep, GT-based choice, new images/semantic labels, production change or automatic follow-up is authorized. All existing positive, negative and unknown findings remain.','','![Primary fused component effects](COMPONENTS.svg)']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
 for ax,(ds,z)in zip(axes,results.items()):
  pp=z['groups']['After']['fused']['paired'];names=ARMS[1:];v=np.array([pp[a+'__minus__Native']['v']['mean']*100 for a in names]);cis=np.array([pp[a+'__minus__Native']['v']['CI95']for a in names])*100;ax.bar(range(4),v,color=['#2f6f80','#c88431','#5a6d9c','#7a7b7e']);ax.errorbar(range(4),v,yerr=np.stack([v-cis[:,0],cis[:,1]-v]),fmt='none',ecolor='black',capsize=4);ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(4),['Full','Length','Fluctuation','MBR ref']);ax.set_title(ds+' | fixed online After, n=32');ax.set_ylabel('vIoU change vs Native (pp)');ax.grid(axis='y',alpha=.2)
 fig.suptitle('Cached64 temporal component decisions | paired parent95% intervals');fig.savefig(P/'COMPONENTS.svg');fig.savefig(D/'COMPONENTS.png',dpi=150);plt.close(fig)
 files=['scripts/run_stvg_projection_components64_v1.py','scripts/report_stvg_projection_components64_v1.py','protocols/stvg_projection_components64_v1.md']+[str((P/f).relative_to(ROOT))for f in ['RESULTS_hc2.json','RESULTS_vidstg.json','VERIFICATION.json','ROOT_INDEPENDENT_AUDIT.json','ACTUAL_CONFIG.json','REPORT.md','COMPONENTS.svg']];put(D/'PUBLIC_FILES.json',dict(files=files));print('REPORT_AUDIT_COMPLETE',checks,flush=True)
if __name__=='__main__':run()
