"""Post-barrier paired five-row evaluation; no labels used in inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import numpy as np
from scripts.run_stvg_spatial_strong_bidirectional256_v1 import D,P,ROWS,read,put,sha,load,verify

def paired(d):
 x=np.asarray(d,dtype=float);g=np.random.default_rng(20261020);samples=np.concatenate([x[g.integers(0,len(x),(100,len(x)))].mean(1)for _ in range(100)])
 return dict(mean=float(x.mean()),CI95=np.quantile(samples,[.025,.975]).tolist(),n=len(x))

def main():
 import torch
 torch.set_num_threads(4);p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['parents']==256 and bar['all_before_GT'];start=time.perf_counter()
 from methods.stvg_referdino_opd_v1.temporal import final_mbr
 from vg_tta.stvg_posterior_decision_tree_v1 import brute_utility
 checks=0;utility_err=0.
 for ds in ['hc2','vidstg']:
  b=read(D/f'BARRIER_{ds}.json');assert sha(D/f'BARRIER_{ds}.json')==bar['barriers'][ds]and b['count']==128
  for f,h in b['files'].items():
   assert sha(R/f)==h;z=load(R/f);assert not z['GT_read']
   for kind,q in z['posterior'].items():
    dec=final_mbr(z['frame_ids'],q);assert dec['interval']==z['intervals'][kind];ii,jj=np.triu_indices(len(q),1);pick=np.unique(np.r_[np.linspace(0,len(ii)-1,19).astype(int),np.where((ii==dec['indices'][0])&(jj==dec['indices'][1]))[0]]);err=float(abs(brute_utility(z['frame_ids'],q,pick)-dec['utility_grid'][ii[pick],jj[pick]]).max());assert err<1e-10;utility_err=max(utility_err,err);checks+=len(pick)
 put(D/'PRE_GT_AUDIT.json',dict(status='pass',posterior_and_direct_utility_checks=checks,max_error=utility_err,GT_read=False))
 put(D/'GT_EXPOSURE.json',dict(epoch=time.time(),global_barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json'),all256_predictions_sealed=True))
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official,box_iou
 from vg_tta.tastvg_paper48_metrics_v1 import xyxy
 allresults={};officialchecks=0
 for ds in ['hc2','vidstg']:
  records=[r for r in p['records']if r['dataset']==ds];raw=read(R/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows'];gt,spans,prov=truths(ds,dict(parents=[r['query_ordinal']for r in records]));rows=[]
  for rec in records:
   z=load(D/'predictions'/ds/f"{rec['alias']}.pt");qidx=rec['query_ordinal'];r=raw[qidx];values={};seedvalues={};fixed_support=[f for f in z['frame_ids']if f in gt[qidx]];unobserved=[f for f in fixed_support if f not in {z['frame_ids'][i]for i in z['planned_positions']}]
   def metric(box,iv):
    nonlocal officialchecks
    tube=DenseTube(box,r,gt[qidx],spans[qidx],clip=ds=='hc2');v=tube.score(iv);off=official(box,r,gt[qidx],spans[qidx],iv,ds);assert max(abs(v[k]-off[k])for k in ['v','t','s'])<2e-10;officialchecks+=1
    pixel=xyxy(box,r['input']['width'],r['input']['height']);pixel=np.maximum(pixel,0)if ds=='hc2'else pixel;lookup={f:i for i,f in enumerate(z['frame_ids'])}
    for name,ids in [('fixed_actual_sIoU',fixed_support),('unobserved_actual_sIoU',unobserved)]:v[name]=float(np.mean([float(box_iou(pixel[lookup[f]],gt[qidx][f]))for f in ids]))if ids else None
    v['boundary_MAE']=(abs(iv[0]-spans[qidx][0])+abs(iv[1]-spans[qidx][1]))/2
    return v
   N=z['intervals']['Native'];S=z['intervals']['Strong']
   values['Source']=metric(z['Source'],N);values['Temporal_only']=metric(z['Source'],S);values['Without_spatial_adaptation']=metric(z['E'],S)
   for arm,iv in [('Spatial_only',N),('Full',S)]:
    seedvalues[arm]={str(s):metric(z['Live'][s],iv)for s in p['seeds']};values[arm]={k:float(np.mean([v[k]for v in seedvalues[arm].values()]))if all(v[k]is not None for v in seedvalues[arm].values())else None for k in next(iter(seedvalues[arm].values()))}
   rows.append(dict(alias=rec['alias'],values=values,seeds=seedvalues,GT_supported_actual_nodes=len(fixed_support),unobserved_actual_nodes=len(unobserved),cost=z['cost'],new_updates=z['actual_new_updates'],reused_updates=z['reused_updates']))
  means={a:{k:float(np.mean([r['values'][a][k]for r in rows if r['values'][a][k]is not None]))if any(r['values'][a][k]is not None for r in rows)else None for k in rows[0]['values'][a]}for a in ROWS};comparisons={}
  for a,b in [('Spatial_only','Source'),('Temporal_only','Source'),('Full','Source'),('Full','Without_spatial_adaptation'),('Full','Spatial_only'),('Without_spatial_adaptation','Temporal_only')]:
   dd={k:paired([100*(r['values'][a][k]-r['values'][b][k])for r in rows if r['values'][a][k]is not None and r['values'][b][k]is not None])for k in ['v','t','s','fixed_actual_sIoU','unobserved_actual_sIoU']};v=np.array([100*(r['values'][a]['v']-r['values'][b]['v'])for r in rows]);dd['tail']=dict(positive=int((v>1e-10).sum()),negative=int((v< -1e-10).sum()),unchanged=int((abs(v)<=1e-10).sum()),below_minus5=int((v< -5).sum()),positive_contribution_pp=float(np.maximum(v,0).mean()),negative_contribution_pp=float(np.minimum(v,0).mean()),worst=[dict(alias=rows[i]['alias'],delta_pp=float(v[i]))for i in np.argsort(v)[:8]]);comparisons[a+'_minus_'+b]=dd
  allresults[ds]=dict(direction='VidSTG_to_HC2'if ds=='hc2'else'HC2_to_VidSTG',parents=128,queries=128,means=means,paired=comparisons,rows=rows,GT_provenance=prov,exposure=p['cohorts'][ds]['historical_exposure'])
 result=dict(status='completed_frozen_panel_evaluation',directions=allresults,seeds=p['seeds'],seed_aggregation='metrics averaged within query/parent before macro mean and bootstrap; no averaged-box trajectory',independent_confirmation=False,algorithm_selection=False,source_checkpoints=p['checkpoints'],configuration=p['configs'],official_metric_checks=officialchecks,pre_GT_audit=read(D/'PRE_GT_AUDIT.json'),score_seconds=time.perf_counter()-start)
 put(P/'RESULTS_PUBLIC.json',result)
 lines=['# Frozen spatial OPD + Strong: two directions, 128 parents each','','Five rows, same physical decision support/MBR, frozen spatial configuration and seeds41/42/43. One prelocked query per parent. HC exact prior development128 outputs reused after full-input compatibility audit; reverse128 freshly inferred. This is a panel evaluation, not the full validation split or independent confirmation. No parameter selection. Strong posterior = 0.5 high-resolution448 final + 0.5 native224 second-pass six-layer posterior mean. Spatial adaptation and temporal evidence enhancement have separate mechanisms.','','| Direction / row | full vIoU % | tIoU % | fixed actual-node sIoU % |','|---|---:|---:|---:|']
 for ds,res in allresults.items():
  for a in ROWS:
   m=res['means'][a];s=m['fixed_actual_sIoU'];lines.append(f"| {res['direction']} / {a} | {100*m['v']:.6f} | {100*m['t']:.6f} | {100*s:.6f} |")
  lines+=['',f"## {res['direction']} paired differences",'', '| Comparison | full-v pp [95% paired parent CI] | tIoU pp | severe losses >5 pp |','|---|---|---:|---:|']
  for k,v in res['paired'].items():
   z=v['v'];lines.append(f"| {k} | {z['mean']:+.6f} [{z['CI95'][0]:+.6f},{z['CI95'][1]:+.6f}] | {v['t']['mean']:+.6f} | {v['tail']['below_minus5']} |")
  c=res['paired']['Full_minus_Without_spatial_adaptation']['v'];lines+=['',f"Adaptation contribution relative to E+Strong: {c['mean']:+.6f} pp, CI {c['CI95']}. Whole-method improvement over Source cannot by itself establish OPD contribution."]
 lines+=['','## Cost and scope','','All per-query measured cost components and peak GPU are in RESULTS_PUBLIC.json. Common Source224/448, GroundingDINO and ReferDINO are charged, including model loading separately. Three-seed matrix time is not single-inference latency. Single-seed component sums include each expert/common stage once; they are component sums, not independent end-to-end wall measurements. HC cache-reuse rows do not claim zero deployment cost. HC RC001 complete-input compatibility case provides a measured complete-chain profile; reverse cases run the complete chain. No weights, raw caches, private video/annotations, captions or parent identifiers are published. Historical exposure is disclosed; no methods registry is automatically promoted.']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n');put(P/'AUDIT_PUBLIC.json',dict(status='pass',official_metric_checks=officialchecks,posterior=read(D/'PRE_GT_AUDIT.json'),GT_after_global_barrier=True,protected_registries_unchanged=True));print('SCORED_BOTH_DIRECTIONS',flush=True)
if __name__=='__main__':main()
