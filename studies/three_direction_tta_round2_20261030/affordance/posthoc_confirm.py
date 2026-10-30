"""Finite post-confirm diagnostics on sealed predictions. No new model, optimizer, or calibrated prediction.
Affine quantities are output-to-output orthogonal projections without GT and are not rescored.
"""
import pathlib,json,hashlib,collections
import numpy as np
from scipy.stats import spearmanr
ROOT=pathlib.Path(__file__).resolve().parent;RUN=ROOT/'runs/confirm_native_fps_support_v1';CACHE=ROOT/'caches/confirm_native_fps_support_v1'
def digest(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def scalar(x):return float(x) if np.isfinite(x) else None
def region(p,t):
 p=p.reshape(-1);t=t.reshape(-1);y=t>=.5;pred=p>=.5
 return {'target_count':int(y.sum()),'predicted_positive_count_at05':int(pred.sum()),'TP_at05':int((pred&y).sum()),'FP_at05':int((pred&~y).sum()),'recall_at05':scalar(pred[y].mean()) if y.any() else None,'precision_at05':scalar(y[pred].mean()) if pred.any() else None,'mean_target_probability':scalar(p[y].mean()) if y.any() else None,'mean_non_target_probability':scalar(p[~y].mean()) if (~y).any() else None,'target_MAE':scalar(np.abs(p[y]-t[y]).mean()) if y.any() else None,'non_target_MAE':scalar(np.abs(p[~y]-t[~y]).mean()) if (~y).any() else None,'all_point_mean_probability':float(p.mean())}
def projection(z,p,mask):
 zz=z.reshape(-1)[mask].astype(np.float64);q=p.reshape(-1)[mask].astype(np.float64);d=np.log(q/(1-q))-zz;A=np.c_[np.ones(len(zz)),zz];coef,_,rank,_=np.linalg.lstsq(A,d,rcond=None);fit=A@coef;energy=float((d*d).sum());centered=float(((d-d.mean())**2).sum())
 return {'numeric_band_points':len(zz),'effective_residual_mean':float(d.mean()),'effective_residual_std':float(d.std()),'shift_only_energy_fraction':float(len(d)*d.mean()**2/energy) if energy else None,'affine_projection_energy_fraction':float((fit*fit).sum()/energy) if energy else None,'affine_residual_RMS':float(np.sqrt(((d-fit)**2).mean())),'affine_variance_R2':float(1-((d-fit)**2).sum()/centered) if centered else None,'beta_shift':float(coef[0]),'alpha_logit_scaling_adjustment':float(coef[1]),'implied_positive_temperature_slope':bool(1+coef[1]>0),'least_squares_rank':int(rank),'no_GT_used_in_projection':True,'no_new_prediction_generated_or_scored':True}
def main():
 for n,h in json.loads((RUN/'SEALED.json').read_text()).items():assert digest(RUN/n)==h
 score=json.loads((RUN/'SCORED.json').read_text());assert score['seal_verified'];refs=json.loads((ROOT/'cohorts/confirm_scoring.json').read_text());infer=json.loads((ROOT/'cohorts/confirm_inference.json').read_text());look={(r['id'],r['objective'],r['step'],r['arm']):r for r in score['rows']};rows=[]
 for i,(ref,ir) in enumerate(zip(refs,infer)):
  full=np.load(RUN/f'{i:04d}_hm_support_k3.npz',allow_pickle=False);hm=np.load(RUN/f'{i:04d}_hm_only_k3.npz',allow_pickle=False);cache=np.load(CACHE/(ir['id']+'.npz'),allow_pickle=False);z=np.median(cache['view_logits'],axis=0);target=np.load(ref['native_annotation_container'],allow_pickle=False)[:,3:]
  maps={'source':full['source'],'median':full['median'],'q75':full['q75'],'full':full['adapted'],'hm_only':hm['adapted']};mask=np.ones(2048,dtype=bool)
  for a in ['median','full','hm_only']:mask&=(maps[a].reshape(-1)>=1e-5)&(maps[a].reshape(-1)<=1-1e-5)
  rank={f'{a}_to_{b}':scalar(spearmanr(maps[a].reshape(-1),maps[b].reshape(-1)).statistic) for a,b in [('source','median'),('median','full'),('median','hm_only'),('q75','full')]}
  metrics={}
  for name,objective,arm in [('source','hm_support','source'),('median','hm_support','median'),('q75','hm_support','q75'),('full','hm_support','adapted'),('hm_only','hm_only','adapted')]:metrics[name]={k:look[(ir['id'],objective,3,arm)][k] for k in ['auc','aiou','sim','mae']}
  rows.append({'id':ir['id'],'parent_id':ir['parent_id'],'object_class':ref['object_class'],'affordance':ref['affordance'],'cohort_index':i,'spearman':rank,'projection':{name:projection(z,maps[name],mask) if mask.sum()>=16 else None for name in ['full','hm_only']},'numeric_mask_is_common_to_both_variants_and_median':True,'region':{name:region(p,target) for name,p in maps.items()},'metrics':metrics})
 paired=[]
 for left,right in [('full','q75'),('hm_only','q75'),('full','hm_only'),('full','median'),('hm_only','median')]:
  item={'candidate':left,'reference':right,'n_parents':len(rows),'metrics':{}}
  for metric in ['auc','aiou','sim','mae']:
   sign=-1 if metric=='mae' else 1;diff=np.array([sign*(r['metrics'][left][metric]-r['metrics'][right][metric]) for r in rows if r['metrics'][left][metric] is not None and r['metrics'][right][metric] is not None]);rng=np.random.default_rng(42);samples=diff[rng.integers(0,len(diff),(10000,len(diff)))].mean(1);item['metrics'][metric]={'mean_signed_benefit':float(diff.mean()),'improved':int((diff>0).sum()),'worsened':int((diff<0).sum()),'unchanged':int((diff==0).sum()),'min':float(diff.min()),'max':float(diff.max()),'paired_parent_bootstrap_95ci_descriptive_only':np.quantile(samples,[.025,.975]).tolist()}
  paired.append(item)
 aggregate={'mean_spearman':{k:float(np.mean([r['spearman'][k] for r in rows if r['spearman'][k] is not None])) for k in rows[0]['spearman']},'projection':{},'mean_parent_region':{}}
 for name in ['full','hm_only']:
  vals=[r['projection'][name] for r in rows if r['projection'][name] is not None];aggregate['projection'][name]={key:float(np.mean([v[key] for v in vals if v[key] is not None])) for key in ['numeric_band_points','effective_residual_mean','effective_residual_std','shift_only_energy_fraction','affine_projection_energy_fraction','affine_residual_RMS','affine_variance_R2','beta_shift','alpha_logit_scaling_adjustment']};aggregate['projection'][name]['positive_temperature_slope_count']=sum(v['implied_positive_temperature_slope'] for v in vals)
 for arm in ['source','median','q75','full','hm_only']:
  aggregate['mean_parent_region'][arm]={key:float(np.mean([r['region'][arm][key] for r in rows if r['region'][arm][key] is not None])) for key in rows[0]['region'][arm]}
 signed=[(r['metrics']['full']['aiou']-r['metrics']['q75']['aiou'],r) for r in rows];extremes={tag:{'cohort_index':r['cohort_index'],'id':r['id'],'object_class':r['object_class'],'affordance':r['affordance'],'full_minus_q75_aiou':d} for tag,(d,r) in [('largest_positive',max(signed,key=lambda x:x[0])),('largest_negative',min(signed,key=lambda x:x[0]))]}
 output={'scope':'Posthoc analysis after confirmation32 labels exposed, not independent additional validation; no new method selection','projection_definition':'effective d=logit(final p)-median native logits; common numeric band p in[1e-5,1-1e-5]; project d onto span(1,z), no GT; do not generate/rescore calibration predictions','bootstrap':'paired parent points,10000 resamples,seed42,descriptive only; approximate physical grouping and small stratified cohort limit inference','aggregate':aggregate,'paired_comparisons':paired,'extreme_case_selection':extremes,'rows':rows,'new_model_calls':0,'new_calibrated_prediction_scores':0,'primary_results_modified':False};(ROOT/'POSTHOC_CONFIRM_DIAGNOSTICS.json').write_text(json.dumps(output,indent=2));print(json.dumps({'aggregate':aggregate,'paired_comparisons':paired,'cases':extremes},indent=2))
if __name__=='__main__':main()
