"""Post-seal descriptive task/support analysis; no prediction changes."""
import json,hashlib,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];H=ROOT/'research/round53'
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 start=time.perf_counter();audit=read(H/'saved_scalar_audit_attempt001/AUDIT_RESULT.json');assert audit['state'].startswith('PASS_');rows=read(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');summary=read(H/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336'];out=H/'analysis_attempt001';assert not out.exists();out.mkdir();metrics=[]
 for arm,val in summary['balanced_mIoU_percent'].items():
  record={'arm':arm,'balanced_mIoU_percent':val,'drone_mIoU_percent':summary['drone'][arm]['mIoU_percent'],'quad_mIoU_percent':summary['quad'][arm]['mIoU_percent'],'gain_vs_source_pp':val-summary['balanced_mIoU_percent']['source'],'gain_vs_geometry_full_pp':val-summary['balanced_mIoU_percent']['geometry_full'],'unknown_n':sum(r['statuses'][arm]['unknown'] for r in rows),'fallback_n':sum(r['statuses'][arm]['fallback'] for r in rows)}
  for ref in ['source','geometry_full']:
   d=np.asarray([r['IoU'][arm]-r['IoU'][ref] for r in rows]);record['improved_vs_'+ref]=int((d>1e-10).sum());record['worsened_vs_'+ref]=int((d< -1e-10).sum());record['query_changes_vs_'+ref]=sum(r['queries'][arm]!=r['queries'][ref] for r in rows)
  for threshold,name in [(.25,'25'),(.5,'50')]:
   record['balanced_Acc'+name+'_percent']=(summary['drone'][arm]['Acc'+name+'_percent']+summary['quad'][arm]['Acc'+name+'_percent'])/2;record['rescue'+name]=sum(r['IoU']['source']<=threshold and r['IoU'][arm]>threshold for r in rows);record['break'+name]=sum(r['IoU']['source']>threshold and r['IoU'][arm]<=threshold for r in rows)
  metrics.append(record)
 dump(out/'ARM_METRICS.json',metrics)
 comparisons=[]
 for suffix in ['full','V','VR']:
  for before,after in [('geometry','pre'),('geometry','native'),('pre','native'),('shuffle','native')]:
   a=before+'_'+suffix;b=after+'_'+suffix;d=np.asarray([r['IoU'][b]-r['IoU'][a] for r in rows]);comparisons.append({'before':a,'after':b,'balanced_delta_pp':summary['balanced_mIoU_percent'][b]-summary['balanced_mIoU_percent'][a],'improved':int((d>1e-10).sum()),'worsened':int((d< -1e-10).sum()),'query_changes':sum(r['queries'][b]!=r['queries'][a] for r in rows)})
 dump(out/'MATCHED_COMPARISONS.json',comparisons)
 coverage=read(H/'prediction_audit_attempt001/COVERAGE_STATS.json');support_counts={'groups':sum(r['groups'] for r in coverage),'seedless_groups':sum(r['seedless_groups'] for r in coverage),'T_mask_available':sum(r['T_mask_available'] for r in coverage),'R_masks_available':sum(r['R_masks_available'] for r in coverage),'R_masks_total':sum(r['R_masks_total'] for r in coverage),'separator_only_span_bridges':sum(r['separator_only_bridges'] for r in coverage),'visible_groups_at_least8':0,'visible_groups_at_least8_one_distinct_seed':0};distinct=[];ratios=[];native_entropy=[];bad_errors=[];anchoroutside=0
 for row in rows:
  sid=row['id_PRIVATE'];pred=ROOT/'crossplatform/round53/predicted336_attempt001';lc=read(ROOT/'crossplatform/round53/latent336_attempt001/cases_PRIVATE'/(sid+'.json'));case=read(pred/'cases_PRIVATE'/(sid+'.json'));anchoroutside+=case['counts']['anchor_outside_box_groups']
  if not lc['native_fp2_index_xyz_exact']:bad_errors.append(lc['native_max_coordinate_error'])
  with np.load(pred/'supports_PRIVATE'/(sid+'.npz'),allow_pickle=False) as z:
   for m in range(case['counts']['groups']):
    w=z['native__T__'+str(m)];nn=z['nearest_seed__'+str(m)]
    if len(w)>=8:
     support_counts['visible_groups_at_least8']+=1;unique=len(np.unique(nn)) if len(nn) else 0;distinct.append(unique);support_counts['visible_groups_at_least8_one_distinct_seed']+=unique==1
     if w.sum()>0:ratios.append(float(w.sum()**2/(w@w)/len(w)))
  with np.load(ROOT/'crossplatform/round53/latent336_attempt001/samples_PRIVATE'/(sid+'.npz'),allow_pickle=False) as z:
   qp=z['query_point_5'].astype(float);native_entropy.extend((-np.sum(qp*np.log(np.maximum(qp,1e-300)),axis=1)/np.log(qp.shape[1])).tolist())
 support_counts.update(native_index_mismatch_cases=len(bad_errors),native_index_max_coordinate_error_median=float(np.median(bad_errors)),native_index_max_coordinate_error_max=float(np.max(bad_errors)),anchor_outside_predicted_box_groups=anchoroutside,distinct_nearest_seed_median=float(np.median(distinct)),native_effective_pixel_ratio_median=float(np.median(ratios)),native_query_point_normalized_entropy_median=float(np.median(native_entropy)))
 dump(out/'SUPPORT_DIAGNOSTICS.json',support_counts)
 scenes=read(H/'score_dev336_attempt001/ALL_SCENE_SUMMARIES_PRIVATE.json');scene_rows=[]
 for platform in ['drone','quad']:
  values=sorted({r['scene_PRIVATE'] for r in rows if r['platform']==platform})
  for r in scenes:
   if r['scope']=='ALL336' and r['platform']==platform:scene_rows.append({'platform':platform,'scene_alias':('D' if platform=='drone' else 'Q')+'_S'+str(values.index(r['scene_PRIVATE'])+1).zfill(2),'n':r['n'],'mIoU_percent':r['mIoU_percent']})
 dump(out/'ALL_SCENE_METRICS.json',scene_rows);dump(out/'ANALYSIS_RECEIPT.json',{'state':'R53_POSTSEAL_DESCRIPTIVE_NO_TUNING','n':336,'arms':14,'comparisons':len(comparisons),'scene_clusters':5,'no_predictions_changed':True,'new_GT_or_model_calls':0,'seconds':time.perf_counter()-start});print(json.dumps(support_counts),flush=True)
if __name__=='__main__':main()
