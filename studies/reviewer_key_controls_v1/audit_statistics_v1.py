"""Separate-process independent aggregation audit of reviewer diagnostics."""
from common_v1 import *
import numpy as np,math
from statistics_v1 import COUNTS
def close(x,y,tol=1e-12):
 if x is None or y is None:assert x is None and y is None
 else:assert abs(x-y)<=tol,(x,y)
def means(report,rr,fields):
 assert report['queries']==len(rr)and report['parents']==len({r['parent']for r in rr})
 # fsum is independent from the derivation's numpy reduction.
 for k in fields:
  vals=[r[k]for r in rr if r.get(k)is not None];z=report['statistics'][k];assert z['queries']==len(vals);close(z['mean'],math.fsum(vals)/len(vals)if vals else None)
def ci(report,rr,field):
 from full_score import cluster
 if not rr:return
 zz=[dict(parent=r['parent'],values={'A':{'v':r[field]},'B':{'v':0. if r[field]is not None else None}})for r in rr]
 out=cluster(zz,'A','B',[('v',1)])['v'];z=report['statistics'][field];close(z['mean'],out['mean']);assert np.allclose(z['CI95'],out['CI95'],rtol=0,atol=1e-12)
def offline(ds):
 start=time.perf_counter();root=R/f'outputs/review_strengthening_v1/offline_current/{ds}';a=read(root/'audit_receipt.json');assert sha(root/'derived.json')==a['derived_sha256'];oo=read(root/'derived.json')['rows'];truth=selected.labels(ds);cells={}
 for x in oo:
  e=fixed(selected.D/f'components_{ds}_sealed.json')['files'][x['id']];assert sha(selected.D/e['path'])==e['sha256']
  with np.load(selected.D/e['path'])as z:iv=z['FullMAP_interval'].copy()
  allframes=sum(x['distance'],[])
  for mode,span in [('all_unobserved',None),('predicted_event',iv),('GT_event',truth[x['id']]['interval'])]:
   for b in range(-1,5):
    ff=[f for f in(allframes if b<0 else x['distance'][b])if span is None or int(span[0])<=f['frame']<int(span[1])];name=mode+'/'+('all_distances'if b<0 else f'bin{b}');cells.setdefault(name,[])
    if ff:
     ours=100*math.fsum(f['values']['Ours']for f in ff)/len(ff);inter=100*math.fsum(f['values']['Interpolation']for f in ff)/len(ff);cells[name].append(dict(parent=x['parent'],ours_IoU_percent=ours,interpolation_IoU_percent=inter,delta_IoU_pp=ours-inter,frames=len(ff)))
 report=read(P/'offline'/ds/'distance_event_membership.json');assert set(cells)==set(report)
 for name,rr in cells.items():
  means(report[name],rr,['ours_IoU_percent','interpolation_IoU_percent','delta_IoU_pp']);assert report[name]['annotated_frames']==sum(r['frames']for r in rr);ci(report[name],rr,'delta_IoU_pp')
 metrics=read(R/f'outputs/temporal_feedback_risk_v1/{ds}/metrics.json')['private_rows'];risk=read(P/'offline'/ds/'risk_thresholds.json')
 for name,g in risk.items():
  rr=metrics if name=='all'else[r for r in metrics if r['values']['Source'][name.split('_')[1]]>=float(name.split('_')[-1])];z=[]
  for r in rr:
   d=100*(r['values']['FullMAP']['v']-r['values']['Source']['v']);z.append(dict(parent=r['parent'],delta_v_pp=d,gain_gt1_percent=float(d>1)*100,loss_lt_minus1_percent=float(d< -1)*100,loss_le_minus10_percent=float(d<=-10)*100,negative_percent=float(d< -1e-10)*100))
  means(g,z,list(z[0])[1:]if z else[])
  for f in g['statistics']:ci(g,z,f)
 # Check every query-form x quality bin against native audited quality and exact metadata join.
 joined=read(D/'offline'/ds/'joined_private.json')['rows'];quality=read(P/'offline'/ds/'query_form_expert_quality_spatial.json');by={r['id']:r for r in oo};p4={r['id']:r for r in read(R/f'outputs/p4_cached_risk_v1/{ds}/private_scalar_rows.json')['rows']}
 for r in joined:
  q=by[r['id']]['quality'];expected='unscorable'if q is None else'low'if q<1/3 else'medium'if q<2/3 else'high';assert r['quality_group']==expected and r['form']==p4[r['id']]['query_form']
 for name,v in quality.items():
  form,qg=name.split('/');rr=[r for r in joined if(form=='all'or r['form']==form)and(qg=='all'or r['quality_group']==qg)];assert v['spatial_v']['queries']==len(rr);dd=[]
  for r in rr:dd.append(dict(parent=r['parent'],delta_v_pp=100*(r['values']['Ours']['v']-r['values']['ExpertInterpolation']['v'])))
  if dd:ci(v['spatial_v'],dd,'delta_v_pp')
 receipt=dict(status='independently_verified_cached_membership_quality_risk',queries=len(oo),all_distance_masks_counts_means_and_paired_CIs=True,all_risk_threshold_counts_means_CIs=True,all_metadata_quality_joins=True,code_sha256=sha(Path(__file__)),source_plan_sha256=sha(D/'statistics_plan_v1.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_GT_metric_scoring=0,new_model=0);atomic(P/'offline'/ds/'audit_receipt.json',receipt);print(ds,'offline independent audit passed',flush=True)
def tta(ds):
 start=time.perf_counter();private=read(D/'tta'/ds/'private_rows.json');report=read(P/'tta'/ds/'diagnostics.json')['methods'];root=R/'outputs/tta_full_v1';seal=read(root/f'{ds}_sealed.json');audit=read(root/f'{ds}_reporting_receipt.json');assert len(seal['files'])==COUNTS[ds]
 for m,rr in private.items():
  means(report[m],rr,list(rr[0])[1:]);ci(report[m],rr,'entropy_delta');ci(report[m],rr,'native_delta_t_pp');ci(report[m],rr,'native_delta_v_pp');assert sum(r['parameter_update_L2']>0 for r in rr)==audit['diagnostics'][m]['nonzero_parameter_delta_queries'];assert round(sum(r['changed_native_interval_percent']/100 for r in rr))==audit['diagnostics'][m]['changed_native_queries'];assert sum(r['logit_change_L2']>0 for r in rr)==round(report[m]['statistics']['changed_logits_percent']['mean']*len(rr)/100)
 atomic(P/'tta'/ds/'audit_receipt.json',dict(status='independently_verified_saved_optimizer_diagnostic_aggregates',queries=COUNTS[ds],original_full_readout_and_logit_SHA_audit_sha256=sha(root/f'{ds}_reporting_receipt.json'),all_group_means_and_primary_CIs=True,exact_original_update_and_interval_counts=True,code_sha256=sha(Path(__file__)),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_model=0,new_optimizer=0));print(ds,'TTA aggregate audit passed',flush=True)
def evidence(ds):
 start=time.perf_counter();rr=read(D/'temporal_evidence'/ds/'private_rows.json');report=read(P/'temporal_evidence'/ds/'report.json');seal=read(D/'temporal'/ds/'sealed.json');truth=selected.labels(ds)
 for r in rr:
  e=seal['files'][r['id']];assert sha(D/e['path'])==e['sha256']
  with np.load(D/e['path'])as z:
   y=(z['frame_ids']>=truth[r['id']]['interval'][0])&(z['frame_ids']<truth[r['id']]['interval'][1]);assert r['event_frames']==int(y.sum())and r['background_frames']==int((~y).sum())
   for n in ['SourceROI','CorrectedROI']:
    L=z['evidence_'+n];pos=L[y];neg=L[~y];v=float(((pos[:,None]>neg).sum()+.5*(pos[:,None]==neg).sum())/(len(pos)*len(neg)))if len(pos)and len(neg)else None;close(v,r[n+'_AUROC']);close(float(pos.mean())if len(pos)else None,r[n+'_event_mean']);close(float(neg.mean())if len(neg)else None,r[n+'_background_mean'])
 fields=[k for k in rr[0]if k not in ['parent','id','source_t_group','native_frames','event_frames','background_frames']]
 for name,v in report.items():
  rows0=rr if name=='all'else[r for r in rr if(r['delta_t_pp']>1e-10 if name=='temporal_improved'else r['delta_t_pp']< -1e-10 if name=='temporal_harmed'else r['source_t_group']==name)];means(v,rows0,fields)
  for f in ['delta_t_pp','delta_v_pp','CorrectedROI_AUROC','Corrected_minus_Source_AUROC']:ci(v,rows0,f)
 atomic(P/'temporal_evidence'/ds/'audit_receipt.json',dict(status='independent_pairwise_AUROC_event_masks_means_CIs_verified',queries=len(rr),all_saved_evidence_SHA=True,code_sha256=sha(Path(__file__)),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_GT_metric_scoring=0,new_model=0));print(ds,'evidence independent audit passed',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('task',choices=['offline','tta','evidence']);ap.add_argument('--dataset',choices=COUNTS,required=True);a=ap.parse_args();globals()[a.task](a.dataset)
