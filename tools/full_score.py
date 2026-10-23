"""Independent CPU scoring of each complete, sealed paper-package domain."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse,json,sys,time,math,csv
from pathlib import Path
import numpy as np,torch
from full import ROOT,PROVIDER,RUN,PUBLIC,sha,read,atomic,check
sys.path.insert(0,str(ROOT))
from stvg.metrics import official,DenseTube,box_iou
from stvg.metrics_base import xyxy
from stvg.evaluation import finite_mean,add_reporting_metrics,main_table
from stvg.paper_panel import ARMS

def metric(boxes,r,gt,span,iv,pos):
 ds=r['dataset'];m=r['metadata'];ids=m['frame_ids'];row=dict(input=m,frame_ids=ids)
 v=official(boxes,row,gt,span,iv,ds);d=DenseTube(boxes,row,gt,span,clip=ds=='hc2').score(iv);err=max(abs(v[k]-d[k])for k in ('v','t','s'));assert err<2e-10
 pixels=xyxy(boxes,m['width'],m['height']);pixels=np.maximum(pixels,0)if ds=='hc2'else pixels;obs={ids[int(x)]for x in pos};lookup={f:i for i,f in enumerate(ids)};fixed=[f for f in ids if f in gt]
 for name,rr in [('fixed_actual_sIoU',fixed),('observed_actual_sIoU',[f for f in fixed if f in obs]),('unobserved_actual_sIoU',[f for f in fixed if f not in obs])]:v[name]=float(np.mean([box_iou(pixels[lookup[f]],gt[f])for f in rr]))if rr else None
 fps=m['fps'];v.update(boundary_MAE=(abs(iv[0]-span[0])+abs(iv[1]-span[1]))/2,start_error_seconds=abs(iv[0]-span[0])/fps,end_error_seconds=abs(iv[1]-span[1])/fps,interval_length_seconds=(iv[1]-iv[0])/fps,length_error_seconds=abs(iv[1]-iv[0]-span[1]+span[0])/fps)
 return add_reporting_metrics(v,m),err

def cluster(rows,a,b,fields=None):
 fields=fields or [('v',100),('t',100),('observed_actual_sIoU',100),('unobserved_actual_sIoU',100),('boundary_MAE_seconds',1)]
 parents=sorted({r['parent']for r in rows});ix={p:i for i,p in enumerate(parents)};n=len(parents);w=np.random.default_rng(20261022).multinomial(n,[1/n]*n,size=10000);result={}
 for f,scale in fields:
  sums=np.zeros(n);counts=np.zeros(n);delta=[]
  for r in rows:
   x,y=r['values'][a].get(f),r['values'][b].get(f)
   if x is None or y is None:continue
   z=scale*(x-y);delta.append(z);j=ix[r['parent']];sums[j]+=z;counts[j]+=1
  d=np.asarray(delta);den=w@counts;boot=(w@sums)[den>0]/den[den>0]
  result[f]=dict(mean=float(d.mean())if len(d)else None,CI95=np.quantile(boot,[.025,.975]).tolist()if len(boot)else None,queries=len(d),parents=int((counts>0).sum()),improved=int(((d< -1e-10)if f in ('boundary_MAE_seconds','missing')else(d>1e-10)).sum()),harmed=int(((d>1e-10)if f in ('boundary_MAE_seconds','missing')else(d< -1e-10)).sum()),unchanged=int((abs(d)<=1e-10).sum()),declined_over5pp=int((d< -5).sum())if scale==100 else None)
 return result

def run(ds):
 torch.set_num_threads(1);p=check();dest=RUN/ds;seal=read(dest/'sealed.json');job=p['jobs'][ds]
 assert seal['fingerprint']==p['fingerprint']and seal['queries']==job['queries']and seal['GT_read']is False
 if(dest/'score_receipt.json').exists():
  rec=read(dest/'score_receipt.json');assert rec['seal_sha256']==sha(dest/'sealed.json')and all(sha(ROOT/f)==h for f,h in rec['public_files'].items());return
 for e in seal['files'].values():assert sha(dest/e['path'])==e['sha256']and sha(dest/e['receipt_path'])==e['receipt_sha256']
 label=PROVIDER/job['labels'];assert sha(label)==job['labels_sha256'];truth={x['id']:x for x in [json.loads(s)for s in label.read_text().splitlines()]if x['dataset']==ds};records=[json.loads(s)for s in(PROVIDER/job['manifest']).read_text().splitlines()];assert set(seal['files'])=={r['id']for r in records}==set(truth)
 audit=read(RUN/'tta_audit.json')['historical_project_OPD'][ds];arms=tuple(ARMS)+(('HistoricalOPD',)if audit['eligible']else());rows=[];extent=[];distance=[[]for _ in range(5)];edge=[0,.5,1,2,4,math.inf];maxerr=0.;checks=0;costs=[];started=time.perf_counter()
 for i,r in enumerate(records):
  e=seal['files'][r['id']];g=truth[r['id']];gt={int(k):v for k,v in g['boxes'].items()};vals={};ext={};dist=[[]for _ in range(5)]
  with np.load(dest/e['path'],allow_pickle=False)as z:
   for a in ARMS:vals[a],err=metric(z[a],r,gt,g['interval'],z[a+'_interval'].tolist(),z['positions8']);maxerr=max(maxerr,err);checks+=1
   if audit['eligible']:
    historical=Path(audit['private_prediction_root'])/'predictions'/(r['id']+'.npz')
    with np.load(historical,allow_pickle=False)as h:vals['HistoricalOPD'],err=metric(h['Live'],r,gt,g['interval'],h['Strong_interval'].tolist(),h['positions8']);maxerr=max(maxerr,err);checks+=1
   ids=z['frame_ids'];pos=set(map(int,z['positions8']));W,H=r['metadata']['width'],r['metadata']['height'];fps=r['metadata']['fps']
   cb=xyxy(z['complete_boxes'].reshape(-1,4),W,H).reshape(len(ids),2,4);mb=xyxy(z['mask_boxes'].reshape(-1,4),W,H).reshape(len(ids),2,4)
   if ds=='hc2':cb=np.maximum(cb,0);mb=np.maximum(mb,0)
   all_complete=[];all_mask=[];paired_complete=[];paired_mask=[];cmissing=[];mmissing=[]
   our=xyxy(z['OursS'],W,H);init=xyxy(z['SharedInit'],W,H)
   if ds=='hc2':our=np.maximum(our,0);init=np.maximum(init,0)
   for t,f in enumerate(ids):
    f=int(f)
    if f not in gt or t in pos:continue
    d=float(z['nearest_observation_seconds_placeholder'][t])/fps;k=min(int(np.searchsorted(edge,d,side='right'))-1,4);k=max(k,0);dist[k].append(box_iou(our[t],gt[f])-box_iou(init[t],gt[f]))
    for side in (0,1):
     if z['source_anchors'][t,side]<0:continue
     ca=bool(z['complete_available'][t,side]);ma=bool(z['mask_available'][t,side]);c=box_iou(cb[t,side],gt[f])if ca else 0.;m=box_iou(mb[t,side],gt[f])if ma else 0.;all_complete.append(c);all_mask.append(m);cmissing.append(not ca);mmissing.append(not ma)
     if ca and ma:paired_complete.append(c);paired_mask.append(m)
   ext=dict(complete=dict(iou_all=finite_mean(all_complete),iou_both_available=finite_mean(paired_complete),missing=finite_mean(cmissing)),mask=dict(iou_all=finite_mean(all_mask),iou_both_available=finite_mean(paired_mask),missing=finite_mean(mmissing)))
   extent.append(dict(parent=str(g['parent']),values=ext,slots=len(all_complete),paired_slots=len(paired_complete)))
   for k,d in enumerate(dist):
    distance[k].append(dict(parent=str(g['parent']),values=dict(OursS={'delta':finite_mean(d)},SharedInit={'delta':0. if d else None}),nodes=len(d)))
  rows.append(dict(id=r['id'],parent=str(g['parent']),values=vals,duration_group='short_le5s'if(g['interval'][1]-g['interval'][0])/r['metadata']['fps']<=5 else'long_gt5s'));costs.append(read(dest/e['receipt_path']))
  if(i+1)%100==0:atomic(dest/'score_progress.json',dict(completed=i+1,total=len(records),CPU_only=True))
 means={a:{f:finite_mean([r['values'][a][f]for r in rows])for f in rows[0]['values'][a]}for a in arms};table=main_table(means)
 for a in arms:
  for f in ('start_error_seconds','end_error_seconds','interval_length_seconds','length_error_seconds'):table[a][f]=means[a][f]
 old=read(PROVIDER/'results/boxfixed_full_v1'/ds/'report.json')
 for a,b in [('Source','Source'),('Global6','Final')]:
  for f in ('m_vIoU','m_tIoU','m_sIoU_unobserved_actual','vIoU@0.3','vIoU@0.5'):assert abs(table[a][f]-old['means'][b][f])<1e-12,(ds,a,f)
 pairs=[('OursS','Source'),('OursS','Interpolation'),('Full','OursS'),('Full','Global6'),('Global6','Source'),('Full','Source'),('Full','NoLocal'),('Full','Independent'),('SharedInit','Source'),('OursS','SharedInit')]
 pairs += [('HistoricalOPD','Source'),('Full','HistoricalOPD')]if audit['eligible']else[]
 paired={a+'-'+b:cluster(rows,a,b)for a,b in pairs};valid_counts={a:{f:sum(x['values'][a][f]is not None for x in rows)for f in ('observed_actual_sIoU','unobserved_actual_sIoU')}for a in arms}
 extent_report=dict(queries=sum(x['slots']>0 for x in extent),total_slots=sum(x['slots']for x in extent),paired_available_slots=sum(x['paired_slots']for x in extent),mean_per_query={a:{f:finite_mean([x['values'][a][f]for x in extent])for f in ('iou_all','iou_both_available','missing')}for a in ('complete','mask')},paired=cluster(extent,'complete','mask',[('iou_all',100),('iou_both_available',100),('missing',100)]),scope='Same anchor+direction, actual GT-supported unobserved nodes; largest mask-mass component, no GT direction selection. Missing zero-filled all-slot IoU and jointly available IoU separately reported.')
 curve=[dict(seconds=[edge[k],edge[k+1]if np.isfinite(edge[k+1])else None],nodes=sum(x['nodes']for x in rr),effective_queries=sum(x['nodes']>0 for x in rr),paired=cluster(rr,'OursS','SharedInit',[('delta',100)])['delta'])for k,rr in enumerate(distance)]
 segments={}
 for c in costs:
  key=f"{c['worker']['pid']}:{c['worker']['startticks']}";z=segments.setdefault(key,dict(queries=0,complete_capture_reused=0,fresh_queries=0,wall_sum=0.,visual_hits=0,visual_misses=0,RGB_hits=0,RGB_misses=0,expert_forwards=0));z['queries']+=1;z['complete_capture_reused']+=int(c['complete_capture_reused']);z['fresh_queries']+=int(not c['complete_capture_reused']);z['wall_sum']+=c['query_wall_seconds'];z['visual_hits']+=c['visual_delta']['hits'];z['visual_misses']+=c['visual_delta']['misses'];z['RGB_hits']+=c['RGB_delta']['hits'];z['RGB_misses']+=c['RGB_delta']['misses'];z['expert_forwards']+=c['expert_forwards']
 for z in segments.values():z['mean_query_wall_seconds']=z['wall_sum']/z['queries'];z['visual_hit_rate']=z['visual_hits']/(z['visual_hits']+z['visual_misses'])if z['visual_hits']+z['visual_misses']else None
 time_groups={g:dict(queries=sum(x['duration_group']==g for x in rows),means={a:{f:finite_mean([x['values'][a][f]for x in rows if x['duration_group']==g])for f in ['t','v','boundary_MAE_seconds','interval_length_seconds']}for a in arms})for g in ['short_le5s','long_gt5s']}
 report=dict(time_groups=time_groups,status='complete_scored',dataset=ds,backbone='TA-STVG',queries=len(rows),parents=len({r['parent']for r in rows}),aggregation='equal-query; parent-cluster10000 resamples',means=table,paired=paired,valid_queries=valid_counts,extent=extent_report,distance_curve=curve,efficiency_segments=segments,proof=dict(all_predictions_sealed_before_labels=True,official_dense_checks=checks,max_error=maxerr,Source_and_Global6_reproduce_published_v1=True,fingerprint=p['fingerprint'],seal_sha256=sha(dest/'sealed.json')),exposure=p['exposure'],aliases=dict(NativeTemporal='OursS',Global6='old v1 Final bitwise; no duplicate baseline',Full='NEW frozen Native2D; never oldFinal alias'))
 pub=PUBLIC/ds;atomic(pub/'report.json',report);public_files={str((pub/'report.json').relative_to(ROOT)):sha(pub/'report.json')};pm={p:f'P{i+1:03d}'for i,p in enumerate(sorted({x['parent']for x in rows}))};fields=['query','parent']+[a+'_'+f for a in arms for f in ('v','t','observed_actual_sIoU','unobserved_actual_sIoU','boundary_MAE_seconds')]+[a+'-'+b+'_'+f for a,b in pairs for f in ('v','t')]
 for off in range(0,len(rows),1000):
  f=pub/f'signed_{off//1000+1:02d}.tsv';f.parent.mkdir(parents=True,exist_ok=True)
  with f.open('w')as out:
   writer=csv.writer(out,delimiter='\t',lineterminator='\n');writer.writerow(fields)
   for i,x in enumerate(rows[off:off+1000],off+1):writer.writerow([f'{ds[0].upper()}{i:05d}',pm[x['parent']]]+[x['values'][a][f]for a in arms for f in ('v','t','observed_actual_sIoU','unobserved_actual_sIoU','boundary_MAE_seconds')]+[100*(x['values'][a][f]-x['values'][b][f])for a,b in pairs for f in ('v','t')])
  public_files[str(f.relative_to(ROOT))]=sha(f)
 f=pub/'severe_declines.tsv'
 with f.open('w')as out:
  w=csv.writer(out,delimiter='\t',lineterminator='\n');w.writerow(['query','parent','method','baseline','metric','delta_pp'])
  for i,x in enumerate(rows,1):
   for a,b in pairs:
    for metric in ['v','t']:
     d=100*(x['values'][a][metric]-x['values'][b][metric])
     if d< -5:w.writerow([f'{ds[0].upper()}{i:05d}',pm[x['parent']],a,b,metric,d])
 public_files[str(f.relative_to(ROOT))]=sha(f);atomic(dest/'metrics.json',dict(report,private_rows=rows,private_extent=extent,private_distance=distance));atomic(dest/'score_receipt.json',dict(status='scored',seal_sha256=sha(dest/'sealed.json'),public_files=public_files,seconds=time.perf_counter()-started));assert not torch.cuda.is_initialized();print(json.dumps(dict(dataset=ds,queries=len(rows),means=table)))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=['hc2','vidstg']);a=p.parse_args();run(a.dataset)
