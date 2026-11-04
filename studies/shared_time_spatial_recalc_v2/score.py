from common import *
import argparse
from stvg.metrics import official

def pixels_at_gt(z,arm,f,meta,ds,reference=False):
 if arm=='PureExpertLinear':ids=z['dense_input_ids'];pix=z['PureExpertLinear_xyxy'].copy()
 else:
  ids=z['frame_ids'];pix=xyxy(z[arm],meta['width'],meta['height'])
  if ds=='hc2':pix=np.maximum(pix,0) # Preserve original native-output benchmark rule.
 if reference:
  scale=np.array([meta['width'],meta['height']]*2);pix=pix/scale
  j=np.clip(np.searchsorted(ids,f,side='right')-1,0,len(ids)-2);u=(f-ids[j])/(ids[j+1]-ids[j]);pred=pix[j]+u[:,None]*(pix[j+1]-pix[j]);pred[f<=ids[0]]=pix[0];pred[f>=ids[-1]]=pix[-1]
 else:pred=np.column_stack([np.interp(f,ids,pix[:,k])for k in range(4)])
 if ds=='hc2':pred=np.maximum(pred,0)
 return pred,(f>=ids[0])&(f<=ids[-1])

def measure(z,arm,m,truth,gtspan,ds,reference=False):
 f=np.array(sorted(truth));g=np.array([truth[int(k)]for k in f]);p,valid=pixels_at_gt(z,arm,f,m,ds,reference)
 if reference:g=g/np.array([m['width'],m['height']]*2)
 iw=np.maximum(0,np.minimum(p[:,2],g[:,2])-np.maximum(p[:,0],g[:,0]));ih=np.maximum(0,np.minimum(p[:,3],g[:,3])-np.maximum(p[:,1],g[:,1]));inter=iw*ih;den=np.maximum(0,p[:,2]-p[:,0])*np.maximum(0,p[:,3]-p[:,1])+np.maximum(0,g[:,2]-g[:,0])*np.maximum(0,g[:,3]-g[:,1])-inter;iou=np.divide(inter,den,out=np.zeros(len(f)),where=den>0);iou[~valid]=0;a,b=map(int,z['shared_interval']);c,d=map(int,gtspan);over=max(0,min(b,d)-max(a,c));t=over/(b-a+d-c-over);event=(f>=max(a,c))&(f<min(b,d));assert valid[event].all();v=float(iou[event].sum()/max(max(b,d)-min(a,c),1));obs=np.isin(f,z['frame_ids'][z['positions8']]);eu=event&~obs;eo=event&obs
 return dict(t=float(t),v=v,vIoU_ge_03=float(v>.3),vIoU_ge_05=float(v>.5),event_unobserved_IoU=float(iou[eu].mean())if eu.any()else None,event_observed_IoU=float(iou[eo].mean())if eo.any()else None,event_GT_frames=int(event.sum()),event_unobserved_frames=int(eu.sum()),event_observed_frames=int(eo.sum()))

def report(out,ds):
 from full_score import cluster
 fields=[('v',100),('event_unobserved_IoU',100),('event_observed_IoU',100),('t',100)];means={a:{k:float(np.mean(vv))if(vv:=[r['values'][a][k]for r in out if r['values'][a][k]is not None])else None for k in out[0]['values'][a]}for a in ARMS};paired={a+'-'+b:cluster(out,a,b,fields)for a,b in PAIRS};groups={g:dict(queries=len(rr),paired={a+'-'+b:cluster(rr,a,b,fields)for a,b in PAIRS})for g in ['short_le5s','long_gt5s']if(rr:=[r for r in out if r['duration_group']==g])};folder=P/ds
 atomic(folder/'report.json',dict(queries=len(out),parents=len({r['parent']for r in out}),means=means,paired=paired,timegroups=groups,unobserved_scope='All SAME dense annotated frames in common predicted/GT event intersection, excluding exact raw8expert physical times. Empty intersection query null only for conditional IoU; primaryv/t retains all queries.',valid_event_unobserved_queries={a:sum(r['values'][a]['event_unobserved_IoU']is not None for r in out)for a in ARMS}))
 with(folder/'all_signed_scalars.tsv').open('w')as f:
  w=csv.writer(f,delimiter='\t');w.writerow(['query','group','arm','tIoU','vIoU','R03','R05','event_observed_IoU','event_unobserved_IoU','event_GT_frames','event_observed_frames','event_unobserved_frames'])
  for i,r in enumerate(out,1):
   for a in ARMS:v=r['values'][a];w.writerow([f'Q{i:05}',r['duration_group'],a,*[v[k]for k in ['t','v','vIoU_ge_03','vIoU_ge_05','event_observed_IoU','event_unobserved_IoU','event_GT_frames','event_observed_frames','event_unobserved_frames']]])
 return dict(means=means,paired=paired)

def main(ds):
 validate();sys.addaudithook(old.deny_gpu_media);torch.set_num_threads(1);start=time.perf_counter();dest=D/ds;seal=read(dest/'sealed.json');assert not seal['GT_read']and seal['queries']==COUNTS[ds];truth=old.selected.labels(ds);out=[];checks=0;mx=0.;oldscalar={r['id']:r for r in fixed(CHECK/ds/'metrics.json')['private_rows']}
 for i,row in enumerate(rows(ds),1):
  resources(start);key=row['id'];e=seal['files'][key];assert sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256'];g=truth[key];gt={int(k):b for k,b in g['boxes'].items()};values={}
  with np.load(D/e['path'])as z:
   for arm in ARMS:
    v=measure(z,arm,row['metadata'],gt,g['interval'],ds);values[arm]=v
    if arm=='PureExpertLinear':
     ids=z['dense_input_ids'];b=z['PureExpertLinear_xyxy']/np.array([row['metadata']['width'],row['metadata']['height']]*2);boxes=np.column_stack([(b[:,0]+b[:,2])/2,(b[:,1]+b[:,3])/2,b[:,2]-b[:,0],b[:,3]-b[:,1]])
    else:ids=z['frame_ids'];boxes=z[arm]
    gold=official(boxes,dict(input=row['metadata'],frame_ids=ids.tolist()),gt,g['interval'],z['shared_interval'],ds);err=max(abs(v[k]-gold[k])for k in ['t','v']);assert err<2e-10;mx=max(mx,err);checks+=1
   assert len({x['t']for x in values.values()})==1
   for field in ['event_GT_frames','event_observed_frames','event_unobserved_frames']:assert len({v[field]for v in values.values()})==1
   assert abs(values['Full']['v']-oldscalar[key]['values']['Full']['v'])<2e-10
  out.append(dict(id=key,parent=str(g['parent']),duration_group='short_le5s'if(g['interval'][1]-g['interval'][0])/row['metadata']['fps']<=5 else'long_gt5s',values=values))
  if i%100==0:atomic(dest/'score_progress.json',dict(completed=i,total=COUNTS[ds],worker=identity()))
 report(out,ds);atomic(dest/'metrics.json',dict(private_rows=out));rec=dict(status='fresh_whole_three_spatial_paths_common_time_common_dense_support_official_scored_PENDING_audit',queries=len(out),official_checks=checks,max_official_discrepancy=mx,same_perquery_tIoU_and_event_frame_counts=True,unchanged_currentFull_metric_reproduced=True,seal_sha256=sha(dest/'sealed.json'),metrics_sha256=sha(dest/'metrics.json'),report_sha256=sha(P/ds/'report.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start);atomic(dest/'score_receipt.json',rec);print(json.dumps(rec),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=COUNTS,required=True);main(p.parse_args().dataset)
