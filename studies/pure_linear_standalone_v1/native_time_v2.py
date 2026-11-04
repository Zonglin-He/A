"""Fix only native Source evaluation binding; NEVER regenerate pure trajectories."""
from common import *
from score import measure
import argparse
N=D/'native_time_v2';Q=P/'native_time_v2'

def native_interval(ds,key):
 e=fixed(CHECK/ds/'sealed.json')['files'][key];assert sha(CHECK/e['path'])==e['sha256']
 with np.load(CHECK/e['path'])as z:iv=z['Source_interval'].copy()
 # Second independently audited native Source producer, not historical MBR alias.
 root=R/'outputs/reviewer_key_controls_v1';e=fixed(root/'temporal'/ds/'sealed.json')['files'][key];assert sha(root/e['path'])==e['sha256']
 with np.load(root/e['path'])as z:assert np.array_equal(iv,z['Source_interval'])
 return iv

def check():
 p=validate();x=read(D/'native_execution_v2.json');assert x['fingerprint']==digest({k:v for k,v in x.items()if k!='fingerprint'})and x['scientific_plan_sha256']==sha(D/'plan.json')
 for f,h in x['code_sha256'].items():assert sha(R/f)==h
 return p

def score(ds):
 check();sys.addaudithook(guard);torch.set_num_threads(1);start=time.perf_counter();seal=read(D/ds/'sealed.json');audit=read(D/ds/'audit_receipt.json');assert audit['metrics_sha256']==sha(D/ds/'metrics.json')and audit['seal_sha256']==sha(D/ds/'sealed.json');old={r['id']:r for r in read(D/ds/'metrics.json')['private_rows']};reference={r['id']:r for r in read(CHECK/ds/'metrics.json')['private_rows']};truth=labels(ds);out=[];checks=0;mx=0.
 from stvg.metrics import official
 from tools.full_score import cluster
 for i,row in enumerate(rows(ds),1):
  resources(start);key=row['id'];e=seal['files'][key];assert sha(D/e['path'])==e['sha256'];iv=native_interval(ds,key);g=truth[key];gt={int(k):b for k,b in g['boxes'].items()}
  with np.load(D/e['path'])as z:
   xy=z['boxes_xyxy'];ids=z['frame_ids'];m={**row['metadata'],'dataset':ds};v=measure(xy,ids,z['anchor_times'],m,gt,g['interval'],iv);b=xy/np.array([m['width'],m['height']]*2);cx=np.column_stack([(b[:,0]+b[:,2])/2,(b[:,1]+b[:,3])/2,b[:,2]-b[:,0],b[:,3]-b[:,1]]);gold=official(cx,dict(input=m,frame_ids=ids.tolist()),gt,g['interval'],iv,ds);err=max(abs(v[k]-gold[k])for k in ['t','v','s']);assert err<2e-10;mx=max(mx,err);checks+=1
  assert v['t']==reference[key]['values']['Source']['t'],(key,'native_Source_readout_mismatch')
  values={ARMS[0]:v,ARMS[1]:old[key]['values'][ARMS[1]],'OriginalFull':reference[key]['values']['Full'],'OldSparseLinearFixed':reference[key]['values']['PureLinearFixed']};assert str(g['parent'])==old[key]['parent']==str(reference[key]['parent']);assert values[ARMS[1]]['t']==values['OriginalFull']['t'];out.append(dict(id=key,parent=str(g['parent']),duration_group=old[key]['duration_group'],values=values))
  if i%100==0:atomic(N/ds/'score_progress.json',dict(completed=i,total=COUNTS[ds],worker=identity()))
 arms=list(out[0]['values']);means={a:{k:float(np.mean(vv))if(vv:=[r['values'][a][k]for r in out if r['values'][a][k]is not None])else None for k in ['t','v','vIoU_ge_03','vIoU_ge_05','observed_actual_sIoU','unobserved_actual_sIoU','boundary_MAE_seconds']}for a in arms};pairs=[('OriginalFull',ARMS[1]),(ARMS[1],'OldSparseLinearFixed')];paired={a+'-'+b:cluster(out,a,b)for a,b in pairs};groups={g:dict(queries=len(rr),paired={a+'-'+b:cluster(rr,a,b)for a,b in pairs})for g in ['short_le5s','long_gt5s']if(rr:=[r for r in out if r['duration_group']==g])}
 atomic(N/ds/'metrics.json',dict(private_rows=out));atomic(Q/ds/'report.json',dict(queries=len(out),parents=len({r['parent']for r in out}),means=means,paired=paired,timegroups=groups,scope='One pure raw8corner-linear spatial estimator. SourceTime is actual restored per-offset MAP hull native Source time, verified every query against original native Source scores and independent two producer intervals. FullTime is explicitly separate evaluation protocol only, not expert-alone event prediction. FullTime old v1 scores reused exact, never rescored. Unobserved pure statistic is dense input-frame support and differs from historical native-sampled support; vIoU same official dense labels.'))
 with(Q/ds/'all_signed_scalars.tsv').open('w')as f:
  import csv
  w=csv.writer(f,delimiter='\t');w.writerow(['query','timegroup','arm','tIoU','vIoU','R03','R05','observedIoU','unobservedIoU'])
  for i,r in enumerate(out,1):
   for a in arms:v=r['values'][a];w.writerow([f'Q{i:05}',r['duration_group'],a,*[v[k]for k in ['t','v','vIoU_ge_03','vIoU_ge_05','observed_actual_sIoU','unobserved_actual_sIoU']]])
 rec=dict(status='actual_native_Source_only_rescored_FullTime_reused_PENDING_independent_audit',queries=len(out),official_checks=checks,max_official_discrepancy=mx,native_Source_time_exact_all_queries=True,pure_trajectory_regenerated=False,FullTime_reevaluated=False,prediction_seal_sha256=sha(D/ds/'sealed.json'),previous_FullTime_metrics_sha256=sha(D/ds/'metrics.json'),previous_FullTime_audit_sha256=sha(D/ds/'audit_receipt.json'),metrics_sha256=sha(N/ds/'metrics.json'),report_sha256=sha(Q/ds/'report.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start);atomic(N/ds/'score_receipt.json',rec);print(json.dumps(rec),flush=True)

def audit(ds):
 check();sys.addaudithook(guard);torch.set_num_threads(1);start=time.perf_counter();seal=read(D/ds/'sealed.json');sc=read(N/ds/'score_receipt.json');assert sc['metrics_sha256']==sha(N/ds/'metrics.json')and sc['prediction_seal_sha256']==sha(D/ds/'sealed.json');truth=labels(ds);out=read(N/ds/'metrics.json')['private_rows'];m={r['id']:r for r in out};old={r['id']:r for r in read(D/ds/'metrics.json')['private_rows']};checks=0;mx=0.
 for i,row in enumerate(rows(ds),1):
  resources(start);key=row['id'];e=seal['files'][key];assert sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256'];iv=native_interval(ds,key);g=truth[key];gt={int(k):v for k,v in g['boxes'].items()}
  with np.load(D/e['path'])as z:
   xy=z['boxes_xyxy'];ids=z['frame_ids'];scale=np.array([row['metadata']['width'],row['metadata']['height']]*2);f=np.array(sorted(gt));valid=(f>=ids[0])&(f<=ids[-1]);p=np.zeros((len(f),4));p[valid]=xy[f[valid]-ids[0]]/scale
   if ds=='hc2':p=np.maximum(0,p)
   t=np.array([gt[int(k)]for k in f])/scale;inter=np.maximum(0,np.minimum(p[:,2:],t[:,2:])-np.maximum(p[:,:2],t[:,:2])).prod(1)
   # Explicit independent component areas avoid shared scoring helper.
   union=np.maximum(0,p[:,2]-p[:,0])*np.maximum(0,p[:,3]-p[:,1])+np.maximum(0,t[:,2]-t[:,0])*np.maximum(0,t[:,3]-t[:,1])-inter;iou=np.divide(inter,union,out=np.zeros(len(f)),where=union>0);iou[~valid]=0;a,b=map(int,iv);c,d=map(int,g['interval']);use=(f>=max(a,c))&(f<min(b,d));v=float(iou[use].sum()/max(max(b,d)-min(a,c),1));err=abs(v-m[key]['values'][ARMS[0]]['v']);mx=max(mx,err);assert err<2e-10;checks+=1
  assert m[key]['values'][ARMS[1]]==old[key]['values'][ARMS[1]]and m[key]['parent']==str(g['parent']);checks+=1
 rep=read(Q/ds/'report.json');mc=0
 for arm in ARMS:
  for f,v in rep['means'][arm].items():
   vv=[r['values'][arm][f]for r in out if r['values'][arm][f]is not None];assert(v is None and not vv)or abs(v-float(np.mean(vv)))<1e-12;mc+=1
 rec=dict(status='verified_actual_native_Source_readout_denseIoU_allarrays_SHA_parent_reused_FullTime_means',queries=len(out),parents=len({r['parent']for r in out}),checks=checks,max_independent_metric_error=mx,means_checks=mc,metrics_sha256=sha(N/ds/'metrics.json'),report_sha256=sha(Q/ds/'report.json'),score_sha256=sha(N/ds/'score_receipt.json'),previous_trajectory_audit_sha256=sha(D/ds/'audit_receipt.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start);atomic(N/ds/'audit_receipt.json',rec);atomic(Q/ds/'audit_receipt.json',rec);print(json.dumps(rec),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['score','audit']);p.add_argument('--dataset',choices=COUNTS,required=True);a=p.parse_args();globals()[a.stage](a.dataset)
