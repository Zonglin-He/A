from common import *
import argparse,csv
from stvg.metrics import official

def measure(xy,ids,anchors,m,gt,span,iv):
 # Independent dense pixel IoU, no interpolation of the already dense trajectory.
 f=np.array(sorted(gt));g=np.asarray([gt[int(k)]for k in f],float);inside=(f>=ids[0])&(f<=ids[-1]);p=np.zeros((len(f),4));p[inside]=xy[np.searchsorted(ids,f[inside])]
 if m['dataset']=='hc2':p=np.maximum(p,0)
 w=np.maximum(0,np.minimum(p[:,2],g[:,2])-np.maximum(p[:,0],g[:,0]));h=np.maximum(0,np.minimum(p[:,3],g[:,3])-np.maximum(p[:,1],g[:,1]));inter=w*h;area=np.maximum(0,p[:,2]-p[:,0])*np.maximum(0,p[:,3]-p[:,1]);ga=np.maximum(0,g[:,2]-g[:,0])*np.maximum(0,g[:,3]-g[:,1]);den=area+ga-inter;iou=np.divide(inter,den,out=np.zeros(len(f)),where=den>0);iou[~inside]=0
 a,b=map(int,iv);c,d=map(int,span);overlap=max(0,min(b,d)-max(a,c));t=overlap/(b-a+d-c-overlap);use=(f>=max(a,c))&(f<min(b,d));v=float(iou[use].sum()/max(max(b,d)-min(a,c),1));obs=inside&np.isin(f,anchors);unobs=inside&~np.isin(f,anchors)
 return dict(t=float(t),v=v,s=float(iou.mean()),vIoU_ge_03=float(v>.3),vIoU_ge_05=float(v>.5),observed_actual_sIoU=float(iou[obs].mean())if obs.any()else None,unobserved_actual_sIoU=float(iou[unobs].mean())if unobs.any()else None,boundary_MAE_seconds=(abs(a-c)+abs(b-d))/(2*m['fps']))

def report(out,ds):
 # Only one spatial algorithm; two explicitly separate temporal evaluation protocols.
 from tools.full_score import cluster
 gold={r['id']:r for r in fixed(CHECK/ds/'metrics.json')['private_rows']}
 q=[]
 for r in out:q.append({**r,'values':{**r['values'],'OriginalFull':gold[r['id']]['values']['Full'],'OldSparseLinearFixed':gold[r['id']]['values']['PureLinearFixed']}})
 arms=ARMS+['OriginalFull','OldSparseLinearFixed'];means={a:{k:float(np.mean(vv))if(vv:=[r['values'][a][k]for r in q if r['values'][a][k]is not None])else None for k in q[0]['values'][a]if k in ['v','t','vIoU_ge_03','vIoU_ge_05','observed_actual_sIoU','unobserved_actual_sIoU','boundary_MAE_seconds']}for a in arms};pairs=[('OriginalFull',ARMS[1]),(ARMS[1],'OldSparseLinearFixed')];paired={a+'-'+b:cluster(q,a,b)for a,b in pairs}
 groups={g:dict(queries=len(rr),paired={a+'-'+b:cluster(rr,a,b)for a,b in pairs})for g in ['short_le5s','long_gt5s']if(rr:=[r for r in q if r['duration_group']==g])};tails={}
 for a,b in pairs:
  dv=np.array([100*(r['values'][a]['v']-r['values'][b]['v'])for r in q]);dt=np.array([100*(r['values'][a]['t']-r['values'][b]['t'])for r in q]);tails[a+'-'+b]=dict(improved=int((dv>1e-10).sum()),harmed=int((dv< -1e-10).sum()),ties=int((abs(dv)<=1e-10).sum()),severe_v_over5pp=int((dv< -5).sum()),severe_t_over5pp=int((dt< -5).sum()),severe_union_once=int(((dv< -5)|(dt< -5)).sum()),rescues_union_once=int(((dv>5)|(dt>5)).sum()))
 folder=P/ds;atomic(folder/'report.json',dict(queries=len(q),parents=len({r['parent']for r in q}),means=means,paired=paired,timegroups=groups,tails=tails,scope='Single pure expert spatial algorithm; native Source interval is independent baseline, Full interval separate controlled evaluation only. Unobserved support now all actual input video frames, OLD unobserved metric used Source sample grid; do not interpret cross-support unobserved differences as path-only effect. vIoU has same official dense annotation support.'))
 with(folder/'all_signed_scalars.tsv').open('w')as f:
  w=csv.writer(f,delimiter='\t');w.writerow(['query','timegroup','arm','tIoU','vIoU','R03','R05','observedIoU','unobservedIoU'])
  for i,r in enumerate(q,1):
   for a in arms:z=r['values'][a];w.writerow([f'Q{i:05}',r['duration_group'],a,*[z[k]for k in ['t','v','vIoU_ge_03','vIoU_ge_05','observed_actual_sIoU','unobserved_actual_sIoU']]])
 return q

def main(ds):
 validate();sys.addaudithook(guard);torch.set_num_threads(1);start=time.perf_counter();dest=D/ds;seal=read(dest/'sealed.json');assert seal['queries']==COUNTS[ds]and not seal['GT_read']and seal['no_temporal_interval_in_prediction'];truth=labels(ds);out=[];mx=0.;checks=0
 for i,row in enumerate(rows(ds),1):
  resources(start);key=row['id'];e=seal['files'][key];assert sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256'];g=truth[key];gt={int(k):b for k,b in g['boxes'].items()};ivs=evaluation_intervals(ds,key);values={}
  with np.load(D/e['path'])as z:
   xy=z['boxes_xyxy'];ids=z['frame_ids'];b=xy/np.array([row['metadata']['width'],row['metadata']['height']]*2);cx=np.column_stack([(b[:,0]+b[:,2])/2,(b[:,1]+b[:,3])/2,b[:,2]-b[:,0],b[:,3]-b[:,1]]);m={**row['metadata'],'dataset':ds}
   for arm,iv in ivs.items():
    v=measure(xy,ids,z['anchor_times'],m,gt,g['interval'],iv);gold=official(cx,dict(input=row['metadata'],frame_ids=ids.tolist()),gt,g['interval'],iv,ds);err=max(abs(gold[k]-v[k])for k in ['v','t','s']);assert err<2e-10;mx=max(mx,err);values[arm]=v;checks+=1
  out.append(dict(id=key,parent=str(g['parent']),duration_group='short_le5s'if(g['interval'][1]-g['interval'][0])/row['metadata']['fps']<=5 else'long_gt5s',values=values))
  if i%100==0:atomic(dest/'score_progress.json',dict(completed=i,total=COUNTS[ds],worker=identity()))
 report(out,ds);atomic(dest/'metrics.json',dict(private_rows=out));rec=dict(status='whole_single_pure_trajectory_separate_two_time_protocols_official_scored_PENDING_audit',queries=len(out),official_checks=checks,max_official_discrepancy=mx,seal_sha256=sha(dest/'sealed.json'),metrics_sha256=sha(dest/'metrics.json'),report_sha256=sha(P/ds/'report.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start);atomic(dest/'score_receipt.json',rec);print(json.dumps(rec),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=COUNTS,required=True);main(p.parse_args().dataset)
