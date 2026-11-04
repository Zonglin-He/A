"""Fresh evaluator independent of stvg.metrics/DenseTube/full_score.metric."""
from common import *
import argparse,csv

def evaluate_array(boxes,ids,positions,metadata,truth,span,iv,ds,reference=False):
 W,H=metadata['width'],metadata['height'];ids=np.asarray(ids);f=np.array(sorted(truth));gt=np.array([truth[int(t)]for t in f],dtype=float)
 if not reference:
  pix=corner(boxes,W,H);pix=np.maximum(pix,0)if ds=='hc2'else pix;pred=np.column_stack([np.interp(f,ids,pix[:,k])for k in range(4)])
  inter=np.maximum(0,np.minimum(pred[:,2:],gt[:,2:])-np.maximum(pred[:,:2],gt[:,:2])).prod(1);den=np.maximum(0,pred[:,2:]-pred[:,:2]).prod(1)+np.maximum(0,gt[:,2:]-gt[:,:2]).prod(1)-inter;iou=np.divide(inter,den,out=np.zeros(len(f)),where=den>0)
 else:
  # Independent normalized-coordinate interpolation by bracket search,
  # explicit componentwise area formula (no shared corner/IoU helper).
  b=np.asarray(boxes,dtype=float);corn=np.stack((b[:,0]-b[:,2]/2,b[:,1]-b[:,3]/2,b[:,0]+b[:,2]/2,b[:,1]+b[:,3]/2),axis=1);corn=np.maximum(0,corn)if ds=='hc2'else corn;target=gt/np.array([W,H,W,H]);j=np.clip(np.searchsorted(ids,f,side='right')-1,0,len(ids)-2);q=(f-ids[j])/(ids[j+1]-ids[j]);pred=corn[j]+q[:,None]*(corn[j+1]-corn[j]);pred[f<=ids[0]]=corn[0];pred[f>=ids[-1]]=corn[-1];iw=np.maximum(0,np.minimum(pred[:,2],target[:,2])-np.maximum(pred[:,0],target[:,0]));ih=np.maximum(0,np.minimum(pred[:,3],target[:,3])-np.maximum(pred[:,1],target[:,1]));num=iw*ih;area=np.maximum(0,pred[:,2]-pred[:,0])*np.maximum(0,pred[:,3]-pred[:,1]);garea=np.maximum(0,target[:,2]-target[:,0])*np.maximum(0,target[:,3]-target[:,1]);union=area+garea-num;iou=np.divide(num,union,out=np.zeros(len(f)),where=union>0)
 iou[(f<ids[0])|(f>ids[-1])]=0;a,b=map(int,iv);g,h=map(int,span);assert a<b and g<h;overlap=max(0,min(b,h)-max(a,g));t=overlap/(b-a+h-g-overlap);mask=(f>=max(a,g))&(f<min(b,h));v=float(iou[mask].sum()/max(max(b,h)-min(a,g),1))
 # Native GT support only for observed/unobserved diagnostic, independent dense scores above.
 native_mask=np.isin(f,ids);obs=np.isin(f,ids[np.asarray(positions,int)]);fixed=iou[native_mask]
 out=dict(t=float(t),v=v,s=float(iou.mean()),fixed_actual_sIoU=float(fixed.mean())if len(fixed)else None,observed_actual_sIoU=float(iou[obs].mean())if obs.any()else None,unobserved_actual_sIoU=float(iou[native_mask&~obs].mean())if(native_mask&~obs).any()else None,vIoU_ge_03=float(v>=.3),vIoU_ge_05=float(v>=.5),boundary_MAE_seconds=(abs(a-g)+abs(b-h))/(2*metadata['fps']))
 return out

def main(ds):
 validate();torch.set_num_threads(1);sys.addaudithook(old.deny_gpu_media);start=time.perf_counter();dest=D/ds;seal=read(dest/'sealed.json');assert seal['queries']==COUNTS[ds]and not seal['GT_read'];rr=rows(ds);truth=old.selected.labels(ds);keys={r['id']for r in rr};assert set(seal['files'])==set(truth)==keys
 historical={x['id']:x for x in read(TEMP/'temporal'/ds/'metrics.json')['private_rows']};pure={x['id']:x for x in read(PURE/'pure_full'/ds/'metrics.json')['private_rows']};combo={x['id']:x for x in read(COMBO/'combination'/ds/'metrics.json')['private_rows']};assert set(historical)==set(pure)==set(combo)==keys
 from stvg.metrics import official
 out=[];max_official=0.;max_old=0.;checks=0;oldchecks=0
 for i,row in enumerate(rr,1):
  resources(start);key=row['id'];e=seal['files'][key];assert sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256'];g=truth[key];gt={int(f):b for f,b in g['boxes'].items()};values={}
  with np.load(D/e['path'])as z:
   ids=z['frame_ids'];pos=z['positions8']
   for arm in ARMS:
    value=evaluate_array(z[arm],ids,pos,row['metadata'],gt,g['interval'],z[arm+'_interval'],ds);values[arm]=value
    if arm in ['Full','ComboLog','PureLogFixed']:
     gold=official(z[arm],dict(input=row['metadata'],frame_ids=ids.tolist()),gt,g['interval'],z[arm+'_interval'],ds);err=max(abs(value[k]-gold[k])for k in ['t','v','s']);max_official=max(max_official,err);assert err<2e-10,(key,arm,err);checks+=1
    oldvalue=(historical[key]['values'][{'Source':'Source','SourceTemporal':'SourceTR','Full':'Ours'}[arm]]if arm in ['Source','SourceTemporal','Full']else pure[key]['values']['PureMatched']if arm=='PureLogFixed'else combo[key]['values']['SimpleCombination']if arm=='ComboLog'else None)
    if oldvalue is not None:
     for field in value:
      if field not in oldvalue:continue
      a,b=value[field],oldvalue[field];assert(a is None)==(b is None)
      if a is not None:delta=abs(a-b);max_old=max(max_old,delta);assert delta<2e-10,(key,arm,field,a,b);oldchecks+=1
   assert values['Full']['t']==values['PureLogFixed']['t']==values['PureLinearFixed']['t']and values['SourceTemporal']['t']==values['ComboLog']['t']==values['ComboLinear']['t']
  assert str(g['parent'])==str(historical[key]['parent'])==str(pure[key]['parent'])==str(combo[key]['parent']);out.append(dict(id=key,parent=str(g['parent']),duration_group='short_le5s'if(g['interval'][1]-g['interval'][0])/row['metadata']['fps']<=5 else'long_gt5s',values=values))
  if i%100==0:atomic(dest/'score_progress.json',dict(completed=i,total=len(rr),worker=identity()))
 old.selected.suite().parent_report(out,ARMS,PAIRS,P/ds);old.selected.suite().signed_tails(out,PAIRS,P/ds);atomic(dest/'metrics.json',dict(private_rows=out));rec=dict(status='fresh_all7arm_independent_dense_scores_all3keyarms_official_perquery_checks_complete_PENDING_audit',queries=len(out),parents=len({r['parent']for r in out}),new_official_checks=checks,max_official_discrepancy=max_official,old_perquery_scalar_checks=oldchecks,max_old_scalar_discrepancy=max_old,seal_sha256=sha(dest/'sealed.json'),metrics_sha256=sha(dest/'metrics.json'),report_sha256=sha(P/ds/'report.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_Source=0,new_expert=0,new_GPU=0,new_model=0);atomic(dest/'score_receipt.json',rec);print(json.dumps(rec),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=COUNTS,required=True);main(p.parse_args().dataset)
