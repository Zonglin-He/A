from common import *
import argparse,gc,traceback

def derive(ds,key):
 import numpy as np
 e=read(selected.D/f'components_{ds}_sealed.json')['files'][key]
 for f,h in [('path','sha256'),('receipt_path','receipt_sha256')]:assert sha(selected.D/e[f])==e[h]
 with np.load(selected.D/e['path'])as z:a={k:z[k].copy()for k in ['Source','Source_interval','FullMAP','FullMAP_interval','frame_ids','positions8']}
 b=a['FullMAP'];assert np.isfinite(b).all()and(b[:,2:]>=0).all();lo=b[:,:2]-b[:,2:]/2;hi=b[:,:2]+b[:,2:]/2;ll=np.clip(lo,0,1);hh=np.clip(hi,0,1)
 a['ClipToImage']=np.concatenate([(ll+hh)/2,hh-ll],axis=1);assert(a['ClipToImage'][:,2:]>=0).all();area=np.maximum(0,np.minimum(hi,1)-np.maximum(lo,0)).prod(1);bad=(b[:,2]<=0)|(b[:,3]<=0)|(area==0);fb=b.copy();fb[bad]=a['Source'][bad];a['SourceFallbackFullOutside']=fb
 for arm in ['ClipToImage','SourceFallbackFullOutside']:a[arm+'_interval']=a['FullMAP_interval'].copy()
 proof=dict(parent_prediction_sha256=e['sha256'],parent_receipt_sha256=e['receipt_sha256'],clipped_frames=int(((lo<0).any(1)|(hi>1).any(1)).sum()),fallback_frames=int(bad.sum()),same_current_predicted_event=True,no_GT_gate=True,new_Source=0,new_expert=0)
 return a,proof

def predict(ds,preflight=False):
 import numpy as np,torch
 torch.set_num_threads(1);validate();gt_guard();start=time.perf_counter();files={}
 for i,row in enumerate(records(ds)[:2]if preflight else records(ds),1):
  resources(start);a,p=derive(ds,row['id'])
  if preflight:
   b,q=derive(ds,row['id']);assert p==q and all(np.array_equal(v,b[k])for k,v in a.items())
  else:
   path=D/ds/'predictions'/(row['id']+'.npz');assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True)
   with path.with_suffix('.tmp').open('wb')as f:np.savez_compressed(f,**a)
   os.replace(path.with_suffix('.tmp'),path)
   with np.load(path)as z:assert set(z.files)==set(a)and all(np.array_equal(v,z[k])for k,v in a.items())
   files[row['id']]=dict(path=str(path.relative_to(D)),sha256=sha(path),provenance=p)
  if i%250==0:atomic(D/ds/'predict_progress.json',dict(completed=i,total=COUNTS[ds],worker=identity(),GT_read=False))
  del a
 atomic(D/ds/('preflight.json'if preflight else'sealed.json'),dict(status='passed_two_actual_cached_geometric_controls_repeat_exact'if preflight else'sealed_whole_domain_before_GT',queries=i,files=files,worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,GT_read=False,new_model=0,new_Source=0,new_expert=0))

def score(ds):
 import numpy as np,torch
 from full_score import metric
 torch.set_num_threads(1);validate();seal=read(D/ds/'sealed.json');assert seal['queries']==COUNTS[ds]and not seal['GT_read'];truth=selected.labels(ds);old={r['id']:r for r in read(selected.D/'components'/ds/'metrics.json')['private_rows']};start=time.perf_counter();rr=[];checks=0;maxerr=0.
 for i,row in enumerate(records(ds),1):
  resources(start);key=row['id'];g=truth[key];e=seal['files'][key];assert sha(D/e['path'])==e['sha256'];o=old[key];assert str(g['parent'])==o['parent'];values={a:o['values'][a]for a in ['Source','FullMAP']}
  with np.load(D/e['path'])as z:
   inp={**row,'metadata':{**row['metadata'],'frame_ids':z['frame_ids'].tolist()}}
   for a in ['ClipToImage','SourceFallbackFullOutside']:values[a],err=metric(z[a],inp,{int(k):v for k,v in g['boxes'].items()},g['interval'],z[a+'_interval'].tolist(),z['positions8']);assert err<2e-10;checks+=1;maxerr=max(maxerr,err)
  assert values['ClipToImage']['t']==values['SourceFallbackFullOutside']['t']==values['FullMAP']['t'];rr.append(dict(id=key,parent=o['parent'],duration_group=o['duration_group'],values=values))
  if i%250==0:atomic(D/ds/'score_progress.json',dict(completed=i,total=COUNTS[ds],worker=identity()))
 out=P/ds;rep=selected.suite().parent_report(rr,ARMS,PAIRS,out);selected.suite().signed_tails(rr,PAIRS,out);atomic(D/ds/'metrics.json',dict(private_rows=rr,report=rep));receipt=dict(status='scored_complete_geometry_controls_with_exact_audited_Source_Full_scalar_inheritance',queries=len(rr),new_official_checks=checks,max_error=maxerr,whole_seal_sha256=sha(D/ds/'sealed.json'),inherited_scalar_sha256=sha(selected.D/'components'/ds/'metrics.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start);atomic(D/ds/'score_receipt.json',receipt);atomic(out/'score_receipt.json',receipt)

def audit(ds):
 import numpy as np,torch
 torch.set_num_threads(1);validate();gt_guard();seal=read(D/ds/'sealed.json');assert read(D/ds/'score_receipt.json')['whole_seal_sha256']==sha(D/ds/'sealed.json');start=time.perf_counter()
 for row in records(ds):
  e=seal['files'][row['id']];assert sha(D/e['path'])==e['sha256'];a,p=derive(ds,row['id']);assert p==e['provenance']
  with np.load(D/e['path'])as z:assert set(z.files)==set(a)and all(np.array_equal(v,z[k])for k,v in a.items())
 rr=read(D/ds/'metrics.json')['private_rows'];rep=read(P/ds/'report.json');assert len(rr)==COUNTS[ds]
 for arm,m in rep['means'].items():
  for k,v in m.items():
   x=[r['values'][arm][k]for r in rr if r['values'][arm][k]is not None];assert(v is None and not x)or abs(v-float(np.mean(x)))<1e-12
 rec=dict(status='verified_whole_domain_geometric_controls_arrays_parent_SHA_inherited_baselines_and_all_means',queries=len(rr),all_arrays_rederived_exact=True,score_sha256=sha(D/ds/'score_receipt.json'),metrics_sha256=sha(D/ds/'metrics.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_model=0,new_Source=0,new_expert=0,new_GT_scoring=0);atomic(D/ds/'reporting_receipt.json',rec);atomic(P/ds/'reporting_receipt.json',rec)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['preflight','predict','score','audit']);p.add_argument('--dataset',choices=COUNTS,required=True);a=p.parse_args()
 try:predict(a.dataset,True)if a.command=='preflight'else globals()[a.command](a.dataset)
 except BaseException:atomic(D/'failures'/(a.command+'_'+a.dataset+'.json'),dict(error=traceback.format_exc(),worker=identity()));raise
