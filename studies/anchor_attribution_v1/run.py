"""Frozen-cache attribution: direct eight boxes before/after shared calibration."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json,time,argparse,hashlib,csv
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'tools'));sys.path.insert(0,str(R))
from full import RUN,PROVIDER,read,sha,atomic,check,identity
D=R/'outputs/anchor_attribution_v1';PUB=R/'results/anchor_attribution_v1'
ARMS=('Direct8','SharedDirect8')
def frozen():
 p=read(D/'plan.json');body={k:v for k,v in p.items()if k!='fingerprint'};assert hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()==p['fingerprint'];assert sha(Path(__file__))==p['code_sha256'];check()
 for ds,h in p['parent_seals'].items():assert sha(RUN/ds/'sealed.json')==h
 return p

def predict():
 import numpy as np
 p=frozen();assert not(D/'sealed.json').exists();atomic(D/'predictor.json',identity());started=time.perf_counter();files={}
 label_paths={(PROVIDER/v['labels']).resolve()for v in read(RUN/'plan.json')['jobs'].values()}
 def guard(event,args):
  if event=='open'and args and isinstance(args[0],(str,bytes,Path))and Path(os.fsdecode(args[0])).resolve()in label_paths:raise PermissionError('Prediction cannot read labels')
 sys.addaudithook(guard)
 for ds,total in [('hc2',1902),('vidstg',10303)]:
  seal=read(RUN/ds/'sealed.json');files[ds]={}
  for i,(key,e)in enumerate(seal['files'].items(),1):
   parent=RUN/ds/e['path'];assert sha(parent)==e['sha256'];rc=read(RUN/ds/e['receipt_path']);assert rc['fingerprint']==p['parent_fingerprint']and rc['expert_forwards']==0
   with np.load(parent,allow_pickle=False)as z:
    ids=z['frame_ids'];pos=z['positions8'];native=z['native_boxes'];assert native.shape==(8,4)and len(set(map(int,pos)))==8
    direct=z['Source'].copy();shared=z['SharedInit'].copy();direct[pos]=native;shared[pos]=native
    un=np.ones(len(ids),bool);un[pos]=False;assert np.array_equal(direct[un],z['Source'][un])and np.array_equal(shared[un],z['SharedInit'][un])and np.array_equal(direct[pos],shared[pos])
    iv=z['Source_interval'].copy();assert np.array_equal(iv,z['OursS_interval'])and np.array_equal(iv,z['Interpolation_interval'])
    target=D/ds/(key+'.npz');target.parent.mkdir(exist_ok=True);assert not target.exists();np.savez_compressed(target,Direct8=direct,SharedDirect8=shared,interval=iv,positions8=pos,frame_ids=ids)
   files[ds][key]=dict(path=str(target.relative_to(D)),sha256=sha(target),parent_prediction_sha256=e['sha256'])
   if i%100==0:atomic(D/'predict_progress.json',dict(dataset=ds,completed=i,total=total,GT_read=False,**identity()))
  assert len(files[ds])==total
 atomic(D/'sealed.json',dict(status='sealed_before_labels',fingerprint=p['fingerprint'],files=files,queries=12205,GT_read=False,new_model=0,new_expert=0,CPU_seconds=time.perf_counter()-started));print('Predictions sealed: HC1902 + Vid10303; no labels/model/expert',flush=True)

def score():
 import numpy as np,torch
 from full_score import metric,cluster
 torch.set_num_threads(1);p=frozen();seal=read(D/'sealed.json');assert seal['fingerprint']==p['fingerprint']and seal['GT_read']is False
 atomic(D/'scorer.json',identity());started=time.perf_counter();allreports={};checks=0;maxerr=0.
 parentplan=read(RUN/'plan.json')
 for ds in ['hc2','vidstg']:
  job=parentplan['jobs'][ds];label=PROVIDER/job['labels'];assert sha(label)==job['labels_sha256'];truth={x['id']:x for x in map(json.loads,label.read_text().splitlines())if x['dataset']==ds};records=[json.loads(s)for s in(PROVIDER/job['manifest']).read_text().splitlines()];old={x['id']:x for x in read(RUN/ds/'metrics.json')['private_rows']};rows=[]
  for i,r in enumerate(records,1):
   key=r['id'];e=seal['files'][ds][key];assert sha(D/e['path'])==e['sha256'];g=truth[key];gt={int(k):v for k,v in g['boxes'].items()};vals={a:old[key]['values'][a]for a in ['Source','SharedInit','Interpolation','OursS','Full']}
   with np.load(D/e['path'],allow_pickle=False)as z:
    for a in ARMS:
     vals[a],err=metric(z[a],r,gt,g['interval'],z['interval'].tolist(),z['positions8']);maxerr=max(maxerr,err);checks+=1
    for f in ['unobserved_actual_sIoU']:
     assert vals['Direct8'][f]==vals['Source'][f]and vals['SharedDirect8'][f]==vals['SharedInit'][f]
    for a in ARMS:assert vals[a]['t']==vals['Source']['t']
   rows.append(dict(id=key,parent=old[key]['parent'],values=vals,duration_group=old[key]['duration_group']))
   if i%100==0:atomic(D/'score_progress.json',dict(dataset=ds,completed=i,total=len(records),CPU_only=True))
  arms=list(rows[0]['values']);fields=['v','t','observed_actual_sIoU','unobserved_actual_sIoU'];means={a:{f:float(np.mean([x['values'][a][f]for x in rows if x['values'][a][f]is not None]))for f in fields}for a in arms}
  pairs=[('Direct8','Source'),('SharedDirect8','Direct8'),('OursS','SharedDirect8'),('Interpolation','SharedDirect8'),('OursS','Interpolation'),('Full','OursS')];paired={a+'-'+b:cluster(rows,a,b)for a,b in pairs};report=dict(status='completed',dataset=ds,queries=len(rows),parents=len({x['parent']for x in rows}),means=means,paired=paired,scope='Post-full attribution on exposed official sets, fixed native interval and all eight boxes. Direct8=Source+raw8; SharedDirect8=SharedInit+raw8. Both overwrite all eight observed positions, including outside predicted interval; official v clips to native interval. OursS may select initialization instead of raw anchor at observed positions and gates propagation by native interval, so OursS-SharedDirect8 is a spatial-system contrast, not pure propagation causality.',fingerprint=p['fingerprint'])
  allreports[ds]=report;atomic(PUB/ds/'report.json',report);atomic(D/ds/'metrics.json',dict(report,private_rows=rows));f=PUB/ds/'signed.tsv'
  with f.open('w')as out:
   w=csv.writer(out,delimiter='\t',lineterminator='\n');w.writerow(['query']+[a+'_'+v for a in arms for v in fields])
   for i,x in enumerate(rows,1):w.writerow([f'{ds[0].upper()}{i:05d}']+[x['values'][a][v]for a in arms for v in fields])
 atomic(D/'score_receipt.json',dict(status='scored',fingerprint=p['fingerprint'],seal_sha256=sha(D/'sealed.json'),official_dense_checks=checks,max_error=maxerr,CPU_seconds=time.perf_counter()-started,public_files={str(f.relative_to(R)):sha(f)for f in PUB.rglob('*')if f.is_file()},new_model=0,new_expert=0));print(json.dumps(dict(status='completed',official_dense_checks=checks,max_error=maxerr),indent=2),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['predict','score']);args=a.parse_args();predict()if args.command=='predict'else score()
