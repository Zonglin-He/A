"""Task-aware expert pseudo-box TTA baseline: four episodic box residuals.

Not VLOD-TTA or network training. Same eight native teacher outputs, no GT.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json,time,hashlib,argparse,csv
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R),str(R/'tools')]
from full import RUN,PROVIDER,read,sha,atomic,check,identity
D=R/'outputs/paper_review_suite_v1';P=R/'results/paper_review_suite_v1'

def predict():
 import numpy as np,torch
 from stvg.consensus import encode,decode
 torch.set_num_threads(1);p=read(D/'pseudobox_plan.json');assert sha(Path(__file__))==p['code_sha256'];check();selection=read(D/'selection.json');assert sha(D/'selection.json')==p['selection_sha256'];atomic(D/'pseudobox_worker.json',identity());files={};started=time.perf_counter()
 from stvg.runtime import guard
 sys.addaudithook(guard)
 def loss_giou(a,b):
  a=torch.cat((a[:,:2]-a[:,2:]/2,a[:,:2]+a[:,2:]/2),1);b=torch.cat((b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2),1);inter=(torch.minimum(a[:,2:],b[:,2:])-torch.maximum(a[:,:2],b[:,:2])).clamp(min=0).prod(1);union=(a[:,2:]-a[:,:2]).prod(1)+(b[:,2:]-b[:,:2]).prod(1)-inter;cover=(torch.maximum(a[:,2:],b[:,2:])-torch.minimum(a[:,:2],b[:,:2])).clamp(min=0).prod(1);return 1-inter/union.clamp(min=1e-12)+(cover-union)/cover.clamp(min=1e-12)
 for ds,groups in selection.items():
  seal=read(RUN/ds/'sealed.json');files[ds]={}
  for stage,rows in groups.items():
   files[ds][stage]={}
   for i,row in enumerate(rows,1):
    key=row['id'];e=seal['files'][key];assert sha(RUN/ds/e['path'])==e['sha256'];dest=D/'pseudobox'/ds;dest.mkdir(parents=True,exist_ok=True);path=dest/(key+'.npz');assert not path.exists()
    with np.load(RUN/ds/e['path'])as z:
     original=z['Source'].copy();target=torch.tensor(z['native_boxes'],dtype=torch.float64);pos=z['positions8'].copy();nativeiv=z['Source_interval'].copy();ids=z['frame_ids'].copy()
    encoded=torch.tensor(encode(original),dtype=torch.float64);delta=torch.zeros(4,dtype=torch.float64,requires_grad=True);assert np.array_equal(decode(encoded.numpy()),original)or np.allclose(decode(encoded.numpy()),original,atol=1e-15,rtol=0)
    opt=torch.optim.Adam([delta],lr=.01,betas=(.9,.999),eps=1e-8);trace=[]
    for step in range(21):
     zz=encoded+delta;boxes=torch.cat((zz[:,:2],zz[:,2:].exp()),1);anchors=boxes[pos];loss=5*(anchors-target).abs().mean()+2*loss_giou(anchors,target).mean();trace.append(float(loss.detach()))
     if step<20:opt.zero_grad();loss.backward();assert torch.isfinite(delta.grad).all();opt.step()
    boxes=decode(encoded.numpy()+delta.detach().numpy());np.savez_compressed(path,PseudoBox=boxes,interval=nativeiv,frame_ids=ids,positions8=pos,loss_path=np.array(trace),delta=delta.detach().numpy());files[ds][stage][key]=dict(path=str(path.relative_to(D)),sha256=sha(path),parent_prediction_sha256=e['sha256']);atomic(D/'pseudobox_progress.json',dict(dataset=ds,stage=stage,completed=i,total=len(rows),GT_read=False))
 atomic(D/'pseudobox_sealed.json',dict(files=files,queries=384,GT_read=False,fingerprint=p['fingerprint'],seconds=time.perf_counter()-started,optimizer_steps=7680,extra_expert=0));print('PseudoBox384 sealed',flush=True)

def score():
 import numpy as np,torch
 from full_score import metric,cluster
 torch.set_num_threads(1);p=read(D/'pseudobox_plan.json');assert sha(Path(__file__))==p['code_sha256'];seal=read(D/'pseudobox_sealed.json');assert seal['queries']==384 and not seal['GT_read'];selection=read(D/'selection.json');checks=0;maxerr=0.
 for ds,groups in selection.items():
  j=read(RUN/'plan.json')['jobs'][ds];f=PROVIDER/j['labels'];assert sha(f)==j['labels_sha256'];truth={x['id']:x for x in map(json.loads,f.read_text().splitlines())if x['dataset']==ds};orig={x['id']:x for x in read(RUN/ds/'metrics.json')['private_rows']}
  for stage,records in groups.items():
   rows=[];decreased=0
   for r in records:
    e=seal['files'][ds][stage][r['id']];assert sha(D/e['path'])==e['sha256'];g=truth[r['id']]
    with np.load(D/e['path'])as z:v,err=metric(z['PseudoBox'],r,{int(k):v for k,v in g['boxes'].items()},g['interval'],z['interval'].tolist(),z['positions8']);decreased+=bool(z['loss_path'][-1]<z['loss_path'][0])
    maxerr=max(maxerr,err);checks+=1;vals={a:orig[r['id']]['values'][a]for a in ['Source','SharedInit','Interpolation','OursS','Full']};vals['PseudoBox']=v;rows.append(dict(id=r['id'],parent=str(g['parent']),values=vals))
   arms=list(rows[0]['values']);fields=['v','t','observed_actual_sIoU','unobserved_actual_sIoU'];report=dict(dataset=ds,stage=stage,queries=len(rows),means={a:{f:float(np.mean([x['values'][a][f]for x in rows if x['values'][a][f]is not None]))for f in fields}for a in arms},paired={a+'-'+b:cluster(rows,a,b)for a,b in [('PseudoBox','Source'),('Full','PseudoBox'),('PseudoBox','SharedInit')]},loss_decreased=decreased,scope=p['description'],fingerprint=p['fingerprint']);atomic(P/'pseudobox'/ds/stage/'report.json',report);atomic(D/'pseudobox'/ds/(stage+'_metrics.json'),dict(report,private_rows=rows))
   with(P/'pseudobox'/ds/stage/'signed.tsv').open('w')as f:
    w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['query']+[a+'_'+k for a in arms for k in fields])
    for i,x in enumerate(rows,1):w.writerow([f'Q{i:03d}']+[x['values'][a][k]for a in arms for k in fields])
 atomic(D/'pseudobox_score_receipt.json',dict(checks=checks,max_error=maxerr,seal_sha256=sha(D/'pseudobox_sealed.json'),all384_sealed_before_GT=True));print('PseudoBox scored',flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['predict','score']);args=a.parse_args();predict()if args.command=='predict'else score()
