"""Spatial x temporal branch ablation: repool the missing Time-only branch."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json,time,argparse,gc,hashlib,csv
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tools'))
from full import RUN,PROVIDER,read,sha,atomic,check,identity,binding
D=R/'outputs/coupled_ablation_v1';PUB=R/'results/coupled_ablation_v1';ARMS=('Base','Space','Time','Full')
def frozen():
 p=read(D/'plan.json');assert hashlib.sha256(json.dumps({k:v for k,v in p.items()if k!='fingerprint'},sort_keys=True).encode()).hexdigest()==p['fingerprint'];assert sha(Path(__file__))==p['code_sha256']and sha(D/'selection.json')==p['selection_sha256'];check()
 for ds,h in p['parent_seals'].items():assert sha(RUN/ds/'sealed.json')==h
 return p

def predict():
 import numpy as np,torch
 from stvg.runtime import setup,lease,load_source
 from stvg.visual_prefix import native_appearance_grid
 from stvg.decode import decode
 from stvg.spatial import SpatialGeometry,evidence_dp
 from stvg.representation import extract
 from stvg.prior import from_capture
 from stvg.prototype import one_step
 from stvg.temporal import final_mbr
 p=frozen();setup();gpu=lease();atomic(D/'predictor.json',identity());selection=read(D/'selection.json');started=time.perf_counter();files={};proofs=[]
 try:
  for ds,rows in selection.items():
   model=load_source(read(R/'configs/default.json')['datasets'][ds]);files[ds]={}
   for i,row in enumerate(rows,1):
    tick=time.perf_counter();key=row['id'];entry=read(RUN/ds/'sealed.json')['files'][key];assert sha(RUN/ds/entry['capture_path'])==entry['capture_sha256']and sha(RUN/ds/entry['path'])==entry['sha256'];cap=torch.load(RUN/ds/entry['capture_path'],map_location='cpu',weights_only=False);rec=read(RUN/ds/entry['receipt_path']);assert rec['input_binding']==binding(row,read(RUN/'plan.json')['provider_runtime_fingerprint'])
    frames,ids=decode(row['metadata'],ds);assert ids==list(cap['frame_ids']);torch.manual_seed(20260920);np.random.seed(20260920);grid=native_appearance_grid(model,frames,ids,row['metadata']);g=SpatialGeometry.from_cached_geometry(cap['geometry']);q0=from_capture(cap)
    fullaux=evidence_dp(g.for_interval([ids[0],ids[-1]+1]))['final'].numpy();features,_=extract(dict(frame_ids=ids,native_local_grid=grid),fullaux);assert np.array_equal(features,np.asarray(cap['content_features'])),'Full native-grid/pool no-op failed'
    fq,_=one_step(features,q0);fiv=final_mbr(ids,fq)['interval'];fboxes=evidence_dp(g.for_interval(fiv))['final'].numpy()
    with np.load(RUN/ds/entry['path'],allow_pickle=False)as z:
     assert np.array_equal(fiv,z['Full_interval'])and np.array_equal(fboxes,z['Full']);base=z['SharedInit'].copy();base[z['positions8']]=z['native_boxes'];assert np.array_equal(base[z['positions8']],z['native_boxes'])
     basefeatures,_=extract(dict(frame_ids=ids,native_local_grid=grid),base);tq,_=one_step(basefeatures,q0);tiv=final_mbr(ids,tq)['interval'];out=dict(Base=base,Space=z['OursS'].copy(),Time=base.copy(),Full=z['Full'].copy(),Base_interval=z['Source_interval'].copy(),Space_interval=z['Source_interval'].copy(),Time_interval=np.asarray(tiv),Full_interval=z['Full_interval'].copy(),positions8=z['positions8'].copy(),frame_ids=z['frame_ids'].copy())
    file=D/ds/(key+'.npz');file.parent.mkdir(exist_ok=True);assert not file.exists();np.savez_compressed(file,**out);files[ds][key]=dict(path=str(file.relative_to(D)),sha256=sha(file),parent_prediction_sha256=entry['sha256']);proofs.append(dict(dataset=ds,query=i,Full_grid_pool_and_output_no_op=True,base_pool_changed=not np.array_equal(basefeatures,features),prefix_forward=1,new_expert=0,seconds=time.perf_counter()-tick));atomic(D/'predict_progress.json',dict(dataset=ds,completed=i,total=len(rows),**identity(),GT_read=False));print(json.dumps(proofs[-1]),flush=True);del cap,g,grid,frames,features,basefeatures,out;gc.collect()
   model.cpu();del model;gc.collect();torch.cuda.empty_cache()
  atomic(D/'sealed.json',dict(status='sealed_before_labels',fingerprint=p['fingerprint'],files=files,queries=256,GT_read=False,new_expert=0,prefix_forwards=256,proofs=proofs,wall_seconds=time.perf_counter()-started))
 finally:gpu.close()

def score():
 os.environ['CUDA_VISIBLE_DEVICES']=''
 import numpy as np,torch
 from full_score import metric,cluster
 p=frozen();seal=read(D/'sealed.json');assert seal['queries']==256 and seal['GT_read']is False and seal['fingerprint']==p['fingerprint'];torch.set_num_threads(1);atomic(D/'scorer.json',identity());started=time.perf_counter();checks=0;maxerr=0.
 for ds,records in read(D/'selection.json').items():
  job=read(RUN/'plan.json')['jobs'][ds];f=PROVIDER/job['labels'];assert sha(f)==job['labels_sha256'];truth={x['id']:x for x in map(json.loads,f.read_text().splitlines())if x['dataset']==ds};old={x['id']:x for x in read(RUN/ds/'metrics.json')['private_rows']};baseold={x['id']:x for x in read(R/'outputs/anchor_attribution_v1'/ds/'metrics.json')['private_rows']};rows=[]
  for r in records:
   key=r['id'];e=seal['files'][ds][key];assert sha(D/e['path'])==e['sha256'];gt=truth[key];boxes={int(k):v for k,v in gt['boxes'].items()};vals={}
   with np.load(D/e['path'],allow_pickle=False)as z:
    for a in ARMS:vals[a],err=metric(z[a],r,boxes,gt['interval'],z[a+'_interval'].tolist(),z['positions8']);maxerr=max(maxerr,err);checks+=1
   assert vals['Base']==baseold[key]['values']['SharedDirect8']and vals['Space']==old[key]['values']['OursS']and vals['Full']==old[key]['values']['Full'];vals['Source']=old[key]['values']['Source'];rows.append(dict(id=key,parent=old[key]['parent'],values=vals))
  fields=['v','t','observed_actual_sIoU','unobserved_actual_sIoU'];means={a:{f:float(np.mean([x['values'][a][f]for x in rows if x['values'][a][f]is not None]))for f in fields}for a in [*ARMS,'Source']};pairs=[('Space','Base'),('Time','Base'),('Full','Space'),('Full','Time'),('Full','Base')];paired={a+'-'+b:cluster(rows,a,b)for a,b in pairs}
  interaction=[]
  for x in rows:
   value={f:x['values']['Full'][f]-x['values']['Space'][f]-x['values']['Time'][f]+x['values']['Base'][f]if all(x['values'][a][f]is not None for a in ARMS)else None for f in fields};interaction.append(dict(parent=x['parent'],values=dict(Interaction=value,Zero={f:0. if v is not None else None for f,v in value.items()})))
  report=dict(status='complete_scored',dataset=ds,queries=128,parents=len({x['parent']for x in rows}),means=means,paired=paired,interaction=cluster(interaction,'Interaction','Zero',[(f,100)for f in fields]),fingerprint=p['fingerprint'],scope='Fixed metadata-selected exposed128parents/domain; no tuning or new holdout claim. Base=SharedDirect8, Space=OursS, Time=base-track Native2D feedback with base final boxes, Full=current fixed system. Shared calibration/eight evidence held across four arms. Full native-grid/pool/posterior interval/boxes no-op on every actual prefix.')
  atomic(PUB/ds/'report.json',report);atomic(D/ds/'metrics.json',dict(report,private_rows=rows));pm={v:f'P{i+1:03d}'for i,v in enumerate(sorted({x['parent']for x in rows}))}
  with(PUB/ds/'signed.tsv').open('w')as f:
   w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['query','parent']+[a+'_'+v for a in [*ARMS,'Source']for v in fields])
   for i,x in enumerate(rows,1):w.writerow([f'{ds[0].upper()}{i:03d}',pm[x['parent']]]+[x['values'][a][v]for a in [*ARMS,'Source']for v in fields])
 atomic(D/'score_receipt.json',dict(status='scored',official_dense_checks=checks,max_error=maxerr,all_three_existing_endpoints_exact=True,seal_sha256=sha(D/'sealed.json'),score_seconds=time.perf_counter()-started,public_files={str(f.relative_to(R)):sha(f)for f in PUB.rglob('*')if f.is_file()}));print(json.dumps(dict(status='complete_scored',checks=checks,maxerr=maxerr)),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['predict','score']);args=a.parse_args();predict()if args.command=='predict'else score()
