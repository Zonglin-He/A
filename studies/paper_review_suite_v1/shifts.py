"""Fixed controlled RGB shifts: recompute Source AND same main on each view."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json,time,hashlib,argparse,subprocess,gc,csv,io
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R),str(R/'tools')]
from full import RUN,PROVIDER,read,sha,atomic,check,identity
D=R/'outputs/paper_review_suite_v1';P=R/'results/paper_review_suite_v1';TRANSFORMS=['Blur','JPEG','Dim']

def transform(frames,name):
 import numpy as np
 if name=='Dim':return np.ascontiguousarray((frames.astype(np.float32)*.5).clip(0,255).astype(np.uint8))
 if name=='Blur':
  import cv2
  return np.stack([cv2.GaussianBlur(x,(5,5),1.5,borderType=cv2.BORDER_REFLECT_101)for x in frames])
 from PIL import Image
 output=[]
 for frame in frames:
  stream=io.BytesIO();Image.fromarray(frame).save(stream,format='JPEG',quality=30,subsampling=2);stream.seek(0)
  with Image.open(stream)as im:output.append(np.array(im.convert('RGB')))
 return np.stack(output)

def frozen():
 p=read(D/'shifts_plan.json');assert sha(Path(__file__))==p['code_sha256']and sha(D/'selection.json')==p['selection_sha256'];check();return p

def predict(stage):
 import torch,numpy as np
 from stvg.runtime import setup,lease,load_source
 from stvg.teacher_bridge import ResidentTeacher
 from stvg.predictor import Predictor
 from stvg.decode import decode
 from stvg.query import QuerySubjectParser
 p=frozen();setup();gpu=lease();selection=read(D/'selection.json');files={};atomic(D/('shifts_'+stage+'_worker.json'),identity());started=time.perf_counter();seen=0
 try:
  for ds,groups in selection.items():
   files[ds]={};model=Predictor(ds);model.source=load_source(model.config);model.source.cpu();teacher=ResidentTeacher(cache_dir=D/'shift_teacher_cache',fingerprint=p['runtime_fingerprint']);model.teacher_service=teacher;parser=QuerySubjectParser(R/'checkpoints/stanza')
   try:
    for row in groups[stage]:
     key=row['id'];frames,ids=decode(row['metadata'],ds);files[ds][key]={};originalsha=hashlib.sha256(frames.tobytes()).hexdigest();parses=dict(subject=parser(row['metadata']['caption'])['subject'])
     for name in TRANSFORMS:
      dest=D/'shifts'/stage/ds;dest.mkdir(parents=True,exist_ok=True);out=dest/(key+'.'+name+'.npz');assert not out.exists();assert time.perf_counter()-started<21600;tick=time.perf_counter();view=transform(frames,name);assert view.shape==frames.shape and view.dtype==np.uint8;rgbsha=hashlib.sha256(view.tobytes()).hexdigest();assert rgbsha!=originalsha;transform_s=time.perf_counter()-tick;torch.manual_seed(20260920);np.random.seed(20260920);cap=model.capture(view,ids,row['metadata'],parses=parses);result=model.finish(cap);assert not cap['GT_read']and not result['GT_read'];np.savez_compressed(out,Source=cap['Source'],Full=result['boxes'],Source_interval=np.asarray(cap['intervals']['Native']),Full_interval=np.asarray(result['interval']),frame_ids=np.asarray(ids),positions8=np.asarray(cap['planned_positions']));files[ds][key][name]=dict(path=str(out.relative_to(D)),sha256=sha(out),transformed_RGB_sha256=rgbsha,original_RGB_sha256=originalsha,transform_seconds=transform_s,pipeline_seconds=time.perf_counter()-tick,capture_cost=cap['cost'],finish_cost=result['cost'],expert_forwards=cap['cost']['ReferDINO_query_forwards'],GT_read=False);del cap,result,view;gc.collect();torch.cuda.empty_cache()
     seen+=1;atomic(D/'shifts_progress.json',dict(stage=stage,dataset=ds,queries=seen,total=128 if stage=='screen'else 256,**identity(),GT_read=False));print(json.dumps(dict(stage=stage,dataset=ds,queries=seen)),flush=True)
   finally:teacher.close();del model;gc.collect();torch.cuda.empty_cache()
  atomic(D/('shifts_'+stage+'_sealed.json'),dict(files=files,queries=seen,GT_read=False,fingerprint=p['fingerprint'],seconds=time.perf_counter()-started))
 finally:gpu.close()

def score(stage):
 os.environ['CUDA_VISIBLE_DEVICES']=''
 import torch,numpy as np
 from full_score import metric,cluster
 torch.set_num_threads(1);p=frozen();seal=read(D/('shifts_'+stage+'_sealed.json'));assert not seal['GT_read'];checks=0;maxerr=0.
 for ds,groups in read(D/'selection.json').items():
  j=read(RUN/'plan.json')['jobs'][ds];f=PROVIDER/j['labels'];assert sha(f)==j['labels_sha256'];gt={r['id']:r for r in map(json.loads,f.read_text().splitlines())if r['dataset']==ds};old={r['id']:r for r in read(RUN/ds/'metrics.json')['private_rows']}
  for name in TRANSFORMS:
   rows=[];cost=[]
   for r in groups[stage]:
    key=r['id'];e=seal['files'][ds][key][name];assert sha(D/e['path'])==e['sha256'];g=gt[key];vals={'CleanSource':old[key]['values']['Source'],'CleanFull':old[key]['values']['Full']}
    with np.load(D/e['path'])as z:
     for a in ['Source','Full']:vals[a],err=metric(z[a],r,{int(k):v for k,v in g['boxes'].items()},g['interval'],z[a+'_interval'].tolist(),z['positions8']);checks+=1;maxerr=max(maxerr,err)
    rows.append(dict(parent=str(g['parent']),values=vals));cost.append(e)
   report=dict(dataset=ds,stage=stage,shift=name,queries=len(rows),means={a:{f:float(np.mean([x['values'][a][f]for x in rows]))for f in ['v','t']}for a in ['Source','Full','CleanSource','CleanFull']},paired={a+'-'+b:cluster(rows,a,b)for a,b in [('Full','Source'),('Source','CleanSource'),('Full','CleanFull')]},mean_transform_seconds=float(np.mean([x['transform_seconds']for x in cost])),mean_pipeline_seconds=float(np.mean([x['pipeline_seconds']for x in cost])),new_expert=sum(x['expert_forwards']for x in cost),fingerprint=p['fingerprint'],scope='Controlled synthetic shifts, same exact Source/Full input perpair, all parameters frozen. Natural-domain evidence not replaced; no shift-specific tuning.');pub=P/'shifts'/ds/stage/name;atomic(pub/'report.json',report);atomic(D/'shifts'/stage/ds/(name+'_metrics.json'),dict(report,private_rows=rows))
   with(pub/'signed.tsv').open('w')as f:
    w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['query','Source_t','Source_v','Full_t','Full_v']);w.writerows([[f'Q{i:03d}',*[x['values'][a][k]for a in ['Source','Full']for k in ['t','v']]]for i,x in enumerate(rows,1)])
 atomic(D/('shifts_'+stage+'_score_receipt.json'),dict(status='scored',checks=checks,max_error=maxerr,seal_sha256=sha(D/('shifts_'+stage+'_sealed.json'))));print('Shifts '+stage+' scored',flush=True)

def controller():
 p=frozen();atomic(D/'shifts_controller.json',identity());atomic(D/'shifts_state.json',dict(status='queued_after_budget',controller=identity()))
 while not(D/'budget_confirm_score_receipt.json').exists():
  who=read(D/'budget_launch.json')
  try:st=Path(f"/proc/{who['pid']}/stat").read_text().rsplit(')',1)[1].split();assert int(st[19])==who['startticks']and st[0]!='Z'
  except(OSError,AssertionError):atomic(D/'shifts_state.json',dict(status='dependency_failed_no_start'));raise RuntimeError('Budget dependency failed; root must inspect before shifts')
  time.sleep(10)
 for stage in ['screen','confirm']:
  for command in ['predict','score']:
   atomic(D/'shifts_state.json',dict(status=stage+'_'+command,controller=identity()));c=subprocess.Popen([sys.executable,'-B',__file__,command,'--stage',stage]);atomic(D/('shifts_'+stage+'_'+command+'_child.json'),identity(c.pid));assert c.wait()==0,'Shifts failed; no automatic retry or parameter change'
 atomic(D/'shifts_state.json',dict(status='complete_scored_pending_publication',fingerprint=p['fingerprint']))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['run','predict','score']);a.add_argument('--stage',choices=['screen','confirm']);args=a.parse_args();{'run':controller,'predict':lambda:predict(args.stage),'score':lambda:score(args.stage)}[args.command]()
