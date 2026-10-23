"""Predeclared full-cache temporal controls. Prediction cannot open labels."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json,time,hashlib,argparse,fcntl,csv,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R),str(R/'tools')]
from full import RUN,PROVIDER,read,sha,atomic,identity,check
D=R/'outputs/paper_review_suite_v1';P=R/'results/paper_review_suite_v1'
ARMS=['P0Native','P0MBR','P0MAP','Soft','P1MAP','Hard','EventOnly','EventCentered','NoShrink','UniformPrior','Kappa16','Kappa256','BetaHalf','BetaDouble']

def freeze():
 p=read(D/'cached_plan.json');assert p['fingerprint']==hashlib.sha256(json.dumps({k:v for k,v in p.items()if k!='fingerprint'},sort_keys=True).encode()).hexdigest()
 assert sha(Path(__file__))==p['code_sha256'];check()
 for ds,h in p['parent_seals'].items():assert sha(RUN/ds/'sealed.json')==h
 return p

def prepare():
 assert not(D/'cached_plan.json').exists()
 p=dict(schema='paper_review_cached_v1',seed=20261024,code_sha256=sha(Path(__file__)),parent_seals={ds:sha(RUN/ds/'sealed.json')for ds in ['hc2','vidstg']},parent_fingerprint=read(RUN/'plan.json')['fingerprint'],arms=ARMS,queries={'hc2':1902,'vidstg':10303},new_models=0,new_experts=0,GT_in_prediction=False,
 spatial_control='All temporal arms evaluated on identical frozen Source boxes AND identical frozen Full boxes. Neither comparison re-decodes space. E3 main causal table uses Source boxes. These isolate temporal decisions; they are not reconfigured end-to-end pipelines.',
 definitions=dict(P0Native='FP32 per-offset native MAP endpoints then physical hull; original Source boxes. Distinct from Source published MBR.',P0MBR='Sealed exact physical-hull native prior + original MBR; exact existing Source interval.',P0MAP='Joint physical-hull prior MAP; NOT per-offset native hull.',Soft='Original ONE round kappa64 beta1 all-video soft pair + MBR; exact Full temporal output.',P1MAP='Same Soft posterior, joint physical MAP.',Hard='ONE pair from hard membership of P0MBR interval; kappa64; native final prior.',EventOnly='Log positive compatibility alone; no background denominator. Raw length bias retained.',EventCentered='EventOnly minus video mean, predeclared length-bias control; never promote by outcome.',NoShrink='Same Soft with kappa0.',UniformPrior='Same second-stage Soft evidence, uniform on original positive support as final prior; native seed remains.',Kappa16='Sensitivity ONLY kappa16',Kappa256='Sensitivity ONLY kappa256',BetaHalf='Sensitivity ONLY beta.5',BetaDouble='Sensitivity ONLY beta2'),
 statistics='Equal query. Paired parent-cluster 10000 bootstrap seed20261024. All signed, time groups <=5sec, severe decline>5pp t/v/union once, offline frame AUROC/evidence distribution and native coverage. No setting selection or production promotion; both full sets previously examined.',
 stages='All12205 prediction records seal before independent CPU scorer. Domain jobs parallel CPU only. New support features and first-DP controls in independent collector stage.')
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();atomic(D/'cached_plan.json',p);atomic(P/'cached_protocol.json',p)
 target=D/'source_snapshot/cached.py';target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(__file__).read_bytes());atomic(D/'cached_state.json',dict(status='prepared',fingerprint=p['fingerprint']))
 print(p['fingerprint'],flush=True)

def predict(ds):
 import numpy as np,torch
 from stvg.prototype import one_step,membership,evidence_from_membership,inclusive_scores,update
 from stvg.temporal import final_mbr
 from stvg.readout import native_view_indices
 from threadpoolctl import threadpool_limits
 torch.set_num_threads(1);threadpool_limits(1);p=freeze();dest=D/'cached'/ds;dest.mkdir(parents=True,exist_ok=True)
 lock=(dest/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);atomic(dest/'worker.json',identity())
 labels={(PROVIDER/j['labels']).resolve()for j in read(RUN/'plan.json')['jobs'].values()}
 def guard(event,args):
  if event=='open'and args and isinstance(args[0],(str,bytes,Path))and Path(os.fsdecode(args[0])).resolve()in labels:raise PermissionError('No GT before all prediction seals')
 sys.addaudithook(guard);files={};started=time.perf_counter();seal=read(RUN/ds/'sealed.json')
 for i,(key,e)in enumerate(seal['files'].items(),1):
  target=dest/(key+'.npz');receipt=dest/(key+'.json')
  if receipt.exists():
   rec=read(receipt);assert rec['fingerprint']==p['fingerprint']and sha(target)==rec['sha256'];files[key]=rec;continue
  assert not target.exists(),'Unbound output preserved; no overwrite'
  cp=RUN/ds/e['capture_path'];npz=RUN/ds/e['path'];assert sha(cp)==e['capture_sha256']and sha(npz)==e['sha256']
  c=torch.load(cp,map_location='cpu',weights_only=False);ids=np.asarray(c['frame_ids']);q0=np.asarray(c['native_prior'],np.float64);z=np.asarray(c['content_features']);assert q0.shape==(len(ids),len(ids))and abs(q0.sum()-1)<1e-10
  q1,stats=one_step(z,q0);q16,_=one_step(z,q0,kappa=16);q256,_=one_step(z,q0,kappa=256);qn,_=one_step(z,q0,kappa=0)
  p0iv=final_mbr(ids,q0)['interval'];hard=((ids>=p0iv[0])&(ids<p0iv[1])).astype(float);lh,_=evidence_from_membership(z,hard,64)
  from stvg.prototype import unit
  x=unit(z-np.where(np.ptp(z,axis=0)==0,z[0],z.mean(0)));le=np.log(np.maximum((1+x@stats['prototypes']['event'])/2,1e-12));lec=le-le.mean()
  uniform=(q0>0).astype(float);uniform/=uniform.sum()
  posts=dict(P0MBR=q0,P0MAP=q0,Soft=q1,P1MAP=q1,Hard=update(q0,inclusive_scores(lh)),EventOnly=update(q0,inclusive_scores(le)),EventCentered=update(q0,inclusive_scores(lec)),NoShrink=qn,UniformPrior=update(uniform,stats['S']),Kappa16=q16,Kappa256=q256,BetaHalf=update(q0,.5*stats['S']),BetaDouble=update(q0,2*stats['S']))
  raw=[]
  for rec,logits in zip(c['records'],c['native_logits']):
   a,b=native_view_indices(logits);raw.append([rec['frame_ids'][a],rec['frame_ids'][b]+1])
  ivs={'P0Native':[min(a for a,b in raw),max(b for a,b in raw)]}
  for arm,q in posts.items():
   if arm.endswith('MAP'):
    a,b=np.unravel_index(int(q.argmax()),q.shape);ivs[arm]=[int(ids[a]),int(ids[b])+1]
   else:ivs[arm]=p0iv if arm=='P0MBR'else final_mbr(ids,q)['interval']
  with np.load(npz,allow_pickle=False)as old:
   assert np.array_equal(ivs['P0MBR'],old['Source_interval'])and np.array_equal(ivs['Soft'],old['Full_interval'])
  np.savez_compressed(target,intervals=np.array([ivs[a]for a in ARMS],np.int64),L=stats['L'],frame_ids=ids,native_membership=membership(q0))
  rec=dict(path=str(target.relative_to(D)),sha256=sha(target),parent_capture_sha256=e['capture_sha256'],parent_prediction_sha256=e['sha256'],fingerprint=p['fingerprint'],no_op=True,GT_read=False);atomic(receipt,rec);files[key]=rec
  if i%25==0:atomic(dest/'progress.json',dict(completed=i,total=len(seal['files']),seconds=time.perf_counter()-started,**identity(),GT_read=False))
  del c
 atomic(dest/'sealed.json',dict(status='sealed_before_labels',fingerprint=p['fingerprint'],files=files,queries=len(files),GT_read=False,seconds=time.perf_counter()-started));print(ds+' sealed '+str(len(files)),flush=True)

def score():
 import numpy as np,torch
 from stvg.metrics import official,DenseTube
 from scipy.stats import rankdata
 torch.set_num_threads(1);p=freeze();seals={ds:read(D/'cached'/ds/'sealed.json')for ds in ['hc2','vidstg']};assert all(seals[ds]['queries']==p['queries'][ds]and not seals[ds]['GT_read']for ds in seals)
 atomic(D/'cached_all_sealed.json',dict(fingerprint=p['fingerprint'],seals={ds:sha(D/'cached'/ds/'sealed.json')for ds in seals},GT_read=False));atomic(D/'cached_scorer.json',identity());checks=0;maxerr=0.;started=time.perf_counter()
 for ds in seals:
  job=read(RUN/'plan.json')['jobs'][ds];label=PROVIDER/job['labels'];assert sha(label)==job['labels_sha256'];gt={x['id']:x for x in map(json.loads,label.read_text().splitlines())if x['dataset']==ds};records=list(map(json.loads,(PROVIDER/job['manifest']).read_text().splitlines()));rows=[];auc=[];coverage=[]
  parentseal=read(RUN/ds/'sealed.json');oldmetrics={r['id']:r for r in read(RUN/ds/'metrics.json')['private_rows']}
  for i,r in enumerate(records,1):
   key=r['id'];e=seals[ds]['files'][key];assert sha(D/e['path'])==e['sha256'];g=gt[key];truth={int(k):v for k,v in g['boxes'].items()};vals={};row=dict(input=r['metadata'],frame_ids=r['metadata']['frame_ids'])
   pe=parentseal['files'][key];assert sha(RUN/ds/pe['path'])==e['parent_prediction_sha256']
   with np.load(D/e['path'])as a,np.load(RUN/ds/pe['path'])as base:
    for support in ['Source','Full']:
     dense=DenseTube(base[support],row,truth,g['interval'],clip=ds=='hc2')
     for arm,iv in zip(ARMS,a['intervals']):
      v=official(base[support],row,truth,g['interval'],iv.tolist(),ds);ref=dense.score(iv);err=max(abs(v[k]-ref[k])for k in ['t','v','s']);assert err<2e-10;maxerr=max(maxerr,err);checks+=1;vals[support+'Box_'+arm]=v
    assert vals['SourceBox_P0MBR']['v']==oldmetrics[key]['values']['Source']['v']and vals['FullBox_Soft']['v']==oldmetrics[key]['values']['Full']['v']
    ids=a['frame_ids'];mask=(ids>=g['interval'][0])&(ids<g['interval'][1]);L=a['L'];n1=int(mask.sum());n0=len(mask)-n1
    A=float((rankdata(L)[mask].sum()-n1*(n1+1)/2)/(n1*n0))if n1 and n0 else None
    auc.append(dict(parent=str(g['parent']),auc=A,inside_mean=float(L[mask].mean())if n1 else None,outside_mean=float(L[~mask].mean())if n0 else None,inside=n1,outside=n0))
    native=a['intervals'][ARMS.index('P0Native')];mbr=a['intervals'][ARMS.index('P0MBR')]
    coverage.append({name:dict(recall=max(0,min(iv[1],g['interval'][1])-max(iv[0],g['interval'][0]))/(g['interval'][1]-g['interval'][0]),no_overlap=bool(iv[1]<=g['interval'][0]or iv[0]>=g['interval'][1]))for name,iv in [('Native',native),('MBR',mbr)]})
   rows.append(dict(id=key,parent=str(g['parent']),values=vals,duration_group=oldmetrics[key]['duration_group']))
   if i%50==0:atomic(D/'cached_score_progress.json',dict(dataset=ds,completed=i,total=len(records),checks=checks))
  arms=list(rows[0]['values']);parents=sorted({x['parent']for x in rows});ix={v:i for i,v in enumerate(parents)};indices=np.array([ix[x['parent']]for x in rows]);counts=np.bincount(indices,minlength=len(parents));weights=np.random.default_rng(20261024).multinomial(len(parents),[1/len(parents)]*len(parents),size=10000);den=weights@counts;means={a:{f:float(np.mean([x['values'][a][f]for x in rows]))for f in ['t','v','s']}for a in arms};paired={}
  for support in ['Source','Full']:
   reference=support+'Box_P0MBR'
   for arm in ARMS:
    name=support+'Box_'+arm;d=np.array([[x['values'][name][f]-x['values'][reference][f]for f in ['t','v']]for x in rows])*100;sums=np.zeros((len(parents),2));np.add.at(sums,indices,d);boot=(weights@sums)/den[:,None]
    paired[name+'-P0MBR']=dict(delta_pp=d.mean(0).tolist(),CI95_pp=np.quantile(boot,[.025,.975],axis=0).T.tolist(),order=['t','v'],win=(d>1e-8).sum(0).tolist(),harm=(d< -1e-8).sum(0).tolist(),tie=(abs(d)<=1e-8).sum(0).tolist(),severe_t=int((d[:,0]< -5).sum()),severe_v=int((d[:,1]< -5).sum()),severe_union=int((d.min(1)< -5).sum()))
  groups={group:{a:{f:float(np.mean([x['values'][a][f]for x in rows if x['duration_group']==group]))for f in ['t','v']}for a in arms}for group in sorted({x['duration_group']for x in rows})}
  report=dict(status='scored',dataset=ds,queries=len(rows),parents=len(parents),means=means,paired=paired,time_groups=groups,offline_event_diagnostic=dict(queries_with_both_classes=sum(x['auc']is not None for x in auc),macro_AUROC=float(np.mean([x['auc']for x in auc if x['auc']is not None])),mean_inside=float(np.mean([x['inside_mean']for x in auc if x['inside_mean']is not None])),mean_outside=float(np.mean([x['outside_mean']for x in auc if x['outside_mean']is not None]))),native_coverage={name:dict(mean_recall=float(np.mean([x[name]['recall']for x in coverage])),no_overlap=sum(x[name]['no_overlap']for x in coverage))for name in ['Native','MBR']},fingerprint=p['fingerprint'],scope=p['spatial_control'],exposure='All official full lists previously scored; sensitivity descriptive, never model selection.')
  atomic(P/'cached'/ds/'report.json',report);atomic(D/'cached'/ds/'metrics.json',dict(report,private_rows=rows,private_auc=auc,private_coverage=coverage))
  pm={v:f'P{i+1:03d}'for i,v in enumerate(parents)}
  with(P/'cached'/ds/'signed.tsv').open('w')as f:
   w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['query','parent','duration_group']+[a+'_'+v for a in arms for v in ['t','v']])
   for i,x in enumerate(rows,1):w.writerow([f'{ds[0].upper()}{i:05d}',pm[x['parent']],x['duration_group']]+[x['values'][a][v]for a in arms for v in ['t','v']])
  with(P/'cached'/ds/'severe.tsv').open('w')as f:
   w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['query','parent','arm','baseline','delta_t_pp','delta_v_pp','severe_union_count_once'])
   for i,x in enumerate(rows,1):
    for support in ['Source','Full']:
     for arm in ARMS:
      name=support+'Box_'+arm;ref=support+'Box_P0MBR';dt,dv=[100*(x['values'][name][v]-x['values'][ref][v])for v in ['t','v']]
      if min(dt,dv)< -5:w.writerow([f'{ds[0].upper()}{i:05d}',pm[x['parent']],name,ref,dt,dv,1])
 atomic(D/'cached_score_receipt.json',dict(status='scored',official_dense_checks=checks,max_error=maxerr,seconds=time.perf_counter()-started,all_12205_sealed_before_GT=True,public_files={str(f.relative_to(R)):sha(f)for f in(P/'cached').rglob('*')if f.is_file()}));print(json.dumps(dict(status='scored',checks=checks,maxerr=maxerr)),flush=True)

def controller():
 import subprocess
 p=freeze();lock=(D/'cached_controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);state=dict(status='predicting',fingerprint=p['fingerprint'],controller=identity(),children=[]);atomic(D/'cached_state.json',state)
 try:
  children=[]
  for ds in ['hc2','vidstg']:
   if not(D/'cached'/ds/'sealed.json').exists():
    log=(D/(ds+'_cached.log')).open('ab');c=subprocess.Popen([sys.executable,'-B',__file__,'predict','--dataset',ds],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();children.append(c);state['children'].append(dict(**identity(c.pid),dataset=ds));atomic(D/'cached_state.json',state)
  for c in children:assert c.wait()==0,'Prediction child failed; preserve outputs'
  state['status']='scoring';state['children']=[];atomic(D/'cached_state.json',state)
  if not(D/'cached_score_receipt.json').exists():
   log=(D/'cached_score.log').open('ab');c=subprocess.Popen([sys.executable,'-B',__file__,'score'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();state['children']=[dict(**identity(c.pid),role='scorer')];atomic(D/'cached_state.json',state);assert c.wait()==0,'CPU scoring failed'
  state.update(status='complete_scored_pending_publication',children=[]);atomic(D/'cached_state.json',state)
 except BaseException:state.update(status='failed',error=traceback.format_exc());atomic(D/'cached_state.json',state);raise

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','predict','score','run']);ap.add_argument('--dataset',choices=['hc2','vidstg']);a=ap.parse_args();{'prepare':prepare,'predict':lambda:predict(a.dataset),'score':score,'run':controller}[a.command]()
