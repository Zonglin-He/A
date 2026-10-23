"""Independent post-seal official scoring of E3 support and coupled DP controls."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json,time,csv,hashlib,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R),str(R/'tools'),str(Path(__file__).parent)]
from full import RUN,PROVIDER,read,sha,atomic,identity,check
D=R/'outputs/paper_review_suite_v1';P=R/'results/paper_review_suite_v1'

def score():
 import numpy as np,torch
 from full_score import metric,cluster
 from supports import ARMS,frozen
 torch.set_num_threads(1);p=frozen();seal=read(D/'supports_sealed.json');assert seal['queries']==384 and seal['GT_read']is False and seal['fingerprint']==p['fingerprint'];assert sha(Path(__file__))==read(D/'supports_finish_plan.json')['code_sha256'];atomic(D/'supports_scorer.json',identity());started=time.perf_counter();checks=0;maxerr=0.
 for ds,groups in read(D/'selection.json').items():
  job=read(RUN/'plan.json')['jobs'][ds];f=PROVIDER/job['labels'];assert sha(f)==job['labels_sha256'];truth={r['id']:r for r in map(json.loads,f.read_text().splitlines())if r['dataset']==ds};original={r['id']:r for r in read(RUN/ds/'metrics.json')['private_rows']}
  for stage,records in groups.items():
   rows=[]
   for r in records:
    key=r['id'];e=seal['files'][ds][stage][key];assert sha(D/e['path'])==e['sha256'];g=truth[key];gt={int(k):v for k,v in g['boxes'].items()};vals={}
    with np.load(D/e['path'])as z:
     for a in ARMS:
      vals[a],err=metric(z[a],r,gt,g['interval'],z[a+'_interval'].tolist(),z['positions8']);checks+=1;maxerr=max(maxerr,err)
     for a in ['SourceROI','FullFrame','DPDP']:
      name='FixedSource_'+a;vals[name],err=metric(z['Source'],r,gt,g['interval'],z[a+'_interval'].tolist(),z['positions8']);checks+=1;maxerr=max(maxerr,err)
    for a in ['Source','OursS','SharedInit','Interpolation']:vals[a]=original[key]['values'][a]
    assert vals['DPDP']==original[key]['values']['Full'];rows.append(dict(id=key,parent=str(g['parent']),duration_group=original[key]['duration_group'],values=vals))
   arms=list(rows[0]['values']);fields=['v','t','observed_actual_sIoU','unobserved_actual_sIoU','boundary_MAE_seconds'];means={a:{f:float(np.mean([x['values'][a][f]for x in rows if x['values'][a][f]is not None]))if any(x['values'][a][f]is not None for x in rows)else None for f in fields}for a in arms};pairs=[('DPDP','Source'),('DPDP','OursS'),('DPDP','GreedyDP'),('DPGreedy','GreedyGreedy'),('DPDP','DPGreedy'),('GreedyDP','GreedyGreedy'),('FixedSource_DPDP','Source'),('FixedSource_DPDP','FixedSource_SourceROI'),('FixedSource_DPDP','FixedSource_FullFrame'),('DPDP','SourceROI'),('DPDP','FullFrame')];paired={a+'-'+b:cluster(rows,a,b)for a,b in pairs};severe={}
   for a,b in pairs:
    d=np.array([[x['values'][a][f]-x['values'][b][f]for f in ['t','v']]for x in rows])*100;severe[a+'-'+b]=dict(t=int((d[:,0]< -5).sum()),v=int((d[:,1]< -5).sum()),union=int((d.min(1)< -5).sum()))
   groupsummary={group:{a:{f:float(np.mean([x['values'][a][f]for x in rows if x['duration_group']==group]))for f in ['t','v']}for a in arms}for group in sorted({x['duration_group']for x in rows})}
   report=dict(dataset=ds,stage=stage,queries=len(rows),parents=len({x['parent']for x in rows}),means=means,paired=paired,severe=severe,time_groups=groupsummary,fingerprint=p['fingerprint'],exposure=p['outcome'],DP_main_no_op=True,screen_confirm_parent_disjoint=True,parameter_selection=False)
   pub=P/'supports'/ds/stage;atomic(pub/'report.json',report);atomic(D/'supports'/ds/(stage+'_metrics.json'),dict(report,private_rows=rows));pm={v:f'P{i+1:03d}'for i,v in enumerate(sorted({x['parent']for x in rows}))}
   with(pub/'signed.tsv').open('w')as out:
    w=csv.writer(out,delimiter='\t',lineterminator='\n');w.writerow(['query','parent','duration_group']+[a+'_'+f for a in arms for f in fields])
    for i,x in enumerate(rows,1):w.writerow([f'Q{i:03d}',pm[x['parent']],x['duration_group']]+[x['values'][a][f]for a in arms for f in fields])
 atomic(D/'supports_score_receipt.json',dict(status='scored',checks=checks,max_error=maxerr,all384_sealed_before_GT=True,seconds=time.perf_counter()-started,seal_sha256=sha(D/'supports_sealed.json'),public_files={str(f.relative_to(R)):sha(f)for f in(P/'supports').rglob('*')if f.is_file()}));assert not torch.cuda.is_initialized();print('Supports scored '+str(checks),flush=True)

if __name__=='__main__':
 p=read(D/'supports_finish_plan.json');assert sha(Path(__file__))==p['code_sha256'];atomic(D/'supports_finish_worker.json',identity())
 while not(D/'supports_sealed.json').exists():
  # Verify owner incarnation, and detect actual exit rather than waiting forever.
  launch=read(D/'supports_launch.json');alive=False
  try:
   st=Path(f"/proc/{launch['pid']}/stat").read_text().rsplit(')',1)[1].split();alive=int(st[19])==launch['startticks']and st[0]!='Z'
  except OSError:pass
  if not alive:raise RuntimeError('Shared-grid predictor exited before seal; preserve its log, do not auto-restart')
  time.sleep(10)
 score()
