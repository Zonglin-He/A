"""Read-only recheck of original center/log-size interpolation with E tails."""
import os,sys,json,time,hashlib,csv,argparse
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R/'studies/reviewer_key_controls_v1'),str(R),str(R/'tools')]
from common_v1 import selected,read,sha,atomic,digest,identity,deny_gpu_media
import numpy as np,torch
D=R/'outputs/old_interpolation_verification_v1';P=R/'results/old_interpolation_verification_v1';OLD=R/'outputs/review_strengthening_v1';FRESH=R/'outputs/shared_time_spatial_recalc_v2';DS='vidstg'
def derive(row,oseal,nseal):
 key=row['id'];oe=oseal['files'][key];ne=nseal['files'][key];assert sha(OLD/oe['path'])==oe['sha256']and sha(FRESH/ne['path'])==ne['sha256']
 with np.load(OLD/oe['path'])as z,np.load(FRESH/ne['path'])as n:
  ids=z['frame_ids'].copy();pos=z['positions8'].copy();iv=z['Ours_interval'].copy();full=z['Ours'].copy();stored=z['ExpertInterpolation'].copy();E=n['SharedCorrection'].copy()
  assert np.array_equal(ids,n['frame_ids'])and np.array_equal(pos,n['positions8'])and np.array_equal(iv,z['ExpertInterpolation_interval'])and np.array_equal(iv,n['shared_interval'])and np.array_equal(full,n['Full'])
  # Recover the original raw8 center-size coordinates from their independent
  # raw pixel-corner representation; only floating roundoff is tolerated.
  raw=n['raw_expert_xyxy']/np.array([row['metadata']['width'],row['metadata']['height']]*2);boxes=np.column_stack([(raw[:,0]+raw[:,2])/2,(raw[:,1]+raw[:,3])/2,raw[:,2]-raw[:,0],raw[:,3]-raw[:,1]])
  result=E.copy();inside=(ids>=ids[pos[0]])&(ids<=ids[pos[-1]]);times=ids[pos]
  for k in range(2):result[inside,k]=np.interp(ids[inside],times,boxes[:,k])
  for k in range(2,4):result[inside,k]=np.exp(np.interp(ids[inside],times,np.log(boxes[:,k])))
  err=float(np.max(abs(result-stored)));assert err<2e-14,(key,err)
  assert np.array_equal(result[~inside],E[~inside])and np.array_equal(stored[~inside],E[~inside])
 return dict(ids=ids,positions=pos,interval=iv,Full=full,Interpolation=result),err
def prepare():
 assert not(D/'plan.json').exists();oseal=read(OLD/'matched_cpu'/DS/'sealed.json');nseal=read(FRESH/DS/'sealed.json');rows=selected.records(DS);assert len(rows)==10303 and set(oseal['files'])==set(nseal['files'])=={r['id']for r in rows};cases=[]
 for j in [0,1,len(rows)//2,len(rows)-1]:
  a,e=derive(rows[j],oseal,nseal);b,f=derive(rows[j],oseal,nseal);assert all(np.array_equal(a[k],b[k])for k in a);cases.append(dict(row_index=j,all_array_repeat_exact=True,old_Interpolation_reconstruction_max_error=e))
 job=read(selected.RUN/'plan.json')['jobs'][DS];files=[Path(__file__),R/'stvg/paper_panel.py',R/'stvg/metrics.py',R/'stvg/metrics_base.py',OLD/'matched_cpu'/DS/'sealed.json',OLD/'matched_cpu'/DS/'metrics.json',FRESH/DS/'sealed.json',FRESH/DS/'audit_receipt.json',selected.PROVIDER/job['labels'],selected.PROVIDER/job['manifest']]
 p=dict(schema='read_only_old_Interpolation_Vid_full_verification_v1',queries=10303,definition='Original raw8 center linear/log-width-height interpolation; outside raw8 span exact current shared E. Same original Full interval and expert8. Not pure corner-linear/nearest-expert tails.',authorization='User challenges negative Full-minus-original-Interpolation; independently rederive arrays and rescore unchanged complete sealed cohort, no outcome guarantee.',GT_phase='Original prediction lists sealed long before this check. Separate official scoring and independent normalized-coordinate manual scoring; no prediction/model/method changes.',preflight=cases,code_input_sha256={str(f.relative_to(R))if f.is_relative_to(R)else str(f):sha(f)for f in files},new_Source=0,new_expert=0,new_model=0,new_GPU=0,new_predictions=0)
 p['fingerprint']=digest(p);atomic(D/'plan.json',p);atomic(P/'protocol.json',{k:v for k,v in p.items()if k!='code_input_sha256'});print(p['fingerprint'])
def validate():
 p=read(D/'plan.json');assert digest({k:v for k,v in p.items()if k!='fingerprint'})==p['fingerprint']
 for f,h in p['code_input_sha256'].items():assert sha(R/f)==h,f
 return p
def manual(boxes,ids,iv,m,g):
 # Independent normalized pixel-corner interpolation and scalar IoU expression.
 from bisect import bisect_right
 truth={int(k):np.array(v)/[m['width'],m['height'],m['width'],m['height']]for k,v in g['boxes'].items()};corners=np.column_stack([boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2]);total=0.;a,b=map(int,iv);c,d=map(int,g['interval']);support=0
 for f,target in truth.items():
  if not(max(a,c)<=f<min(b,d)):continue
  support+=1
  if f<ids[0]or f>ids[-1]:continue
  j=min(max(bisect_right(ids,f)-1,0),len(ids)-2);w=(f-ids[j])/float(ids[j+1]-ids[j]);p=corners[j]+w*(corners[j+1]-corners[j]);iw=max(0.,min(p[2],target[2])-max(p[0],target[0]));ih=max(0.,min(p[3],target[3])-max(p[1],target[1]));inter=iw*ih;den=max(p[2]-p[0],0)*max(p[3]-p[1],0)+max(target[2]-target[0],0)*max(target[3]-target[1],0)-inter;total+=inter/den if den>0 else 0.
 overlap=max(0,min(b,d)-max(a,c));return dict(v=total/max(max(b,d)-min(a,c),1),t=overlap/(b-a+d-c-overlap))
def run(stage):
 plan=validate();torch.set_num_threads(1);sys.addaudithook(deny_gpu_media);start=time.perf_counter();assert not(D/(stage+'_receipt.json')).exists();rows=selected.records(DS);truth=selected.labels(DS);oldrows={r['id']:r for r in read(OLD/'matched_cpu'/DS/'metrics.json')['private_rows']};oseal=read(OLD/'matched_cpu'/DS/'sealed.json');nseal=read(FRESH/DS/'sealed.json');out=[];maxerr=0.;arrayerr=0.;checks=0
 from stvg.metrics import official
 for i,row in enumerate(rows,1):
  z,err=derive(row,oseal,nseal);arrayerr=max(arrayerr,err);g=truth[row['id']];values={}
  for arm,ref in [('Full','Ours'),('Interpolation','ExpertInterpolation')]:
   if stage=='official':v=official(z[arm],dict(input=row['metadata'],frame_ids=z['ids'].tolist()),{int(k):v for k,v in g['boxes'].items()},g['interval'],z['interval'],DS)
   else:v=manual(z[arm],z['ids'],z['interval'],row['metadata'],g)
   values[arm]={k:v[k]for k in ['t','v']};e=max(abs(v[k]-oldrows[row['id']]['values'][ref][k])for k in ['t','v']);assert e<1e-11,(row['id'],arm,e);maxerr=max(maxerr,e);checks+=1
  assert values['Full']['t']==values['Interpolation']['t'];out.append(dict(id=row['id'],parent=str(g['parent']),values=values))
  if i%500==0:atomic(D/(stage+'_progress.json'),dict(completed=i,total=10303,worker=identity()))
 atomic(D/(stage+'_rows.json'),dict(private_rows=out));means={a:{k:float(np.mean([r['values'][a][k]for r in out]))for k in ['t','v']}for a in ['Full','Interpolation']};rec=dict(status='verified_original_Interpolation_complete_Vid_'+stage,queries=len(out),checks=checks,max_old_metric_error=maxerr,max_array_reconstruction_roundoff=arrayerr,means=means,delta_v_pp=100*(means['Full']['v']-means['Interpolation']['v']),all_perquery_times_and_Full_reference_exact=True,rows_sha256=sha(D/(stage+'_rows.json')),worker=identity(),elapsed_seconds=time.perf_counter()-start,new_predictions=0,new_Source=0,new_expert=0,new_model=0,new_GPU=0,plan_fingerprint=plan['fingerprint'])
 if stage=='manual':
  orig=read(D/'official_rows.json')['private_rows'];assert len(orig)==len(out)
  for a,b in zip(orig,out):assert a['id']==b['id']and max(abs(a['values'][x][k]-b['values'][x][k])for x in ['Full','Interpolation']for k in ['t','v'])<1e-11
 atomic(D/(stage+'_receipt.json'),rec);atomic(P/(stage+'_verification.json'),rec);print(json.dumps(rec),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','official','manual']);x=p.parse_args();prepare()if x.stage=='prepare'else run(x.stage)
