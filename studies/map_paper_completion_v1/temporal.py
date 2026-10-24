"""Full-target temporal controls with real final MAP-conditioned spatial DP."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from common import *
import argparse,gc,traceback,fcntl,subprocess
ARMS=['Source','SourceJointMAP','FullMAP','NoPosteriorUpdate','HardMembership','EventOnly','EventCentered','NoShrink','UniformPrior']
def prepare():
 p=freeze('temporal',dict(arms=ARMS,cohort=COUNTS,posterior='All changed posterior arms use legal jointMAP then actual interval-conditioned original spatialDP; Source controls retain Source boxes.',HardMembership='Hard weights of q0 jointMAP interval; one all-video event/background pair, common-mean shrinkage64.',EventOnly='Log positive compatibility without background denominator; no centering.',EventCentered='Predeclared EventOnly video-mean correction, no result selection.',NoShrink='Identical soft pair with kappa0.',UniformPrior='Original q0 seed membership and evidence, uniform positive-support final prior.',fixed='Same8, q0, original fulltime auxiliary pooled features, beta1/context1.5/one round; no optimizer/newSource/expert/GPU.',exposure='All fullsets historically exposed; no parameter/configuration selection; all signed/tails/CI/cost.'),[Path(__file__),S/'common.py',R/'studies/map_full_v1/map_core.py',R/'stvg/prototype.py',R/'stvg/spatial.py'])
 atomic(D/'temporal_state.json',dict(status='prepared',fingerprint=p['fingerprint']));print(p['fingerprint'])
def derive(c):
 import numpy as np,torch
 from stvg.prototype import one_step,membership,evidence_from_membership,inclusive_scores,update,unit
 from stvg.spatial import SpatialGeometry,evidence_dp
 from map_core import map_interval
 from stvg.readout import prediction
 ids=np.asarray(c['frame_ids']);q0=np.asarray(c['native_prior']);f=np.asarray(c['content_features']);g=SpatialGeometry.from_cached_geometry(c['geometry']);q1,st=one_step(f,q0);qns,_=one_step(f,q0,kappa=0);iv0=map_interval(ids,q0);hard=((ids>=iv0[0])&(ids<iv0[1])).astype(float);lh,_=evidence_from_membership(f,hard,64)
 x=unit(f-np.where(np.ptp(f,axis=0)==0,f[0],f.mean(0)));le=np.log(np.maximum((1+x@st['prototypes']['event'])/2,1e-12));u=(q0>0).astype(float);u/=u.sum()
 posts=dict(FullMAP=q1,NoPosteriorUpdate=q0,HardMembership=update(q0,inclusive_scores(lh)),EventOnly=update(q0,inclusive_scores(le)),EventCentered=update(q0,inclusive_scores(le-le.mean())),NoShrink=qns,UniformPrior=update(u,st['S']))
 native=prediction(c['native_logits'],torch.as_tensor(c['Source']),c['records'],ids)['physical_interval'];a=dict(Source=np.asarray(c['Source']).copy(),Source_interval=np.asarray(native),SourceJointMAP=np.asarray(c['Source']).copy(),SourceJointMAP_interval=np.asarray(iv0),positions8=np.asarray(c['positions8']),frame_ids=ids)
 for name,q in posts.items():
  iv=map_interval(ids,q);a[name]=evidence_dp(g.for_interval(iv))['final'].numpy();a[name+'_interval']=np.asarray(iv)
 return a

def preflight():
 import torch,numpy as np
 from stvg.runtime import guard
 sys.addaudithook(guard);torch.set_num_threads(1);p=validate('temporal');proof=[]
 for ds in COUNTS:
  for row in records(ds)[:2]:
   c,parent=load_parent(ds,row);a=derive(c);b=derive(c);assert all(np.array_equal(a[k],b[k])for k in a);mp,h=current_MAP(ds,row['id'])
   with np.load(mp)as z:assert np.array_equal(a['FullMAP'],z['FullMAP'])and np.array_equal(a['FullMAP_interval'],z['FullMAP_interval'])and np.array_equal(a['Source'],z['Source'])and np.array_equal(a['Source_interval'],z['Source_interval'])
   proof.append(dict(dataset=ds,parent=parent,MAP_prediction_sha256=h,full_and_Source_bitwise=True,all_arrays_repeat_exact=True))
 assert not torch.cuda.is_initialized();atomic(D/'temporal_preflight_receipt.json',dict(status='passed',cases=4,proof=proof,GT_read=False,new_Source=0,new_expert=0,new_GPU=0,fingerprint=p['fingerprint']));atomic(P/'temporal_preflight_receipt.json',dict(status='passed',cases=4,all_arrays_repeat_exact=True,FullMAP_and_nativeSource_bitwise=True,new_Source=0,new_expert=0,new_GPU=0,GT_read=False,fingerprint=p['fingerprint']))
def predict(ds):
 import torch,numpy as np
 from stvg.runtime import guard
 sys.addaudithook(guard);torch.set_num_threads(1);p=validate('temporal');assert read(D/'temporal_preflight_receipt.json')['status']=='passed';dest=D/'temporal'/ds;dest.mkdir(parents=True,exist_ok=True);lock=(dest/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);who=identity();atomic(D/f'temporal_{ds}_worker.json',who);files={};start=time.perf_counter()
 for i,row in enumerate(records(ds),1):
  assert time.perf_counter()-start<259200;out=dest/'predictions'/(row['id']+'.npz');rp=dest/'receipts'/(row['id']+'.json');assert not out.exists()and not rp.exists(),'No blind overwrite/retry';tick=time.perf_counter();c,parent=load_parent(ds,row);a=derive(c);mp,h=current_MAP(ds,row['id'])
  with np.load(mp)as z:assert np.array_equal(a['FullMAP'],z['FullMAP'])and np.array_equal(a['FullMAP_interval'],z['FullMAP_interval'])
  out.parent.mkdir(parents=True,exist_ok=True)
  with out.with_suffix('.tmp').open('wb')as f:np.savez_compressed(f,**a)
  os.replace(out.with_suffix('.tmp'),out)
  with np.load(out)as z:assert all(np.array_equal(a[k],z[k])for k in a)
  e=dict(path=str(out.relative_to(D)),sha256=sha(out),parent=parent,current_MAP_prediction_sha256=h,worker=who,query_CPU_seconds=time.perf_counter()-tick,GT_read=False,fingerprint=p['fingerprint']);atomic(rp,e);files[row['id']]=dict(e,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp));atomic(D/f'temporal_{ds}_progress.json',dict(completed=i,total=COUNTS[ds],worker=who,seconds=time.perf_counter()-start,GT_read=False));print(ds,i,flush=True);del c,a;gc.collect()
 assert not torch.cuda.is_initialized();atomic(D/f'temporal_{ds}_sealed.json',dict(queries=len(files),files=files,worker=who,seconds=time.perf_counter()-start,GT_read=False,fingerprint=p['fingerprint']))
def pairs():return [(a,'FullMAP')for a in ARMS if a not in ['Source','SourceJointMAP','FullMAP']]+[('FullMAP','Source'),('FullMAP','SourceJointMAP')]
def score(ds):
 import torch,numpy as np
 from full_score import metric
 torch.set_num_threads(1);p=validate('temporal');sp=D/f'temporal_{ds}_sealed.json';seal=read(sp);assert seal['queries']==COUNTS[ds]and not seal['GT_read'];truth=labels(ds);rr=[];errmax=0.;checks=0
 for row in records(ds):
  e=seal['files'][row['id']];assert sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256'];g=truth[row['id']];vals={}
  with np.load(D/e['path'])as z:
   r={**row,'metadata':{**row['metadata'],'frame_ids':z['frame_ids'].tolist()}}
   for a in ARMS:vals[a],err=metric(z[a],r,{int(k):v for k,v in g['boxes'].items()},g['interval'],z[a+'_interval'].tolist(),z['positions8']);assert err<2e-10;errmax=max(errmax,err);checks+=1
  rr.append(dict(id=row['id'],parent=str(g['parent']),duration_group='short_le5s'if(g['interval'][1]-g['interval'][0])/row['metadata']['fps']<=5 else'long_gt5s',values=vals))
 folder=P/'temporal'/ds;res=parent_report(rr,ARMS,pairs(),folder);signed_tails(rr,pairs(),folder);atomic(D/'temporal'/ds/'metrics.json',dict(private_rows=rr,report=res));rec=dict(status='scored',queries=len(rr),checks=checks,max_error=errmax,seal_sha256=sha(sp),whole_domain_sealed_before_GT=True,fingerprint=p['fingerprint']);atomic(D/f'temporal_{ds}_score_receipt.json',rec);atomic(folder/'score_receipt.json',rec)
def audit(ds):
 import torch,numpy as np
 torch.set_num_threads(1);p=validate('temporal');seal=read(D/f'temporal_{ds}_sealed.json');score=read(D/f'temporal_{ds}_score_receipt.json');assert score['seal_sha256']==sha(D/f'temporal_{ds}_sealed.json');segments={};checks=0
 for row in records(ds):
  e=seal['files'][row['id']];assert sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256'];c,parent=load_parent(ds,row);assert parent==e['parent'];a=derive(c)
  with np.load(D/e['path'])as z:assert all(np.array_equal(a[k],z[k])for k in a)
  checks+=5;wk=f"{e['worker']['pid']}:{e['worker']['startticks']}";segments[wk]=segments.get(wk,0.)+e['query_CPU_seconds']
 rr=read(D/'temporal'/ds/'metrics.json')['private_rows'];report=read(P/'temporal'/ds/'report.json')
 for arm in ARMS:
  for f,v in report['means'][arm].items():
   x=[r['values'][arm][f]for r in rr if r['values'][arm][f]is not None];assert(v is None and not x)or abs(v-float(np.mean(x)))<1e-12
 out=dict(status='verified_whole_domain',queries=COUNTS[ds],SHA_checks=checks,all_arrays_cold_rederived_exact=True,all_means_recomputed=True,worker_query_CPU_seconds=segments,score_sha256=sha(D/f'temporal_{ds}_score_receipt.json'),new_Source=0,new_expert=0,new_GPU=0,new_GT_scoring=0,scope='Cache-only research wall, inherited acquisition costs nonzero; no deployment speed claim.');atomic(D/f'temporal_{ds}_reporting_receipt.json',out);atomic(P/'temporal'/ds/'reporting_receipt.json',out)
def domain(ds):
 validate('temporal');who=identity()
 try:
  for stage,suffix in [('predict','sealed'),('score','score_receipt'),('audit','reporting_receipt')]:
   assert not(D/f'temporal_{ds}_{suffix}.json').exists(),'New domain must not be blindly relaunched';atomic(D/f'temporal_{ds}_state.json',dict(status=stage,worker=who));log=(D/f'temporal_{ds}_{stage}.log').open('ab');c=subprocess.Popen([sys.executable,'-B',__file__,stage,'--dataset',ds],stdout=log,stderr=subprocess.STDOUT);atomic(D/f'temporal_{ds}_{stage}_child.json',identity(c.pid));rc=c.wait();log.close();atomic(D/f'temporal_{ds}_{stage}_exit.json',dict(code=rc));assert rc==0,'Preserve failure, no retry'
  atomic(D/f'temporal_{ds}_state.json',dict(status='complete_scored_audited',worker=who))
 except BaseException:atomic(D/f'temporal_{ds}_state.json',dict(status='failed_preserved',worker=who,error=traceback.format_exc()));raise
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','preflight','predict','score','audit','domain']);a.add_argument('--dataset',choices=list(COUNTS));z=a.parse_args();dict(prepare=prepare,preflight=preflight,predict=lambda:predict(z.dataset),score=lambda:score(z.dataset),audit=lambda:audit(z.dataset),domain=lambda:domain(z.dataset))[z.command]()
