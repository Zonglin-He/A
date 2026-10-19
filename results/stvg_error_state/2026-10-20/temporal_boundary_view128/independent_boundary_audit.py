"""No-GT independent temporal observation and heterogeneous-risk arithmetic audit."""
from pathlib import Path
import sys,hashlib,numpy as np
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_boundary_view128_v1 import D,P,PRE,FIN,DEV,HR,ST


def main():
 from methods.stvg_referdino_opd_v1.runtime import setup,decode
 from scripts.stvg_temporal_marginal_decode_util256_v1 import independent_cdf
 setup();p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==128 and bar['plan_sha256']==sha(D/'PLAN_LOCK.json')and not(D/'GT_EXPOSURE.json').exists();assert read(D/'NATIVE_CONTROL.json')['status']=='pass';raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];n=cd=checks=rgb=spaces=0;error=0.
 for r in p['records']:
  a=r['alias'];z=load(D/'predictions'/f'{a}.pt');old=load(DEV/'priors'/f'{a}.pt');st=load(ST/'predictions'/f'{a}.pt');ids=np.array(old['ids']);sp=load(FIN/'Live/predictions'/f'{a}.pt');inp=load(PRE/'inputs'/f'{a}.pt');hr=load(HR/'predictions'/f'{a}.pt');assert z['original_RGB_sha256']==hr['observation']['RGB_sha256']
  for state,q in enumerate([old['q'],st['outputs']['Live41']['q']]):
   w=np.array([sum(q[k,:])+sum(q[:,k])for k in range(65)]);w/=w.sum();edges=[float(ids[0])]+[(int(ids[k])+int(ids[k+1]))/2 for k in range(64)]+[float(ids[-1]+1)];taken=set(ids[::2]);extra=[]
   for k in range(32):
    quant=(k+.25)/32;cum=0.
    for cell in range(65):
     if cum+w[cell]>quant:break
     cum+=w[cell]
    target=edges[cell]+(quant-cum)/w[cell]*(edges[cell+1]-edges[cell]);free=[x for x in range(int(ids[0]),int(ids[-1])+1)if x not in taken];v=min(free,key=lambda x:(abs(x-target),x));taken.add(v);extra.append(v)
   assert sorted(taken)==r['schedules'][state]and extra==r['schedule_meta'][state]['adaptive_frames'];n+=1
  new=np.array(z['new_ids']);nearest=[min(range(65),key=lambda j:(abs(int(t)-int(ids[j])),j))for t in new];assert nearest==r['schedule_meta'][0]['nearest_original_indices']and np.array_equal(new,r['schedules'][0])
  if a=='RC001':
   union=sorted(set(ids)|set(new));frames,physical=decode(dict(raw[r['query_ordinal']]['input'],frame_ids=union));original=frames[np.searchsorted(union,ids)];assert hashlib.sha256(original.tobytes()).hexdigest()==z['original_RGB_sha256'];actual=frames[np.searchsorted(union,new)]
   for kind,pixels in [('Actual',actual),('Nearest',original[nearest])]:assert hashlib.sha256(pixels.tobytes()).hexdigest()==z['observations'][kind]['RGB_sha256'];rgb+=1
  for kind,obs in z['observations'].items():
   assert sorted(t for rec in obs['records']for t in rec['frame_ids'])==new.tolist()
   assert all(rec['frame_ids']==new[o::2].tolist()for o,rec in enumerate(obs['records']))
   grids=[]
   for rec,ll in zip(obs['records'],obs['logits']):
    ll=ll.numpy()[0].astype(float);ii,jj=np.triu_indices(len(ll),1);v=ll[ii,0]+ll[jj,1];v=np.exp(v-v.max());v/=v.sum();grids.append(dict(ids=rec['frame_ids'],ij=np.stack([ii,jj]),p=v))
   q=independent_cdf(grids,new.tolist());e=float(abs(q-obs['q']).max());assert e<1e-12;error=max(error,e);cd+=1
   for rec,ll,orec,all_ in zip(obs['records'],obs['logits'],old['records'],obs['aligned_logits']):
    ll=ll.numpy()[0];output=all_.numpy()[0]
    for t,row in zip(orec['frame_ids'],output):
     j=np.searchsorted(rec['frame_ids'],t)
     if j==0:value=ll[0]
     elif j==len(ll):value=ll[-1]
     else:
      a0,b0=rec['frame_ids'][j-1:j+1];alpha=(t-a0)/(b0-a0);value=((1-alpha)*ll[j-1].astype(float)+alpha*ll[j].astype(float)).astype(np.float32)
     assert abs(value-row).max()<3e-7
  ii,jj=np.triu_indices(65,1);S=ids[ii];E=ids[jj]+1
  for arm,o in z['outputs'].items():
   chosen=np.where((S==o['interval'][0])&(E==o['interval'][1]))[0];assert len(chosen)==1;pick=np.unique(np.r_[np.linspace(0,2079,19).astype(int),chosen]);pick=np.arange(2080)if a in ['RC001','RC002']else pick;utility=np.zeros(len(pick))
   for part in o['sources']:
    hid=np.array(part['ids']);a0,b0=np.triu_indices(len(hid),1);s=hid[a0];e=hid[b0]+1;prob=part['q'][a0,b0];assert abs(prob.sum()-1)<1e-10
    for k,x in enumerate(pick):
     inter=np.maximum(0,np.minimum(E[x],e)-np.maximum(S[x],s));u=inter/(E[x]-S[x]+e-s-inter);utility[k]+=part['weight']*float(sum(u*prob))
   assert utility[np.where(pick==chosen[0])[0][0]]>=utility.max()-1e-12;checks+=len(pick)
  for k,v in z['trajectories'].items():assert np.array_equal(v,inp['E']if k=='E'else sp['outputs'][f'T20_Live_{k[4:]}']);spaces+=1
 put(P/'BOUNDARY_INDEPENDENT_AUDIT.json',dict(status='pass',all128_before_GT=True,independent_schedules=n,independent_CDFs=cd,max_CDF_error=error,actual_RGB_checks=rgb,direct_heterogeneous_MBR_utilities=checks,two_full_old_action_grids_each_arm=True,spatial_arrays_exact=spaces,no_GT_read=True,new_model_calls=0));print('independent pass',n,checks,flush=True)

if __name__=='__main__':main()
