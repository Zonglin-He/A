"""Pre-GT audit revision: separate numerical gradient verification from exact Adam replay.
Original scorer/predictions/labels/optimizer remain pinned and unchanged.
"""
from scripts.score_stvg_temporal_strong_relative_preference_opd128_v1 import *
import scripts.score_stvg_temporal_strong_relative_preference_opd128_v1 as original

def replay(Bgrid,dd,bb,maps,support,out):
 B=Bgrid[tuple(support.T)];r=np.zeros(len(B));m1=r.copy();m2=r.copy();rng=np.random.default_rng(out['seed']);u=rng.random((40,32,2));cdf=np.cumsum(B);cdf[-1]=1.
 bank=np.stack([np.searchsorted(cdf,u[:,:,0],side='right'),np.searchsorted(cdf,u[:,:,1],side='right')],axis=-1);assert np.array_equal(bank,out['bank'])
 maximum=0.;changed=admitted=total=collisions=0;seen=set();accepted=set();pref=[];globalKL=[]
 def close(a,b):
  nonlocal maximum
  e=float(np.max(np.abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float))));maximum=max(maximum,e);assert e<2e-10,e
 for k,tr in enumerate(out['trace']):
  p=B.copy()if not r.any()else np.exp(np.log(B)+r-logsumexp(np.log(B)+r))
  pairs=bank.reshape(-1,2)if out['mode']=='Direct'else bank[k].copy()
  if out['mode']=='Live':
   cp=np.cumsum(p);cp[-1]=1.;pairs[:,0]=np.searchsorted(cp,u[k,:,0],side='right');changed+=int(not np.array_equal(pairs,bank[k]))
  assert np.array_equal(pairs,tr['pairs'])and np.array_equal(u[k],tr['uniforms'])
  fb=independent_feedback(dd,bb,maps,pairs)
  for key in fb:close(fb[key],tr['feedback'][key])
  # Raw-log arithmetic orders can differ at roundoff; the strict admitted sign must remain exact.
  assert np.array_equal(fb['w'],tr['feedback']['w'])
  v=r[pairs[:,0]]-r[pairs[:,1]];t=fb['t'];w=fb['w'];a=(expit(v)-t)*w/len(pairs)
  g=p*(r-p@r);np.add.at(g,pairs[:,0],a);np.add.at(g,pairs[:,1],-a)
  pk=float(np.mean(w*(np.logaddexp(0,v)-t*v+xlogy(t,t)+xlogy(1-t,1-t))))
  kk=float(p@r-logsumexp(np.log(B)+r)+logsumexp(np.log(B)))
  close(r,tr['eta_before']);close(p,tr['pi_before']);close(g,tr['gradient']);close(pk,tr['preference_KL']);close(kk,tr['global_KL']);close(pk+kk,tr['loss'])
  # Gradient checked independently above; replay recorded Adam using its audited exact gradient.
  actual_g=tr['gradient'];m1=.9*m1+.1*actual_g;m2=.999*m2+.001*actual_g*actual_g;r-=.1*(m1/(1-.9**(k+1)))/(np.sqrt(m2/(1-.999**(k+1)))+1e-8);r-=r.mean()
  close(m1,tr['adam_first']);close(m2,tr['adam_second']);close(r,tr['eta_after'])
  admitted+=int(w.sum());total+=len(w);collisions+=int(fb['collision'].any(1).sum());seen.update(pairs.ravel().tolist());accepted.update(pairs[w>0].ravel().tolist());pref.append(pk);globalKL.append(kk)
 close(r,out['eta']);expected=np.zeros_like(Bgrid);expected[tuple(support.T)]=B if not r.any()else np.exp(np.log(B)+r-logsumexp(np.log(B)+r));close(expected,out['q'])
 return dict(steps=len(out['trace']),maxerror=maximum,Live_batches_different_from_initial=changed,admitted=admitted,pairs_evaluated=total,
  admission_fraction=admitted/total,unique_sampled_intervals=len(seen),unique_admitted_intervals=len(accepted),native_collision_pairs=collisions,
  initial_preference_KL=pref[0],final_preupdate_preference_KL=pref[-1],initial_global_KL=globalKL[0],final_preupdate_global_KL=globalKL[-1],
  eta_norm=float(np.linalg.norm(r)),posterior_L1=float(abs(out['q']-Bgrid).sum()))


if __name__=='__main__':
 def timeout(signum,frame):raise TimeoutError('finite900s revised preaudit and score')
 signal.signal(signal.SIGALRM,timeout);signal.signal(signal.SIGTERM,timeout);signal.alarm(900)
 lock=(D/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  supplement=read(D/'PRE_GT_AUDIT_REVISION001_LOCK.json')
  assert all(sha(R/f)==h for f,h in supplement['pins'].items())
  original.replay=replay;original.main()
 except BaseException as e:put(D/f'REVISED_AUDIT_FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e)));raise
 finally:lock.close()
