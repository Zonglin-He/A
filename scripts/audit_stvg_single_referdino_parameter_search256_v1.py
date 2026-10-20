"""Independent NumPy audit of new optimizer rollouts, then saved GT aggregation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='2'
import sys,time,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_single_referdino_parameter_search256_common_v1 import ROOT,D,P,BASE as OLD,read,put,sha,load,verify
RUN=D;PUBLIC=P;DS=None
def bind(ds):
 global D,P,DS
 DS=ds;D=RUN/ds;P=PUBLIC/ds
from scripts.audit_stvg_spatial_strong_bidirectional256_v1 import array,softmax,giou
def pregt(phase):
 p=verify();bar=read(D/f'{phase.upper()}_FIT_BARRIER.json');beg=time.perf_counter();errors={k:0. for k in ['chart','actions','cost','weights','mean_gradient','state_increment']};fits=steps=reused=0
 for rec in [r for r in p['records']if r['dataset']==DS]:
  alias=rec['alias'];old=load(OLD/f'predictions/{DS}/{alias}.pt');cache=load(OLD/f'inputs/{DS}/{alias}.pt');first={}
  for cid,cfg in bar['configs'].items():
   ent=bar['files'][cid+'/'+alias];assert sha(ROOT/ent['path'])==ent['sha256'];z=load(ROOT/ent['path']);f=z['fit']
   assert np.array_equal(array(f['path'][0]['boxes']),old['E_Ref'])and not array(f['path'][0]['state']).any();assert np.array_equal(array(f['target']),array(cache['teacher_native']['boxes']))
   assert f['actual_updates']==40 and len(f['path'])==41
   for k in ['lr','tau','steps','scales','samples']:assert f['configuration'][k]==cfg[k]
   first[cid]=f['rounds'][0]
   if ent['historical_reuse']:reused+=1;continue
   fits+=1
   for k,rr in enumerate(f['rounds']):
    steps+=1;r=rr['rollout'];mu=array(rr['current_policy_mean']);Dd=array(r['D']);ac=array(r['actions']);w=array(r['weights']);eps=array(r['eps']);proposal=array(r['proposal_mean']);b=array(f['path'][k]['boxes'])[f['positions']];cm=np.concatenate((b[:,:2],np.log(b[:,2:])),axis=-1)
    assert np.array_equal(proposal,mu);assert np.array_equal(eps[:,:8],-eps[:,8:16])and np.array_equal(eps[:,16:24],-eps[:,24:])
    sig=np.repeat(np.array(cfg['scales']),16);std=Dd[:,None,:]*np.array(cfg['scales'])[None,:,None];out=np.concatenate((ac[...,:2],np.exp(ac[...,2:])),axis=-1);target=array(f['target'])[:,None]
    c=5*np.abs(out-target).sum(-1)+2*(1-giou(out,target));q=softmax(-c/cfg['tau']);diff=(ac[:,:,None,:]-mu[:,None,None,:])/std[:,None,:,:];comp=-.5*(diff*diff).sum(-1)-np.log(std).sum(-1)[:,None,:]-2*math.log(2*math.pi);resp=softmax(comp)
    grad=(w[:,:,None,None]*resp[:,:,:,None]*(mu[:,None,None,:]-ac[:,:,None,:])/std[:,None,:,:]**2).sum((1,2))/len(mu)
    checks=dict(chart=np.max(abs(cm-mu)),actions=np.max(abs(ac-(proposal[:,None]+Dd[:,None]*sig[None,:,None]*eps))),cost=np.max(abs(c-array(r['cost']))),weights=np.max(abs(q-w)),mean_gradient=np.max(abs(grad-array(rr['policy_mean_gradient']))),state_increment=np.max(abs(array(f['path'][k+1]['state'])-array(f['path'][k]['state'])-array(rr['state_step']))))
    for key,e in checks.items():errors[key]=max(errors[key],float(e))
    assert max(checks.values())<2e-8,(alias,cid,k,checks)
  vals=list(first.values())
  for rr in vals[1:]:assert np.array_equal(array(rr['rollout']['actions']),array(vals[0]['rollout']['actions'])), 'first-step candidate mismatch'
  if fits and fits%80==0:print('PRE_GT_AUDIT',phase,fits,steps,flush=True)
 put(P/phase/'PRE_GT_AUDIT_PUBLIC.json',dict(status='pass',new_fits=fits,new_updates_independently_audited=steps,reused_prior_audited_fits=reused,first_step_candidates_identical_across_optimizers=True,paired_random_stream=True,max_errors=errors,GT_read=False,seconds=time.perf_counter()-beg))
 print('PRE_GT_AUDIT_PASS',phase,fits,steps,flush=True)
def scored(phase):
 p=verify();r=read(P/phase/'RESULTS_PUBLIC.json');beg=time.perf_counter();checks=0;maxerr=0.;n=128
 def close(a,b):
  nonlocal checks,maxerr
  e=float(np.max(abs(np.asarray(a)-np.asarray(b))));assert e<1e-10,e;checks+=1;maxerr=max(maxerr,e)
 for row in r['rows']:
  with np.load(D/phase/'private_metric_receipts'/f"{row['alias']}.npz")as z:
   a,end=z['interval'];g,h=z['GT_span'];length=max(end,h)-min(a,g);overlap=max(0,min(end,h)-max(a,g));union=end-a+h-g-overlap
   mask=(z['dense_GT_ids']>=max(a,g))&(z['dense_GT_ids']<min(end,h));obs=np.isin(z['actual_ids'],z['observed_ids'])
   for i,arm in enumerate(z['arms'].tolist()):
    v=row['values'][arm];close(np.sum(z['dense_GT_IoU'][i,mask])/max(length,1),v['v']);close(overlap/union,v['t']);close(np.mean(z['dense_GT_IoU'][i]),v['s']);close(np.mean(z['actual_IoU'][i]),v['fixed_actual_sIoU']);close(np.mean(z['actual_IoU'][i,~obs]),v['unobserved_actual_sIoU'])
    if obs.any():close(np.mean(z['actual_IoU'][i,obs]),v['observed_actual_sIoU'])
 for arm,metrics in r['means'].items():
  for key,v in metrics.items():close(sum(x['values'][arm][key]for x in r['rows']if x['values'][arm][key]is not None)/sum(x['values'][arm][key]is not None for x in r['rows']),v)
 rng=np.random.default_rng(20261020);idx=rng.integers(0,128,(10000,128))
 for comparison,metrics in r['paired'].items():
  a,b=comparison.split('_minus_')
  # Independent gathered-resample implementation for the primary endpoint;
  # all other point estimates and signed tails are separately recomputed.
  x=np.array([100*(row['values'][a]['v']-row['values'][b]['v'])for row in r['rows']]);close(np.quantile(x[idx].mean(1),[.025,.975]),metrics['v']['CI95'])
  for key,m in metrics.items():
   if key=='tail':continue
   close(np.mean([100*(row['values'][a][key]-row['values'][b][key])for row in r['rows']]),m['mean'])
  tail=metrics['tail'];assert [int((x>1e-10).sum()),int((x< -1e-10).sum()),int((abs(x)<=1e-10).sum()),int((x< -5).sum())]==[tail[k]for k in ['positive','negative','unchanged','below_minus5']]
 put(P/phase/'ROOT_AUDIT_PUBLIC.json',dict(status='pass',independent_checks=checks,max_abs=maxerr,all128_and_signed_tails=True,primary_parent_bootstrap_independent=True,private_receipt_metric_aggregation=True,seconds=time.perf_counter()-beg));print('ROOT_AUDIT_PASS',phase,checks,maxerr,flush=True)
if __name__=='__main__':
 bind(sys.argv[2])
 if sys.argv[1]=='pre-gt':pregt(sys.argv[3])
 elif sys.argv[1]=='scored':scored(sys.argv[3])
