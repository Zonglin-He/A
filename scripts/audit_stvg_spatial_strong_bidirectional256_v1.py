"""Independent NumPy rollout-density/feedback/mean-gradient audit before GT."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,math
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import numpy as np
from scripts.run_stvg_spatial_strong_bidirectional256_v1 import D,P,PRE,FIN,read,put,load,verify,sha

def array(x):return x.detach().cpu().numpy()if hasattr(x,'detach')else np.asarray(x)
def softmax(x,axis=-1):
 z=np.exp(x-np.max(x,axis=axis,keepdims=True));return z/z.sum(axis=axis,keepdims=True)
def giou(a,b):
 a=np.concatenate((a[...,:2]-a[...,2:]/2,a[...,:2]+a[...,2:]/2),-1);b=np.concatenate((b[...,:2]-b[...,2:]/2,b[...,:2]+b[...,2:]/2),-1)
 intersection=np.maximum(np.minimum(a[...,2:],b[...,2:])-np.maximum(a[...,:2],b[...,:2]),0).prod(-1);ua=(a[...,2:]-a[...,:2]).prod(-1)+(b[...,2:]-b[...,:2]).prod(-1)-intersection;enc=(np.maximum(a[...,2:],b[...,2:])-np.minimum(a[...,:2],b[...,:2])).prod(-1)
 ua=np.maximum(ua,1e-7);enc=np.maximum(enc,1e-7)
 return intersection/ua-(enc-ua)/enc
def main():
 import torch
 torch.set_num_threads(2);p=verify();assert read(D/'GLOBAL_PREDICTION_BARRIER.json')['parents']==256;beg=time.perf_counter();errs={k:0. for k in ['actions','cost','weights','mean_gradient','chart','proposal','final','state_step']};steps=fits=0;changed=0
 for rec in p['records']:
  ds=rec['dataset'];a=rec['alias'];pred=load(D/'predictions'/ds/f'{a}.pt')
  if ds=='hc2'and a!='RC001':full=load(FIN/'Live/predictions'/f'{a}.pt');fs={s:full['fits'][f'T20_Live_{s}']for s in p['seeds']}
  else:full=load(D/'traces'/ds/f'{a}.pt');fs={s:full['fits'][f'Live_{s}']for s in p['seeds']}
  for s,f in fs.items():
   fits+=1;cfg=f['configuration'];target=array(f['target']);pos=f['positions'];assert len(pos)==8 and f['selected_step']==40 and f['actual_updates']==40;assert len(f['rounds'])==40 and len(f['path'])==41;assert not array(f['path'][0]['state']).any();assert np.array_equal(array(f['path'][0]['boxes']),pred['E'])
   for k,rr in enumerate(f['rounds']):
    steps+=1;r=rr['rollout'];mu=array(rr['current_policy_mean']);D0=array(r['D']);ac=array(r['actions']);w=array(r['weights']);eps=array(r['eps']);proposal=array(r['proposal_mean']);bs=array(f['path'][k]['boxes'])[pos];cm=np.concatenate([bs[:,:2],np.log(bs[:,2:])],-1);sig=np.repeat(np.array(cfg['scales']),16);std=D0[:,None,:]*np.array(cfg['scales'])[None,:,None]
    e=dict(chart=float(abs(mu-cm).max()),proposal=float(abs(proposal-mu).max()),actions=float(abs(ac-(proposal[:,None]+D0[:,None]*sig[None,:,None]*eps)).max()))
    ab=np.concatenate((ac[...,:2],np.exp(ac[...,2:])),axis=-1);cost=5*abs(ab-target[:,None]).sum(-1)+2*(1-giou(ab,target[:,None]));e['cost']=float(abs(cost-array(r['cost'])).max());weights=softmax(-cost/cfg['tau']);e['weights']=float(abs(weights-w).max())
    dif=(ac[:,:,None,:]-mu[:,None,None,:])/std[:,None,:,:];comp=-.5*(dif*dif).sum(-1)-np.log(std).sum(-1)[:,None,:]-2*math.log(2*math.pi);resp=softmax(comp);gm=(w[:,:,None,None]*resp[:,:,:,None]*(mu[:,None,None,:]-ac[:,:,None,:])/(std[:,None,:,:]**2)).sum((1,2))/len(mu);e['mean_gradient']=float(abs(gm-array(rr['policy_mean_gradient'])).max());state0=array(f['path'][k]['state']);state1=array(f['path'][k+1]['state']);e['state_step']=float(abs(state1-state0-array(rr['state_step'])).max())
    for key,val in e.items():errs[key]=max(errs[key],val)
    assert max(e.values())<2e-8,(a,s,k,e)
   e=float(abs(array(f['final'])-pred['Live'][s]).max());errs['final']=max(errs['final'],e);assert e==0.;changed+=int(not np.array_equal(pred['Live'][s],pred['E']))
  del full,fs,pred
  if fits%96==0:print('INDEPENDENT',fits,steps,flush=True)
 assert fits==768 and steps==30720
 out=dict(status='pass',fits=fits,updates=steps,zero_state_E_checks=fits,actual_changed_final_trajectories=changed,independent_NumPy_checks=['cxcylogwh chart','Live current proposal','actions and scale mapping','L1 plus GIoU cost','teacher softmax','two-component mixture responsibility and mean gradient','actual state increments','final output identity'],max_errors=errs,spatial_hyperparameters_unchanged=True,GT_read=False,seconds=time.perf_counter()-beg)
 put(D/'INDEPENDENT_SPATIAL_AUDIT.json',out);put(P/'INDEPENDENT_SPATIAL_AUDIT_PUBLIC.json',out);print(out,flush=True)
if __name__=='__main__':main()
