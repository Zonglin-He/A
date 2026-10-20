"""Independent initialization, normalized proposal, gradient and readout audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='2'
import sys,math,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_single_referdino_initialization_coupled_common_v1 import ROOT,D,P,BASE,read,put,sha,load,initial,verify
from scripts.audit_stvg_spatial_strong_bidirectional256_v1 import array,softmax,giou

def pregt(ds,phase):
 p=verify();bar=read(D/ds/f'{phase.upper()}_FIT_BARRIER.json');pb=read(D/ds/phase/'PREDICTION_BARRIER.json');errors={k:0. for k in ['initialization','chart','actions','cost','weights','gradient','state_increment','readout']};count=steps=readouts=0;beg=time.perf_counter()
 for rec in [r for r in p['records']if r['dataset']==ds]:
  a=rec['alias'];old=load(BASE/f'predictions/{ds}/{a}.pt');inp=load(BASE/f'inputs/{ds}/{a}.pt');S=array(inp['Source']).astype(float);delta=np.array(old['initialization']['delta']);first_eps=None
  for cid,cfg in bar['configs'].items():
   entry=bar['files'][cid+'/'+a];assert sha(ROOT/entry['path'])==entry['sha256'];f=load(ROOT/entry['path'])['fit'];g=cfg['initialization_gamma'];E=initial(S,old['E_Ref'],g);ind=np.concatenate((S[:,:2]+g*delta[:2],np.exp(np.log(S[:,2:])+g*delta[2:])),axis=-1);errors['initialization']=max(errors['initialization'],float(abs(E-ind).max()));assert np.array_equal(array(f['path'][0]['boxes']),E)and not array(f['path'][0]['state']).any();assert np.array_equal(array(f['target']),array(inp['teacher_native']['boxes']));assert f['actual_updates']==40 and len(f['path'])==41;count+=1
   if first_eps is None:first_eps=array(f['rounds'][0]['rollout']['eps'])
   else:assert np.array_equal(first_eps,array(f['rounds'][0]['rollout']['eps']))
   for k,rr in enumerate(f['rounds']):
    steps+=1;r=rr['rollout'];mu=array(rr['current_policy_mean']);d=array(r['D']);acts=array(r['actions']);w=array(r['weights']);eps=array(r['eps']);b=array(f['path'][k]['boxes'])[f['positions']];chart=np.concatenate((b[:,:2],np.log(b[:,2:])),axis=-1);assert np.array_equal(mu,array(r['proposal_mean']));assert np.array_equal(eps[:,:8],-eps[:,8:16])and np.array_equal(eps[:,16:24],-eps[:,24:]);sig=np.repeat(np.array(cfg['scales']),16);std=d[:,None,:]*np.array(cfg['scales'])[None,:,None];boxes=np.concatenate((acts[...,:2],np.exp(acts[...,2:])),axis=-1);target=array(f['target'])[:,None];cost=5*abs(boxes-target).sum(-1)+2*(1-giou(boxes,target));q=softmax(-cost/cfg['tau']);diff=(acts[:,:,None,:]-mu[:,None,None,:])/std[:,None,:,:];comp=-.5*np.square(diff).sum(-1)-np.log(std).sum(-1)[:,None,:]-2*math.log(2*math.pi);resp=softmax(comp);grad=(w[:,:,None,None]*resp[:,:,:,None]*(mu[:,None,None,:]-acts[:,:,None,:])/std[:,None,:,:]**2).sum((1,2))/len(mu)
    checks=dict(chart=float(abs(chart-mu).max()),actions=float(abs(acts-(mu[:,None]+d[:,None]*sig[None,:,None]*eps)).max()),cost=float(abs(cost-array(r['cost'])).max()),weights=float(abs(q-w).max()),gradient=float(abs(grad-array(rr['policy_mean_gradient'])).max()),state_increment=float(abs(array(f['path'][k+1]['state'])-array(f['path'][k]['state'])-array(rr['state_step'])).max()))
    assert max(checks.values())<2e-8,(a,cid,k,checks)
    for key,e in checks.items():errors[key]=max(errors[key],e)
  cache={};fnpz=D/ds/phase/'predictions'/f'{a}.npz';assert sha(fnpz)==pb['files'][str(fnpz.relative_to(ROOT))]
  with np.load(fnpz)as z:
   for arm,c in pb['definitions'].items():
    key=c['fit_phase'],c['config_id']
    if key not in cache:
     fb=read(D/ds/f'{key[0].upper()}_FIT_BARRIER.json');entry=fb['files'][key[1]+'/'+a];cache[key]=load(ROOT/entry['path'])['fit']
    f=cache[key];E=initial(S,old['E_Ref'],c['initialization_gamma']);b=array(f['path'][c['steps']]['boxes']);alpha=c['alpha'];expected=np.concatenate(((1-alpha)*E[:,:2]+alpha*b[:,:2],np.exp((1-alpha)*np.log(E[:,2:])+alpha*np.log(b[:,2:]))),-1);e=float(abs(expected-z[arm]).max());errors['readout']=max(errors['readout'],e);assert e<1e-10;readouts+=1
  if count%64==0:print('GAMMA_INDEPENDENT',ds,phase,count,steps,flush=True)
 result=dict(status='pass',new_fits=count,new_updates_independently_audited=steps,readout_checks=readouts,max_errors=errors,paired_normalized_noise_across_different_initializations=True,teacher_evidence_unchanged=True,all128=True,GT_read=False,seconds=time.perf_counter()-beg);put(P/ds/phase/'PRE_GT_AUDIT_PUBLIC.json',result);put(P/ds/phase/'READOUT_AUDIT_PUBLIC.json',dict(status='pass',checks=readouts,max_abs=errors['readout'],GT_read=False));print('GAMMA_PRE_GT_AUDIT_PASS',ds,phase,count,steps,flush=True)

def scored(ds,phase):
 import scripts.audit_stvg_single_referdino_parameter_search256_v1 as module
 module.D=D/ds;module.P=P/ds;module.DS=ds;module.verify=verify;module.scored(phase)
if __name__=='__main__':
 if sys.argv[1]=='pre-gt':pregt(sys.argv[2],sys.argv[3])
 else:scored(sys.argv[2],sys.argv[3])
