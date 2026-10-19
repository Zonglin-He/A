"""No-GT independent native frame gates, query means and every head posterior."""
from pathlib import Path
import sys,numpy as np
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_native_stage128_v1 import D,P,MEM,HR,DEV
from scripts.stvg_temporal_marginal_decode_util256_v1 import independent_cdf


def main():
 p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==128 and bar['plan_sha256']==sha(D/'PLAN_LOCK.json')and not(D/'GT_EXPOSURE.json').exists();checks=gates=queries=mixtures=0;maxquery=maxprob=maxgate=0;tol=16*np.finfo(np.float32).eps
 for r in p['records']:
  a=r['alias'];z=load(D/'predictions'/f'{a}.pt');cache=load(MEM/'temporal_inputs'/f'{a}.pt');hr=load(HR/'predictions'/f'{a}.pt');old=load(DEV/'priors'/f'{a}.pt');native=z['outputs']['Native']['q'];assert np.array_equal(native,old['q'])and np.array_equal(native,z['outputs']['P2L6']['q'])
  for off,(tr,kw,g)in enumerate(zip(z['native_stage_traces'],cache['inputs'],cache['grids'])):
   lm=tr['initial_motion_logits'].numpy().astype(float);la=tr['initial_appearance_logits'].numpy().astype(float);ap=tr['action_logits1'].numpy().astype(float);prob=(1/(1+np.exp(-lm))+1/(1+np.exp(-la)))/2;action=1/(1+np.exp(-ap));err=max(float(abs(prob-tr['initial_gate_prob'].numpy()).max()),float(abs(action-tr['action1'].numpy()).max()));assert err<tol;maxgate=max(maxgate,err)
   fallback=np.flatnonzero(tr['initial_gate_prob'].numpy()>0).tolist();first=np.flatnonzero(tr['initial_gate_prob'].numpy()>tr['theta']).tolist()or fallback;second=np.flatnonzero(tr['action1'].numpy()>.5).tolist()or fallback;assert first==tr['first']and second==tr['second']and fallback==tr['fallback'];gates+=2
   l=g['h']*g['w'];H=kw['encoded_feature'][-l:].permute(1,0,2).numpy().astype(float)
   for selected,attention,q in [(first,tr['attention1'],tr['query1']),(second,tr['attention2'],tr['query2'])]:
    ex=(H[selected]*attention.numpy().astype(float)[...,None]).mean((0,1));err=float((abs(ex-q.numpy())/(1+abs(ex))).max());assert err<tol;maxquery=max(maxquery,err);queries+=1
   assert np.array_equal(tr['query2'].numpy(),tr['native_query2'].numpy())and np.array_equal(z['pass_logits'][1][off][-1].numpy(),old['logits'][off].numpy())
  allq=[]
  for stage in [0,1]:
   stageq=[]
   for layer in range(6):
    grids=[]
    for rec,ll in zip(z['records'],z['pass_logits'][stage]):
     v=ll[layer].numpy()[0].astype(float);i,j=np.triu_indices(len(v),1);s=v[i,0]+v[j,1];pr=np.exp(s-s.max());pr/=pr.sum();grids.append(dict(ids=rec['frame_ids'],ij=np.stack([i,j]),p=pr))
    q=independent_cdf(grids,z['frame_ids']);exact=z['outputs'][f'P{stage+1}L{layer+1}']['q'];err=float(abs(q-exact).max());assert err<1e-12;maxprob=max(maxprob,err);checks+=1;stageq.append(exact)
   allq.append(stageq)
  p1=sum(allq[0])/6;p2=sum(allq[1])/6;u=(native>0).astype(float);u/=u.sum();expected=dict(FirstPassMix=(native+allq[0][-1])/2,Pass1Mean6=p1,Pass2Mean6=p2,BothMean12=(p1+p2)/2,HiResAuxMix=(hr['observation']['q']+p2)/2,NativeUniformMix=(native+u)/2,HiResUniformMix=(hr['observation']['q']+u)/2)
  for k,v in expected.items():assert np.max(abs(z['outputs'][k]['q']-v))<1e-15;mixtures+=1
 put(P/'STAGE_INDEPENDENT_AUDIT.json',dict(status='pass',all128_before_GT=True,raw_head_posterior_CDF_checks=checks,native_gate_checks=gates,independent_native_query_means=queries,native_second_query_and_last_logits_exact=256,fixed_scaled_query_and_gate_tolerance=tol,max_scaled_query_error=maxquery,max_gate_sigmoid_error=maxgate,max_CDF_error=maxprob,posterior_mixture_checks=mixtures,no_new_model_calls=True,no_GT=True));print('independent native stage pass',checks,maxquery,maxprob,flush=True)

if __name__=='__main__':main()
