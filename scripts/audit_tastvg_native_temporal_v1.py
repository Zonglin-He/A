"""Independent CPU probability/projection/state audit for the native head."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_proximal_common_v1 import *
def native_logp(logits,pairs):
 vals=[]
 for o,z in enumerate(logits):
  z=np.asarray(z,float).reshape(-1,2);ii,jj=np.triu_indices(len(z),1);raw=z[ii,0]+z[jj,1];raw-=np.logaddexp.reduce(raw);mapping={(int(s),int(e)):j for j,(s,e) in enumerate(zip(ii,jj))}
  vals.append(np.asarray([raw[mapping[tuple(p[o])]] for p in pairs]))
 v=vals[0]+vals[1];return v-np.logaddexp.reduce(v)
def run(ds):
 import torch
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.tastvg_spatial_rank_s11_v1 import average_ranks
 p=verify(ds);req=read(BASE/ds/'T/REQUEST.json');out=BASE/ds/'T';hs=read(out/'HEAD_SUPPORT.json');checks=0;rows=[]
 for cond in p['conditions']:
  for order,seq in p['splits']['search']['orders'].items():
   prev=hs['initial_sha256']
   for at,parent in enumerate(seq):
    x=load(out/'online'/cond/order/f'{at:05}.pt');base=load(BASE/ds/req['spatial_arm']/'online'/cond/order/f'{at:05}.pt');assert base['pre_sha']==x['pre_sha'] and base['post_sha']==x['post_sha'] and torch.equal(base['slow']['boxes'],x['slow']['boxes'])
    assert prev==x['temporal_pre_sha']==state_hash(x['temporal_pre_state']);prev=x['temporal_post_sha'];assert prev==state_hash(x['temporal_post_state']);checks+=1;u=x['native_temporal_update']
    if u is None:assert x['temporal_pre_sha']==x['temporal_post_sha'];continue
    assert at%4==0 and u['probability_scope']=='restricted_native_legal_span_pair_support'
    rank=average_ranks(u['scores']);np.testing.assert_array_equal(rank,u['rank'])
    lp=native_logp(u['logits_before'],u['pairs']);lq=-rank/u['teacher_temperature'];lq-=np.logaddexp.reduce(lq)
    spread=float(np.ptp(u['scores']));strength=spread/(spread+u['s_ref']);assert abs(strength-u['strength'])<1e-14
    target=(1-strength)*np.exp(lp)+strength*np.exp(lq);np.testing.assert_allclose(np.exp(lp),u['p'],atol=1e-12,rtol=1e-10);np.testing.assert_allclose(target,u['target'],atol=1e-12,rtol=1e-10)
    np.testing.assert_allclose(np.sum(np.exp(lp)*(lp-np.log(target))),u['loss_before'],atol=1e-10,rtol=1e-8)
    proposed={n:v.clone().add_(u['gradients'][n],alpha=-u['lr']) for n,v in x['temporal_pre_state'].items()};dd={n:proposed[n]-x['temporal_pre_state'][n] for n in proposed};pn=float(torch.sqrt(sum(d.double().square().sum() for d in dd.values())));fac=min(1.,hs['arrival_radius']/(pn+1e-30))
    assert abs(pn-u['proposed_arrival_norm'])<1e-9 and abs(fac-u['projection_factor'])<1e-9
    for n,z in proposed.items():expect=x['temporal_pre_state'][n]+fac*dd[n] if fac<1 else z;np.testing.assert_allclose(expect,x['temporal_post_state'][n],atol=1e-7,rtol=1e-6);checks+=z.numel()
    postlp=native_logp(u['logits_after'],u['pairs']);np.testing.assert_allclose(np.sum(np.exp(postlp)*(postlp-np.array(u['frozen_log_target']))),u['loss_after'],atol=1e-10,rtol=1e-8)
    if spread==0:assert u['loss_before']==0 and x['temporal_pre_sha']==x['temporal_post_sha']
    rows.append(dict(parent=parent,order=order,condition=cond,arrival=at,strength=strength,spread=spread,loss_before=u['loss_before'],loss_after=u['loss_after'],gradient_norm=u['global_gradient_norm'],projection_factor=fac,actual_arrival_norm=u['actual_arrival_norm']))
 assert not torch.cuda.is_initialized();write(out/'TEMPORAL_STEP_ROWS.json',rows);write(out/'TEMPORAL_AUDIT.json',dict(status='pass',checks=checks,spatial_stream_bitwise=True,offset_probability_mapping=True,GPU_initialized=False,time=time.time()))
if __name__=='__main__':run(sys.argv[1])
