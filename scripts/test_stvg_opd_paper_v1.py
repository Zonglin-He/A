"""CPU numerical contracts and exact cohort/configuration authorization checks."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.stvg_opd_paper_common_v1 import prepare,read,BASE,OLD,PAPER,committed,write
from vg_tta.decota_spatial_opd_tunable_v1 import likelihood_loss,optimizer_step

def run():
    torch.set_num_threads(2);d=prepare();old=read(OLD/'DESIGN_LOCK.json');checks=0
    for ds in ['hc2','vidstg']:
        s=d['stages']['P0_'+ds];o=old['stages']['confirm_'+ds]
        assert s['parents']==o['parents'] and s['orders']==o['orders'];checks+=2
        p=read(PAPER/ds/'PLAN.json')
        for seq in s['orders'].values():
            assert len(seq)==128 and len(set(seq))==128 and set(seq)==set(s['parents']);checks+=1
        assert len({p['rows'][q]['source'] for q in s['parents']})==128;checks+=1
        cfg=d['datasets'][ds]['config'];assert cfg['sigma']==.1 and cfg['steps']==(20 if ds=='hc2' else 10);checks+=1
        bal=d['datasets'][ds]['balanced_one_query_per_parent']
        assert len(bal)==len({p['rows'][q]['source'] for q in bal})==(237 if ds=='hc2' else 732);checks+=1
    rng=np.random.default_rng(20261008)
    for sigma in [.1,.25,.5]:
        initial=rng.normal(size=(2,4));mu=torch.tensor(initial+.03,dtype=torch.float64,requires_grad=True)
        samples=torch.tensor(initial[:,None]+sigma*rng.normal(size=(2,32,4)),dtype=torch.float64)
        w=rng.uniform(size=(2,32));w/=w.sum(1,keepdims=True);weights=torch.tensor(w,dtype=torch.float64)
        loss=likelihood_loss(mu,torch.tensor(initial),samples,weights,sigma)
        grad=torch.autograd.grad(loss,mu)[0].numpy()
        expected=((mu.detach().numpy()-initial)-(w-1/32)[:,:,None].__mul__(samples.numpy()).sum(1))/(sigma*sigma*2)
        assert np.max(np.abs(grad-expected))<1e-10;checks+=1
    p=torch.tensor([1.],requires_grad=True);opt=torch.optim.Adam([p],lr=.03)
    assert optimizer_step(opt,p.sum(),[p],False) is None and p.item()==1 and not opt.state;checks+=1
    initial={'spatial.query_residual':torch.ones(256),'decoder.LN':torch.arange(1536,dtype=torch.float32)}
    final={n:v+.1 for n,v in initial.items()}
    for alpha in [0.,1/16,1/8]:
        c=committed(initial,final,alpha)
        assert torch.count_nonzero(c['spatial.query_residual'])==0
        assert torch.equal(c['decoder.LN'],initial['decoder.LN']+(final['decoder.LN']-initial['decoder.LN'])*alpha);checks+=2
    result=dict(status='pass',CPU_contracts=checks,GT_read=False,GPU_qualification=False)
    if not (BASE/'CPU_CONTRACTS.json').exists():write(BASE/'CPU_CONTRACTS.json',result)
    print(result)

if __name__=='__main__':run()
