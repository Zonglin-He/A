"""CPU scope contracts, not effectiveness tests or mirrored scorer checks."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import vg_tta.decota_actuation_scope_v1 as m
from vg_tta.c1_enabling_tricks_v1 import QUERY

class Toy:
    def __init__(self,base,ids,key,settings):self.named=base.named
    def state(self):return {n:p.detach().clone() for n,p in self.named}
    def restore(self,state):
        with torch.no_grad():
            for n,p in self.named:p.copy_(state[n].to(p))
    def values(self):
        v=torch.stack([p.mean() for _,p in self.named]).sum();center=.4+.02*v
        return dict(boxes=torch.stack([torch.stack([center,center,center*0+.2,center*0+.2])]))

class Base:
    def __init__(self):self.named=[(QUERY,torch.zeros(256,requires_grad=True))]+[(f'ln{i}',torch.ones(256,requires_grad=True)) for i in range(6)]
    def restore(self,state):Toy(self,None,None,None).restore(state)

def run():
    actual=m.TrickReplay;m.TrickReplay=Toy
    try:
        base=Base();initial=Toy(base,None,None,None).state();ex=dict(observations={(0,0):dict(probe=dict(boxes=[[.6,.6,.2,.2],[.45,.45,.2,.2]],target_scores=[.6,.5],accepted=True),receipt=dict(context_active=False))})
        cfg=dict(optimizer='sgd',lr=1.,evidence='frame',authority_source='frame');z=m.fit_scope(base,initial,ex,[0],'toy',cfg,'u_only')
        assert z['active_parameters']==256 and z['slow_LN']['steps']==1
        for n in initial:
            if n!=QUERY:assert torch.equal(z['state'][n],initial[n])
        assert any(not torch.equal(z['write_proposal'][n],z['state'][n]) for n in initial if n!=QUERY)
        # The slow write must never alter the current prediction saved before it.
        rp=Toy(base,None,None,None);rp.restore(z['state']);assert torch.equal(rp.values()['boxes'],z['final']);checks=4
        c=m.commit_state(initial,z['write_proposal']);assert c[QUERY].abs().sum()==0
        for n in initial:
            if n!=QUERY:assert torch.equal(c[n],initial[n]+(z['write_proposal'][n]-initial[n])/16)
        checks+=2
        rp.restore(initial)
        for n,p in base.named:assert torch.equal(p,initial[n]);checks+=1
        s=m.fit_scope(base,initial,ex,[0],'toy',cfg,'small_LN');a=s['authority']
        for n in initial:
            expected=s['proposal_state'][n] if n==QUERY else initial[n]+a*(s['proposal_state'][n]-initial[n])
            assert torch.equal(s['state'][n],expected);checks+=1
        empty=m.fit_scope(base,initial,dict(observations={}),[0],'toy',cfg,'u_only');assert empty['gradient_calls']==0 and empty['slow_LN'] is None;checks+=1
        print('ACTUATION_SCOPE_CONTRACTS_PASS',checks)
    finally:m.TrickReplay=actual

if __name__=='__main__':run()
