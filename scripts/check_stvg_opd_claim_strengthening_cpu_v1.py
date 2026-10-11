"""Synthetic full-step equivalence and independently calculated paired statistics.

No media, GT, source weights or real model execution is used by this check.
The real decoder/expert qualification remains a separate mandatory GPU stage.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from methods.decota_final_simplified_v1.tensors import ParameterState
from scripts.decota_matrix_common_v1 import write,sha
from vg_tta import decota_spatial_opd_tunable_v1 as original
from vg_tta.stvg_opd_deployment_profile_v1 import fit,compare_complete_trace,compare_output,cpu_contracts


class SyntheticBase(ParameterState):
    def __init__(self):
        gen=torch.Generator().manual_seed(20261011)
        self.named=[('spatial.query_residual',torch.zeros(256,requires_grad=True)),
                    ('synthetic.LN',torch.ones(1536,requires_grad=True))]
        self.initial=self.state()
        self.matrix=torch.randn(28,1792,generator=gen)*.003
        self.bias=torch.linspace(-.8,.8,28).reshape(7,4)


class SyntheticReplay(ParameterState):
    def __init__(self,base,ids,key,spec):
        self.base=base;self.named=base.named
    def values(self):
        vec=torch.cat([p.reshape(-1) for _,p in self.named])
        origin=torch.cat([self.base.initial[n].reshape(-1) for n,_ in self.named])
        self.native_raw=self.base.bias+(self.base.matrix@(vec-origin)).reshape(7,4)
        return dict(boxes=self.native_raw.sigmoid())


def run():
    torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    contract=cpu_contracts();counts=[]
    prior=original.TrickReplay;original.TrickReplay=SyntheticReplay
    try:
        for steps in [10,40]:
            for mode in ['empty','rejected','informative','constant_zero_reward']:
                base=SyntheticBase();initial=base.state();expert=dict(observations={})
                if mode!='empty':
                    for pos in range(7):
                        box=[3.,3.,.01,.01] if mode=='constant_zero_reward' else [.5,.5,.45,.45]
                        expert['observations']['single4',pos]=dict(probe=dict(boxes=[box],
                            target_scores=[.9],accepted=mode!='rejected'),receipt=dict(context_active=False))
                cfg=dict(original.CONFIG,steps=steps)
                old=original.fit(base,initial,expert,list(range(7)),'synthetic-contract','on_policy',config=cfg)
                traced=fit(base,initial,expert,list(range(7)),'synthetic-contract',config=cfg,trace=True,replay_type=SyntheticReplay)
                plain=fit(base,initial,expert,list(range(7)),'synthetic-contract',config=cfg,trace=False,replay_type=SyntheticReplay)
                complete=compare_complete_trace(traced,old);output=compare_output(plain,traced)
                assert all(torch.equal(v,initial[n]) for n,v in base.state().items())
                if mode in ['empty','rejected','constant_zero_reward']:assert plain['gradient_calls']==0
                else:assert plain['gradient_calls']==steps
                counts.append(dict(steps=steps,mode=mode,complete=complete,trace_disabled=output))
        changed=dict(traced);changed['final']=traced['final'].clone();changed['final'][0,0]+=.01
        try:compare_complete_trace(changed,old)
        except AssertionError:pass
        else:raise AssertionError('Changed synthetic output accepted')
    finally:original.TrickReplay=prior
    from scripts.finalize_stvg_opd_design_ablation_v1 import statistics
    from scripts.review_stvg_opd_supplement_v1 import independent_stats,compare_stats
    rows=[dict(query_ordinal=q,source_id=str(q),order=o,condition='clean',
               direct=(q-3)*.01+(0.002 if o=='order2' else 0.)) for o in ['order1','order2'] for q in range(7)]
    checks=compare_stats(statistics(rows,['direct']),independent_stats(rows,['direct'],balanced=True))
    assert not torch.cuda.is_initialized()
    files=['vg_tta/stvg_opd_deployment_profile_v1.py','scripts/check_stvg_opd_claim_strengthening_cpu_v1.py',
           'scripts/finalize_stvg_opd_expanded_state_v1.py','scripts/run_stvg_opd_deployment_latency_v1.py']
    write(ROOT/'artifacts/stvg_opd_claim_strengthening_v1/CPU_IMPLEMENTATION_READBACK.json',
        dict(status='pass',scope='synthetic CPU full-step/trace-off equivalence and paired bootstrap; no real-model qualification',
             synthetic_fit_cases=8,synthetic_complete_rounds=sum(x['steps'] for x in counts),
             synthetic_fits_executed=24,changed_output_rejected=1,config_contracts=contract,
             paired_statistics_checks=checks,records=counts,pins={p:sha(ROOT/p) for p in files},
             actual_GPU_fits=0,new_DINO_calls=0,GT_read=False,CUDA_initialized=False,time=time.time()))
    print('CLAIM_STRENGTHENING_CPU_CONTRACTS',len(counts),checks,flush=True)


if __name__=='__main__':run()
