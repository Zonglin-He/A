"""Synthetic component contracts; explicitly not real STVG GPU qualification."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from methods.decota_final_simplified_v1.tensors import ParameterState
from methods.decota_final_simplified_v1.observations import uniform_positions
from vg_tta.tastvg_deployment_corruption_v2 import burst_spec,apply_burst
from vg_tta.stvg_opd_paper_component_audit_v1 import audit,giou
from scripts.stvg_opd_paper_inputs_v1 import observation
from scripts.stvg_opd_paper_common_v1 import BASE,read,write
import vg_tta.stvg_opd_paper_ablations_v1 as component
import vg_tta.decota_spatial_opd_tunable_v1 as core

class SyntheticBase(ParameterState):
    def __init__(self):
        self.delta=torch.zeros(256,requires_grad=True)
        self.LN=torch.zeros(1536,requires_grad=True)
        self.named=[('spatial.query_residual',self.delta),('spatial.layers.5.norm1.weight',self.LN)]
        self.initial=self.state()
    def values(self):
        offset=self.delta.reshape(-1,4).mean(0)+self.LN.reshape(-1,4).mean(0)
        boxes=torch.sigmoid(torch.tensor([[0.,0.,-.8,-.8],[.05,-.05,-.7,-.9]])+offset)
        return {'boxes':boxes}

class SyntheticReplay(ParameterState):
    def __init__(self,base,*args):self.base=base;self.named=base.named
    def values(self):return self.base.values()

def run():
    torch.set_num_threads(2);checks=0
    original_component,original_core=component.TrickReplay,core.TrickReplay
    component.TrickReplay=core.TrickReplay=SyntheticReplay
    ex=dict(observations={('original',0):dict(probe=dict(boxes=torch.tensor([[.54,.5,.37,.34]]),target_scores=[.8],accepted=True),receipt=dict(context_active=True))})
    cfg=read(BASE/'DESIGN_LOCK.json')['datasets']['vidstg']['config']
    try:
        a=SyntheticBase();baseline=core.fit(a,a.initial,ex,[0,1],'synthetic','on_policy',config=cfg)
        b=SyntheticBase();full=component.component_fit(b,b.initial,ex,[0,1],'synthetic','on_policy',config=cfg)
        assert torch.equal(baseline['final'],full['final'])
        assert all(torch.equal(baseline['state'][n],full['state'][n]) for n in baseline['state']);checks+=2
        for variant,scope in [('query_only',256),('LN_only',1536),('direct_L1_GIoU',1792)]:
            b=SyntheticBase();fit=component.fit_variant(b,b.initial,ex,[0,1],'synthetic',variant,cfg)
            assert fit['active_parameters']==scope and fit['selected_step']==10
            assert audit(fit,ex)['status']=='pass';checks+=3
            for n in fit['initial']:
                if n not in fit['optimizer_parameter_names']:assert torch.equal(fit['initial'][n],fit['state'][n]);checks+=1
            assert all(torch.equal(b.initial[n],b.state()[n]) for n in b.initial);checks+=1
        for variant in ['query_only','LN_only','direct_L1_GIoU']:
            b=SyntheticBase();empty=dict(observations={})
            fit=component.fit_variant(b,b.initial,empty,[0,1],'synthetic',variant,cfg)
            assert fit['gradient_calls']==0 and torch.equal(fit['before'],fit['final'])
            assert audit(fit,empty)['status']=='pass';checks+=2
    finally:component.TrickReplay=original_component;core.TrickReplay=original_core
    ids=list(range(100));frames=np.random.default_rng(13).integers(0,256,size=(100,24,32,3),dtype=np.uint8)
    row=dict(source='synthetic-video',frame_ids=ids,input=dict(frame_count=100,frame_ids=ids))
    for coverage in [2.5,5.,10.]:
        spec=burst_spec(row['source'],100,ids,coverage)
        assert spec['length']==int(np.ceil(coverage)) and spec['actual_physical_fraction']==spec['length']/100;checks+=2
        for family in ['frame_drop','frame_freeze','motion_blur','occlusion','exposure']:
            actual,_,_=observation('hc2',row,f'{family}_{coverage}',frames)
            expected=apply_burst(frames,row['input'],spec,family,row['source'])
            assert np.array_equal(actual,expected)
            assert np.array_equal(actual[[p for p in ids if p not in spec['positions']]],frames[[p for p in ids if p not in spec['positions']]])
            checks+=2
    assert observation('vidstg',row,'clean',frames)[0] is frames;checks+=1
    grid=[0,3,8,12,18,29,40,57,80,99]
    for k in [1,2,4,8]:
        positions=uniform_positions(grid,[0,9],k)
        assert len(positions)==len(set(positions))==k and positions==sorted(positions);checks+=2
    result=dict(status='pass',synthetic_contracts=checks,real_GPU_qualification=False,GT_read=False,
        interventions=['standard direct L1/GIoU','query256 versus LN1536 frozen coordinates','no-feedback no Adam',
            'full arithmetic parity','physical 2.5/5/10 percent burst','Uniform K distinct physical samples'],time=time.time())
    write(BASE/'COMPONENT_CPU_CONTRACTS.json',result);print(result)

if __name__=='__main__':run()
