"""F37 S4 and conditional real-pseudolabel private-output comparison.

No labels/evaluator imported. Parents and current registries remain read-only.
"""
import argparse
import copy
import fcntl
import gc
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,load,save,sha
from scripts.prepare_spatial4_attribute_v1 import OUT,F35,F36
from scripts.run_spatial10_components_v1 import capture_timed,observations,existing
from scripts.run_parametric_observation_v1 import cpu_tree


def plan():
    p=read(OUT/'LOCK.json')
    for f,h in p['protected_pins'].items():assert sha(ROOT/f)==h,f
    return p


class Budget:
    def __init__(self,p):
        self.caps=p['caps'];self.file=OUT/'COUNTERS.json'
        self.counts=read(self.file) if self.file.exists() else {k:0 for k in self.caps}
    def charge(self,k,n):
        assert self.counts[k]+n<=self.caps[k],('budget',k,self.counts)
        self.counts[k]+=n;status(self.file,self.counts)
    def reserve(self,**needs):
        assert all(self.counts[k]+n<=self.caps[k] for k,n in needs.items())


def dest(stage,key):return OUT/stage/(key.replace(':','_')+'.pt')


def record(p,x):
    save(p,cpu_tree(x));write(p.with_suffix('.json'),dict(key=x['key'],sha256=sha(p),completed=time.time(),
        code_pins={f:sha(ROOT/f) for f in ['scripts/run_spatial4_attribute_v1.py','vg_tta/simplification_partial_v1.py','vg_tta/partial_absorption_v1.py']}))


def reference(r):
    pp=F35/'spatial'/(r['key'].replace(':','_')+'.pt')
    if existing(pp):
        x=load(pp);z=x['fits']['C0']
    else:
        pp=F36/'space'/(r['key'].replace(':','_')+'.pt');assert existing(pp)
        x=load(pp);z=x['fits']['A0']
    return x,z,dict(path=str(pp),sha256=sha(pp))


def one(model,expert,r,c,budget,stage,smoke=False):
    import torch
    from vg_tta.parametric_observation_v1 import ObservationReplay
    from vg_tta.simplification_partial_v1 import fit
    from vg_tta.partial_absorption_v1 import FinalBBoxReplay,full_bbox_prediction
    from vg_tta.decota_tastvg_episode_v1 import fitted_merge
    from scripts.audit_parametric_reinsertion_v1 import full_prediction
    from methods.decota_refine_uniform_v1.api import reconstruct
    budget.reserve(fits=8,backward=65,new_DINO=0)
    old,parent,pinfo=reference(r)
    frames,base,records,views,qa,cost=capture_timed(model,r);ids=r['input']['frame_ids']
    assert qa==old['qa']
    it=ObservationReplay(model,views,len(ids),'spatial')
    with torch.no_grad():zero=it.values()
    assert torch.equal(zero['boxes'].cpu(),old['native_boxes'])
    assert list(fitted_merge(zero['logits'],records,ids))==old['native_indices']
    assert all(torch.equal(a.cpu(),b) for a,b in zip(zero['logits'],old['native_logits']))
    fits={'S0':parent};audits={};reuse={'S0':pinfo};ex=old['expert']
    lr=.01 if c=='hcstvg1_test' else .1
    if stage=='space':
        # panel=False verifies EXACT 4 original+weak cached requests; no DINO inference.
        ex=observations(expert,r,frames,ids,old['native_indices'],ex,False,budget)
        assert ex['new_DINO']==0 and ex['positions4']==old['expert']['positions4']
        z=fit(it,records,ids,anchors=ex['anchors']['single4'],planned=4,kappa=None,gamma=0.,lr=lr,steps=10,charge=budget.charge)
        fits['S4']=z
        audits['S4']=full_prediction(model,frames,ids,r['input'],r['subject'],z['state'],z['final'])['audit']
    else:
        expanded=FinalBBoxReplay(model,views,len(ids))
        with torch.no_grad():ez=expanded.values()
        assert torch.equal(ez['boxes'],zero['boxes'])
        assert all(torch.equal(a,b) for a,b in zip(ez['logits'],zero['logits']))
        if stage=='practical_dev':specs=[(n,m) for n in (1792,2820) for m in (.5,1.)]
        else:
            selected=read(OUT/'PRACTICAL_SELECTION.json')['directions'][c]
            specs=[(n,selected[str(n)]['multiplier']) for n in (1792,2820)]
        for n,m in specs:
            name=f'H{n}_m{m:g}'
            if n==1792 and m==1:
                fits[name]=parent;reuse[name]=pinfo;continue
            interface=it if n==1792 else expanded
            z=fit(interface,records,ids,anchors=parent['anchors'],planned=4,kappa=None if c=='hcstvg1_test' else 2.,
                gamma=1e-4,lr=lr*m,steps=10,charge=budget.charge)
            assert z['parameter_count']==n
            fn=full_prediction if n==1792 else full_bbox_prediction
            audits[name]=fn(model,frames,ids,r['input'],r['subject'],z['state'],z['final'])['audit']
            fits[name]=z
    if smoke:
        for n,lr0,steps in [('steps0',lr,0),('lr0',0.,1)]:
            z=fit(it,records,ids,anchors=ex['anchors']['single4'],planned=4,kappa=None,gamma=0.,lr=lr0,steps=steps,charge=budget.charge)
            assert z['state_delta']==0 and torch.equal(z['final']['boxes'],zero['boxes'].cpu())
            a=full_prediction(model,frames,ids,r['input'],r['subject'],z['state'],z['final'])['audit']
            record(dest('noops',r['key']+'_'+n),dict(key=r['key'],fit=z,audit=a))
    controls={'Frozen':dict(boxes=zero['boxes'].cpu(),indices=old['native_indices'])}
    for name,z in fits.items():
        assert z['final']['indices']==old['native_indices']
        assert all(torch.equal(a,b.cpu()) for a,b in zip(z['final']['logits'][:2],zero['logits']))
        if name in ('S0','S4'):
            bb,aa=reconstruct(zero['boxes'].cpu(),z['anchors'],ids,'absolute')
            controls[name+'_absolute']=dict(boxes=bb,indices=old['native_indices'],audit=aa)
    record(dest(stage,r['key']),dict(key=r['key'],cohort=c,source=r['source'],group=r['group'],
        role=r.get('f37_role','watched'),frame_ids=ids,native_boxes=zero['boxes'].cpu(),native_indices=old['native_indices'],
        native_logits=[z.cpu() for z in zero['logits']],records=records,qa=qa,cost=cost,expert=ex,fits=fits,controls=controls,
        audits=audits,reuse=reuse,GT_online=False,actual_new_DINO=0,parent=pinfo))
    print('F37',stage,r['key'],{n:(z['best_step'],round(z['state_delta'],5)) for n,z in fits.items()},flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['space','practical_dev','practical_eval','practical_guard'])
    ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
    from scripts.run_decota_refine_v1 import configure
    from scripts.run_closure_v1 import model_for
    from vg_tta.foreground_runtime import state_digest
    from vg_tta.tg_spatial_tta_v1 import SpatialExpert
    import torch
    configure();p=plan();budget=Budget(p)
    lease=open(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock','a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tick=time.time();failure=None;expert=None;n=0
    try:
        if a.stage=='space':
            cfg=read(ROOT/'artifacts/decota_refine_v1/lock.json');expert=SpatialExpert(cfg['expert_snapshot']);ed=state_digest(expert.model)
        for c in p['rows']:
            if a.stage=='space':rr=p['space_extension'][c]
            elif a.stage=='practical_guard':rr=[r for r in p['H_rows'] if r['key'].startswith(c+':')]
            else:rr=[r for r in p['rows'][c] if r['f37_role']==('development' if a.stage=='practical_dev' else 'evaluation')]
            rr=[r for r in rr if not existing(dest(a.stage,r['key']))]
            if not rr or (a.limit and n>=a.limit):continue
            model=model_for(c);md=state_digest(model)
            for r in rr:
                if a.limit and n>=a.limit:break
                smoke=a.stage=='space' and not list((OUT/'noops').glob(c+'*_lr0.pt'))
                one(model,expert,r,c,budget,a.stage,smoke)
                assert state_digest(model)==md
                if expert:assert state_digest(expert.model)==ed
                n+=1;status(OUT/'STATUS.json',dict(stage=a.stage,last=r['key'],completed_invocation=n,counts=budget.counts))
            del model;gc.collect();torch.cuda.empty_cache()
    except BaseException as e:failure=repr(e);raise
    finally:
        write(OUT/'leases'/f'{a.stage}_{time.time_ns()}.json',dict(stage=a.stage,seconds=time.time()-tick,failure=failure,counts=budget.counts))
        fcntl.flock(lease,fcntl.LOCK_UN);lease.close()

if __name__=='__main__':main()
