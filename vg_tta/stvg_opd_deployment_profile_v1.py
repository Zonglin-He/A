"""Local audit-off deployment fit, qualified separately; CURRENT is unchanged.

Sampling, rewards, student likelihood and Adam use the original functions.
Trace mode is only for qualification. Timed mode avoids per-round CPU traces,
independent audit arithmetic and diagnostic post-update decoder forwards.
"""
import torch
from vg_tta.c1_enabling_tricks_v1 import TrickReplay
from vg_tta.decota_optimizer_posterior_r1_v1 import Energy,flat
from vg_tta.decota_spatial_opd_tunable_v1 import rollout,likelihood_loss,CONFIG
from vg_tta.decota_spatial_opd_boundary_chart_proposal003 import proposed_coordinates


class DeploymentReplay(TrickReplay):
    def values(self):
        outputs=[]
        hook=self.base.decoder.decoder.bbox_embed.register_forward_hook(
            lambda module,args,out:outputs.append(out))
        try: result=super().values()
        finally:hook.remove()
        assert len(outputs)==12
        self.native_raw=torch.stack([outputs[5+(i%2)*6][i//2,0]
                                    for i in range(len(self.ids))]).float()
        assert torch.equal(self.native_raw.sigmoid(),result['boxes'])
        return result


def check_config(cfg):
    if set(cfg)!=set(CONFIG) or cfg['samples']!=32:
        raise ValueError('Original fixed Gaussian configuration required')
    if any(isinstance(cfg[k],bool) or not isinstance(cfg[k],(float,int)) or
           not 0<cfg[k]<float('inf') for k in ['sigma','tau','lr']):
        raise ValueError('Invalid fixed positive scalar')
    if isinstance(cfg['steps'],bool) or not isinstance(cfg['steps'],int) or cfg['steps']<1:
        raise ValueError('Invalid original round count')
    if isinstance(cfg['writeback'],bool) or not 0<=cfg['writeback']<=1:
        raise ValueError('Invalid original consolidation')


@torch.enable_grad()
def fit(base,initial,expert,ids,key,*,config,trace=False,replay_type=DeploymentReplay):
    cfg=dict(config);check_config(cfg)
    base.restore(initial);rp=replay_type(base,ids,key,{})
    origin=rp.state();params=[p for _,p in rp.named];names=[n for n,_ in rp.named]
    assert sum(p.numel() for p in params)==1792
    with torch.no_grad():before=rp.values()['boxes'].detach().clone()
    support=Energy(expert,before,'top1');positions=[int(f[0]) for f in support.frames]
    evidence=torch.cat([f[1] for f in support.frames],0).detach() if positions else before.new_empty((0,4))
    opt=torch.optim.Adam(params,lr=cfg['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0.)
    traces=[];updates=0
    try:
        for k in range(cfg['steps']):
            boxes=rp.values()['boxes']
            entry=dict(step=k,boxes=boxes.detach().cpu().clone(),state={n:v.cpu().clone() for n,v in rp.state().items()}) if trace else None
            if not positions:
                if trace:traces.append(dict(**entry,round=dict(updated=False,no_information=True)))
                continue
            mu,boundary=proposed_coordinates(boxes[positions],rp.native_raw[positions])
            rr=rollout(mu.detach(),evidence,key,k,False,config=cfg)
            loss=likelihood_loss(mu,rr['mean'],rr['samples'],rr['weights'],sigma=cfg['sigma'])
            gmu=torch.autograd.grad(loss,mu,retain_graph=True)[0] if trace else None
            opt.zero_grad(set_to_none=True)
            update=None
            if rr['informative']:
                gradients=torch.autograd.grad(loss,params)
                assert all(torch.isfinite(g).all() for g in gradients)
                previous=flat(params).detach().clone() if trace else None
                for p,g in zip(params,gradients):p.grad=g.detach()
                opt.step();updates+=1
                assert all(torch.isfinite(p).all() for p in params)
                if trace:update=dict(gradient=flat(gradients).detach().cpu(),raw=(flat(params).detach()-previous).cpu(),names=names)
            if trace:
                entry['round']=dict(updated=rr['informative'],no_information=not rr['informative'],
                    loss=float(loss.detach()),rollout={n:(v.detach().cpu().clone() if torch.is_tensor(v) else v) for n,v in rr.items()},
                    autograd_mean_gradient=gmu.detach().cpu(),mean_before=mu.detach().cpu())
                if update is not None:entry['update']=update
                traces.append(entry)
        finalstate={n:v.cpu().clone() for n,v in rp.state().items()}
        with torch.no_grad():final=rp.values()['boxes'].detach().cpu().clone()
        initialcpu={n:v.cpu().clone() for n,v in origin.items()}
        if not positions:
            assert torch.equal(final,before.cpu()) and all(torch.equal(finalstate[n],initialcpu[n]) for n in finalstate)
        return dict(initial=initialcpu,state=finalstate,before=before.cpu(),final=final,
                    positions=positions,gradient_calls=updates,selected_step=cfg['steps'],
                    active_parameters=1792,GT_used=False,config=cfg,trace=traces,
                    empty=not positions,original_scientific_operations=True,
                    complete_round_trace_enabled=trace,independent_CPU_audit_in_timed_path=False)
    finally:rp.restore(origin)


def compare_complete_trace(new,old):
    """Actual full saved path and gradient comparisons, not a Jacobian proof."""
    assert new['complete_round_trace_enabled'] and len(new['trace'])==old['config']['steps']
    count=0
    def same(a,b):
        nonlocal count
        if torch.is_tensor(a):
            assert torch.equal(a,b);count+=a.numel()
        elif isinstance(a,dict):
            assert set(a)==set(b)
            for k in a:same(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b)
            for x,y in zip(a,b):same(x,y)
        else:assert a==b
    for k in ['initial','state','before','final','positions','gradient_calls','selected_step','config','empty']:
        same(new[k],old[k])
    for k,t in enumerate(new['trace']):
        same(t['boxes'],old['path'][k]['boxes']);same(t['state'],old['path'][k]['state'])
        r=old['rounds'][k]
        if not old['positions']:
            same(t['round'],dict(updated=r['updated'],no_information=r['no_information']))
        else:
            for field in ['updated','no_information','loss','rollout','autograd_mean_gradient','mean_before']:
                same(t['round'][field],r[field])
            if r['updated']:
                for field in ['gradient','raw','names']:same(t['update'][field],old['path'][k]['update'][field])
    return dict(status='pass',actual_complete_rounds=len(new['trace']),tensor_coordinates=count,
                independent_full_decoder_Jacobian=False)


def compare_output(new,qualified):
    coordinates=0
    for field in ['initial','state','before','final']:
        a,b=new[field],qualified[field]
        if isinstance(a,dict):
            assert set(a)==set(b)
            for n in a:assert torch.equal(a[n],b[n]);coordinates+=a[n].numel()
        else:assert torch.equal(a,b);coordinates+=a.numel()
    for f in ['positions','gradient_calls','selected_step','active_parameters','GT_used','config','empty']:
        assert new[f]==qualified[f]
    return dict(status='pass',output_state_coordinates=coordinates)


def cpu_contracts():
    good=0;rejected=0
    for steps in [10,40]:
        for alpha in [0.,1/16,1/8,1.]:
            check_config(dict(CONFIG,steps=steps,writeback=alpha));good+=1
    for field,value in [('samples',16),('steps',True),('steps',0),('sigma',0),('tau',float('nan')),
                        ('lr',float('inf')),('writeback',True),('writeback',1.1)]:
        try:check_config(dict(CONFIG,**{field:value}))
        except ValueError:rejected+=1
        else:raise AssertionError('Invalid deployment config accepted')
    return dict(status='pass',scope='CPU fixed-config and invalid-input contracts only; GPU qualification still required',
                valid_contracts=good,invalid_contracts_rejected=rejected,GT_read=False,actual_GPU_fits=0)
