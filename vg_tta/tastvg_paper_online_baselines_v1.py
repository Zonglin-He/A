"""Persistent TENT/SAR/ViTTA optimization cores for the STVG port.

No target labels or specialists. Call reset only at stream boundaries. Backbone
binding and GPU smoke qualification are separate; this module alone is not a
completed benchmark. The existing episodic implementations remain unchanged.
"""
import copy
import torch
from vg_tta.native_probability_interface_v1 import offsets_entropy,entropy_ceiling
from vg_tta.vitta_paper_v1 import feature_statistics,alignment_loss,temporal_consistency,validate_source

class OnlineOptimizer:
    def __init__(self,parameters,method,config=None,source=None):
        if method not in ('TENT','SAR','ViTTA'):raise ValueError(method)
        self.params=list(parameters)
        if not self.params or len({id(p) for p in self.params})!=len(self.params):raise ValueError('Invalid parameter scope')
        self.method=method;self.initial=[p.detach().clone() for p in self.params]
        self.flags=[p.requires_grad for p in self.params]
        defaults={'TENT':dict(lr=.001,steps=1),'SAR':dict(lr=.001,steps=3,rho=.05,margin_fraction=.4,reset_entropy=.2),'ViTTA':dict(lr=1e-5,steps=1,momentum=.9,weight_decay=5e-4,consistency_weight=.1,momentum_mvg=.1)}
        self.cfg={**defaults[method],**(config or {})}
        if set(self.cfg)!=set(defaults[method]) or any(not torch.isfinite(torch.tensor(float(v))) or v<0 for v in self.cfg.values()):raise ValueError('Invalid configuration')
        if int(self.cfg['steps'])!=self.cfg['steps']:raise ValueError('Integer steps required')
        if method=='ViTTA' and not 0<self.cfg['momentum_mvg']<=1:raise ValueError('EMA rate')
        self.source=validate_source(source) if method=='ViTTA' else None
        self.optimizer=(torch.optim.Adam(self.params,lr=self.cfg['lr']) if method=='TENT' else torch.optim.SGD(self.params,lr=self.cfg['lr'],momentum=.9,weight_decay=self.cfg.get('weight_decay',0.)))
        self.opt_initial=copy.deepcopy(self.optimizer.state_dict());self.target=None;self.entropy_ema=None;self.arrivals=0

    def _copy(self,values):
        with torch.no_grad():
            for p,v in zip(self.params,values):p.copy_(v)

    def reset(self):
        self._copy(self.initial);self.optimizer.load_state_dict(copy.deepcopy(self.opt_initial))
        self.target=None;self.entropy_ema=None;self.arrivals=0
        for p,flag in zip(self.params,self.flags):p.grad=None;p.requires_grad_(flag)

    def state_dict(self):
        return dict(method=self.method,config=self.cfg,parameters=[p.detach().clone() for p in self.params],optimizer=copy.deepcopy(self.optimizer.state_dict()),target=copy.deepcopy(self.target),entropy_ema=self.entropy_ema,arrivals=self.arrivals)

    def load_state_dict(self,state):
        assert state['method']==self.method and state['config']==self.cfg
        assert len(state['parameters'])==len(self.params)
        self._copy(state['parameters']);self.optimizer.load_state_dict(state['optimizer'])
        self.target=copy.deepcopy(state['target']);self.entropy_ema=state['entropy_ema'];self.arrivals=state['arrivals']

    @torch.enable_grad()
    def arrive(self,closure,inference,full_ids=None):
        """closure: native offsets (TENT/SAR) or two live View objects (ViTTA).

        inference runs with original requires_grad flags and no_grad, retaining
        current adapted values. Return both before/after predictions explicitly.
        """
        before=[p.detach().clone() for p in self.params];trace=[]
        def predict():
            for p,f in zip(self.params,self.flags):p.requires_grad_(f)
            with torch.no_grad():result=inference()
            return result
        pre=predict()
        try:
            for p in self.params:p.requires_grad_(True)
            for _ in range(self.cfg['steps']):
                self.optimizer.zero_grad(set_to_none=True);native=closure()
                if self.method=='ViTTA':
                    current=feature_statistics(native);rate=self.cfg['momentum_mvg']
                    target=current if self.target is None else {k:{q:(1-rate)*self.target[k][q].to(v[q])+rate*v[q] for q in v} for k,v in current.items()}
                    loss=alignment_loss(target,self.source)+self.cfg['consistency_weight']*temporal_consistency(native,full_ids)
                    self.target={k:{q:z.detach().clone() for q,z in v.items()} for k,v in target.items()}
                else:loss=offsets_entropy(native)
                if not torch.isfinite(loss) or not loss.requires_grad:raise RuntimeError('Invalid live loss')
                rec=dict(loss=float(loss.detach()),updated=False)
                if self.method=='SAR':
                    margin=self.cfg['margin_fraction']*sum(entropy_ceiling(x) for x in native)/len(native)
                    rec['margin']=margin
                    if loss.detach()>=margin:trace.append(rec);continue
                loss.backward();grad=[p.grad for p in self.params if p.grad is not None]
                if not grad or not all(torch.isfinite(g).all() for g in grad):raise RuntimeError('Invalid gradients')
                if self.method=='SAR':
                    base=[p.detach().clone() for p in self.params];norm=torch.sqrt(sum(g.square().sum() for g in grad))
                    try:
                        with torch.no_grad():
                            for p in self.params:
                                if p.grad is not None:p.add_(p.grad*self.cfg['rho']/(norm+1e-12))
                        self.optimizer.zero_grad(set_to_none=True);second=offsets_entropy(closure())
                        if not torch.isfinite(second):raise RuntimeError('Nonfinite SAR perturbed loss')
                        rec['second_loss']=float(second.detach())
                        if second.detach()<margin:
                            second.backward();self.entropy_ema=rec['second_loss'] if self.entropy_ema is None else .9*self.entropy_ema+.1*rec['second_loss']
                        else:trace.append(rec);continue
                    finally:self._copy(base)
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in self.params):raise RuntimeError('Nonfinite gradients')
                self.optimizer.step();rec['updated']=True
                if self.method=='SAR' and self.entropy_ema<self.cfg['reset_entropy']:
                    self._copy(self.initial);self.optimizer.load_state_dict(copy.deepcopy(self.opt_initial));self.entropy_ema=None;rec['recovered']=True
                if not all(torch.isfinite(p).all() for p in self.params):raise RuntimeError('Nonfinite parameters')
                trace.append(rec)
            post=predict();self.arrivals+=1
            return pre,post,dict(method=self.method,arrival=self.arrivals,trace=trace,persistent=True,GT_online=False,expert_calls=0,parameter_delta=float(torch.sqrt(sum((p.detach()-v).double().square().sum() for p,v in zip(self.params,before)))))
        finally:
            for p,f in zip(self.params,self.flags):p.requires_grad_(f);p.grad=None
