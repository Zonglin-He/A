"""R7 single-sample, frozen-feature final-linear-regression adapter.
No dataset/checkpoint/model constructor, G implementation, teacher or yaw output.
Caller supplies audited source model + GT-free P/text inputs and detached G target.
"""
from dataclasses import dataclass
import hashlib, random, re
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn.functional as F

PARAMETER_NAMES = tuple('prediction_heads.5.'+head+'.net.8.'+suffix
    for head in ('center_residual_head','size_pred_head') for suffix in ('weight','bias'))
LEXICON = ('car','vehicle','sedan','van','truck','bus','pedestrian','person','man','woman','people','child','boy','girl','cyclist','biker','bicycle','bike','taxi','suv','jeep','trailer','walker','worker')

@dataclass(frozen=True)
class StepConfig:
    steps: int
    lr: float
    beta_metres: float
    def validate(self):
        assert isinstance(self.steps,int) and self.steps>=0
        assert np.isfinite(self.lr) and self.lr>0
        assert np.isfinite(self.beta_metres) and self.beta_metres>0

@dataclass(frozen=True)
class SourceCache:
    base_xyz: torch.Tensor             # [1,Q,3], source model coordinates
    center_features: torch.Tensor      # [1,288,Q], input to final center Conv
    size_features: torch.Tensor        # [1,288,Q], input to final size Conv
    boxes_raw6: torch.Tensor           # [1,Q,6], no size clamp/no yaw
    text_scores: torch.Tensor          # [1,Q], immutable pure-text BBF readout
    query_index: int                   # fixed source selection, no reselection
    selected_parameter_sha256: str

@dataclass(frozen=True)
class PseudoTarget:
    query_index: int
    target6: torch.Tensor              # absolute [cx,cy,cz,sx,sy,sz], same frame/metres
    reliability6: torch.Tensor         # [6] in [0,1], zero means unobserved


def layers(model):
    assert len(model.prediction_heads)==6
    h=model.prediction_heads[5]
    assert not h.heading
    a=h.center_residual_head.net[8];b=h.size_pred_head.net[8]
    for layer in (a,b):
        assert isinstance(layer,torch.nn.Conv1d)
        assert layer.in_channels==288 and layer.out_channels==3 and layer.kernel_size==(1,) and layer.groups==1 and layer.bias is not None
    names=dict(model.named_parameters());assert all(n in names for n in PARAMETER_NAMES)
    assert sum(names[n].numel() for n in PARAMETER_NAMES)==1734
    return a,b


def parameter_sha(model):
    params=dict(model.named_parameters());h=hashlib.sha256()
    for n in PARAMETER_NAMES:
        x=params[n].detach().cpu().contiguous();h.update(n.encode()+b'\0');h.update(x.numpy().tobytes())
    return h.hexdigest()


@contextmanager
def episode(model):
    """Restore selected values, ALL gradients/requires_grad/modes/buffers and RNG.
    Nonselected parameters are never assigned or optimized; version checks detect
    unauthorized in-place mutation. No CUDA initialization on a CPU-only call.
    """
    params=dict(model.named_parameters());selected=set(PARAMETER_NAMES);layers(model)
    original_values={n:params[n].detach().clone() for n in selected}
    flags={n:p.requires_grad for n,p in params.items()};grads={n:p.grad for n,p in params.items()}
    versions={n:p._version for n,p in params.items() if n not in selected}
    modes=[(m,m.training) for m in model.modules()]
    buffers={n:b.detach().clone() for n,b in model.named_buffers()}
    py=random.getstate();npstate=np.random.get_state();cpu=torch.get_rng_state()
    cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        model.eval()
        for n,p in params.items():p.requires_grad_(n in selected);p.grad=None
        yield
        assert all(params[n]._version==v for n,v in versions.items()),'Frozen parameter mutated'
    finally:
        with torch.no_grad():
            for n,x in original_values.items():params[n].copy_(x)
            for n,b in model.named_buffers():b.copy_(buffers[n])
        for n,p in params.items():p.grad=grads[n];p.requires_grad_(flags[n])
        for m,training in modes:m.training=training
        random.setstate(py);np.random.set_state(npstate);torch.set_rng_state(cpu)
        if cuda is not None:torch.cuda.set_rng_state_all(cuda)


def first_noun_mask(text,offset_mapping):
    """GT-free first lexicon noun. No matched noun -> explicit unsupported input.
    Offset mapping must be from the exact model input including any sentinel.
    All-token softmax denominator is retained separately by text_bbf_scores.
    """
    m=re.search(r'\b(?:'+'|'.join(LEXICON)+r')\b',text,re.I)
    if m is None:raise ValueError('No lexicon noun; no GT fallback permitted')
    off=torch.as_tensor(offset_mapping)
    assert off.ndim==2 and off.shape[1]==2
    mask=(off[:,1]>m.start())&(off[:,0]<m.end())
    assert mask.any(),'Noun outside actual model tokenization'
    return mask


def text_bbf_scores(outputs,token_mask):
    q=outputs['last_proj_queries'];t=outputs['proj_tokens']
    assert q.ndim==t.ndim==3 and q.shape[0]==t.shape[0]==1 and q.shape[-1]==t.shape[-1]==64
    mask=torch.as_tensor(token_mask,device=q.device,dtype=torch.bool)
    assert mask.shape==(t.shape[1],) and mask.any()
    # Preserve specials/sentinel/padding in actual T-token denominator.
    return (((q@t.transpose(-1,-2))/.07).softmax(-1)*mask[None,None]).sum(-1)


def raw_boxes(model,cache):
    a,b=layers(model)
    return torch.cat((cache.base_xyz+a(cache.center_features).transpose(1,2),b(cache.size_features).transpose(1,2)),-1)


def capture_source(model,model_inputs,token_mask,query_index=None):
    """One no-grad eval source forward. Caller owns source-checkpoint/inputs lock.
    No labels accepted. RGB may inform external G but is not sent into 3EED here.
    """
    assert set(model_inputs)=={'point_clouds','text'}
    assert model_inputs['point_clouds'].ndim==3 and model_inputs['point_clouds'].shape[0]==1 and model_inputs['point_clouds'].shape[2]==3
    assert len(model_inputs['text'])==1
    a,b=layers(model);features={}
    def hook(name):
        def f(module,args):
            assert name not in features;features[name]=args[0].detach().clone()
        return f
    with episode(model):
        handles=[a.register_forward_pre_hook(hook('center')),b.register_forward_pre_hook(hook('size'))]
        try:
            with torch.no_grad():out=model(model_inputs)
        finally:
            for h in handles:h.remove()
        boxes=torch.cat((out['last_center'],out['last_pred_size']),-1).detach().clone()
        scores=text_bbf_scores(out,token_mask).detach().clone()
        fixed_q=int(scores.argmax(-1).item()) if query_index is None else int(query_index)
        cache=SourceCache(out['last_base_xyz'].detach().clone(),features['center'],features['size'],boxes,scores,fixed_q,parameter_sha(model))
        assert 0<=fixed_q<boxes.shape[1]
        with torch.no_grad():replay=raw_boxes(model,cache)
        assert torch.equal(replay,boxes),'Final Conv frozen replay is not bitexact to this source forward'
    return cache


def weighted_smooth_l1(box,target,w,beta):
    return (F.smooth_l1_loss(box,target,beta=beta,reduction='none')*w).sum()/w.sum()


def box_gradient(box,target,w,beta):
    """Analytic six-dimensional gradient for nonzero reliability sum."""
    return ((box-target)/beta).clamp(-1,1)*w/w.sum()


def adapt(model,cache,pseudo,config):
    """Returns four arms; model restored before return or any exception.
    SGD is plain, no momentum/decay/clip. Configuration must be explicitly locked
    by caller; this module chooses no experimental lr/beta/step budget.
    """
    config.validate();a,b=layers(model)
    assert pseudo.query_index==cache.query_index and parameter_sha(model)==cache.selected_parameter_sha256
    source=cache.boxes_raw6.detach().clone();q=cache.query_index
    assert source.ndim==3 and source.shape[0]==1 and source.shape[-1]==6 and 0<=q<source.shape[1]
    target=torch.as_tensor(pseudo.target6,device=source.device,dtype=source.dtype).detach().clone()
    w=torch.as_tensor(pseudo.reliability6,device=source.device,dtype=source.dtype).detach().clone()
    assert target.shape==w.shape==(6,) and torch.isfinite(target).all() and torch.isfinite(w).all()
    assert ((w>=0)&(w<=1)).all() and (target[3:]>0).all()
    assert not cache.center_features.requires_grad and not cache.size_features.requires_grad and not cache.base_xyz.requires_grad
    direct=source.clone();direct[0,q]=torch.where(w>0,target,source[0,q])
    output_control=source.clone();trace=[]
    hc=cache.center_features[0,:,q];hs=cache.size_features[0,:,q]
    kc=cache.center_features[0].transpose(0,1)@hc+1
    ks=cache.size_features[0].transpose(0,1)@hs+1
    norms={'center_squared_feature_norm':float(hc.square().sum()),'size_squared_feature_norm':float(hs.square().sum()),'center_self_kernel_with_bias':float(kc[q]),'size_self_kernel_with_bias':float(ks[q])}
    with episode(model):
        with torch.no_grad():assert torch.equal(raw_boxes(model,cache),source),'Stale source cache'
        selected=[dict(model.named_parameters())[n] for n in PARAMETER_NAMES]
        opt=torch.optim.SGD(selected,lr=config.lr,momentum=0,weight_decay=0)
        initial={n:dict(model.named_parameters())[n].detach().clone() for n in PARAMETER_NAMES}
        for step in range(config.steps if float(w.sum())>0 else 0):
            opt.zero_grad(set_to_none=True);pred=raw_boxes(model,cache);current=pred[0,q]
            g=box_gradient(current.detach(),target,w,config.beta_metres)
            loss=weighted_smooth_l1(current,target,w,config.beta_metres)
            assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in selected)
            grad_norm=float(torch.sqrt(sum(p.grad.square().sum() for p in selected)))
            # Exact in real arithmetic: linear SGD induces a feature-Gram kernel
            # box-space step. This independent no-parameter arm keeps all Q boxes.
            gc=box_gradient(output_control[0,q],target,w,config.beta_metres)
            output_control[0,:,:3]-=config.lr*kc[:,None]*gc[:3]
            output_control[0,:,3:]-=config.lr*ks[:,None]*gc[3:]
            opt.step()
            with torch.no_grad():after=raw_boxes(model,cache).detach().clone()
            delta={n:float((dict(model.named_parameters())[n]-v).abs().max()) for n,v in initial.items()}
            trace.append({'step':step+1,'loss':float(loss.detach()),'selected_box_before':current.detach().cpu().tolist(),'selected_box_after':after[0,q].cpu().tolist(),'weighted_SmoothL1_box_gradient6':g.cpu().tolist(),'parameter_gradient_norm':grad_norm,'nonzero_gradient_parameter_tensors':sum(int(torch.count_nonzero(p.grad)>0) for p in selected),'clip_enabled':False,'clip_factor':1.,'parameter_delta_max_by_name':delta,'output_space_replay_max_abs_error':float((after-output_control).abs().max())})
        with torch.no_grad():updated=raw_boxes(model,cache).detach().clone()
        difference=(updated-source).abs();other=torch.ones(source.shape[1],dtype=torch.bool,device=source.device);other[q]=False
        spill={'other_query_count_changed':int((difference[0,other].amax(-1)>0).sum()),'other_query_max_abs_center_change':float(difference[0,other,:3].max()) if other.any() else 0.,'other_query_max_abs_size_change':float(difference[0,other,3:].max()) if other.any() else 0.}
    assert parameter_sha(model)==cache.selected_parameter_sha256
    return {'query_index':q,'source':source,'direct_G':direct,'parameter_SGD':updated,'exact_output_space_SGD':output_control.detach(),'trace':trace,'feature_norms':norms,'spillover':spill,'all_weights_zero':float(w.sum())==0,'yaw':0.,'readout_fixed_source_text_scores':cache.text_scores.detach().clone(),'restored_selected_parameters':True,'interpretation':'Current-input parameter SGD is a feature-kernel box-space step, not evidence of a unique adaptation benefit.'}
