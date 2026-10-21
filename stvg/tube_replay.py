"""One cached TA time-decoder replay with per-head spatial reading priors.

The actual native224 TimeDecoder kwargs are the cache boundary. Its memory is
the original text+motion memory, never an appearance ROI substitute. Only the
last H*W cross-attention probabilities are redistributed within each legal
per-frame visual group. Text probabilities and the visual group's mass remain.
No encoder, expert, spatial decoder, optimizer or model training is used here.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
import inspect
import textwrap
import time
import types

import torch
import torch.nn.functional as F

from .tensors import detached

SCHEMA='ta_native224_time_decoder_inputs_v1'
TIME_FORWARD_AST='ba35c70f4cb2b9b0727776848d09c1d5e68296bdea23fb16110c3002099e1a37'
LAYER_FORWARD_AST='f675458e2dbfd0043a222b4ffd9656915da285b7d06834456a721352f88cba87'
INPUT_KEYS={'query_tgt','query_content','query_time','query_mask','encoded_feature','encoded_pos','encoded_mask'}


def _require(condition,message):
    if not condition:raise ValueError(message)


def _ast(function):
    tree=ast.parse(textwrap.dedent(inspect.getsource(function)))
    return hashlib.sha256(ast.dump(tree.body[0],include_attributes=False).encode()).hexdigest()


def verify_decoder(decoder):
    _require(_ast(type(decoder).forward)==TIME_FORWARD_AST,'Unsupported native TimeDecoder forward')
    _require(len(decoder.layers)>0,'Empty temporal decoder')
    for layer in decoder.layers:
        _require(_ast(type(layer).forward)==LAYER_FORWARD_AST,'Unsupported temporal layer forward')
        a=layer.cross_attn_image
        _require(type(a)is torch.nn.MultiheadAttention and not a.batch_first,
                 'Expected the actual native per-frame MultiheadAttention')
        _require(a._qkv_same_embed_dim and a.bias_k is None and a.bias_v is None and not a.add_zero_attn,
                 'Unsupported attention token layout')
    _require(not decoder.training and not any(p.requires_grad for p in decoder.parameters()),
             'Temporal replay requires frozen eval decoder weights')
    return dict(time_decoder=TIME_FORWARD_AST,time_layer=LAYER_FORWARD_AST,layers=len(decoder.layers))


def _check_inputs(kwargs,grid,frames):
    _require(set(kwargs)==INPUT_KEYS,'Cache must contain the actual seven TimeDecoder kwargs')
    h,w=map(int,grid);_require(h>0 and w>0,'Positive visual token grid required')
    q=kwargs['query_tgt'];memory=kwargs['encoded_feature'];S,N,C=memory.shape
    _require(memory.dtype==torch.float32 and N==frames and S>h*w,'Wrong native memory/frame layout')
    _require(q.shape==(N,1,C) and kwargs['query_content'].shape==q.shape and kwargs['query_time'].shape==q.shape,
             'Wrong temporal query/content/time layout')
    _require(kwargs['encoded_pos'].shape==memory.shape,'Position and memory layout differ')
    _require(kwargs['encoded_mask'].shape==(N,S) and kwargs['encoded_mask'].dtype==torch.bool,
             'Preserve the original encoded padding mask exactly')
    _require(kwargs['query_mask'].shape==(1,N) and kwargs['query_mask'].dtype==torch.bool,'Wrong query padding mask')
    for value in kwargs.values():
        _require(torch.is_tensor(value),'Decoder inputs must be tensors')
        if value.is_floating_point():_require(bool(torch.isfinite(value).all()),'Nonfinite cached input')
    return h,w


def cache_temporal(model,base,records,frame_ids,source_identity=None):
    """Export native second-pass inputs already captured by FrozenNativeReplay.

    ``base.time_inputs`` is populated by hooks at the actual TimeDecoder call;
    ``temporal_inputs`` is the corresponding full-layer hidden output. No
    queries or memory are reconstructed from a spatial cache or class scores.
    """
    decoder=model.ground_decoder.time_decoder;interface=verify_decoder(decoder)
    _require(len(records)==len(base.time_inputs)==len(base.temporal_inputs)==2,'Expected the two original offsets')
    _require(hasattr(base,'time_input_layouts') and len(base.time_input_layouts)==2,
             'Actual input layouts must be captured before cloning the native kwargs')
    ids=[int(x)for x in frame_ids]
    _require(len(ids)>=4 and all(b>a for a,b in zip(ids,ids[1:])),'Physical frame IDs must increase')
    _require(not model.temp_embed.training and not any(p.requires_grad for p in model.temp_embed.parameters()),
             'Frozen temporal head required')
    offsets=[]
    for offset,(record,kwargs,hidden)in enumerate(zip(records,base.time_inputs,base.temporal_inputs)):
        _require(record['offset']==offset and not record.get('flip',False),'Unexpected offset/flip routing')
        physical=[int(x)for x in record['frame_ids']]
        _require(physical==ids[offset::2],'Physical offset grid mismatch')
        grid=tuple(map(int,base.views[offset]['info']['fea_map_size']))
        _check_inputs(kwargs,grid,len(physical))
        layouts=base.time_input_layouts[offset]
        _require(set(layouts)==INPUT_KEYS,'Missing actual native tensor layout')
        for key,value in kwargs.items():
            _require(tuple(layouts[key]['shape'])==tuple(value.shape),'Captured tensor layout shape changed')
        _require(hidden.ndim==4 and hidden.shape[0]==len(decoder.layers) and hidden.shape[1:3]==(1,len(physical)),
                 'Full native temporal hidden batch shape changed')
        logits=base.zero['logits'][offset]
        _require(logits.shape==(1,len(physical),2),'Unexpected native temporal logits')
        offsets.append(dict(offset=offset,frame_ids=physical,frame_positions=list(range(offset,len(ids),2)),
            grid_size=list(grid),inputs=detached(kwargs,'cpu'),input_layouts=layouts,hidden=detached(hidden,'cpu'),
            native_logits=detached(logits,'cpu')))
    return dict(schema=SCHEMA,frame_ids=ids,offsets=offsets,interface=interface,
        source_identity=source_identity,GT_read=False,resolution=224,
        memory_layout='original text prefix then motion visual suffix; actual padding order retained',
        capture='actual final native224 FP32 TimeDecoder boundary; no reconstructed memory')


def occupancy_prior(boxes,grid_size,legal_visual,*,permuted=False):
    """Continuous cell occupancy for fixed 1.5x context; output boxes unchanged.

    The negative control cyclically permutes only legal visual locations by
    floor(nlegal/2), at least one. It preserves every legal M value, its area
    sum and the padding layout separately in every physical frame.
    """
    boxes=torch.as_tensor(boxes,dtype=torch.float64,device='cpu')
    legal=torch.as_tensor(legal_visual,dtype=torch.bool,device='cpu')
    h,w=map(int,grid_size);N=len(boxes)
    _require(boxes.shape==(N,4) and legal.shape==(N,h*w),'Tube/grid shape mismatch')
    _require(bool(torch.isfinite(boxes).all()) and bool((boxes[:,2:]>0).all()),'Finite positive-size boxes required')
    lower=boxes[:,:2]-.75*boxes[:,2:];upper=boxes[:,:2]+.75*boxes[:,2:]
    x=torch.arange(w,dtype=torch.float64)/w;y=torch.arange(h,dtype=torch.float64)/h
    dx=(torch.minimum(upper[:,0,None],x[None]+1/w)-torch.maximum(lower[:,0,None],x[None])).clamp_min(0)*w
    dy=(torch.minimum(upper[:,1,None],y[None]+1/h)-torch.maximum(lower[:,1,None],y[None])).clamp_min(0)*h
    M=(dy[:,:,None]*dx[:,None,:]).reshape(N,h*w).clamp(0,1)
    M=M.masked_fill(~legal,0)
    shifts=[]
    if permuted:
        M=M.clone()
        for t in range(N):
            selected=torch.nonzero(legal[t],as_tuple=True)[0];n=len(selected)
            shift=max(1,n//2)if n else 0
            if n:M[t,selected]=M[t,selected].roll(shift)
            shifts.append(shift%n if n else 0)
    else:shifts=[0]*N
    return M,dict(context_scale=1.5,permuted=bool(permuted),legal_counts=legal.sum(-1).tolist(),
        shifts=shifts,legal_area=M.sum(-1).tolist(),permutation='per-frame legal-cell cyclic floor(n/2), minimum1')


def redistribute_attention(A,M,reliability,legal_visual,heads,alpha=.5):
    """Redistribute the real [frames*heads,queries,keys] post-softmax tensor."""
    _require(A.ndim==3 and A.shape[0]%heads==0,'Unexpected per-head probability layout')
    N=A.shape[0]//heads;V=M.shape[-1]
    _require(M.shape==(N,V) and reliability.shape==(N,) and legal_visual.shape==(N,V),
             'Attention prior/frame layout mismatch')
    _require(A.shape[-1]>V,'Text prefix must remain outside visual redistribution')
    if alpha==0:return A
    visual=A[...,-V:].reshape(N,heads,A.shape[1],V)
    valid=legal_visual[:,None,None,:]
    w=1-float(alpha)*reliability[:,None]*(1-M)
    local=torch.where(valid,visual,torch.zeros_like(visual))
    original_mass=local.sum(-1,keepdim=True)
    weighted=local*w[:,None,None,:];denominator=weighted.sum(-1,keepdim=True)
    safe=torch.where(denominator>0,denominator,torch.ones_like(denominator))
    updated=weighted*(original_mass/safe)
    result=A.clone()
    result[...,-V:]=torch.where(valid,updated,visual).reshape_as(A[...,-V:])
    return result


def _clone(function,namespace):
    result=types.FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=function.__kwdefaults__;result.__annotations__=function.__annotations__
    return result


class _HeadRedistributor:
    """Local function-global injection; no process/global torch monkeypatch.

    Native MultiheadAttention and functional bytecode are retained. Only its
    local `softmax` binding is wrapped, before dropout and value aggregation.
    Alpha zero does not install or call this adapter at all.
    """
    def __init__(self,module,M,r,legal,alpha,audits):
        self.module=module;self.M=M;self.r=r;self.legal=legal;self.alpha=alpha;self.audits=audits
        self.softmax_calls=0
        functional=F.multi_head_attention_forward
        source=inspect.getsource(functional)
        _require(source.count('attn_output_weights = softmax(attn_output_weights, dim=-1)')==1,
                 'Unsupported PyTorch attention implementation')
        namespace=dict(functional.__globals__)
        original_softmax=namespace['softmax']
        def redirected_softmax(*args,**kwargs):
            self.softmax_calls+=1
            _require(self.softmax_calls==1,'Attention performed an unexpected extra softmax')
            A=original_softmax(*args,**kwargs)
            new=redistribute_attention(A,self.M.to(A),self.r.to(A),self.legal.to(A.device),module.num_heads,self.alpha)
            V=self.M.shape[-1];N=len(self.r);Q=A.shape[1]
            valid=self.legal.to(A.device)[:,None,None,:]
            before=A[...,-V:].reshape(N,module.num_heads,Q,V)
            after=new[...,-V:].reshape_as(before)
            mass_before=torch.where(valid,before,torch.zeros_like(before)).sum(-1)
            mass_after=torch.where(valid,after,torch.zeros_like(after)).sum(-1)
            self.audits.append(dict(visual_mass_max_error=(mass_after-mass_before).abs().amax().detach(),
                text_max_error=(new[...,:-V]-A[...,:-V]).abs().amax().detach(),
                padding_max_error=torch.where(valid,torch.zeros_like(after),after-before).abs().amax().detach(),
                heads=module.num_heads,queries=Q,frames=N))
            return new
        namespace['softmax']=redirected_softmax
        modified=_clone(functional,namespace)
        namespace[functional.__name__]=modified
        # The original MHA.forward uses its local F alias; its other helpers
        # remain the real unchanged torch.nn.functional functions.
        proxy=types.SimpleNamespace(**vars(F));proxy.multi_head_attention_forward=modified
        forward=type(module).forward;forward_globals=dict(forward.__globals__);forward_globals['F']=proxy
        self.forward=_clone(forward,forward_globals)

    def __call__(self,*args,**kwargs):
        _require(kwargs.get('need_weights',True),'Native time attention requires the weighted path')
        output=self.forward(self.module,*args,**kwargs)
        _require(self.softmax_calls==1,'Native per-head attention path was bypassed')
        self.softmax_calls=0
        return output


@contextmanager
def guided_attention(decoder,M,reliability,legal_visual,alpha=.5):
    _require(float(alpha)in(0.,.5),'Only locked alpha0 recovery or alpha.5 research is supported')
    if alpha==0:
        yield []
        return
    original=[];audits=[]
    try:
        for layer in decoder.layers:
            module=layer.cross_attn_image
            _require('forward'not in module.__dict__,'Attention already has an instance forward override')
            module.forward=_HeadRedistributor(module,M,reliability,legal_visual,.5,audits)
            original.append(module)
        yield audits
    finally:
        for module in original:del module.forward


def _synchronize(device):
    if torch.device(device).type=='cuda':torch.cuda.synchronize(device)


def _to(value,device):
    if torch.is_tensor(value):return value.to(device)
    if isinstance(value,dict):return {k:_to(v,device)for k,v in value.items()}
    return value


def restore_layout(value,layout,device):
    """Restore the original layout directly on the final execution device.

    CPU .clone() and a subsequent .to(cuda) can materialize an expanded query.
    Copy only unique broadcast values into the final-device backing, then view
    it with the captured shape, stride and storage offset. No arithmetic or
    dtype conversion is performed. Unsupported overlap is rejected explicitly.
    """
    shape=tuple(map(int,layout['shape']));stride=tuple(map(int,layout['stride']))
    offset=int(layout['storage_offset'])
    _require(tuple(value.shape)==shape and len(shape)==len(stride) and offset>=0
             and all(n>0 for n in shape) and all(s>=0 for s in stride),'Invalid actual tensor layout')
    collapsed=tuple(1 if s==0 else n for n,s in zip(shape,stride))
    span=1
    for s,n in sorted((s,n)for s,n in zip(stride,collapsed)if n>1):
        _require(s>=span,'Unsupported non-broadcast internal overlap')
        span+=(n-1)*s
    storage_size=offset+1+sum((n-1)*s for n,s in zip(shape,stride))
    backing=torch.empty(storage_size,dtype=value.dtype,device=device)
    unique=torch.as_strided(backing,collapsed,stride,offset)
    index=tuple(slice(0,1)if s==0 else slice(None)for s in stride)
    source=value[index]
    if any(s==0 and n>1 for n,s in zip(shape,stride)):
        _require(torch.equal(source.expand(shape),value),'Broadcast layout disagrees with cached values')
    unique.copy_(source.to(device=device))
    restored=torch.as_strided(backing,shape,stride,offset)
    _require(restored.stride()==stride and restored.storage_offset()==offset,'Failed to restore native tensor layout')
    return restored


def restore_inputs(inputs,layouts,device):
    _require(set(inputs)==set(layouts)==INPUT_KEYS,'Missing native input layout metadata')
    return {key:restore_layout(value,layouts[key],device)for key,value in inputs.items()}


@torch.no_grad()
def _replay_on_device(model,cache,boxes,reliability,*,alpha=.5,permuted=False):
    """One complete cached time subgraph replay per original offset (two calls)."""
    _require(cache['schema']==SCHEMA and cache['resolution']==224 and cache['GT_read']is False,'Wrong native cache')
    _require(float(alpha)in(0.,.5),'The research strength is fixed at alpha=.5')
    decoder=model.ground_decoder.time_decoder;verify_decoder(decoder)
    head=model.temp_embed;device=next(decoder.parameters()).device
    _require(not head.training and not any(p.requires_grad for p in head.parameters()),'Frozen temporal head required')
    _require(next(head.parameters()).device==device,'Decoder/head devices differ')
    boxes=torch.as_tensor(boxes,dtype=torch.float64,device='cpu')
    r=torch.as_tensor(reliability,dtype=torch.float64,device='cpu')
    N=len(cache['frame_ids']);_require(boxes.shape==(N,4) and r.shape==(N,),'Full physical grid mismatch')
    _require(bool(torch.isfinite(r).all())and bool(((r>=0)&(r<=1)).all()),'Reliability must lie in[0,1]')
    _synchronize(device);begin=time.perf_counter();transfer=decode=head_seconds=prior_seconds=0.
    logits=[];hidden_outputs=[];prior_records=[];audits=[];records=[]
    for record in cache['offsets']:
        offset=record['offset'];positions=record['frame_positions'];kwargs=record['inputs']
        _require(record['frame_ids']==cache['frame_ids'][offset::2],'Offset frame IDs changed')
        _check_inputs(kwargs,record['grid_size'],len(positions))
        V=record['grid_size'][0]*record['grid_size'][1]
        legal=~kwargs['encoded_mask'][:,-V:].cpu()
        tick=time.perf_counter();M,meta=occupancy_prior(boxes[positions],record['grid_size'],legal,permuted=permuted)
        prior_seconds+=time.perf_counter()-tick
        tick=time.perf_counter();inputs=restore_inputs(kwargs,record['input_layouts'],device);_synchronize(device);transfer+=time.perf_counter()-tick
        tick=time.perf_counter()
        # Original FP32 temporal route; alpha0 leaves every native attention
        # method unchanged and performs no hand-written attention arithmetic.
        with torch.autocast(device.type,enabled=False),guided_attention(decoder,M,r[positions],legal,alpha)as layer_audits:
            hidden=decoder(**inputs)
        _synchronize(device);decode+=time.perf_counter()-tick
        tick=time.perf_counter()
        with torch.autocast(device.type,enabled=False):all_logits=head(hidden)
        _synchronize(device);head_seconds+=time.perf_counter()-tick
        tick=time.perf_counter();logits.append(detached(all_logits[-1],'cpu'));hidden_outputs.append(detached(hidden,'cpu'))
        _synchronize(device);transfer+=time.perf_counter()-tick
        for a in layer_audits:audits.append({k:float(v.cpu())if torch.is_tensor(v)else v for k,v in a.items()})
        prior_records.append(dict(offset=offset,frame_ids=record['frame_ids'],M=M,reliability=r[positions].clone(),
            legal_visual=legal,metadata=meta))
        records.append(dict(offset=offset,flip=False,frame_ids=list(record['frame_ids'])))
    if alpha==0:
        for value,record in zip(logits,cache['offsets']):
            _require(torch.equal(value,record['native_logits']),'Alpha0 native temporal logits did not recover bitwise')
        for value,record in zip(hidden_outputs,cache['offsets']):
            _require(torch.equal(value,record['hidden']),'Alpha0 native temporal hidden did not recover bitwise')
    _synchronize(device);wall=time.perf_counter()-begin
    _require(len(logits)==2 and len(audits)==(0 if alpha==0 else 2*len(decoder.layers)),
             'Unexpected temporal replay call/layer count')
    return dict(logits=logits,hidden=hidden_outputs,records=records,priors=prior_records,attention_audit=audits,
        cost=dict(replay_wall_seconds=wall,prior_seconds=prior_seconds,cache_transfer_seconds=transfer,
            time_decoder_seconds=decode,temporal_head_seconds=head_seconds,bookkeeping_seconds=wall-transfer-decode-head_seconds-prior_seconds,
            extra_time_decoder_calls=2,offset_time_decoder_calls=[1,1],extra_encoder_calls=0,extra_448_calls=0,
            extra_expert_calls=0,spatial_decoder_calls=0,optimizer_steps=0),
        metadata=dict(alpha=float(alpha),permuted=bool(permuted),GT_read=False,resolution=224,
            visual='original motion-memory suffix, per-frame and per-head',text='prefix probabilities unchanged directly',
            zero_feedback='original native MHA route, exact saved inputs and native head batch shape'))


@torch.no_grad()
def replay_temporal(model,cache,boxes,reliability,*,alpha=.5,permuted=False,device=None):
    """Transfer only the required frozen temporal modules, replay, then restore.

    The physical posterior and original MBR are computed from final-layer
    logits only. No six-layer Strong posterior or 448 branch is constructed.
    """
    from .temporal import probabilities,mass_table,final_mbr
    begin=time.perf_counter();decoder=model.ground_decoder.time_decoder;head=model.temp_embed
    old_decoder=next(decoder.parameters()).device;old_head=next(head.parameters()).device
    target=torch.device(device)if device is not None else torch.device('cuda'if torch.cuda.is_available()else'cpu')
    move_seconds=0.;result=None
    try:
        tick=time.perf_counter();decoder.to(target);head.to(target);_synchronize(target);move_seconds+=time.perf_counter()-tick
        result=_replay_on_device(model,cache,boxes,reliability,alpha=alpha,permuted=permuted)
        tick=time.perf_counter();U,q=mass_table(probabilities(result['records'],result['logits']))
        _require(U==cache['frame_ids'],'Temporal posterior changed the physical grid')
        interval=final_mbr(U,q)['interval'];post_seconds=time.perf_counter()-tick
        result.update(interval=interval,posterior=q,frame_ids=U)
    finally:
        tick=time.perf_counter();decoder.to(old_decoder);head.to(old_head);_synchronize(target);move_seconds+=time.perf_counter()-tick
    wall=time.perf_counter()-begin;c=result['cost'];transfer=c['cache_transfer_seconds']+move_seconds
    c.update(weight_transfer_seconds=move_seconds,transfer_seconds=transfer,
        decoder_seconds=c['time_decoder_seconds']+c['temporal_head_seconds'],time_postprocess_seconds=post_seconds,
        extra_decoder_calls=c['extra_time_decoder_calls'],replay_wall_seconds=wall,
        bookkeeping_seconds=wall-transfer-c['time_decoder_seconds']-c['temporal_head_seconds']-c['prior_seconds']-post_seconds)
    return result
