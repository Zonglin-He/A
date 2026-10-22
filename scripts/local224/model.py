"""One real visual prefix and the original two-pass temporal flow, offset0 only.

Shares audited primitives without changing the frozen production sources.
Internal native frame selection is retained; no internal score is a new reward.
"""
import copy,gc,time
import torch
from stvg.capture import transfer
from stvg.tensors import floating32,detached
from stvg.temporal_capture import TemporalReplay,TimeOnlyDecoder,_native_contexts,_prefix_view,verify_query_forward

class SingleReplay(TemporalReplay):
    def __init__(self,model,view):
        self.model=model;self.views=floating32([view]);self.keep_audit=False;self.temporal_inputs=[]
        self.norm=copy.deepcopy(model.ground_encoder.encoder.norm).eval().requires_grad_(False)
        self.decoder=TimeOnlyDecoder(model.ground_decoder)
        self.head=copy.deepcopy(model.temp_embed).float().eval().requires_grad_(False)
        self.zero=detached(self.reference_values())
        assert len(self.temporal_inputs)==len(self.zero['logits'])==1

@torch.no_grad()
def forward(model,frames,row):
    from stvg.inputs import make_batch,offset_batch,query_subject
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    verify_query_forward(model.ground_decoder);device=next(model.parameters()).device
    def sync():torch.cuda.synchronize()
    costs={};handles=[];starts={};counts={}
    def hooks(name,module):
        def before(m,a):sync();starts[name]=time.perf_counter()
        def after(m,a,o):sync();costs[name]=costs.get(name,0.)+time.perf_counter()-starts[name];counts[name]=counts.get(name,0)+1
        handles.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
    for name in ['vis_encoder','vid','text_encoder','ground_encoder']:hooks(name,getattr(model,name))
    tick=time.perf_counter();q=dict(row['input']);q.update(height=frames.shape[1],width=frames.shape[2])
    batch=make_batch(frames,row['frame_ids'],q,model);sync();costs['preprocess_and_input_transfer']=time.perf_counter()-tick
    batch['videos']=batch['videos'].to('cpu');gc.collect();torch.cuda.empty_cache()
    try:
        with _native_contexts(model),query_subject(model,batch,row['parses']['subject']):
            tick=time.perf_counter();view=offset_batch(batch,0);view['videos']=view['videos'].to(device);sync();costs['offset_input_transfer']=time.perf_counter()-tick
            tick=time.perf_counter();captured=_prefix_view(model,view,device);sync();costs['native_prefix_wall']=time.perf_counter()-tick
            ids=list(map(int,view['targets'][0]['frame_ids']));shape=list(view['videos'].tensors.shape);del view
            tick=time.perf_counter();gpu=transfer(captured,device);sync();costs['prefix_transfer']=time.perf_counter()-tick
            tick=time.perf_counter();replay=SingleReplay(model,gpu);sync();costs['two_pass_time_with_copy_and_heads']=time.perf_counter()-tick
            logits=replay.zero['logits'][0].cpu();hidden=replay.temporal_inputs[0].cpu()
            del replay,gpu,captured
    finally:
        for h in handles:h.remove()
    assert counts==dict(vis_encoder=1,vid=1,text_encoder=1,ground_encoder=1),counts
    return dict(ids=ids,logits=logits,hidden=hidden,costs=costs,module_calls=counts,input_shape=shape,native_time_passes=2)
