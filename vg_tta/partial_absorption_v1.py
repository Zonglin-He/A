"""F36 H-only final spatial output copy. Not a deployed interface."""
import copy
import torch
from vg_tta.parametric_observation_v1 import ObservationReplay,tensor_hash


class FinalBBoxReplay(ObservationReplay):
    def __init__(self,model,views,n):
        super().__init__(model,views,n,'spatial')
        source=self.decoder.decoder.bbox_embed.layers[-1]
        assert isinstance(source,torch.nn.Linear)
        self.final_bbox=copy.deepcopy(source).float()
        self.shape=dict(in_features=source.in_features,out_features=source.out_features,
            bias=source.bias is not None,parameters=sum(p.numel() for p in source.parameters()),
            layer_count=len(self.decoder.decoder.layers),path='ground_decoder.decoder.bbox_embed.layers[-1]')
        self._add('spatial',[('final_bbox.'+n,p) for n,p in self.final_bbox.named_parameters()])
        self.initial=self.state();self.trace=[]
    def _pos_forward(self,dec,*args,**kwargs):
        calls=[0]
        def output(module,aa,y):
            calls[0]+=1
            active=self.pass_id==1 and calls[0]==len(dec.layers)
            out=self.final_bbox(aa[0]) if active else y
            self.trace.append(dict(view=self.view_id,pass_id=self.pass_id,layer=calls[0]-1,copy_active=active,
                                   input_hash=tensor_hash(aa[0]),output_hash=tensor_hash(out)))
            return out
        h=dec.bbox_embed.layers[-1].register_forward_hook(output)
        try:
            return super()._pos_forward(dec,*args,**kwargs)
        finally:
            h.remove()
            assert calls[0]==len(dec.layers)
    def values(self):
        self.trace=[]
        return super().values()


def full_bbox_prediction(model,frames,ids,metadata,subject,state,expected):
    from scripts.audit_parametric_reinsertion_v1 import full_prediction
    space=model.ground_decoder.decoder
    fc=copy.deepcopy(space.bbox_embed.layers[-1]).float()
    fc.load_state_dict({n[len('spatial.final_bbox.'):]:p for n,p in state.items() if n.startswith('spatial.final_bbox.')})
    source_state={n:p for n,p in state.items() if not n.startswith('spatial.final_bbox.')}
    counters=dict(pass_calls=0,layer=0);trace=[]
    def entering(module,args,kwargs):
        counters['pass_calls']+=1;counters['layer']=0
    def output(module,args,y):
        counters['layer']+=1
        active=counters['pass_calls']%2==0 and counters['layer']==len(space.layers)
        trace.append(dict(pass_call=counters['pass_calls'],layer=counters['layer']-1,copy_active=active))
        return fc(args[0]) if active else y
    hooks=[space.register_forward_pre_hook(entering,with_kwargs=True),
           space.bbox_embed.layers[-1].register_forward_hook(output)]
    try:
        result=full_prediction(model,frames,ids,metadata,subject,source_state,expected)
    finally:
        for h in hooks:h.remove()
    assert counters['pass_calls']==4 and sum(x['copy_active'] for x in trace)==2
    result['audit'].update(final_bbox_copy_only=True,copy_invocations=2,trace=trace)
    return result
