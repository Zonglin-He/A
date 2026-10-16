"""Five-round closure replay. GT-free; original model and F28 remain immutable.

N2 freezes cloned ASA weights, hard routes and iterative spatial references.
It is a derivative diagnostic only, never a deployment graph.
"""
import copy
import types
import torch
from vg_tta.shared_state_v1 import SharedReplay


def floating32(x):
    if isinstance(x, torch.Tensor):
        return x.float() if x.is_floating_point() else x
    if isinstance(x, dict):return {k:floating32(v) for k,v in x.items()}
    if isinstance(x, list):return [floating32(v) for v in x]
    if isinstance(x, tuple):return tuple(floating32(v) for v in x)
    return x


class ClosureReplay(SharedReplay):
    def __init__(self, model, views, n, scope='shared', precision='fp32', frozen=False):
        joint=scope in ('shared_spatial_head','spatial_head')
        super().__init__(model, views, n, 'shared_spatial' if scope=='shared_spatial_head' else 'spatial' if joint else scope)
        if joint:self._add('head', list(self.head.named_parameters()))
        self.scope=scope;self.initial=self.state();self.precision=precision
        self.frozen=frozen;self.constants={};self.capture_constants=False
        self.decoder.decoder.forward=types.MethodType(self._pos_forward, self.decoder.decoder)

    def const(self, key, value):
        if not self.frozen:return value
        if self.capture_constants:
            self.constants[key]=value.detach().clone() if isinstance(value,torch.Tensor) else copy.deepcopy(value)
        return self.constants[key]

    def _pos_forward(self, dec, query_tgt=None, pred_boxes=None, query_time=None,
                     query_mask=None, encoded_feature=None, encoded_pos=None, encoded_mask=None):
        from models.net_utils import gen_sineembed_for_position
        refs=[]
        for j,layer in enumerate(dec.layers):
            # Initial reference has a differentiable native path; only later
            # references cross the explicitly detached iterative-box interface.
            if j:pred_boxes=self.const(('ref',self.view_id,self.pass_id,j),pred_boxes)
            sine=gen_sineembed_for_position(pred_boxes);pos=dec.ref_point_head(sine)
            scale=1 if j==0 else dec.query_scale(query_tgt)
            sine=sine[...,:dec.d_model]*scale
            query_tgt,_=layer(query_tgt=query_tgt,query_pos=pos,query_time_embed=query_time,
                query_sine_embed=sine,query_mask=query_mask,encoded_feature=encoded_feature,
                encoded_pos=encoded_pos,encoded_mask=encoded_mask,good_rf=None,is_first=j==0)
            new=dec.bbox_embed(query_tgt).sigmoid();refs.append(new);pred_boxes=new.detach()
        return torch.stack(refs).transpose(1,2)

    def _queries(self,H,fm,fa,ft,chosen,count):
        with torch.no_grad():
            _,am=self.model.t_spatial_clas(fm[chosen],ft[:,:1])
            _,aa=self.model.s_spatial_clas(fa[chosen],ft[:,:1])
        am=self.const(('am',self.view_id,self.pass_id),am)
        aa=self.const(('aa',self.view_id,self.pass_id),aa)
        self.extrema.append([am.argmin(-1).tolist(),am.argmax(-1).tolist(),aa.argmin(-1).tolist(),aa.argmax(-1).tolist()])
        return ((H[-count:].permute(1,0,2)[chosen]*am.unsqueeze(2)).mean((0,1)),
                (H[:count].permute(1,0,2)[chosen]*aa.unsqueeze(2)).mean((0,1)))

    def values(self, tensor_shift=0.):
        boxes,zs,actions,gates,Hs=[],[],[],[],[];self.extrema=[]
        self.capture_constants=self.frozen and not self.constants
        with torch.autocast('cuda',dtype=torch.float16,enabled=self.precision=='mixed'):
            for vi,raw in enumerate(self.views):
                self.view_id=vi
                v=floating32(raw) if self.precision=='fp32' else raw
                H=self.norm(v['prefix']);Hs.append(H)
                info=dict(v['info']);info.update(encoded_feature=H,frames_cls=H.mean(0),videos_cls=H.mean(0).mean(0))
                h,w=info['fea_map_size'];count=h*w;nf=H.shape[1]
                fm=H[-count:].permute(1,2,0).reshape(nf,256,h,w).detach()
                fa=H[:count].permute(1,2,0).reshape(nf,256,h,w).detach()
                ft=H[count:-count].mean(1).unsqueeze(0).detach()
                with torch.no_grad():
                    lm=self.model.t_temporal_clas(fm,ft);la=self.model.s_temporal_clas(fa,ft)
                    prob=(lm.sigmoid()+la.sigmoid())/2
                    fallback=self._indices(prob>0)
                    first=self.const(('first',vi),self._indices(prob>self.model.theta) or fallback)
                    self.pass_id=0;itq,isq=self._queries(H,fm,fa,ft,first,count)
                    _,hidden=self.decoder(encoded_info=info,vis_pos=v['vis_pos'],itq=itq,isq=isq)
                    ap=self.model.action_embed(hidden)[-1].squeeze().sigmoid()
                    second=self.const(('second',vi),self._indices(ap>.5) or fallback)
                self.pass_id=1;itq,isq=self._queries(H,fm,fa,ft,second,count)
                def add_query(module,args,kwargs):
                    kw=dict(kwargs);q=kw['query_tgt'];kw['query_tgt']=q+self.delta.to(q.dtype)[None,None,:];return args,kw
                hook=self.decoder.decoder.register_forward_pre_hook(add_query,with_kwargs=True)
                try:pos,hidden=self.decoder(encoded_info=info,vis_pos=v['vis_pos'],itq=itq,isq=isq)
                finally:hook.remove()
                boxes.append(pos.flatten(1,2)[-1]);zs.append(self.head(hidden)[-1]);actions.append(self.model.action_embed(hidden)[-1])
                gates.append(dict(first=first,second=second))
        self.capture_constants=False
        return dict(boxes=torch.stack([boxes[i%2][i//2] for i in range(self.n)]).float(),
                    logits=zs,actions=actions,gates=gates,H=Hs,extrema=copy.deepcopy(self.extrema))
