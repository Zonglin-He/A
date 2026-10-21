"""Frozen native capture without trainable adaptation state.

The historical FP16 visual prefix and FP32 two-pass readout remain in the same
order. No optimizer, learnable residual, copied trainable LN or OPD is created.
"""
import gc
import torch
from .capture import transfer
from .tensors import floating32, detached
from ._visual import frozen_visual_host
from ._attention import frozen_attention_storage
from ._host import inactive_modules_on_host
from .source import transient_native_masks

class FrozenNativeReplay:
    def __init__(self,model,views,n):
        assert len(views)==2 and not model.training and not any(p.requires_grad for p in model.parameters())
        self.model=model;self.views=floating32(views);self.n=n
        self.spatial_inputs=[];self.temporal_inputs=[];self.time_inputs=[];self.time_input_layouts=[]
        with torch.no_grad():self.zero=detached(self.read())
        assert len(self.spatial_inputs)==len(self.temporal_inputs)==len(self.time_inputs)==2
    @staticmethod
    def _indices(mask):
        value=torch.nonzero(mask.squeeze()).squeeze().tolist()
        return [value]if isinstance(value,int)else value
    def _queries(self,H,fm,fa,ft,chosen,count):
        _,am=self.model.t_spatial_clas(fm[chosen],ft[:,:1])
        _,aa=self.model.s_spatial_clas(fa[chosen],ft[:,:1])
        return ((H[-count:].permute(1,0,2)[chosen]*am.unsqueeze(2)).mean((0,1)),
                (H[:count].permute(1,0,2)[chosen]*aa.unsqueeze(2)).mean((0,1)))
    def read(self):
        boxes,logits,actions,gates=[],[],[],[]
        with torch.autocast('cuda',enabled=False):
            for view in self.views:
                H=self.model.ground_encoder.encoder.norm(view['prefix']);info=dict(view['info'])
                info.update(encoded_feature=H,frames_cls=H.mean(0),videos_cls=H.mean(0).mean(0))
                h,w=info['fea_map_size'];count=h*w;nf=H.shape[1]
                fm=H[-count:].permute(1,2,0).reshape(nf,256,h,w).detach()
                fa=H[:count].permute(1,2,0).reshape(nf,256,h,w).detach()
                ft=H[count:-count].mean(1).unsqueeze(0).detach()
                lm=self.model.t_temporal_clas(fm,ft);la=self.model.s_temporal_clas(fa,ft)
                prob=(lm.sigmoid()+la.sigmoid())/2;fallback=self._indices(prob>0)
                first=self._indices(prob>self.model.theta)or fallback
                itq,isq=self._queries(H,fm,fa,ft,first,count)
                _,hidden=self.model.ground_decoder(encoded_info=info,vis_pos=view['vis_pos'],itq=itq,isq=isq)
                ap=self.model.action_embed(hidden)[-1].squeeze().sigmoid();second=self._indices(ap>.5)or fallback
                itq,isq=self._queries(H,fm,fa,ft,second,count)
                def spatial(m,args,kw):
                    assert not args;self.spatial_inputs.append(detached(kw))
                    # Preserve the historical zero-residual addition's tensor layout.
                    # This is a fixed constant, not an adaptation parameter.
                    kw=dict(kw);q=kw['query_tgt'];kw['query_tgt']=q+torch.zeros(256,device=q.device,dtype=q.dtype)[None,None,:]
                    return args,kw
                def temporal(m,args,kw):
                    assert not args;self.time_inputs.append(detached(kw))
                    self.time_input_layouts.append({k:{'shape':list(v.shape),'stride':list(v.stride()),'storage_offset':v.storage_offset()}for k,v in kw.items()})
                hooks=[self.model.ground_decoder.decoder.register_forward_pre_hook(spatial,with_kwargs=True),
                       self.model.ground_decoder.time_decoder.register_forward_pre_hook(temporal,with_kwargs=True)]
                try:pos,hidden=self.model.ground_decoder(encoded_info=info,vis_pos=view['vis_pos'],itq=itq,isq=isq)
                finally:
                    for hook in hooks:hook.remove()
                self.temporal_inputs.append(detached(hidden));boxes.append(pos.flatten(1,2)[-1])
                logits.append(self.model.temp_embed(hidden)[-1]);actions.append(self.model.action_embed(hidden)[-1])
                gates.append(dict(first=first,second=second))
        return dict(boxes=torch.stack([boxes[i%2][i//2]for i in range(self.n)]).float(),logits=logits,actions=actions,gates=gates)

@torch.no_grad()
def frozen_native_forward(model,frames,row):
    from .inputs import make_batch,query_subject,offset_batch
    q=dict(row['input']);q.update(height=frames.shape[1],width=frames.shape[2])
    batch=make_batch(frames,row['frame_ids'],q,model);device=next(model.parameters()).device
    batch['videos']=batch['videos'].to('cpu');gc.collect();torch.cuda.empty_cache();views=[];records=[]
    with transient_native_masks(model),frozen_visual_host(model),frozen_attention_storage(model),inactive_modules_on_host(model),query_subject(model,batch,row['parses']['subject']):
        for offset in (0,1):
            prefixes=[];grounds=[]
            hooks=[model.ground_encoder.encoder.norm.register_forward_pre_hook(lambda m,a:prefixes.append(transfer(a[0],'cpu'))),
                model.ground_decoder.register_forward_pre_hook(lambda m,a,kw:grounds.append(transfer(kw,'cpu'))if not grounds else None,with_kwargs=True)]
            view=offset_batch(batch,offset);view['videos']=view['videos'].to(device)
            try:
                with torch.autocast('cuda',dtype=torch.float16):model(view['videos'],view['texts'],view['targets'],iteration_rate=-1)
            finally:
                for hook in hooks:hook.remove()
            assert len(prefixes)==len(grounds)==1
            views.append(dict(prefix=prefixes[0],info=grounds[0]['encoded_info'],vis_pos=grounds[0]['vis_pos']))
            records.append(dict(offset=offset,flip=False,frame_ids=view['targets'][0]['frame_ids']))
            del view,prefixes,grounds,hooks;gc.collect();torch.cuda.empty_cache()
        replay=FrozenNativeReplay(model,transfer(views,device),len(row['frame_ids']))
    return batch,records,replay
