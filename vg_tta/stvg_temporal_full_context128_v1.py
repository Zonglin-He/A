"""Qualified original two-pass Source on one full65 stream, no offset split."""
from contextlib import ExitStack
from types import SimpleNamespace,MethodType
import inspect,textwrap,gc
import numpy as np,torch


def single_posterior(ids,logits):
    from scipy.special import logsumexp
    x=logits.detach().cpu().numpy()if hasattr(logits,'detach')else np.asarray(logits)
    assert x.dtype==np.float32 and x.shape==(1,len(ids),2)and np.isfinite(x).all()
    ii,jj=np.triu_indices(len(ids),1);x=x[0].astype('float64');v=x[ii,0]+x[jj,1];q=np.zeros((len(ids),len(ids)));q[ii,jj]=np.exp(v-logsumexp(v));assert abs(q.sum()-1)<1e-12
    return q


def single_reference(model,view):
    from methods.decota_final_simplified_v1.replay import SpatialReplay
    from methods.decota_final_simplified_v1.tensors import floating32
    source=textwrap.dedent(inspect.getsource(SpatialReplay.reference_values))
    original='boxes=torch.stack([boxes[i % 2][i // 2] for i in range(self.n)]).float()'
    assert source.count(original)==1
    changed=source.replace(original,'boxes=boxes[0].float()');assert changed.replace('boxes=boxes[0].float()',original)==source
    namespace={'torch':torch};exec(compile(changed,__file__,'exec'),namespace)
    proxy=SimpleNamespace(model=model,norm=model.ground_encoder.encoder.norm,decoder=model.ground_decoder,head=model.temp_embed,delta=torch.zeros(256,device=next(model.parameters()).device),views=floating32([view]),n=view['prefix'].shape[1],_indices=SpatialReplay._indices)
    proxy._queries=MethodType(SpatialReplay._queries,proxy)
    result=namespace['reference_values'](proxy,capture=False)
    assert len(result['logits'])==len(result['gates'])==1 and len(result['boxes'])==proxy.n
    return result


@torch.no_grad()
def full_forward(model,frames,row):
    from methods.decota_final_simplified_v1 import backbone
    from methods.stvg_referdino_opd_v1._capture import transfer
    from methods.stvg_referdino_opd_v1._source_forward import transient_native_masks
    from methods.stvg_referdino_opd_v1._visual_host import frozen_visual_host
    meta=dict(row['input']);meta.update(height=frames.shape[1],width=frames.shape[2]);batch=backbone.make_batch(frames,row['frame_ids'],meta,model);prefix=[];ground=[]
    hooks=[model.ground_encoder.encoder.norm.register_forward_pre_hook(lambda m,a:prefix.append(transfer(a[0],'cpu'))),model.ground_decoder.register_forward_pre_hook(lambda m,a,kw:ground.append(transfer(kw,'cpu'))if not ground else None,with_kwargs=True)]
    try:
        with transient_native_masks(model),frozen_visual_host(model),backbone.query_subject(model,batch,row['parses']['subject']),torch.autocast('cuda',dtype=torch.float16):
            model(batch['videos'],batch['texts'],batch['targets'],iteration_rate=-1)
    finally:
        for h in hooks:h.remove()
    assert len(prefix)==len(ground)==1
    device=next(model.parameters()).device;view=transfer(dict(prefix=prefix[0],info=ground[0]['encoded_info'],vis_pos=ground[0]['vis_pos']),'cuda')
    with backbone.query_subject(model,batch,row['parses']['subject']):result=single_reference(model,view)
    info=dict(frame_count=len(frames),normalized_shape=list(batch['videos'].tensors.shape),feature_shape=view['info']['fea_map_size'],full_Source_forward_calls=1,FP32_original_two_pass_replay=True,original_reference_function_only_output_assembly_changed=True)
    result=transfer(result,'cpu');del view,batch,prefix,ground;gc.collect();torch.cuda.empty_cache()
    return result,info


def properties():
    ids=[0,4,9,13];z=torch.zeros(1,4,2,dtype=torch.float32);q=single_posterior(ids,z);assert np.max(abs(q-np.triu(np.full((4,4),1/6),1)))<1e-15
    return dict(status='pass',uniform_single_stream_legal_mass=True,decision_support_unchanged=True)
