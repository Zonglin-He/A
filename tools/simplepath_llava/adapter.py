"""Actual autoregressive temporal candidates, never independent endpoint logits.

Every legal start token has its own end distribution under the native shared
separator syntax. Native generation and the Source track remain unchanged.
"""
import hashlib
from types import MethodType
import numpy as np
import torch
from tools.llava_st import backbone
from tools.llava_st.evidence import SourceEvidence
from tools.simplepath_tube.core.spatial import build_geometry,evidence_dp
from tools.simplepath_tube.core.local import LocalFrame,grid_regions,aggregate_sequence
from tools.simplepath_tube.core.prototype import one_step
from stvg.initialization import initialize
from stvg.predictor import output_writeback
from stvg.temporal import final_mbr


def physical_prior(log_joint,ids100):
    ids=np.asarray(ids100);z=np.asarray(log_joint,dtype=np.float64)
    assert z.shape==(100,100) and ids.shape==(100,) and np.all(np.diff(ids)>=0)
    legal=np.triu(np.ones((100,100),bool),1)&(ids[:,None]<ids[None,:])
    assert np.isfinite(z[legal]).all() and legal.any()
    mass=np.zeros((100,100));value=z[legal];mass[legal]=np.exp(value-value.max());mass/=mass.sum()
    unique,inverse=np.unique(ids,return_inverse=True);q=np.zeros((len(unique),len(unique)))
    a,b=np.where(legal);np.add.at(q,(inverse[a],inverse[b]),mass[a,b])
    assert abs(q.sum()-1)<1e-12 and not np.tril(q).any()
    return q


def legacy_clone(cache):
    if hasattr(cache,'to_legacy_cache'):cache=cache.to_legacy_cache()
    return tuple(tuple(x.detach().clone() for x in layer) for layer in cache)


@torch.inference_mode()
def capture(model,tokenizer,processor,frames,ids,caption,*,prefix=None,allow_invalid=False):
    base=int(model.config.vocab_size)
    assert tokenizer.convert_tokens_to_ids([f'<TEMP-{i:03d}>' for i in range(100)])==list(range(base,base+100))
    track=dict(previous=None,temps=[],between=[],past=None,observed_end=None)
    visual={};tower=model.get_vision_tower();old=tower.forward
    def vision(self,images):
        fingerprint=hashlib.sha256(images.detach().cpu().numpy().tobytes()).hexdigest()
        if prefix is None:
            value,unused=old(images);visual.update(siglip=value.detach().cpu().clone(),pixels_sha256=fingerprint)
            return value,unused
        assert prefix['pixels_sha256']==fingerprint
        visual.update(prefix)
        return torch.as_tensor(prefix['siglip'],device=images.device,dtype=images.dtype),None
    def pre(module,args,kwargs):
        x=kwargs.get('input_ids')
        if x is None or x.numel()!=1:return
        token=int(x.item())
        if len(track['temps'])==1 and not(base<=token<base+100):track['between'].append(token)
        if base<=token<base+100 and len(track['temps'])<2:
            if not track['temps']:
                track['past']=legacy_clone(kwargs['past_key_values']);track['start_lp']=track['previous'].clone()
            else:track['observed_end']=track['previous'].clone()
            track['temps'].append(token-base)
    def post(module,args,kwargs,output):
        if len(track['temps'])<2:
            track['previous']=torch.log_softmax(output.logits[0,-1].float(),dim=-1)[base:base+100].detach().cpu()
    tower.forward=MethodType(vision,tower)
    handles=[model.register_forward_pre_hook(pre,with_kwargs=True),model.register_forward_hook(post,with_kwargs=True)]
    try:c=backbone.capture(model,tokenizer,processor,frames,ids,caption)
    finally:
        tower.forward=old
        for h in handles:h.remove()
    if not c['valid']:
        if allow_invalid:return c
        raise ValueError('predeclared qualification/source failed: '+c.get('reason','unknown'))
    assert len(track['temps'])==2 and track['past'] is not None and len(track['between'])<=16
    token=model.get_input_embeddings().weight.new_empty((1,1),dtype=torch.long)
    conditional=np.zeros((99,100));separator_lp=np.zeros(99);native_error=None
    for start in range(99):
        token.fill_(base+start)
        result=model(input_ids=token,past_key_values=track['past'],use_cache=True,return_dict=True)
        subtotal=0.
        for sep in track['between']:
            subtotal+=float(torch.log_softmax(result.logits[0,-1].float(),-1)[sep])
            token.fill_(sep)
            result=model(input_ids=token,past_key_values=result.past_key_values,use_cache=True,return_dict=True)
        end_lp=torch.log_softmax(result.logits[0,-1].float(),-1)[base:base+100].detach().cpu()
        conditional[start]=end_lp.numpy();separator_lp[start]=subtotal
        if start==track['temps'][0]:native_error=float((end_lp-track['observed_end']).abs().max())
        del result
    assert native_error is not None and native_error<=.002,('conditional path differs from native generation',native_error)
    joint=np.full((100,100),-np.inf);joint[:99]=track['start_lp'][:99].numpy()[:,None]+separator_lp[:,None]+conditional
    q=physical_prior(joint,c['model_frames_100'])
    c.update(native_prior=q,conditional_end_log_prob=conditional,start_log_prob=track['start_lp'].numpy(),
             separator_log_prob=separator_lp,conditional_native_path_max_abs_error=native_error,
             conditional_starts=99,visual_prefix=visual,
             prior_scope='native full-vocabulary token likelihood of each start/shared native separator/conditional end; normalize legal distinct physical endpoints; raw temperature1, Source sampling remains .01')
    del track;return c


def finish(c,teacher,cfg):
    ids=c['frame_ids'];positions=c['positions8'];features=c['features'];valid=c['valid_cells']
    evidence=SourceEvidence.from_native_masks(features,valid,ids,positions,teacher['mask_logits'],
                  teacher['teacher_image_size'],(9,9),selected_query=teacher['selected_query'])
    boxes=teacher['boxes'].double().numpy()
    raw,init=initialize(c['Source'].astype(np.float64),positions,boxes,cfg['E_calibration'])
    E=output_writeback(c['Source'],raw,cfg['initialization_gamma'])
    geometry=build_geometry(evidence,torch.from_numpy(E),torch.from_numpy(boxes))
    aux=evidence_dp(geometry.for_interval([ids[0],ids[-1]+1]))['final'].numpy()
    grids=np.asarray(features);local=[LocalFrame(float(t),grids[j],grid_regions(9,9),np.asarray(valid[j]),(9,9)) for j,t in enumerate(ids)]
    z,support=aggregate_sequence(local,aux,1.5);posterior,trace=one_step(z,c['native_prior'])
    interval=final_mbr(ids,posterior)['interval']
    return dict(Source=c['Source'],Spatial=evidence_dp(geometry.for_interval(c['Native_interval']))['final'].numpy(),
        Temporal=c['Source'],Full=evidence_dp(geometry.for_interval(interval))['final'].numpy(),
        Native_interval=np.asarray(c['Native_interval']),Native_MBR_interval=np.asarray(final_mbr(ids,c['native_prior'])['interval']),
        Selected_interval=np.asarray(interval),frame_ids=np.asarray(ids),positions8=np.asarray(positions),
        native_prior=c['native_prior'],posterior=posterior,conditional_end_log_prob=c['conditional_end_log_prob'],
        start_log_prob=c['start_log_prob'],separator_log_prob=c['separator_log_prob'],frame_feedback=trace['L'],auxiliary_path=aux,
        pooled_pretext=z,support=support,initialization=init)
