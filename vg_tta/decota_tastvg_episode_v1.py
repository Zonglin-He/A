"""Live TA-STVG cross-domain episode with frozen-box DeCoTA and true LN baselines."""
import copy,time
import numpy as np
import torch
from methods.decota_v1 import fit,decode
from vg_tta.foreground_runtime import set_query_subject
from vg_tta.metrics import interval_from_logits

def make_batch(frames,ids,q,subject,model):
    from scripts import run_tastvg_span_tta as ta
    x=torch.from_numpy(np.ascontiguousarray(frames)).permute(0,3,1,2).float()/255.
    height=224;width=min(int(height*q['width']/q['height']),int(height*1.4))
    x=torch.nn.functional.interpolate(x,size=(height,width),mode='bilinear',align_corners=False,antialias=True)
    x=(x-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
    target={'item_id':q['index'],'vid':q.get('original_video_id',q.get('source',q['index'])),'frame_ids':list(ids),
        'actioness':torch.zeros(len(ids),dtype=torch.int64),'img_size':tuple(x.shape[-2:]),'ori_size':(q['height'],q['width'])}
    set_query_subject(model,target,subject)
    return ta._batch_to_device(ta.official_imports()['collate_fn']([(x,q['caption'],target)]),next(model.parameters()).device)

def forward(model,batch,*,grad=False,view_index=0):
    from scripts import run_tastvg_span_tta as ta
    post=ta.official_imports()['postprocessor']();records=[];hs=[];zs=[]
    for offset in [0,1]:
        view=ta._make_temporal_view_batch(batch,offset=offset,flip=False)
        if view_index:
            videos=view['videos'];x=videos.tensors
            mean=x.new_tensor([.485,.456,.406])[None,:,None,None];std=x.new_tensor([.229,.224,.225])[None,:,None,None]
            rgb=(x*std+mean).clamp(0,1)
            if view_index==1:rgb=(rgb*.8).clamp(0,1)
            elif view_index==2:rgb=(rgb*1.2).clamp(0,1)
            elif view_index==3:rgb=(rgb*.7+rgb.mean(1,keepdim=True)*.3).clamp(0,1)
            else:raise ValueError(view_index)
            view['videos']=type(videos)((rgb-mean)/std,videos.mask,videos.durations)
        captured={}
        h=model.temp_embed.register_forward_pre_hook(lambda m,args:captured.update(h=args[0].detach()))
        try:
            with torch.set_grad_enabled(grad),torch.autocast('cuda',dtype=torch.float16):
                o=model(view['videos'],view['texts'],view['targets'],iteration_rate=-1)
        finally:h.remove()
        zs.append(o['pred_sted'][0]);hs.append(captured['h'])
        if not grad:
            r=ta._postprocess_temporal_view(o,view,post,offset=offset,flip=False)
            records.append({k:r[k].detach().cpu() if torch.is_tensor(r[k]) else r[k] for k in ['offset','flip','frame_ids','raw_boxes','boxes_abs','temporal_logits','pred_sted']})
    n=len(batch['targets'][0]['frame_ids']);merged=torch.stack([zs[i%2][i//2] for i in range(n)])
    if grad:return merged
    native=ta._merge_postprocessed_views(records,batch['targets'][0]['frame_ids'])
    return native,hs,records

def fitted_merge(zs,records,ids):
    # Preserve native FP16 arithmetic through log_softmax AND score addition.
    # Promoting logits to FP32 first breaks official rounded-score ties.
    starts=[];ends=[]
    for z,r in zip(zs,records):
        ij=native_view_indices(z)
        starts.append(ids.index(r['frame_ids'][ij[0]]));ends.append(ids.index(r['frame_ids'][ij[1]]))
    return min(starts),max(ends)

def native_view_indices(z):
    """Exact temporal branch of official PostProcess, including dtype/ties."""
    assert z.ndim==3 and z.shape[0]==1 and z.shape[-1]==2 and torch.isfinite(z).all()
    n=z.shape[1]
    mask=(torch.ones(n,n,dtype=torch.float32)*-1e32).tril(0).to(z.device)
    score=mask+(z[:,:,0].log_softmax(1).unsqueeze(2)+z[:,:,1].log_softmax(1).unsqueeze(1))[0]
    return divmod(int(score.flatten().max(0)[1]),n)

def episode(model,batch,config,baseline_configs,*,baselines=True,ablations=False):
    from vg_tta.tastvg_baseline_expansion import decoder_layernorm_scope
    from vg_tta.external_tta_baselines import run_endpoint_tta,capture_parameter_state,restore_parameter_state
    ids=list(batch['targets'][0]['frame_ids']);start=time.perf_counter()
    base,inputs,records=forward(model,batch);native=base['predicted_indices'];z0=base['temporal_logits'];boxes=base['raw_boxes']
    predictions={'frozen':{'boxes':boxes,'indices':native}};audits={};times={'frozen_seconds':time.perf_counter()-start}
    configs={'decota':dict(config)}
    if ablations:
        configs.update({'anchor1e4':{**config,'gamma':1e-4},**{f'steps{s}':{**config,'steps':s} for s in [0,1,3,10,20]},'lr0':{**config,'lr':0.}})
    fitted_state=None
    for name,c in configs.items():
        t=time.perf_counter();head,zs,audit=fit(model.temp_embed,inputs,backbone='tastvg',lr=c['lr'],steps=c['steps'],gamma=c.get('gamma',0.))
        extent=fitted_merge(zs,records,ids);result=decode(z0,extent,ids,native_indices=native)
        predictions[name]={'boxes':boxes,'indices':result['indices']};audits[name]={'fit':audit,'decode':result};times[name+'_seconds']=time.perf_counter()-t
        if name=='decota':
            predictions['coupled_coverage']={'boxes':boxes,'indices':extent}
            fitted_state={k:v.detach().cpu() for k,v in head.state_dict().items()}
        if name in ['lr0','steps0']:
            assert result['indices']==native
            assert all(torch.equal(a.detach().cpu(),b.detach().cpu()) for a,b in zip(head.parameters(),model.temp_embed.parameters()))
        del head,zs
    if baselines:
        names,params,scope=decoder_layernorm_scope(model);saved=capture_parameter_state(params)
        for method,c in baseline_configs.items():
            t=time.perf_counter()
            try:
                audit=run_endpoint_tta(params,lambda v:forward(model,batch,grad=True,view_index=v),method,reset=False,**c)
                updated,_,_=forward(model,batch)
                predictions[method]={'boxes':updated['raw_boxes'],'indices':updated['predicted_indices']}
                audits[method]={'fit':audit,'scope':scope,'live_decoder_forward':True,'spatial_not_artificially_frozen':True}
            finally:
                restore_parameter_state(params,saved)
                for p in params:p.grad=None
            assert all(torch.equal(p.detach(),s.to(p.device)) for p,s in zip(params,saved))
            times[method+'_seconds']=time.perf_counter()-t
    return {'predictions':predictions,'native_logits':z0,'head_inputs':[h.detach().cpu() for h in inputs],
        'native_views':records,'fitted_head_state':fitted_state,'audits':audits,'timing':times,'GT_used':False,'frame_ids':ids}
