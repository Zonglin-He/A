"""Native frozen evidence and deterministic decoder-layer candidate support."""
import heapq
import torch
from methods.decota_final_simplified_v1.objectives import prediction


def envelope(a,b,records,ids):
    start=min(records[0]['frame_ids'][a[0]],records[1]['frame_ids'][b[0]])
    end=max(records[0]['frame_ids'][a[1]],records[1]['frame_ids'][b[1]])
    return [ids.index(start),ids.index(end)]


def ranked_spans(z):
    z=z.detach().cpu();n=z.shape[1]
    s=z[:,:,0].log_softmax(1)[0];e=z[:,:,1].log_softmax(1)[0]
    return sorted([(float(s[i]+e[j]),i,j) for i in range(n) for j in range(i+1,n)],key=lambda x:(-x[0],x[1],x[2]))


def candidates(layers,records,ids,k=8):
    temporal=[];spatial=[]
    def add_interval(indices,origin):
        if any(x['indices']==indices for x in temporal):return
        temporal.append(dict(indices=indices,physical_interval=[ids[indices[0]],ids[indices[1]]+1],origin=origin))
    for i in reversed(range(len(layers))):
        pred=layers[i]
        if len(temporal)<k:add_interval(pred['indices'],f'layer{i+1}')
        if not any(torch.equal(x['boxes'],pred['boxes']) for x in spatial):spatial.append(dict(boxes=pred['boxes'],origin=f'layer{i+1}'))
    aa,bb=[ranked_spans(z) for z in layers[-1]['logits']];heap=[(-(aa[0][0]+bb[0][0]),0,0)];seen={(0,0)}
    while heap and len(temporal)<k:
        _,i,j=heapq.heappop(heap);a,b=aa[i],bb[j]
        add_interval(envelope(a[1:],b[1:],records,ids),f'final_top_pair:{i},{j}')
        for u,v in ((i+1,j),(i,j+1)):
            if u<len(aa) and v<len(bb) and (u,v) not in seen:
                seen.add((u,v));heapq.heappush(heap,(-(aa[u][0]+bb[v][0]),u,v))
    return dict(temporal=temporal,spatial=spatial)


@torch.no_grad()
def capture(model,frames,row,old,want_candidates):
    from methods.decota_final_simplified_v1.backbone import make_batch,query_subject,inserted_state,offset_batch
    from methods.decota_final_simplified_v1.tensors import detached
    from scripts.c1_controlled_corruption_v1 import pixelhash
    q={**row['input'],'height':frames.shape[1],'width':frames.shape[2]};ids=row['frame_ids']
    batch=make_batch(frames,ids,q,model);assert pixelhash(batch['videos'].tensors.cpu().numpy())==old['batch_pixels_sha']
    evidence=[];outputs=[];records=[]
    with query_subject(model,batch,row['parses']['subject']),inserted_state(model,{}):
        for offset in (0,1):
            amap={'app':[],'motion':[]};qq=[];act=[];hooks=[]
            for name,mod in [('app',model.s_spatial_clas),('motion',model.t_spatial_clas)]:
                hooks.append(mod.register_forward_hook(lambda m,a,o,name=name:amap[name].append(detached(o[1],'cpu'))))
            hooks.append(model.ground_decoder.register_forward_pre_hook(lambda m,a,kw:qq.append({'s':detached(kw['isq'],'cpu'),'t':detached(kw['itq'],'cpu')}),with_kwargs=True))
            hooks.append(model.action_embed.register_forward_hook(lambda m,a,o:act.append(detached(o,'cpu'))))
            b=offset_batch(batch,offset)
            try:
                with torch.autocast('cuda',dtype=torch.float16):out=model(b['videos'],b['texts'],b['targets'],iteration_rate=-1)
            finally:
                for h in hooks:h.remove()
            assert len(qq)==2 and len(act)==2 and len(amap['app'])==len(amap['motion'])==2
            out=detached(out,'cpu');prob=(out['logits_f_a'].sigmoid()+out['logits_f_m'].sigmoid())/2
            def idx(z):return torch.nonzero(z.reshape(-1),as_tuple=False).reshape(-1).tolist()
            fallback=idx(prob>0);first=idx(prob>model.theta) or fallback;second=idx(act[0][-1].sigmoid()>.5) or fallback
            ev=dict(TTS_app=out['logits_f_a'],TTS_motion=out['logits_f_m'],selected_stage1=first,selected_stage2=second)
            for stage,chosen in [(1,first),(2,second)]:
                for branch in ('app','motion'):
                    a=amap[branch][stage-1];assert len(a)==len(chosen);ev[f'ASA{stage}_{branch}']=a
                for branch in ('s','t'):ev[f'Q{branch}{stage}']=qq[stage-1][branch]
            evidence.append(ev);outputs.append(out);records.append(dict(offset=offset,flip=False,frame_ids=b['targets'][0]['frame_ids']))
    def combine(layer):
        boxes=torch.stack([layer[i%2]['pred_boxes'][i//2] for i in range(len(ids))])
        return prediction([r['pred_sted'] for r in layer],boxes,records,ids)
    native=combine(outputs)
    assert torch.equal(native['boxes'],old['native']['boxes'])
    assert all(torch.equal(a,b) for a,b in zip(native['logits'],old['native']['logits']))
    assert native['indices']==old['native']['indices']
    result=dict(native=native,evidence=evidence,records=records,frame_ids=ids,exact_historical_native=True,batch_pixels_sha=old['batch_pixels_sha'])
    if want_candidates:
        assert len(outputs[0]['aux_outputs'])==len(outputs[1]['aux_outputs'])==5
        layers=[combine([out['aux_outputs'][i] for out in outputs]) for i in range(5)]+[native]
        result.update(layers=layers,candidates=candidates(layers,records,ids))
    return result
