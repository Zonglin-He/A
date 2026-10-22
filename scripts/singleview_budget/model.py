"""Real one/two-view frozen native temporal branch; no diagnostic sync hooks."""
import gc
import torch
from stvg.capture import transfer
from stvg.temporal_capture import TemporalReplay,_native_contexts,_prefix_view,verify_query_forward
from scripts.local224.model import SingleReplay

@torch.no_grad()
def forward(model,frames,row,offsets=(0,)):
    from stvg.inputs import make_batch,offset_batch,query_subject
    assert tuple(offsets)in[(0,),(1,),(0,1)]
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    verify_query_forward(model.ground_decoder);device=next(model.parameters()).device
    q=dict(row['input']);q.update(height=frames.shape[1],width=frames.shape[2])
    batch=make_batch(frames,row['frame_ids'],q,model);batch['videos']=batch['videos'].to('cpu');gc.collect();torch.cuda.empty_cache()
    views=[];records=[];shapes=[]
    with _native_contexts(model),query_subject(model,batch,row['parses']['subject']):
        for offset in offsets:
            view=offset_batch(batch,offset);view['videos']=view['videos'].to(device)
            views.append(_prefix_view(model,view,device));records.append(dict(offset=offset,flip=False,frame_ids=list(map(int,view['targets'][0]['frame_ids']))));shapes.append(list(view['videos'].tensors.shape));del view
            gc.collect();torch.cuda.empty_cache()
        gpu=transfer(views,device)
        replay=SingleReplay(model,gpu[0])if len(offsets)==1 else TemporalReplay(model,gpu,audit=False)
        logits=[x.detach().cpu()for x in replay.zero['logits']]
    del batch,replay,gpu,views
    return dict(records=records,logits=logits,input_shapes=shapes,actual_visual_views=len(offsets),native_time_passes=2*len(offsets))

def posterior(out,full_ids):
    import numpy as np
    from stvg.temporal import probabilities,mass_table
    recs,z=out['records'],out['logits']
    if len(recs)==1:recs=recs*2;z=z*2
    sub,q=mass_table(probabilities(recs,z));loc={int(f):i for i,f in enumerate(full_ids)}
    ix=[loc[int(f)]for f in sub];p=np.zeros((len(full_ids),len(full_ids)),dtype=np.float64);p[np.ix_(ix,ix)]=q
    assert abs(p.sum()-1)<1e-12
    return p
