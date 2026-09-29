"""Teacher-independent bounded parameter-space spatial rollout support."""
import torch
from methods.decota_final_simplified_v1.backbone import inserted_state
from vg_tta.tastvg_causal_round2_v1 import forward


def central_state(model):
    state={'spatial.query_residual':torch.zeros(256,device=next(model.parameters()).device)}
    for ln in ['norm1','norm3','norm4']:
        for name,p in getattr(model.ground_decoder.decoder.layers[5],ln).named_parameters():state[f'spatial.layers.5.{ln}.{name}']=p.detach().clone()
    assert sum(v.numel() for v in state.values())==1792
    return state


def directions(size=1792,count=4,seed=20260929):
    g=torch.Generator(device='cpu').manual_seed(seed)
    q,r=torch.linalg.qr(torch.randn(size,count,generator=g,dtype=torch.float64),mode='reduced')
    q=q*torch.where(r.diag()>=0,1.,-1.)
    return q.T.contiguous()


def rollout_states(center,rho=.05):
    flat=torch.cat([x.detach().cpu().double().flatten() for x in center.values()]);basis=directions(len(flat));radius=rho*flat.norm()
    vectors=[flat]+[flat+sign*radius*u for u in basis for sign in [1,-1]];states=[]
    for z in vectors:
        state={};offset=0
        for key,value in center.items():
            state[key]=z[offset:offset+value.numel()].reshape(value.shape).to(value);offset+=value.numel()
        states.append(state)
    return states,dict(rho=rho,radius=float(radius),central_norm=float(flat.norm()),dimensions=len(flat),direction_count=4,candidates=9,seed=20260929),basis


@torch.no_grad()
def predict(model,data,state):
    # Existing exact private interface: query residual on second spatial pass,
    # three final-block LNs on both native calls, restored on context exit.
    with inserted_state(model,state):return forward(model,data,[v['H'] for v in data['views']])


@torch.no_grad()
def reinsert(model,frames,row,state,expected):
    from methods.decota_final_simplified_v1.backbone import make_batch,query_subject,offset_batch
    batch=make_batch(frames,row['frame_ids'],row['input'],model)
    with query_subject(model,batch,row['parses']['subject']),inserted_state(model,state):
        for j in (0,1):
            b=offset_batch(batch,j)
            with torch.autocast('cuda',dtype=torch.float16):out=model(b['videos'],b['texts'],b['targets'],iteration_rate=-1)
            for key in ['pred_boxes','pred_sted']:assert torch.equal(out[key],expected[j][key]),(j,key)
    return dict(full_pipeline_exact=True,offsets=2)
