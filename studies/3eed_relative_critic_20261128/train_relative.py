"""Fit four matched Waymo-only rerankers and seal source validation gates."""
import sys

import numpy as np
import torch
from torch import nn
from torch.nn.functional import smooth_l1_loss, softplus
from safetensors.torch import save_file

from source_geometry import HERE,RUN,read,digest,source_guard,write

sys.path.insert(0,str(HERE.parent/'3eed_four_paradigms_20261128'))
from geometry_models import Critic


NAMES=('real_iou','real_pair','real_delta','joint_delta')
EPOCHS=36
QUERY_BATCH=16
THRESHOLDS=(0.,.02,.04,.06,.10,.16,.24)


class RelativeCritic(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.read=nn.Sequential(nn.Linear(4*width,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,1))

    def pair(self,a,b):
        return self.read(torch.cat((a,b,a-b,abs(a-b)),-1)).squeeze(-1)

    def forward(self,descriptors,reference):
        return self.pair(descriptors,reference)-self.pair(reference,reference)


def model_for(name):
    if name in ('real_iou','real_pair'):
        return Critic()
    return RelativeCritic(50)


def logits(model,name,x,base):
    if name in ('real_iou','real_pair'):
        score=model.net(x[...,:48]).squeeze(-1)
        return score.sigmoid() if name=='real_iou' else score
    if name=='real_delta':
        x=torch.cat((x[...,:48],torch.zeros_like(x[...,48:])),-1)
    anchor=x[torch.arange(len(x),device=x.device),base]
    return model(x,anchor[:,None,:].expand_as(x))


def supervised(pred,name,truth,mask,base):
    if name=='real_pair':
        # Sample same-query pairs uniformly, plus comparisons to the reference.
        index=torch.rand(*truth.shape,device=truth.device).masked_fill(~mask,2).argsort(1)
        other=truth.gather(1,index)
        delta=truth-other
        active=mask & mask.gather(1,index) & (delta.abs()>.01)
        pairs=softplus(-delta.sign()*(pred-pred.gather(1,index)))
        anchor=truth.gather(1,base[:,None]); anchor_pred=pred.gather(1,base[:,None])
        ranked=truth-anchor
        usable=mask & (ranked.abs()>.01)
        compare=softplus(-ranked.sign()*(pred-anchor_pred))
        return (pairs[active].sum()+compare[usable].sum())/(active.sum()+usable.sum()).clamp(min=1)
    if name in ('real_delta','joint_delta'):
        truth=truth-truth.gather(1,base[:,None])
    return smooth_l1_loss(pred[mask],truth[mask],beta=.10)


def feature_data(path,stamp,device):
    assert digest(path)==stamp
    with np.load(path,allow_pickle=False) as f:
        return tuple(torch.as_tensor(f[k],device=device) for k in ('x','y','mask','base'))


@torch.inference_mode()
def validation(model,name,data):
    x,y,allowed,base=data
    score=logits(model,name,x,base)
    if name in ('real_delta','joint_delta'):
        np.testing.assert_array_equal(score.gather(1,base[:,None]).cpu().numpy(),0.)
    score=score.masked_fill(~allowed,-torch.inf)
    v,index=score.max(1)
    initial=y.gather(1,base[:,None])[:,0]
    record={}
    gate=(None,) if name in ('real_iou','real_pair') else THRESHOLDS
    for threshold in gate:
        choice=index if threshold is None else torch.where(v>threshold,index,base)
        chosen=y.gather(1,choice[:,None])[:,0]
        record[str(threshold)]=dict(miou=float(chosen.mean()),acc50=float((chosen>.50).float().mean()),
            replacement=int((choice!=base).sum()),mean_gain=float((chosen-initial).mean()))
    best=None if gate==(None,) else max(gate,key=lambda k:(record[str(k)]['miou'],-record[str(k)]['replacement']))
    return dict(baseline=float(initial.mean()),queries=len(y),threshold=best,options=record)


def train():
    source_guard()
    torch.manual_seed(34073)
    torch.set_num_threads(1)
    device='cuda'
    out=RUN/'source'
    pin=read(out/'feature-seal.json')
    assert not pin['used_target']
    assert all(digest(HERE/f)==stamp for f,stamp in pin['files'].items())
    parts={part:feature_data(out/f'features-{part}.npz',pin['info'][part]['sha'],device) for part in ('train','val')}
    models={k:model_for(k).to(device) for k in NAMES}
    models['real_pair'].load_state_dict(models['real_iou'].state_dict())
    models['joint_delta'].load_state_dict(models['real_delta'].state_dict())
    optim=torch.optim.AdamW([p for m in models.values() for p in m.parameters()],lr=.001,fused=True)
    train=parts['train']
    order_count=len(train[0])
    trace=[]
    for e in range(EPOCHS):
        for order in torch.randperm(order_count,device=device).split(QUERY_BATCH):
            x,y,mask,base=(values[order] for values in train)
            optim.zero_grad(set_to_none=True)
            loss=[supervised(logits(model,name,x,base),name,y,mask,base) for name,model in models.items()]
            sum(loss).backward()
            optim.step()
        if e%12==0 or e+1==EPOCHS:
            phase=dict(epoch=e+1,losses=dict(zip(NAMES,[float(it.detach()) for it in loss],strict=True)))
            trace.append(phase)
            print(phase,flush=True)
    trained={}
    for name,model in models.items():
        stored=out/f'{name}.safetensors'
        save_file({k:v.detach().cpu() for k,v in model.state_dict().items()},str(stored))
        val=validation(model.eval(),name,parts['val'])
        trained[name]=dict(sha=digest(stored),source_val=val,parameters=sum(w.numel() for w in model.parameters()))
        print(name,val['threshold'],val['options'][str(val['threshold'])],flush=True)
    write(out/'training-seal.json',dict(models=trained,source_features=digest(out/'feature-seal.json'),
                     used_target=False,qwen_updates=0,trace=trace,code=digest(__file__)))


if __name__=='__main__':
    train()
