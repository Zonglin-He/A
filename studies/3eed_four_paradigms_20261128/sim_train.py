"""Train small geometry networks with source-only synthetic ray observations."""
from hashlib import sha256
from itertools import product
import json
from math import pi
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from safetensors.torch import save_file
from torch.nn.functional import smooth_l1_loss

from geometry_models import Flow, Critic, axial, critic_features, exact_iou, ray_frame, scene_for, slab, unit, yaw_frame


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = ROOT/'runs/four-paradigms-20261128'
SOURCE = ROOT/'runs/waymo-posttraining-20261128'
DEVICE = 'cuda'
SEED = 20261128
STEPS = 1954
BATCH = 512
RAYS = 64


def digest(path):
    with Path(path).open('rb') as f:
        return __import__('hashlib').file_digest(f,'sha256').hexdigest()


def write(path, value):
    staging = path.with_suffix(path.suffix+'.tmp')
    staging.write_text(json.dumps(value,allow_nan=False,indent=2)+'\n')
    staging.replace(path)


def source_dimensions():
    lock=json.loads((SOURCE/'source-seal.json').read_text())
    assert not lock['target_read']
    parts={}
    records={}
    images={}
    for split in ('train','val'):
        dataset=SOURCE/f'source-{split}.json'
        assert digest(dataset)==lock[split]['file']
        rows=json.loads(dataset.read_text())
        split_file=ROOT/f'assets/upstream-3eed/data/splits/waymo_{split}.txt'
        assert digest(split_file)==lock[split]['split']
        permitted=set(split_file.read_text().split())
        records[split]={r['recording'] for r in rows}
        images[split]={r['rgb_sha'] for r in rows}
        assert all(r['frame']['target']=='waymo' and r['recording'] in permitted for r in rows)
        assert len(rows)==lock[split]['n']
        values=np.asarray([r['box'][3:6] for r in rows],dtype=np.float32)
        assert (values>0).all()
        values[:,:2]=np.sort(values[:,:2],axis=1)[:,::-1]
        parts[split]=torch.tensor(values,device=DEVICE)
    assert not records['train'] & records['val'] and not images['train'] & images['val']
    return parts, lock


def virtual_scene(dimension, count):
    dev=dimension.device
    rand=lambda *size:torch.rand(*size,device=dev)
    noise=lambda *size:torch.randn(*size,device=dev)
    span=dimension[torch.randint(len(dimension),(count,),device=dev)]
    orientation=rand(count)*pi-pi/2
    center=torch.stack((span[:,2]*0,span[:,2]*0,span[:,2]/2),1)
    gt=torch.cat((center,span,orientation[:,None]),1)
    up = (rand(count)*100-15)*pi/180
    az = rand(count)*2*pi
    high = .3+(49.7)*rand(count).pow(2)
    below = .30+(center[:,2]-.31).clamp(min=0)*rand(count)
    height = torch.where(up>=0,torch.maximum(high,center[:,2]+.20),below)
    delta = height-center[:,2]
    level = torch.where(delta<0,-up.abs(),up.abs())
    horizontal = torch.maximum(delta.abs()/level.tan().abs().clamp(min=.025),span[:,:2].norm(dim=1)*.65).clamp(max=120)
    eye = torch.stack((horizontal*az.cos(),horizontal*az.sin(),height),1)
    basis = ray_frame(center-eye)
    world_axes=yaw_frame(orientation)
    signs=torch.tensor(list(product((-.5,.5),repeat=3)),device=dev)
    vertices=(signs*span[:,None]) @ world_axes.transpose(-1,-2) + center[:,None]
    image=(vertices-eye[:,None])@basis
    uv=image[...,:2]/image[...,2:3].clamp(min=.15)
    lower,upper=uv.amin(1),uv.amax(1)
    jitter=(upper-lower)*.09
    b_lower,b_upper=lower+jitter*noise(count,2),upper+jitter*noise(count,2)
    rect=torch.cat((b_lower,b_upper),1).clamp(-2,2)
    samples=rand(count,RAYS,2)*1.3-.15
    samples[:,0]=.5
    uv=lower[:,None]+samples*(upper-lower)[:,None]
    dirs=unit(torch.cat((uv,torch.ones_like(uv[...,:1])),-1)@basis.transpose(-1,-2))
    silhouette_depth,silhouette=slab(center,span,world_axes,eye,dirs)
    inward=span*(1-rand(count,3)*.14)
    depths,hit=slab(center,inward,world_axes,eye,dirs)
    keep=rand(count).pow(2).clamp(.01,1.)
    # Random occlusion can remove one side independently of ray dropout.
    threshold=rand(count)*.75
    coordinate=torch.where(rand(count,1)<.5,samples[...,0],1-samples[...,1])
    occlusion=(rand(count,1)<.55) & (coordinate<threshold[:,None])
    mask=hit & (rand(count,RAYS)<=keep[:,None]) & ~occlusion
    fallback=hit.to(torch.long).argmax(1)
    sparse=torch.arange(RAYS,device=dev)[None]==fallback[:,None]
    mask|=(~mask.any(1,keepdim=True)) & sparse & hit
    # The predicted 2D region can include other returns and range noise.
    contam=(~hit) & (rand(count,RAYS)<.10) & (rand(count,1)<.5)
    measured=torch.where(contam,norm_depth(center,eye)[:,None]+noise(count,RAYS)*span[:,0,None]*.4,depths)
    points=eye[:,None]+dirs*(measured+noise(count,RAYS)*.035)[...,None]
    prior=span*(noise(count,3)*.25+noise(count,1)*.12).clamp(-.85,.85).exp()
    rough=center+noise(count,3)*.2*span
    scene=scene_for(points,dirs,mask|contam,eye,prior,rect,rough)
    perturb=noise(count,7)
    gain=rand(count,1)*.80+.08
    guess=torch.cat((center+perturb[:,:3]*span*gain,
                     span*(perturb[:,3:6]*gain).clamp(-1.2,1.2).exp(),
                     axial(orientation+perturb[:,6]*gain[:,0]*pi/2)[:,None]),1)
    _,predicted=slab(guess[:,:3],guess[:,3:6],yaw_frame(guess[:,6]),eye,dirs)
    # The training visual feature is a simulated noisy silhouette, independent of LiDAR dropout.
    truth_mask=silhouette ^ (rand(count,RAYS)<.06)
    visual=((truth_mask & predicted).sum(1)/(truth_mask | predicted).sum(1).clamp(min=1)).float()
    visual=torch.where(rand(count)<.30,0.,visual)
    return scene,gt,guess,visual,dict(first_hits=silhouette.sum(),remaining=(mask|contam).sum())


def norm_depth(center,eye):
    return (center-eye).norm(dim=1)


def check_iou():
    import cv2
    rng=np.random.default_rng(2026)
    a=rng.normal(0,.9,size=(256,7))
    b=rng.normal(0,.9,size=(256,7))
    for x in (a,b):
        x[:,3:6]=np.exp(x[:,3:6])+.2
    results=[]
    for x,y in zip(a,b,strict=True):
        def rect(z):
            return (tuple(z[:2]),tuple(z[3:5]),float(z[6]*180/pi))
        _,poly=cv2.rotatedRectangleIntersection(rect(x),rect(y))
        area=cv2.contourArea(cv2.convexHull(poly)) if poly is not None else 0.
        z=max(0.,min(x[2]+x[5]/2,y[2]+y[5]/2)-max(x[2]-x[5]/2,y[2]-y[5]/2))
        i=area*z
        results.append(i/(np.prod(x[3:6])+np.prod(y[3:6])-i))
    err=float(np.max(np.abs(exact_iou(torch.tensor(a),torch.tensor(b)).numpy()-results)))
    assert err<1.e-5,err
    return err


def flow_term(model,scene,truth):
    target=scene.encode(truth)
    zero=torch.randn_like(target)
    t=torch.rand((len(target),1),device=target.device)
    return ((model(zero+(target-zero)*t,t,model.embed(scene))-(target-zero))**2).mean()


def train():
    start=perf_counter()
    RUN.mkdir(exist_ok=True,parents=True)
    if (RUN/'training-seal.json').exists():
        raise FileExistsError('The source-only models are already sealed')
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    err=check_iou()
    bank,source=source_dimensions()
    flow,critic=Flow().cuda(),Critic().cuda()
    optim=torch.optim.AdamW([*flow.parameters(),*critic.parameters()],lr=.001,fused=True)
    monitor=[]
    totals=torch.zeros(2,device=DEVICE)
    prevalence=torch.zeros(2,device=DEVICE)
    for step in range(STEPS):
        optim.zero_grad(set_to_none=True)
        scene,truth,candidate,visual,stats=virtual_scene(bank['train'],BATCH)
        lm=flow_term(flow,scene,truth)
        target=exact_iou(candidate,truth)
        pred=critic(critic_features(scene,candidate),visual)
        lc=smooth_l1_loss(pred,target,beta=.1)
        (lm+lc).backward()
        optim.step()
        totals+=torch.stack((lm.detach(),lc.detach()))
        prevalence+=torch.stack((stats['first_hits'],stats['remaining']))
        if step==0 or (step+1)%250==0 or step+1==STEPS:
            progress=dict(step=step+1,simulated=(step+1)*BATCH,flow_mse=float(lm),critic_loss=float(lc))
            print(progress,flush=True)
            monitor.append(progress)
    torch.manual_seed(SEED+19)
    losses=[]
    with torch.inference_mode():
        for _ in range(16):
            scene,truth,candidate,visual,_=virtual_scene(bank['val'],BATCH)
            expected=exact_iou(candidate,truth)
            estimated=critic(critic_features(scene,candidate),visual)
            losses.append(torch.stack((flow_term(flow,scene,truth), (expected-estimated).abs().mean(),
                                       (expected-expected.mean()).abs().mean())))
    caps={}
    for name,net in (('flow',flow),('critic',critic)):
        file=RUN/f'{name}.safetensors'
        save_file({k:v.detach().cpu() for k,v in net.state_dict().items()},str(file))
        caps[name]=dict(sha256=digest(file),parameters=sum(v.numel() for v in net.parameters()))
    rate=(prevalence/(STEPS*BATCH)).tolist()
    val=torch.stack(losses).mean(0).tolist()
    record=dict(training_samples=STEPS*BATCH,validation_samples=16*BATCH,source=source,seed=SEED,
                used_target_data=False,qwen_updates=0,checkpoints=caps,exact_iou_max_cv2_error=err,
                simulation=dict(rays_per_scene=RAYS,mean_visible_before_dropout=rate[0],mean_retained=rate[1]),
                source_validation=dict(flow_velocity_mse=val[0],critic_iou_mae=val[1],batch_mean_iou_mae=val[2]),
                monitor=monitor,seconds=perf_counter()-start,code={f:digest(HERE/f) for f in
                ('sim_train.py','geometry_models.py')})
    write(RUN/'training-seal.json',record)
    print('Sealed source-only weights',record['source_validation'],flush=True)


if __name__=='__main__':
    train()
