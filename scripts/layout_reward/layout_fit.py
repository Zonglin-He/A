import time,gc
import torch
from stvg.state import Replay
from stvg.policy import chart,diagonal,feedback,mixture_loss,noise_for,unchart
from stvg.tensors import detached

def fit(model,cache,key,arm,cfg,reward):
    rp=Replay(model,cache,cache['Source'],cache['E'],cache['planned_positions'],lambda:None)
    opt=torch.optim.Adam([rp.state],lr=cfg['lr'],betas=(.9,.999),eps=1e-8)
    path=[];rounds=[];start=time.perf_counter();updates=0
    try:
        for k in range(cfg['steps']+1):
            v=rp.values();bs=v['boxes'][rp.positions];mu=chart(bs)
            score=reward(bs[:,None]).squeeze(1)
            path.append(dict(step=k,boxes=v['boxes'].detach().cpu(),layout_reward=score.detach().cpu()))
            if k==cfg['steps']:break
            active=reward.active
            if not bool(active.any()):break
            if arm=='LayoutDirect':
                loss=(1-score[active]).mean();rr=None
            else:
                D=diagonal(bs)
                with torch.no_grad():
                    raw,rng=noise_for(key,41,k,8,torch.float32)
                    eps=torch.cat((raw[:,:8],raw[:,16:24],raw[:,8:16],raw[:,24:]),1).to(mu)
                    sig=torch.tensor([cfg['scales'][0]]*16+[cfg['scales'][1]]*16,device=mu.device,dtype=mu.dtype)
                    actions=(mu.detach()[:,None]+D[:,None]*sig[None,:,None]*eps).detach();boxes=unchart(actions)
                    rewards=reward(boxes);weights=(rewards/cfg['layout_tau']).softmax(-1).detach()
                    rr=dict(actions=actions.cpu(),boxes=boxes.cpu(),rewards=rewards.cpu(),weights=weights.cpu(),ESS=(1/weights.square().sum(-1)).cpu(),proposal_mean=mu.detach().cpu(),rng_seed=rng)
                loss=mixture_loss(mu[active],actions[active],D[active],weights[active],cfg['scales'])
            opt.zero_grad(set_to_none=True);before=rp.state.detach().clone();grad=torch.autograd.grad(loss,rp.state)[0];assert torch.isfinite(grad).all();rp.state.grad=grad;opt.step();updates+=1
            rounds.append(dict(step=k,loss=float(loss.detach()),rollout=rr,gradient=grad.detach().cpu(),state_step=(rp.state.detach()-before).cpu()))
        return dict(arm=arm,final=path[-1]['boxes'],path=path,rounds=rounds,actual_updates=updates,seconds=time.perf_counter()-start)
    finally:
        with torch.no_grad():rp.state.zero_()
        del rp;gc.collect();torch.cuda.empty_cache()
