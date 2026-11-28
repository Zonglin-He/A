"""Source-only supervised and backend-reward optimizer stages."""
from argparse import ArgumentParser
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter

import numpy as np
import torch

from bridge import CFG, HERE, JOINT, RUN, backend, compact, digest, publish, replies, setup, source_bank, source_guard, truth_iou
from model_ops import completion_logp, generate, image, load, make_model, restore, save, state, text_tokens, visual


def joint_example(row):
    answer = compact([dict(bbox_2d=row['window'],bbox_3d=row['camera_box'],label=row['label'])])
    return image(row['frame']['rgb']), JOINT+row['frame']['text'], answer


def reward_item(job):
    row, text = job
    from score import iou2
    raw=replies(text)
    valid=all(r['status']=='ok' for r in raw.values())
    window=raw['2d']['value']
    pixel=iou2(np.asarray(window),np.asarray(row['window'])) if window is not None else 0.
    if valid and max(raw['3d']['value'][3:6])<=100 and max(map(abs,raw['3d']['value'][:3]))<=300:
        prediction=backend(row['frame'],raw)
        overlap=truth_iou(dict(value=prediction['box']),row['box'])['value']
        route=prediction['route']
    else:
        overlap=0.
        route='invalid'
    return dict(total=overlap+CFG['reward_iou2_weight']*pixel+CFG['reward_format_weight']*valid,
                iou=overlap,pixel=pixel,valid=valid,route=route)


def sft(name, rows, example, processor, model, params, steps, lr, seed):
    opt=torch.optim.AdamW(params.values(),lr=lr,weight_decay=0.)
    rng=np.random.default_rng(seed)
    indices=[]
    trace=[]
    begin=perf_counter()
    batch=CFG['sft_batch']
    for step in range(steps):
        if len(indices)<batch:
            indices.extend(rng.permutation(len(rows)).tolist())
        inputs=[example(rows[indices.pop(0)]) for _ in range(batch)]
        ctx=visual(processor,[(im,prompt) for im,prompt,_ in inputs])
        tokens=text_tokens(processor,[answer for _,_,answer in inputs])
        lp, active=completion_logp(model,ctx,tokens,processor.tokenizer.pad_token_id)
        loss=-lp[active].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gradient=torch.nn.utils.clip_grad_norm_(params.values(),1.)
        assert torch.isfinite(gradient) and torch.isfinite(loss)
        opt.step()
        trace.append(dict(step=step+1,loss=float(loss.detach()),grad=float(gradient),tokens=int(active.sum())))
        if step==0 or (step+1)%16==0:
            print(name,step+1,'/',steps,trace[-1],round(perf_counter()-begin,1),flush=True)
    assert any(k.endswith('.b') and v.count_nonzero().item()>0 for k,v in params.items())
    receipt=save(params,name)
    return dict(**receipt,kind='masked assistant-token SFT',steps=steps,batch=batch,lr=lr,seconds=perf_counter()-begin,
                min_loss=min(x['loss'] for x in trace),peak_cuda=torch.cuda.max_memory_allocated(),trace=trace)


def grpo(rows, processor, model, params):
    ref=state(params)
    rng=np.random.default_rng(CFG['seed']+21)
    schedule=[rows[k] for k in rng.permutation(len(rows))[:CFG['continuation_steps']]]
    opt=torch.optim.AdamW(params.values(),lr=CFG['grpo_lr'],weight_decay=0.)
    trace=[]
    begin=perf_counter()
    group=CFG['grpo_group']
    temp=CFG['grpo_temperature']
    clip=CFG['grpo_clip']
    pad=processor.tokenizer.pad_token_id
    import multiprocessing as mp
    with ProcessPoolExecutor(6,mp_context=mp.get_context('fork')) as pool:
        for i,row in enumerate(schedule,1):
            job=(image(row['frame']['rgb']),JOINT+row['frame']['text'])
            texts,tokens=generate(processor,model,[job],120,count=group,temp=temp)
            all_scores=list(pool.map(reward_item,[(row,x) for x in texts]))
            rewards=torch.tensor([x['total'] for x in all_scores],device='cuda')
            deviation=rewards.std(unbiased=False)
            update=[]
            if deviation.item()>1e-6:
                advantages=(rewards-rewards.mean())/(deviation+1e-4)
                context=visual(processor,[job]*group)
                with torch.no_grad():
                    old,_=completion_logp(model,context,tokens,pad,temp)
                    previous=state(params)
                    restore(params,ref)
                    reference,_=completion_logp(model,context,tokens,pad,temp)
                    restore(params,previous)
                for _ in range(CFG['grpo_epochs']):
                    current,mask=completion_logp(model,context,tokens,pad,temp)
                    ratio=(current-old).exp()
                    objective=torch.minimum(ratio*advantages[:,None],ratio.clamp(1-clip,1+clip)*advantages[:,None])
                    gap=(reference-current).clamp(-8,8)
                    kl=gap.expm1()-gap
                    loss=-((objective-CFG['grpo_kl']*kl)*mask).sum(1).div(mask.sum(1)).mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    gradient=torch.nn.utils.clip_grad_norm_(params.values(),1.)
                    assert torch.isfinite(loss) and torch.isfinite(gradient)
                    opt.step()
                    update.append(dict(loss=float(loss.detach()),gradient=float(gradient),kl=float(kl[mask].mean().detach())))
            trace.append(dict(group=i,rewards=all_scores,updates=update))
            if i==1 or i%8==0:
                print('grpo',i,'/',len(schedule),'valid/reward',sum(x['valid'] for x in all_scores),
                      round(float(rewards.mean()),3),'active groups',sum(bool(x['updates']) for x in trace),
                      'seconds',round(perf_counter()-begin,1),flush=True)
    assert any(x['updates'] for x in trace)
    receipt=save(params,'grpo')
    return dict(**receipt,kind='online outcome GRPO via registered geometry reward',groups=len(trace),samples=group*len(trace),
                active_groups=sum(bool(x['updates']) for x in trace),optimizer_updates=sum(len(x['updates']) for x in trace),
                peak_cuda=torch.cuda.max_memory_allocated(),seconds=perf_counter()-begin,trace=trace)


def run(name):
    source_guard()
    setup()
    torch.manual_seed(CFG['seed'])
    rows=source_bank('train')
    processor,model,_,params=make_model(trainable=True)
    if name in ('grpo','source_sft'):
        warm=RUN/'warm-receipt.json'
        from bridge import read
        stamp=read(warm)
        assert digest(RUN/stamp['file'])==stamp['sha256']
        load(params,'warm')
    if name=='warm':
        result=sft(name,rows,joint_example,processor,model,params,CFG['sft_steps'],CFG['sft_lr'],CFG['seed'])
    elif name=='source_sft':
        result=sft(name,rows,joint_example,processor,model,params,CFG['continuation_steps'],CFG['continuation_sft_lr'],CFG['seed']+21)
    elif name=='grpo':
        result=grpo(rows,processor,model,params)
    else:
        from conditioning import source_examples, conditioned_example
        data=source_examples(name)
        result=sft(name,data,conditioned_example,processor,model,params,CFG['sft_steps'],CFG['sft_lr'],CFG['seed']+39)
        result.update(data_examples=len(data))
    if name in ('grpo','source_sft'):
        result['warm_start']=stamp['sha256']
    publish(RUN/f'{name}-receipt.json',dict(**result,source=digest(RUN/'source-seal.json'),script=digest(Path(__file__)),
        model_code=digest(HERE/'model_ops.py'),protocol=digest(HERE/'protocol.json'),target_read=False))


if __name__=='__main__':
    cli=ArgumentParser(description=__doc__)
    cli.add_argument('stage',choices=('warm','source_sft','grpo','shape','graph'))
    run(cli.parse_args().stage)
