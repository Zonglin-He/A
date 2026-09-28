"""Finite GT-free NTP/PTD capture, temporal monitor, single-update replay."""
import argparse, copy, gc, hashlib, os, shutil, sys, time, traceback
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.ptd_dino_bridge_v1 import rows,file as native_file,checked,read,write,status,sha,verify as old_verify
from scripts.ptd_joint_box_opd_v1 import put
from scripts.ptd_spatial_adapter_ab_v1 import processor_load,model_load,frames_for,inputs_for,infer,CK
from scripts.ptd_opd_information_v1 import Budget
from vg_tta.ptd_spatial_adapter_ab_v1 import Adapter
from vg_tta.ptd_np_safety_v1 import js,matched_step,synthetic_checks
OUT=ROOT/'artifacts/ptd_np_safety_v1'

def digest(k):return hashlib.sha256(k.encode()).hexdigest()
def path(r,folder):return OUT/folder/(digest(r['key'])+'.pt')
def pilot():return [r for d in ['HC','Vid'] for r in [v for v in rows('development') if v['domain']==d][:2]]

def register():
    old=old_verify();torch.set_num_threads(4)
    pins=[Path(__file__),ROOT/'vg_tta/ptd_np_safety_v1.py',ROOT/'protocols/ptd_np_safety_v1.md',
        ROOT/'scripts/ptd_spatial_adapter_ab_v1.py',ROOT/'vg_tta/ptd_spatial_adapter_ab_v1.py',
        ROOT/'scripts/ptd_opd_information_v1.py',ROOT/'external/ParallelTubeDecoding/src/model/ptd_generation.py',
        ROOT/'external/ParallelTubeDecoding/src/train/monkey_patch_forward.py']
    parents={str(native_file(r,'native')):sha(native_file(r,'native')) for r in rows('development')}
    for folder in ['ptd_dino_bridge_v1','ptd_dino_conditional_advantage_v1']:
        f=ROOT/'artifacts'/folder/'RESULT_MANIFEST.json';parents[str(f)]=sha(f)
    write(OUT/'SYNTHETIC_CHECKS.json',synthetic_checks())
    write(OUT/'REGISTRATION.json',dict(time=time.time(),pins={str(p):sha(p) for p in pins},parents=parents,
        protected=old['protected'],keys=[r['key'] for r in rows('development')],pilot_keys=[r['key'] for r in pilot()],
        sources=64,positions=1014,rank=16,epsilon=.001,steps=1,seed=20260924,bootstrap_seed=20260926,
        GPU_seconds=7200,pilot_GPU_seconds=1800,output_bytes=3*2**30,GT_worker=False,prototype=False,
        checkpoint=read(CK/'OFFICIAL_RECEIPT.json'),P2='conditional_on_P1_and_valid_expert_reward_interface'))
    status(OUT/'STATUS.json',dict(time=time.time(),state='registered'))

def verify():
    r=read(OUT/'REGISTRATION.json');pins={**r['pins'],**r['parents'],**r['protected']}
    for p in sorted((OUT/'amendments').glob('*.json')):pins.update(read(p).get('pins',{}))
    for p,h in pins.items():assert sha(p)==h,p
    return r

@torch.inference_mode()
def native_probe(model,pr,inp,z):
    import model.ptd_generation as pg
    tids=[pg.get_token_id(pr.tokenizer,f'<t{i+1}>') for i in range(32)]
    ref_end=pg.get_token_id(pr.tokenizer,'<|object_ref_end|>')
    orig=pg._run_cached_ptd_probe;ctx={};cap={}
    def track(*args,**kw):
        q=torch.as_tensor(kw['query_token_ids']).flatten().tolist();ctx['time']=q==[ref_end]
        if q and all(t in tids for t in q):cap['prefix']=args[1].detach().cpu()
        try:return orig(*args,**kw)
        finally:ctx['time']=False
    def hook(module,args,out):
        if ctx.get('time'):
            cap['time_logits']=out[0,1:3][:,tids].float().cpu()
            cap['time_coordinate_mass']=(torch.logsumexp(out[0,1:3][:,tids].float(),-1)-torch.logsumexp(out[0,1:3].float(),-1)).exp().cpu()
        return out
    h=model.lm_head.register_forward_hook(hook)
    try:
        with patch.object(pg,'_run_cached_ptd_probe',track):base=infer(model,pr,inp,fixed=z)
    finally:h.remove()
    assert torch.equal(base['logits'],z['logits']) and torch.equal(base['h'],z['h'])
    assert base['completion']==z['completion']
    assert cap['prefix'][0,:inp['input_ids'].shape[1]].equal(inp['input_ids'][0].cpu())
    return cap

@torch.inference_mode()
def ntp_pair(model,pr,inp,z,prefix,budget):
    import model.ptd_generation as pg
    cids=z['coord_ids'].cuda();token=pg.build_ptd_token_ids(pr.tokenizer,max_time_tokens=32)
    pre=model(**dict(inp),use_cache=True,return_dict=True,logits_to_keep=1);cache=pre.past_key_values
    prompt=inp['input_ids'].shape[1];suffix=prefix[:,prompt:].cuda()
    if suffix.shape[1]:
        o,_=pg._run_language_model(model,suffix,cache,attention_mask=None,position_offsets=torch.arange(prompt,prompt+suffix.shape[1],device='cuda'),logits_to_keep=1);cache=o.past_key_values
    budget.calls+=1
    cache_ctrl=copy.deepcopy(cache);results={}
    for branch,kv in [('free',cache),('ctrl',cache_ctrl)]:
        logit_rows=[];mass_rows=[];raw_rows=[];chosen_rows=[];generated=[];prefix_lengths=[]
        def append(ids):
            nonlocal kv
            offset=kv.get_seq_length();x=torch.tensor([ids],device='cuda')
            o,l=pg._run_language_model(model,x,kv,attention_mask=None,
                position_offsets=torch.arange(offset,offset+len(ids),device='cuda'),logits_to_keep=1)
            kv=o.past_key_values;generated.extend(ids);budget.calls+=1
            return l[0,-1].float()
        for j,p in enumerate(z['positions']):
            tid=pg.get_token_id(pr.tokenizer,f'<t{p+1}>')
            logits=append([tid,int(token['box_start'])]);lp=[];mass=[];raw=[];chosen=[];lens=[]
            for c in range(4):
                logits_coord=logits[cids];lp.append(logits_coord.cpu());mass.append(float((logits_coord.logsumexp(0)-logits.logsumexp(0)).exp()))
                raw.append(bool((cids==logits.argmax()).any()));lens.append(len(generated))
                index=int(logits_coord.argmax()) if branch=='free' else int(z['base_tokens'][j,c])
                chosen.append(index)
                logits=append([int(cids[index])])
            append([int(token['box_end']),int(token['newline'])])
            logit_rows.append(torch.stack(lp));mass_rows.append(mass);raw_rows.append(raw);chosen_rows.append(chosen);prefix_lengths.append(lens)
            budget.check()
        results[branch]=dict(logits=torch.stack(logit_rows),coordinate_mass=torch.tensor(mass_rows),
            full_vocab_argmax_is_coordinate=torch.tensor(raw_rows),conditioned_tokens=torch.tensor(chosen_rows),
            suffix_tokens=torch.tensor(generated),prefix_lengths=torch.tensor(prefix_lengths),
            spatial_prefix_mode=branch,GT_used=False)
    assert torch.equal(results['free']['logits'][0,0],results['ctrl']['logits'][0,0])
    assert torch.equal(results['ctrl']['conditioned_tokens'],z['base_tokens'])
    assert torch.equal(results['ctrl']['prefix_lengths'],results['free']['prefix_lengths'])
    assert len(results['free']['suffix_tokens'])==len(z['positions'])*8
    # No retokenization or hidden structural differences between the two paths.
    ff=results['free']['suffix_tokens'].view(-1,8);cc=results['ctrl']['suffix_tokens'].view(-1,8)
    assert torch.equal(ff[:,[0,1,6,7]],cc[:,[0,1,6,7]])
    del pre,cache,cache_ctrl,kv;gc.collect();torch.cuda.empty_cache()
    return results

@torch.inference_mode()
def temporal_alternatives(model,pr,inp,z,time_logits,budget):
    import model.ptd_generation as pg
    lp=time_logits.double().log_softmax(-1);joint=lp[0,:,None]+lp[1,None,:]
    legal=torch.triu(torch.ones(32,32,dtype=torch.bool));joint[~legal]=-torch.inf
    joint-=torch.logsumexp(joint.flatten(),0);s,e=map(int,z['interval'])
    candidates=sorted([(a,b) for a in range(32) for b in range(a,32) if (a,b)!=(s,e)],key=lambda ab:(-float(joint[ab]),ab))[:2]
    spans=[(s,e)]+candidates;weights=torch.tensor([joint[a,b].exp() for a,b in spans],dtype=torch.float64)
    mass=float(weights.sum());weights/=weights.sum();alternatives=[];u=torch.zeros(len(z['positions']),dtype=torch.float64)
    for k,(a,b) in enumerate(candidates,1):
        fixed=copy.copy(z);tt=list(z['temporal']['tokens']);tt[1]=pg.get_token_id(pr.tokenizer,f'<t{a+1}>');tt[2]=pg.get_token_id(pr.tokenizer,f'<t{b+1}>')
        fixed['temporal']=dict(tokens=tt,anchors=z['temporal']['anchors'])
        monitor=infer(model,pr,inp,fixed=fixed);budget.calls+=1
        assert monitor['positions']==z['positions'] and monitor['interval']==[a,b]
        block_js=js(z['logits'],monitor['logits']).mean(-1);u+=weights[k]*block_js
        fixed['temporal']=dict(tokens=tt,anchors=[pg.get_token_id(pr.tokenizer,f'<t{i+1}>') for i in range(a,b+1)])
        full=infer(model,pr,inp,fixed=fixed);budget.calls+=1
        assert full['interval']==[a,b] and full['positions']==list(range(a,b+1))
        alternatives.append(dict(interval=[a,b],weight=float(weights[k]),legal_probability=float(joint[a,b].exp()),
            matched_support_logits=monitor['logits'],block_JS=block_js,full={k:full[k] for k in ['positions','interval','format_ok','completion','base_tokens','logits','coordinate_valid']},
            new_spatial_decode=True,GT_used=False))
        budget.check()
    return dict(spans=spans,weights=weights,selected_legal_mass=mass,legal_logp=joint,alternatives=alternatives,U_TS=u,GT_used=False)

def capture_source(r,model,pr,budget):
    z=checked(native_file(r,'native'));frames,_=frames_for(r,'clean');inp,pre=inputs_for(r,pr,frames)
    assert pre['pixel_sha']==z['preprocess']['pixel_sha']
    cap=native_probe(model,pr,inp,z);budget.calls+=1
    pair=ntp_pair(model,pr,inp,z,cap['prefix'],budget)
    temporal=temporal_alternatives(model,pr,inp,z,cap['time_logits'],budget)
    result=dict(key=r['key'],positions=z['positions'],frame_ids=r['input']['frame_ids'],prefix=cap['prefix'],time_logits=cap['time_logits'],
        time_coordinate_mass=cap['time_coordinate_mass'],ntp=pair,temporal=temporal,
        Dfree=js(z['logits'],pair['free']['logits']).mean(-1),Dctrl=js(z['logits'],pair['ctrl']['logits']).mean(-1),
        baseline_replay_exact=True,first_coordinate_equal=True,preprocess=pre,GT_used=False)
    result['cascade']=result['Dfree']-result['Dctrl']
    put(path(r,'capture'),result)
    del frames,inp;gc.collect();torch.cuda.empty_cache()
    return result

def fit_source(r,capture,budget):
    z=checked(native_file(r,'native'));fits={}
    for branch in ['ctrl','free']:
        fits[branch]=[matched_step(z,capture['ntp'][branch]['logits'],j) for j in range(len(z['positions']))]
        budget.check()
    put(path(r,'fits'),dict(key=r['key'],updates=fits,GT_used=False,steps_per_update=1,reset_each_block=True))
    return fits

def replay_source(r,fits,model,pr,budget):
    z=checked(native_file(r,'native'));frames,_=frames_for(r,'clean');inp,_=inputs_for(r,pr,frames)
    j=len(z['positions'])//2;checks=[]
    for branch in ['ctrl','free']:
        f=fits[branch][j];adapter=Adapter(z['h'].shape[-1]);adapter.up.weight.data.copy_(f['up'])
        with torch.no_grad():expected=z['logits']+adapter(z['h'])
        predicted=infer(model,pr,inp,fixed=z,adapter=adapter.cuda());budget.calls+=1
        assert torch.equal(predicted['h'],z['h']) and torch.equal(predicted['logits'],z['logits'])
        error=float((predicted['adapted_logits']-expected).abs().max())
        # CPU/GPU GEMM accumulation may differ; argmax must remain exact.
        assert error<=2e-5 and torch.equal(predicted['adapted_logits'].argmax(-1),f['tokens']),(error,r['key'])
        checks.append(dict(branch=branch,block=j,max_logit_error=error,tokens_exact=True,backbone_h_exact=True))
        del adapter,predicted;budget.check()
    put(path(r,'replay'),dict(key=r['key'],checks=checks,GT_used=False))
    del frames,inp;gc.collect();torch.cuda.empty_cache()

def run(stage):
    reg=verify();rr=pilot() if stage=='pilot' else rows('development')
    if stage=='development':assert read(OUT/'FORECAST.json')['allowed']
    budget=Budget(OUT,stage,1800 if stage=='pilot' else 7200);done=[]
    try:
        assert sha(CK/'model.safetensors')==reg['checkpoint']['sha256']
        pr=processor_load();model=model_load()
        for r in rr:
            start=time.monotonic()
            cap=checked(path(r,'capture')) if path(r,'capture').exists() else capture_source(r,model,pr,budget)
            fitted=checked(path(r,'fits'))['updates'] if path(r,'fits').exists() else fit_source(r,cap,budget)
            if not path(r,'replay').exists():replay_source(r,fitted,model,pr,budget)
            done.append(r['key']);budget.check()
            status(OUT/'STATUS.json',dict(state='running',stage=stage,done=len(done),total=len(rr),key=r['key'],time=time.time()))
            print('NP_SOURCE',r['key'],len(cap['positions']),round(time.monotonic()-start,2),'complete',flush=True)
            del cap,fitted;gc.collect();torch.cuda.empty_cache()
            assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<reg['output_bytes']
        write(OUT/f'{stage.upper()}_BARRIER.json',dict(time=time.time(),keys=done,GT_read=False,
            files={str(path(r,folder)):sha(path(r,folder)) for r in rr for folder in ['capture','fits','replay']}))
        status(OUT/'STATUS.json',dict(state='sealed',stage=stage,completed=len(done),time=time.time()))
    except BaseException as e:
        failure=dict(time=time.time(),stage=stage,error=repr(e),traceback=traceback.format_exc(),done=done)
        write(OUT/'failures'/f'{time.time_ns()}.json',failure);status(OUT/'STATUS.json',dict(state='failed',**failure));raise
    finally:budget.close()

def forecast():
    verify();assert len(read(OUT/'PILOT_BARRIER.json')['keys'])==4
    secs=sum(read(f)['seconds'] for f in (OUT/'receipts').glob('*.json'))
    observed=sum(len(checked(path(r,'capture'))['positions']) for r in pilot())
    estimate=secs/observed*1014*1.5+120
    write(OUT/'FORECAST.json',dict(time=time.time(),pilot_GPU_worker_seconds=secs,pilot_positions=observed,
        total_GPU_worker_estimate_with_margin=estimate,allowed=estimate<7200,GT_used=False))
    print(read(OUT/'FORECAST.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','pilot','development','forecast']);a=p.parse_args()
    if a.action=='register':register()
    elif a.action=='forecast':forecast()
    else:run(a.action)
