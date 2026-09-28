"""P3 bounded corruption inputs and fixed Oracle-KL. T requires a registered contract."""
import argparse, copy, gc, hashlib, os, shutil, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from unittest.mock import patch
from scripts.ptd_np_safety_v1 import rows,native_file,checked,read,write,status,sha,pilot,ntp_pair,native_probe,path as np_path
from scripts.ptd_spatial_adapter_ab_v1 import CK,frames_for,inputs_for,processor_load,model_load,infer
from scripts.ptd_joint_box_opd_v1 import put
from scripts.ptd_opd_information_v1 import Budget
from vg_tta.ptd_np_safety_v1 import js
from vg_tta.ptd_spatial_adapter_ab_v1 import Adapter
from vg_tta.ptd_oracle_credit_v1 import fit,SEEDS,synthetic
OUT=ROOT/'artifacts/ptd_corruption_coupling_v1'
CONDS=['clean','noise_medium','defocus_extreme']
TAU=.11990111548389185
STATES=['F','T']
ARMS={'S':'F','TS':'T','TS-stale':'T'}

def path(r,c,folder,arm='F',seed=None):
    return OUT/folder/c/(hashlib.sha256(r['key'].encode()).hexdigest()+'_'+arm+('' if seed is None else '_'+str(seed))+'.pt')

def selected(stage):return pilot() if stage=='pilot' else rows('development')

def register():
    assert not (OUT/'REGISTRATION.json').exists()
    old=read(ROOT/'artifacts/ptd_oracle_credit_v1/REGISTRATION.json')
    rr=rows('development');assert [r['key'] for r in rr]==old['keys']
    parents={}
    for folder in ['ptd_np_safety_v1','ptd_oracle_credit_v1','ptd_oracle_absorption_v1','ptd_dino_bridge_v1']:
        p=ROOT/'artifacts'/folder/'RESULT_MANIFEST.json';parents[str(p)]=sha(p)
    for p in [ROOT/'artifacts/ptd_np_safety_v1/DESIGN.json',ROOT/'methods/C1_TEMPORAL_RESEARCH_STATUS.json']:
        parents[str(p)]=sha(p)
    assert read(ROOT/'artifacts/ptd_np_safety_v1/DESIGN.json')['cuts']['Dctrl'][2]==TAU
    pins=[Path(__file__),ROOT/'protocols/ptd_corruption_coupling_v1.md',ROOT/'scripts/ptd_np_safety_v1.py',
        ROOT/'scripts/ptd_spatial_adapter_ab_v1.py',ROOT/'scripts/corruption_route_retest_v1.py',
        ROOT/'scripts/c1_controlled_corruption_v1.py',ROOT/'vg_tta/ptd_oracle_credit_v1.py',
        ROOT/'methods/decota_final_simplified_v1/observations.py',ROOT/'vg_tta/ptd_corruption_temporal_v1.py',
        ROOT/'methods/decota_final_simplified_v1/objectives.py',ROOT/'scripts/ptd_temporal_opd_probe_v1.py']
    schedule=[]
    for domain in ['HC','Vid']:
        for i,r in enumerate(x for x in rr if x['domain']==domain):
            schedule.append(dict(key=r['key'],domain=domain,ordinal=i,condition=['clean','defocus_extreme','noise_medium','clean'][i//8]))
    write(OUT/'REGISTRATION.json',dict(time=time.time(),keys=old['keys'],pilot_keys=[r['key'] for r in pilot()],
        pins={str(p):sha(p) for p in pins},parents=parents,protected=old['protected'],checkpoint=old['checkpoint'],
        sources=64,queries=64,conditions=CONDS,changing=schedule,tau_D=TAU,boundary='>= preserving P0 ties',
        arms=['F','T','S','TS','TS-stale'],spatial=dict(rank=16,lr=.002,steps=3,beta=.01,seeds=SEEDS),
        temporal_status='user_authorized_new_PTD_occupancy_bridge',temporal=dict(rank=16,lr=.1,steps=5,eps=1e-4,weight_decay=0,center=.5,prior=.1,margin=.2,eta=.25,seed=20260924,legal_start_le_end=True,single_PTD_time_grid=True),GPU_seconds=7200,pilot_seconds=900,CPU_seconds=1800,output_bytes=8*2**30,
        historical_exposed_development=True,GT_oracle_training=True,critic=False,production_change=False))
    write(OUT/'INPUTS.json',rr);write(OUT/'SYNTHETIC_CHECKS.json',synthetic())
    from vg_tta.ptd_corruption_temporal_v1 import checks
    write(OUT/'TEMPORAL_SYNTHETIC_CHECKS.json',checks())
    write(OUT/'TEMPORAL_INTERFACE_AUDIT.json',dict(status='explicit_new_bridge_authorized_by_user',
        existing_ta='methods/decota_final_simplified_v1/objectives.py:project needs dense actionness and two offsets',
        existing_ptd='scripts/ptd_temporal_opd_probe_v1.py:TimeAdapter exists but target is external teacher OPD',
        user_reply='允许最小 PTD 时间桥接，明确新接口',
        bridge='legal endpoint posterior occupancy -> logit(clamp[1e-6,1-1e-6]) -> original center .5/MAD projection/prior .1 -> NLL+hinge .2',
        changes=['new evidence source','one PTD grid instead of two TA offsets','inclusive legal s<=e matching PTD singleton spans','rank16 temporal output adapter instead of TA head'],
        decoder='legal joint MAP; zero-state equality required; no GT or utility fallback',new_temporal_training=True,utility='untested'))
    status(OUT/'STATUS.json',dict(state='registered',time=time.time()))

@torch.inference_mode()
def time_infer(model,pr,inp,native,adapter=None):
    import model.ptd_generation as pg
    orig=pg._run_cached_ptd_probe;ctx={};cap={};idx=0
    tids=torch.tensor([pg.get_token_id(pr.tokenizer,f'<t{i+1}>') for i in range(32)],device='cuda')
    ref_end=pg.get_token_id(pr.tokenizer,'<|object_ref_end|>')
    def probe(*args,**kw):
        ctx['time']=torch.as_tensor(kw['query_token_ids']).flatten().tolist()==[ref_end]
        try:return orig(*args,**kw)
        finally:ctx['time']=False
    def sem(*args,**kw):
        nonlocal idx
        v=native['semantic'][idx];idx+=1;return v
    def hook(module,args,out):
        if not ctx.get('time'):return out
        h=args[0][0,1:3].float();base=out[0,1:3][:,tids].float()
        cap.update(h=h.cpu(),logits=base.cpu())
        if adapter is None:return out
        z=base+adapter(h);cap['adapted_logits']=z.cpu()
        from vg_tta.ptd_corruption_temporal_v1 import posterior
        lp,ij=posterior(z.cpu());best=ij[:,int(lp.argmax())];cap['legal_interval']=best.tolist()
        y=out.float().clone();y[0,1:3,:]=-torch.inf
        for k in range(2):y[0,k+1,tids[best[k]]]=z[k,best[k]]
        return y
    handle=model.lm_head.register_forward_hook(hook)
    try:
        with patch.object(pg,'_run_cached_ptd_probe',probe),patch.object(pg,'_parse_semantic_block',sem):result=infer(model,pr,inp)
        assert idx==len(native['semantic']) and result['semantic']==native['semantic']
        return result,cap
    finally:handle.remove()

def capture_temporal(r,c,model,pr,budget,stage):
    from vg_tta.ptd_corruption_temporal_v1 import train,TimeAdapter
    base=checked(path(r,c,'capture'))['z'];dest=path(r,c,'capture','T');tick=time.monotonic()
    if dest.exists():return checked(dest)
    if not base['format_ok']:
        result=dict(z=base,Dctrl=torch.empty(0),Dfree=torch.empty(0),invalid_original=True,GT_used=False)
    else:
        frames,_=frames_for(r,c);inp,pre=inputs_for(r,pr,frames);assert pre['pixel_sha']==base['preprocess']['pixel_sha']
        replay,probe=time_infer(model,pr,inp,base);budget.calls+=1
        assert replay['completion']==base['completion'] and torch.equal(replay['logits'],base['logits'])
        assert torch.equal(probe['logits'],checked(path(r,c,'capture'))['time_logits'])
        if stage=='pilot':
            zero,zc=time_infer(model,pr,inp,base,TimeAdapter(probe['h'].shape[-1]).cuda());budget.calls+=1
            assert zero['completion']==base['completion'] and torch.equal(zc['adapted_logits'],probe['logits'])
            del zero,zc
        fitted=train(probe,r['input']['frame_ids']);put(path(r,c,'time_fit'),dict(**fitted,probe=probe,key=r['key'],condition=c))
        adapter=TimeAdapter(probe['h'].shape[-1]);adapter.load_state_dict(fitted['state']);z,tcap=time_infer(model,pr,inp,base,adapter.cuda());budget.calls+=1
        err=float((tcap['adapted_logits']-fitted['logits']).abs().max());assert err<=2e-4,(r['key'],err)
        assert z['interval']==fitted['interval'] and torch.equal(tcap['h'],probe['h'])
        z.update(key=r['key'],condition=c,frame_ids=r['input']['frame_ids'],preprocess=pre)
        cap=native_probe(model,pr,inp,z);budget.calls+=1
        pair=ntp_pair(model,pr,inp,z,cap['prefix'],budget)
        result=dict(z=z,ntp=pair,Dctrl=js(z['logits'],pair['ctrl']['logits']).mean(-1),Dfree=js(z['logits'],pair['free']['logits']).mean(-1),
            prefix=cap['prefix'],time_logits=cap['time_logits'],temporal_logit_error=err,I0=base['interval'],IT=z['interval'],fresh_spatial_and_NTP=True,GT_used=False)
        del frames,inp,replay,adapter;gc.collect();torch.cuda.empty_cache()
    result.update(key=r['key'],condition=c,seconds=time.monotonic()-tick);put(dest,result)
    print('TEMPORAL',c,r['key'],base.get('interval'),result['z'].get('interval'),len(result['Dctrl']),round(result['seconds'],2),flush=True)
    budget.check();return result

def verify():
    r=read(OUT/'REGISTRATION.json');pins={**r['pins'],**r['parents'],**r['protected']}
    for p in sorted((OUT/'amendments').glob('*.json')):pins.update(read(p).get('pins',{}))
    for p,h in pins.items():assert sha(p)==h,p
    return r

def capacity(reg):
    assert shutil.disk_usage(ROOT).free>20*2**30,'DISK_FLOOR'
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<reg['output_bytes'],'OUTPUT_LIMIT'

def gpu(action,stage,fn):
    reg=verify()
    if stage=='development':assert read(OUT/'FORECAST.json')['allowed']
    b=Budget(OUT,action+'_'+stage,reg['pilot_seconds'] if stage=='pilot' and action=='capture' else reg['GPU_seconds'])
    try:
        fn(reg,b);capacity(reg)
    except BaseException as e:
        err=dict(time=time.time(),action=action,stage=stage,error=repr(e),traceback=traceback.format_exc())
        write(OUT/'failures'/f'{time.time_ns()}.json',err);status(OUT/'STATUS.json',dict(state='failed',**err));raise
    finally:b.close()

def capture(stage):
    def work(reg,budget):
        assert sha(CK/'model.safetensors')==reg['checkpoint']['sha256']
        pr=processor_load();model=model_load();files={};times=[]
        for r in selected(stage):
            for c in CONDS:
                dest=path(r,c,'capture')
                if dest.exists():
                    checked(dest);files[str(dest)]=sha(dest)
                    capture_temporal(r,c,model,pr,budget,stage);files[str(path(r,c,'capture','T'))]=sha(path(r,c,'capture','T'));continue
                tick=time.monotonic()
                if c=='clean':
                    z=checked(native_file(r,'native'));cap=checked(np_path(r,'capture'))
                    result=dict(z=z,ntp=cap['ntp'],Dctrl=cap['Dctrl'],Dfree=cap['Dfree'],prefix=cap['prefix'],
                        time_logits=cap['time_logits'],reused_clean=True,parent_hashes={str(native_file(r,'native')):sha(native_file(r,'native')),str(np_path(r,'capture')):sha(np_path(r,'capture'))})
                else:
                    frames,ids=frames_for(r,c);inp,pre=inputs_for(r,pr,frames)
                    z=infer(model,pr,inp);budget.calls+=1
                    z.update(key=r['key'],condition=c,frame_ids=ids,preprocess=pre)
                    if z['format_ok']:
                        cap=native_probe(model,pr,inp,z);budget.calls+=1
                        pair=ntp_pair(model,pr,inp,z,cap['prefix'],budget)
                        result=dict(z=z,ntp=pair,Dctrl=js(z['logits'],pair['ctrl']['logits']).mean(-1),Dfree=js(z['logits'],pair['free']['logits']).mean(-1),prefix=cap['prefix'],time_logits=cap['time_logits'],reused_clean=False)
                        if stage=='pilot':
                            zero=infer(model,pr,inp,fixed=z,adapter=Adapter(z['h'].shape[-1]).cuda());budget.calls+=1
                            assert zero['completion']==z['completion'] and torch.equal(zero['adapted_logits'],z['logits'])
                            result['zero_adapter_exact']=True;del zero
                    else:result=dict(z=z,invalid_original=True,Dctrl=torch.empty(0),Dfree=torch.empty(0),reused_clean=False)
                    del frames,inp;gc.collect();torch.cuda.empty_cache()
                result.update(key=r['key'],condition=c,GT_used=False,seconds=time.monotonic()-tick)
                put(dest,result);files[str(dest)]=sha(dest);times.append(result['seconds']);budget.check()
                status(OUT/'STATUS.json',dict(state='capture',stage=stage,key=r['key'],condition=c,completed=len(list((OUT/'capture').rglob('*.pt'))),time=time.time()))
                print('CAPTURE',c,r['key'],len(result['Dctrl']),int((result['Dctrl']>=TAU).sum()),round(result['seconds'],2),flush=True)
                capture_temporal(r,c,model,pr,budget,stage);files[str(path(r,c,'capture','T'))]=sha(path(r,c,'capture','T'))
            capacity(reg)
        write(OUT/f'CAPTURE_{stage.upper()}_BARRIER.json',dict(time=time.time(),files=files,sources=len(selected(stage)),GT_read=False))
    gpu('capture',stage,work)

def forecast():
    reg=verify();assert len(read(OUT/'CAPTURE_PILOT_BARRIER.json')['files'])==24
    used=sum(read(p)['seconds'] for p in (OUT/'receipts').glob('*.json'))
    costs=[checked(path(r,c,'capture',s))['seconds'] for r in pilot() for c in CONDS for s in STATES]
    total=used+sum(costs)*15*1.5+900
    write(OUT/'FORECAST.json',dict(time=time.time(),used=used,total_with_margin=total,cap=reg['GPU_seconds'],allowed=total<reg['GPU_seconds'],GT_read=False))
    print(read(OUT/'FORECAST.json'),flush=True)

def proposals(stage):
    def work(reg,budget):
        import transformers
        assert transformers.__version__=='4.52.4','DINO must use the registered legacy processor/model environment'
        from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT,EXPERT_SHA256
        from methods.decota_final_simplified_v1.observations import SpatialExpert,ContextView,probe_from_detection,tensor_hash
        from scripts.ptd_dino_bridge_v1 import dest as old_dino,point as old_point
        from scripts.decota_matrix_common_v1 import load
        assert sha(ROOT/EXPERT_SNAPSHOT/'model.safetensors')==EXPERT_SHA256
        expert=SpatialExpert(ROOT/EXPERT_SNAPSHOT);files={}
        for r in selected(stage):
            parses=load(r['parent_file'])['parses'];context,s1=parses['context'],parses['old']
            eligible=context['eligible'] and len(expert.processor.tokenizer(context['context'])['input_ids'])<=256
            for c,state in [(c,s) for c in CONDS for s in STATES]:
                dest=path(r,c,'design',state)
                if dest.exists():checked(dest);files[str(dest)]=sha(dest);continue
                data=checked(path(r,c,'capture',state));z=data['z'];blocks=[];points=[]
                frames=None;old=checked(old_dino(r)) if c=='clean' and state=='F' else None
                for j in (data['Dctrl']>=TAU).nonzero().flatten().tolist():
                    pos=z['positions'][j];pool=[]
                    def add(kind,box):
                        box=list(map(int,box));match=next((v for v in pool if v['box']==box),None)
                        if match is None:pool.append(dict(box=box,kinds=[kind]))
                        else:match['kinds'].append(kind)
                    add('native',z['base_tokens'][j].tolist())
                    for branch in ['ctrl','free']:add('NTP_'+branch,data['ntp'][branch]['logits'][j].argmax(-1).tolist())
                    seed=int(hashlib.sha256(f'cf-v1|{r["key"]}|{j}'.encode()).hexdigest()[:15],16)
                    gen=torch.Generator().manual_seed(seed);prob=z['logits'][j].double().softmax(-1)
                    for k in range(2):add('PTD_sample'+str(k),torch.multinomial(prob,1,generator=gen).flatten().tolist())
                    if old is not None:
                        assert old['positions']==z['positions'] and old['RGB_sha']==z['preprocess']['pixel_sha']
                        accepted=bool(old['mask'][j]);point=dict(position=pos,reused_clean=True,accepted=accepted,RGB_sha=old['RGB_sha'])
                        if accepted:add('DINO',(old['xyxy'][j].double()*1000).round().clamp(0,1000).long().tolist())
                    else:
                        if frames is None:
                            frames,ids=frames_for(r,c);assert hashlib.sha256(frames.tobytes()).hexdigest()==z['preprocess']['pixel_sha']
                        point=dict(position=pos,frame_id=ids[pos],RGB_sha=hashlib.sha256(frames[pos].tobytes()).hexdigest(),GT_used=False)
                        if not eligible and not s1['phrase']:point.update(accepted=False,reason='parser_abstention')
                        else:
                            inp={}
                            def hook(module,args,kwargs):
                                for name in ['pixel_values','pixel_mask','input_ids','attention_mask']:
                                    if name in kwargs:inp[name]=dict(sha256=tensor_hash(kwargs[name]),shape=list(kwargs[name].shape),dtype=str(kwargs[name].dtype))
                            handle=expert.model.register_forward_pre_hook(hook,with_kwargs=True)
                            try:
                                with torch.no_grad():d=ContextView(expert,context)(frames[pos],context['context'],context['entity']) if eligible else expert(frames[pos],s1['phrase'],s1['entity'])
                            finally:handle.remove()
                            budget.calls+=1
                            if eligible:
                                p=probe_from_detection(d,pos,ids[pos]);accepted=p['accepted'];box=torch.as_tensor(p['boxes'][int(np.argmax(p['target_scores']))]).double() if accepted else None
                            else:accepted=d['accepted'];box=torch.as_tensor(d['box']).double() if accepted else None;p=d
                            point.update(accepted=bool(accepted),detection=d,probe=p)
                            point['inputs']=inp
                            if c=='clean':
                                old_positions=checked(old_dino(r))['positions']
                                if pos in old_positions:
                                    prior=checked(old_point(r,old_positions.index(pos)))
                                    assert prior['RGB_sha']==point['RGB_sha'] and prior['inputs']==inp
                                    assert prior['accepted']==bool(accepted)
                                    pe=float((prior['detection']['all_boxes']-d['all_boxes']).abs().max())
                                    se=float((prior['detection']['all_phrase_scores']-d['all_phrase_scores']).abs().max())
                                    assert max(pe,se)<1e-5,(pe,se)
                                    point['legacy_control']=dict(passed=True,pixels_and_text_exact=True,box_error=pe,score_error=se)
                            if accepted:add('DINO',(torch.cat([box[:2]-box[2:]/2,box[:2]+box[2:]/2]).clamp(0,1)*1000).round().long().tolist())
                    points.append(point);blocks.append(dict(j=j,position=pos,frame_id=r['input']['frame_ids'][pos],Dctrl=float(data['Dctrl'][j]),candidates=pool));budget.check()
                put(dest,dict(key=r['key'],condition=c,state=state,blocks=blocks,points=points,GT_used=False));files[str(dest)]=sha(dest)
                print('PROPOSALS',c,state,r['key'],len(blocks),sum(p['accepted'] for p in points),flush=True)
                del frames;gc.collect();torch.cuda.empty_cache()
        for r in selected(stage):
            for c in CONDS:
                f=stale_design(r,c);files[str(f)]=sha(f)
        write(OUT/f'DESIGN_{stage.upper()}_BARRIER.json',dict(time=time.time(),files=files,GT_read=False))
    gpu('proposals',stage,work)

def stale_design(r,c):
    z=checked(path(r,c,'capture','T'))['z']
    old=checked(path(r,c,'design','F'));mapped=[];dropped=[]
    for ob in old['blocks']:
        if ob['position'] not in z.get('positions',[]):dropped.append(ob['position']);continue
        j=z['positions'].index(ob['position']);pool=copy.deepcopy(ob['candidates']);native=z['base_tokens'][j].tolist()
        match=next((v for v in pool if v['box']==native),None)
        if match is not None:pool.remove(match)
        pool.insert(0,dict(box=native,kinds=['T_native_zero_credit_baseline']))
        mapped.append(dict(**{k:v for k,v in ob.items() if k not in ['j','candidates']},j=j,candidates=pool))
    d=dict(key=r['key'],condition=c,blocks=mapped,dropped_positions=dropped,GT_used=False)
    put(path(r,c,'design','TS-stale'),d)
    return path(r,c,'design','TS-stale')

def prepare():
    verify();bar=read(OUT/'DESIGN_DEVELOPMENT_BARRIER.json')
    for p,h in bar['files'].items():assert sha(p)==h
    from scripts.ptd_dino_bridge_v1 import OLD,geometry,support
    from scripts.score_ptd_spatial_adapter_ab_v1 import truth
    labels=read(OLD/'LABELS_SCORER_ONLY.json');files={};start=time.monotonic()
    for r in rows('development'):
        g=truth(r,labels)
        for c,arm,state in [(c,a,s) for c in CONDS for a,s in ARMS.items()]:
            data=checked(path(r,c,'capture',state));z=data['z'];d=checked(path(r,c,'design',state));blocks=[]
            if arm=='TS-stale':d=checked(path(r,c,'design','TS-stale'))
            if z['format_ok']:
                pos,valid,den=support(r,z,g);base=geometry(z['base_tokens'].numpy(),g['boxes'][pos]);output=z['base_tokens'].clone()
                for b in d['blocks']:
                    j=b['j'];boxes=torch.tensor([v['box'] for v in b['candidates']]);ious=geometry(boxes.numpy(),g['boxes'][pos[j]])
                    credit=(ious-base[j])/den if valid[j] else np.zeros(len(ious));assert abs(credit[0])<1e-12;credit[0]=0
                    best=int(credit.argmax());output[j]=boxes[best]
                    blocks.append(dict(j=j,boxes=boxes,credit=torch.tensor(credit),best=best,valid_GT=bool(valid[j]),denominator=int(den),candidate_kinds=[v['kinds'] for v in b['candidates']]))
            else:output=z.get('base_tokens',torch.empty(0,4,dtype=torch.long))
            dest=path(r,c,'oracle',arm);put(dest,dict(key=r['key'],condition=c,blocks=blocks,oracle_output=output,GT_oracle=True));files[str(dest)]=sha(dest)
    write(OUT/'ORACLE_BARRIER.json',dict(time=time.time(),files=files,seconds=time.monotonic()-start,privileged_GT=True))

def train():
    verify();assert (OUT/'ORACLE_BARRIER.json').exists();torch.set_num_threads(4);start=time.monotonic();files={}
    for r in rows('development'):
        for c,arm,state in [(c,a,s) for c in CONDS for a,s in ARMS.items()]:
            z=checked(path(r,c,'capture',state))['z'];o=checked(path(r,c,'oracle',arm))
            for seed in SEEDS:
                dest=path(r,c,'fits',arm,seed)
                if not dest.exists():
                    f=fit(z,o['blocks'],seed) if z['format_ok'] else dict(invalid_native_noop=True,history=[],seed=seed)
                    put(dest,dict(**f,key=r['key'],condition=c))
                files[str(dest)]=sha(dest)
            assert time.monotonic()-start<1800
    write(OUT/'FIT_BARRIER.json',dict(time=time.time(),files=files,seconds=time.monotonic()-start,GT_oracle=True))
    history=OUT/'implementation_history/preprocessor_v5_attempt'
    if (history/'fits').exists():
        reused=[]
        for r in rows('development'):
            for c,arm,seed in [(c,a,s) for c in CONDS for a in ARMS for s in SEEDS]:
                dest=path(r,c,'replay',arm,seed);old_replay=history/dest.relative_to(OUT)
                if not old_replay.exists():continue
                new_fit=checked(path(r,c,'fits',arm,seed));old_fit=checked(history/path(r,c,'fits',arm,seed).relative_to(OUT))
                if new_fit.get('invalid_native_noop'):
                    same=bool(old_fit.get('invalid_native_noop'))
                else:
                    nh,oh=new_fit['history'][-1],old_fit['history'][-1]
                    same=torch.equal(nh['tokens'],oh['tokens']) and all(torch.equal(v,oh['state'][k]) for k,v in nh['state'].items())
                if same:
                    checked(old_replay);dest.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(old_replay,dest);shutil.copy2(old_replay.with_suffix('.json'),dest.with_suffix('.json'))
                    reused.append(dict(path=str(dest),historical_path=str(old_replay),sha256=sha(dest),invalid_native=bool(new_fit.get('invalid_native_noop'))))
        write(OUT/'REPLAY_REUSE.json',dict(rule='same captured state and bitwise identical final adapter parameters and tokens; no metric-based reuse',records=reused,count=len(reused)))

def replay(stage):
    def work(reg,budget):
        assert (OUT/'FIT_BARRIER.json').exists();pr=processor_load();model=model_load();files={}
        for r in selected(stage):
            for c,arm,state in [(c,a,s) for c in CONDS for a,s in ARMS.items()]:
                z=checked(path(r,c,'capture',state))['z'];frames=None;inp=None
                for seed in SEEDS:
                    dest=path(r,c,'replay',arm,seed)
                    if dest.exists():checked(dest);files[str(dest)]=sha(dest);continue
                    f=checked(path(r,c,'fits',arm,seed))
                    if z['format_ok']:
                        if inp is None:
                            frames,_=frames_for(r,c);inp,pre=inputs_for(r,pr,frames);assert pre['pixel_sha']==z['preprocess']['pixel_sha']
                        adapter=Adapter(z['h'].shape[-1],seed=seed);adapter.load_state_dict(f['history'][-1]['state'])
                        with torch.no_grad():expected=z['logits']+adapter(z['h'])
                        result=infer(model,pr,inp,fixed=z,adapter=adapter.cuda());budget.calls+=1
                        assert torch.equal(result['h'],z['h']) and torch.equal(result['logits'],z['logits'])
                        err=float((result['adapted_logits']-expected).abs().max());assert err<=2e-5
                        assert torch.equal(result['adapted_logits'].argmax(-1),f['history'][-1]['tokens'])
                        assert result['interval']==z['interval'] and result['semantic']==z['semantic']
                        out=dict(tokens=f['history'][-1]['tokens'],max_logit_error=err,tokens_exact=True,completion=result['completion'],format_ok=result['format_ok'])
                        del result,adapter
                    else:out=dict(tokens=z.get('base_tokens',torch.empty(0,4,dtype=torch.long)),invalid_native_noop=True,format_ok=False)
                    put(dest,dict(**out,key=r['key'],condition=c,seed=seed,GT_file_read=False,GT_oracle_trained=True));files[str(dest)]=sha(dest);budget.check()
                print('REPLAY',c,arm,r['key'],flush=True);del frames,inp;gc.collect();torch.cuda.empty_cache()
            capacity(reg)
        write(OUT/f'REPLAY_{stage.upper()}_BARRIER.json',dict(time=time.time(),files=files,sources=len(selected(stage))))
    gpu('replay',stage,work)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','capture','forecast','proposals','prepare','train','replay']);p.add_argument('--stage',choices=['pilot','development'],default='pilot');a=p.parse_args()
    if a.action in ['capture','proposals','replay']:globals()[a.action](a.stage)
    else:globals()[a.action]()
