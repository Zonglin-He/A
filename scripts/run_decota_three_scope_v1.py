"""Finite no-label features/acquisition/predictions. No old queue mutation."""
import sys,os,time,gc,hashlib,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_three_scope_common_v1 import *

def prepare():
    assert not (BASE/'RUNTIME_LOCK.json').exists()
    assert read(ROOT/'artifacts/decota_ln_spectrum_v1/FINAL_COMPLETION.json')['status']=='completed_verified_publication'
    lock=old.verify();pins=dict(lock['pins'])
    for f in sorted((old.BASE/'revisions').glob('*.json')):pins.update(read(f)['pin_overrides'])
    pins.update({p:sha(ROOT/p) for p in OWN})
    pins.update({p:sha(ROOT/p) for p in ['external/TA-STVG/models/pipeline.py','external/TA-STVG/models/language_model/bert.py']})
    inputs={};cohort=[]
    def pin(f):inputs[str(f.relative_to(ROOT))]=sha(f)
    for f in [old.BASE/'ONLINE_GLOBAL_BARRIER.json',old.BASE/'ONLINE_LOCK.json',ROOT/'artifacts/decota_ln_spectrum_v1/PAIR_SELECTION_BARRIER.json']:
        if f.exists():pin(f)
    for ds in DATASETS:
        p=read(old.BASE/ds/'PLAN.json');assert len(p['rows'])==48
        write(BASE/ds/'PLAN.json',p);pin(BASE/ds/'PLAN.json')
        for row in p['rows']:
            for cond in p['conditions']:
                _,rc=c1.cache(ds,row['pool_parent'],cond,content=False)
                pin(c1.POOL/ds/'capture'/cond/f"{row['pool_parent']:05}.json")
                ef=c1.BASE/ds/'evidence'/cond/f"{row['ordinal']:05}.pt";pin(ef);pin(ef.with_suffix('.json'))
        for split,sp in p['splits'].items():
            for cond in p['conditions']:
                for order,seq in sp['orders'].items():
                    for at,parent in enumerate(seq):
                        refs={}
                        for stream in ['episodic','online100']:
                            f=old.BASE/'online'/ds/stream/split/cond/order/f'{at:05}.pt'
                            pin(f);pin(f.with_suffix('.json'));refs[stream]=str(f.relative_to(ROOT))
                        cohort.append(dict(dataset=ds,split=split,condition=cond,order=order,arrival=at,parent=parent,references=refs))
    assert len(cohort)==1152
    write(BASE/'COHORT.json',dict(cells=cohort));pin(BASE/'COHORT.json')
    write(BASE/'RUNTIME_LOCK.json',dict(version='decota_three_scope_v1',time=time.time(),pins=pins,inputs=inputs,protected=lock['protected'],
        unique_inputs=576,logical_scope_outputs=2304,new_DINO_cap=2304,parameters=1792,Adam_lr=.03,steps=10,LN_writeback=1/16,
        TTS_delta=.5,TTS_rule='physical stratified max, earliest tie, old empty-bin farthest fallback',
        streams=['episodic','online100'],expert_rate=1.,no_backbone_forward=True,temporal='Native',GT_online=False,
        boundary='linear probability values/merged frame grid/zero extrapolation/floor1e-12/geometric/strict i<j',
        ridge_alpha=1.,no_total_deadline=True,no_search=True))
    status(BASE/'STATUS.json',dict(status='locked_pending_contracts_features',predictions=0,GT_read=False))
    archive('新名单/规则/代码与旧匹配缓存哈希锁定；零新预测，待合同与选帧准备')

def gpu():
    from scripts.run_tastvg_decota_c1_same_domain_v1 import start_gpu
    return start_gpu()

def data_for(ds,row,cond):
    from scripts.run_tastvg_decota_c1_same_domain_v1 import data_input
    return data_input(ds,row,cond)

def smoke_uniform(ds,model,p):
    import torch
    from scripts.run_decota_identity_commitment_v1 import compare
    from vg_tta.tastvg_decota_c1_same_domain_v1 import NormalizedSpatialReplay
    from vg_tta.decota_identity_commitment_v1 import fit
    cases=[]
    for at,parent in enumerate(p['splits']['search']['orders']['order1'][:2]):
        row=p['rows'][parent];data,rc=data_for(ds,row,'clean');base=NormalizedSpatialReplay(model,data)
        ef=c1.BASE/ds/'evidence'/'clean'/f'{parent:05}.pt';ex=c1.checked(ef)
        ref=old.checked(old.BASE/'online'/ds/'episodic'/'search'/'clean'/'order1'/f'{at:05}.pt')
        z=fit(base,base.initial,ex['expert'],row['frame_ids'],row['key'],'top1');compare(z,ref['fit'])
        assert torch.equal(base.zero['boxes'].cpu(),ref['native']['boxes'])
        cases.append(dict(dataset=ds,parent=parent,original_Top1_all_steps_bitwise=True,native_bitwise=True,backward_calls=z['gradient_calls']))
    write(BASE/ds/'SMOKE.json',dict(status='pass',cases=cases,GT_read=False,time=time.time()))

def features(ds):
    import torch,numpy as np
    from scripts.run_tastvg_decota_c1_same_domain_v1 import model_for
    from vg_tta.tastvg_decota_c1_same_domain_v1 import NormalizedSpatialReplay
    from vg_tta.decota_three_scope_v1 import tts_positions,tube_summary
    from methods.decota_final_simplified_v1.tensors import state_hash
    verify();lease=gpu();model=model_for(ds);mh=state_hash(model.state_dict());p=read(BASE/ds/'PLAN.json');files={};start=time.time();qcache={}
    try:
        smoke_uniform(ds,model,p)
        for row in p['rows']:
            parent=row['ordinal']
            with torch.no_grad():
                tok=model.text_encoder.tokenizer([row['input']['caption']],padding=True,return_tensors='pt').to('cuda')
                h=model.text_encoder.body(**tok).last_hidden_state
                hq=(h*tok['attention_mask'][...,None]).sum(1)/tok['attention_mask'].sum(1)[:,None]
                qcache[parent]=hq[0].float().cpu().numpy()
            for cond in p['conditions']:
                budget();f=BASE/'features'/ds/cond/f'{parent:05}.pt';assert not f.exists()
                data,rc=data_for(ds,row,cond);base=NormalizedSpatialReplay(model,data);scores=[]
                with torch.no_grad(),torch.autocast('cuda',enabled=False):
                    for v in data['views']:
                        H=v['H'];hh,ww=v['info']['fea_map_size'];n=hh*ww;nf=H.shape[1]
                        a=H[:n].permute(1,2,0).reshape(nf,256,hh,ww).detach()
                        m=H[-n:].permute(1,2,0).reshape_as(a);text=H[n:-n].mean(1).unsqueeze(0)
                        app=model.s_temporal_clas(a,text);motion=model.t_temporal_clas(m,text)
                        scores.append(((app.sigmoid()+motion.sigmoid())/2).reshape(-1).cpu().numpy())
                    ids=row['frame_ids'];tts=np.array([scores[i%2][i//2] for i in range(len(ids))],float)
                    norms=[];latents=[]
                    hn=base.decoder.decoder.norm.register_forward_hook(lambda m,a,o:norms.append(o.detach()))
                    def last(m,a,o):
                        assert norms;latents.append(norms[-1]);norms.clear()
                    hd=base.decoder.decoder.register_forward_hook(last)
                    try:values=base.values()
                    finally:hn.remove();hd.remove()
                    assert len(latents)==2 and torch.equal(values['boxes'].cpu(),data['prediction']['boxes'].cpu())
                    latent=torch.cat([x.reshape(-1,256) for x in latents]).mean(0).float().cpu().numpy()
                    qs=torch.stack([x['query_tgt'].reshape(-1,256).mean(0) for x in base.spatial_inputs]).mean(0).float().cpu().numpy()
                feat=dict(query=qcache[parent],spatial=latent,query_native=qs,geometry=tube_summary(data['prediction']['boxes'].cpu().numpy(),ids,data['prediction']['physical_interval'],tts))
                positions=tts_positions(ids,data['prediction']['indices'],tts)
                commit(f,dict(dataset=ds,parent=parent,condition=cond,positions=positions,tts=tts,keys=feat,
                    pixel_sha256=rc['pixel_sha256'],frame_ids=ids,native=data['prediction']['physical_interval'],GT_read=False,
                    source_initialized=True,recipient_delta_accessed=False,temporal_logits_changed=False))
                files[str(f.relative_to(BASE))]=sha(f)
                status(BASE/ds/'FEATURE_STATUS.json',dict(status='running',done=len(files),total=288,pid=os.getpid(),seconds=time.time()-start,GT_read=False))
                if len(files)%12==0:print('THREE_SCOPE_FEATURES',ds,len(files),288,round(time.time()-start,1),flush=True);gc.collect()
                del data,base,values
        assert state_hash(model.state_dict())==mh
        write(BASE/ds/'FEATURE_BARRIER.json',dict(status='sealed',files=files,cells=288,GT_read=False,seconds=time.time()-start,
            text_forwards=48,frozen_TTS_head_forwards=1152,cached_native_suffix_inputs=288,video_backbone_forwards=0))
        status(BASE/ds/'FEATURE_STATUS.json',dict(status='completed',done=288,seconds=time.time()-start,GT_read=False))
    finally:lease.close()

def evidence(ds):
    import torch,numpy as np
    from scripts.run_tastvg_decota_c1_same_domain_v1 import decode_for
    from scripts.run_tastvg_full_b1_experts_v1 import observation
    from methods.decota_final_simplified_v1.observations import SpatialExpert,ContextView,probe_from_detection,tensor_hash
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
    from methods.decota_final_simplified_v1.tensors import state_hash,detached
    verify();assert read(BASE/ds/'FEATURE_BARRIER.json')['status']=='sealed';lease=gpu();model=SpatialExpert(ROOT/EXPERT_SNAPSHOT);mh=state_hash(model.model.state_dict())
    p=read(BASE/ds/'PLAN.json');files={};start=time.time();calls=reused=requests=0
    try:
        decode=decode_for(ds)
        for row in p['rows']:
            frames,ids=decode(row['input']);assert ids==row['frame_ids'];parent=row['ordinal']
            for cond in p['conditions']:
                budget();f=BASE/'evidence'/ds/cond/f'{parent:05}.pt';assert not f.exists()
                ft=checked(BASE/'features'/ds/cond/f'{parent:05}.pt');shifted,pixel,spec=observation(row,cond,frames);assert pixel==ft['pixel_sha256']
                oldf=c1.BASE/ds/'evidence'/cond/f'{parent:05}.pt';oldex=c1.checked(oldf);assert oldex['pixel_sha256']==pixel
                context,s1=row['parses']['context'],row['parses']['old']
                eligible=context['eligible'] and len(model.processor.tokenizer(context['context'])['input_ids'])<=256
                obs={};anchors=[];new=overlap=0
                for pos in ft['positions']:
                    if not eligible and not s1['phrase']:continue
                    rgb=shifted[pos];rgbhash=hashlib.sha256(rgb.tobytes()).hexdigest();key=('original',pos)
                    if key in oldex['expert']['observations']:
                        ob=copy.deepcopy(oldex['expert']['observations'][key]);assert ob['receipt']['rgb_sha256']==rgbhash
                        assert ob['receipt']['context_active']==eligible
                        expected=context['context'] if eligible else s1['phrase'].lower().strip()+'.';assert ob['receipt']['text']==expected
                        ob['receipt'].update(reused_from=str(oldf.relative_to(ROOT)),reused_sha256=sha(oldf),new_forward=False);overlap+=1
                    else:
                        inp={}
                        def hook(m,a,kw):
                            for k in ('pixel_values','pixel_mask','input_ids','attention_mask'):
                                if k in kw:inp[k]=dict(sha256=tensor_hash(kw[k]),shape=list(kw[k].shape),dtype=str(kw[k].dtype))
                        handle=model.model.register_forward_pre_hook(hook,with_kwargs=True);torch.cuda.synchronize();tick=time.perf_counter()
                        try:d=ContextView(model,context)(rgb,context['context'],context['entity']) if eligible else model(rgb,s1['phrase'],s1['entity'])
                        finally:handle.remove()
                        torch.cuda.synchronize()
                        if eligible:z=probe_from_detection(d,pos,ids[pos])
                        else:z=dict(position=pos,frame_id=ids[pos],accepted=d['accepted'],reason=d['reason'],margin=d['margin'],boxes=torch.tensor([d['box']]) if d['box'] else torch.empty(0,4),target_scores=[d['score']] if d['box'] else [],candidate_ids=[0] if d['box'] else [])
                        receipt=dict(position=pos,frame_id=ids[pos],view='original',scale=1.,rgb_sha256=rgbhash,inputs=inp,
                            seconds=time.perf_counter()-tick,forward_seconds=d['seconds'],text=d['text'],target_span=context['span'] if eligible else None,context_active=eligible,new_forward=True)
                        ob=dict(probe=z,receipt=receipt);new+=1
                    obs[key]=ob;z=ob['probe']
                    if z['accepted']:
                        j=int(np.argmax(z['target_scores']));anchors.append(dict(position=pos,frame_id=ids[pos],box=torch.as_tensor(z['boxes'][j]).tolist(),score=float(z['target_scores'][j]),weight=1.))
                ex=dict(observations=obs,positions4=ft['positions'],anchors={'single4':anchors},new_DINO=new,
                    reused_DINO=overlap,actual_observation_positions=sorted(pos for _,pos in obs))
                commit(f,dict(expert=detached(ex,'cpu'),pixel_sha256=pixel,spec=spec,positions=ft['positions'],GT_read=False))
                files[str(f.relative_to(BASE))]=sha(f);calls+=new;reused+=overlap;requests+=len(ft['positions'])
                status(BASE/ds/'EVIDENCE_STATUS.json',dict(status='running',done=len(files),total=288,new_DINO=calls,reused_DINO=reused,requests=requests,pid=os.getpid(),seconds=time.time()-start,GT_read=False))
                if len(files)%12==0:print('THREE_SCOPE_DINO',ds,len(files),288,'new',calls,'reuse',reused,round(time.time()-start,1),flush=True)
                del shifted,ex
            del frames;gc.collect();torch.cuda.empty_cache()
        assert calls<=1152 and state_hash(model.model.state_dict())==mh
        write(BASE/ds/'EVIDENCE_BARRIER.json',dict(status='sealed',files=files,cells=288,new_DINO=calls,reused_DINO=reused,requests=requests,GT_read=False,seconds=time.time()-start,frozen_weights_unchanged=True))
        status(BASE/ds/'EVIDENCE_STATUS.json',dict(status='completed',done=288,new_DINO=calls,reused_DINO=reused,GT_read=False))
    finally:lease.close()

def online(ds):
    import torch
    from scripts.run_tastvg_decota_c1_same_domain_v1 import model_for
    from vg_tta.tastvg_decota_c1_same_domain_v1 import NormalizedSpatialReplay
    from vg_tta.spatial_online_state_v1 import arrival
    from vg_tta.decota_actuation_scope_v1 import commit_state
    from vg_tta.decota_identity_commitment_v1 import fit
    from vg_tta.c1_enabling_tricks_v1 import QUERY,TrickReplay
    from methods.decota_final_simplified_v1.tensors import detached,state_hash
    verify();assert read(BASE/ds/'EVIDENCE_BARRIER.json')['status']=='sealed';lease=gpu();model=model_for(ds);mh=state_hash(model.state_dict())
    p=read(BASE/ds/'PLAN.json');files={};start=time.time();backwards=0;episodes={}
    try:
        for stream in ['episodic','online100']:
            for split,sp in p['splits'].items():
                for cond in p['conditions']:
                    for order,seq in sp['orders'].items():
                        prev=None;prevsha=None
                        for at,parent in enumerate(seq):
                            budget();f=BASE/'online'/ds/stream/split/cond/order/f'{at:05}.pt';assert not f.exists();row=p['rows'][parent];key=(split,cond,parent)
                            if stream=='episodic' and key in episodes:
                                ref=episodes[key];x=checked(ref);x={**x,'reused_episodic':str(ref.relative_to(ROOT)),'reuse_sha256':sha(ref),'actual_backward_calls':0,'seconds':0.}
                            else:
                                data,rc=data_for(ds,row,cond);base=NormalizedSpatialReplay(model,data);source=detached(base.initial,'cpu');initial=arrival(base.initial,prev,'O-split')
                                assert torch.count_nonzero(initial[QUERY])==0;base.restore(initial);rp=TrickReplay(base,row['frame_ids'],row['key'],{})
                                with torch.no_grad():before=detached(rp.values()['boxes'],'cpu')
                                ef=BASE/'evidence'/ds/cond/f'{parent:05}.pt';ex=checked(ef);assert ex['pixel_sha256']==rc['pixel_sha256'];tick=time.perf_counter()
                                z=fit(base,initial,ex['expert'],row['frame_ids'],row['key'],'top1');after=z['final'];committed=commit_state(detached(initial,'cpu'),z['state'])
                                assert state_hash(base.state())==state_hash(initial)
                                x=dict(source_state=source,initial=detached(initial,'cpu'),committed=committed,before=before,after=after,fit=z,native=detached(data['prediction'],'cpu'),
                                    interval=data['prediction']['physical_interval'],expert=True,evidence_path=str(ef.relative_to(ROOT)),evidence_sha256=sha(ef),pixel_sha256=rc['pixel_sha256'],
                                    seconds=time.perf_counter()-tick,GT_read=False,actual_backward_calls=z['gradient_calls'])
                            x.update(dataset=ds,stream=stream,split=split,condition=cond,order=order,arrival=at,parent=parent,previous_payload_sha256=prevsha,query_reset=True,Adam_reset=True)
                            commit(f,x);files[str(f.relative_to(BASE))]=sha(f);backwards+=x['actual_backward_calls']
                            if stream=='episodic':episodes[key]=f
                            else:prev=x['committed'];prevsha=sha(f)
                            status(BASE/ds/'ONLINE_STATUS.json',dict(status='running',done=len(files),total=1152,stream=stream,pid=os.getpid(),seconds=time.time()-start,GT_read=False))
                            if len(files)%24==0:print('THREE_SCOPE_ONLINE',ds,stream,len(files),1152,round(time.time()-start,1),flush=True);gc.collect()
        assert state_hash(model.state_dict())==mh
        write(BASE/ds/'ONLINE_BARRIER.json',dict(status='sealed',files=files,arrivals=1152,backward_calls=backwards,seconds=time.time()-start,GT_read=False))
        status(BASE/ds/'ONLINE_STATUS.json',dict(status='completed',done=1152,GT_read=False))
    finally:lease.close()

def seal():
    verify();files={};total=0
    for ds in DATASETS:
        for stage in ['FEATURE','EVIDENCE','ONLINE']:
            z=read(BASE/ds/(stage+'_BARRIER.json'));assert z['status']=='sealed' and not z['GT_read'];files.update(z['files'])
            if stage=='ONLINE':total+=z['arrivals']
    assert total==2304
    for f,h in files.items():assert sha(BASE/f)==h,f
    write(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='sealed',files=files,arrivals=total,GT_read=False,time=time.time()))
    status(BASE/'STATUS.json',dict(status='predictions_completed_pending_CPU_audit',predictions=total,GT_read=False))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','features','evidence','online','seal']);p.add_argument('dataset',nargs='?',choices=DATASETS);a=p.parse_args()
    globals()[a.action](a.dataset) if a.action in ['features','evidence','online'] else globals()[a.action]()
