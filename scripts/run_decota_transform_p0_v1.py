"""Finite, genuine transformed-input forwards; no labels or gradient updates."""
import sys,os,time,gc,copy,traceback,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_transform_common_v1 import *

def prepare():
    assert not (BASE/'RUNTIME_LOCK.json').exists()
    assert read(ROOT/'artifacts/decota_three_scope_v1/FINAL_COMPLETION.json')['status']=='completed_verified_publication'
    z=old.verify();pins=dict(z['pins'])
    for f in sorted((old.BASE/'revisions').glob('*.json')):pins.update(read(f)['pin_overrides'])
    pins.update({f:sha(ROOT/f) for f in OWN})
    extra=['scripts/run_spatial_ssl_gpu_v1.py','scripts/run_tastvg_decota_c1_same_domain_v1.py',
           'scripts/run_tastvg_full_b1_experts_v1.py','vg_tta/tastvg_deployment_corruption_v2.py',
           'vg_tta/exact_frame_decode_audit_v2.py','vg_tta/tastvg_paper48_hc2_decode_v1.py']
    extra += [str(f.relative_to(ROOT)) for f in (ROOT/'methods/decota_final_simplified_v1').glob('*.py')]
    pins.update({f:sha(ROOT/f) for f in extra})
    cells=[];inputs={}
    for ds in DATASETS:
        p=read(old.BASE/ds/'PLAN.json');write(BASE/ds/'PLAN.json',p)
        inputs[str((BASE/ds/'PLAN.json').relative_to(ROOT))]=sha(BASE/ds/'PLAN.json')
        for split,sp in p['splits'].items():
            for cond in p['conditions']:
                for order,seq in sp['orders'].items():
                    for at,parent in enumerate(seq):
                        refs={}
                        for stream in ['episodic','online100']:
                            f=old.BASE/'online'/ds/stream/split/cond/order/f'{at:05}.pt'
                            refs[stream]=str(f.relative_to(ROOT));inputs[str(f.relative_to(ROOT))]=sha(f)
                            inputs[str(f.with_suffix('.json').relative_to(ROOT))]=sha(f.with_suffix('.json'))
                        cells.append(dict(dataset=ds,split=split,condition=cond,order=order,arrival=at,parent=parent,references=refs))
        for row in p['rows']:
            for cond in p['conditions']:
                _,r=c1.cache(ds,row['pool_parent'],cond,content=False)
                f=c1.POOL/ds/'capture'/cond/f"{row['pool_parent']:05}.json";inputs[str(f.relative_to(ROOT))]=sha(f)
    assert len(cells)==1152
    write(BASE/'COHORT.json',dict(cells=cells));inputs[str((BASE/'COHORT.json').relative_to(ROOT))]=sha(BASE/'COHORT.json')
    write(BASE/'RUNTIME_LOCK.json',dict(version='decota_transform_p0_v1',pins=pins,inputs=inputs,protected=z['protected'],
        unique_inputs=576,logical_cells=1152,spatial_streams=['episodic','online100'],temporal_views=['shift','crop'],
        spatial_views=['flip','dim95'],real_full_view_forward_cap=2304,offset_forward_cap=4608,
        new_expert_calls=0,new_backwards=0,GT_online=False,predecessor_commit='53c76f7affd3c6ffed21c435d92007f8c868db89',time=time.time()))
    status(BASE/'STATUS.json',dict(status='locked_pending_contracts_smoke',GT_read=False,predictions=0))
    archive('配置/原名单/真实四输入变换与代码锁定，尚无新预测')

def real(model,pixels,ids,row):
    from scripts.run_spatial_ssl_gpu_v1 import frozen_forward
    r=copy.deepcopy(row);r['frame_ids']=list(ids);r['input']['frame_ids']=list(ids)
    _,records,replay=frozen_forward(model,pixels,r)
    return records,replay

def run(ds,smoke=False):
    import torch,numpy as np
    from scripts.run_tastvg_decota_c1_same_domain_v1 import model_for,start_gpu,decode_for,data_input
    from scripts.run_tastvg_full_b1_experts_v1 import observation
    from methods.decota_final_simplified_v1.tensors import state_hash,detached
    from methods.decota_final_simplified_v1.objectives import prediction
    from vg_tta.tastvg_decota_c1_same_domain_v1 import NormalizedSpatialReplay
    from vg_tta.decota_transform_p0_v1 import temporal_specs,temporal_pixels,inverse_interval,consensus,spatial_pixels,inverse_boxes,consistency,directional_query
    verify();lease=start_gpu();model=model_for(ds);mh=state_hash(model.state_dict());p=read(BASE/ds/'PLAN.json')
    cells=[c for c in read(BASE/'COHORT.json')['cells'] if c['dataset']==ds]
    index={}
    for c in cells:index.setdefault((c['parent'],c['condition']),[]).append(c)
    rows=p['rows'] if not smoke else [p['rows'][i] for i in p['splits']['search']['orders']['order1'][:2]]
    tick=time.time();full=offsets=replays=done=noop=0;files={};parity=[]
    try:
        decode=decode_for(ds)
        for row in rows:
            frames,ids=decode(row['input']);assert ids==row['frame_ids'];parent=row['ordinal']
            for cond in (['clean'] if smoke else p['conditions']):
                budget();f=BASE/('smoke' if smoke else 'predictions')/ds/cond/f'{parent:05}.pt'
                assert not f.exists(),f
                shifted,pixel,spec=observation(row,cond,frames);data,rc=data_input(ds,row,cond)
                assert pixel==rc['pixel_sha256'];native=data['prediction']['physical_interval']
                refs={};before={};after={};states={};ehashes=set()
                for cell in index[parent,cond]:
                    for stream,path in cell['references'].items():
                        x=old.checked(ROOT/path);assert x['pixel_sha256']==pixel and x['parent']==parent and x['interval']==native
                        key=stream+'_'+cell['order'];refs[key]=dict(path=path,sha256=sha(ROOT/path))
                        states[key+'_before']=x['initial'];states[key+'_after']=x['fit']['state'];before[key]=x['before'];after[key]=x['after']
                        if stream=='episodic':ehashes.add(state_hash(x['fit']['state']))
                assert len(ehashes)==1,'episodic order reuse requires exact state identity'
                if smoke:
                    rec,rp=real(model,shifted,ids,row);full+=1;offsets+=2
                    assert torch.equal(rp.zero['boxes'].cpu(),data['prediction']['boxes'].cpu())
                    assert all(torch.equal(a.cpu(),b.cpu()) for a,b in zip(rp.zero['logits'],data['prediction']['logits']))
                    rp.restore(states['episodic_order1_after'])
                    with torch.no_grad():bv=rp.values()['boxes'].cpu()
                    assert torch.equal(bv,after['episodic_order1'])
                    parity.append(dict(parent=parent,native_bitwise=True,current_corrected_bitwise=True,GT_read=False))
                    del rp,rec,bv
                specs=temporal_specs(ids,native);time_views={}
                for name,speci in specs.items():
                    if not speci['effective']:
                        noop+=1;time_views[name]=dict(spec=speci,interval=native,mapped=dict(interval=native,raw=native,clipped=False,invalid=False,fallback=False),identity_reused=True)
                        continue
                    pp=temporal_pixels(shifted,speci,name);rec,rp=real(model,pp,speci['ids'],row);full+=1;offsets+=2
                    pr=prediction(rp.zero['logits'],rp.zero['boxes'],rec,speci['ids']);mapped=inverse_interval(pr['physical_interval'],speci,name,ids,native)
                    time_views[name]=dict(spec=speci,interval=pr['physical_interval'],mapped=mapped,logits=pr['logits'],records=rec,
                        image_sha256=hashlib.sha256(pp.tobytes()).hexdigest(),identity_reused=False)
                    del rp,pp,rec,pr
                cv=consensus(native,time_views['shift']['mapped']['interval'],time_views['crop']['mapped']['interval'])
                space_views={};native_hash=None
                for name in ['flip','dim95']:
                    pp=spatial_pixels(shifted,name);rec,rp=real(model,pp,ids,row);full+=1;offsets+=2
                    sh=state_hash(rp.initial)
                    if native_hash is None:native_hash=sh
                    assert native_hash==sh
                    results={};reuse={}
                    for key,st in states.items():
                        hh=state_hash(st)
                        if hh not in reuse:
                            rp.restore(st)
                            with torch.no_grad():raw=rp.values()['boxes'].cpu()
                            results[key]=dict(raw=raw,mapped=inverse_boxes(raw.numpy(),name),state_sha256=hh)
                            reuse[hh]=results[key];replays+=1
                        else:results[key]=reuse[hh]
                    space_views[name]=dict(results=results,image_sha256=hashlib.sha256(pp.tobytes()).hexdigest(),full_native_logits=detached(rp.zero['logits'],'cpu'),records=rec)
                    del pp,rp,rec
                cc={}
                for key in before:
                    c0=consistency(space_views['flip']['results'][key+'_before']['mapped'],space_views['dim95']['results'][key+'_before']['mapped'],ids,native)
                    c1v=consistency(space_views['flip']['results'][key+'_after']['mapped'],space_views['dim95']['results'][key+'_after']['mapped'],ids,native)
                    cc[key]=dict(before=c0,after=c1v,delta_event=c1v['event']-c0['event'],delta_full=c1v['full']-c0['full'])
                val=dict(dataset=ds,parent=parent,condition=cond,pixel_sha256=pixel,native_interval=native,frame_ids=ids,
                    temporal_views=time_views,consensus_interval=cv,spatial_views=space_views,consistency=cc,references=refs,
                    directional=directional_query(row['input']['caption']),GT_read=False,new_gradients=0,new_expert=0)
                commit(f,val);files[str(f.relative_to(BASE))]=sha(f);done+=1
                status(BASE/ds/('SMOKE_STATUS.json' if smoke else 'P0_STATUS.json'),dict(status='running',done=done,total=len(rows)*(1 if smoke else 6),
                    actual_full_view_passes=full,actual_offset_forwards=offsets,suffix_replays=replays,seconds=time.time()-tick,pid=os.getpid(),GT_read=False))
                print('TRANSFORM_P0',ds,'smoke' if smoke else 'full',done,len(rows)*(1 if smoke else 6),round(time.time()-tick,1),flush=True)
                del shifted,data,val,space_views,states,before,after;gc.collect()
            del frames;torch.cuda.empty_cache()
        assert state_hash(model.state_dict())==mh
        out=dict(status='pass' if smoke else 'sealed',cells=done,files=files,native_parity=parity,GT_read=False,time=time.time(),seconds=time.time()-tick,
            actual_full_view_passes=full,actual_offset_forwards=offsets,suffix_replays=replays,crop_noop=noop,backwards=0,new_DINO=0,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),source_weights_unchanged=True)
        write(BASE/ds/('SMOKE.json' if smoke else 'PREDICTION_BARRIER.json'),out)
        status(BASE/ds/('SMOKE_STATUS.json' if smoke else 'P0_STATUS.json'),{**out,'status':'completed'})
    finally:lease.close()

def allrun():
    from scripts.test_decota_transform_p0_v1 import run as contracts
    if not (BASE/'CONTRACTS.json').exists():
        write(BASE/'CONTRACTS.json',dict(status='pass',checks=contracts(),GT_read=False,time=time.time()))
    else:assert read(BASE/'CONTRACTS.json')['status']=='pass'
    # Dataset subprocesses avoid accumulating guards and GPU state between phases.
    import subprocess
    for ds in DATASETS:
        for phase in ['smoke','run']:
            barrier=BASE/ds/('SMOKE.json' if phase=='smoke' else 'PREDICTION_BARRIER.json')
            if barrier.exists():
                b=read(barrier);assert b['status']==('pass' if phase=='smoke' else 'sealed') and not b['GT_read']
                for f,h in b['files'].items():assert sha(BASE/f)==h
                status(BASE/ds/('SMOKE_STATUS.json' if phase=='smoke' else 'P0_STATUS.json'),{**b,'status':'completed'})
                continue
            subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B',__file__,phase,ds],cwd=ROOT,check=True)
    bb={ds:read(BASE/ds/'PREDICTION_BARRIER.json') for ds in DATASETS}
    files={k:v for b in bb.values() for k,v in b['files'].items()}
    assert len(files)==576 and all(not b['GT_read'] for b in bb.values())
    write(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='sealed',unique_inputs=576,logical_cells=1152,files=files,GT_read=False,time=time.time(),cost={ds:{k:v for k,v in b.items() if k not in ['files','native_parity']} for ds,b in bb.items()}))
    status(BASE/'STATUS.json',dict(status='completed_predictions_pending_CPU_audit',GT_read=False,time=time.time()))
    archive('576真实变换输入及1152逻辑读出/双空间状态诊断全部封存；待CPU计分审计，不称P0通过')

if __name__=='__main__':
    try:
        if sys.argv[1]=='prepare':prepare()
        elif sys.argv[1]=='all':allrun()
        else:run(sys.argv[2],sys.argv[1]=='smoke')
    except BaseException as e:
        status(BASE/'FAILURE.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),pid=os.getpid(),time=time.time()))
        raise
