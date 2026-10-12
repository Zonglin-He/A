"""Actual frozen-only P7-A qualification and missing temporal capture."""
import collections,gc,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_temporal_guided_p7_common_v1 import *

def one(model,r,cache):
    import torch
    from scripts.run_decota_paper_main_v1 import read_row,frames_for
    from scripts.stvg_opd_paper_inputs_v1 import observation
    from scripts.stvg_opd_legacy_input_bridge003 import normalized
    from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward
    from methods.decota_final_simplified_v1.objectives import prediction
    from methods.decota_final_simplified_v1.tensors import detached
    raw=read_row(r['dataset'],r['query_ordinal']);frames,ids,_=frames_for(r['dataset'],raw,cache)
    shifted,pixel,spec=observation(r['dataset'],raw,'clean',frames)
    batch,records,base=frozen_forward(model,shifted,raw)
    z=prediction(base.zero['logits'],base.zero['boxes'],records,ids);ip=ROOT/r['original_Full']['input_path']
    assert sha(ip)==r['original_Full']['input_sha256'];inp=normalized(load(ip))
    assert pixel==inp['pixel_sha256'] and spec==inp['corruption_spec'] is None and ids==inp['frame_ids']
    assert model._fixed_full_state_hash==inp['source_model_state_sha256'] and z['physical_interval']==inp['interval']
    assert torch.equal(z['boxes'].cpu(),inp['native_boxes']) and not any(p.requires_grad or p.grad is not None for p in model.parameters())
    capture=dict(records=records,frame_ids=ids,zero=detached({k:base.zero[k] for k in ['boxes','logits','actions']},'cpu'),
        source_model_state_sha256=model._fixed_full_state_hash)
    evidence=dict(index=r['index'],dataset=r['dataset'],source=r['source'],query_ordinal=r['query_ordinal'],parent_id=r['parent_id'],
        previous_P6=r['previous_P6'],capture=capture,pixel_sha256=pixel,frame_ids=ids,corruption_spec=spec,
        Native_interval=z['physical_interval'],input_binding=r['original_Full'],GT_read=False,
        new_DINO_calls=0,optimizer_steps=0,new_spatial_fits=0,backwards=0,runtime_sha256=sha(BASE/'RUNTIME.json'))
    evidence['Projection_interval'],evidence['projection_check']=projection(evidence)
    del frames,shifted,batch,base;gc.collect();torch.cuda.empty_cache()
    return evidence

def compared(a,b):
    from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
    n=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
    for key in ['capture','pixel_sha256','frame_ids','Native_interval','Projection_interval']:equal(a[key],b[key],n)
    return n

def p6_control(z,r):
    import torch
    p=ROOT/r['P6_binding']['path'];assert sha(p)==r['P6_binding']['sha256'];old=load(p)
    from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
    n=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
    c=old['capture'];current=z['capture']
    for k in ['records','frame_ids','source_model_state_sha256']:equal(c[k],current[k],n)
    for k in ['boxes','logits','actions']:equal(c['zero'][k],current['zero'][k],n)
    assert z['pixel_sha256']==old['pixel_sha256'] and z['Native_interval']==old['predictions']['Native']['physical_interval']
    assert z['Projection_interval']==old['predictions']['ActionnessProjection']['physical_interval']
    return n

def run(qualification):
    import torch
    from scripts.stvg_opd_design_ablation_common_v1 import activate_old
    activate_old()
    from scripts.run_decota_paper_main_v1 import model_for
    from methods.decota_final_simplified_v1.tensors import state_hash
    verify();d=read(BASE/'DESIGN_LOCK.json');records=d['captures']
    selection=[records[i] for i in d['qualification_indices']] if qualification else [r for r in records if not r['previous_P6']]
    if not qualification:assert read(A/'ROOT_QUALIFICATION.json')['status']=='pass'
    lease_=lease(A);model=None;dataset=None;cache=collections.OrderedDict();files={};count=0;start=time.time()
    try:
        for r in selection:
            if dataset!=r['dataset']:
                if model is not None:
                    assert state_hash(model.state_dict())==model_hash;del model;cache.clear();gc.collect();torch.cuda.empty_cache()
                model=model_for(r['source']);model_hash=model._fixed_full_state_hash;dataset=r['dataset']
            repeats=[0,1] if qualification else [0]
            for repeat in repeats:
                path=A/('qualification' if qualification else 'capture')/f"{r['index']:05}"/f'repeat{repeat}.pt'
                assert not path.exists();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
                evidence=one(model,r,collections.OrderedDict() if qualification else cache)
                evidence['compute']=dict(actual_frozen_capture_seconds=time.perf_counter()-tick,CUDA_peak_allocated=torch.cuda.max_memory_allocated(),
                    actual_temporal_capture_forwards=1,actual_DINO_calls=0,backwards=0,optimizer_steps=0)
                save(path.parent/('UNACCEPTED_'+path.name),evidence)
                if r['previous_P6']:evidence['original_P6_exact']=p6_control(evidence,r)
                if repeat:evidence['complete_repeat_bitwise']=compared(evidence,load(path.parent/'repeat0.pt'))
                if not qualification and r['index'] in d['qualification_indices']:
                    evidence['qualified_formal_bitwise']=compared(evidence,load(A/'qualification'/f"{r['index']:05}"/'repeat0.pt'))
                save(path,evidence);h=sha(path);write(path.with_suffix('.json'),dict(status='accepted_complete_frozen_temporal_capture',
                    sha256=h,bytes=path.stat().st_size,GT_read=False,actual_GPU_capture_forwards=1,actual_new_spatial_fits=0,time=time.time()))
                files[str(path.relative_to(ROOT))]=h;count+=1
                status(A/'GPU_STATUS.json',dict(status='running_qualification' if qualification else 'running_missing_capture',
                    dataset=dataset,index=r['index'],repeat=repeat,done=count,total=8 if qualification else 192,pid=os.getpid(),GT_read=False,time=time.time()))
                print('P7A_FROZEN_CAPTURE',qualification,count,8 if qualification else 192,dataset,r['index'],flush=True)
                del evidence;gc.collect();torch.cuda.empty_cache()
        assert count==(8 if qualification else 192) and state_hash(model.state_dict())==model_hash
        write(A/('QUALIFICATION_BARRIER.json' if qualification else 'TEMPORAL_CAPTURE_BARRIER.json'),dict(status='pass' if qualification else 'sealed',
            scope='four fixed source-bound queries, two fresh complete capture repeats' if qualification else 'all192 missing frozen temporal captures',
            files=files,capture_forwards=count,new_spatial_fits=0,optimizer_steps=0,backwards=0,new_DINO_calls=0,
            source_state_restored=True,GT_read=False,elapsed_seconds=time.time()-start,time=time.time()))
    finally:
        if model is not None:del model
        cache.clear();gc.collect();torch.cuda.empty_cache();lease_.close()

def qualification_root():
    verify();b=read(A/'QUALIFICATION_BARRIER.json');assert b['status']=='pass' and b['capture_forwards']==8
    count=0;projection_values=0;old_controls=0
    for p,h in b['files'].items():
        f=ROOT/p;assert sha(f)==h;z=load(f);r=read(BASE/'DESIGN_LOCK.json')['captures'][z['index']]
        interval,check=projection(z);assert interval==z['Projection_interval'];projection_values+=sum(x.numel() for x in z['capture']['zero']['logits'])
        compared(z,load(f.parent/'repeat0.pt'))
        if z['previous_P6']:p6_control(z,r);old_controls+=1
        assert not z['GT_read'] and z['backwards']==z['optimizer_steps']==z['new_spatial_fits']==z['new_DINO_calls']==0;count+=1
    write(A/'ROOT_QUALIFICATION.json',dict(status='pass',scope='all8 actual GPU frozen captures/whole repeat/source/P6 exact and independent unchanged projection',
        actual_GPU_capture_forwards=count,P6_exact_controls=old_controls,projection_values=projection_values,
        new_spatial_fits=0,new_DINO_calls=0,optimizer_steps=0,backwards=0,CPU_only=True,GT_read=False,time=time.time()))
    print('P7A_QUALIFICATION_ROOT_PASS',count,flush=True)

def assemble():
    import torch
    from scripts.stvg_opd_legacy_input_bridge003 import normalized
    verify();d=read(BASE/'DESIGN_LOCK.json');b=read(A/'TEMPORAL_CAPTURE_BARRIER.json');assert b['status']=='sealed' and b['capture_forwards']==192
    binding={};sourcefiles={};oldcount=0
    for r in d['captures']:
        if r['previous_P6']:
            p=ROOT/r['P6_binding']['path'];assert sha(p)==r['P6_binding']['sha256'];old=load(p)
            z=dict(dataset=r['dataset'],capture=old['capture'],pixel_sha256=old['pixel_sha256'],frame_ids=old['frame_ids'],
                Native_interval=old['predictions']['Native']['physical_interval']);z['Projection_interval'],_=projection(z)
            assert z['Projection_interval']==old['predictions']['ActionnessProjection']['physical_interval'];oldcount+=1
        else:
            p=A/'capture'/f"{r['index']:05}"/'repeat0.pt';assert sha(p)==b['files'][str(p.relative_to(ROOT))];z=load(p)
        inp=normalized(load(ROOT/r['original_Full']['input_path']));assert inp['pixel_sha256']==z['pixel_sha256'] and inp['frame_ids']==z['frame_ids']
        assert inp['source_model_state_sha256']==z['capture']['source_model_state_sha256'] and inp['interval']==z['Native_interval']
        assert torch.equal(inp['native_boxes'],z['capture']['zero']['boxes'].cpu())
        binding[r['index']]=dict(path=str(p.relative_to(ROOT)),sha256=sha(p),T0=z['Native_interval'],T1=z['Projection_interval']);sourcefiles[str(p.relative_to(ROOT))]=sha(p)
    assert oldcount==64 and len(binding)==256
    files={}
    for r in d['records']:
        ip=ROOT/r['original_Full']['input_path'];assert sha(ip)==r['original_Full']['input_sha256'];inp=normalized(load(ip))
        tb=binding[r['capture_index']];assert inp['interval']==tb['T0'];boxes=dict(Native=inp['native_boxes']);fits={};initial=None
        for arm,oldarm in [('OPD','Full'),('Direct','Direct'),('Capped_Direct','Direct_capped')]:
            p=O/'common_state'/f"{r['index']:05}"/(oldarm+'.pt');rc=read(p.with_suffix('.json'));assert sha(p)==rc['sha256'] and not rc['GT_read']
            z=load(p);assert z['input_sha256']==r['original_Full']['input_sha256'] and z['interval']==tb['T0']
            if initial is None:initial=z['fit']['initial']
            else:assert all(torch.equal(v,z['fit']['initial'][k]) for k,v in initial.items())
            boxes[arm]=z['fit']['final'];fits[arm]=dict(path=str(p.relative_to(ROOT)),sha256=sha(p));sourcefiles[str(p.relative_to(ROOT))]=sha(p)
        p=A/'readouts'/f"{r['public_cell']:05}.pt";save(p,dict(**r,boxes=boxes,temporal_binding=tb,spatial_bindings=fits,
            frame_ids=inp['frame_ids'],intervals=dict(T0=tb['T0'],T1=tb['T1']),GT_read=False))
        h=sha(p);files[str(p.relative_to(ROOT))]=h;write(p.with_suffix('.json'),dict(status='accepted_eight_readouts',sha256=h,bytes=p.stat().st_size,outputs=8,GT_read=False,time=time.time()))
        sourcefiles[str(ip.relative_to(ROOT))]=sha(ip)
        status(A/'CPU_STATUS.json',dict(status='assembling_all512_eight_readouts',done=r['public_cell']+1,total=512,CPU_only=True,GT_read=False,time=time.time()))
    write(A/'GLOBAL_READOUT_BARRIER.json',dict(status='sealed',scope='all clean512 fixed cells and4096 eight-way outputs before new GT scoring',
        files=files,source_files=sourcefiles,unique_temporal_captures=256,new_GPU_formal_captures=192,exact_reused_P6_captures=64,
        cells=512,logical_outputs=4096,parents=256,new_spatial_fits=0,GT_read=False,runtime_sha256=sha(BASE/'RUNTIME.json'),time=time.time()))
    print('P7A_GLOBAL_SEALED',512,4096,flush=True)

if __name__=='__main__':
    job=sys.argv[1]
    try:{'qualification':lambda:run(True),'formal':lambda:run(False),'qualification_root':qualification_root,'assemble':assemble}[job]()
    except BaseException:
        p=A/'failures'/str(time.time_ns());p.mkdir(parents=True);(p/'traceback.txt').write_text(traceback.format_exc())
        status(A/'STATUS.json',dict(status='failed_preserved_pending_actual_root',stage=job,evidence=str(p.relative_to(ROOT)),GT_read=False,time=time.time()));raise
