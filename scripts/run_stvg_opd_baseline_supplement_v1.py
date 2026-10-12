"""Serial finite native baseline qualification/dev/formal runs. No GT or experts."""
import collections
import copy
import gc
import os
import sys
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_baseline_supplement_common_v1 import *

def equal_state(a,b):
    from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
    count=dict(tensors=0,tensor_coordinates=0,scalar_values=0); equal(a,b,count); return count

def execute(ds,candidate,phase):
    verify();activate_old(); assert phase in ['qualification','formal_qualification','development','formal']
    import torch
    from scripts.run_decota_paper_main_v1 import gpu,model_for,read_row,frames_for
    from methods.decota_final_simplified_v1.backbone import make_batch
    from methods.decota_final_simplified_v1.tensors import state_hash,detached
    from vg_tta.tastvg_baseline_expansion import decoder_layernorm_scope
    from vg_tta.native_baselines_paper_v1 import live_output
    from vg_tta.decota_paper_baselines_20261005_v1 import OnlineBaseline
    original_state_loader=OnlineBaseline.load_state_dict
    def isolated_state_loader(self,snapshot):
        # PyTorch may retain the CPU Adam step tensor supplied during restore.
        # Protect saved/qualification witnesses from subsequent in-place steps.
        return original_state_loader(self,copy.deepcopy(snapshot))
    OnlineBaseline.load_state_dict=isolated_state_loader
    d=design(); definition=d['candidates'][candidate]; method=definition['method'];cfg=definition['config']; source=d['datasets'][ds]['source']
    dest=BASE/phase/ds/candidate
    if (dest/'PREDICTION_BARRIER.json').exists(): return
    if phase not in ['qualification']: assert read(BASE/'ROOT_QUALIFICATION_READBACK.json')['status']=='pass'
    plan=read(PAPER/ds/'PLAN.json')
    orders=({'order1':d['datasets'][ds]['development_orders']['order1'][:2]} if phase=='qualification' else {o:seq[:2] for o,seq in plan['orders'].items()} if phase=='formal_qualification' else d['datasets'][ds]['development_orders'] if phase=='development' else plan['orders'])
    if phase=='formal': assert read(BASE/'formal_qualification'/ds/candidate/'PREDICTION_BARRIER.json')['status']=='sealed'
    lease=gpu(); model=model_for(source); mh=state_hash(model.state_dict()); names,params,scope=decoder_layernorm_scope(model)
    opt=OnlineBaseline(names,params,method,config=cfg);opt.audit_updates=True;cache=collections.OrderedDict();files={};controls=[];done=0;start=time.time()
    total=sum(map(len,orders.values()))
    try:
        if not (dest/'PARAMETER_SCOPE.json').exists(): write(dest/'PARAMETER_SCOPE.json',dict(names=names,count=sum(p.numel() for p in params),scope=scope,source_model_state_sha256=mh,config=cfg))
        def arrival(batch,ids,subject):
            before=detached(opt.state_dict(),'cpu'); evidence=[]
            def closure():
                native,_=live_output(model,batch,ids,subject)
                evidence.append([dict(sted=o.sted.detach().cpu().clone(),actionness_logits=o.actionness_logits.detach().cpu().clone(),valid=o.valid.detach().cpu().clone(),frame_ids=list(o.frame_ids)) for o in native])
                return native
            pre,post,a=opt.arrive(closure,lambda:live_output(model,batch,ids,subject)[1])
            z=dict(method=method,config=cfg,Before=detached(pre,'cpu'),After=detached(post,'cpu'),audit=detached(a,'cpu'),
                state_before=before,state_after=detached(opt.state_dict(),'cpu'),closure_evidence=evidence,GT_read=False)
            return z
        for order,seq in orders.items():
            opt.reset(); previous=None; previous_state=None
            for at,q in enumerate(seq):
                from scripts.decota_paper_common_v1 import budget
                budget()
                row=read_row(ds,q); path=dest/order/f'{at:05}.pt'
                if path.exists():
                    rc=read(path.with_suffix('.json')); assert sha(path)==rc['sha256'] and not rc['GT_read']
                    z=load(path); assert z['query_ordinal']==q and z['config']==cfg and z['previous_payload_sha256']==previous
                    if previous_state is not None: equal_state(previous_state,z['state_before'])
                    opt.load_state_dict(z['state_after']); assert audit_baseline(z)==z['math_audit']
                else:
                    tick=time.perf_counter(); frames,ids,reuse=frames_for(ds,row,cache); batch=make_batch(frames,ids,row['input'],model); decode=time.perf_counter()-tick
                    subject=row['parses']['subject']; initial=detached(opt.state_dict(),'cpu')
                    zero=None
                    if phase in ['qualification','formal_qualification']:
                        with torch.no_grad(): zero=detached(live_output(model,batch,ids,subject)[1],'cpu')
                        lr=opt.optimizer.param_groups[0]['lr'];opt.optimizer.param_groups[0]['lr']=0.
                        zr=arrival(batch,ids,subject); assert torch.equal(zr['After']['boxes'],zero['boxes']) and zr['After']['indices']==zero['indices'] and zr['After']['physical_interval']==zero['physical_interval']
                        assert all(torch.equal(a,b) for a,b in zip(initial['parameters'],zr['state_after']['parameters']))
                        opt.load_state_dict(initial);opt.optimizer.param_groups[0]['lr']=lr
                        save(dest/'lr0_controls'/order/f'{at:05}.pt',zr)
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter(); z=arrival(batch,ids,subject);torch.cuda.synchronize();seconds=time.perf_counter()-tick
                    z.update(dataset=ds,source=source,phase=phase,candidate=candidate,order=order,arrival=at,query_ordinal=q,
                        frame_ids=ids,input_metadata_sha256=hashlib.sha256(__import__('json').dumps(row['input'],sort_keys=True).encode()).hexdigest(),
                        previous_payload_sha256=previous,runtime_sha256=runtime_sha(),
                        compute=dict(decode_seconds=decode,model_seconds=seconds,CUDA_peak_allocated=torch.cuda.max_memory_allocated(),CUDA_peak_reserved=torch.cuda.max_memory_reserved(),new_DINO_calls=0,decoded_frame_reuse=reuse))
                    failed=dest/'unaccepted'/order/f'{at:05}.pt'
                    # Preserve complete evidence before CPU checks; never accept failed arithmetic.
                    save(failed,z); z['math_audit']=audit_baseline(z)
                    if previous_state is not None: equal_state(previous_state,z['state_before'])
                    if phase in ['qualification','formal_qualification']:
                        end=detached(opt.state_dict(),'cpu');opt.load_state_dict(initial);repeat=arrival(batch,ids,subject)
                        cf=dest/'repeat_controls'/order/f'{at:05}.pt';save(cf,repeat)
                        pair=equal_state({k:z[k] for k in ['Before','After','audit','state_before','state_after','closure_evidence']},{k:repeat[k] for k in ['Before','After','audit','state_before','state_after','closure_evidence']})
                        assert audit_baseline(repeat)==z['math_audit']; opt.load_state_dict(end)
                        controls.append(dict(order=order,arrival=at,query_ordinal=q,lr0_exact_native=True,complete_repeat_bitwise=pair,repeat_path=str(cf.relative_to(ROOT)),repeat_sha256=sha(cf),actual_arrive_calls=3,GT_read=False))
                    elif at<2 and (phase=='formal' or (phase=='development' and order=='order1')):
                        qf=BASE/('formal_qualification' if phase=='formal' else 'qualification')/ds/candidate/order/f'{at:05}.pt'; qz=load(qf)
                        counts=equal_state({k:z[k] for k in ['Before','After','audit','state_before','state_after','closure_evidence']},{k:qz[k] for k in ['Before','After','audit','state_before','state_after','closure_evidence']})
                        write(dest/'first_formal'/order/f'{at:05}.json',dict(status='pass',complete_bitwise=counts,qualified_sha256=sha(qf),GT_read=False,time=time.time()))
                    save(path,z);write(path.with_suffix('.json'),dict(sha256=sha(path),bytes=path.stat().st_size,GT_read=False,time=time.time()))
                    del frames,batch
                files[str(path.relative_to(ROOT))]=sha(path);previous=sha(path);previous_state=z['state_after'];done+=1
                del z;gc.collect();torch.cuda.empty_cache()
                status(dest/'STATUS.json',dict(status='running',done=done,total=total,pid=os.getpid(),phase=phase,order=order,arrival=at,elapsed_seconds=time.time()-start,GT_read=False,time=time.time()))
                if done%4==0: print('BASELINE_SUPPLEMENT_PROGRESS',phase,ds,candidate,done,total,flush=True)
            opt.reset();assert state_hash(model.state_dict())==mh
        assert done==len(files)==total
        write(dest/'PREDICTION_BARRIER.json',dict(status='sealed',dataset=ds,candidate=candidate,method=method,config=cfg,phase=phase,arrivals=done,files=files,orders=orders,actual_source_model_hash=mh,source_restored_after_all_orders=True,qualification_controls=controls,GT_read=False,runtime_sha256=runtime_sha(),time=time.time()))
        status(dest/'STATUS.json',dict(status='sealed',done=done,total=total,GT_read=False,time=time.time()))
    finally: OnlineBaseline.load_state_dict=original_state_loader;lease.close()

def root_qualification():
    verify();count=checks=coords=0; records=[]
    for ds in design()['datasets']:
        for candidate in design()['candidates']:
            f=BASE/'qualification'/ds/candidate; b=read(f/'PREDICTION_BARRIER.json'); assert b['arrivals']==2 and len(b['qualification_controls'])==2
            for c in b['qualification_controls']:
                p=f/c['order']/f"{c['arrival']:05}.pt";z=load(p);assert sha(p)==b['files'][str(p.relative_to(ROOT))]
                assert audit_baseline(z)==z['math_audit']; repeat=load(ROOT/c['repeat_path']);assert sha(ROOT/c['repeat_path'])==c['repeat_sha256']
                keys=['Before','After','audit','state_before','state_after','closure_evidence']
                result=equal_state({k:z[k] for k in keys},{k:repeat[k] for k in keys});assert result==c['complete_repeat_bitwise']
                checks+=z['math_audit']['optimizer']['scalar_checks'];coords+=result['tensor_coordinates'];count+=1
                records.append(dict(dataset=ds,candidate=candidate,arrival=c['arrival'],sha256=sha(p),math_status='pass',repeat_status='pass',lr0_status='pass'))
    assert count==36
    write(BASE/'ROOT_QUALIFICATION_READBACK.json',dict(status='pass',qualification_arrivals=count,actual_arrive_calls=count*3,independent_optimizer_scalar_checks=checks,bitwise_tensor_coordinates=coords,formal_predictions_accepted=0,GT_read=False,records=records,time=time.time()))

if __name__=='__main__':
    try:
        if sys.argv[1]=='root_qualification': root_qualification()
        else: execute(sys.argv[2],sys.argv[3],sys.argv[1])
    except BaseException:
        p=BASE/'failures'/str(time.time_ns());p.mkdir(parents=True);(p/'traceback.txt').write_text(traceback.format_exc());status(BASE/'STATUS.json',dict(status='failed_preserved_pending_actual_root',evidence=str(p.relative_to(ROOT)),GT_read=False,time=time.time()));raise
