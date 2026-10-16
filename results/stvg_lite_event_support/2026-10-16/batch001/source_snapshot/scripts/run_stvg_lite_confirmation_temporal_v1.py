"""Frozen original caption/alignment protocol, then locked final temporal fitter.

No labels are opened here. Fresh confirmation cases cannot select the recipe.
"""
from pathlib import Path
import gc
import os
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stvg_lite_event_common_v1 import BASE as STUDY,read,write,sha,load_case,save_pt,guard
BASE=STUDY/'confirmation'


def verify():
    runtime=read(BASE/'TEMPORAL_RUNTIME.json')
    assert runtime['status']=='locked' and not runtime['GT_read']
    for rel,h in {**runtime['pins'],**runtime['protected']}.items():assert sha(ROOT/rel)==h,rel
    assert runtime['recipe_sha256']==sha(STUDY/'FINAL_RECIPE_LOCK.json')
    assert runtime['source_barrier_sha256']==sha(BASE/'SOURCE_CAPTURE_BARRIER.json')
    rr=read(BASE/'ROSTER_LOCK.json');bar=read(BASE/'SOURCE_CAPTURE_BARRIER.json')
    assert rr['status']=='locked' and rr['count']==bar['count']==64 and bar['status']=='complete' and not bar['GT_read']
    assert bar['roster_sha256']==sha(BASE/'ROSTER_LOCK.json')
    files={x['case_id']:x for x in bar['files']};assert len(files)==64
    for rec in rr['records']:
        c,p=load_case(rec);item=files[rec['probe_alias']]
        assert item['sha256']==sha(p) and item['source_case_sha256']==c['source_case_sha256'] and not c['GT_read']
    return rr['records'],read(STUDY/'FINAL_RECIPE_LOCK.json')


def run():
    rows,recipe=verify();sys.addaudithook(guard)
    import torch
    from transformers import AutoProcessor
    from scripts import run_stvg_native_instance_teacher_storage002 as teacher
    from scripts import stvg_table1_temporal_adapter_v1 as adapter
    from scripts import run_stvg_table1_source_v1 as source
    from scripts import run_stvg_table1_teacher_v1 as render
    from scripts import run_stvg_table1_native_instance_adapter_v1 as canonical
    from scripts import stvg_temporal_lite_fit_v1 as fitter
    torch.set_num_threads(2)
    teacher.verify_local_model_receipt();teacher.MAX_MEMORY={0:'20GiB','cpu':'4GiB'}
    _,model,torch,device,mapping,load_seconds,peak=teacher._load_qwen(ROOT/teacher.CHECKPOINT_REL,BASE/'offload_temporal')
    model.generation_config.use_cache=True
    processor=AutoProcessor.from_pretrained(ROOT/teacher.CHECKPOINT_REL,local_files_only=True,trust_remote_code=False)
    files=[];times=[];start=time.perf_counter();requests=0;fits=0;fallbacks=0
    for rec in rows:
        cid=rec['probe_alias'];case,path=load_case(rec);_,plan=source._validate_roster_record(rec)
        query=plan['input']['caption'];request=canonical.canonicalize(case,query,BASE/'canonical_cache')
        rp=BASE/'requests'/(cid+'.json');write(rp,request)
        eligible,reason=render.temporal_clip_eligibility(case,request)
        tick=time.perf_counter();fit=None
        if eligible:
            rgb=render.rgb_for_candidates(case,request,case['temporal_candidates'])
            feedback=adapter.adapt_hot_query(case,query,case['temporal_candidates'],model=model,processor=processor,device=device,decoded_rgb_by_physical_id=rgb,frame_map=request['frame_map'])
            requests+=len(feedback['teacher_responses']);del rgb
            rewards=[adapter._alignment_reward(c['canonical_id'],r)for c,r in zip(case['temporal_candidates'],feedback['teacher_responses'])]
            fit=fitter.fit(case,adapter._hidden_from_source_case(case),case['temporal_candidates'],rewards,recipe['selected_temporal'])
            interval=list(fit['final_interval']);fits+=1
        else:
            feedback=dict(status='unavailable_temporal_input_native_fallback',reason=reason,final_interval=case['Native_interval'],GT_read=False,optimizer_steps=0)
            interval=list(case['Native_interval']);fallbacks+=1
        payload=dict(case_id=cid,dataset=rec['dataset'],source_file_sha256=sha(path),source_case_sha256=case['source_case_sha256'],request_sha256=sha(rp),feedback=feedback,selected_temporal_fit=fit,selected_interval=interval,recipe_sha256=sha(STUDY/'FINAL_RECIPE_LOCK.json'),runtime_sha256=sha(BASE/'TEMPORAL_RUNTIME.json'),wall_seconds=time.perf_counter()-tick,GT_read=False)
        item=save_pt(BASE/'temporal'/(cid+'.pt'),payload);item['case_id']=cid;files.append(item)
        times.append(dict(case_id=cid,dataset=rec['dataset'],source_case_sha256=case['source_case_sha256'],selected_interval=interval,temporal_output_sha256=item['sha256'],GT_read=False))
        write(BASE/'TEMPORAL_STATUS.json',dict(status='running',done=len(files),total=64,GT_read=False,time=time.time()),mutable=True)
        print('LITE_CONFIRM_TEMPORAL',len(files),64,flush=True)
        del case,feedback,payload,fit;gc.collect();torch.cuda.empty_cache()
    write(BASE/'TEMPORAL_LOCK.json',dict(status='locked',count=64,records=times,GT_read=False,selection_used_confirmation_GT=False,recipe_sha256=sha(STUDY/'FINAL_RECIPE_LOCK.json')))
    write(BASE/'TEMPORAL_GLOBAL_BARRIER.json',dict(status='complete',count=64,files=files,request_pairs=requests,fit_count=fits,no_fit_count=fallbacks,model_load_seconds=load_seconds,wall_seconds=time.perf_counter()-start,temporal_lock_sha256=sha(BASE/'TEMPORAL_LOCK.json'),runtime_sha256=sha(BASE/'TEMPORAL_RUNTIME.json'),GT_read=False))


def readback():
    rows,recipe=verify()
    import torch
    from scripts import stvg_table1_temporal_adapter_v1 as adapter
    from scripts import stvg_temporal_lite_fit_v1 as fitter
    from scripts.run_stvg_component_temporal005 import _exact_equal
    bar=read(BASE/'TEMPORAL_GLOBAL_BARRIER.json');assert bar['status']=='complete' and bar['count']==64 and not bar['GT_read']
    assert bar['runtime_sha256']==sha(BASE/'TEMPORAL_RUNTIME.json') and bar['temporal_lock_sha256']==sha(BASE/'TEMPORAL_LOCK.json')
    files={x['case_id']:x for x in bar['files']};times={x['case_id']:x for x in read(BASE/'TEMPORAL_LOCK.json')['records']};fits=fallbacks=0
    torch.set_num_threads(2)
    for rec in rows:
        cid=rec['probe_alias'];case,cp=load_case(rec);entry=files[cid];p=ROOT/entry['path'];assert sha(p)==entry['sha256']
        d=torch.load(p,map_location='cpu',weights_only=False)
        assert not d['GT_read'] and d['source_case_sha256']==case['source_case_sha256'] and d['source_file_sha256']==sha(cp)
        assert d['recipe_sha256']==sha(STUDY/'FINAL_RECIPE_LOCK.json') and d['request_sha256']==sha(BASE/'requests'/(cid+'.json'))
        f=d['selected_temporal_fit'];feedback=d['feedback']
        if f is None:
            assert feedback['status']=='unavailable_temporal_input_native_fallback' and d['selected_interval']==case['Native_interval'];fallbacks+=1
        else:
            rewards=[adapter._alignment_reward(c['canonical_id'],r)for c,r in zip(case['temporal_candidates'],feedback['teacher_responses'])]
            repeat=fitter.fit(case,adapter._hidden_from_source_case(case),case['temporal_candidates'],rewards,recipe['selected_temporal'])
            assert _exact_equal(repeat,f) and d['selected_interval']==f['final_interval'];fits+=1
        assert times[cid]['selected_interval']==d['selected_interval'] and times[cid]['temporal_output_sha256']==sha(p)
    write(BASE/'ROOT_TEMPORAL_READBACK.json',dict(status='pass',count=64,fit_exact= fits,no_fit=fallbacks,GT_read=False,CPU_only=True,scope='all source/feedback/selected fitter exact replay; no GT or visual model replay',barrier_sha256=sha(BASE/'TEMPORAL_GLOBAL_BARRIER.json')))


if __name__=='__main__':
    {'run':run,'readback':readback}[sys.argv[1]]()
