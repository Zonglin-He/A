"""Only requested GT-event images reach the unchanged frozen Sa2VA interface."""
import os,sys,time,hashlib,gc,traceback,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
from scripts.tastvg_oracle_event5_common_v1 import *
def run():
    import torch,numpy as np
    from PIL import Image
    from scripts.run_final_simplification_v1 import lease
    from scripts.run_tastvg_full_b1_experts_v1 import observation
    from vg_tta.tastvg_spatial_expansion_s0_v1 import mask_boxes
    from vg_tta.exact_frame_decode_audit_v2 import decode as vd
    from vg_tta.tastvg_paper48_hc2_decode_v1 import decode as hd
    verified();assert read(BASE/'EXPERIMENT1_ROOT_AUDIT.json')['status']=='pass'
    sys.addaudithook(guard);lh=lease();tick=time.monotonic();torch.set_num_threads(4)
    torch.manual_seed(20260929);np.random.seed(20260929);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    registry={};calls=reuses=unsupported=0;model=tokenizer=None
    events=read(BASE/'EVENT_PLAN.json')['events'];assert len(events)==288
    for c in events:
        for branch in ['U','U2','R']:
            _,r=cached_evidence(c,branch);rinput=r['input_sha256']
            if rinput not in registry:registry[rinput]=(ROOT/r['cache'],r['cache_sha256'],'old_'+branch)
    def model_ready():
        nonlocal model,tokenizer
        if model is not None:return
        from transformers import AutoModel,AutoTokenizer
        mp=ROOT/'checkpoints/Sa2VA-4B'
        for r in read(mp/'DOWNLOAD_RECEIPT.json')['files']:assert sha(mp/r['file'])==r['sha256']
        for f,h in read(mp/'OFFICIAL_CODE_RECEIPT.json')['files'].items():assert sha(mp/f)==h['sha256']
        model,info=AutoModel.from_pretrained(str(mp),torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,
            use_flash_attn=False,trust_remote_code=True,local_files_only=True,output_loading_info=True)
        assert not any(info[k] for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
        model=model.eval().cuda().requires_grad_(False)
        tokenizer=AutoTokenizer.from_pretrained(str(mp),trust_remote_code=True,use_fast=False,local_files_only=True)
        model.preparing_for_generation(tokenizer,max_new_tokens=256,torch_dtype=torch.bfloat16)
        assert all(float(m.fill_hole_area)==0 for m in model.modules() if hasattr(m,'fill_hole_area'))
    for done,c in enumerate(events,1):
        budget();ds=c['dataset'];row=plan(ds)['rows'][c['parent']];path=BASE/ds/'expert_receipts'/f'{prefix(c)}.json'
        if path.exists():
            r=read(path)
            if r['eligible']:assert sha(ROOT/r['cache'])==r['cache_sha256']
            continue
        if not c['eligible']:
            write(path,dict(eligible=False,available=c['available'],reason=c['reason'],positions=[],new_call=False,
                GT_assisted_observation=True,raw_GT_read=False,time=time.time()));unsupported+=1;continue
        from vg_tta import exact_frame_decode_audit_v2 as binding
        binding.decode=hd if ds=='hc2' else vd
        frames,ids=(hd if ds=='hc2' else vd)(row['input']);assert ids==row['frame_ids']
        shifted,pixel,_=observation(row,c['condition'],frames);assert pixel==c['pixel_sha256']
        pos=c['positions'];assert len(pos)==len(set(pos))==5
        samples=[shifted[i] for i in pos]
        digest=hashlib.sha256(row['input']['caption'].encode()+str(pos).encode()+str(shifted.shape).encode()+b''.join(a.tobytes() for a in samples)).hexdigest()
        cf=BASE/ds/'expert_cache'/f'{digest}.pt';start=time.monotonic();fresh=digest not in registry
        if cf.exists():registry[digest]=(cf,sha(cf),'new_GT_event_existing');fresh=False
        if fresh:
            model_ready();ledger=BASE/'EXPERT_ATTEMPTS.jsonl'
            attempts=[json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
            assert len(attempts)<288 and all(a['input_sha256']!=digest for a in attempts)
            attempt=dict(attempt=len(attempts),dataset=ds,input_sha256=digest,cell=cell_key(c),
                requested_frames=5,GT_assisted_observation=True,time=time.time())
            with ledger.open('a') as f:f.write(json.dumps(attempt)+'\n');f.flush();os.fsync(f.fileno())
            prompt='<image>Please segment the object described by: '+row['input']['caption'].rstrip('.')+'.'
            with torch.inference_mode():raw=model.predict_forward(video=[Image.fromarray(a) for a in samples],text=prompt,tokenizer=tokenizer)
            masks=raw['prediction_masks'];mask=masks[0] if len(masks) else None;valid,boxes=mask_boxes(mask,pos,len(ids))
            assert np.isfinite(boxes).all() and all(not valid[i] for i in range(len(ids)) if i not in pos)
            value=dict(valid=valid,boxes=boxes,positions=pos,mask_count=len(masks),
                mask_shape=list(np.asarray(mask).shape) if mask is not None else None,
                mask_bits=np.packbits(np.asarray(mask,dtype=bool)) if mask is not None else None,
                prediction_text=raw['prediction'],input_sha256=digest,GT_assisted_observation=True,raw_GT_read=False)
            save(cf,value);registry[digest]=(cf,sha(cf),'new_GT_event');calls+=1
        else:cf,h,origin=registry[digest];assert sha(cf)==h;value=load(cf);reuses+=1
        assert value['input_sha256']==digest and value['positions']==pos
        write(path,dict(eligible=True,available=c['available'],positions=pos,cache=str(cf.relative_to(ROOT)),
            cache_sha256=sha(cf),input_sha256=digest,pixel_sha256=pixel,new_call=fresh,origin=registry[digest][2],
            valid_frames=int(value['valid'].sum()),seconds=time.monotonic()-start,GT_assisted_observation=True,raw_GT_read=False,time=time.time()))
        status(BASE/'EXPERT_STATUS.json',dict(status='running',done=done,total=288,new_calls=calls,reused=reuses,
            worker_pid=os.getpid(),dataset=ds,raw_GT_read=False,GT_assisted_observation=True,time=time.time()))
        print('GT_EVENT_OBSERVATIONS',done,288,ds,flush=True);del frames,shifted;gc.collect();torch.cuda.empty_cache()
    verified();write(BASE/'EXPERT_RESOURCES.json',dict(logical_requests=288,eligible=sum(c['eligible'] for c in events),
        unsupported=sum(not c['eligible'] for c in events),new_calls=calls,reused=reuses,
        worker_wall_seconds=time.monotonic()-tick,peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(),
        wall_includes_loading_IO=True,extra_expert_control_calls=0,raw_GT_read=False,GT_assisted_observation=True,time=time.time()))
    status(BASE/'EXPERT_STATUS.json',dict(status='completed',done=288,total=288,new_calls=calls,time=time.time()));lh.close()
if __name__=='__main__':
    try:run()
    except BaseException as e:
        status(BASE/'EXPERT_STATUS.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
