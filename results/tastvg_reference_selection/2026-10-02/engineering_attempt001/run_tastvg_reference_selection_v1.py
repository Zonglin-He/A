"""Serial local Sa2VA acquisition; no student inference, optimization, or GT."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
from tastvg_reference_common_v1 import *
import gc,hashlib,traceback,numpy as np

def run():
    import torch
    from PIL import Image
    from transformers import AutoModel,AutoTokenizer
    from vg_tta.exact_frame_decode_audit_v2 import decode as vid_decode
    from vg_tta.tastvg_paper48_hc2_decode_v1 import decode as hc_decode
    from vg_tta.tastvg_spatial_expansion_s0_v1 import mask_boxes
    from scripts.run_tastvg_full_b1_experts_v1 import observation
    from scripts.run_final_simplification_v1 import lease
    from vg_tta.tastvg_reference_selection_v1 import decision
    lock=verify();tick=time.monotonic();handle=lease();sys.addaudithook(guard)
    torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    model_path=ROOT/'checkpoints/Sa2VA-4B'
    for r in read(model_path/'DOWNLOAD_RECEIPT.json')['files']:assert sha(model_path/r['file'])==r['sha256']
    model,info=AutoModel.from_pretrained(str(model_path),torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,use_flash_attn=False,
        trust_remote_code=True,local_files_only=True,output_loading_info=True)
    assert not any(info[k] for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    model=model.eval().cuda().requires_grad_(False)
    tokenizer=AutoTokenizer.from_pretrained(str(model_path),trust_remote_code=True,use_fast=False,local_files_only=True)
    model.preparing_for_generation(tokenizer,max_new_tokens=256,torch_dtype=torch.bfloat16)
    assert all(float(m.fill_hole_area)==0 for m in model.modules() if hasattr(m,'fill_hole_area'))
    calls=0;resources=[];barriers={}
    for ds in DATASETS:
        p=read(BASE/ds/'PLAN.json');old=read(ROOT/p['row_plan']);frames=None;previous=None;receipts=[]
        for cell in p['cells']:
            budget();row=old['rows'][cell['parent']]
            if previous!=cell['parent']:
                frames,ids=(vid_decode if ds=='vidstg' else hc_decode)(row['input']);assert ids==row['frame_ids'];previous=cell['parent']
            shifted,pixel,spec=observation(row,cell['condition'],frames);assert pixel==cell['pixel_sha256']
            x=load(ROOT/cell['payload']);candidate_boxes=[c['prediction']['boxes'] for c in x['update_steps'][0]['candidates']]
            jobs=[('uniform_parity',cell['uniform_positions'])] if cell['cell']==p['uniform_parity_cell'] else []
            jobs.append(('student_routed',cell['router']['positions']))
            for strategy,pos in jobs:
                assert len(pos)==len(set(pos))==5
                start=time.monotonic();samples=[shifted[i] for i in pos]
                digest=hashlib.sha256(row['input']['caption'].encode()+str(pos).encode()+str(shifted.shape).encode()+b''.join(a.tobytes() for a in samples)).hexdigest()
                attempt=BASE/ds/'attempts'/f'{cell["cell"]:03}_{strategy}.json'
                assert calls<lock['max_new_specialist_calls'];calls+=1
                write(attempt,dict(call_number=calls,cell=cell['cell'],strategy=strategy,input_sha256=digest,time=time.time(),GT_read=False))
                prompt='<image>Please segment the object described by: '+row['input']['caption'].rstrip('.')+'.'
                with torch.inference_mode():raw=model.predict_forward(video=[Image.fromarray(im) for im in samples],text=prompt,tokenizer=tokenizer)
                mm=raw['prediction_masks'];mask=mm[0] if len(mm) else None;valid,boxes=mask_boxes(mask,pos,len(ids))
                value=dict(valid=valid,boxes=boxes,positions=pos,mask_count=len(mm),mask_shape=list(np.asarray(mask).shape) if mask is not None else None,
                    mask_bits=np.packbits(np.asarray(mask,dtype=bool)) if mask is not None else None,prediction_text=raw['prediction'],input_sha256=digest,GT_read=False)
                parity=None
                if strategy=='uniform_parity':
                    baseline=load(ROOT/cell['uniform_cache']);assert baseline['input_sha256']==digest
                    assert np.array_equal(valid,baseline['valid']) and np.array_equal(boxes,baseline['boxes'])
                    assert value['mask_shape']==baseline['mask_shape'] and np.array_equal(value['mask_bits'],baseline['mask_bits'])
                    assert value['prediction_text']==baseline['prediction_text'];parity=True
                cache=BASE/ds/'expert_cache'/f'{cell["cell"]:03}_{strategy}.pt';save(cache,value)
                receipt=dict(cell=cell['cell'],strategy=strategy,cache=str(cache.relative_to(BASE)),cache_sha256=sha(cache),
                    input_sha256=digest,pixel_sha256=pixel,positions=pos,valid_frames=int(valid.sum()),mask_count=len(mm),
                    seconds=time.monotonic()-start,decision=decision(candidate_boxes,boxes,valid),uniform_bitwise_parity=parity,GT_read=False)
                rf=BASE/ds/'receipts'/f'{cell["cell"]:03}_{strategy}.json';write(rf,receipt);receipts.append(str(rf.relative_to(BASE)))
                resources.append(dict(dataset=ds,cell=cell['cell'],strategy=strategy,seconds=receipt['seconds']))
                status(BASE/'STATUS.json',dict(status='running_specialist',dataset=ds,cell=cell['cell'],done=calls,total=62,worker_pid=os.getpid(),GT_read=False,time=time.time()))
                print('REFERENCE',ds,cell['cell'],strategy,calls,62,'sec',round(receipt['seconds'],2),flush=True)
                del raw,mm,mask;gc.collect();torch.cuda.empty_cache()
            del shifted
        barrier=BASE/ds/'PREDICTION_BARRIER.json';write(barrier,dict(routed_cells=30,uniform_checks=1,files={r:sha(BASE/r) for r in receipts},GT_read=False,time=time.time()));barriers[ds]=sha(barrier)
    assert calls==62
    verify();write(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(new_specialist_calls=calls,routed_cells=60,datasets=barriers,GT_read=False,time=time.time()))
    write(BASE/'RESOURCES.json',dict(specialist_calls=62,student_forward=0,backward=0,worker_wall_seconds=time.monotonic()-tick,
        peak_vram_bytes=torch.cuda.max_memory_allocated(),calls=resources,wall_includes_IO_and_loading=True))
    status(BASE/'STATUS.json',dict(status='sealed_pending_root_cpu_GT',done=62,total=62,GT_read=False,time=time.time()));handle.close()
if __name__=='__main__':
    try:run()
    except BaseException as e:
        write(BASE/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.time()))
        status(BASE/'STATUS.json',dict(status='failed',error=repr(e),worker_pid=os.getpid(),time=time.time()));raise
