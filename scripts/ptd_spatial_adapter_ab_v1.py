"""Isolated PTD research driver. prepare reads historical metadata; capture has no GT."""
import argparse, gc, hashlib, importlib.util, json, math, os, re, sys, time
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.decota_matrix_common_v1 import read,write,save,load,sha
OUT=ROOT/'artifacts/ptd_spatial_adapter_ab_v1'
PTD=ROOT/'external/ParallelTubeDecoding'
CK=ROOT/'checkpoints/ParallelTubeDecoding-Qwen3-VL-4B'
CONDITIONS=['clean','noise_medium','defocus_extreme']

def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def path(row,condition,view='original'):return OUT/'cache'/condition/(row['key'].replace(':','_')+'_'+view+'.pt')

def prepare():
    if (OUT/'LOCK.json').exists():return read(OUT/'LOCK.json')
    parent=read(ROOT/'artifacts/spatial_consolidation_roles_v1/LOCK.json');rows=[]
    for cohort in ['hcstvg1_test','vidstg_test']:
        pool=sorted([r for r in parent['rows'].values() if r['cohort']==cohort],key=lambda r:digest('ptd-ab-v1|'+cohort+'|'+r['source']))
        assert len(pool)>=40
        for i,r in enumerate(pool[:40]):
            assert sha(r['path'])==r['sha256'];x=load(r['path']);q=x['input'];n=len(q['frame_ids'])
            pos=np.unique(np.linspace(0,n-1,min(n,32)).round().astype(int)).tolist()
            rows.append(dict(key=r['key'],source=r['source'],cohort=cohort,split='development' if i<8 else 'confirmation',
                             smoke=i<2,p0=i<4,ordinal=i,input={**q,'frame_ids':[q['frame_ids'][j] for j in pos]},
                             parent_positions=pos,parent_frame_ids=q['frame_ids'],parent_file=r['path'],parent_sha=r['sha256']))
    assert len({r['source'] for r in rows})==80
    assert len({r['input']['video_sha256'] for r in rows})==80
    write(OUT/'INPUTS.json',rows)
    protected=['methods/CURRENT_METHOD.json','methods/CURRENT_WORKING_METHOD.json','methods/C1_FINAL_RESEARCH_CONFIG.json',
               'methods/C1_TEMPORAL_RESEARCH_STATUS.json','artifacts/c1_fresh_confirmation_v1/vid_train_audit/PROSPECTIVE_SOURCE_RESERVATION.json']
    deps=['scripts/ptd_spatial_adapter_ab_v1.py','vg_tta/ptd_spatial_adapter_ab_v1.py','protocols/ptd_spatial_adapter_ab_v1.md',
          'scripts/c1_controlled_corruption_v1.py','scripts/corruption_route_retest_v1.py',
          'external/ParallelTubeDecoding/src/model/ptd_generation.py','external/ParallelTubeDecoding/src/dataset/data_utils.py']
    lock=dict(created=time.time(),inputs_sha=sha(OUT/'INPUTS.json'),labels=parent['labels'],labels_sha=parent['labels_sha256'],
              frames=32,long_edge=448,rank=16,seed=20260924,conditions=CONDITIONS,
              checkpoint=read(CK/'OFFICIAL_RECEIPT.json'),teacher_revision='ebb281ec70b05090aa6165b016eac8ec08e71b17',
              pins={f:sha(ROOT/f) for f in deps},protected={f:sha(ROOT/f) for f in protected},
              max_seconds=21600,stage_seconds=7200,max_bytes=20*2**30,peak_bytes=28*2**30,
              historical_exposure=True,GT_worker=False,GT_p0_only=True,cross_video_state=False)
    write(OUT/'LOCK.json',lock);return lock

def verify():
    p=read(OUT/'LOCK.json');assert sha(OUT/'INPUTS.json')==p['inputs_sha']
    for f,h in p['protected'].items():assert sha(ROOT/f)==h,f
    pins=dict(p['pins'])
    for f in sorted((OUT/'amendments').glob('*.json')):pins.update(read(f).get('pins',{}))
    for f,h in pins.items():assert sha(ROOT/f)==h,f
    return p

def processor_load():
    sys.path.insert(0,str(PTD/'src'))
    spec=importlib.util.spec_from_file_location('ptd_ab_data_utils',PTD/'src/dataset/data_utils.py')
    d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
    from transformers import AutoProcessor
    pr=AutoProcessor.from_pretrained(CK,local_files_only=True)
    d.patch_qwen3_video_processor(pr);d.patch_processor_with_time_tokens(pr)
    return pr

def model_load():
    from train.monkey_patch_forward import replace_qwen3_with_ptd_forward
    replace_qwen3_with_ptd_forward()
    from transformers import AutoModelForImageTextToText
    model,info=AutoModelForImageTextToText.from_pretrained(CK,local_files_only=True,dtype=torch.bfloat16,
                            attn_implementation='sdpa',device_map='cuda',output_loading_info=True)
    assert not info['missing_keys'] and not info['unexpected_keys']
    return model.eval().requires_grad_(False)

def frames_for(row,condition,view='original'):
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from scripts.corruption_route_retest_v1 import corrupt
    frames,ids=decode(row['input']);frames=corrupt(frames,ids,row['source'],condition)
    if view!='original':
        gamma=float(view.removeprefix('gamma'))
        frames=np.rint(255*np.power(frames.astype(np.float32)/255,gamma)).clip(0,255).astype(np.uint8)
    return frames,ids

def inputs_for(row,processor,frames):
    from transformers.video_utils import VideoMetadata
    q=row['input'];h,w=frames.shape[1:3];scale=min(1.,448/max(h,w))
    hh=max(32,min(448,round(h*scale/32)*32));ww=max(32,min(448,round(w*scale/32)*32))
    x=torch.from_numpy(frames).permute(0,3,1,2).float()
    x=torch.nn.functional.interpolate(x,size=(hh,ww),mode='bilinear',align_corners=False,antialias=True)
    query=q['caption'].strip()
    if not query.endswith(('.', '?', '!')):query+='.'
    text=f"Given the query: '{query}' Localize the described object throughout the video. Use object reference tokens, time tokens, and box tokens. Return the object reference, event time segment, and per-time bbox coordinates."
    msg=[dict(role='user',content=[dict(type='video'),dict(type='text',text=text)])]
    prompt=processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
    meta=VideoMetadata(total_num_frames=q['frame_count'],fps=q['fps'],frames_indices=q['frame_ids'],width=w,height=h,video_backend='exact_ffmpeg_frame_ids')
    z=processor(text=[prompt],videos=[x],video_metadata=[meta],do_resize=False,do_sample_frames=False,return_tensors='pt')
    assert int(z['video_grid_thw'][0,0])==len(frames)
    return z.to('cuda'),dict(size=[hh,ww],pixel_sha=hashlib.sha256(frames.tobytes()).hexdigest(),grid=z['video_grid_thw'].tolist())

@torch.inference_mode()
def infer(model,pr,inputs,*,fixed=None,adapter=None):
    import model.ptd_generation as pg
    from vg_tta.ptd_spatial_adapter_ab_v1 import boxes_from_tokens
    coord=torch.tensor([pg.get_token_id(pr.tokenizer,f'<{j}>') for j in range(1001)],device='cuda')
    times=[pg.get_token_id(pr.tokenizer,f'<t{j}>') for j in range(1,int(inputs['video_grid_thw'][0,0])+1)]
    time_map={t:i for i,t in enumerate(times)}
    original_probe=pg._run_cached_ptd_probe;original_sem=pg._parse_semantic_block;original_temp=pg._parse_temporal_block
    context={};captured={};semantic=[];temporal={};sem_index=0
    def probe(*args,**kw):
        qq=torch.as_tensor(kw['query_token_ids']).flatten().tolist()
        context['positions']=[time_map[v] for v in qq] if all(v in time_map for v in qq) else None
        return original_probe(*args,**kw)
    def sem(*args,**kw):
        nonlocal sem_index
        result=original_sem(*args,**kw) if fixed is None else fixed['semantic'][sem_index]
        semantic.append(result);sem_index+=1;return result
    def temp(*args,**kw):
        result=original_temp(*args,**kw) if fixed is None else (fixed['temporal']['tokens'],fixed['temporal']['anchors'])
        temporal.update(tokens=result[0],anchors=result[1]);return result
    def hook(module,args,output):
        pos=context.get('positions')
        if pos is None:return output
        n=len(pos);h=args[0][0].reshape(n,6,-1)[:,1:5].float()
        full=output[0].reshape(n,6,-1);logits=full[:,1:5][:,:,coord].float()
        raw=full.argmax(-1);restricted=coord[logits.argmax(-1)]
        coordinate_valid=(raw[:,1:5]==restricted).all(-1)
        captured.update(h=h.cpu(),logits=logits.cpu(),positions=pos,raw_blocks=raw.cpu(),coordinate_valid=coordinate_valid.cpu(),
                        base_tokens=logits.argmax(-1).cpu(),coord_ids=coord.cpu())
        if adapter is not None:
            delta=adapter(h);adapted=logits+delta
            changed=output.float().clone().reshape(1,n,6,-1)
            # Only modify originally valid coordinates; malformed output cannot be repaired.
            for j in range(n):
                if coordinate_valid[j]:
                    changed[0,j,1:5,:]=-torch.inf
                    changed[0,j,1:5,coord]=adapted[j]
            captured['adapted_logits']=adapted.cpu()
            return changed.reshape_as(output)
        return output
    handle=model.lm_head.register_forward_hook(hook);start=time.perf_counter()
    try:
        with patch.object(pg,'_run_cached_ptd_probe',probe),patch.object(pg,'_parse_semantic_block',sem),patch.object(pg,'_parse_temporal_block',temp):
            tokens,result=pg.generate_ptd(model,pr.tokenizer,dict(inputs),max_new_tokens=1024,max_time_tokens=len(times),
                                        temperature=0.,ptd_attn_implementation='sdpa')
        text=pr.tokenizer.decode(tokens[0],skip_special_tokens=False)
        # Official finish() already strips prompt_len; these are completion-only IDs.
        completion=text
        parsed=re.search(r'<\|time_start\|>\s*<t(\d+)>\s*<t(\d+)>\s*<\|time_end\|>',completion)
        interval=[int(parsed[1])-1,int(parsed[2])-1] if parsed else None
        format_ok=bool(result.stopped and captured and captured['coordinate_valid'].all() and interval is not None)
        if captured:
            tt=captured.get('adapted_logits',captured['logits']).argmax(-1)
            boxes,valid=boxes_from_tokens(tt);captured.update(boxes=boxes,geometry_valid=valid)
        return dict(**captured,semantic=semantic,temporal=temporal,interval=interval,format_ok=format_ok,
                    completion=completion,seconds=time.perf_counter()-start,GT_used=False)
    finally:handle.remove()

def capture(stage,condition='clean',views=False):
    from scripts.run_final_simplification_v1 import lease
    from vg_tta.ptd_spatial_adapter_ab_v1 import Adapter
    p=verify();rows=read(OUT/'INPUTS.json');rows=[r for r in rows if r['smoke']] if stage=='smoke' else [r for r in rows if r['split']==stage]
    start=time.perf_counter();guard=lease();counts={};torch.set_num_threads(4);torch.manual_seed(p['seed']);torch.cuda.reset_peak_memory_stats()
    try:
        assert sha(CK/'model.safetensors')==p['checkpoint']['sha256']
        pr=processor_load();model=model_load();loaded=time.perf_counter()-start
        for row in rows:
            for view in (['original','gamma0.9','gamma1.1'] if views else ['original']):
                f=path(row,condition,view)
                if f.exists():assert sha(f)==read(f.with_suffix('.json'))['sha'];continue
                frames,ids=frames_for(row,condition,view);inputs,pre=inputs_for(row,pr,frames)
                fixed=load(path(row,condition)) if view!='original' else None
                if fixed is not None and not fixed['format_ok']:
                    z=dict(format_ok=False,skipped='original_format_failure',GT_used=False)
                else:z=infer(model,pr,inputs,fixed=fixed)
                z.update(key=row['key'],condition=condition,view=view,frame_ids=ids,preprocess=pre)
                save(f,z);write(f.with_suffix('.json'),dict(sha=sha(f),lock_sha=sha(OUT/'LOCK.json')))
                if stage=='smoke' and view=='original' and z['format_ok']:
                    adapter=Adapter(z['h'].shape[-1],seed=p['seed']).cuda()
                    repeat=infer(model,pr,inputs);zero=infer(model,pr,inputs,adapter=adapter)
                    assert repeat['completion']==zero['completion']==z['completion']
                    assert torch.equal(repeat['logits'],z['logits']) and torch.equal(zero['adapted_logits'],z['logits'])
                    with torch.no_grad():adapter.up.weight.normal_(0,.001)
                    expected=z['logits']+adapter(z['h'].cuda()).cpu()
                    replay=infer(model,pr,inputs,adapter=adapter)
                    assert torch.equal(replay['h'],z['h']) and torch.equal(replay['adapted_logits'],expected)
                    assert replay['interval']==z['interval'] and replay['semantic']==z['semantic']
                    save(OUT/'smoke'/(row['key'].replace(':','_')+'.pt'),dict(zero_exact=True,repeat_exact=True,
                            nonzero_adapter_full_replay_exact=True,adapter=adapter.cpu().state_dict(),expected=expected,
                            actual=replay['adapted_logits'],parameters=sum(a.numel() for a in adapter.parameters())))
                    del adapter,repeat,zero,replay
                counts[row['key']+'|'+view]=dict(format_ok=z['format_ok'],seconds=z.get('seconds'),positions=len(z.get('positions',[])))
                print('PTD_CAPTURE',stage,condition,row['key'],view,counts[row['key']+'|'+view],flush=True)
                del inputs,frames,z;gc.collect();torch.cuda.empty_cache()
                assert torch.cuda.max_memory_allocated()<=p['peak_bytes'],'PEAK_BUDGET'
                assert time.perf_counter()-start<=p['stage_seconds'],'STAGE_TIME_BUDGET'
        write(OUT/(stage+'_'+condition+'_CAPTURE.json'),dict(status='completed',counts=counts,worker_seconds=time.perf_counter()-start,
              load_seconds=loaded,peak_bytes=torch.cuda.max_memory_allocated(),GT_worker=False))
    finally:
        write(OUT/'worker_receipts'/f'capture_{time.time_ns()}.json',dict(stage=stage,condition=condition,seconds=time.perf_counter()-start,
              peak_bytes=torch.cuda.max_memory_allocated(),counts=counts))
        guard.close()

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','capture']);a.add_argument('--stage',default='smoke');a.add_argument('--condition',default='clean');a.add_argument('--views',action='store_true');x=a.parse_args()
    prepare() if x.action=='prepare' else capture(x.stage,x.condition,x.views)
