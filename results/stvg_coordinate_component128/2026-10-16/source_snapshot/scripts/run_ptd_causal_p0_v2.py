"""Modern PTD diagnostic; NEVER a HC->Vid or Vid->HC transfer score.

The released model was trained on BOTH datasets. Frozen parameter diagnostics
use one exact pixel grid, native time tokens or GT/matched temporal tokens,
and identical dense spatial query times. No repository source is modified.
"""
import argparse,sys,importlib.util,subprocess,time,re,gc,math,json
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.run_st_causal_audit_v2 import OUT,plan
from scripts.decota_matrix_common_v1 import read,write,save,load,sha,status
from vg_tta.exact_frame_decode_audit_v2 import decode
from vg_tta.st_component_diagnostics_v1 import contiguous_controls
PTD=ROOT/'external/ParallelTubeDecoding';CK=ROOT/'checkpoints/ParallelTubeDecoding-Qwen3-VL-4B'
DEST=OUT/'ptd_2026';SEGMENT=re.compile(r'<\|time_start\|>\s*<t(\d+)>\s*<t(\d+)>\s*<\|time_end\|>')
BOX=re.compile(r'<t(\d+)>\s*<\|box_start\|>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*<\|box_end\|>')


def prepare():
    p=plan();rows={}
    for g,rr in p['rows'].items():
        rows[g]=[]
        for row in rr:
            n=len(row['input']['frame_ids']);pos=np.unique(np.linspace(0,n-1,min(n,64)).round().astype(int));q=dict(row['input']);q['frame_ids']=[q['frame_ids'][i] for i in pos]
            gt=row['diagnostic_GT'];v=np.array(gt['valid'])[pos];assert v.any()
            rows[g].append(dict(parent_ordinal=row['ordinal'],input=q,positions_in_parent=pos.tolist(),diagnostic_GT=dict(interval=gt['interval'],boxes=np.array(gt['boxes'])[pos].tolist(),valid=v.tolist(),full_track_centers=None if gt['full_track_centers'] is None else np.array(gt['full_track_centers'])[pos].tolist())))
    write(DEST/'lock.json',dict(rows=rows,parent_sha256=sha(OUT/'lock.json'),checkpoint_revision='b7758863446db22d3a5b587ab2e5f204be69b77a',
        official_repository='https://github.com/mbzuai-oryx/ParallelTubeDecoding',commit=subprocess.check_output(['git','-C',str(PTD),'rev-parse','HEAD'],text=True).strip(),
        year=2026,source_training='joint VidSTG + HC-STVG v1/v2; NOT a source-isolated transfer checkpoint',
        primary_comparison='GT time tokens vs native time tokens, SAME dense spatial time queries; GT vs matched controls secondary',
        preprocessing='original frame numbers, max64 fixed uniformly selected parent positions, resize224x320, allSDPA; diagnostic not paper-resolution benchmark',
        spatial_oracle='full-track GT centers only; bias to cached visual prefix keys, fixed token count on every frame',
        seeds=p['seeds'],no_parameter_updates=True,formal_TTA=False,development_only=True,created_unix=time.time()))


def utilities():
    sys.path.insert(0,str(PTD/'src'))
    # Import just the official preprocessing helper. The dataset package's
    # __init__ imports unrelated DPO/GRPO training data dependencies.
    spec=importlib.util.spec_from_file_location('ptd_diagnostic_data_utils',PTD/'src/dataset/data_utils.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
    return d


def inputs_for(row,processor):
    from transformers.video_utils import VideoMetadata
    raw,ids=decode(row['input']);x=torch.from_numpy(raw).permute(0,3,1,2).float()
    x=torch.nn.functional.interpolate(x,size=(224,320),mode='bilinear',align_corners=False,antialias=True)
    query=row['input']['caption'].strip()
    if not query.endswith(('.', '?', '!')):query+='.'
    text=f"Given the query: '{query}' Localize the described object throughout the video. Use object reference tokens, time tokens, and box tokens. Return the object reference, event time segment, and per-time bbox coordinates."
    msg=[dict(role='user',content=[dict(type='video'),dict(type='text',text=text)])]
    prompt=processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
    q=row['input'];fps=q.get('fps')
    if fps is None:
        from fractions import Fraction
        probe=json.loads(subprocess.check_output([str(ROOT/'.conda/tubedetr/bin/ffprobe'),'-v','error','-select_streams','v:0','-show_entries','stream=avg_frame_rate','-of','json',q['video_path']],text=True))
        fps=float(Fraction(probe['streams'][0]['avg_frame_rate']))
    assert fps>0
    meta=VideoMetadata(total_num_frames=q['frame_count'],fps=fps,frames_indices=ids,width=q['width'],height=q['height'],video_backend='exact_ffmpeg_frame_ids')
    # Time tokens encode sampled position rather than invented physical FPS.
    inputs=processor(text=[prompt],videos=[x],video_metadata=[meta],do_resize=False,do_sample_frames=False,return_tensors='pt')
    assert int(inputs['video_grid_thw'][0,0])==len(ids)
    return inputs.to('cuda')


def parse(text,n):
    matches=list(SEGMENT.finditer(text));ij=None
    if len(matches)==1:
        a,b=map(int,matches[0].groups())
        if 1<=a<=b<=n:ij=[a-1,b-1]
    boxes=torch.zeros(n,4);present=torch.zeros(n,dtype=torch.bool)
    for m in BOX.finditer(text):
        j,*xy=map(int,m.groups())
        if not 1<=j<=n or present[j-1]:raise ValueError('duplicate/out-of-grid output box')
        x=np.clip(np.array(xy,dtype=float)/1000.,0,1);lo=np.minimum(x[:2],x[2:]);hi=np.maximum(x[:2],x[2:]);boxes[j-1]=torch.tensor(np.r_[(lo+hi)/2,hi-lo]);present[j-1]=True
    return dict(text=text,indices=ij,boxes=boxes,present=present)


def infer(model,processor,inputs,*,interval=None,dense=False,prefix_bias=None,temporal_only=False,query_positions=None):
    import model.ptd_generation as pg
    n=int(inputs['video_grid_thw'][0,0]);parse_original=pg._parse_temporal_block;run_original=pg._run_language_model;calls=[]
    def temporal(block,tokens,*,block_size):
        if interval is None:
            visible,anchors=parse_original(block,tokens,block_size=block_size)
        else:
            a,b=interval;visible=[tokens['time_start'],tokens['ordered_time_tokens'][a],tokens['ordered_time_tokens'][b],tokens['time_end']];anchors=tokens['ordered_time_tokens'][a:b+1]
        if dense:anchors=tokens['ordered_time_tokens']
        if query_positions is not None:anchors=[tokens['ordered_time_tokens'][i] for i in query_positions]
        return visible,anchors
    ref_end_id=pg.get_token_id(processor.tokenizer,'<|object_ref_end|>')
    def language(*args,**kw):
        mask=kw.get('attention_mask')
        query_ids=kw.get('input_ids',args[1] if len(args)>1 else None)
        is_temporal_probe=query_ids is not None and query_ids.shape[-1]>=6 and int(query_ids[0,-6])==ref_end_id
        if prefix_bias is not None and mask is not None and is_temporal_probe:
            assert mask.ndim==4 and mask.shape[-1]>=len(prefix_bias)
            changed=mask.clone()
            if changed.dtype==torch.bool:changed=torch.zeros_like(changed,dtype=torch.float32).masked_fill(~changed,-torch.inf)
            bias=torch.zeros(changed.shape[-1],device=changed.device,dtype=changed.dtype);bias[:len(prefix_bias)]=prefix_bias.to(bias)
            # Only the six temporal probe rows; do not perturb catch-up rows
            # that commit already generated semantic-reference tokens to KV.
            changed[:,:,-6:,:]+=bias[None,None,None,:]
            kw['attention_mask']=changed;calls.append(dict(query_length=mask.shape[-2],changed_query_rows=6,key_length=mask.shape[-1],phase='temporal_probe_only'))
        return run_original(*args,**kw)
    start=time.perf_counter()
    with patch.object(pg,'_parse_temporal_block',temporal),patch.object(pg,'_run_language_model',language):
        out,result=pg.generate_ptd(model,processor.tokenizer,dict(inputs),max_new_tokens=1024,max_time_tokens=n,temperature=0.,ptd_attn_implementation='sdpa',generation_format='temporal_localization' if temporal_only else 'spatio_temporal_grounding')
    text=processor.tokenizer.decode(out[0],skip_special_tokens=False)
    pred=parse(text,n);pred.update(seconds=time.perf_counter()-start,stopped=result.stopped,bias_calls=calls)
    if prefix_bias is not None:assert calls,'No actual attention intervention occurred'
    return pred


def run(group,limit=None,preprocess_only=False):
    p=read(DEST/'lock.json');plan();assert p['parent_sha256']==sha(OUT/'lock.json');d=utilities()
    from transformers import AutoProcessor,AutoModelForImageTextToText
    processor=AutoProcessor.from_pretrained(CK,local_files_only=True);d.patch_qwen3_video_processor(processor);d.patch_processor_with_time_tokens(processor)
    if preprocess_only:
        x=inputs_for(p['rows'][group][0],processor);print('PREPROCESS_OK',{k:tuple(v.shape) for k,v in x.items() if torch.is_tensor(v)},flush=True);return
    receipt=read(CK/'OFFICIAL_RECEIPT.json');assert sha(CK/'model.safetensors')==receipt['sha256']
    from train.monkey_patch_forward import replace_qwen3_with_ptd_forward
    replace_qwen3_with_ptd_forward();torch.manual_seed(20260910);torch.set_num_threads(4)
    model,info=AutoModelForImageTextToText.from_pretrained(CK,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',device_map='cuda',output_loading_info=True)
    assert not info['missing_keys'] and not info['unexpected_keys'];model.eval().requires_grad_(False)
    receipts=[]
    ordered=sorted(p['rows'][group],key=lambda r:(r['diagnostic_GT']['full_track_centers'] is None,r['parent_ordinal']))
    for row in ordered[:limit]:
        j=row['parent_ordinal'];dest=DEST/'runs'/group/f'{j:03d}.pt'
        if not dest.exists():
            x=inputs_for(row,processor);n=int(x['video_grid_thw'][0,0]);gt=np.array(row['diagnostic_GT']['valid']);ix=np.flatnonzero(gt)
            preds={'native_sparse':infer(model,processor,x),'native_dense':infer(model,processor,x,dense=True)}
            repeat=infer(model,processor,x,dense=True);assert repeat['text']==preds['native_dense']['text']
            preds['GT_dense']=infer(model,processor,x,interval=[int(ix[0]),int(ix[-1])],dense=True)
            controls=contiguous_controls(gt,row['input']['frame_ids'],p['seeds'])
            for name,c in controls.items():
                ii=np.flatnonzero(c['mask']);preds[name]=infer(model,processor,x,interval=[int(ii[0]),int(ii[-1])],dense=True)
            # Optional Space->Time: full-track supervision ONLY, no temporal
            # support mask and no box annotation existence cue in inputs.
            centers=row['diagnostic_GT']['full_track_centers'];space={};space_audit={}
            if centers is not None and bool(preds['native_dense']['present'].all()):
                from vg_tta.st_causal_audit_v2 import center_roi
                from vg_tta.matched_direction_diagnostics_v1 import spatial_directions
                _,hh,ww=map(int,x['video_grid_thw'][0]);grid=(hh//2,ww//2)
                vp=torch.where(x['input_ids'][0]==model.config.video_token_id)[0];assert len(vp)==n*grid[0]*grid[1]
                allc=np.array(centers);variants,space_audit=spatial_directions(preds['native_dense']['boxes'][:,:2].numpy(),allc)
                zero=torch.zeros(x['input_ids'].shape[1],device='cuda')
                space['zero_bias']=infer(model,processor,x,prefix_bias=zero,temporal_only=True)
                assert space['zero_bias']['indices']==preds['native_dense']['indices']
                for name,c in variants.items():
                    roi=center_roi(c,grid).flatten().cuda();bias=torch.zeros(x['input_ids'].shape[1],device='cuda');bias[vp]=roi.float()*math.log(4.)
                    space[name]=infer(model,processor,x,prefix_bias=bias,temporal_only=True)
                roi=center_roi(allc,grid);uniform=math.log1p(float(roi.float().mean())*3.)
                bias=torch.zeros(x['input_ids'].shape[1],device='cuda');bias[vp]=uniform
                space['uniform_mass']=infer(model,processor,x,prefix_bias=bias,temporal_only=True)
                # The semantic object reference is generated BEFORE the
                # intervention. Verify it is identical, not merely assumed.
                semantic=lambda text:text.split('<|object_ref_end|>')[0]
                assert all('<|object_ref_end|>' in z['text'] for z in space.values())
                assert all(semantic(z['text'])==semantic(preds['native_dense']['text']) for z in space.values())
                space_audit['zero_bias_native_interval_exact']=True
                space_audit['semantic_prefix_exact_all_arms']=True
            assert all(not m.training for m in model.modules()) and all(not v.requires_grad for v in model.parameters())
            save(dest,dict(predictions=preds,spatial=space,spatial_direction_audit=space_audit,spatial_eligible_full_track=centers is not None,spatial_skip_reason='missing dense native boxes' if centers is not None and not space else None,query=row,controls={k:{kk:vv for kk,vv in c.items() if kk!='mask'} for k,c in controls.items()},
                strict_checkpoint_load=True,loading_info=info,pixel_grid_preserved=True,dense_query_grid_matched=True,repeat_exact=True,
                GT_diagnostic_only=True,source_sha256=sha(__file__),protocol_sha256=sha(DEST/'lock.json'),checkpoint_sha256=receipt['sha256'],prefix_tokens=x['input_ids'].shape[1]))
            del x,preds,space;gc.collect();torch.cuda.empty_cache();print('PTD_DONE',group,j+1,flush=True)
        receipts.append(dict(path=str(dest),sha256=sha(dest),ordinal=j))
        status(DEST/f'progress_{group}.json',dict(done=len(receipts),total=len(p['rows'][group]),unix=time.time()))
    if limit is None:write(DEST/f'barrier_{group}.json',dict(receipts=receipts,frozen_parameters=True,GT_diagnostic_only=True))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run']);ap.add_argument('--group',default='hc_to_vid');ap.add_argument('--limit',type=int);ap.add_argument('--preprocess-only',action='store_true');a=ap.parse_args()
    if a.stage=='prepare':prepare()
    else:run(a.group,a.limit,a.preprocess_only)
