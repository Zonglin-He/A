"""True text-only PTD interface audit; no labels or learned state persistence."""
import argparse,gc,hashlib,json,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.ptd_visual_evidence_decomposition_v1 import OUT,DENSE,Budget,digest,read,write,load,save,sha,frames_for,inputs_for,processor_load,model_load,parent_path,infer

def strip_visual(inputs,tok):
    ids=inputs['input_ids'][0]
    forbidden=[tok.convert_tokens_to_ids(t) for t in ['<|vision_start|>','<|vision_end|>','<|video_pad|>','<|image_pad|>']]
    # Patched PTD processor nests a video wrapper around per-frame wrappers.
    # Remove exactly the visual tokens, preserving every original time/text token.
    keep=[i for i,v in enumerate(ids.tolist()) if v not in forbidden];removed=len(ids)-len(keep)
    assert removed>0
    clean=ids[keep][None]
    assert not any(t in clean for t in forbidden)
    return dict(input_ids=clean,attention_mask=torch.ones_like(clean)),dict(removed=removed,kept_indices=keep,original_tokens=len(ids),clean_tokens=len(keep),visual_tokens_remaining=0)

@torch.inference_mode()
def probe(model,pr,inputs,fixed,text_only=False):
    import model.ptd_generation as pg
    coord=fixed['coord_ids'].cuda();time_map={pg.get_token_id(pr.tokenizer,f'<t{i+1}>'):i for i in range(32)}
    original_probe=pg._run_cached_ptd_probe;original_language=pg._run_language_model;context={};result={};si=0;records=[];prefills=[]
    def track(*args,**kw):
        q=torch.as_tensor(kw['query_token_ids']).flatten().tolist();context['pos']=[time_map[v] for v in q] if all(v in time_map for v in q) else None
        if context['pos'] is not None:
            records.append(dict(query_tokens=q,positions=context['pos'],prefix_tokens=args[1][0].detach().cpu(),context_limits=kw['context_limits'].cpu(),probe_position_starts=kw['probe_position_starts'].cpu(),cache_before=args[3].get_seq_length()))
        return original_probe(*args,**kw)
    def sem(*args,**kw):
        nonlocal si
        x=fixed['semantic'][si];si+=1;return x
    def temp(*args,**kw):return fixed['temporal']['tokens'],fixed['temporal']['anchors']
    def hook(module,args,out):
        pos=context.get('pos')
        if pos is None:return out
        n=len(pos);h=args[0][0].reshape(n,6,-1)[:,1:5].float();f=out[0].reshape(n,6,-1);logits=f[:,1:5][:,:,coord].float();raw=f.argmax(-1)
        result.update(h=h.cpu(),logits=logits.cpu(),positions=pos,raw_blocks=raw.cpu(),coordinate_valid=(raw[:,1:5]==coord[logits.argmax(-1)]).all(-1).cpu())
        return out
    def text_prefill(**kwargs):
        assert kwargs.get('past_key_values') is None
        assert not any(k in kwargs for k in ['pixel_values','pixel_values_videos','image_grid_thw','video_grid_thw'])
        ids=kwargs['input_ids'];n=ids.shape[1];core=model.model
        core.rope_deltas=torch.zeros((1,1),dtype=torch.long,device=ids.device)
        pos=torch.arange(n,device=ids.device).view(1,1,-1).expand(3,1,-1)
        out=core.language_model(input_ids=ids,attention_mask=kwargs['attention_mask'],position_ids=pos,past_key_values=None,use_cache=True,visual_pos_masks=None,deepstack_visual_embeds=None,return_dict=True)
        prefills.append(dict(fresh_KV=True,cache_length=out.past_key_values.get_seq_length(),input_length=n,rope_delta=core.rope_deltas.cpu(),position_ids=pos.cpu(),visual_forward_calls=0))
        return SimpleNamespace(past_key_values=out.past_key_values,logits=model.lm_head(out.last_hidden_state[:,-1:]))
    handle=model.lm_head.register_forward_hook(hook)
    def deny_visual(*a,**kw):raise AssertionError('VISUAL_FORWARD_IN_TEXT_ONLY_BRANCH')
    try:
        from contextlib import ExitStack
        with ExitStack() as stack:
            stack.enter_context(patch.object(pg,'_run_cached_ptd_probe',track));stack.enter_context(patch.object(pg,'_parse_semantic_block',sem));stack.enter_context(patch.object(pg,'_parse_temporal_block',temp))
            if text_only:
                stack.enter_context(patch.object(model,'forward',text_prefill));stack.enter_context(patch.object(model.model.visual,'forward',deny_visual))
            tokens,meta=pg.generate_ptd(model,pr.tokenizer,dict(inputs),max_new_tokens=1024,max_time_tokens=32,temperature=0.,ptd_attn_implementation='sdpa')
        assert result['positions']==fixed['positions'];result.update(completion=pr.tokenizer.decode(tokens[0]),stopped=meta.stopped,spatial_probe_records=records,text_only_prefills=prefills,GT_used=False)
        if text_only:assert len(prefills)==1 and prefills[0]['cache_length']==inputs['input_ids'].shape[1]
        return result
    finally:handle.remove()

def run(stage):
    lock=read(OUT/'STAGE0_LOCK.json');rows={r['key']:r for r in read(DENSE/'INPUTS.json')};keys=lock['pilot_keys'] if stage=='stage0' else list(rows)
    expected=read(OUT/'STUDENT_CODE_LOCK.json')['code_sha']
    for f in sorted((OUT/'amendments').glob('student_*.json')):expected=read(f)['student_code_sha']
    assert sha(__file__)==expected;budget=Budget(stage);done=[]
    try:
        pr=processor_load();model=model_load()
        for key in keys:
            dest=OUT/'student'/f'{digest(key)}.pt'
            if dest.exists():assert sha(dest)==read(dest.with_suffix('.json'))['sha'];continue
            row=rows[key];fixed=load(parent_path(row,'clean'));frames,_=frames_for(row,'clean');inputs,pre=inputs_for(row,pr,frames)
            plus=probe(model,pr,inputs,fixed);budget.calls+=1;assert torch.equal(plus['logits'],fixed['logits']) and torch.equal(plus['h'],fixed['h']),'BASELINE_REINSERTION_MISMATCH'
            clean,ablation=strip_visual(inputs,pr.tokenizer);minus=probe(model,pr,clean,fixed,text_only=True);budget.calls+=1
            donor=rows[lock['donors'][key]];df,_=frames_for(donor,'clean');h,w=frames.shape[1:3]
            df=np.stack([np.asarray(Image.fromarray(im).resize((w,h),Image.Resampling.LANCZOS)) for im in df]);wrong_inputs,_=inputs_for(row,pr,df);wrong=probe(model,pr,wrong_inputs,fixed);budget.calls+=1
            repeat=probe(model,pr,clean,fixed,text_only=True);budget.calls+=1
            assert torch.equal(repeat['logits'],minus['logits']) and torch.equal(repeat['h'],minus['h']),'TEXT_ONLY_STALE_KV_OR_ROPE'
            # Queries preserve time identities; generated semantic/time prefix is unchanged after removing visual tokens.
            p=plus['spatial_probe_records'][0];m=minus['spatial_probe_records'][0]
            assert p['query_tokens']==m['query_tokens'];assert p['prefix_tokens'][len(inputs['input_ids'][0]):].equal(m['prefix_tokens'][len(clean['input_ids'][0]):])
            assert (p['context_limits']-m['context_limits']==ablation['removed']).all();assert (p['probe_position_starts']-m['probe_position_starts']==ablation['removed']).all()
            result=dict(key=key,plus=plus,minus=minus,wrong=wrong,ablation=ablation,true_full_replay_exact=True,minus_after_wrong_exact=True,donor=donor['key'],GT_used=False)
            save(dest,result);write(dest.with_suffix('.json'),dict(sha=sha(dest)));done.append(str(dest));print('STUDENT',key,'true_exact',True,'no_visual',True,flush=True)
            del inputs,clean,wrong_inputs,frames,df,plus,minus,wrong,repeat,result;gc.collect();torch.cuda.empty_cache();budget.check()
        write(OUT/(stage+'_STUDENT_COMPLETE.json'),dict(status='completed',files=done,GT_used=False))
    except BaseException as e:
        write(OUT/'failures'/f'{stage}_student_{time.time_ns()}.json',dict(error=repr(e),traceback=traceback.format_exc(),completed=done));raise
    finally:budget.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',default='stage0');a=p.parse_args();run(a.stage)
