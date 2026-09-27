"""Frozen PTD interface for independent event/spatial DESTA-3D v2 paths.

No dataset labels are accessed here. Caption features come from one stock
prompt prefill; temporal and spatial generation each use their own new KV cache.
"""
from __future__ import annotations

from contextlib import contextmanager
import re
from typing import Any, Mapping
from unittest.mock import patch

import torch

from vg_tta.desta3d_v1 import reshape_merged_video_tokens


def caption_token_mask(processor, inputs: Mapping[str, torch.Tensor], caption: str):
    """Map caption character offsets to the actual expanded multimodal tokens.

    The complete post-video text suffix must match byte-for-token. Tokens whose
    offsets overlap the caption are selected, including any merged punctuation
    at its boundaries; instruction/template-only tokens are excluded.
    """
    query = caption.strip()
    if not query:
        raise ValueError('empty caption')
    if not query.endswith(('.', '?', '!')):
        query += '.'
    prefix = "Given the query: '"
    text = (prefix + query + "' Localize the described object throughout the video. "
            'Use object reference tokens, time tokens, and box tokens. Return the object reference, '
            'event time segment, and per-time bbox coordinates.')
    messages = [dict(role='user', content=[dict(type='video'), dict(type='text', text=text)])]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert prompt.count(text) == 1
    start = prompt.index(text) + len(prefix)
    end = start + len(query)
    enc = processor.tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    template_ids = enc['input_ids']
    actual_ids = inputs['input_ids'][0].detach().cpu().tolist()
    suffix = 0
    while suffix < min(len(template_ids), len(actual_ids)) and template_ids[-1-suffix] == actual_ids[-1-suffix]:
        suffix += 1
    selected = [i for i, (a,b) in enumerate(enc['offset_mapping']) if a < end and b > start]
    if not selected or min(selected) < len(template_ids)-suffix:
        raise ValueError('caption tokens not entirely in the exact matching text suffix')
    shift = len(actual_ids)-len(template_ids)
    positions = [i+shift for i in selected]
    if any(actual_ids[j] != template_ids[i] for i,j in zip(selected,positions)):
        raise ValueError('caption token mapping failed')
    mask = torch.zeros_like(inputs['input_ids'], dtype=torch.bool)
    mask[0, positions] = True
    if not torch.all(inputs['attention_mask'][mask].bool()):
        raise ValueError('caption token overlaps input padding')
    return mask, {'caption':query, 'positions':positions, 'token_count':len(positions),
                  'matching_suffix_tokens':suffix, 'query_definition':'stock caption-only prompt hidden token sequence; no generated response or GT'}


def capture_stock_fields(model, processor, inputs, caption, frame_ids, fps):
    mask, metadata = caption_token_mask(processor, inputs, caption)
    video_id = int(model.config.video_token_id)
    video_positions = inputs['input_ids'][0].eq(video_id).nonzero().flatten()
    assert len(video_positions) and min(metadata['positions']) > int(video_positions[-1])
    captured = {}
    def merger(_m,_a,out):
        if 'tokens' in captured:
            raise RuntimeError('repeated visual merger in stock prefill')
        captured['tokens'] = out.detach()
    def language(_m,_a,out):
        hidden = out.last_hidden_state
        assert hidden.shape[:2] == mask.shape
        captured['query_tokens'] = hidden[mask].reshape(1,-1,hidden.shape[-1]).float().detach()
    handles = [model.model.visual.merger.register_forward_hook(merger),
               model.model.language_model.register_forward_hook(language)]
    keep = {'input_ids','attention_mask','mm_token_type_ids','pixel_values_videos','video_grid_thw','second_per_grid_ts'}
    try:
        with torch.no_grad():
            model.model(**{k:v for k,v in inputs.items() if k in keep}, use_cache=False)
    finally:
        for h in reversed(handles):h.remove()
    visual = reshape_merged_video_tokens(captured['tokens'].float(), inputs['video_grid_thw'][0])
    assert visual.shape[1] == len(frame_ids) and fps > 0
    q = captured['query_tokens']
    assert torch.isfinite(visual).all() and torch.isfinite(q).all()
    return {'visual_grid':visual, 'query_tokens':q,
            'query_mask':torch.ones(q.shape[:2],device=q.device,dtype=torch.bool),
            'frame_times':torch.tensor(frame_ids,device=visual.device,dtype=torch.float32)[None]/float(fps),
            'metadata':metadata}


@contextmanager
def branch_injection(model, adapter, inputs, fields, branch, *, gate_override=None):
    if branch not in {'event','spatial'}:
        raise ValueError(branch)
    state = {'branch':branch, 'calls':0}
    def hook(_m,_a,tokens):
        state['calls'] += 1
        if state['calls'] != 1:
            raise RuntimeError('each decode must have exactly one own multimodal prefill')
        visual = reshape_merged_video_tokens(tokens.float(), inputs['video_grid_thw'][0])
        assert torch.allclose(visual,fields['visual_grid'],rtol=1e-3,atol=1e-3)
        result = adapter(visual,fields['query_tokens'],query_mask=fields['query_mask'],
                         frame_times=fields['frame_times'],gate_override=gate_override)
        state['fields'] = result
        updated = result['updated_tokens_'+branch].reshape_as(tokens).to(tokens.dtype)
        state['zero_exact'] = bool(torch.equal(updated,tokens))
        state['changed_elements'] = int((updated != tokens).sum())
        state['relative_injection_norm'] = float((updated.float()-tokens.float()).norm()/tokens.float().norm().clamp_min(1e-12))
        state['updated_tokens'] = updated.detach().cpu()
        return updated
    handle = model.model.visual.merger.register_forward_hook(hook)
    try:
        yield state
    finally:
        handle.remove()
    if state['calls'] != 1:
        raise RuntimeError('branch did not execute an independent visual prefill')


@torch.inference_mode()
def temporal_decode(model, processor, inputs):
    import model.ptd_generation as pg
    n = int(inputs['video_grid_thw'][0,0])
    saved, probe_queries = {}, []
    parse = pg._parse_temporal_block
    probe = pg._run_cached_ptd_probe
    def temporal(*args,**kw):
        tokens, anchors = parse(*args,**kw)
        saved.update(tokens=list(tokens),anchors=list(anchors))
        return tokens,anchors
    def observed_probe(*args,**kw):
        probe_queries.append(torch.as_tensor(kw['query_token_ids']).flatten().tolist())
        return probe(*args,**kw)
    with patch.object(pg,'_parse_temporal_block',temporal), patch.object(pg,'_run_cached_ptd_probe',observed_probe):
        tokens,res = pg.generate_ptd(model,processor.tokenizer,dict(inputs),max_new_tokens=1024,
                                    max_time_tokens=n,temperature=0.,generation_format='temporal_localization',
                                    ptd_attn_implementation='sdpa')
    completion = processor.tokenizer.decode(tokens[0],skip_special_tokens=False)
    m = re.search(r'<\|time_start\|>\s*<t(\d+)>\s*<t(\d+)>\s*<\|time_end\|>',completion)
    interval = [int(m[1])-1,int(m[2])-1] if m else None
    tids = {pg.get_token_id(processor.tokenizer,f'<t{i+1}>') for i in range(n)}
    assert not any(q and all(x in tids for x in q) for q in probe_queries), 'event pass unexpectedly decoded boxes'
    valid = bool(res.stopped and saved and interval is not None and 0 <= interval[0] <= interval[1] < n)
    return {'interval':interval,'format_ok':valid,'temporal':saved,'completion':completion,
            'probe_queries':probe_queries,'spatial_probe_count':0,'GT_used':False}


@torch.inference_mode()
def spatial_decode_fixed_interval(model,processor,inputs,event_result):
    from scripts.ptd_spatial_adapter_ab_v1 import infer
    import model.ptd_generation as pg
    if not event_result['format_ok']:
        raise ValueError('spatial pass cannot invent a missing event interval')
    tokens = event_result['temporal']['tokens']
    anchors = event_result['temporal']['anchors']
    calls = []
    def force_temporal(_raw, token_ids, *, block_size):
        # Validate the carried event decision against the same official grammar.
        block = list(tokens)+[token_ids['null']]*(block_size-len(tokens))
        parsed = original_parse(block,token_ids,block_size=block_size)
        assert parsed == (tokens,anchors)
        calls.append(1)
        return list(tokens),list(anchors)
    original_parse = pg._parse_temporal_block
    with patch.object(pg,'_parse_temporal_block',force_temporal):
        # Only time is fixed. The spatial branch produces its own semantic ref.
        result = infer(model,processor,dict(inputs))
    if result.get('interval') is not None:
        assert result['interval'] == event_result['interval']
    if result.get('positions'):
        assert result['positions'] == list(range(event_result['interval'][0],event_result['interval'][1]+1))
    result['temporal_override_calls'] = len(calls)
    return result


@torch.inference_mode()
def decode_two_pass(model,processor,inputs,adapter,fields,*,gate_override=None):
    with branch_injection(model,adapter,inputs,fields,'event',gate_override=gate_override) as event_state:
        event = temporal_decode(model,processor,inputs)
    if not event['format_ok']:
        return {'event':event,'spatial':None,'interval':event['interval'],'format_ok':False,
                'event_injection':event_state,'spatial_injection':None,'GT_used':False}
    with branch_injection(model,adapter,inputs,fields,'spatial',gate_override=gate_override) as spatial_state:
        spatial = spatial_decode_fixed_interval(model,processor,inputs,event)
    return {'event':event,'spatial':spatial,'interval':event['interval'],
            'format_ok':bool(spatial['format_ok']), 'event_injection':event_state,
            'spatial_injection':spatial_state,'GT_used':False,
            'cache_policy':'independent fresh event/spatial KV; only event interval and time-token anchors passed'}
