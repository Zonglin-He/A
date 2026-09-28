"""Differentiable replay of the exact native PTD cached probe schedule.

Teacher tokens specify support, never labels. Each branch starts from a new
multimodal prefill. No inference-decorated generator is used for the student.
"""
from __future__ import annotations
from unittest.mock import patch
import torch
from vg_tta.desta3d_v2_ptd import branch_injection


def _selected_logits(logits, query, token_ids, coordinate_ids):
    if query == [int(token_ids['ref_end'])]:
        assert logits.shape[:2] == (1, 6)
        ids = torch.tensor(token_ids['ordered_time_tokens'], device=logits.device)
        return 'time', logits[0, 1:3].index_select(-1, ids).float()
    if query and all(q in token_ids['ordered_time_tokens'] for q in query):
        assert logits.shape[:2] == (1, len(query)*6)
        ids = torch.tensor(coordinate_ids, device=logits.device)
        return 'coordinate', logits.reshape(len(query), 6, -1)[:, 1:5].index_select(-1, ids).float()
    return None, None


def capture_teacher(model, processor, inputs, adapter, fields):
    """Observe native cached-v3 generation and retain every probe's inputs."""
    import model.ptd_generation as pg
    from vg_tta.desta3d_v2_shared_reference_cached import decode_shared_reference_two_pass
    n = int(inputs['video_grid_thw'][0, 0])
    token_ids = pg.build_ptd_token_ids(processor.tokenizer, max_time_tokens=n)
    coordinates = [int(pg.get_token_id(processor.tokenizer, f'<{i}>')) for i in range(1001)]
    branches = []
    original_probe = pg._run_cached_ptd_probe
    prompt_len = inputs['input_ids'].shape[1]

    def observe(m, generated, registry, cache, **kwargs):
        if generated.shape[1] == prompt_len:
            branches.append({'probes': [], 'logits': {}})
        assert 1 <= len(branches) <= 2
        query = torch.as_tensor(kwargs['query_token_ids']).flatten().tolist()
        row = {'generated': generated.detach().cpu().clone(),
               'cache_length_before': pg._cache_length(cache),
               'kwargs': {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
                          for k, v in kwargs.items()}}
        original_lm = pg._run_language_model
        def lm(*args, **kw):
            output, logits = original_lm(*args, **kw)
            kind, selected = _selected_logits(logits, query, token_ids, coordinates)
            if kind:
                assert kind not in branches[-1]['logits']
                branches[-1]['logits'][kind] = selected.detach().cpu().clone()
            return output, logits
        with patch.object(pg, '_run_language_model', lm):
            result = original_probe(m, generated, registry, cache, **kwargs)
        row['cache_length_after'] = pg._cache_length(result[1])
        branches[-1]['probes'].append(row)
        return result

    with patch.object(pg, '_run_cached_ptd_probe', observe):
        native = decode_shared_reference_two_pass(model, processor, inputs, adapter, fields)
    # Outside inference_mode, clone again so replay inputs are ordinary tensors.
    from vg_tta.optimizer_checkpoint import cpu_clone
    trace = cpu_clone({'branches': branches, 'token_ids': token_ids,
                      'coordinate_ids': coordinates, 'prompt_length': prompt_len})
    return native, trace


def replay_branch(model, inputs, adapter, fields, trace, branch, *, pg=None):
    """Return graph-connected logits at frozen teacher support, plus cache audit."""
    if torch.is_inference_mode_enabled():
        raise RuntimeError('student replay cannot run under inference_mode')
    if pg is None:
        import model.ptd_generation as pg
    index = {'event': 0, 'spatial': 1}[branch]
    if index >= len(trace['branches']):
        raise ValueError('teacher has no branch support')
    schedule = trace['branches'][index]['probes']
    if not schedule:
        raise ValueError('empty teacher schedule')
    device = inputs['input_ids'].device
    prefix = schedule[0]['generated'].to(device)
    if not torch.equal(prefix, inputs['input_ids']):
        raise ValueError('teacher and student prompt tokens differ')
    pg.configure_ptd_model(model, 6)
    prefill = dict(inputs)
    for key in ('position_ids','cache_position','past_key_values','labels',
                'ptd_position_ids','ptd_prefix_lengths','ptd_context_limits'):
        prefill.pop(key, None)
    prefill.update(use_cache=True, return_dict=True, logits_to_keep=1)
    selected, audits = {}, []
    with branch_injection(model, adapter, inputs, fields, branch) as injection:
        output = model(**prefill)
    cache = output.past_key_values
    assert cache is not None
    # The graph stays in the cache; temporary probes are cropped exactly as native.
    for row in schedule:
        assert pg._cache_length(cache) == row['cache_length_before']
        generated = row['generated'].to(device)
        assert torch.equal(generated[:, :prefix.shape[1]], prefix)
        kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in row['kwargs'].items()}
        assert kwargs['ptd_attn_implementation'] == 'sdpa'
        query = torch.as_tensor(kwargs['query_token_ids']).flatten().tolist()
        original_lm = pg._run_language_model
        def lm(*args, **kw):
            output, logits = original_lm(*args, **kw)
            kind, value = _selected_logits(logits, query, trace['token_ids'], trace['coordinate_ids'])
            if kind:
                assert kind not in selected
                selected[kind] = value
            return output, logits
        with patch.object(pg, '_run_language_model', lm):
            _, cache = pg._run_cached_ptd_probe(model, generated, trace['token_ids'], cache, **kwargs)
        assert pg._cache_length(cache) == row['cache_length_after']
        audits.append({'before': row['cache_length_before'], 'after': pg._cache_length(cache),
                       'query': query})
    expected = 'time' if branch == 'event' else 'coordinate'
    if expected not in selected:
        raise ValueError('teacher has no requested distribution support')
    return selected[expected], {'branch': branch, 'cache': audits,
        'injection': {k: injection[k] for k in ('calls','changed_elements','relative_injection_norm')}}


def output_kl(student, teacher):
    """Mean over endpoints or frame/coordinates; sum over categorical support."""
    if student.shape != teacher.shape or student.ndim not in (2,3):
        raise ValueError('output distribution support mismatch')
    teacher = teacher.detach().to(device=student.device, dtype=torch.float32)
    lp, lq = teacher.log_softmax(-1), student.float().log_softmax(-1)
    return (lp.exp()*(lp-lq)).sum(-1).mean()
