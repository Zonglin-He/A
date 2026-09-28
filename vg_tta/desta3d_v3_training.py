"""Frozen-backbone source preparation utilities; no driver side effects."""
import math
import torch
from vg_tta.optimizer_checkpoint import cpu_clone,restore_optimizer,validate_serialized_optimizer

def training_inputs(processor,prompt,record):
    """Use the already prepared (possibly mildly augmented) physical pixels."""
    if not record['response_eligible']:return None
    from scripts.ptd_8b_teacher_feasibility_v1 import append_targets
    prompt_ids=prompt['input_ids'][0].detach().cpu();start=prompt_ids.numel()
    response=torch.tensor(processor.tokenizer.encode(record['response']+'<|im_end|>\n',add_special_tokens=False),dtype=torch.long)
    mm=prompt.get('mm_token_type_ids',torch.zeros_like(prompt['input_ids']))[0].detach().cpu()
    d=dict(input_ids=torch.cat([prompt_ids,response]),labels=torch.cat([torch.full((start,),-100,dtype=torch.long),response]),
           mm_token_type_ids=torch.cat([mm,torch.zeros(len(response),dtype=mm.dtype)]),
           video_grid_thw=prompt['video_grid_thw'].detach().cpu(),pixel_values_videos=prompt['pixel_values_videos'].detach().cpu())
    d=append_targets(processor,d,start);d['ptd_prefix_lengths']=d.pop('ptd_prefix_length').reshape(1)
    for k in ['input_ids','labels','mm_token_type_ids','attention_mask','ptd_position_ids','ptd_context_limits']:d[k]=d[k][None]
    if 'second_per_grid_ts' in prompt:d['second_per_grid_ts']=prompt['second_per_grid_ts'].detach().cpu()
    return {k:v.to(prompt['input_ids'].device) for k,v in d.items()}

def optimizer_counters(adapter,optimizer):
    parameters={p:n for n,p in adapter.named_parameters()}
    assert all(isinstance(p,torch.nn.Parameter) and p in parameters for p in optimizer.state)
    return {parameters[p]:int(state['step'].item()) for p,state in optimizer.state.items() if state}

def checked_step(adapter,optimizer):
    before=optimizer_counters(adapter,optimizer)
    grads={n for n,p in adapter.named_parameters() if p.grad is not None}
    if not grads:raise ValueError('empty gradient window')
    norm=float(torch.nn.utils.clip_grad_norm_([p for p in adapter.parameters() if p.requires_grad],1.))
    if not math.isfinite(norm):raise ValueError('nonfinite full-source gradient')
    optimizer.step();after=optimizer_counters(adapter,optimizer)
    if not all(torch.isfinite(p).all() for p in adapter.parameters()):raise ValueError('nonfinite adapter after Adam')
    if not all(torch.isfinite(v).all() for state in optimizer.state.values() for v in state.values() if isinstance(v,torch.Tensor)):
        raise ValueError('nonfinite live Adam state')
    for name in set(before)|set(after)|grads:
        assert after.get(name,0)==before.get(name,0)+int(name in grads),name
    validate_serialized_optimizer(optimizer.state_dict());optimizer.zero_grad(set_to_none=True)
    return dict(norm=norm,clipped=norm>1,actual_counters=after,gradient_parameter_names=sorted(grads))
