"""Structured source objectives and explicit branch actuation scopes.

No data access, GPU initialization, optimizer or state selection at import.
The GT-conditioned MTP objective deliberately does not claim free native
cached support equivalence. Original masks and model support stay unchanged.
"""
from __future__ import annotations
import math
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from vg_tta.desta3d_v2_source import split_source_loss_masks, _single_token_id


def configure_branch(adapter, branch: str, level: int = 0):
    if branch not in ('event','spatial') or level not in (0,1,2,3):
        raise ValueError((branch,level))
    if adapter.architecture != 'dual3d':raise ValueError('requires independent dual readers')
    prefixes=[f'film_{branch}.',f'norm_{branch}.']
    exact=set()
    if level>=1:prefixes.append(f'out_proj_{branch}.');exact.add(f'gate_{branch}')
    if level>=2:prefixes.append(f'{branch}_reader.pointwise.')
    if level>=3:prefixes.extend(['input_proj.','shared_stem.'])
    named=[]
    for name,p in adapter.named_parameters():
        p.grad=None;p.requires_grad_(name in exact or any(name.startswith(s) for s in prefixes))
        if p.requires_grad:named.append((name,p))
    assert named and all(not n.startswith('norm_stem.') for n,p in named)
    return named


def detached_referent_pool(event_features, referent_logits, tau=1.):
    """Separate experimental factor; never mutates the v2 event head."""
    if not math.isfinite(tau) or tau<=0:raise ValueError('positive finite tau required')
    if event_features.shape[:-1]!=referent_logits.shape or event_features.ndim!=5:
        raise ValueError('expected [B,T,H,W,C] and [B,T,H,W]')
    weights=(referent_logits.detach().float()/tau).flatten(-2).softmax(-1).reshape_as(referent_logits)
    return (weights[...,None]*event_features).sum((2,3)),weights


def structured_support(data, tokenizer):
    old=split_source_loss_masks(data,tokenizer)
    spans=old['source_response_spans']; prefix=int(data['ptd_prefix_lengths'].reshape(-1)[0])
    inp=data['input_ids'].detach().cpu()[0]; labels=data['labels'].detach().cpu()[0]
    pos=data['ptd_position_ids'].detach().cpu()[0]; T=int(data['video_grid_thw'][0,0])
    if not 1<=T<=100:raise ValueError('actual time support outside learned vocabulary')
    time_ids=[_single_token_id(tokenizer,f'<t{i}>') for i in range(1,T+1)]
    coord_ids=[_single_token_id(tokenizer,f'<{i}>') for i in range(1001)]
    time_start=spans['time_span_inclusive'][0]
    endpoint_positions=[time_start+1,time_start+2]
    box_positions=[p for start,end in spans['box_spans_inclusive'] for p in range(start+2,start+6)]
    positions={k:[] for k in ['event','spatial']};targets={k:[] for k in positions}
    # Select only real MTP targets matching those prefix positions, not NTP
    # copies, frame anchors, markers, semantic tokens, or MTP <null> padding.
    mapping={}
    null=spans['token_ids']['null']
    for target_pos in old['target_positions'].tolist():
        if target_pos<=prefix or int(labels[target_pos])==null:continue
        origin=int(pos[target_pos-1])+1
        if origin in endpoint_positions+box_positions:
            if origin in mapping:raise ValueError('duplicated structured MTP origin')
            if int(labels[target_pos])!=int(inp[origin]):raise ValueError('MTP target differs from original prefix')
            mapping[origin]=target_pos-1
    for branch,origins,classes in [('event',endpoint_positions,time_ids),('spatial',box_positions,coord_ids)]:
        for origin in origins:
            if origin not in mapping:raise ValueError('missing structured MTP position')
            positions[branch].append(mapping[origin]);targets[branch].append(classes.index(int(inp[origin])))
    assert len(positions['event'])==2 and len(positions['spatial'])%4==0
    return dict(positions=positions,targets=targets,time_ids=time_ids,coordinate_ids=coord_ids,
                T=T,boxes=len(box_positions)//4,original=old,
                conditioning='official GT reference/time prefix; MTP context unchanged')


def structured_ce(logits, targets, branch):
    if branch not in ('event','spatial'):raise ValueError(branch)
    if logits.ndim!=2 or not torch.isfinite(logits).all():raise ValueError('finite full-support logits required')
    if branch=='event' and logits.shape[0]!=2:raise ValueError('two endpoints required')
    if branch=='spatial' and (logits.shape[0]%4 or not logits.shape[0]):raise ValueError('four coordinates per frame required')
    loss=F.cross_entropy(logits.float(),targets.long(),reduction='sum' if branch=='event' else 'mean')
    return loss


def source_branch_objective(model,data,support,branch,regularizer_weight=.05):
    """Original frozen BF16 head; structured restricted support + full-vocab CE.

    Across event and spatial calls, .05 of each original branch CE equals
    lambda_PTD=.1 times their mean. No synthetic precision replacement.
    """
    if branch not in ('event','spatial') or regularizer_weight<0:raise ValueError(branch)
    labels=data['labels'];old=support['original'];mask=old[branch].to(labels.device)[0]
    target_positions=torch.nonzero(mask,as_tuple=False).flatten()
    h=model.model(**{k:v for k,v in data.items() if k!='labels'},use_cache=False).last_hidden_state[0]
    structured_positions=support['positions'][branch]
    classes=support['time_ids' if branch=='event' else 'coordinate_ids']
    selected_index={p:i for i,p in enumerate(structured_positions)}
    reduced=[];original_sum=h.sum()*0
    target=labels[0,target_positions]
    for start in range(0,len(target_positions),32):
        ps=target_positions[start:start+32]-1
        local=[(j,selected_index[int(p)]) for j,p in enumerate(ps) if int(p) in selected_index]
        # Captured loop metadata is bound to defaults, including recomputation.
        def project(hh,tt,local=tuple(local),classes=tuple(classes)):
            logits=model.lm_head(hh).float()
            ce=F.cross_entropy(logits,tt,reduction='sum')
            ix=[j for j,_ in local]
            return ce,logits[ix][:,list(classes)]
        full,sel=checkpoint(project,h[ps],target[start:start+32],use_reentrant=False)
        original_sum=original_sum+full
        for j,(_,order) in enumerate(local):reduced.append((order,sel[j]))
    if len(reduced)!=len(structured_positions):raise ValueError('structured target not covered by branch')
    selected=torch.stack([v for _,v in sorted(reduced)])
    truth=torch.tensor(support['targets'][branch],device=selected.device)
    structured=structured_ce(selected,truth,branch)
    original=original_sum/len(target_positions)
    return structured+regularizer_weight*original,dict(structured=structured,original=original,
        selected_logits=selected,targets=truth,original_targets=len(target_positions),
        structured_targets=len(structured_positions),regularizer_weight=regularizer_weight)
