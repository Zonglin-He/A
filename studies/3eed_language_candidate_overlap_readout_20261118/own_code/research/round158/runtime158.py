"""Inference entry for a new observation. All tensors are produced by the frozen source model.

A view is a dict with prefix (physical XYZ/indices and the six decoder positions),
visual (1024 x 288 bottom-up point features), and graph (calibrated RGB bank).
For optional B, the prefix uses the same A anchor indices and the existing source
position normalization. Its point features and RGB graph are registered onto A.
"""
import hashlib
import numpy as np
import torch

from evidence158 import head_span,phrase_scores,role_plan,token_input
from model51 import SENTINEL,firstnoun_mask
from reader158 import select_from_description


@torch.inference_mode()
def observe_roles(base,matcher,caption,graph,a,b=None):
    items,phrases,edges=role_plan(caption,graph)
    subject=items[0][1] if items else None
    fallback=firstnoun_mask(base,caption+SENTINEL)[0].numpy()
    conditioned=[(caption,head_span(caption,subject),fallback)]
    conditioned.extend((text,head_span(text,entity,True),None) for text,(_,entity) in zip(phrases,items))
    language=[]
    for text,span,prior in conditioned:
        tokens,mask=token_input(base,text,span,prior)
        language.append((base.text_projector(base.text_encoder(**tokens).last_hidden_state),mask))
    def encode(view):
        return phrase_scores(base,matcher,view['prefix'],view['graph'],conditioned,view['visual'],language)
    scores_a=encode(a)
    if b is None:
        return scores_a,scores_a.copy(),np.zeros(256),edges
    za,zb=a['prefix'],b['prefix']
    if not torch.equal(za['query_points_sample_inds'],zb['query_points_sample_inds']) or not torch.equal(za['xyz'],zb['xyz']):
        raise ValueError('B must be registered on A anchor slots')
    coverage=np.asarray(b['coverage'])
    if coverage.shape!=(256,) or not np.isfinite(coverage).all() or np.any((coverage<0)|(coverage>1)):
        raise ValueError('invalid local coverage')
    return scores_a,encode(b),coverage,edges


def ground_description(base,matcher,caption,graph,boxes,projected_bounds,visible,a,b=None):
    """Return one A256 index. No labels, prior winners, or historical output caches."""
    left,right,coverage,edges=observe_roles(base,matcher,caption,graph,a,b)
    target=next((ent['span'] for ent in graph['entities'] if ent['role']=='T'),None)
    evidence=dict(A=left,B=right,coverage=coverage,rect=projected_bounds,seen=visible,
                  signs=np.array([sign for _,sign in edges]),boxes=boxes,
                  box_sha=hashlib.sha256(boxes.tobytes()).hexdigest())
    return select_from_description(evidence,caption,target)


@torch.inference_mode()
def ground_context_description(base,matcher,caption,parsed,boxes,projected_bounds,visible,
                              main_view,full_sentence_competition,second_view=None):
    """Revised winner for a live frame. The existing full competition vector is one input.

    full_sentence_competition is the frozen A+B readout on A anchors (full A for a
    single observation). Obtain it before argmax. No previous winner is used.
    """
    from context158 import select_contextual_winner
    subject=next((e for e in parsed['entities'] if e['role']=='T'),None)
    fallback=firstnoun_mask(base,caption+SENTINEL)[0].numpy()
    tok,mask=token_input(base,caption,head_span(caption,subject),fallback)
    lang=[(base.text_projector(base.text_encoder(**tok).last_hidden_state),mask)]
    def native(view):
        return phrase_scores(base,matcher,view['prefix'],view['graph'],[caption],view['visual'],lang)[0]
    la=native(main_view)
    lb=None; coverage=None
    if second_view is not None:
        for key in ('xyz','query_points_sample_inds'):
            if not torch.equal(main_view['prefix'][key],second_view['prefix'][key]):
                raise ValueError('unregistered observation in context readout')
        lb=native(second_view);coverage=second_view['coverage']
    return select_contextual_winner(boxes,caption,None if subject is None else subject['span'],
            projected_bounds,visible,full_sentence_competition,la,lb,coverage)
