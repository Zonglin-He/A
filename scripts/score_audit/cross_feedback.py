"""Single-hop held-anchor feedback from fixed Source feature correspondence.

Proposal construction must happen before this module reads any held teacher
target. ``source_only_affinity`` and ``predict_held_support`` therefore accept
no teacher masks, scores, boxes, or query-selection outputs. Only the final
``held_feedback`` operation accepts the independently held target.

This is an offline scoring probe, not an adaptation method. Top20 cosine/.1
temperature follow the existing Source transport arithmetic. Local feature
aliasing, incorrect correspondences, occlusion, and soft target extent can still
make this score prefer a wrong, overly large, or shrunken box. No empirical
reliability or OPD advantage is established by implementing it.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


TOPK = 20
TEMPERATURE = .1
EPS = 1e-15


def _require(condition,message):
    if not condition:
        raise ValueError(message)


def _cpu(value,name,dtype=None):
    value = torch.as_tensor(value)
    _require(value.device.type=='cpu',name+' must already be on CPU')
    return value.to(dtype=dtype) if dtype is not None else value


@dataclass(frozen=True)
class CrossAffinity:
    """Source-only transport from one node's cells to a distinct held node.

    Rows index held cells; columns index source cells. The owner may cache this
    object for a fixed Source-feature/valid-mask pair. A (source,held) key alone
    is safe only within that immutable query; never reuse it across queries.
    """
    matrix: torch.Tensor
    source_valid: torch.Tensor
    held_valid: torch.Tensor
    source_node: int
    held_node: int


def _validate_affinity(affinity):
    _require(isinstance(affinity,CrossAffinity),'A Source-only CrossAffinity is required')
    _require(affinity.source_node!=affinity.held_node,'A held node cannot score itself')
    _require(affinity.matrix.device.type=='cpu' and affinity.matrix.dtype==torch.float64,
             'Cached affinity must be FP64 on CPU')
    _require(affinity.source_valid.dtype==affinity.held_valid.dtype==torch.bool,
             'Cached cell support must be boolean')
    _require(affinity.matrix.shape==(len(affinity.held_valid),len(affinity.source_valid)),
             'Affinity dimensions must match its cell support')


def source_only_affinity(source_features,held_features,source_valid,held_valid,
                         *,source_node,held_node):
    """Build direct t→held row-normalized correspondence without teacher data.

    Normalize/cosine/topk/softmax use FP32, including sorted=False and operation
    order from propagation_v0.bank.transport. Selected weights are then cast to
    FP64 and normalized again before map transport. Padding cells are excluded.
    Empty valid support produces zero transport, never an invented match.
    """
    _require(isinstance(source_node,int) and isinstance(held_node,int) and
             source_node>=0 and held_node>=0 and source_node!=held_node,
             'Distinct nonnegative source and held node indices are required')
    source = _cpu(source_features,'source_features',torch.float32).detach()
    held = _cpu(held_features,'held_features',torch.float32).detach()
    sv = _cpu(source_valid,'source_valid',torch.bool).detach().clone()
    hv = _cpu(held_valid,'held_valid',torch.bool).detach().clone()
    _require(source.ndim==held.ndim==2 and source.shape[1]==held.shape[1] and
             source.shape[1]>0,'Source features must be compatible [cells,channels] grids')
    _require(sv.shape==(len(source),) and hv.shape==(len(held),),
             'Valid masks must match their Source grids')
    _require(bool(torch.isfinite(source[sv]).all()) and bool(torch.isfinite(held[hv]).all()),
             'Valid Source feature cells must be finite')
    source = torch.where(sv[:,None],source,torch.zeros_like(source))
    held = torch.where(hv[:,None],held,torch.zeros_like(held))
    source = F.normalize(source,dim=-1,eps=1e-12)
    held = F.normalize(held,dim=-1,eps=1e-12)
    source_ids = torch.nonzero(sv,as_tuple=False).flatten()
    held_ids = torch.nonzero(hv,as_tuple=False).flatten()
    matrix = torch.zeros(len(held),len(source),dtype=torch.float64)
    if len(source_ids) and len(held_ids):
        cosine = held[held_ids]@source[source_ids].T
        values,indices = torch.topk(cosine,min(TOPK,len(source_ids)),dim=1,sorted=False)
        weights = (values/TEMPERATURE).softmax(1).double()
        weights /= weights.sum(1,keepdim=True)
        matrix[held_ids[:,None].expand_as(indices),source_ids[indices]] = weights
    result = CrossAffinity(matrix,sv,hv,source_node,held_node)
    _validate_affinity(result)
    return result


def _shape(shape,name):
    _require(isinstance(shape,(tuple,list)) and len(shape)==2 and
             all(isinstance(x,int) and x>0 for x in shape),name+' must be positive (height,width)')
    return tuple(shape)


def box_occupancy(boxes,shape):
    """FP64 fractional cell occupancy of normalized cxcywh boxes, [K,H,W].

    Intersection with image cells naturally restricts visible support. Box
    centers and extents themselves are not clipped or changed. Zero-size or
    entirely off-image boxes have empty support; negative dimensions fail.
    """
    H,W = _shape(shape,'source_shape')
    boxes = _cpu(boxes,'boxes',torch.float64)
    _require(boxes.ndim==2 and boxes.shape[1]==4 and len(boxes)>0 and
             bool(torch.isfinite(boxes).all()) and bool((boxes[:,2:]>=0).all()),
             'Candidate boxes must be finite nonnegative-size [K,4] cxcywh')
    x0,y0 = (boxes[:,:2]-boxes[:,2:]/2).unbind(-1)
    x1,y1 = (boxes[:,:2]+boxes[:,2:]/2).unbind(-1)
    x = torch.arange(W+1,dtype=torch.float64)/W
    y = torch.arange(H+1,dtype=torch.float64)/H
    ox = (torch.minimum(x1[:,None],x[1:])-torch.maximum(x0[:,None],x[:-1])).clamp_min(0)*W
    oy = (torch.minimum(y1[:,None],y[1:])-torch.maximum(y0[:,None],y[:-1])).clamp_min(0)*H
    return oy[:,:,None]*ox[:,None,:]


def predict_held_support(boxes,affinity,*,source_shape,held_shape):
    """Predict support at the held node without receiving its teacher target.

    Batched matrix multiplication supports the three fixed candidates in one
    call. Affinity is one direct hop, not recursive or multi-hop propagation.
    """
    _validate_affinity(affinity)
    source_shape,held_shape = _shape(source_shape,'source_shape'),_shape(held_shape,'held_shape')
    _require(source_shape[0]*source_shape[1]==len(affinity.source_valid) and
             held_shape[0]*held_shape[1]==len(affinity.held_valid),
             'Grid shapes must match cached Source-only affinity')
    occupancy = box_occupancy(boxes,source_shape).flatten(1)
    occupancy = occupancy*affinity.source_valid[None]
    predicted = occupancy@affinity.matrix.T
    return predicted.reshape(len(occupancy),*held_shape)


def held_feedback(predicted_support,held_target,affinity):
    """Evaluate 2D soft Dice against the independently held teacher mask.

    This is the first and only API that accepts the held target. Empty/empty
    returns zero, and invalid held cells are excluded from both maps. No source
    candidate or affinity is constructed, changed, selected, or normalized using
    target content. Values are FP64; a scalar score is returned per candidate.
    """
    _validate_affinity(affinity)
    predicted = _cpu(predicted_support,'predicted_support',torch.float64)
    target = _cpu(held_target,'held_target',torch.float64).detach()
    _require(predicted.ndim==3 and target.ndim==2 and predicted.shape[1:]==target.shape
             and target.numel()==len(affinity.held_valid),
             'Held predictions and target must share [K,H,W] / [H,W] geometry')
    _require(bool(torch.isfinite(predicted).all()) and bool(torch.isfinite(target).all()) and
             bool(((predicted>=-1e-12)&(predicted<=1+1e-12)).all()) and
             bool(((target>=0)&(target<=1)).all()),'Support maps must be finite probabilities')
    valid = affinity.held_valid.reshape_as(target)
    p = (predicted*valid[None]).flatten(1)
    q = (target*valid).flatten()[None]
    score = 2*(p*q).sum(-1)/(p.sum(-1)+q.sum(-1)).clamp_min(EPS)
    return score.clamp(0,1)
