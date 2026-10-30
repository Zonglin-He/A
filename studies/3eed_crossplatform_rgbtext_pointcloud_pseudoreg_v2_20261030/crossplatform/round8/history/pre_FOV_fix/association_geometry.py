"""R8 non-learning physical-proposal association and robust center consensus.
Pure NumPy. No model/annotation/GT/RGB decoder; caller supplies full-caption frozen
CLIP cosine per explicitly returned tight/context RGB ROI. Never changes q_source.
"""
from dataclasses import dataclass,asdict
import hashlib
import numpy as np

@dataclass(frozen=True)
class AssociationConfig:
    text_pool: int=32
    pc_complement: int=16
    max_physical_rois: int=16
    source_cover_rois: int=8
    min_visible_support: int=8
    support_margin_fraction: float=.05
    support_margin_m: float=.05
    dedup_jaccard_min: float=.45
    dedup_overlap_min: float=.65
    dedup_box_iou_min: float=.10
    dedup_shared_points_min: int=8
    context_expansion_fraction: float=.20
    context_min_pixels: float=4.
    min_crop_side_pixels: int=4
    tight_semantic_weight: float=.5
    context_semantic_weight: float=.5
    semantic_pc_term: float=.02
    semantic_margin_scale: float=.05
    support_confidence_full_points: int=32
    geometry_representative_max_normalized_difference: float=.02
    min_center_spread_m: float=.01
    member_text_weight_floor: float=.05
    huber_fraction_of_anchor_size: float=.10
    huber_min_m: float=.03
    irls_iterations: int=3
    consensus_dispersion_fraction: float=.15
    max_center_delta_fraction: float=.15
    max_center_delta_m: float=.30
DEFAULT_CONFIG=AssociationConfig()


def aabb_iou(a,b):
    inter=np.maximum(0,np.minimum(a[:3]+a[3:]/2,b[:3]+b[3:]/2)-np.maximum(a[:3]-a[3:]/2,b[:3]-b[3:]/2)).prod()
    union=a[3:].prod()+b[3:].prod()-inter
    return float(inter/union) if union>0 else 0.


def visible_observations(points,uv,depth,image_size_wh):
    p=np.asarray(points,np.float64);u=np.asarray(uv,np.float64);d=np.asarray(depth,np.float64);W,H=map(int,image_size_wh)
    assert p.ndim==2 and p.shape[1]==3 and u.shape==(len(p),2) and d.shape==(len(p),) and W>0 and H>0
    valid=np.isfinite(p).all(1)&np.isfinite(u).all(1)&np.isfinite(d)&(d>0)
    pix=np.zeros_like(u,dtype=np.int64);pix[valid]=np.floor(u[valid]+.5).astype(np.int64)
    valid&=(pix[:,0]>=0)&(pix[:,0]<W)&(pix[:,1]>=0)&(pix[:,1]<H)
    ids=np.flatnonzero(valid)
    if not len(ids):return np.empty((0,3)),np.empty((0,2),np.int64)
    key=pix[ids,1]*W+pix[ids,0];x=p[ids];order=np.lexsort((x[:,2],x[:,1],x[:,0],d[ids],key));ids=ids[order];key=key[order]
    ids=ids[np.r_[True,key[1:]!=key[:-1]]]
    # Canonical XYZ ordering ensures support sets/consensus independent of P order.
    xyz,first=np.unique(p[ids],axis=0,return_index=True)
    return xyz,pix[ids[first]]


def support_ids(points,box,cfg):
    margin=cfg.support_margin_fraction*box[3:]+cfg.support_margin_m
    return frozenset(np.flatnonzero((np.abs(points-box[:3])<=box[3:]/2+margin).all(1)).tolist())


def pair_evidence(a,b,sa,sb,cfg):
    shared=len(sa&sb);union=len(sa|sb);jac=shared/union if union else 0.;over=shared/min(len(sa),len(sb)) if sa and sb else 0.;iou=aabb_iou(a,b)
    same=(shared>=cfg.dedup_shared_points_min and jac>=cfg.dedup_jaccard_min and over>=cfg.dedup_overlap_min and iou>=cfg.dedup_box_iou_min)
    return {'shared_points':shared,'jaccard':jac,'support_overlap_coefficient':over,'box_AABB_IoU':iou,'duplicate_edge':same}


def crop_from_support(pixels,ids,image_size_wh,cfg):
    if not ids:return None
    W,H=map(int,image_size_wh);xy=pixels[np.array(sorted(ids),np.int64)]
    lo=xy.min(0).astype(float);hi=xy.max(0).astype(float)+1
    if np.any(hi-lo<cfg.min_crop_side_pixels):return None
    tight=np.array([lo[0],lo[1],hi[0],hi[1]])
    extra=np.maximum((hi-lo)*cfg.context_expansion_fraction,cfg.context_min_pixels)
    context=np.array([max(0,lo[0]-extra[0]),max(0,lo[1]-extra[1]),min(W,hi[0]+extra[0]),min(H,hi[1]+extra[1])])
    return tight.astype(np.int64),np.array([np.floor(context[0]),np.floor(context[1]),np.ceil(context[2]),np.ceil(context[3])],np.int64)


def prepare_candidates(*,points,uv,depth,image_size_wh,source_boxes,text_scores,source_q=None,config=DEFAULT_CONFIG):
    """Stage1, no semantic scores/GT. ROI order locks before CLIP encoding.
    At most48 queries -> complete-link physical groups -> deterministic source/PC
    coverage <=16 groups, each tight and context ROI. q_source stays original.
    """
    cfg=config;boxes=np.asarray(source_boxes,np.float64);scores=np.asarray(text_scores,np.float64)
    assert boxes.ndim==2 and boxes.shape[1]==6 and scores.shape==(len(boxes),) and np.isfinite(boxes).all() and np.isfinite(scores).all()
    q0=int(np.argmax(scores)) if source_q is None else int(source_q);assert q0==int(np.argmax(scores))
    p,pix=visible_observations(points,uv,depth,image_size_wh)
    supports=[support_ids(p,b,cfg) if (b[3:]>0).all() else frozenset() for b in boxes]
    valid=[q for q in range(len(boxes)) if (boxes[q,3:]>0).all() and len(supports[q])>=cfg.min_visible_support]
    text_order=sorted(valid,key=lambda q:(-scores[q],q));pc_order=sorted(valid,key=lambda q:(-len(supports[q]),-scores[q],q))
    pool=text_order[:cfg.text_pool];pool+= [q for q in pc_order if q not in pool][:cfg.pc_complement]
    groups=[];edges={}
    def compatible(a,b):
        key=tuple(sorted((a,b)))
        if key not in edges:edges[key]=pair_evidence(boxes[a],boxes[b],supports[a],supports[b],cfg)
        return edges[key]['duplicate_edge']
    for q in pool:
        # Complete-link forbids transitive overlap chains merging distant objects.
        options=[g for g in groups if all(compatible(q,r) for r in g)]
        if options:
            best=min(options,key=lambda g:(-min(edges[tuple(sorted((q,r)))]['jaccard'] for r in g),groups.index(g)));best.append(q)
        else:groups.append([q])
    descriptions=[]
    for gid,members in enumerate(groups):
        anchor=min(members,key=lambda q:(-scores[q],q));union=frozenset().union(*(supports[q] for q in members));crop=crop_from_support(pix,union,image_size_wh,cfg)
        descriptions.append({'group_id':gid,'members':members,'anchor_query':anchor,'source_text_score':float(scores[anchor]),'support_ids':union,'visible_support_count':len(union),'tight_xyxy':None if crop is None else crop[0],'context_xyxy':None if crop is None else crop[1]})
    eligible=[g for g in descriptions if g['tight_xyxy'] is not None]
    bytext=sorted(eligible,key=lambda g:(-g['source_text_score'],g['anchor_query']));bypc=sorted(eligible,key=lambda g:(-g['visible_support_count'],-g['source_text_score'],g['anchor_query']))
    chosen=bytext[:min(cfg.source_cover_rois,cfg.max_physical_rois)]
    for g in bypc:
        if g not in chosen:chosen.append(g)
        if len(chosen)>=cfg.max_physical_rois:break
    chosen=chosen[:cfg.max_physical_rois]
    membership=np.full(len(boxes),-1,np.int64)
    for g in descriptions:
        for q in g['members']:membership[q]=g['group_id']
    return {'q_source':q0,'source_boxes':boxes,'source_text_scores':scores,'supports':supports,'visible_points':p,'visible_pixels':pix,'groups':descriptions,'roi_groups':chosen,'candidate_pool':pool,'dedup_membership':membership,'config':cfg,'image_size_wh':tuple(image_size_wh),'pair_evidence':edges,'presemantic_pool_policy':'top32text+16PC complement; complete-link support+spatial; up to8source then PCfill to16ROIs'}


def weighted_median(values,weights):
    order=np.argsort(values,kind='stable');v=values[order];w=weights[order]
    return float(v[np.searchsorted(np.cumsum(w),w.sum()/2,side='left')])


def geometry_representatives(members,boxes,scores,cfg):
    reps=[]
    for q in sorted(members,key=lambda q:(-scores[q],q)):
        if all(max(np.max(np.abs(boxes[q,:3]-boxes[r,:3])/np.maximum(boxes[r,3:],1e-9)),np.max(np.abs(boxes[q,3:]-boxes[r,3:])/np.maximum(boxes[r,3:],1e-9)))>cfg.geometry_representative_max_normalized_difference for r in reps):reps.append(q)
    return reps


def associate_and_consensus(prepared,semantic_cosines,*,mode='rgb_fulltext',target_phrase_cosines=None,image_plane_relation_diagnostics=None):
    """Stage2. semantic_cosines[G,2]=tight/context fullcaption frozen CLIP cosine.
    Optional phrase/relation fields are retained diagnostics only, not extra rules.
    Modes rgb_fulltext / pc_only / semantic_shuffle. No learned/GT box generator.
    """
    assert mode in ('rgb_fulltext','pc_only','semantic_shuffle')
    p=prepared;cfg=p['config'];groups=p['roi_groups'];boxes=p['source_boxes'];scores=p['source_text_scores'];q0=p['q_source']
    result={'original_source_q':q0,'associated_q':q0,'association_only_box':boxes[q0].copy(),'pseudo6':boxes[q0].copy(),'raw_consensus_center':boxes[q0,:3].copy(),'r_global':0.,'w6':np.zeros(6),'mode':mode,'yaw':0.,'diagnostics':{'reason':'no_eligible_physical_ROI','physical_group_count':len(p['groups']),'semantic_ROI_count':len(groups),'source_candidate_pool_count':len(p['candidate_pool']),'q0_not_retagged_as_q1':True,'optional_target_phrase_scores_used_for_decision':False,'optional_image_relation_used_for_decision':False}}
    if not groups:return result
    cos=np.asarray(semantic_cosines,np.float64);assert cos.shape==(len(groups),2) and np.isfinite(cos).all() and (np.abs(cos)<=1+1e-6).all()
    used=np.roll(cos,1,axis=0) if mode=='semantic_shuffle' else cos.copy()
    semantic=cfg.tight_semantic_weight*used[:,0]+cfg.context_semantic_weight*used[:,1]
    support=np.array([g['visible_support_count'] for g in groups],float);pc=np.log1p(support)/max(np.log1p(support).max(),1.)
    rank=pc if mode=='pc_only' else semantic+cfg.semantic_pc_term*pc
    best=min(range(len(groups)),key=lambda i:(-rank[i],groups[i]['anchor_query']))
    group=groups[best];q1=group['anchor_query'];anchor=boxes[q1];others=[i for i in range(len(groups)) if i!=best]
    if mode=='pc_only':margin=(support[best]-max([support[i] for i in others],default=support[best]))/max(support[best],1.);margin_conf=float(np.clip(margin,0,1))
    else:margin=float(semantic[best]-max([semantic[i] for i in others],default=semantic[best]));margin_conf=float(np.clip(margin/cfg.semantic_margin_scale,0,1))
    result.update(associated_q=q1,association_only_box=anchor.copy(),pseudo6=anchor.copy(),raw_consensus_center=anchor[:3].copy())
    source_gid=int(p['dedup_membership'][q0]);same=source_gid==group['group_id'] and source_gid>=0
    reps=geometry_representatives(group['members'],boxes,scores,cfg)
    diag=result['diagnostics'];diag.update(reason='insufficient_geometric_consensus',selected_group=group['group_id'],selected_group_members=list(group['members']),geometry_representatives=reps,source_group=source_gid,source_and_associated_same_dedup_group=same,q0_equals_q1=q0==q1,canonical_anchor_rule='maximum original puretext source score within selected physical group',semantic_cosines_original=cos.tolist(),semantic_cosines_used=used.tolist(),group_semantic_fused=semantic.tolist(),group_pc_support=support.tolist(),group_joint_rank=rank.tolist(),semantic_or_PC_margin=margin,margin_confidence=margin_conf,target_phrase_diagnostics_provided=target_phrase_cosines is not None,image_relation_diagnostics_provided=image_plane_relation_diagnostics is not None)
    if len(reps)<2:return result
    centers=boxes[reps,:3];spread=np.ptp(centers,axis=0)
    if float(spread.max())<cfg.min_center_spread_m:diag['reason']='geometric_center_spread_below_resolution';return result
    textmax=max(float(scores[reps].max()),1e-12)
    base=np.sqrt(np.array([len(p['supports'][q]) for q in reps],float))*np.clip(scores[reps]/textmax,cfg.member_text_weight_floor,1.)
    center=np.array([weighted_median(centers[:,j],base) for j in range(3)])
    delta_scale=np.maximum(cfg.huber_fraction_of_anchor_size*anchor[3:],cfg.huber_min_m)
    for _ in range(cfg.irls_iterations):
        residual=centers-center;robust=np.minimum(1.,delta_scale[None]/np.maximum(np.abs(residual),1e-12));weights=base[:,None]*robust;center=(weights*centers).sum(0)/weights.sum(0)
    mad=np.array([weighted_median(np.abs(centers[:,j]-center[j]),base) for j in range(3)])
    support_conf=float(np.clip((group['visible_support_count']-cfg.min_visible_support)/(cfg.support_confidence_full_points-cfg.min_visible_support),0,1))
    member_conf=float(np.clip((len(reps)-1)/3,0,1));dispersion=1/(1+(mad/np.maximum(cfg.consensus_dispersion_fraction*anchor[3:],1e-12))**2)
    delta=center-anchor[:3];limit=np.minimum(cfg.max_center_delta_fraction*anchor[3:],cfg.max_center_delta_m)
    qualified=np.abs(delta)<=limit
    r_global=margin_conf*support_conf*member_conf
    # Global confidence shrinks the *same pseudo target* used by direct and TTA.
    # It is not hidden in normalized w, where common factors would cancel.
    w=dispersion*qualified if r_global>0 else np.zeros(3)
    result['raw_consensus_center']=center.copy();result['r_global']=float(r_global)
    result['pseudo6'][:3]=anchor[:3]+np.where(qualified,r_global*delta,0.)
    result['w6'][:3]=w
    # No jitter, no forced minimum positive w or manufactured displacement.
    diag.update(reason='consensus_available' if w.any() else 'zero_confidence_or_axis_shift_rejected',consensus_center_spread_m=spread.tolist(),consensus_weighted_MAD_m=mad.tolist(),raw_consensus_minus_anchor_m=delta.tolist(),r_global=float(r_global),final_pseudo_minus_anchor_m=(result['pseudo6'][:3]-anchor[:3]).tolist(),per_axis_shift_limit_m=limit.tolist(),qualified_center_axes=qualified.tolist(),support_confidence=support_conf,multiple_representative_confidence=member_conf,per_axis_dispersion_confidence=dispersion.tolist(),consensus_member_base_weights=base.tolist(),nonzero_pseudo_supervision=bool(np.any((w>0)&(np.abs(result['pseudo6'][:3]-anchor[:3])>0))))
    return result
