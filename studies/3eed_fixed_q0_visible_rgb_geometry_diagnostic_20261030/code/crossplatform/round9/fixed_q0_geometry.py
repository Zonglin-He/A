"""R9 identity-preserving geometry: only original q0's prepared ROI group.
Consensus/size prior and conservative margin arithmetic retained from frozen R8.
No query reselection, GT consumer or RGB-mismatch loss; pure NumPy CPU module.
"""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'round8'))
from association_geometry import (prepare_candidates,DEFAULT_CONFIG,weighted_median,geometry_representatives)

def fixed_q0_pseudo(prepared,semantic_cosines,*,mode='rgb_fulltext',target_phrase_cosines=None,image_plane_relation_diagnostics=None):
    """Stage2. semantic_cosines[G,2]=tight/context fullcaption frozen CLIP cosine.
    Optional phrase/relation fields are retained diagnostics only, not extra rules.
    Modes rgb_fulltext / pc_only / semantic_shuffle. No learned/GT box generator.
    """
    assert mode in ('rgb_fulltext','pc_only')
    p=prepared;cfg=p['config'];groups=p['roi_groups'];boxes=p['source_boxes'];scores=p['source_text_scores'];q0=p['q_source']
    result={'original_source_q':q0,'associated_q':q0,'association_only_box':boxes[q0].copy(),'pseudo6':boxes[q0].copy(),'raw_consensus_center':boxes[q0,:3].copy(),'r_global':0.,'w6':np.zeros(6),'mode':mode,'yaw':0.,'diagnostics':{'reason':'no_eligible_physical_ROI','physical_group_count':len(p['groups']),'semantic_ROI_count':len(groups),'source_candidate_pool_count':len(p['candidate_pool']),'q0_not_retagged_as_q1':True,'optional_target_phrase_scores_used_for_decision':False,'optional_image_relation_used_for_decision':False}}
    result['diagnostics']['selection_policy']='force_original_q0_group_no_reselect'
    if np.any(boxes[q0,3:]<=0):result['diagnostics']['reason']='q0_nonpositive_source_size';return result
    group_id=int(p['dedup_membership'][q0])
    if group_id<0:result['diagnostics']['reason']='q0_not_in_prepared_pool';return result
    eligible=[i for i,g in enumerate(groups) if g['group_id']==group_id and q0 in g['members']]
    if not eligible:result['diagnostics']['reason']='q0_group_not_in_semantic_ROIs';return result
    assert len(eligible)==1
    cos=np.asarray(semantic_cosines,np.float64);assert cos.shape==(len(groups),2) and np.isfinite(cos).all() and (np.abs(cos)<=1+1e-6).all()
    used=cos.copy()
    semantic=cfg.tight_semantic_weight*used[:,0]+cfg.context_semantic_weight*used[:,1]
    support=np.array([g['visible_support_count'] for g in groups],float);pc=np.log1p(support)/max(np.log1p(support).max(),1.)
    rank=pc if mode=='pc_only' else semantic+cfg.semantic_pc_term*pc
    unconstrained_best=min(range(len(groups)),key=lambda i:(-rank[i],groups[i]['anchor_query']))
    best=eligible[0]
    group=groups[best];q1=q0;anchor=boxes[q0];others=[i for i in range(len(groups)) if i!=best]
    if mode=='pc_only':margin=(support[best]-max([support[i] for i in others],default=support[best]))/max(support[best],1.);margin_conf=float(np.clip(margin,0,1))
    else:margin=float(semantic[best]-max([semantic[i] for i in others],default=semantic[best]));margin_conf=float(np.clip(margin/cfg.semantic_margin_scale,0,1))
    result.update(associated_q=q1,association_only_box=anchor.copy(),pseudo6=anchor.copy(),raw_consensus_center=anchor[:3].copy())
    source_gid=int(p['dedup_membership'][q0]);same=source_gid==group['group_id'] and source_gid>=0
    reps=geometry_representatives(group['members'],boxes,scores,cfg)
    diag=result['diagnostics'];diag.update(reason='insufficient_geometric_consensus',selected_group=group['group_id'],selected_group_members=list(group['members']),geometry_representatives=reps,source_group=source_gid,source_and_associated_same_dedup_group=same,q0_equals_q1=q0==q1,canonical_anchor_rule='fixed original q0, never select a different canonical query',unconstrained_rank_winner_group_diagnostic_only=groups[unconstrained_best]['group_id'],fixed_q0_group_is_unconstrained_winner=(unconstrained_best==best),RGB_mismatch_diagnostic_only=(mode=='rgb_fulltext' and unconstrained_best!=best),semantic_cosines_original=cos.tolist(),semantic_cosines_used=used.tolist(),group_semantic_fused=semantic.tolist(),group_pc_support=support.tolist(),group_joint_rank=rank.tolist(),semantic_or_PC_margin=margin,margin_confidence=margin_conf,target_phrase_diagnostics_provided=target_phrase_cosines is not None,image_relation_diagnostics_provided=image_plane_relation_diagnostics is not None)
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
