"""R50 CPU-only pool/W/readout draft. No files, labels, models or optimizers.

All geometry and evidence enter as fixed arrays. This module never changes
source candidates, source scores, subject prompts, crops, reference evidence,
fusion weights, anchor rules, clipping or query tie breaking.
"""
import numpy as np

ALLOWED_K=(3,10)
ARM_NAMES=['source','old_G','wideentry_oldW','oldentry_newW','wideentry_newW']

def exact(a,b,name):
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape or a.dtype!=b.dtype or a.tobytes()!=b.tobytes():
        raise AssertionError(('old_endpoint_bridge_not_exact',name))

def validate_counts(mask_counts,allvisible,unassigned,ambiguous,source_boxes):
    c=np.asarray(mask_counts);n=np.asarray(allvisible);u=np.asarray(unassigned);a=np.asarray(ambiguous);boxes=np.asarray(source_boxes)
    assert c.ndim==2 and c.shape[0]==256 and n.shape==u.shape==a.shape==(256,) and boxes.shape==(256,6)
    assert all(np.issubdtype(x.dtype,np.integer) and (x>=0).all() for x in (c,n,u,a))
    assert np.isfinite(boxes).all() and np.array_equal(c.sum(1)+u+a,n)
    invalid=(boxes[:,3:]<=0).any(1)
    assert (n[invalid]==0).all() and (c[invalid]==0).all()
    return c,n,u,a,invalid

def native_crop_valid(detector_boxes,image_size_wh,nonempty_masks):
    """Well-formed existing tight crop, not category/ranking/area selection.

    Same original-pixel int conversion/clipping as visualZ.crops(pad=0).
    A valid tight crop also permits the original pad=.2 context crop.
    """
    b=np.asarray(detector_boxes);nonempty=np.asarray(nonempty_masks,bool);W,H=map(int,image_size_wh)
    assert b.shape==(len(nonempty),4) and np.isfinite(b).all() and W>0 and H>0
    valid=np.zeros(len(b),bool)
    for m,(x1,y1,x2,y2) in enumerate(b):
        valid[m]=bool(nonempty[m] and x1<x2 and y1<y2 and min(W,int(x2))>max(0,int(x1)) and min(H,int(y2))>max(0,int(y1)))
    return valid

def wide_entry(old_pool,mask_counts,crop_valid):
    """Strict expansion: old pool union all mask C[:,m]>0 with a native crop.

    Counts are globally nearest-depth visible, uniquely mask-owned points
    strictly inside at least one positive-size original source box. No coarse
    category, source score, topK, GT, new threshold search or instance oracle.
    Retaining the original pool avoids deleting historical +.1m-halo support
    while claiming a pure expansion. The retained-old reason is explicit.
    """
    c=np.asarray(mask_counts);valid=np.asarray(crop_valid,bool);M=c.shape[1]
    old=np.asarray(old_pool,np.int64);assert len(np.unique(old))==len(old) and ((old>=0)&(old<M)).all()
    assert valid.shape==(M,) and valid[old].all(), 'historical_scored_mask_crop_contract_changed'
    supported=(c>0).any(0);extra=supported&valid
    pool=np.array(sorted(set(old.tolist())|set(np.flatnonzero(extra).tolist())),np.int64)
    reasons=[];oldset=set(old.tolist())
    for m in range(M):
        reasons.append('OLD_POOL_RETAINED' if m in oldset else 'NEW_EXCLUSIVE_PROJECTED_SUPPORT' if extra[m] else
                       'UNKNOWN_INVALID_NATIVE_CROP' if not valid[m] else 'NO_EXCLUSIVE_VISIBLE_POINT_INSIDE_SOURCE_POOL')
    return dict(pool=pool,additional_masks=np.array([m for m in pool if m not in oldset],np.int64),
                has_strict_projected_support=supported,crop_valid=valid,entry_reasons=reasons)

def rect_iou_association13(a,b):
    # R17 geometry imports this version, while the P/R relation scorer uses
    # round20/detect's separate denominator guard. Do not interchange them.
    a=np.asarray(a,float);b=np.asarray(b,float)
    inter=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod()
    union=np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-inter
    return float(inter/union) if union>0 else 0.

def legacy_geometry_from_visible(points,owner,depth,boxes,projected_rectangles,projected_valid,detector_boxes):
    """Exact matrix portion of R17 prepare, with precomputed original projections.

    Rectangles/valid must come from the unchanged camera/box_corners/project
    functions: all8 corners depth>.1 and finite UV, then untrimmed min/max UV.
    No sameclass pairing/component_mapping stage is required to recover these
    matrices. They cannot be reconstructed from normalized old W alone.
    """
    points=np.asarray(points);owner=np.asarray(owner);depth=np.asarray(depth);boxes=np.asarray(boxes)
    rect=np.asarray(projected_rectangles);validproj=np.asarray(projected_valid,bool);det=np.asarray(detector_boxes)
    assert boxes.shape==(256,6) and owner.shape==depth.shape==(len(points),) and rect.shape==(256,4) and validproj.shape==(256,)
    M=len(det);assert det.shape==(M,4) and np.isfinite(points).all() and np.isfinite(depth).all()
    masks=[]
    for m in range(M):
        ids=np.flatnonzero(owner==m)
        if len(ids)>=4:
            d=depth[ids];q1,q3=np.quantile(d,[.25,.75]);spread=max(q3-q1,.1)
            ids=ids[(d>=q1-1.5*spread)&(d<=q3+1.5*spread)]
        masks.append(ids)
    support=np.zeros((256,M),int);coverage=np.zeros((256,M));project_iou=np.zeros((256,M))
    for q,b in enumerate(boxes):
        if (b[3:]<=0).any():continue
        for m,ids in enumerate(masks):
            support[q,m]=np.count_nonzero((np.abs(points[ids]-b[:3])<=b[3:]/2+.1).all(1))
            coverage[q,m]=support[q,m]/max(len(ids),1)
            if validproj[q]:project_iou[q,m]=rect_iou_association13(rect[q],det[m])
    return dict(point_support_count=support,point_support_fraction=coverage,projection_iou=project_iou,
                point_count_after_depth_trim=np.array([len(x) for x in masks],np.int64))

def legacy_weights(pool,support,coverage,projection_iou,source_boxes):
    """R23 original quality/validity/foreground-normalization, any fixed pool."""
    support=np.asarray(support);coverage=np.asarray(coverage);pi=np.asarray(projection_iou);boxes=np.asarray(source_boxes)
    assert support.shape==coverage.shape==pi.shape and support.shape[0]==256
    assert np.isfinite(coverage).all() and np.isfinite(pi).all()
    valid=(support>=3)&(coverage>=.15)&(pi>=.1)&(boxes[:,3:]>0).all(1)[:,None]
    quality=np.where(valid,coverage*np.sqrt(np.maximum(pi,0)),0.)
    weights=quality[:,pool].copy() if len(pool) else np.zeros((256,0))
    weights/=np.maximum(weights.sum(1,keepdims=True),1e-12)
    return dict(weights=weights,quality=quality,valid=valid)

def visible_mass_weights(pool,mask_counts,allvisible,unassigned,ambiguous,source_boxes,k):
    """W_im=C_im/(N_i+k): explicit unknown mass, no foreground renormalization.

    N includes exclusive foreground of ALL masks, unassigned and ambiguous.
    The pseudo-count k is unknown mass, never a named foreground instance.
    Equivalent to (C/N)*(N/(N+k)); N=0 gives W=0 and unknown=1.
    """
    assert type(k) is int and k in ALLOWED_K
    c,n,u,a,invalid=validate_counts(mask_counts,allvisible,unassigned,ambiguous,source_boxes)
    denom=n.astype(np.float64)+k;selected=c[:,pool].astype(np.float64)
    weights=selected/denom[:,None];mass=weights.sum(1)
    prior=k/denom;outside=(c.sum(1)-c[:,pool].sum(1))/denom
    unassigned_mass=u/denom;ambiguous_mass=a/denom
    assert np.allclose(mass+prior+outside+unassigned_mass+ambiguous_mass,1,rtol=0,atol=1e-12)
    return dict(weights=weights,row_mass=mass,unknown_mass=1-mass,unknown_prior_mass=prior,
                out_of_pool_foreground_mass=outside,unassigned_mass=unassigned_mass,ambiguous_mass=ambiguous_mass,
                count_shrinkage=n/denom,invalid_source_geometry=invalid,k=k)

def legacy_operator(source_scores,weights,pool,e_all,relation_all):
    """Original legacy anchoring/clip/2V+2P+2R. Changed W propagates all cues.

    No calibrated meaning is assigned to e. Unknown reference observations
    remain zero exactly as the original relation scorer specifies. Required
    newly entered masks must have actual frozen-model evidence, never NaN->0.
    """
    s=np.asarray(source_scores);w=np.asarray(weights);pool=np.asarray(pool,np.int64)
    e=np.asarray(e_all)[pool];rv=np.asarray(relation_all)[pool]
    assert s.shape==(256,) and w.shape==(256,len(pool)) and rv.shape==(len(pool),2)
    assert np.isfinite(s).all() and np.isfinite(w).all()
    if not np.isfinite(e).all() or not np.isfinite(rv).all():raise ValueError('Required mask lacks cached/scored evidence; do not impute zero')
    q0=int(s.argmax());mapped=w.sum(1)>0;v=np.zeros(256);p=np.zeros(256);r=np.zeros(256)
    vv=w@e;rr=w@rv;av=ar=None
    if len(pool):
        av=vv[q0] if mapped[q0] else float(np.median(e))
        ar=rr[q0] if mapped[q0] else np.median(rv,axis=0)
        v[mapped]=np.clip((vv[mapped]-av)/2,-2,2)
        p[mapped]=np.clip(rr[mapped,0]-ar[0],-2,2)
        r[mapped]=np.clip(rr[mapped,1]-ar[1],-2,2)
    F=s.astype(float)+2*v+2*p+2*r
    return dict(scores=F,query=int(F.argmax()),vision=v,position=p,relation=r,mapped=mapped,
                vv=vv,rr=rr,anchor_v=av,anchor_r=ar,source_q0_mapped=bool(mapped[q0]),
                anchor_fallback_original_mask_median=bool(len(pool) and not mapped[q0]),row_mass=w.sum(1))

def five_readouts(source_scores,source_boxes,old_pool,crop_valid,mask_counts,allvisible,unassigned,ambiguous,
                  support,coverage,projection_iou,e_all,relation_all,k,*,expected_old_weights,expected_old_vpr):
    """Five conceptual rows for ONE k; root may lock k3/k10 as two trials.

    Both k trials share source/oldG/wideentry-oldW, so their union has seven
    unique evaluated configurations, not five. Report all trials/selection.
    """
    c,n,u,a,invalid=validate_counts(mask_counts,allvisible,unassigned,ambiguous,source_boxes)
    old=np.asarray(old_pool,np.int64);entry=wide_entry(old,c,crop_valid);wide=entry['pool']
    ow=legacy_weights(old,support,coverage,projection_iou,source_boxes)['weights']
    ww=legacy_weights(wide,support,coverage,projection_iou,source_boxes)['weights']
    exact(ow,expected_old_weights,'oldentry_oldW')
    oldstate=legacy_operator(source_scores,ow,old,e_all,relation_all)
    for name in ('vision','position','relation'):exact(oldstate[name],expected_old_vpr[name],name)
    oldnew=visible_mass_weights(old,c,n,u,a,source_boxes,k);widenew=visible_mass_weights(wide,c,n,u,a,source_boxes,k)
    states=[dict(scores=np.asarray(source_scores).astype(float),query=int(np.asarray(source_scores).argmax())),oldstate,
            legacy_operator(source_scores,ww,wide,e_all,relation_all),
            legacy_operator(source_scores,oldnew['weights'],old,e_all,relation_all),
            legacy_operator(source_scores,widenew['weights'],wide,e_all,relation_all)]
    qs=np.array([state['query'] for state in states],np.int64)
    return dict(arm_names=ARM_NAMES.copy(),scores=np.stack([state['scores'] for state in states]),queries=qs,
                selected_original_boxes=np.asarray(source_boxes)[qs],states=states,entry=entry,
                old_weights=ow,wide_old_weights=ww,oldentry_new=oldnew,wideentry_new=widenew,
                old_endpoint_bitexact=True,k=k,no_parameter_or_box_change=True)
