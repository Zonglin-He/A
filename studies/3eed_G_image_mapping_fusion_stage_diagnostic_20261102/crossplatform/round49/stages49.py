"""Unexecuted pure-array draft: exact legacy stage replay, no I/O or model imports.

Root must freeze the arm set and authorize callers before real-data execution.
Mask ownership/geometry is evidence, never an independent identity annotation.
"""
import numpy as np

def mask_scene_lite(instances):
    """R15 image-only geometry using existing original-resolution mask arrays.

    Does not read RGB, raw point cloud, calibration or task metadata. Preserve
    R15 dtype and operation order so cached-bank comparisons can demand exact.
    """
    H,W=map(int,instances['original_hw'])
    masks=np.asarray(instances['masks_bool']);boxes=np.asarray(instances['boxes_xyxy'],np.float64)
    assert masks.dtype==bool and masks.ndim==3 and masks.shape[1:]==(H,W)
    assert boxes.shape==(len(masks),4) and np.isfinite(boxes).all()
    objects=[]
    for j,mask in enumerate(masks):
        yx=np.argwhere(mask);defined=bool(len(yx))
        centroid=(yx[:,::-1].mean(0)+.5)/np.array([W,H]) if defined else None
        border=bool(defined and (mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any()))
        objects.append({'mask_centroid_norm2':centroid,'touches_image_border':border,
                        'detector_box_xyxy':boxes[j].copy(),'mask_geometry_defined':defined})
    return {'image_size_wh':[W,H],'objects':objects}

def rect_iou_detect20(a,b):
    # Exact historical rank23 `from detect import rect_iou`, not the different
    # rect_iou denominator guards in round13/association.py.
    a=np.asarray(a,float);b=np.asarray(b,float);lo=np.maximum(a[:2],b[:2]);hi=np.minimum(a[2:],b[2:]);inter=np.maximum(hi-lo,0).prod()
    return float(inter/max((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-inter,1e-12))

def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-30,30)))

def relation_value(g,obj,evidence,W,H):
    """Literal rank23/rankM24 operations; unknown relations remain zero."""
    if obj['mask_centroid_norm2'] is None:raise ValueError('Undefined mask centroid; do not invent a geometry fallback')
    x=float(obj['mask_centroid_norm2'][0]);global_values=[];reference_values=[]
    if obj['touches_image_border']:return 0.,0.
    for rel in g['global_relations']:
        p=rel['predicate']
        if p in ['GLOBAL_LEFT','LEFT_OF_CENTER']:global_values.append(float(-np.tanh((x-.5)/.15)))
        elif p in ['GLOBAL_RIGHT','RIGHT_OF_CENTER']:global_values.append(float(np.tanh((x-.5)/.15)))
        elif p=='GLOBAL_CENTER':global_values.append(float(1-4*abs(x-.5)))
    for rel in g['reference_relations']:
        if rel['frame']!='camera_image':continue
        anchors=evidence['references'].get(rel['subject'],[]);weighted=0.;total=0.
        for a in anchors:
            if rect_iou_detect20(a['box_PRIVATE'],obj['detector_box_xyxy'])>=.3:continue
            box=np.asarray(a['box_PRIVATE'])/np.array([W,H,W,H]);center=(box[0]+box[2])/2;uncertainty=.1*(box[2]-box[0]);direction=1 if rel['predicate']=='RIGHT_OF' else -1
            weight=float(sigmoid(a['full']['log_odds_yes_no'])*sigmoid(a['visual_evidence'])*a['detector_score'])
            score=float(np.tanh(direction*(center-x)/(.15+uncertainty)));weighted+=weight*score;total+=weight
        if total:reference_values.append(weighted/(1+total))
    return (float(np.mean(global_values)) if global_values else 0.,float(np.mean(reference_values)) if reference_values else 0.)

def anchored_vision(s,w,ev):
    """Original 1/2 scale, clip and unmapped handling; no score calibration."""
    q0=int(s.argmax());mapped=w.sum(1)>0;vv=w@ev
    anchor=vv[q0] if mapped[q0] else float(np.median(ev))
    unclipped=np.zeros(256);unclipped[mapped]=(vv[mapped]-anchor)/2
    vision=np.zeros(256);vision[mapped]=np.clip(unclipped[mapped],-2,2)
    return dict(mapped=mapped,vv=vv,anchor_v=anchor,vision_unclipped=unclipped,vision=vision)

def replay(s,weights,pool,evidence,text,scene_lite):
    """Reconstruct full original bank before any counterfactual is evaluated."""
    s=np.asarray(s);assert s.shape==(256,) and np.isfinite(s).all()
    w=np.asarray(weights);assert w.shape==(256,len(pool)) and np.isfinite(w).all()
    assert set(evidence['targets_PRIVATE'])=={str(m) for m in pool}
    q0=int(s.argmax());v=np.zeros(256);p=np.zeros(256);r=np.zeros(256)
    if not pool:
        return dict(source_q0=q0,mapped=np.zeros(256,bool),image_e=np.zeros(0),image_full=np.zeros(0),image_gray=np.zeros(0),
                    W=w,mask_relations=np.zeros((0,2)),vv=np.zeros(256),rr=np.zeros((256,2)),anchor_v=None,anchor_r=None,
                    vision_unclipped=v.copy(),position_unclipped=p.copy(),relation_unclipped=r.copy(),vision=v,position=p,relation=r,F=s.astype(float))
    ev=np.array([np.mean([x['visual_evidence'] for x in evidence['targets_PRIVATE'][str(m)]]) for m in pool])
    ef=np.array([np.mean([x['full']['log_odds_yes_no'] for x in evidence['targets_PRIVATE'][str(m)]]) for m in pool])
    eg=np.array([np.mean([x['gray']['log_odds_yes_no'] for x in evidence['targets_PRIVATE'][str(m)]]) for m in pool])
    W,H=scene_lite['image_size_wh']
    rv=np.array([relation_value(text,scene_lite['objects'][m],evidence,W,H) for m in pool])
    stage=anchored_vision(s,w,ev);mapped=stage['mapped'];rr=w@rv
    ar=rr[q0] if mapped[q0] else np.median(rv,axis=0)
    pu=np.zeros(256);ru=np.zeros(256);pu[mapped]=rr[mapped,0]-ar[0];ru[mapped]=rr[mapped,1]-ar[1]
    p[mapped]=np.clip(pu[mapped],-2,2);r[mapped]=np.clip(ru[mapped],-2,2)
    v=stage['vision']
    return dict(stage,source_q0=q0,W=w,image_e=ev,image_full=ef,image_gray=eg,mask_relations=rv,rr=rr,anchor_r=ar,
                position_unclipped=pu,relation_unclipped=ru,position=p,relation=r,F=s.astype(float)+2*v+2*p+2*r)

def assert_bank_exact(stage,bank):
    for key in ('vision','position','relation'):
        if not np.array_equal(stage[key],bank[key]):
            raise AssertionError(('measurement_bridge_not_exact',key,float(np.max(abs(stage[key]-bank[key])))))

def proposed_readouts(s,stage,joint_e=None):
    """Draft arm set, not execution authorization or a selected improvement.

    Return all vectors with explicit mapped-only tie convention separately.
    No root-approved driver exists until the exact arm set is locked.
    """
    s=np.asarray(s).astype(float);v=stage['vision'];p=stage['position'];r=stage['relation'];m=stage['mapped']
    scores={'source':s.copy(),'legacy_full':s+2*v+2*p+2*r,'legacy_vision':s+2*v,
            'legacy_vision_position':s+2*v+2*p,'legacy_vision_unclipped':s+2*stage['vision_unclipped']}
    queries={name:int(a.argmax()) for name,a in scores.items()}
    mapped=np.flatnonzero(m)
    if len(mapped):
        maxset=mapped[stage['vv'][mapped]==stage['vv'][mapped].max()]
        queries['legacy_mapped_cue_only']=int(maxset[s[maxset].argmax()])
    else:queries['legacy_mapped_cue_only']=int(s.argmax())
    for name,ev in [('legacy_full_yes_vision',stage['image_full']),('joint_fulltext_matched_legacy_vision',joint_e)]:
        if ev is None:continue
        cue=anchored_vision(s,stage['W'],np.asarray(ev))['vision'] if len(ev) else np.zeros(256)
        scores[name]=s+2*cue;queries[name]=int(scores[name].argmax())
    # The cue-only query is not argmax of an invented finite score vector. A
    # future writer must persist its exact override/tie-set contract explicitly.
    return scores,queries

ARM_NAMES=[f'legacy_V{v}P{p}R{r}' for v in (0,1) for p in (0,1) for r in (0,1)]+[
    'legacy_rawvv_mapped_cue_only','legacy_postclipV_mapped_cue_only',
    'legacy_fullG_mapped_only','legacy_fullG_V_unclipped',
    'legacy_fullG_no_gray','joint_e_fullG_same_operator',
    'joint_literal_source_plus_rawcue','joint_rawcue_mapped_only']

def _domain_choice(score,source,domain,source_ties):
    eligible=np.flatnonzero(domain)
    if not len(eligible):return int(source.argmax()),np.zeros(0,np.int64),True
    ties=eligible[score[eligible]==score[eligible].max()]
    q=int(ties[source[ties].argmax()]) if source_ties else int(ties[0])
    return q,ties,False

def readouts16(s,stage,joint_e):
    """Root-fixed 16 arms; immutable native source and original cached cues.

    Four domain-restricted controls expose their candidate domain and exact
    maxima; cue-only plateaus use source then lowest-index exact tie break.
    Empty mapped domains return source q0 explicitly, never an arbitrary zero.
    """
    s=np.asarray(s).astype(float);v=stage['vision'];p=stage['position'];r=stage['relation'];mapped=stage['mapped']
    joint_e=np.asarray(joint_e);assert joint_e.shape==stage['image_e'].shape
    if len(joint_e):
        joint=anchored_vision(s,stage['W'],joint_e)
        raw_full=anchored_vision(s,stage['W'],stage['image_full'])
    else:
        joint={'vision':np.zeros(256),'vv':np.zeros(256)};raw_full={'vision':np.zeros(256)}
    vectors=[s+2*a*v+2*b*p+2*c*r for a in (0,1) for b in (0,1) for c in (0,1)]
    vectors += [stage['vv'].copy(),v.copy(),stage['F'].copy(),s+2*stage['vision_unclipped']+2*p+2*r,
                s+2*raw_full['vision']+2*p+2*r,s+2*joint['vision']+2*p+2*r,
                s+2*joint['vv'],joint['vv'].copy()]
    assert len(vectors)==len(ARM_NAMES)==16
    domains=np.ones((16,256),bool);domains[[8,9,10,15]]=mapped
    queries=[];ties=[];fallback=[]
    for i,score in enumerate(vectors):
        q,t,fb=_domain_choice(score,s,domains[i],source_ties=i in (8,9,15))
        queries.append(q);ties.append(t.tolist());fallback.append(fb)
    return dict(names=ARM_NAMES.copy(),scores=np.stack(vectors),domains=domains,
                queries=np.array(queries,np.int64),max_tie_sets=ties,empty_mapped_source_fallback=fallback,
                joint_rawcue=joint['vv'],joint_legacy_operator_vision=joint['vision'],
                full_yes_legacy_operator_vision=raw_full['vision'])
