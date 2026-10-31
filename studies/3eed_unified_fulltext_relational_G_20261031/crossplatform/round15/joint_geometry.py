"""R15 scene-wide candidate mapping then inequality geometry. Pure CPU arrays.

A unique supported graph hypothesis and reciprocal geometric correspondence are
necessary gates, not ground-truth identity. q0 is fallback only. No model/GT/I/O.
"""
from dataclasses import asdict,dataclass,field
from pathlib import Path
import hashlib,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'round13'))
from geometry_observation import ObservationConfig,generate_observation_targets
from scene_observation import array_sha

ARM_NAMES=('source','association_only','association_geometry_full','association_geometry_noRGB','association_geometry_half')

@dataclass(frozen=True)
class JointConfig:
    minimum_exclusive_support:int=8
    positive_distance_tolerance_m:float=.1
    minimum_positive_support_fraction:float=.90
    foreign_minimum_points:int=3
    foreign_fraction_max:float=.10
    dedup_minimum_AABB_IoU:float=.5
    dedup_minimum_visible_Jaccard:float=.5
    geometry:ObservationConfig=field(default_factory=ObservationConfig)
DEFAULT_CONFIG=JointConfig()

def config_sha256(config=DEFAULT_CONFIG):return hashlib.sha256(json.dumps(asdict(config),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def _aabb_iou(a,b):
    if (a[3:]<=0).any() or (b[3:]<=0).any():return 0.
    lo=np.maximum(a[:3]-a[3:]/2,b[:3]-b[3:]/2);hi=np.minimum(a[:3]+a[3:]/2,b[:3]+b[3:]/2)
    inter=float(np.maximum(hi-lo,0).prod());union=float(a[3:].prod()+b[3:].prod()-inter)
    return inter/union if union>0 else 0.

def _inside_tolerance(box,points,tolerance):
    if (box[3:]<=0).any():return np.zeros(len(points),bool)
    outside=np.maximum(np.abs(points-box[:3])-box[3:]/2,0.)
    return np.linalg.norm(outside,axis=1)<=tolerance

def _pool(boxes_raw6,text_scores):
    boxes=np.asarray(boxes_raw6,np.float32);scores=np.asarray(text_scores)
    if boxes.ndim!=2 or boxes.shape[1]!=6 or not len(boxes) or not np.isfinite(boxes).all():raise ValueError('Finite original source Qx6 pool required')
    if scores.shape!=(len(boxes),) or not np.isfinite(scores).all():raise ValueError('One finite original source text score per query required')
    return boxes,scores

def _dedup(boxes,support,config):
    """Complete-link geometric compatibility. Source scores/q0 not arguments."""
    Q=len(boxes);pair=np.eye(Q,dtype=bool);counts={'unordered_pairs':Q*(Q-1)//2,'IoU_or_exact_pairs_tested':0,'merge_compatible_pairs':0,'exact_empty_support_pairs':0}
    # Geometry-hash order has no model score/target-mask priority. Hash is only
    # deterministic ordering, never a geometric or semantic confidence.
    order=sorted(range(Q),key=lambda q:(array_sha(boxes[q].astype('<f4')),q))
    for ii,q in enumerate(order):
        for r in order[ii+1:]:
            exact=np.array_equal(boxes[q],boxes[r]);iou=_aabb_iou(boxes[q].astype(float),boxes[r].astype(float))
            if not exact and iou<config.dedup_minimum_AABB_IoU:continue
            union=int(np.count_nonzero(support[q]|support[r]));inter=int(np.count_nonzero(support[q]&support[r]));jac=inter/union if union else None
            okay=bool(exact or (iou>=config.dedup_minimum_AABB_IoU and jac is not None and jac>=config.dedup_minimum_visible_Jaccard))
            pair[q,r]=pair[r,q]=okay
            counts['IoU_or_exact_pairs_tested']+=1;counts['merge_compatible_pairs']+=int(okay);counts['exact_empty_support_pairs']+=int(exact and union==0)
    groups=[];membership=np.full(Q,-1,np.int64)
    for q in order:
        destination=next((i for i,members in enumerate(groups) if all(pair[q,r] for r in members)),None)
        if destination is None:destination=len(groups);groups.append([])
        groups[destination].append(q);membership[q]=destination
    return groups,membership,{'query_geometry_hash_order':order,'pair_compatibility_counts':counts,
        'complete_link':True,'q0_or_source_score_used':False,'empty_support_only_exact_coordinate_duplicate_merge':True,
        'grouping_not_verified_instance_partition':True}

def _mapping_rows(feasible,groups,membership,objects,scores):
    Q,M=feasible.shape
    group_masks=[sorted({int(m) for q in members for m in np.flatnonzero(feasible[q])}) for members in groups]
    rows=[]
    for m,obj in enumerate(objects):
        qs=np.flatnonzero(feasible[:,m]);gids=sorted(set(int(membership[q]) for q in qs))
        row={'mask_id':obj['mask_id'],'feasible_query_indices':qs.tolist(),'feasible_group_ids':gids,
             'mapped_group_id':None,'canonical_query_index':None,'status':'NA_no_compatible_group'}
        if len(gids)>1:row['status']='NA_multiple_compatible_groups'
        elif len(gids)==1:
            gid=gids[0]
            if len(group_masks[gid])!=1:row['status']='NA_group_compatible_with_multiple_independent_masks'
            else:
                # Source text prior only ranks original members of the one
                # reciprocally unique group. No inter-entity score competition.
                q=sorted(qs.tolist(),key=lambda j:(-float(scores[j]),j))[0]
                row.update(status='reciprocal_unique_geometric_correspondence_hypothesis',mapped_group_id=gid,canonical_query_index=int(q))
        rows.append(row)
    return rows,group_masks

def map_scene_to_candidate_groups(*,scene,boxes_raw6,text_scores,config=DEFAULT_CONFIG):
    boxes,scores=_pool(boxes_raw6,text_scores);cfg=config
    if not isinstance(scene,dict) or scene.get('schema')!='R15_whole_scene_observation_v1':raise ValueError('R15 independent scene observation required')
    points=np.asarray(scene['visible_points_model'],np.float64);owners=np.asarray(scene['point_owner']);objects=scene['objects'];M=len(objects);Q=len(boxes)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all() or owners.shape!=(len(points),) or not np.issubdtype(owners.dtype,np.integer) or ((owners< -2)|(owners>=M)).any():raise ValueError('Scene ownership/point shape mismatch')
    if len({o['mask_id'] for o in objects})!=M:raise ValueError('Unique independent mask IDs required')
    for j,obj in enumerate(objects):
        ids=np.flatnonzero(owners==j)
        if obj['n_exclusive']!=len(ids) or not np.array_equal(obj['points_model'],scene['visible_points_model'][ids]) or not np.array_equal(obj['physical_raw_ids'],scene['physical_raw_ids'][ids]):raise ValueError('Object support must match frozen whole-scene exclusive ownership')
    support=np.stack([_inside_tolerance(box.astype(float),points,cfg.positive_distance_tolerance_m) for box in boxes])
    groups,membership,dd=_dedup(boxes,support,cfg)
    counts=np.zeros((Q,M),np.int64)
    for q in range(Q):
        known=owners[support[q] & (owners>=0)]
        counts[q]=np.bincount(known,minlength=M) if M else np.zeros(0,np.int64)
    denom=counts.sum(1);n=np.array([o['n_exclusive'] for o in objects],np.int64)
    positive=counts/np.maximum(n,1)[None,:]
    positive_feasible=(positive>=cfg.minimum_positive_support_fraction)&(n[None,:]>=cfg.minimum_exclusive_support)&(boxes[:,3:]>0).all(1)[:,None]
    foreign=denom[:,None]-counts;foreign_fraction=foreign/np.maximum(denom,1)[:,None]
    contamination=(foreign>=cfg.foreign_minimum_points)&(foreign_fraction>cfg.foreign_fraction_max)
    feasible=positive_feasible & ~contamination
    rows,group_masks=_mapping_rows(feasible,groups,membership,objects,scores)
    noforeign_rows,noforeign_group_masks=_mapping_rows(positive_feasible,groups,membership,objects,scores)
    for j,row in enumerate(rows):
        row['exclusive_support_n']=int(n[j]);row['other_mask_exclusive_points_total']=int(n.sum()-n[j])
        row['other_masks_with_at_least_three_exclusive_points']=int(np.sum(np.delete(n,j)>=cfg.foreign_minimum_points))
        row['foreign_evidence_available_somewhere_in_scene']=bool(n.sum()-n[j]>=cfg.foreign_minimum_points)
        cq=row['canonical_query_index']
        row['canonical_positive_fraction']=float(positive[cq,j]) if cq is not None else None
        row['canonical_foreign_count']=int(foreign[cq,j]) if cq is not None else None
        row['canonical_foreign_fraction']=float(foreign_fraction[cq,j]) if cq is not None else None
        row['without_foreign_gate_status']=noforeign_rows[j]['status']
        row['without_foreign_gate_query_index']=noforeign_rows[j]['canonical_query_index']
        row['foreign_gate_changed_mapping']=bool((row['status'],row['canonical_query_index'])!=(noforeign_rows[j]['status'],noforeign_rows[j]['canonical_query_index']))
    return {'schema':'R15_reciprocal_scene_mapping_v1','groups':[{'group_id':i,'query_indices':g,'compatible_mask_ids':[objects[j]['mask_id'] for j in group_masks[i]]} for i,g in enumerate(groups)],
        'membership':membership,'mask_mappings':rows,'positive_feasible':positive_feasible,'feasible':feasible,
        'positive_fraction':positive,'inside_exclusive_counts':counts,'foreign_exclusive_counts':foreign,
        'foreign_fraction':foreign_fraction,'pollution_rejected':contamination,
        'candidate_unassigned_contained_n':np.sum(support & (owners[None,:]==-1),axis=1),
        'candidate_ambiguous_contained_n':np.sum(support & (owners[None,:]==-2),axis=1),
        'candidate_visible_support_sha256':np.asarray([array_sha(s) for s in support]),
        'config':asdict(cfg),'config_sha256':config_sha256(cfg),
        'diagnostics':{'source_query_count':Q,'independent_mask_count':M,'dedup_group_count':len(groups),'dedup':dd,
            'minimum_positive_support_required':cfg.minimum_exclusive_support,
            'n_positive_edges':int(positive_feasible.sum()),'n_foreign_rejected_positive_edges':int((positive_feasible & contamination).sum()),
            'n_effective_edges':int(feasible.sum()),'n_reciprocal_unique_masks':sum(r['canonical_query_index'] is not None for r in rows),
            'n_mapping_changed_by_foreign_gate':sum(r['foreign_gate_changed_mapping'] for r in rows),
            'other_mask_points_are_predicted_correspondence_not_GT':True,'whole_cloud_not_q0_ROI':True,
            'q0_compatibility_priority':False,'scores_only_within_reciprocally_confirmed_group':True,
            'single_mask_or_missing_foreign_points_cannot_exclude_giant_box':True,
            'unassigned_or_multimask_points_are_not_negative_evidence':True}}

def generate_joint_targets(*,scene,boxes_raw6,text_scores,q0,graph_match,semantic_branch='graph',config=DEFAULT_CONFIG):
    """Association/readout and geometric increment are separate direct arms.

    graph_match is label-free semantic agent output; only unique_supported plus
    complete search can propose a target. Agreement is not certified identity.
    """
    boxes,scores=_pool(boxes_raw6,text_scores)
    if semantic_branch not in ('graph','proposal_control'):raise ValueError('Explicit graph or proposal_control branch required')
    if not isinstance(q0,(int,np.integer)) or isinstance(q0,(bool,np.bool_)) or not 0<=int(q0)<len(boxes):raise ValueError('Valid fresh source q0 required')
    if not isinstance(graph_match,dict) or not {'status','selected_target_mask_id','search_complete'}<=set(graph_match):raise ValueError('Explicit semantic graph result required')
    if not isinstance(graph_match['search_complete'],(bool,np.bool_)):raise ValueError('Explicit semantic search completeness required')
    result={'arm_names':np.asarray(ARM_NAMES),'arm_query_index':np.full(5,int(q0),np.int64),
        'arm_boxes_raw6':np.tile(boxes[q0],(5,1)),'arm_status':['source']+['source_fallback']*4,
        'q_source':int(q0),'q_association':int(q0),'selected_target_mask_id':None,
        'semantic_proposed_target_mask_id':graph_match.get('selected_target_mask_id'),
        'mapping':None,'geometry':None,'association_accepted':False,
        'diagnostics':{'config':asdict(config),'config_sha256':config_sha256(config),'q0_used_for_fallback_only':True,
            'direct_no_parameter_update':True,'semantic_and_geometry_uniqueness_not_identity_GT':True,
            'semantic_branch':semantic_branch,'semantic_original_status':graph_match['status'],
            'semantic_relation_mode':graph_match.get('relation_mode'),
            'semantic_relation_checks_enabled':graph_match.get('relation_check'),
            'semantic_literal_relations_satisfied':graph_match.get('literal_relations_satisfied'),
            'full_noRGB_share_independent_mask_support_and_gate':True,
            'geometry_size_source_or_expand_prior_not_amodal_observation':True}}
    def fail(reason):
        result['arm_status'][1:]=['source_fallback:'+reason]*4;result['diagnostics']['reason']=reason;return result
    expected_status={'graph':'unique_supported','proposal_control':'proposal_control_unique'}[semantic_branch]
    if graph_match['status']!=expected_status or not graph_match['search_complete']:
        return fail('semantic_unknown_ambiguous_incomplete_or_unsupported')
    mid=graph_match['selected_target_mask_id'];obj=next((o for o in scene['objects'] if o['mask_id']==mid),None)
    if obj is None:return fail('semantic_target_mask_not_in_independent_scene')
    mapping=map_scene_to_candidate_groups(scene=scene,boxes_raw6=boxes,text_scores=scores,config=config);result['mapping']=mapping
    row=next(r for r in mapping['mask_mappings'] if r['mask_id']==mid)
    if row['canonical_query_index'] is None:return fail(row['status'])
    q=int(row['canonical_query_index']);anchor=boxes[q].copy()
    result.update(q_association=q,selected_target_mask_id=mid,association_accepted=True)
    result['arm_query_index'][1:]=q;result['arm_boxes_raw6'][1:]=anchor
    result['arm_status'][1:]=['mapped_original_candidate']*4
    geo=generate_observation_targets(points_support_model=obj['points_model'],physical_raw_ids=obj['physical_raw_ids'],
        mask_bool=obj['mask_bool'],camera_spec=scene['camera_spec'],source6=anchor,q0=q,
        segmentation_valid=True,config=config.geometry)
    result['geometry']=geo;result['arm_boxes_raw6'][2:]=geo['arm_boxes_raw6'][1:]
    result['arm_status'][2:]=geo['arm_status'][1:]
    result['diagnostics'].update(reason='semantic_proposal_and_reciprocal_geometric_correspondence_accepted',
        source_and_associated_query_same=bool(q==q0),selected_exclusive_support_n=obj['n_exclusive'],
        selected_support_sha256=obj['support_sha256'],selected_mask_sha256=obj['mask_sha256'],
        no_q0_ROI_support_cropping=True,identity_uncertainty_retained=True,
        R13_geometry_fixed_q0_means_associated_anchor_query_not_original_q_source=True,
        geometry_failure_fallback_is_associated_anchor_not_original_q0=True,
        geometry_never_reselects_query=True)
    assert np.all(result['arm_query_index'][1:]==q)
    assert np.isfinite(result['arm_boxes_raw6']).all()
    assert np.array_equal(result['arm_boxes_raw6'][0],boxes[q0])
    assert np.array_equal(result['arm_boxes_raw6'][1],boxes[q])
    assert np.array_equal(result['arm_boxes_raw6'][4],anchor+np.float32(.5)*(result['arm_boxes_raw6'][2]-anchor))
    return result
