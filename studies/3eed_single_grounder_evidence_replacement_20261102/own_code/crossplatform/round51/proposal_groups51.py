"""Pure R51 proposal-support bags for conditional TTA eligibility only.

No G carrier/color/reference/readout is changed here. A bag is the complete
real point support of an ORIGINAL candidate, possibly several components.
Groups are proposal-support equivalences, not semantic instance identity.
"""
import numpy as np

MINIMUM_POINTS=6
MINIMUM_COMPONENTS=1
MINIMUM_JACCARD=.70
MINIMUM_EACH_COVERAGE=.75
MAXIMUM_PAIR_SHARED_FRACTION=.25


def _bag(value):
    a=np.asarray(value)
    if a.ndim!=1 or a.dtype.kind not in 'iu' or (a<0).any():
        raise ValueError('Unique original raw-point row IDs required')
    if len(np.unique(a))!=len(a):
        raise ValueError('Repeated raw IDs must not multiply support')
    return np.sort(a.astype(np.int64,copy=False))


def _metrics(box_a,aa,box_b,bb,intersection=None):
    a=np.asarray(box_a);b=np.asarray(box_b)
    if a.shape!=(6,) or b.shape!=(6,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Finite original center/size6 required')
    size_ok=bool((a[3:]>0).all() and (b[3:]>0).all())
    difference=np.abs(a[:3].astype(np.float64)-b[:3].astype(np.float64))
    centers=bool(size_ok and (difference<=a[3:].astype(np.float64)/2).all() and
                 (difference<=b[3:].astype(np.float64)/2).all())
    n=len(np.intersect1d(aa,bb,assume_unique=True)) if intersection is None else intersection
    ca=n/len(aa) if len(aa) else 0.;cb=n/len(bb) if len(bb) else 0.
    j=n/(len(aa)+len(bb)-n) if len(aa)+len(bb)-n else 0.
    compatible=bool(len(aa)>=MINIMUM_POINTS and len(bb)>=MINIMUM_POINTS and centers and
                    j>=MINIMUM_JACCARD and ca>=MINIMUM_EACH_COVERAGE and cb>=MINIMUM_EACH_COVERAGE)
    return {'intersection':n,'a_points':len(aa),'b_points':len(bb),'jaccard':float(j),
            'a_coverage':float(ca),'b_coverage':float(cb),'mutual_centers':centers,'compatible':compatible}


def bag_metrics(box_a,ids_a,box_b,ids_b):
    """No IoU: measured point overlap + exact mutual ORIGINAL-center veto."""
    return _metrics(box_a,_bag(ids_a),box_b,_bag(ids_b))


def compatible_bags(box_a,ids_a,box_b,ids_b):
    return bag_metrics(box_a,ids_a,box_b,ids_b)['compatible']


def _csr(point_offsets,point_indices,Q):
    off=np.asarray(point_offsets);flat=np.asarray(point_indices)
    if off.shape!=(Q+1,) or off.dtype.kind not in 'iu' or off[0]!=0 or off[-1]!=len(flat) or (np.diff(off)<0).any():
        raise ValueError('Valid complete query-support CSR required')
    bags=[_bag(flat[off[q]:off[q+1]]) for q in range(Q)]
    return off.astype(np.int64,copy=False),flat.astype(np.int64,copy=False),bags


def build_proposal_groups(*,source_boxes,point_offsets,point_indices,component_counts):
    """Return JSON info + NPZ arrays. No source scores/class/GT accepted.

    Query order is the original256. Canonical groups retain all original member
    bags; no union replaces them. Multiple complete-link matches leave qid=-1.
    """
    boxes=np.asarray(source_boxes);cc=np.asarray(component_counts)
    if boxes.shape!=(256,6) or not np.isfinite(boxes).all() or cc.shape!=(256,) or cc.dtype.kind not in 'iu' or (cc<0).any():
        raise ValueError('Original256 source boxes and actual component counts required')
    offsets,flat,bags=_csr(point_offsets,point_indices,256)
    support=np.array([len(a) for a in bags],np.int64)
    point_sets=[frozenset(map(int,a)) for a in bags]
    eligible=(support>=MINIMUM_POINTS)&(cc>=MINIMUM_COMPONENTS)&(boxes[:,3:]>0).all(1)
    compatible=np.zeros((256,256),bool);mutual=np.zeros((256,256),bool)
    intersections=np.zeros((256,256),np.int64)
    for a in range(256):
        for b in range(a,256):
            m=_metrics(boxes[a],bags[a],boxes[b],bags[b],len(point_sets[a]&point_sets[b]))
            intersections[a,b]=intersections[b,a]=m['intersection']
            mutual[a,b]=mutual[b,a]=m['mutual_centers']
            compatible[a,b]=compatible[b,a]=m['compatible'] and eligible[a] and eligible[b]
    # Point-count/min-raw-ID/original-query tie ordering; source scores absent.
    order=sorted(np.flatnonzero(eligible),key=lambda q:(-int(support[q]),int(bags[q][0]),int(q)))
    groups=[];qid=np.full(256,-1,np.int64);matches=[[] for _ in range(256)]
    status=['UNKNOWN_insufficient_real_support_or_components' for _ in range(256)]
    for q in range(256):
        if not (boxes[q,3:]>0).all():status[q]='UNKNOWN_nonpositive_original_size'
        elif support[q]<MINIMUM_POINTS:status[q]='UNKNOWN_fewer_than_six_real_points'
        elif cc[q]<MINIMUM_COMPONENTS:status[q]='UNKNOWN_no_original_local_component'
    for q in order:
        candidates=[g['group_id'] for g in groups if all(compatible[q,m] for m in g['members'])]
        matches[q]=list(map(int,candidates))
        if len(candidates)>1:
            status[q]='UNKNOWN_multiple_complete_link_proposal_groups'
            continue
        if not candidates:
            g=len(groups)
            groups.append({'group_id':g,'members':[int(q)],'representative_query':int(q)})
        else:
            g=candidates[0];groups[g]['members'].append(int(q))
        qid[q]=g;status[q]='UNIQUE_PROPOSAL_SUPPORT_EQUIVALENCE_not_semantic_identity'
    # Global sharing is descriptive only. Duplicate proposals are alternatives,
    # not extra physical objects or negative evidence against this bag.
    raw_membership={};group_membership={}
    for q in np.flatnonzero(eligible):
        for point in bags[q]:
            key=int(point);raw_membership[key]=raw_membership.get(key,0)+1
            if qid[q]>=0:group_membership.setdefault(key,set()).add(int(qid[q]))
    global_alt=np.zeros(256);cross_group=np.zeros(256)
    for q,a in enumerate(bags):
        if len(a):
            global_alt[q]=np.mean([raw_membership.get(int(v),0)>int(eligible[q]) for v in a])
            own=int(qid[q]);cross_group[q]=np.mean([bool(group_membership.get(int(v),set())-{own}) for v in a])
    arrays={'source_boxes':boxes.copy(),'support_offsets':offsets.copy(),'support_indices':flat.copy(),
            'component_counts':cc.copy(),'query_group_ids':qid,'query_support_counts':support,
            'pair_intersection_counts':intersections,'pair_compatible':compatible,'pair_mutual_centers':mutual,
            'query_global_alternative_shared_fraction':global_alt,
            'query_cross_group_shared_fraction':cross_group}
    info={'schema':'R51_WHOLE_PROPOSAL_SUPPORT_EQUIVALENCE_V1','groups':groups,
          'query_group_ids':qid.tolist(),'query_status':status,'compatible_group_ids_at_insertion':matches,
          'construction_order':list(map(int,order)),
          'rules':{'minimum_real_points':MINIMUM_POINTS,'minimum_original_components':MINIMUM_COMPONENTS,
                   'jaccard_at_least':MINIMUM_JACCARD,'each_coverage_at_least':MINIMUM_EACH_COVERAGE,
                   'mutual_original_box_centers_required':True,'complete_link_all_members':True,
                   'local_pair_shared_at_most':MAXIMUM_PAIR_SHARED_FRACTION,
                   'dominance_gate':False,'global_shared_gate':False,'box_IoU_grouping':False},
          'semantic_identity_certified':False,'G_and_original_256_boxes_unchanged':True,
          'canonical_members_keep_original_bags_no_cross_group_union':True,
          'global_shared_fractions_diagnostic_only':True}
    return info,arrays


def pair_geometry(group_json,positive_group,negative_group,point_offsets,point_indices):
    """Max over ALL cross-group member pairs, both directed overlap ratios.

    May also be used for B's query groups whose canonical IDs have gaps.
    No source probabilities, labels, connected-component dominance or global
    alternative sharing enters this local P/N test.
    """
    groups={int(g['group_id']):list(map(int,g['members'])) for g in group_json['groups']}
    if len(groups)!=len(group_json['groups']) or any(len(v)!=len(set(v)) for v in groups.values()):
        raise ValueError('Duplicate group IDs or member queries')
    Q=len(point_offsets)-1;_,_,bags=_csr(point_offsets,point_indices,Q)
    p=int(positive_group);n=int(negative_group);reject=[]
    if p<0 or n<0 or p not in groups or n not in groups:
        reject.append('UNKNOWN_missing_positive_or_negative_proposal_group')
    elif p==n:reject.append('SAME_duplicate_proposal_support_group')
    P=groups.get(p,[]) if p>=0 else [];N=groups.get(n,[]) if n>=0 else []
    if not P or not N:reject.append('UNKNOWN_empty_positive_or_negative_group')
    if any(q<0 or q>=Q or len(bags[q])<MINIMUM_POINTS for q in P+N):
        raise ValueError('Group member without valid original support')
    pmax=nmax=0.;witnessP=witnessN=None
    for a in P:
        for b in N:
            intersection=len(np.intersect1d(bags[a],bags[b],assume_unique=True))
            pa=intersection/len(bags[a]);nb=intersection/len(bags[b])
            if witnessP is None or pa>pmax:pmax=pa;witnessP=[a,b]
            if witnessN is None or nb>nmax:nmax=nb;witnessN=[a,b]
    if pmax>MAXIMUM_PAIR_SHARED_FRACTION or nmax>MAXIMUM_PAIR_SHARED_FRACTION:
        reject.append('UNKNOWN_local_P_N_point_support_overlap')
    return {'eligible':not reject,'positive_group':p,'negative_group':n,
            'positive_queries':P,'negative_queries':N,
            'max_positive_direction_overlap':float(pmax),'max_negative_direction_overlap':float(nmax),
            'max_directional_overlap':float(max(pmax,nmax)),
            'positive_overlap_witness_queries':witnessP,'negative_overlap_witness_queries':witnessN,
            'rejection_reasons':reject,'global_shared_used_for_acceptance':False,
            'candidate_support_preference_not_semantic_identity':True}
