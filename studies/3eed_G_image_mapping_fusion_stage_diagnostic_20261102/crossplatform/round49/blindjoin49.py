"""UNEXECUTED draft. Both visual phases must be sealed and root-authorized.

No task annotations, IoU, image pixels, raw XYZ or new models. Partial-subject
compatibility and complete-caption identity are deliberately distinct fields.
"""
from pathlib import Path
import json,hashlib,datetime,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'research/round49/experiment'
OUT=ROOT/'crossplatform/round49/blindjoin48_attempt001'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8<<20),b''):h.update(x)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def verify_directory(folder):
    folder=Path(folder);seal=read(folder/'SEALED.json')
    for name,h in seal['files'].items():
        p=Path(name);assert not p.is_absolute() and '..' not in p.parts and sha(folder/p)==h

def check():
    plan=read(EXP/'BLIND_JOIN_PLAN_LOCK.json');auth=read(EXP/'BLIND_JOIN_EXECUTION_AUTHORIZATION.json')
    assert auth['state']=='ROOT_AUTHORIZED_R49_POSTSEAL_TWO_PHASE_JOIN'
    assert auth['plan_sha256']==sha(EXP/'BLIND_JOIN_PLAN_LOCK.json')
    assert plan['n']==48 and plan['no_GT_or_new_model_or_adaptation'] is True
    for field in ('phase1_seal','phase2_seal','phase1_annotations','phase2_annotations','cohort_file','render_bindings','reviewer_allocation'):
        assert plan[field] in plan['inputs_sha256'],('missing_direct_input_pin',field)
    # This checks only immutable file digests; annotations are decoded later.
    for rel,h in plan['programs_sha256'].items():assert sha(ROOT/rel)==h
    for rel,h in plan['inputs_sha256'].items():assert sha(ROOT/rel)==h
    for phase in ('phase1','phase2'):
        assert auth[phase+'_seal_sha256']==sha(ROOT/plan[phase+'_seal'])
    verify_directory(ROOT/'crossplatform/round49/stage336_attempt001')
    verify_directory(ROOT/'crossplatform/round49/ownership336_attempt001')
    return plan

def image_top_status(top,positive,negative=None):
    if not top:return 'UNKNOWN_no_visual_evidence_pool'
    if not positive and negative is None:return 'UNKNOWN_no_resolved_fullcaption_identity'
    top=set(top);positive=set(positive)
    if top and top<=positive:return 'SUPPORTED'
    if negative is None:return 'DISJOINT_FROM_SINGLE_VISIBLE_TARGET' if top.isdisjoint(positive) else 'TIE_TARGET_OTHER'
    negative=set(negative)
    if top and top<=negative:return 'VISIBLY_INCOMPATIBLE_WITH_PARTIAL_SUBJECT'
    return 'UNKNOWN_OR_MIXED'

def point_support(counts,q,target_indices,status,geometry_status,allvisible,unassigned,ambiguous):
    row=counts[q];foreground=int(row.sum());visible=int(allvisible[q]);nu=int(unassigned[q]);na=int(ambiguous[q])
    result={'allvisible':visible,'unassigned':nu,'ambiguous':na,'foreground_exclusive_points':foreground,
            'unassigned_over_allvisible':float(nu/visible) if visible else None,
            'ambiguous_over_allvisible':float(na/visible) if visible else None,
            'target_exclusive_points':None,'other_foreground_exclusive_points':None,
            'target_fraction':None,'target_share_of_exclusive_foreground':None,
            'target_exclusive_points_over_allvisible':None,
            'fraction_scope':'counts_of_visible_points_inside_fixed_source_box;not_full_object_or_complete_box_purity',
            'R28_fixed_support_proxy_not_gold':True}
    assert foreground+nu+na==visible
    if status!='SINGLE_VISIBLE_TARGET':return dict(result,status='UNKNOWN_fullcaption_'+status)
    if str(geometry_status[q]).startswith('UNKNOWN'):return dict(result,status=str(geometry_status[q]))
    target=int(row[target_indices].sum())
    fraction=target/max(foreground,1)
    label='target_supported' if foreground>=5 and fraction>=.8 else 'other_supported' if foreground>=5 and fraction<=.2 else 'UNKNOWN_sparse_or_mixed'
    result.update(status=label,target_exclusive_points=target,other_foreground_exclusive_points=foreground-target,
        target_fraction=float(fraction) if foreground else None,
        target_share_of_exclusive_foreground=float(fraction) if foreground else None,
        target_exclusive_points_over_allvisible=float(target/visible) if visible else None)
    return result

def main():
    plan=check();assert not OUT.exists();OUT.mkdir()
    # Only after both seals and explicit root grant are checked do labels enter.
    p1list=read(ROOT/plan['phase1_annotations']);p2list=read(ROOT/plan['phase2_annotations'])
    p1={x['review_alias']:x for x in p1list};p2={x['review_alias']:x for x in p2list}
    cohort=read(ROOT/plan['cohort_file']);bindings={x['review_alias']:x for x in read(ROOT/plan['render_bindings'])}
    allocation=read(ROOT/plan['reviewer_allocation'])
    assigned={alias:reviewer for reviewer,key in [('root','root_aliases'),('influence','influence_aliases')] for alias in allocation[key]}
    assert len(cohort)==len(p1list)==len(p2list)==len(p1)==len(p2)==48
    assert set(p1)==set(p2)=={x['review_alias'] for x in cohort}
    assert len(assigned)==48 and set(assigned)==set(p1) and not set(allocation['root_aliases'])&set(allocation['influence_aliases'])
    phase1sha=sha(ROOT/plan['phase1_annotations']);rows=[]
    for c in cohort:
        alias=c['review_alias'];rid=c['id_PRIVATE'];a=p1[alias];b=p2[alias];display=bindings[alias]
        assert b['phase1_annotation_sha256']==phase1sha
        assert b['subject_query_validity'] in ('AVAILABLE','UNKNOWN_EMPTY_OR_INVALID_TEXT')
        assert a['status'] in ('SINGLE_VISIBLE_TARGET','AMBIGUOUS_TARGET','NONE','UNKNOWN')
        source=ROOT/'crossplatform/round49/stage336_attempt001';ownership=ROOT/'crossplatform/round49/ownership336_attempt001'
        z=npz(source/'samples_PRIVATE'/(rid+'.npz'));o=npz(ownership/'samples_PRIVATE'/(rid+'.npz'))
        stagecase=read(source/'cases_PRIVATE'/(rid+'.json'))
        assert b['subject_query_cache_sha256']==sha(ROOT/stagecase['visual_text_source']['path_PRIVATE'])
        assert np.array_equal(z['source_boxes'],o['source_boxes'])
        candidates=display['numbered_candidates_PRIVATE'];ids={x['display_id'] for x in candidates}
        mapping={};hashduplicates=collections.Counter(o['mask_sha256'].tolist())
        for candidate in candidates:
            m=candidate['mask_index_PRIVATE'];assert str(o['mask_sha256'][m])==candidate['mask_sha256']
            mapping[candidate['display_id']]=m
        assert len(mapping)==len(candidates)==len(o['mask_sha256'])
        assert set(mapping.values())==set(range(len(o['mask_sha256']))), 'display_to_original_mask_not_a_complete_bijection'
        targetids=set(a['target_display_ids']);assert targetids<=ids
        if a['status']=='SINGLE_VISIBLE_TARGET':assert targetids,'SINGLE_VISIBLE_TARGET_requires_nonempty_target_ids'
        target_indices=np.array(sorted(mapping[x] for x in targetids),np.int64)
        if a['status'] in ('NONE','UNKNOWN'):assert not targetids
        compatible=set(b['subject_compatible_ids']);incompatible=set(b['subject_incompatible_ids']);unknown=set(b['subject_UNKNOWN_ids'])
        assert not (compatible&incompatible or compatible&unknown or incompatible&unknown)
        assert compatible|incompatible|unknown==ids
        if b['subject_query_validity']=='UNKNOWN_EMPTY_OR_INVALID_TEXT':
            assert not compatible and not incompatible and unknown==ids
        pool=z['pool'].tolist();inverse={m:key for key,m in mapping.items()}
        pool_display={inverse[m] for m in pool}
        compatible_pool_count=len(pool_display&compatible)
        incompatible_pool_count=len(pool_display&incompatible)
        unknown_pool_count=len(pool_display&unknown)
        legacytops=[inverse[pool[i]] for i in np.flatnonzero(z['image_e']==z['image_e'].max())] if pool else []
        rawfulltops=[inverse[pool[i]] for i in np.flatnonzero(z['image_full']==z['image_full'].max())] if pool else []
        jointtops=[inverse[pool[i]] for i in np.flatnonzero(z['joint_e']==z['joint_e'].max())] if pool else []
        complete_positive=targetids if a['status']=='SINGLE_VISIBLE_TARGET' else set()
        absent=bool(targetids and not any(m in pool for m in target_indices.tolist()))
        single_outside=bool(a['status']=='SINGLE_VISIBLE_TARGET' and absent)
        ambiguous_outside=bool(a['status']=='AMBIGUOUS_TARGET' and absent)
        coverage=('SINGLE_target_outside_pool' if single_outside else 'SINGLE_target_intersects_pool') if a['status']=='SINGLE_VISIBLE_TARGET' else (
            ('AMBIGUOUS_all_declared_alternatives_outside_pool' if ambiguous_outside else 'AMBIGUOUS_at_least_one_alternative_in_pool_or_no_declared_ids')
            if a['status']=='AMBIGUOUS_TARGET' else 'UNKNOWN_target_coverage_'+a['status'])
        legacy_full_status=image_top_status(legacytops,complete_positive)
        joint_full_status=image_top_status(jointtops,complete_positive)
        if single_outside and pool:
            legacy_full_status=joint_full_status='UNKNOWN_known_target_not_in_compared_pool'
        inventory={'status':'UNKNOWN_fullcaption_'+a['status'],'target_fullframe_exclusive_points':None,
            'all256_support_status_counts':None,'any_original_query_target_supported':None}
        allstatus=None
        if a['status']=='SINGLE_VISIBLE_TARGET':
            # Count each frame-visible point once. Summing support across the
            # overlapping 256 proposals would multiply the same point.
            owncase=read(ownership/'cases_PRIVATE'/(rid+'.json'))
            with np.load(ownership/'frames_PRIVATE'/(owncase['frame_key_PRIVATE']+'.npz'),allow_pickle=False) as framecache:
                owner=framecache['point_owner'].copy()
            unique_target_points=int(np.isin(owner,target_indices).sum())
            allstatus=[point_support(o['mask_counts'],q,target_indices,a['status'],o['geometry_status'],
                o['allvisible'],o['unassigned'],o['ambiguous'])['status'] for q in range(256)]
            supported=np.array([s=='target_supported' for s in allstatus],bool)
            targetcolumns=[j for j,m in enumerate(pool) if m in set(target_indices.tolist())]
            targetmass=z['weights'][:,targetcolumns].sum(1)
            inventory={'status':'DEFINED_SINGLE_TARGET_SUPPORT_INVENTORY','target_fullframe_exclusive_points':unique_target_points,
                'fullframe_exclusive_foreground_points':int((owner>=0).sum()),
                'fullframe_unassigned_points':int((owner==-1).sum()),'fullframe_ambiguous_points':int((owner==-2).sum()),
                'target_frame_point_count_is_unique_not_query_sum':True,
                'all256_support_status_counts':dict(collections.Counter(allstatus)),
                'target_supported_query_count':int(supported.sum()),'any_original_query_target_supported':bool(supported.any()),
                'target_supported_queries_with_any_W_mapping':int((supported&z['mapped']).sum()),
                'target_supported_queries_with_positive_target_W_mass':int((supported&(targetmass>0)).sum()),
                'target_supported_queries_without_target_W_mass':int((supported&(targetmass==0)).sum()),
                'fullframe_positive_target_points_does_not_certify_depth_completeness':True,
                'zero_target_W_mass_does_not_identify_which_coarse_or_geometric_gate_caused_it':True}
        detail={}
        for name,q in zip(z['arm_names'].tolist(),z['queries'].tolist()):
            supp=point_support(o['mask_counts'],q,target_indices,a['status'],o['geometry_status'],o['allvisible'],o['unassigned'],o['ambiguous'])
            targetcolumns=[j for j,m in enumerate(pool) if m in set(target_indices.tolist())]
            supp.update(query_PRIVATE=q,mapped=bool(z['mapped'][q]),W_target_mass=float(z['weights'][q,targetcolumns].sum()))
            detail[name]=supp
        tie_support={}
        for armindex,key in [(8,'rawcue'),(9,'postclipcue')]:
            top=stagecase['max_tie_sets_PRIVATE'][armindex];representative=int(z['queries'][armindex])
            assert not top or representative in top
            states=[allstatus[q] for q in top] if allstatus is not None else ['UNKNOWN_fullcaption_'+a['status'] for _ in top]
            representative_status=detail[z['arm_names'].tolist()[armindex]]['status']
            tie_support[key]={'maxset_size':len(top),'no_visual_maxset':not bool(top),
                'maxset_support_status_counts':dict(collections.Counter(states)),
                'contains_target_supported':any(s=='target_supported' for s in states) if allstatus is not None else None,
                'target_supported_members_count':sum(s=='target_supported' for s in states) if allstatus is not None else None,
                'representative_support_status':representative_status,
                'representative_is_in_maxset':representative in top,
                'representative_is_target_supported_member':bool(top and representative_status=='target_supported') if allstatus is not None else None,
                'maxset_mixed_support_statuses':len(set(states))>1,
                'source_fallback_due_empty_maxset':not bool(top),
                'representative_failure_does_not_certify_entire_W_maxset_wrong':True}
        rows.append({'review_alias':alias,'id_PRIVATE':rid,'platform':c['platform'],'phase1_status':a['status'],
            'assigned_reviewer_before_images':assigned[alias],
            'phase1_reviewer':a.get('reviewer',assigned[alias]),'phase2_reviewer':b.get('reviewer',assigned[alias]),
            'phase1_exposure_record':{k:v for k,v in a.items() if 'exposure' in k or k in ('historical_project_exposure','nonexpert_not_gold')},
            'phase2_exposure_record':{k:v for k,v in b.items() if 'exposure' in k or k in ('historical_project_exposure','nonexpert_not_gold')},
            'fullcaption_target_display_ids':sorted(targetids),'actual_partial_subject_compatible_ids':sorted(compatible),
            'actual_partial_subject_UNKNOWN_ids':sorted(unknown),'legacy_image_top_ids':legacytops,
            'legacy_raw_full_image_top_ids':rawfulltops,'joint_image_top_ids':jointtops,
            'legacy_visual_vs_actual_partial_subject':image_top_status(legacytops,compatible,incompatible),
            'legacy_raw_full_vs_actual_partial_subject':image_top_status(rawfulltops,compatible,incompatible),
            'legacy_gray_corrected_vs_actual_partial_subject':image_top_status(legacytops,compatible,incompatible),
            'raw_full_and_gray_corrected_topsets_equal':set(rawfulltops)==set(legacytops),
            'legacy_visual_vs_fullcaption_identity':legacy_full_status,
            'joint_visual_vs_fullcaption_identity':joint_full_status,
            'legacy_compared_pool_mask_count':len(pool),'has_legacy_visual_pool':bool(pool),
            'partial_compatible_candidates_in_compared_pool':compatible_pool_count,
            'partial_incompatible_candidates_in_compared_pool':incompatible_pool_count,
            'partial_UNKNOWN_candidates_in_compared_pool':unknown_pool_count,
            'partial_compatible_candidate_exists_in_compared_pool':bool(compatible_pool_count),
            'partial_compatibility_topset_interpretation':('UNKNOWN_no_visual_evidence_pool' if not pool else
                'ORDERING_CHECK_WITH_CONFIRMED_COMPATIBLE_ALTERNATIVE' if compatible_pool_count else
                'NO_CONFIRMED_COMPATIBLE_ALTERNATIVE_FORCED_ARGMAX_NOT_BINARY_RECOGNITION_ERROR'),
            'legacy_rawfull_all_mask_means_negative':bool((z['image_full']<0).all()) if pool else None,
            'legacy_rawfull_max_mask_mean_logodds':float(z['image_full'].max()) if pool else None,
            'legacy_rawfull_mean_definition':'mean_of_two_crop_YesNo_logodds_per_mask_then_max_over_original_pool',
            'gray_corrected_e_is_not_PYes':True,
            'rawfull_mean_negative_is_not_all_views_negative':True,
            'target_mask_not_in_legacy_pool':single_outside,
            'single_target_outside_legacy_pool':single_outside,'ambiguous_all_alternatives_outside_legacy_pool':ambiguous_outside,
            'legacy_pool_identity_coverage_status':coverage,
            'original256_target_support_inventory':inventory,
            'cue_maxset_support_diagnostics':tie_support,
            'duplicate_exact_pixelmask_hashes_n':sum(v>1 for v in hashduplicates.values()),
            'arms':detail,'nonexpert_blind_labels_not_gold':True,'no_taskGT_or_IoU':True,
            'legacy_partial_vs_fullcaption_disagreement_not_automatically_VLMerror':True})
    summary={'n':48,'by_platform':{},'reviewer_allocation_counts':dict(collections.Counter(r['assigned_reviewer_before_images'] for r in rows)),
             'two_assistant_nonexpert_reviews_not_gold':True,'reviewers_have_different_fixed_cohorts_not_a_reviewer_accuracy_comparison':True}
    for platform in ('drone','quad'):
        rr=[r for r in rows if r['platform']==platform];ss={'n':len(rr)}
        for k in ('phase1_status','legacy_visual_vs_actual_partial_subject','legacy_raw_full_vs_actual_partial_subject',
                  'legacy_gray_corrected_vs_actual_partial_subject','legacy_visual_vs_fullcaption_identity',
                  'joint_visual_vs_fullcaption_identity','legacy_pool_identity_coverage_status'):
            ss[k]=dict(collections.Counter(r[k] for r in rr))
        ss['target_mask_not_in_legacy_pool']=sum(r['target_mask_not_in_legacy_pool'] for r in rr)
        ss['ambiguous_all_alternatives_outside_legacy_pool']=sum(r['ambiguous_all_alternatives_outside_legacy_pool'] for r in rr)
        ss['raw_full_and_gray_corrected_topsets_differ']=sum(not r['raw_full_and_gray_corrected_topsets_equal'] for r in rr)
        ss['partial_compatibility_strata']={}
        for name,condition in [('no_visual_pool',lambda x:not x['has_legacy_visual_pool']),
            ('pool_with_confirmed_compatible',lambda x:x['has_legacy_visual_pool'] and x['partial_compatible_candidate_exists_in_compared_pool']),
            ('pool_without_confirmed_compatible',lambda x:x['has_legacy_visual_pool'] and not x['partial_compatible_candidate_exists_in_compared_pool'])]:
            sr=[x for x in rr if condition(x)]
            ss['partial_compatibility_strata'][name]={'n':len(sr),
                'rawfull_top_status_counts':dict(collections.Counter(x['legacy_raw_full_vs_actual_partial_subject'] for x in sr)),
                'gray_corrected_top_status_counts':dict(collections.Counter(x['legacy_gray_corrected_vs_actual_partial_subject'] for x in sr)),
                'all_mask_mean_rawfull_logodds_negative':sum(x['legacy_rawfull_all_mask_means_negative'] is True for x in sr),
                'without_compatible_does_not_establish_binary_visual_recognition_error':True}
        singles=[r['original256_target_support_inventory'] for r in rr if r['phase1_status']=='SINGLE_VISIBLE_TARGET']
        ss['single_target_support_inventory']={'n':len(singles),
            'target_fullframe_exclusive_point_count_zero':sum(x['target_fullframe_exclusive_points']==0 for x in singles),
            'target_fullframe_exclusive_point_count_1_to_4_descriptive':sum(1<=x['target_fullframe_exclusive_points']<=4 for x in singles),
            'no_original_query_target_supported':sum(not x['any_original_query_target_supported'] for x in singles),
            'target_support_exists_but_no_supported_query_has_target_W_mass':sum(x['any_original_query_target_supported'] and x['target_supported_queries_with_positive_target_W_mass']==0 for x in singles),
            'point_proxy_and_mapping_loss_diagnostics_not_independent_world_identity_gold':True}
        ss['point_support_by_arm']={arm:dict(collections.Counter(r['arms'][arm]['status'] for r in rr)) for arm in rows[0]['arms']}
        summary['by_platform'][platform]=ss
    dump(OUT/'ROWS_PRIVATE.json',rows);dump(OUT/'SUMMARY.json',summary)
    dump(OUT/'SEALED.json',{'state':'R49_TWO_PHASE_AND_POINT_SUPPORT_DESCRIPTIVE_JOIN_NO_GT',
        'files':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()},'plan_sha256':sha(EXP/'BLIND_JOIN_PLAN_LOCK.json'),
        'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'does_not_modify_labels_G_or_method':True})
    print('BLIND_JOIN49_SEALED',len(rows))

if __name__=='__main__':main()
