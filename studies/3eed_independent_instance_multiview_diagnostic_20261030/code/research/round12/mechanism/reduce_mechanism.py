"""R12 release-gated author-side mechanism census. No GT/media/P/features.
Decode only selected sealed prediction arrays and scalar/structural trace fields.
"""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json
import numpy as np
ARMS=('source','grabcut_direct','single_instance_direct','multiview_instance_direct','wrongview_instance_direct','single_instance_half','multiview_instance_half','multiview_maskmatch_only_direct')
KEYS=('sample_id','q_source','arm_names','arm_query_index','arm_status','arm_boxes_raw6','acceptance_mask6','actual_changed_axes','point_support_sha256')
LINEAGE_FILES=('crossplatform/round12/history/preflight_G_v1_before_perpoint_corroboration/multiview_feedback.py','crossplatform/round12/history/preflight_G_v2_SIFT_before_PnP_backend/multiview_feedback.py','crossplatform/round12/history/preflight_G_v2_SIFT_before_PnP_backend/FEEDBACK_IMPLEMENTATION_LOCK.json','crossplatform/round12/history/preflight_PnP_v1_before_empty_B_holdout/sensor_pnp.py','research/round12/experiment/REAL_VIEW_PROBE_PLAN_LOCK.json','research/round12/experiment/REAL_VIEW_ICP_PROBE_PLAN_LOCK.json','research/round12/experiment/REAL_VIEW_PNP_PROBE_PLAN_LOCK.json')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def describe(values):
    a=np.asarray([x for x in values if x is not None],float);assert np.isfinite(a).all()
    return {'n':len(a),'zero_count':int((a==0).sum()),'min':float(a.min()) if len(a) else None,'median':float(np.median(a)) if len(a) else None,'mean':float(a.mean()) if len(a) else None,'max':float(a.max()) if len(a) else None,'sum':float(a.sum())}
def _solver(g):
    x=g.get('diagnostics',{}).get('solver',{}).get('full_rgb')
    if x is None:return None
    return {k:x[k] for k in ('success','status','numerical_invalid','iterations')}
def view(v,role):
    reg=v.get('registration',{});ob=reg.get('observation_diagnostics',{});n=v.get('target_depthbound_unique_pairs');static=v.get('static_support')
    globalpass=bool(reg.get('accepted',False) and ob.get('background_only_qualification',False))
    unique='neighbor_mask_index' in v
    enough=n is not None and n>=6
    accepted=bool(v.get('verified',False));contradiction=bool(v.get('contradiction',False))
    if unique:assert globalpass
    if enough:assert unique
    if accepted:assert enough and static is not None and static['accepted']
    if contradiction:assert enough and static is not None and not static['accepted']
    return {'role':role,'frame_alias':v.get('frame_alias'),'status':v.get('status','unknown'),'registration_status':reg.get('status','not_called'),'registration_backend':v.get('registration_backend'),'global_PnP_pass':globalpass,'mask_LK_unique':unique,'depth_pairs':n,'depth_at_least6':enough,'static_pass':accepted,'static_contradiction':contradiction,'insufficient_depth_pairs_not_motion':unique and not enough,'static_residuals':static.get('residuals') if static else None,'fit_pairs':reg.get('n_fit'),'heldout_image_pairs':reg.get('n_holdout'),'heldout_Bcloud_points':reg.get('holdout_residuals',{}).get('n'),'unknown_clock':True}
def one(trace,z):
    assert tuple(z['arm_names'].tolist())==ARMS and str(z['sample_id'])==trace['id']
    q=int(z['q_source']);assert q==trace['q_source'] and np.all(z['arm_query_index']==q)
    b=z['arm_boxes_raw6'];acc=np.asarray(z['acceptance_mask6'],bool)
    assert b.shape==acc.shape==(8,6) and b.dtype==np.float32 and np.isfinite(b).all()
    change=b!=b[0];assert np.array_equal(change,z['actual_changed_axes'])
    for half,full in ((5,2),(6,3)):assert np.array_equal(b[half],b[0]+np.float32(.5)*(b[full]-b[0]))
    d=trace['diagnostics'];noun=d.get('target_noun',{});assoc=d.get('anchor_association',{});mv=d.get('multiview',{})
    roles=[x for x in trace['roles'] if 'frame_alias' in x];lookup={r['frame_alias']:r['role'] for r in roles}
    real=[view(v,lookup[v['frame_alias']]) for v in d.get('neighbor_attempts',[])]
    wrongd=d.get('wrongview',{});wrong=[view(wrongd['attempt'],'wrongview')] if 'attempt' in wrongd else []
    assert len(real)<=len(roles) and len(wrong)<=int(trace['wrong_frame_alias'] is not None)
    main='multiview_geometry' in d;unfiltered='unfiltered_multiview_geometry' in d
    assert main==unfiltered
    if main:assert real and any(v['static_pass'] for v in real) and not any(v['static_contradiction'] for v in real)
    if any(v['static_contradiction'] for v in real):assert not main and not change[[3,6,7]].any()
    if main:
        mg=d['multiview_geometry'];ug=d['unfiltered_multiview_geometry']
        assert np.array_equal(mg['arm_query_index'],ug['arm_query_index']) and mg['rgb_mask_sha256']==ug['rgb_mask_sha256']
    perpoint=[]
    for p in mv.get('neighbors',[]):
        perpoint.append({k:p[k] for k in ('visible_instance_points_before_transform','anchor_FOV','anchor_mask','anchor_mask_and_source_ROI','pixel_within3px','pixel_and_depth_within_threshold','fully_corroborated_neighbor_points')})
        perpoint[-1]['matched_anchor_unique_raw_n']=len(p['matched_anchor_raw_ids_unique'])
    mapping=d.get('multiview_geometry',{}).get('diagnostics',{}).get('canonical_allowed_model_mapping',{})
    if main:assert mapping['applied'] and mapping['same_raw_to_model_then_unique_return_inverse']
    matched_A_union=set(i for p in mv.get('neighbors',[]) for i in p['matched_anchor_raw_ids_unique'])
    selected=d.get('selected_anchor_mask_index');anchor_observed_n=next((c['visible_unique_foreground_in_source_ROI'] for c in assoc.get('candidate_tests',[]) if c['instance_index']==selected),None)
    graphs={1:d.get('grabcut',{}).get('geometry',{}),2:d.get('single_geometry',{}),3:d.get('multiview_geometry',{}),4:d.get('wrongview_geometry',{}),7:d.get('unfiltered_multiview_geometry',{})}
    return {'id':trace['id'],'platform':trace['platform'],'q0_fixed_verified':True,'anchor_raw_full_points':trace['anchor_raw_full_points'],'noun_status':noun.get('scope','not_parsed_invalid_source'),'noun_supported':bool(noun.get('COCO_class_family')),'anchor_association_status':assoc.get('status','not_called'),'anchor_candidate_n':len(assoc.get('candidates',[])) if assoc else None,'anchor_classfamily_detector_candidates_tested':len(assoc.get('candidate_tests',[])) if assoc else None,'roles_missing':sum('frame_alias' not in x for x in trace['roles']),'real_retrieved_n':len(roles),'wrong_retrieved_n':int(trace['wrong_frame_alias'] is not None),'real_attempts':real,'wrong_attempts':wrong,'real_not_attempted_due_anchor_or_noun':len(roles)-len(real),'wrong_not_attempted_due_anchor_or_noun':int(trace['wrong_frame_alias'] is not None)-len(wrong),
      'arm_status':z['arm_status'].tolist(),'arm_changed':change.any(1).tolist(),'arm_center_changed':change[:,:3].any(1).tolist(),'arm_size_changed':change[:,3:].any(1).tolist(),'arm_changed_axis6':change.tolist(),'arm_accepted_axis6':acc.tolist(),'arm_accepted_any':acc.any(1).tolist(),'center_increment_l2_m':np.linalg.norm(b[:,:3].astype(float)-b[0,:3].astype(float),axis=1).tolist(),'size_increment_l2_m':np.linalg.norm(b[:,3:].astype(float)-b[0,3:].astype(float),axis=1).tolist(),'solver_by_arm':{ARMS[i]:_solver(g) for i,g in graphs.items()},'half_arms_have_no_new_solver':True,
      'verified_real_neighbor_n':sum(v['static_pass'] for v in real),'MV_static_contradiction_veto':bool(mv.get('contradiction_veto',False)),'main_and_unfiltered_same_admission_verified':main==unfiltered,'MV_admitted':main,'perpoint':perpoint,'corroborating_anchor_raw_unique_union_n':len(matched_A_union) if main else None,'anchor_visible_instance_ROI_n':anchor_observed_n,'anchor_crossview_corroborated_fraction':len(matched_A_union)/anchor_observed_n if main and anchor_observed_n else None,'main_allowed_canonical_model_n':mapping.get('allowed_canonical_count'),'main_context_model_unique_n':mapping.get('model_unique_count'),'main_allowed_model_XYZ_sha256':mapping.get('allowed_model_XYZ_sha256'),'merged_raw_unique_n':mv.get('merged_raw_unique_n'),'unfiltered_merged_raw_unique_n':mv.get('unfiltered_merged_raw_unique_n'),'corroborated_added_unique_n':mv.get('new_raw_unique_count'),'unfiltered_added_unique_n':mv.get('unfiltered_new_raw_unique_count'),'allowed_corroborated_raw_XYZ_n':mv.get('allowed_corroborated_raw_XYZ_count'),
      'main_unfiltered_identical_float32':bool(np.array_equal(b[3],b[7])),'main_unfiltered_maxabs_difference_m':float(np.max(np.abs(b[3].astype(float)-b[7].astype(float)))),'main_only_accepted_axis_n':int(np.sum(acc[3]&~acc[7])),'unfiltered_only_accepted_axis_n':int(np.sum(acc[7]&~acc[3])),'main_and_unfiltered_input_mask_equal':True if main else None}
def stages(views,retrieved):
    return {'retrieved_preselected':retrieved,'actually_attempted':len(views),'not_attempted_anchor_or_noun':retrieved-len(views),'global_PnP_pass':sum(v['global_PnP_pass'] for v in views),'unique_mask_LK_track':sum(v['mask_LK_unique'] for v in views),'depth_pairs_at_least6':sum(v['depth_at_least6'] for v in views),'static_pass':sum(v['static_pass'] for v in views),'static_contradiction':sum(v['static_contradiction'] for v in views),'unique_track_insufficient_depth_not_proven_motion':sum(v['insufficient_depth_pairs_not_motion'] for v in views),'registration_statuses':dict(Counter(v['registration_status'] for v in views)),'terminal_statuses':dict(Counter(v['status'] for v in views)),'depth_pair_distribution_when_bound':describe([v['depth_pairs'] for v in views]),'fit_pairs':describe([v['fit_pairs'] for v in views]),'heldout_image_pairs':describe([v['heldout_image_pairs'] for v in views]),'heldout_Bcloud_points':describe([v['heldout_Bcloud_points'] for v in views]),'denominators_distinguished_not_attempted_is_not_registration_failure':True}
def aggregate(rows):
    real=[v for r in rows for v in r['real_attempts']];wrong=[v for r in rows for v in r['wrong_attempts']];points=[p for r in rows for p in r['perpoint']]
    d={'n':len(rows),'all_q0_fixed':all(r['q0_fixed_verified'] for r in rows),'noun_supported':sum(r['noun_supported'] for r in rows),'noun_statuses':dict(Counter(r['noun_status'] for r in rows)),'anchor_association_statuses':dict(Counter(r['anchor_association_status'] for r in rows)),'anchor_multicandidate':sum(r['anchor_candidate_n'] is not None and r['anchor_candidate_n']>1 for r in rows),'anchor_zero_candidate':sum(r['anchor_candidate_n']==0 for r in rows),'anchor_unique_candidate':sum(r['anchor_candidate_n']==1 for r in rows),'missing_real_roles':sum(r['roles_missing'] for r in rows),'real_views':stages(real,sum(r['real_retrieved_n'] for r in rows)),'wrong_views':stages(wrong,sum(r['wrong_retrieved_n'] for r in rows)),'verified_neighbor_count_per_anchor':dict(Counter(str(r['verified_real_neighbor_n']) for r in rows)),'static_contradiction_veto_anchors':sum(r['MV_static_contradiction_veto'] for r in rows),'MV_admitted_anchors':sum(r['MV_admitted'] for r in rows),'main_unfiltered_same_gate_all':all(r['main_and_unfiltered_same_admission_verified'] for r in rows),'main_unfiltered_identical_float32':sum(r['main_unfiltered_identical_float32'] for r in rows),'main_unfiltered_different_float32':sum(not r['main_unfiltered_identical_float32'] for r in rows),'main_unfiltered_maxabs_difference_m':describe([r['main_unfiltered_maxabs_difference_m'] for r in rows]),'main_only_accepted_axis_n':sum(r['main_only_accepted_axis_n'] for r in rows),'unfiltered_only_accepted_axis_n':sum(r['unfiltered_only_accepted_axis_n'] for r in rows),'perpoint_verified_nonveto_view_n':len(points),'perpoint':{k:describe([p[k] for p in points]) for k in ('visible_instance_points_before_transform','anchor_FOV','anchor_mask','anchor_mask_and_source_ROI','pixel_within3px','pixel_and_depth_within_threshold','fully_corroborated_neighbor_points','matched_anchor_unique_raw_n')}}
    for key in ('corroborating_anchor_raw_unique_union_n','anchor_visible_instance_ROI_n','anchor_crossview_corroborated_fraction','main_allowed_canonical_model_n','main_context_model_unique_n','merged_raw_unique_n','unfiltered_merged_raw_unique_n','corroborated_added_unique_n','unfiltered_added_unique_n','allowed_corroborated_raw_XYZ_n'):d[key]=describe([r[key] for r in rows])
    d['arms']={}
    for i,name in enumerate(ARMS):
        ss=[r['solver_by_arm'].get(name) for r in rows if r['solver_by_arm'].get(name) is not None]
        d['arms'][name]={'changed_samples':sum(r['arm_changed'][i] for r in rows),'center_changed_samples':sum(r['arm_center_changed'][i] for r in rows),'size_changed_samples':sum(r['arm_size_changed'][i] for r in rows),'accepted_samples':sum(r['arm_accepted_any'][i] for r in rows),'changed_axis6':np.sum([r['arm_changed_axis6'][i] for r in rows],axis=0).tolist() if rows else [0]*6,'accepted_axis6':np.sum([r['arm_accepted_axis6'][i] for r in rows],axis=0).tolist() if rows else [0]*6,'statuses':dict(Counter(r['arm_status'][i] for r in rows)),'center_increment_l2_m':describe([r['center_increment_l2_m'][i] for r in rows]),'size_increment_l2_m':describe([r['size_increment_l2_m'][i] for r in rows]),'solver_objects':len(ss),'solver_success':sum(s['success'] for s in ss),'solver_numerical_invalid':sum(s['numerical_invalid'] for s in ss),'solver_statuses':dict(Counter(s['status'] for s in ss)),'half_no_new_solver':i in (5,6)}
    return d

def main(a):
    assert sha(a.release)==a.release_sha256;release=read(a.release)
    assert release['mechanism_reduction_permitted'] is True and release['GT_read_permitted'] is False and release['reducer_sha256']==sha(Path(__file__))
    gd=Path(release['geometry_dir']);assert sha(gd/'SEALED.json')==release['geometry_seal_sha256']
    seal=read(gd/'SEALED.json');plan=read(gd/'registered_plan.json');root=Path(__file__).resolve().parents[3]
    assert seal['state']=='sealed_R12_8arm_independent_instance_multiview_direct_before_GT' and seal['n_arms']==8 and tuple(seal['arm_names'])==ARMS
    assert seal['plan_sha256']==sha(gd/'registered_plan.json') and seal['program_sha256']==plan['program_sha256']
    for pathkey,sealkey in [('source_dir','source_seal_sha256'),('mask_dir','mask_seal_sha256')]:assert sha(Path(plan[pathkey])/'SEALED.json')==seal[sealkey]
    for n,h in seal['program_sha256'].items():assert sha(root/n)==h,n
    for n,h in seal['files'].items():assert Path(n).name==n and sha(gd/n)==h,n
    traces=read(gd/'trace.json');assert len(traces)==seal['n_samples'];rows=[]
    for tr in traces:
        with np.load(gd/(tr['id']+'.npz'),allow_pickle=False) as z:fields={k:z[k] for k in KEYS}
        rows.append(one(tr,fields))
    assert len({r['id'] for r in rows})==len(rows)
    assert all((root/n).is_file() for n in LINEAGE_FILES),'required preflight lineage file missing'
    lineage={n:sha(root/n) for n in LINEAGE_FILES}
    public={'state':'author_no_GT_R12_mechanism_census_not_independent_quality_audit','binding':{'geometry_seal_sha256':release['geometry_seal_sha256'],'source_seal_sha256':seal['source_seal_sha256'],'mask_seal_sha256':seal['mask_seal_sha256'],'source_checkpoint_sha256':seal['source_checkpoint_sha256'],'mask_weights_sha256':seal['mask_weights_sha256'],'manifest_sha256':seal['manifest_sha256'],'pairmap_sha256':seal['pairmap_sha256'],'frame_index_sha256':seal['frame_index_sha256'],'program_sha256':seal['program_sha256'],'reducer_sha256':sha(Path(__file__)),'release_sha256':sha(a.release)},'all':aggregate(rows),'by_platform':{p:aggregate([r for r in rows if r['platform']==p]) for p in sorted({r['platform'] for r in rows})},'history_and_probe_plan_hashes_only':lineage,'lineage_scope':'Hash-only preservation of old G/old PnP and three input-probe plans; no probe data/results reread or metric comparison. Empty-Bholdout current guard stays bound in actual PnP program hash.','limits':['No target GT, task scores, annotations/index sidecars, images, masks, raw points or source features decoded.','Input-only frame index and pairmap contents are not reread; their existing hashes bind the observation selection.','Retrieved, actually attempted, accepted pair and geometry-active counts have different denominators.','Independent masks/class family/unique track/static support do not verify literal referring-expression identity.','Clock is unverified; this is accepted overlapping-view evidence, not a certified continuous-video/velocity/online claim.','Current PnP backend was registered after five input-development pairs with different-validation SIFT/ICP/PnP probes; counts do not prove algorithmic superiority.','Legacy named multiview_maskmatch_only_direct is unfiltered samegate aggregation, not anchor-only.','Main perpoint filtering changes loss support and normal context; samegate does not mean same accepted face set.','Confidence, coverage, small residual or changed geometry is not task utility. No neural TTA/model update is executed.']}
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'SUMMARY.json').write_text(json.dumps(public,indent=2,allow_nan=False)+'\n');(a.output/'CASES_PRIVATE.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    lines=['# R12 无标签机制计数','', '作者侧实现解释；未读取任务标签、分数、图像、点云或特征。','']
    for name,v in [('all',public['all']),*public['by_platform'].items()]:lines.extend([f"{name}: N={v['n']}, anchorunique={v['anchor_unique_candidate']}, realretrieved={v['real_views']['retrieved_preselected']}, attempted={v['real_views']['actually_attempted']}, PnPpass={v['real_views']['global_PnP_pass']}, staticpass={v['real_views']['static_pass']}, admittedanchors={v['MV_admitted_anchors']}.",''])
    lines.extend(['各臂接受/实际变化/尺寸/求解器、每层拒绝与逐点核验数据见SUMMARY.json。所有NA和零变化样本保持在原分母。','',*['- '+x for x in public['limits']]])
    (a.output/'SUMMARY.md').write_text('\n'.join(lines)+'\n');files={n:sha(a.output/n) for n in ('SUMMARY.json','CASES_PRIVATE.json','SUMMARY.md')}
    (a.output/'SEALED.json').write_text(json.dumps({'state':'sealed_R12_author_no_GT_mechanism_census','files':files,'release_sha256':sha(a.release),'reducer_sha256':sha(Path(__file__))},indent=2)+'\n');print(json.dumps({'n':len(rows),'seal_sha256':sha(a.output/'SEALED.json')}))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--release',type=Path,required=True);ap.add_argument('--release-sha256',required=True);ap.add_argument('--output',type=Path,required=True);main(ap.parse_args())
