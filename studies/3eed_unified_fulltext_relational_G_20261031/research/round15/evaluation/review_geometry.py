"""Independent bounded R15 CPU audit; synthetic arrays only, no task inputs."""
from pathlib import Path
import hashlib,json,sys
import numpy as np

R=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(R/'crossplatform/round15'))
import check_geometry_cpu as fixture
from joint_geometry import generate_joint_targets

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    dest=Path(__file__).with_name('GEOMETRY_IMPLEMENTATION_INDEPENDENT_REVIEW.json')
    if dest.exists():raise FileExistsError(dest)
    lockpath=R/'research/round15/geometry/IMPLEMENTATION_LOCK.json'
    lock=json.loads(lockpath.read_text());checks={}
    for p,h in lock['program_and_dependency_sha256'].items():
        checks['bound_sha:'+p]=(sha(R/p)==h)
    replaypath=Path(__file__).with_name('GEOMETRY_CPU_INDEPENDENT_REPLAY.json')
    replay=json.loads(replaypath.read_text())
    checks['author_21_synthetic_checks_independently_reexecuted']=replay['check_count']==21 and all(replay['checks'].values())
    raw,inst,scene,boxes,scores=fixture.fixture()
    args=dict(scene=scene,boxes_raw6=boxes,text_scores=scores,graph_match=fixture.graph())
    a=generate_joint_targets(q0=0,**args);b=generate_joint_targets(q0=3,**args)
    checks['changing_original_q0_does_not_change_accepted_mapping_or_geometry']=a['q_association']==b['q_association']==2 and np.array_equal(a['arm_boxes_raw6'][1:],b['arm_boxes_raw6'][1:])
    permutation=np.array([4,2,0,3,1]);p=generate_joint_targets(scene=scene,boxes_raw6=boxes[permutation],text_scores=scores[permutation],q0=2,graph_match=fixture.graph())
    checks['candidate_permutation_retains_physical_selected_box_and_geometry']=permutation[p['q_association']]==a['q_association'] and np.array_equal(a['arm_boxes_raw6'],p['arm_boxes_raw6'])
    bad=generate_joint_targets(q0=3,**{**args,'graph_match':fixture.graph('M99999')})
    checks['unlisted_mask_does_not_guess_candidate']=not bad['association_accepted'] and np.array_equal(bad['arm_boxes_raw6'],np.tile(boxes[3],(5,1)))
    incomplete=generate_joint_targets(q0=3,**{**args,'graph_match':{**fixture.graph(),'search_complete':False}})
    checks['incomplete_semantic_search_fails_closed_without_geometric_override']=not incomplete['association_accepted'] and np.array_equal(incomplete['arm_boxes_raw6'],bad['arm_boxes_raw6'])
    checks['successful_geometry_half_is_same_q1_anchor_half_displacement']=np.array_equal(a['arm_boxes_raw6'][4],boxes[2]+np.float32(.5)*(a['arm_boxes_raw6'][2]-boxes[2])) and np.all(a['arm_query_index'][1:]==2)
    static={
      'whole_scene_projection_and_zbuffer_precede_instance_ownership':True,
      'exclusive_ownership_ambiguous_points_not_foreign_negative':True,
      'candidate_dedup_complete_link_score_independent':True,
      'reciprocal_mask_group_uniqueness_precedes_original_query_score_readout':True,
      'foreign_gate_requires_both_three_points_and_fraction_strictly_above_point_one':True,
      'source_q0_used_only_for_association_fallback':True,
      'geometry_is_one_sided_containment_plus_source_prior_no_visible_face_equality':True,
      'full_noRGB_share_mask_support_gate_and_search_domain':True,
      'accepted_geometry_failure_fallback_is_q1_anchor':True,
      'no_teacher_ema_parameter_update_or_GT_input_API':True}
    checks.update({'static_review:'+k:v for k,v in static.items()})
    checks={k:bool(v) for k,v in checks.items()}
    assert all(checks.values()),[k for k,v in checks.items() if not v]
    out={'state':'PASS_R15_BOUNDED_GEOMETRY_IMPLEMENTATION_AND_SYNTHETIC_CONTROLS_NOT_TASK_EFFICACY',
      'implementation_lock_sha256':sha(lockpath),'root_authorization_sha256':lock['root_authorization_sha256'],
      'auditor_sha256':sha(__file__),'independent_replay_sha256':sha(replaypath),
      'checks':checks,'check_count':len(checks),'independent_extra_synthetic_cases':5,
      'blocking_implementation_issues':[],
      'scope':{'code_read':True,'author_fixture_independently_executed':True,'separate_synthetic_metamorphic_checks':True,
       'real_sensor_or_target_data_read':False,'GT_or_task_IoU_computed':False,'model_GPU_or_parameter_updates_executed':False,
       'task_effectiveness_established':False,'future_eight_arm_wrapper_reviewed':False},
      'scientific_limits':['Predicted masks and reciprocal geometry do not certify target identity.',
       'Single independent mask or absent foreign support cannot exclude an oversized proposal; explicit synthetic negative retained.',
       'Same-mask RGB-gated points in noRGB remain RGB-conditioned, not a deployment PC-only control.',
       'Size expansion and center displacement resolve partial-observation ambiguity using a source prior; optimization domain is not coordinate identifiability.',
       'Geometry failure returns associated q1, but semantic association failure returns original q0; all denominators must retain both.',
       'Relation-disabled or flipped semantic paths may change q1/support and therefore are whole-G counterfactuals unless these are explicitly frozen.'],
      'input_author_role_boundary':'This agent authored R15 input registration; independent input replay must be performed by root or another reviewer.',
      'old_scientific_files_modified':False,
      'auditor_draft_failure_lineage':'Initial own-auditor attempt passed assertions then JSON rejected numpy.bool_; old draft preserved history/review_geometry_before_bool_serialization_fix. Scalar bool conversion only; no method, real input, or existing evidence changed.'}
    dest.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'state':out['state'],'checks':len(checks),'path':str(dest),'sha256':sha(dest)}))
if __name__=='__main__':main()
