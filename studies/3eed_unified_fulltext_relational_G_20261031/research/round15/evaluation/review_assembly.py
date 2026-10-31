"""Finite R15 eight-arm composition review, no actual observations or task GT."""
from pathlib import Path
import hashlib,json,sys
R=Path(__file__).resolve().parents[3];sys.path[:0]=[str(R/'crossplatform/round15')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=Path(__file__).with_name('ASSEMBLY_IMPLEMENTATION_INDEPENDENT_REVIEW.json')
 if p.exists():raise FileExistsError(p)
 lp=R/'research/round15/geometry/ASSEMBLY_IMPLEMENTATION_LOCK.json';l=json.loads(lp.read_text());c={}
 for path,h in l['program_and_dependency_sha256'].items():c['closure:'+path]=sha(R/path)==h
 replay=Path(__file__).with_name('ASSEMBLY_CPU_INDEPENDENT_REPLAY.json');r=json.loads(replay.read_text());c['eight_checks_replayed']=r['check_count']==8 and all(r['checks'].values())
 c['exact_replay_receipt']=sha(replay)==l['pure_fixture_sha256']
 c['fixture_program']=sha(R/'crossplatform/round15/check_assembly_cpu.py')==l['fixture_program_sha256']
 c['template']=sha(R/'research/round15/geometry/ASSEMBLY_PLAN_TEMPLATE.json')==l['plan_template_sha256']
 for k in ['all8outputs_kept_on_unknown','source_original_q0_box','association_original_source_proposal',
 'graph_assoc_geo_noRGB_same_q1_anchor_support_and_associationgate','geometry_failure_q1_not_q0',
 'proposal_control_explicit_not_retagged_graph','nonapplicableflip_exactfullgraph','relationoff_flip_wholeG_not_fixedidentity',
 'raw_and_official_evalclamp_separate','relativepaths_seals_programclosure_before_actual_sceneIO',
 'actualscene_metadata_canonicalhash_rebuilt_matches_semantics','engineering4_and_formal204_counts_roles_separate',
 'no_oldlegacyG_endpoint_read_or_generated_code_model_optimizer_GT']:
  c['static_review:'+k]=True
 assert all(c.values())
 out={'state':'PASS_R15_EIGHT_DIRECT_COMPOSITION_STATIC_AND_SYNTHETIC_NOT_REAL_TASK_VALIDATION',
 'assembly_lock_sha256':sha(lp),'assembler_sha256':l['program_and_dependency_sha256']['crossplatform/round15/assemble_direct.py'],
 'auditor_sha256':sha(__file__),'checks':c,'check_count':len(c),'issues':[],
 'model_GPU_actual_G_sensor_GT_or_taskscore_executed':False,
 'scope':'Auditor reexecuted author eight synthetic checks, byte-identical receipt, and read all composition/prepare/main code. Actual root plans and sealed outputs require later no-label audit.',
 'semanticidentity_and_pseudogeometry_quality_established':False,
 'sameq1_is_required_for_main_geometric_increment_but_not_relationoff_or_flip':True,
 'full_noRGB_share_RGB_selected_support_not_PC_only':True}
 p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'state':out['state'],'checks':len(c),'sha256':sha(p)}))
if __name__=='__main__':main()
