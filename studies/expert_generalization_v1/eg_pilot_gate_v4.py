"""Do not require paused CG to complete or call it completed."""
from eg_execution_common_v4 import *
from eg_pilot_gate_v1 import live
def gate():
 validate_execution_v4();pause=R/'outputs/maintenance/CG_STVG_paused_by_user_20261026/owned_exit.json';z=read(pause);assert z['status']=='paused_by_user_actual_owned_exit_GPU_lease_free'and z['both_actual_exited']and not live(z['controller'])and not live(z['worker'])
 edge=R/'outputs/edgetam_comparison_v1';w=read(edge/'pilot_controller_v2.json')
 if live(w):return None
 assert read(edge/'pilot_controller_state_v2.json')['status']=='fixed256_EdgeTAM_scored_audited_pending_root_all_signed_cost_analysis_publication'and read(edge/'pilot_GPU_exit_v2.json')['code']==0
 for ds in rc.COUNTS:
  a=read(edge/'pilot_v1'/ds/'reporting_receipt.json');assert a['status']=='verified_fixed128_all_mask_support_parent_teacher_predictions_and_means'and a['queries']==a['parents']==128 and a['all_means_recomputed']and a['all_mask_support_and_reference_arrays_cold_exact']
 pub=edge/'small_comparison_publication_v1.json'
 if not pub.exists():return None
 q=read(pub);cp=Path(q['completion_path']);assert q['status']=='fixed256_EdgeTAM_complete_published'and q['all_remote_verified']and sha(cp)==q['completion_sha256']and read(cp)['status']=='verified_all_remote_tree_files_bytes_SHA_ancestors'
 t=R/'outputs/expert_matched_tta_v1';who=read(t/'controller_runtime_v4.json');assert who==dict(pid=390224,startticks=255906478)
 if live(who):return None
 assert read(t/'runtime_v4_state.json')['status']=='both_full_domains_scored_audited_pending_root_analysis_publication'
 release=D/'pilot_predecessor_completion_v4.json'
 if not release.exists():return None
 rec=read(release);assert rec['status']=='active_same_expert_TTA_and_EdgeTAM_complete_published_CG_user_paused'and rec['CG_pause_sha256']==sha(pause)and rec['EdgeTAM_publication_sha256']==sha(pub)
 assert set(rec['same_expert_TTA'])==set(rc.COUNTS)
 for ds,item in rec['same_expert_TTA'].items():
  a=t/ds/'reporting_receipt.json';v=read(a);assert sha(a)==item['audit_sha256']and v['status']=='verified_whole_domain'and v['queries']==rc.COUNTS[ds]and v['all_arrays_cold_rederived_exact']and v['all_means_recomputed'];root=t/ds/'root_export_receipt.json';assert sha(root)==item['root_export_sha256'];p=Path(item['completion_path']);assert sha(p)==item['completion_sha256']and read(p)['status']=='verified_all_remote_tree_files_bytes_SHA_ancestors'
 return dict(CG_paused_not_completed_sha256=sha(pause),prioritized_EdgeTAM_publication_sha256=sha(pub),active_same_expert_TTA_release_sha256=sha(release),zero_new_Source=True,full_grid_recovery_deferred=True)
