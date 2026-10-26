"""Actual small EdgeTAM result first; no cancelled or paused requirements."""
from eg_execution_common_v3 import *
from eg_pilot_gate_v2 import gate as active_gate,live
def gate():
 validate_execution_v3();dependency=active_gate()
 if dependency is None:return None
 edge=R/'outputs/edgetam_comparison_v1';w=read(edge/'pilot_controller_v1.json');assert w==dict(pid=792176,startticks=256423011)
 if live(w):return None
 assert read(edge/'pilot_controller_state_v1.json')['status']=='fixed256_EdgeTAM_scored_audited_pending_root_all_signed_cost_analysis_publication'
 assert read(edge/'pilot_GPU_exit_v1.json')['code']==0
 for ds in ['hc2','vidstg']:
  a=read(edge/'pilot_v1'/ds/'reporting_receipt.json');assert a['status']=='verified_fixed128_all_mask_support_parent_teacher_predictions_and_means'and a['queries']==a['parents']==128 and a['all_means_recomputed']and a['all_mask_support_and_reference_arrays_cold_exact']
 pub=edge/'small_comparison_publication_v1.json'
 if not pub.exists():return None
 p=read(pub);assert p['status']=='fixed256_EdgeTAM_complete_published'and p['all_remote_verified'];cp=Path(p['completion_path']);assert sha(cp)==p['completion_sha256']and read(cp)['status']=='verified_all_remote_tree_files_bytes_SHA_ancestors'
 return dict(active_CG_TTA=dependency,prioritized_EdgeTAM_publication_sha256=sha(pub),scope='Fixed cached256 ReferFormer only. Full missing Source-grid recovery deferred by latest user; no new Source, no automatic full extension.')
