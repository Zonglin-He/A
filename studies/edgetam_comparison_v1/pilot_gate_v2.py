"""CG is user-paused, not complete; no paid predecessor is repeated."""
from pilot_execution_v2 import *
from pilot_gate_v1 import live
def gate():
 p=validate_execution();cp=R/'outputs/maintenance/CG_STVG_paused_by_user_20261026/owned_exit.json';a=read(cp);assert a['status']=='paused_by_user_actual_owned_exit_GPU_lease_free'and a['both_actual_exited']and a['GPU_lease_acquired_and_released']
 assert not live(a['controller'])and not live(a['worker'])
 release=D/'pilot_root_release_v2.json'
 if not release.exists():return None
 x=read(release);assert x['status']=='released_fixed256_EdgeTAM_after_explicit_CG_user_pause'and x['execution_fingerprint']==p['fingerprint']and x['CG_paused_proof_sha256']==sha(cp)
 assert x['latest_scope_sha256']==sha(R/'outputs/review_strengthening_v1/active_scope_cached_external_priority_after_CG_pause_v5.json')
 assert read(D/'pilot_preflight_v1.json')['status']=='passed_four_actual_cached_native_RGB_eightboxes_current_intervals_and_references_repeat_exact_gate_waits'and x['new_Source']==x['new_original_expert']==0
 return dict(user_CG_pause_sha256=sha(cp),release_sha256=sha(release),CG_not_completed=True,full_gamma_and_corruption_paused_not_completed=True)
