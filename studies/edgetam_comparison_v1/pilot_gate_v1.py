"""Next GPU priority after actual active CG success; no deferred prerequisite."""
from pilot_common_v1 import *
def live(w):
 try:s=Path('/proc',str(w['pid']),'stat').read_text().rsplit(') ',1)[1].split();return s[0]!='Z'and int(s[19])==w['startticks']
 except FileNotFoundError:return False
def gate():
 cg=R/'outputs/shared_corrected_roi_CG_v1';who=read(cg/'controller.json');assert who==dict(pid=3374443,startticks=252798286)
 if live(who):return None
 assert read(cg/'state.json')['status']=='both_CG_domains_scored_audited_pending_root_analysis_publication'
 result={}
 for ds in ['hc2','vidstg']:
  for stage in ['GPU','CPU']:assert read(cg/f'{ds}_{stage}_exit.json')['code']==0
  a=cg/f'{ds}_reporting_receipt.json';z=read(a);assert z['status']=='verified_whole_domain'and z['all_compact_arrays_cold_rederived_exact']and z['all_means_recomputed']
  pub=cg/f'{ds}_publication.json'
  if not pub.exists():return None
  q=read(pub);assert q['status']=='complete_published'and q['all_remote_verified'];cp=Path(q['completion_path']);assert sha(cp)==q['completion_sha256']and read(cp)['status']=='verified_all_remote_tree_files_bytes_SHA_ancestors';result[ds]=dict(audit_sha256=sha(a),publication_sha256=sha(pub),completion_sha256=sha(cp))
 release=D/'pilot_root_release_v1.json'
 if not release.exists():return None
 q=read(release);assert q['status']=='released_fixed256_EdgeTAM_after_active_CG_complete_cached_priority'and q['CG_complete']==result
 assert sha(R/'outputs/review_strengthening_v1/active_scope_cached_external_priority_v4.json')==q['latest_scope_sha256']
 return dict(CG_complete=result,release_sha256=sha(release),gamma_and_corruption_deferred_CG_HC_control_cancelled=True)
