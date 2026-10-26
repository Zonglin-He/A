"""No GPU/model before actual predecessor exits, complete audits and root release."""
from eg_common_v1 import *

def live(w):
 try:
  s=Path('/proc',str(w['pid']),'stat').read_text().rsplit(') ',1)[1].split()
  return s[0]!='Z'and int(s[19])==w['startticks']
 except FileNotFoundError:return False

SPECS={
 'CG':dict(folder='shared_corrected_roi_CG_v1',identity='controller.json',state='state.json',status='both_CG_domains_scored_audited_pending_root_analysis_publication',owner=dict(pid=3374443,startticks=252798286)),
 'gamma':dict(folder='review_gamma_full_v1',identity='controller_v4.json',state='state_v4.json',status='both_gamma_domains_scored_audited_pending_root_curves_analysis_publication',owner=dict(pid=486668,startticks=256050365)),
 'same_expert_TTA':dict(folder='expert_matched_tta_v1',identity='controller_runtime_v4.json',state='runtime_v4_state.json',status='both_full_domains_scored_audited_pending_root_analysis_publication',owner=dict(pid=390224,startticks=255906478)),
}

def gate():
 for spec in SPECS.values():
  folder=R/'outputs'/spec['folder'];w=read(folder/spec['identity']);assert w==spec['owner']
  if live(w):return None
  assert read(folder/spec['state'])['status']==spec['status'],'Predecessor failed; preserve, do not construct model'
 # A root release is generated only after the actual complete publication and
 # independent official/all-array audits. Missing it is normal waiting.
 release=D/'pilot_predecessor_completion_v1.json'
 if not release.exists():return None
 rec=read(release);assert rec['status']=='all_named_noncorruption_predecessors_complete_published'
 assert set(rec['predecessors'])==set(SPECS)
 out={}
 for name,spec in SPECS.items():
  data=rec['predecessors'][name];assert data['owner']==spec['owner']and not live(data['owner'])
  folder=R/'outputs'/spec['folder'];assert sha(folder/spec['state'])==data['successful_state_sha256']
  assert set(data['domains'])==set(rc.COUNTS)
  for ds,item in data['domains'].items():
   audit=Path(item['audit_path']);assert sha(audit)==item['audit_sha256'];a=read(audit)
   assert a['status']=='verified_whole_domain'and a['queries']==rc.COUNTS[ds]and a['all_means_recomputed']
   exact=a.get('all_compact_arrays_cold_rederived_exact',False)or a.get('all_five_gamma_arrays_cold_rederived_exact',False)or a.get('all_arrays_cold_rederived_exact',False)
   assert exact
   root=Path(item['root_export_path']);assert sha(root)==item['root_export_sha256']
   cp=Path(item['completion_path']);assert sha(cp)==item['completion_sha256']and read(cp)['status']=='verified_all_remote_tree_files_bytes_SHA_ancestors'
  out[name]=data
 pause=R/'outputs/maintenance/corruption_paused_by_user_20261026/owned_exit.json'
 assert sha(pause)==rec['corruption_deferral_sha256']and read(pause)['status']=='paused_by_user'
 return dict(release_sha256=sha(release),predecessors=out,corruption_paused_not_completed=True)
