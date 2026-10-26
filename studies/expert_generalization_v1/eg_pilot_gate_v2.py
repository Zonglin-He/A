"""CG and same-expert TTA complete; gamma and corruption explicitly deferred."""
from eg_common_v1 import *
from eg_execution_common_v2 import validate_execution
from eg_pilot_gate_v1 import SPECS as ORIGINAL_SPECS,live
SPECS={k:v for k,v in ORIGINAL_SPECS.items()if k!='gamma'}

def gate():
 validate_execution()
 for spec in SPECS.values():
  folder=R/'outputs'/spec['folder'];w=read(folder/spec['identity']);assert w==spec['owner']
  if live(w):return None
  assert read(folder/spec['state'])['status']==spec['status'],'Failed actual predecessor; preserve'
 release=D/'pilot_predecessor_completion_v2.json'
 if not release.exists():return None
 rec=read(release);assert rec['status']=='all_active_CG_TTA_predecessors_complete_published_after_gamma_deferral'and set(rec['predecessors'])==set(SPECS)
 out={}
 for name,spec in SPECS.items():
  data=rec['predecessors'][name];folder=R/'outputs'/spec['folder'];assert data['owner']==spec['owner']and not live(data['owner'])and sha(folder/spec['state'])==data['successful_state_sha256'];assert set(data['domains'])==set(rc.COUNTS)
  for ds,item in data['domains'].items():
   a=Path(item['audit_path']);assert sha(a)==item['audit_sha256'];audit=read(a);assert audit['status']=='verified_whole_domain'and audit['queries']==rc.COUNTS[ds]and audit['all_means_recomputed'];assert audit.get('all_compact_arrays_cold_rederived_exact',False)or audit.get('all_arrays_cold_rederived_exact',False)
   root=Path(item['root_export_path']);assert sha(root)==item['root_export_sha256'];cp=Path(item['completion_path']);assert sha(cp)==item['completion_sha256']and read(cp)['status']=='verified_all_remote_tree_files_bytes_SHA_ancestors'
  out[name]=data
 for name,path in [('gamma',R/'outputs/maintenance/gamma_paused_by_user_20261026/review_gamma_full_v1/owned_exit.json'),('corruption',R/'outputs/maintenance/corruption_paused_by_user_20261026/owned_exit.json')]:assert sha(path)==rec[name+'_deferral_sha256']and read(path)['status']=='paused_by_user'
 return dict(release_sha256=sha(release),predecessors=out,gamma_and_corruption_paused_not_completed=True)
