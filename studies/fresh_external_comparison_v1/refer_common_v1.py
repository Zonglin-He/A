"""Same fixed256 cached-or-fresh current evidence; separate pinned alternative expert."""
from pathlib import Path
import os,sys,time,json,hashlib,shutil
from functools import lru_cache
sys.path.insert(0,str(Path(__file__).parent))
import current_evidence_runtime_v2 as preparation
import current_evidence_v1 as current
from current_evidence_v1 import R,N,read,sha,atomic,identity,digest,fixed
rc=current.rc;S=R/'studies/expert_generalization_v1';OLDREF=R/'outputs/expert_generalization_v1';D=N/'referformer_v1';P=R/'results/fresh_external_comparison_v1/referformer_v1'
sys.path.insert(0,str(S))
def pilot_rows(ds):return current.pilot_rows(ds)
def resources(start):
 current.resources(start);assert sum(f.stat().st_size for f in N.rglob('*')if f.is_file())<160*2**30
def validate_pilot():
 from eg_pilot_v1 import validate_pilot as original_assets
 preparation.validate_execution();original_assets();p=read(N/'refer_plan_v1.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert sha(R/f)==h,f
 return p
def payload(ds,row):
 import torch,numpy as np
 cap,raw,frames,ip=current.inputs(ds,row);e=fixed(N/ds/'sealed.json')['files'][row['id']];assert ip==e['input_binding'];t=e['current_teacher'];tp=Path(t['payload_path']);assert sha(tp)==t['payload_sha256']and sha(t['receipt_path'])==t['receipt_sha256'];oldteacher=torch.load(tp,map_location='cpu',weights_only=False,mmap=True);assert sha(N/e['path'])==e['sha256']and sha(N/e['receipt_path'])==e['receipt_sha256']
 with np.load(N/e['path'])as z:reference=dict(CurrentSimple=z['Simple'].copy(),CurrentOurs=z['Ours'].copy(),CurrentOurs_interval=z['Ours_interval'].copy())
 proof=dict(actual_input_binding=ip,current_teacher=t,current_baseline_prediction_sha256=e['sha256'],current_baseline_receipt_sha256=e['receipt_sha256'],current_baseline_audit_sha256=sha(N/ds/'current_audit_receipt_v1.json'),caption_sha256=hashlib.sha256(' '.join(row['metadata']['caption'].lower().split()).encode()).hexdigest(),positions8=cap['positions8'],frame_ids=cap['frame_ids'],new_Source=0,new_original_expert=0,GT_read=False)
 return cap,raw,frames,oldteacher,reference,proof
def require_release():
 q=read(D/'root_predecessor_release_v1.json');assert q['status']=='actual_EdgeTAM_fixed256_complete_published_current256_audited_CG_gamma_corruption_user_paused'
 ed=N/'edgetam_v1';pub=read(ed/'small_comparison_publication_v1.json');assert pub['status']=='fixed256_EdgeTAM_complete_published'and pub['all_remote_verified'];assert q['EdgeTAM_publication_sha256']==sha(ed/'small_comparison_publication_v1.json');completion=Path(pub['completion_path']);assert sha(completion)==pub['completion_sha256'];assert read(completion)['status']=='verified_all_remote_tree_files_bytes_SHA_ancestors'
 assert read(ed/'GPU_exit.json')['code']==0
 for f in ['controller_v1.json','GPU_child.json']:
  x=read(ed/f);st=Path('/proc')/str(x['pid'])/'stat';assert not st.exists()or int(st.read_text().split(') ',1)[1].split()[19])!=x['startticks']
 for ds in ['hc2','vidstg']:
  a=read(ed/'pilot_v1'/ds/'reporting_receipt.json');assert a['status']=='verified_fixed128_all_mask_support_parent_teacher_predictions_and_means'and a['queries']==128;assert q['EdgeTAM_audit_sha256'][ds]==sha(ed/'pilot_v1'/ds/'reporting_receipt.json');assert sha(Path(audit_path:=q['current_audit_paths'][ds]))==q['current_audit_sha256'][ds]and read(Path(audit_path))['queries']==128
 assert q['current256_audit_completion_sha256']==sha(N/'current_audit_completion_v1.json');assert read(N/'current_audit_completion_v1.json')['totals']['queries']==256
 assert sha(R/'outputs/maintenance/CG_STVG_paused_by_user_20261026/owned_exit.json')==q['CG_pause_ownedexit_sha256']
 tta=read(R/'outputs/expert_matched_tta_v1/root_completion.json');assert tta['status']=='both_full_domains_complete_published'and sha(R/'outputs/expert_matched_tta_v1/root_completion.json')==q['same_expert_TTA_completion_sha256']
 return q
