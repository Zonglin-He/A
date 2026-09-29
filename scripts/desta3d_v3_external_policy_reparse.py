"""Sealed provider raw reused once; isolated official-prose parser, CPU only."""
import sys,time,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v3_external_policy_provider_v2 import D,OUT,read,write,sha,local_dependencies
from vg_tta.external_qualification_io import seal,verify_seal

def main():
 from vg_tta.llava_st_evidence_parser_v3 import parse_teacher_text
 from vg_tta.external_privileged_views import dense_spatial_support
 start=time.monotonic();old=OUT/'extgate_evidence002';dest=OUT/'extgate_evidence_parse_v3';assert not dest.exists()
 done,sealed=verify_seal(old);assert read(old/'ROOT_EVIDENCE_READBACK.json')['status']=='passed'
 p=subprocess.run([sys.executable,'-B','-m','pytest','-q','tests/test_llava_st_evidence_parser_v3.py'],text=True,capture_output=True);assert p.returncode==0,(p.stdout,p.stderr)
 write(dest/'CPU_PREFLIGHT.json',dict(status='passed',output=p.stdout));rows=read(old/'INPUTS.json');write(dest/'INPUTS.json',rows);write(dest/'CONFIG.json',read(old/'CONFIG.json'))
 paths=local_dependencies([Path(__file__),ROOT/'protocols/desta3d_v3_external_policy_parser_repair_v3.md',ROOT/'tests/test_llava_st_evidence_parser_v3.py',old/'PREDICTIONS_SEAL.json',old/'ROOT_EVIDENCE_READBACK.json']);write(dest/'LOCK.json',dict(pins={str(p):sha(p) for p in paths}));write(dest/'REGISTRATION.json',dict(status='registered_before_CPU_reparse',time=time.time(),GPU=False,GT=False))
 changes=[]
 for i,row in enumerate(rows):
  ep=dest/'episodes'/f'{i:02}';ep.mkdir(parents=True);orig=old/'episodes'/f'{i:02}'
  for n in ['INPUT.json','RAW_PREDICTION.json']:os.link(orig/n,ep/n)
  ident=read(ep/'INPUT.json');raw=read(ep/'RAW_PREDICTION.json');prior=read(orig/'EVIDENCE.json');ev=parse_teacher_text(raw['raw_text'],ident['frame_ids'],clip_bounds=ident['clip_bounds'])
  for k in ['boxes','frame_groups','spatial_usable','observation_frame_ids','duplicate_count']:assert ev[k]==prior[k]
  _,diag=dense_spatial_support(ident['frame_ids'],ev);assert diag==read(orig/'SUPPORT.json')
  write(ep/'EVIDENCE.json',ev);write(ep/'SUPPORT.json',diag);write(ep/'COMPLETE.json',dict(index=i,raw_exact=True,old_evidence_sha=sha(orig/'EVIDENCE.json')))
  changes.append(dict(query=i+1,old_errors=prior['errors'],new_errors=ev['errors'],temporal_before=prior['temporal_usable'],temporal_after=ev['temporal_usable'],spatial_usable=ev['spatial_usable'],format_valid=ev['format_valid'],syntax_repairs=ev['syntax_repairs']))
 write(dest/'COMPLETE.json',dict(status='completed_unscored_CPU_reparse',queries=16,parents=16,predictions=16,seal_sha=seal(dest,16),GPU=False,GT=False))
 verify_seal(dest)
 # Separate arithmetic check of interval support and retained box coordinates.
 for i,row in enumerate(rows):
  ep=dest/'episodes'/f'{i:02}';ev=read(ep/'EVIDENCE.json')
  if ev['temporal_usable']:
   a,b=ev['all_interval_mentions'][0];lo,hi=ev['clip_bounds'];assert ev['interval_physical']==[lo+a*(hi-lo),lo+b*(hi-lo)]
  assert sha(ep/'RAW_PREDICTION.json')==sha(old/'episodes'/f'{i:02}'/'RAW_PREDICTION.json')
 write(dest/'ROOT_EVIDENCE_READBACK.json',dict(status='passed',queries=16,parents=16,changes=changes,all_raw_exact=True,all_spatial_support_exact=True,CPU_seconds=time.monotonic()-start,GT_read=False,target_read=False));print('REPARSE_PASSED',changes,flush=True)
if __name__=='__main__':main()
