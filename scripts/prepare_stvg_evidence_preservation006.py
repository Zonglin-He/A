"""Exact fixed-reference control and one separately locked preservation ablation."""
import sys,copy,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_evidence_common_v1 import *
def main():
 lock=read(BASE/'SPATIAL_SELECTION_LOCK.json');assert lock['conditional_preservation_required'];v=lock['variant'];b=BASE/'preservation';b.mkdir(exist_ok=False);old=BASE/'round2'
 for name in ['TEACHER_GLOBAL_BARRIER.json','TEACHER_JOB_LOCK.json','CANDIDATE_INPUT_BARRIER.json']:
  shutil.copyfile(old/name,b/name)
 write(b/'VARIANT_LOCK.json',dict(status='locked',variants=[v,'preserved'],GT_read=False))
 refs=[]
 for z in read(old/'REFERENCE_LOCK.json')['records']:
  refs.append({**{k:z[k]for k in ['case_id','dataset','source_case_sha256','GT_read']},v:copy.deepcopy(z[v]),'preserved':copy.deepcopy(z[v])})
 write(b/'REFERENCE_LOCK.json',dict(status='locked',records=refs,GT_read=False));(b/'temporal').mkdir()
 for rec in roster():
  cid=rec['probe_alias'];shutil.copyfile(old/'temporal'/(cid+'.pt'),b/'temporal'/(cid+'.pt'))
 root=read(old/'ROOT_INPUT_READBACK.json');assert root['status']=='pass'
 write(b/'ROOT_INPUT_READBACK.json',dict(status='pass',scope='exact reuse of complete round2 independently pixel/geometry-read input and references, no new teacher or input construction',parent_root_sha256=sha(old/'ROOT_INPUT_READBACK.json'),parent_reference_sha256=sha(old/'REFERENCE_LOCK.json'),reference_lock_sha256=sha(b/'REFERENCE_LOCK.json'),case_count=32,GT_read=False,CPU_only=True))
 # Every preservation reference must equal the selected source reference, including the original teacher receipt hash.
 assert all(r[v]==r['preserved']for r in refs)
 print('preservation prepared',v,len(refs))
if __name__=='__main__':main()
