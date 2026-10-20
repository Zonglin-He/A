"""User steering: primary evaluation seed41 only; preserve original three-seed work."""
import sys,time,copy
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts import run_stvg_spatial_strong_bidirectional256_v1 as base
original_verify=base.verify

def effective_verify():
 p=original_verify();rev=base.read(base.D/'SEED41_REVISION001.json');assert rev['primary_seeds']==[41]and rev['original_plan_sha256']==base.sha(base.D/'PLAN_LOCK.json')
 p=copy.deepcopy(p);p['seeds']=[41]
 for ds in p['configs']:p['configs'][ds]['seeds']=[41]
 return p

def revision():
 assert not (base.D/'SEED41_REVISION001.json').exists()
 p=original_verify();files={str(f.relative_to(R)):base.sha(f)for f in (base.D/'predictions').rglob('*.pt')};counts={ds:len(list((base.D/'predictions'/ds).glob('*.pt')))for ds in ['hc2','vidstg']}
 rev=dict(reason='explicit latest user instruction: one seed instead of three',primary_seeds=[41],original_seeds=p['seeds'],original_plan_sha256=base.sha(base.D/'PLAN_LOCK.json'),checkpoint_counts=counts,original_completed_files=files,additional_completed_seed42_43='retained as historical compute, excluded from primary table and bootstrap',all_other_parameters_unchanged=True,GT_read=False,epoch=time.time())
 base.put(base.D/'SEED41_REVISION001.json',rev);base.put(base.P/'SEED41_REVISION_PUBLIC.json',{k:v for k,v in rev.items()if k not in ['original_completed_files','epoch']})

def main():
 base.verify=effective_verify;action=sys.argv[1]
 if action=='measure':base.measure(sys.argv[2])
 elif action=='complete':
  from scripts import audit_stvg_spatial_strong_bidirectional256_v1 as audit
  from scripts import score_stvg_spatial_strong_bidirectional256_v1 as score
  audit.verify=score.verify=effective_verify
  import inspect,textwrap
  source=textwrap.dedent(inspect.getsource(audit.main));assert source.count('assert fits==768 and steps==30720')==1
  source=source.replace('assert fits==768 and steps==30720','assert fits==256 and steps==10240')
  exec(compile(source,__file__,'exec'),vars(audit));audit.main();score.main()
  report=base.P/'REPORT.md';text=report.read_text().replace('seeds41/42/43','seed41 (latest user revision; completed42/43 excluded from primary results)').replace('Three-seed matrix time is not single-inference latency.','The original three-seed and revised single-seed matrix compute are separately disclosed; do not pool them as repeated independent parents.')
  report.write_text(text)
 elif action=='profile':
  from scripts import profile_stvg_spatial_strong_bidirectional256_v1 as profile
  profile.verify=effective_verify;profile.main()
 elif action=='review':
  from scripts import review_stvg_spatial_strong_bidirectional256_v1 as review
  review.verify=effective_verify;review.main()
 else:raise ValueError(action)
if __name__=='__main__':
 if sys.argv[1]=='revision':revision()
 else:main()
