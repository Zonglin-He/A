"""One low-priority reporting child, no automatic retry of failed work."""
from pathlib import Path
import os,json,time,subprocess,hashlib,traceback
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent;D=R/'outputs/reviewer_key_controls_v1';P=R/'results/reviewer_key_controls_v1'
def read(p):return json.loads(p.read_text())
def put(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)
def who(pid):return dict(pid=pid,start_ticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]))
def run(key,script,args,receipt):
 if receipt.exists():return
 dest=D/'statistics_execution';assert not(dest/(key+'_exit.json')).exists();dest.mkdir(parents=True,exist_ok=True);start=time.perf_counter();cmd=['taskset','-c','14-15','nice','-n','19','ionice','-c','3',str(R/'.conda/tubedetr/bin/python'),'-B',str(S/script),*args];env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 with(dest/(key+'.log')).open('w')as f:
  child=subprocess.Popen(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);ident=who(child.pid);put(dest/'state.json',dict(status='running',key=key,child=ident));code=child.wait()
 put(dest/(key+'_exit.json'),dict(worker=ident,actual_subprocess_wait=True,exit_code=code,elapsed_wall_seconds=time.perf_counter()-start));assert code==0 and receipt.exists(),(key,code)
if __name__=='__main__':
 dest=D/'statistics_execution';assert not(dest/'launch.json').exists();me=who(os.getpid());put(dest/'launch.json',dict(worker=me,entry=__file__,code_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest()for f in [Path(__file__),S/'statistics_v1.py',S/'audit_statistics_v2.py',S/'root_export_v1.py']}))
 try:
  for ds in ['hc2','vidstg']:
   run('offline_audit_'+ds,'audit_statistics_v2.py',['offline','--dataset',ds],P/'offline'/ds/'audit_receipt.json')
   run('offline_export_'+ds,'root_export_v1.py',['offline','--dataset',ds],P/'offline'/ds/'root_export_receipt.json')
   run('tta_derive_'+ds,'statistics_v1.py',['tta','--dataset',ds],P/'tta'/ds/'derive_receipt.json')
   run('tta_audit_'+ds,'audit_statistics_v2.py',['tta','--dataset',ds],P/'tta'/ds/'audit_receipt.json')
  run('corruption_degradation','root_export_v1.py',['corruption'],P/'corruption'/'root_export_receipt.json')
  for ds in ['hc2','vidstg']:
   start=time.perf_counter()
   while not(D/'temporal'/ds/'reporting_receipt.json').exists():
    if(D/'execution_failure_v1.json').exists():raise RuntimeError('CPU prediction prerequisite failed')
    assert time.perf_counter()-start<259200;put(dest/'state.json',dict(status='waiting_temporal_whole_audit',dataset=ds));time.sleep(10)
   run('temporal_root_'+ds,'root_export_v1.py',['temporal','--dataset',ds],D/'temporal'/ds/'root_export_receipt.json')
   run('evidence_derive_'+ds,'statistics_v1.py',['evidence','--dataset',ds],P/'temporal_evidence'/ds/'derive_receipt.json')
   run('evidence_audit_'+ds,'audit_statistics_v2.py',['evidence','--dataset',ds],P/'temporal_evidence'/ds/'audit_receipt.json')
  for ds in ['hc2','vidstg']:
   start=time.perf_counter()
   while not(D/'regions'/ds/'reporting_receipt.json').exists():
    if(D/'execution_failure_v1.json').exists():raise RuntimeError('CPU region prerequisite failed')
    assert time.perf_counter()-start<259200;time.sleep(10)
   run('regions_root_'+ds,'root_export_v1.py',['regions','--dataset',ds],D/'regions'/ds/'root_export_receipt.json')
  put(dest/'state.json',dict(status='all_CPU_diagnostics_and_root_exports_complete_pending_root_review_publication',worker=me))
 except BaseException:put(dest/'failure.json',dict(worker=me,error=traceback.format_exc()));raise
