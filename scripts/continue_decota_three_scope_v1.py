"""One finite serialized GPU queue. Conditional phases are handed to root."""
import sys,os,time,subprocess,fcntl,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_three_scope_common_v1 import BASE,read,write,status

def main():
    with (BASE/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);write(BASE/'LAUNCH.json',dict(pid=os.getpid(),time=time.time(),finite=True))
        try:
            for ds in ['vidstg','hc2']:
                for stage in ['features','evidence','online']:
                    tag={'features':'FEATURE','evidence':'EVIDENCE','online':'ONLINE'}[stage]
                    if (BASE/ds/(tag+'_BARRIER.json')).exists():continue
                    status(BASE/'STATUS.json',dict(status='running',stage=stage,dataset=ds,pid=os.getpid(),time=time.time()))
                    with (BASE/(ds+'_'+stage+'.log')).open('a') as out:
                        subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/run_decota_three_scope_v1.py',stage,ds],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
            subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/run_decota_three_scope_v1.py','seal'],cwd=ROOT,check=True)
            status(BASE/'STATUS.json',dict(status='completed_pending_root_CPU_audit_conditional_stages',pid=os.getpid(),time=time.time()))
        except Exception as e:
            write(BASE/'FAILURE.json',dict(error=str(e),traceback=traceback.format_exc(),time=time.time(),pid=os.getpid()))
            status(BASE/'STATUS.json',dict(status='engineering_failed_pending_root',pid=os.getpid(),time=time.time()));raise
if __name__=='__main__':main()
