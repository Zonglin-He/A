"""One finite controller; no historical queues or automatic new methods."""
import os,sys,time,subprocess,fcntl,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_identity_common_v1 import BASE,read,write,status
PY=ROOT/'.conda/tubedetr/bin/python'
def run(script,*args):
    name='_'.join(args)
    status(BASE/'STATUS.json',dict(status='running',stage=name,pid=os.getpid(),time=time.time()))
    with (BASE/(name+'.log')).open('a') as out:subprocess.run([str(PY),'-B',script,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
def main():
    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(BASE/'LAUNCH.json',dict(pid=os.getpid(),time=time.time(),finite=True))
        try:
            # Vid worker was root-launched before this continuation; never duplicate it.
            if not (BASE/'vidstg/MATCHED_BARRIER.json').exists():
                s=read(BASE/'vidstg/STATUS.json');pid=s['pid']
                status(BASE/'STATUS.json',dict(status='waiting_existing_matched_vidstg',worker_pid=pid,pid=os.getpid(),time=time.time()))
                while not (BASE/'vidstg/MATCHED_BARRIER.json').exists():
                    assert Path(f'/proc/{pid}').exists(),'Existing Vid worker disappeared without barrier'
                    time.sleep(2)
            # Wait for the original worker to release its GPU lease.
            s=read(BASE/'vidstg/STATUS.json')
            while Path(f"/proc/{s['pid']}").exists():time.sleep(2)
            if not (BASE/'hc2/MATCHED_BARRIER.json').exists():run('scripts/run_decota_identity_commitment_v1.py','predict','hc2')
            if not (BASE/'MATCHED_GLOBAL_BARRIER.json').exists():run('scripts/run_decota_identity_commitment_v1.py','seal')
            if not (BASE/'matched_CPU_LOCK.json').exists():run('scripts/score_decota_identity_commitment_v1.py','lock','matched')
            if not (BASE/'SEARCH_SELECTION.json').exists():run('scripts/score_decota_identity_commitment_v1.py','score','matched')
            if not (BASE/'ONLINE_LOCK.json').exists():run('scripts/run_decota_identity_commitment_v1.py','online_prepare')
            for ds in ['vidstg','hc2']:
                if not (BASE/ds/'ONLINE_BARRIER.json').exists():run('scripts/run_decota_identity_commitment_v1.py','online',ds)
            if not (BASE/'ONLINE_GLOBAL_BARRIER.json').exists():run('scripts/run_decota_identity_commitment_v1.py','online_seal')
            if not (BASE/'online_CPU_LOCK.json').exists():run('scripts/score_decota_identity_commitment_v1.py','lock','online')
            if not (ROOT/'results/decota_identity_commitment/2026-10-05/online/ROOT_AUDIT.json').exists():run('scripts/score_decota_identity_commitment_v1.py','score','online')
            status(BASE/'STATUS.json',dict(status='completed_pending_root_review_publication',pid=os.getpid(),time=time.time()))
        except Exception as e:
            write(BASE/'FAILURE.json',dict(time=time.time(),pid=os.getpid(),error=str(e),traceback=traceback.format_exc()))
            status(BASE/'STATUS.json',dict(status='engineering_failed_pending_root',pid=os.getpid(),time=time.time()));raise
if __name__=='__main__':main()
