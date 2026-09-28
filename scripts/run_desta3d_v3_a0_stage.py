"""Run one explicitly requested A0 stage; never auto-expand or retry."""
import argparse,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v3_a0_fast_screen import D,OUT
from vg_tta.desta3d_v3_oracle_io import read,write,sha,total_prior

def main(action,name):
    if action=='fit':assert read(D/'ROOT_CACHE_READBACK.json')['status']=='passed'
    if action=='native':assert read(D/'ROOT_DIRECTION_READBACK.json')['decision']=='native_dev64'
    cmd=[str(ROOT/'.venv-ptd-audit/bin/python'),'-B','scripts/desta3d_v3_a0_fast_screen.py',action,'--name',name]
    log=D/(name+'.log');start=time.monotonic()
    write(D/(name+'_WRAPPER_REGISTRATION.json'),dict(time=time.time(),action=action,code_sha=sha(Path(__file__)),cmd=cmd))
    with log.open('xb') as stream:
        child=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        active=dict(status='running',time=time.time(),pid=child.pid,wrapper_pid=os.getpid(),action=action,name=name,log=str(log),cmd=cmd)
        p=D/'ACTIVE.tmp.json';p.write_text(__import__('json').dumps(active,indent=2)+'\n');os.replace(p,D/'ACTIVE.json')
        code=child.wait()
    elapsed=time.monotonic()-start;p=OUT/name/'RECEIPT.json';worker=read(p)['seconds'] if p.exists() else 0.
    prior=total_prior();seconds=max(0.,elapsed-worker)
    write(OUT/(name+'_wrapper')/'RECEIPT.json',dict(status='completed' if code==0 else 'failed',seconds=seconds,child_wall_seconds=elapsed,worker_seconds=worker,prior_seconds=prior,cumulative_seconds=prior+seconds,cap=None))
    active.update(status='completed_waiting_root_audit' if code==0 else 'failed_root_repair_required',exit_code=code,time=time.time())
    p=D/'ACTIVE.tmp.json';p.write_text(__import__('json').dumps(active,indent=2)+'\n');os.replace(p,D/'ACTIVE.json')
    sys.exit(code)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['cache','fit','native']);p.add_argument('--name',required=True);a=p.parse_args();main(a.action,a.name)
