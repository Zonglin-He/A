"""Complete the single registered A0 chain; errors stop, no retries or expansion."""
import json,os,select,subprocess,sys,time,traceback,ctypes,errno,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v3_a0_fast_screen import D,OUT
from vg_tta.desta3d_v3_oracle_io import read,write,sha,total_prior
PY=ROOT/'.venv-ptd-audit/bin/python'

def command(args,log):
    with (D/log).open('xb') as f:
        p=subprocess.run([str(PY),'-B',*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError(f'{args}: exit{p.returncode}; preserve failure and wait for root')

def active(data):
    p=D/'ACTIVE.tmp.json';p.write_text(json.dumps(data,indent=2)+'\n');os.replace(p,D/'ACTIVE.json')

def native_stage():
    command(['scripts/desta3d_v3_a0_native.py','prepare'],'NATIVE_REGISTRATION.log')
    name='a0_native001';cmd=[str(PY),'-B','scripts/desta3d_v3_a0_native.py','run','--name',name];start=time.monotonic()
    with (D/(name+'.log')).open('xb') as f:
        p=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        active(dict(status='running',action='native',pid=p.pid,wrapper_pid=os.getpid(),name=name,time=time.time(),log=str(D/(name+'.log')),cmd=cmd))
        rc=p.wait()
    elapsed=time.monotonic()-start;rp=OUT/name/'RECEIPT.json';worker=read(rp)['seconds'] if rp.exists() else 0.
    seconds=max(0.,elapsed-worker);prior=total_prior()
    write(OUT/(name+'_wrapper')/'RECEIPT.json',dict(status='completed' if rc==0 else 'failed',seconds=seconds,worker_seconds=worker,child_wall_seconds=elapsed,prior_seconds=prior,cumulative_seconds=prior+seconds,cap=None))
    if rc:raise RuntimeError('Conditional native failed; no retry')

def main():
    a=read(D/'ACTIVE.json');assert a['name']=='a0_cache001'
    paths=[Path(__file__),ROOT/'scripts/audit_desta3d_v3_a0_screen.py',ROOT/'scripts/run_desta3d_v3_a0_stage.py',ROOT/'scripts/desta3d_v3_a0_native.py',ROOT/'scripts/score_desta3d_v3_a0_native.py',ROOT/'scripts/summarize_desta3d_v3_a0_screen.py']
    write(D/'PIPELINE_V2_REGISTRATION.json',dict(time=time.time(),pid=os.getpid(),pins={str(p):sha(p) for p in paths},scope='only cache audit,200 fixed cached steps,direction audit,conditional64 native,score; never full/fresh/expert/OPD',already_running_cache_pid=a['pid']))
    # Kernel wait on the existing wrapper: no polling, no duplicate cache work.
    try:
        assert platform.machine()=='x86_64'
        libc=ctypes.CDLL(None,use_errno=True);fd=libc.syscall(434,int(a['wrapper_pid']),0)
        if fd<0:
            err=ctypes.get_errno()
            if err==errno.ESRCH:raise ProcessLookupError(err,'wrapper exited')
            raise OSError(err,os.strerror(err))
        select.select([fd],[],[]);os.close(fd)
    except ProcessLookupError:pass
    assert read(OUT/'a0_cache001/RECEIPT.json')['status']=='completed'
    assert read(OUT/'a0_cache001_wrapper/RECEIPT.json')['status']=='completed'
    assert read(OUT/'a0_cache001/COMPLETE.json')['queries']==192
    command(['scripts/audit_desta3d_v3_a0_screen.py','cache'],'CACHE_AUDIT.log')
    command(['scripts/run_desta3d_v3_a0_stage.py','fit','--name','a0_fit001'],'FIT_WRAPPER.log')
    command(['scripts/audit_desta3d_v3_a0_screen.py','direction'],'DIRECTION_AUDIT.log')
    if read(D/'ROOT_DIRECTION_READBACK.json')['decision']=='native_dev64':
        native_stage();command(['scripts/score_desta3d_v3_a0_native.py'],'NATIVE_SCORE.log')
    command(['scripts/summarize_desta3d_v3_a0_screen.py'],'SUMMARY.log')
    active(dict(status='completed_waiting_root_final_review',time=time.time(),controller_pid=os.getpid(),decision=read(D/'ROOT_COMPLETION_SUMMARY.json')['decision']))
    write(D/'PIPELINE_V2_COMPLETE.json',dict(time=time.time(),decision=read(D/'ROOT_COMPLETION_SUMMARY.json')['decision']))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(D/'PIPELINE_V2_FAILURE.json',dict(time=time.time(),error=repr(e),traceback=traceback.format_exc()))
        active(dict(status='failed_root_repair_required',time=time.time(),controller_pid=os.getpid(),error=repr(e)));raise
