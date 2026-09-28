"""Continue only registered whole episodes; failure stops for root, never retries."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v3_oracle_mixer_gap import D,OUT
from vg_tta.desta3d_v3_oracle_io import read,write,sha,check_pins,total_prior
PY=ROOT/'.venv-ptd-audit/bin/python'
def active(data):
    p=D/'ACTIVE.tmp.json';p.write_text(json.dumps(data,indent=2)+'\n');os.replace(p,D/'ACTIVE.json')
def execute(name,limit):
    cmd=[str(PY),'-B','scripts/desta3d_v3_oracle_mixer_gap.py','run','--name',name,'--limit',str(limit)]
    log=D/(name+'.log');start=time.monotonic()
    with log.open('xb') as f:
        p=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        active(dict(status='running',time=time.time(),controller_pid=os.getpid(),pid=p.pid,name=name,cmd=cmd,log=str(log)))
        rc=p.wait()
    elapsed=time.monotonic()-start;receipt=OUT/name/'RECEIPT.json';worker=read(receipt) if receipt.exists() else None
    seconds=max(0.,elapsed-(worker['seconds'] if worker else 0.));prior=total_prior()
    write(OUT/(name+'_wrapper')/'RECEIPT.json',dict(status='completed' if rc==0 else 'failed',seconds=seconds,
        child_wall_seconds=elapsed,worker_seconds=worker['seconds'] if worker else 0.,prior_seconds=prior,
        cumulative_seconds=prior+seconds,cap=None,includes='actual child setup/hash validation/teardown beyond worker timer'))
    if rc:
        active(dict(status='failed',time=time.time(),name=name,exit_code=rc,log=str(log)));raise RuntimeError(f'{name} failed; root repair required')
    return read(OUT/name/'COMPLETE.json')
def main(mode):
    check_pins(read(D/'LOCK.json')['pins'])
    if mode=='first':
        execute('oracle_mixer_gap_probe001',1)
        active(dict(status='first_episode_complete_waiting_root_raw_audit',time=time.time()));return
    assert read(D/'ROOT_FIRST_RAW_READBACK.json')['status']=='passed'
    write(D/'CONTROLLER_STARTED.json',dict(time=time.time(),pid=os.getpid(),code_sha=sha(Path(__file__)),scope='fixed remaining source447 audit only'))
    for i in range(1,13):
        if (D/'COMPLETE.json').exists():break
        done=execute(f'oracle_mixer_gap_run{i:03}',447);assert done['status'] in ('complete','safe_pause')
    else:
        if not (D/'COMPLETE.json').exists():raise RuntimeError('12 allocation engineering review boundary')
    active(dict(status='all_predictions_complete_awaiting_root_raw_and_score',time=time.time()))
    write(D/'CONTROLLER_COMPLETE.json',dict(time=time.time(),scope='predictions only; raw and score audit pending'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['first','continue']);a=p.parse_args()
    try:main(a.mode)
    except BaseException as e:
        dest=D/('CONTROLLER_'+a.mode+'_FAILURE.json')
        if not dest.exists():write(dest,dict(error=repr(e),traceback=traceback.format_exc(),time=time.time()))
        raise
