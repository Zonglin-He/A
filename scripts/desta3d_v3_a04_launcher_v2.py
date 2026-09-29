"""Isolated orchestration repair: mutable ACTIVE status, original scientific worker/pins untouched."""
import argparse,os,time,subprocess,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v3_a04_screen import D
from vg_tta.desta3d_v3_oracle_io import OUT,read,write,sha,check_pins,total_prior

def main(steps):
    dest=D/f'S{steps}';check_pins(read(dest/'LOCK.json')['pins']);assert not (dest/'STARTED.json').exists()
    assert steps==2000 and read(D/'S200/ROOT_READBACK.json')['decision']=='continue_same_trajectory_to_2000'
    start=time.monotonic();cmd=[str(ROOT/'.venv-ptd-audit/bin/python'),'-B',str(ROOT/'scripts/desta3d_v3_a04_screen.py'),'worker','--steps',str(steps)]
    env={**os.environ,'OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    with (dest/'RUN001.log').open('xb') as log:
        child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        (D/'ACTIVE.json').write_text(json.dumps(dict(status='running',steps=steps,pid=child.pid,wrapper_pid=os.getpid(),cmd=cmd,time=time.time(),launcher_version=2),indent=2)+'\n')
        code=child.wait()
    wall=time.monotonic()-start
    receipt=read(dest/'RECEIPT.json') if (dest/'RECEIPT.json').exists() else dict(status='failed_before_allocation',seconds=0)
    write(OUT/f'a04_s{steps}'/'RECEIPT.json',receipt);overhead=max(0,wall-receipt['seconds']);prior=total_prior()
    write(OUT/f'a04_s{steps}_wrapper'/'RECEIPT.json',dict(status='completed' if code==0 else 'failed',seconds=overhead,worker_seconds=receipt['seconds'],child_wall_seconds=wall,prior_seconds=prior,cumulative_seconds=prior+overhead,cap=None))
    state=dict(status='completed_pending_audit' if code==0 else 'failed',steps=steps,exit_code=code,time=time.time(),launcher_version=2)
    write(dest/'EXIT.json',state);(D/'ACTIVE.json').write_text(json.dumps(state,indent=2)+'\n')
    assert code==0,'Failure preserved; no replay'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,choices=[2000],default=2000);main(p.parse_args().steps)
