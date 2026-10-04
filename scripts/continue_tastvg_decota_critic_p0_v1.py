"""Finite serial P0, then sealed CPU scoring/report; no historical queue."""
import os, sys, time, subprocess, traceback, fcntl
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.tastvg_decota_critic_common_v1 import *


def run():
    handle = (BASE/'controller.lock').open('a')
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify(); assert read(BASE/'SMOKE_ROOT_ACCEPTANCE.json')['status'] == 'pass'
    write(BASE/'LAUNCH.json', dict(pid=os.getpid(), time=time.time(), argv=sys.argv, finite=True))
    try:
        stages = [('predict', ds) for ds in DATASETS] + [('seal', None)]
        for stage, ds in stages:
            log = BASE/(f'{ds}_{stage}.log' if ds else f'{stage}.log')
            assert not log.exists()
            cmd = ['bash','scripts/with_local_cuda.sh', str(ROOT/'.conda/tubedetr/bin/python'),
                   '-B','scripts/run_tastvg_decota_critic_p0_v1.py', stage]
            if ds: cmd.append(ds)
            with log.open('w') as out:
                p = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
                status(BASE/'STATUS.json', dict(status='running', stage=stage, dataset=ds,
                    controller_pid=os.getpid(), worker_pid=p.pid, log=str(log), GT_read=False))
                assert p.wait() == 0, (stage, ds, p.returncode)
        for script, stage in [('scripts/score_audit_tastvg_decota_critic_p0_v1.py','run'),
                              ('scripts/report_tastvg_decota_critic_p0_v1.py',None)]:
            with (BASE/('CPU.log' if stage else 'REPORT.log')).open('w') as out:
                cmd = [str(ROOT/'.conda/tubedetr/bin/python'),'-B',script]
                if stage: cmd.append(stage)
                p = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
                status(BASE/'STATUS.json', dict(status='CPU_scoring_audit' if stage else 'reporting',
                    controller_pid=os.getpid(), worker_pid=p.pid, GT_read=bool(stage)))
                assert p.wait() == 0, (script, p.returncode)
        status(BASE/'STATUS.json', dict(status='completed_pending_root_visual_publication',
            unique_inputs=576, logical_arrivals=1152, time=time.time()))
    except BaseException as e:
        write(BASE/'CONTROLLER_FAILURE.json', dict(error=repr(e), traceback=traceback.format_exc(), time=time.time()))
        status(BASE/'STATUS.json', dict(status='failed_pending_root', error=repr(e), time=time.time()))
        raise
    finally:
        handle.close()


if __name__ == '__main__': run()
