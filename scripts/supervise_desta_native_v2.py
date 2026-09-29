"""Finish the registered 27 -> 6 -> 1 experiment, serially, without new gates."""
from pathlib import Path
import os
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.desta_native_common import D,OUT,read,write,sha
from scripts.launch_desta_native_v2 import launch


def command(script,phase):
    result=subprocess.run([str(ROOT/'.venv-ptd-audit/bin/python'),'-B',script,phase],cwd=ROOT)
    if result.returncode:raise RuntimeError(f'CPU verification failed: {script} {phase}')


def main():
    assert (D/'experts/temporal/COMPLETE.json').exists() and (D/'experts/spatial/COMPLETE.json').exists()
    write(D/'CONTROLLER_STARTED_V2.json',dict(pid=os.getpid(),time=time.time(),script_sha=sha(Path(__file__))))
    try:
        for phase in ('dev16','dev64','ablations'):
            while not (D/(phase+'_SEAL.json')).exists():
                number=1
                while (OUT/f'desta_native_{phase}{number:03}').exists():number+=1
                name=f'{phase}{number:03}'
                code=launch(phase,name)
                if code:raise RuntimeError(f'{name} failed; root review required, no silent retry')
                state=read(OUT/('desta_native_'+name)/'COMPLETE.json')
                assert state['status'] in ('safe_pause','complete')
            command('scripts/audit_desta_native.py',phase)
            command('scripts/score_desta_native.py',phase)
        command('scripts/summarize_desta_native.py','final')
        write(D/'CONTROLLER_COMPLETE.json',dict(time=time.time(),status='finished_pending_root_review'))
    except BaseException as exc:
        write(D/'CONTROLLER_FAILURE_V2.json',dict(time=time.time(),error=repr(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
