"""Run the finite mechanistic probe once, without retries or old queue resumption."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from stvg.tube_runner import write

OUT=ROOT/'outputs/tubedetr_reward_mechanism'
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()

def main():
    from tools.tubedetr_reward_mechanism_probe import inputs
    inputs()
    for stage in ['cpu','network','score']:
        command=[sys.executable,'-B','-u','tools/tubedetr_reward_mechanism_probe.py','--stage',stage]
        log=ROOT/'logs'/f'tubedetr_reward_mechanism_{stage}.log'
        write(OUT/'status.json',dict(status=stage,as_of=now(),new_parameter_search=False))
        env=dict(os.environ)
        if stage!='network':env['CUDA_VISIBLE_DEVICES']=''
        with log.open('a')as stream:
            p=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=env)
            reg=ROOT/'runtime/tubedetr_active_jobs.json';r=json.loads(reg.read_text())
            r['jobs']['reward_mechanism_worker']=dict(pid=p.pid,command=shlex.join(command),log=str(log.relative_to(ROOT)),stage=stage)
            write(reg,r)
            if p.wait():raise RuntimeError('Mechanism probe stage failed: '+stage)
    write(OUT/'status.json',dict(status='complete',as_of=now(),cpu_queries=128,network_queries=16,old_queues_still_deferred=True))

if __name__=='__main__':
    try:main()
    except Exception as e:
        v=dict(status='failed',error=type(e).__name__,message=str(e),as_of=now());write(OUT/'failure.json',v);write(OUT/'status.json',v);raise
