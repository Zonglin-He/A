"""Transient own-scope supervisor: bounded memory, fixed finite call roster, no elapsed deadline."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_temporal_caption_query_v1 import BASE,read,write,sha
from scripts.stvg_temporal_caption_teacher_guard_v1 import cgroup,sample,monitor


def main():
    admission=sample(cgroup(),gpu=True)
    if admission['available']<8*1024**3 or admission['gpu_free']<8*1024**3:
        raise RuntimeError('RAM/GPU admission margin below 8GiB')
    stop=threading.Event()
    threading.Thread(target=monitor,args=(stop,BASE),daemon=True).start()
    child=None
    def term(signum,frame):
        write(BASE/'SUPERVISOR_STOP_RECEIPT.json',{'status':'own_unit_received_termination',
              'pid':os.getpid(),'worker_pid':child.pid if child else None,'GT_read':False,'time':time.time()},mutable=True)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,term)
    command=[str(ROOT/'.venv/stvg_native_distill_v1/bin/python'),'-B',str(ROOT/'scripts/run_stvg_temporal_caption_teacher_v1.py')]
    with (BASE/'logs/temporal_caption_worker001.log').open('xb') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT)
        write(BASE/'LAUNCH.json',{'status':'actual_launched','stage':'P0','controller_pid':os.getpid(),
              'worker_pid':child.pid,'runtime_sha256':sha(BASE/'RUNTIME.json'),'GT_read':False,'time':time.time()},mutable=True)
        code=child.wait()
    stop.set()
    write(BASE/'WORKER_EXIT_RECEIPT.json',{'status':'actual_P0_worker_exited','stage':'P0','worker_pid':child.pid,
          'exit_code':code,'GT_read':False,'time':time.time()},mutable=True)
    return code


if __name__=='__main__':raise SystemExit(main())
