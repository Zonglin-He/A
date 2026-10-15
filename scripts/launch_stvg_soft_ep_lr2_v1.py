#!/usr/bin/env python3
"""Own-unit launch wrapper for the finite LR-x2 Student stage; not run by preparation."""
from __future__ import annotations
import os, subprocess, sys, threading, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
from scripts import launch_stvg_instance_identity_v1 as resource
BASE=ROOT/'artifacts/stvg_soft_ep_opd_v1/lr2'

def main():
    rt=read(BASE/'RUNTIME.json')
    if rt.get('status')!='locked' or rt.get('GT_read') is not False: raise RuntimeError('LR2 runtime is not locked GT-free')
    for rel,digest in {**rt['pins'],**rt['protected']}.items():
        if sha(ROOT/rel)!=digest: raise RuntimeError('runtime pin mismatch: '+rel)
    unit='stvg-soft-ep-lr2-001.service'; out=BASE/'execution_lr2_001'
    out.mkdir(parents=True,exist_ok=False)
    resource.BASE,resource.UNIT,resource.STAGE=out,unit,'development'
    gib=1024**3; sample=resource.sample()
    if sample['available']<8*gib or sample['gpu_free']<26*gib or sample['disk_free']<30*gib:
        raise RuntimeError('resource admission guard failed; no worker started')
    props=dict(row.split('=',1) for row in subprocess.check_output(['systemctl','--user','show',unit,'--property=MemoryHigh,MemoryMax,MemorySwapMax'],text=True).splitlines() if '=' in row)
    if [int(props[k]) for k in ('MemoryHigh','MemoryMax','MemorySwapMax')]!=[16*gib,20*gib,gib]:
        raise RuntimeError('systemd resource properties differ from locked limits')
    write(out/'ACTUAL_LIMITS.json',dict(status='pass',sample=sample,properties=props,GT_read=False))
    stop=threading.Event(); threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
    env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (out/'worker.log').open('xb') as log:
        proc=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/run_stvg_soft_ep_lr2_v1.py','development'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        write(out/'LAUNCH.json',dict(status='running',worker_pid=proc.pid,unit=unit,runtime_sha256=sha(BASE/'RUNTIME.json'),time=time.time(),GT_read=False))
        code=proc.wait()
    stop.set(); write(out/'WORKER_EXIT.json',dict(status='exited',exit_code=code,time=time.time(),GT_read=False))
    return code

if __name__=='__main__': raise SystemExit(main())
