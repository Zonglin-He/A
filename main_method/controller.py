"""Finite supervised engineering32 chain; every successor requires real receipts."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
from .runtime import ROOT,BASE,read,put,sha,PinVerifier,resource_guard,status
from .run import STAGES

def main():
    a=argparse.ArgumentParser();a.add_argument('--scope',choices=['engineering32'],default='engineering32');args=a.parse_args()
    permit=read(BASE/'CLEANUP_PUBLICATION_RECEIPT.json')
    assert permit['status']=='pass' and permit['remote_verified'] and permit['model_calls_before_cleanup_publication']==0
    pins=read(BASE/'CODE_LOCK.json')['pins'];checker=PinVerifier(pins);checker.verify(full=True)
    put(BASE/'RUN_AUTHORIZATION.json',dict(status='active',scope=args.scope,method_sha256=sha(BASE/'METHOD_LOCK.json'),
        human_authorized=True,cleanup_commit=permit['commit'],start_epoch=time.time(),GT_read=False))
    stages=[('gpu_equivalence',['-m','main_method.qualification','gpu_source'])]
    for stage in STAGES:
        if stage in ('source','inputs'):continue
        stages.append((stage,['-m','main_method.run',stage,'--scope',args.scope]))
    try:
        for stage,cmd in stages:
            checker.verify();resource_guard();log=BASE/'execution'/stage;log.mkdir(parents=True,exist_ok=False)
            put(log/'LAUNCH.json',dict(status='launched',stage=stage,argv=[sys.executable,'-B',*cmd],GT_read=False))
            status(stage,0,32,GT_read=False)
            with (log/'worker.log').open('xb') as f:
                worker=subprocess.Popen([sys.executable,'-B',*cmd],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
                put(log/'PID.json',dict(pid=worker.pid,start_epoch=time.time(),stage=stage,GT_read=False))
                # Parent never issues GPU-worker signals. Each worker owns the common GPU lease.
                result=worker.wait()
            put(log/'EXIT.json',dict(status='success' if result==0 else 'failed',exit_code=result,GT_read=False))
            if result:raise RuntimeError('stage failed: '+stage)
        checker.verify(full=True)
        put(BASE/'GROUP_PUBLICATION_HANDOFF.json',dict(status='ready',count=32,
            actual_ROOT_complete=True,actual_CPU_complete=True,not_official_full_result=True,
            next_action='root independent scalar/output/pixel/privacy review and immediate GitHub publication; then official dispatcher qualification'))
        status('engineering32_complete_publication_pending',32,32,GT_read=True)
    except BaseException as e:
        put(BASE/'FAILURE.json',dict(status='failed',error_type=type(e).__name__,error=str(e),timestamp_epoch=time.time(),
            original_outputs_preserved=True,automatic_scientific_retry=False))
        raise

if __name__=='__main__':main()
