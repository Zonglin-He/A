"""Owned CPU-only sequential launcher; each prediction/score/audit is independent."""
from pathlib import Path
import os, sys, json, subprocess, time, traceback
R=Path(__file__).resolve().parents[2]
D=R/'outputs/temporal_alignment_object_evidence_v1'
def atomic(p,x):
    p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)
def ident(pid):
    return dict(pid=pid,start_ticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]))
if __name__=='__main__':
    assert not (D/'execution_launch_v1.json').exists()
    me=ident(os.getpid());atomic(D/'execution_launch_v1.json',dict(worker=me,entry=__file__,plan='plan_v1.json'))
    try:
        for ds in ['hc2','vidstg']:
            for stage in ['predict','score','audit']:
                assert not (D/ds/f'{stage}_exit_v1.json').exists()
                command=['taskset','-c','14-15','nice','-n','19','ionice','-c','3',str(R/'.conda/tubedetr/bin/python'),'-B',str(Path(__file__).with_name('run.py')),stage,'--dataset',ds]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                log=D/ds/f'{stage}_execution_v1.log';log.parent.mkdir(parents=True,exist_ok=True)
                start=time.perf_counter()
                with log.open('w') as h:
                    child=subprocess.Popen(command,cwd=R,env=env,stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
                    who=ident(child.pid);atomic(D/'execution_state_v1.json',dict(status='running',dataset=ds,stage=stage,worker=me,child=who,command=command))
                    code=child.wait()
                atomic(D/ds/f'{stage}_exit_v1.json',dict(worker=who,exit_code=code,actual_subprocess_wait=True,elapsed_wall_seconds=time.perf_counter()-start,log=str(log.relative_to(R))))
                assert code==0,(ds,stage,code)
        atomic(D/'execution_state_v1.json',dict(status='all_two_domains_scored_audited_pending_root_analysis_publication',worker=me,new_Source=0,new_expert=0,new_optimizer=0))
    except BaseException:
        atomic(D/'execution_failure_v1.json',dict(worker=me,error=traceback.format_exc()));raise
