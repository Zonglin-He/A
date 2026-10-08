"""Finite CPU-only post-seal evaluation; hands visual/publication work to root."""
import os,sys,time,subprocess,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_paper_baseline_scoring_common_v1 import BASE,read,write,status,sha,seal

def run():
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';BASE.mkdir(parents=True,exist_ok=True)
    status(BASE/'STATUS.json',dict(status='verifying_selected_seals',pid=os.getpid(),GT_read=False,time=time.time()))
    seal()
    workers=[]
    for ds in ['vidstg','hc2']:
        log=BASE/(ds+'_score.log');f=log.open('x')
        cmd=[sys.executable,'-B',str(ROOT/'scripts/score_decota_paper_baseline_scoring_v1.py'),ds]
        p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,env=os.environ);f.close()
        workers.append((ds,p))
        write(BASE/(ds+'_LAUNCH.json'),dict(pid=p.pid,command=cmd,log=str(log.relative_to(ROOT)),CPU_only=True,time=time.time()))
    status(BASE/'STATUS.json',dict(status='scoring_selected_sealed_baselines_CPU',pid=os.getpid(),workers={ds:p.pid for ds,p in workers},time=time.time()))
    failures=[]
    for ds,p in workers:
        rc=p.wait()
        if rc:failures.append(dict(dataset=ds,returncode=rc))
    assert not failures,failures
    for script in ['merge','audit','report']:
        if script=='merge':
            from scripts.score_decota_paper_baseline_scoring_v1 import merge
            merge();continue
        name=f'{script}_decota_paper_baseline_scoring_v1.py';log=BASE/(script+'.log')
        status(BASE/'STATUS.json',dict(status='independent_statistics' if script=='audit' else 'writing_report_and_figures',pid=os.getpid(),time=time.time()))
        with log.open('x') as f:subprocess.run([sys.executable,'-B',str(ROOT/'scripts'/name)],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
    status(BASE/'STATUS.json',dict(status='pending_actual_root_visual_publication_archive',pid=os.getpid(),GPU_seconds=0,all_original_paper_complete=False,time=time.time()))

if __name__=='__main__':
    try:run()
    except BaseException:
        status(BASE/'STATUS.json',dict(status='failed_preserved_needs_root_engineering_readback',pid=os.getpid(),time=time.time(),traceback=traceback.format_exc()))
        raise
