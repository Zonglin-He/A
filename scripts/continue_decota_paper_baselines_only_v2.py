"""User-authorized baseline-only continuation after actual parameter selection."""
import sys,os,time,subprocess,fcntl,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_opd_tuning_common_v1 import *

def run():
    verify();selected=read(BASE/'SELECTION_BARRIER.json')
    assert set(selected['datasets'])=={'vidstg','hc2'} and selected['status']=='sealed'
    gates=[]
    for name in ['CONTROLLER.lock','TABLE1_CONTINUATION.lock']:
        f=(PAPER/name).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);gates.append(f)
    for f,h in read(PAPER/'TABLE1_CONTINUATION_RUNTIME.json')['pins'].items():assert sha(ROOT/f)==h,f
    def step(action):
        verify();budget();log=BASE/('baseline_'+action+'.log')
        with log.open('ab') as out:
            command=[str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/decota_baseline_bridge_v2.py',action]
            p=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
            md=dict(status='running',controller_pid=os.getpid(),worker_pid=p.pid,action=action,log=str(log),
                    scope='original_paper_baselines_only_after_tuning',GT_read=False,time=time.time())
            status(BASE/'BASELINE_STATUS.json',md);status(PAPER/'STATUS.json',md)
            assert p.wait()==0,'Baseline continuation failure: '+action
    step('lock')
    if not (PAPER/'baselines/TENT_vidstg/PREDICTION_BARRIER.json').exists():step('resume_TENT_vidstg')
    for method in ['TENT','SAR']:
        for ds in ['hc2','vidstg']:
            if not (PAPER/'baselines'/f'{method}_{ds}'/'PREDICTION_BARRIER.json').exists():
                step('smoke_'+method+'_'+ds);step(method+'_'+ds)
    step('fisher_vidstg')
    if not (PAPER/'baselines/EATA_hc2/PREDICTION_BARRIER.json').exists():
        step('smoke_EATA_hc2');step('EATA_hc2')
    if (PAPER/'source_Fisher/hc2/MEDIA_BARRIER.json').exists():
        step('fisher_hc2')
        if not (PAPER/'baselines/EATA_vidstg/PREDICTION_BARRIER.json').exists():
            step('smoke_EATA_vidstg');step('EATA_vidstg')
        final='all_original_baseline_predictions_sealed_pending_root_audit'
    else:final='available_baselines_sealed_HC_source_Fisher_media_dependency_pending'
    md=dict(status=final,scope='baselines_only',other_method_experiments_paused=True,time=time.time())
    status(BASE/'BASELINE_STATUS.json',md);status(PAPER/'STATUS.json',md)
    archive('调参两集参数已锁后原baseline队列真实接续到当前有限交接；'+final+'；未恢复任何旧Ours或方法组件实验')

if __name__=='__main__':
    try:run()
    except BaseException:
        d=BASE/'baseline_failures'/str(time.time_ns());d.mkdir(parents=True,exist_ok=True)
        (d/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'BASELINE_STATUS.json',dict(status='failed_preserved',failure=str(d),time=time.time()));raise
