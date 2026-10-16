"""One finite CPU continuation bound to the actual P0 controller identity."""
from pathlib import Path
import os,sys,time,subprocess,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
from scripts.stvg_opd_p7a_paper_pidfd001 import wait_once
BASE=ROOT/'artifacts/stvg_temporal_evidence_budget_v1/revision002'

def main():
    rt=read(BASE/'POSTSEAL_RUNTIME002.json');assert rt['status']=='locked'and not rt['GT_read']
    for rel,h in rt['pins'].items():assert sha(ROOT/rel)==h,rel
    p=rt['parent'];props=dict(x.split('=',1)for x in subprocess.check_output(['systemctl','--user','show',p['unit'],'--property=MainPID,ActiveState'],text=True).splitlines()if'='in x)
    if props['ActiveState']=='active':
        assert int(props['MainPID'])==p['pid']
        stat=Path('/proc')/str(p['pid'])/'stat';ticks=int(stat.read_text().split(') ',1)[1].split()[19]);assert ticks==p['start_ticks']
        assert sha(Path('/proc')/str(p['pid'])/'cmdline')==p['cmdline_sha256']
        write(BASE/'POSTSEAL_STATUS002.json',dict(status='waiting',stage='four_arm_prediction_and_root_barrier',GT_read=False),mutable=True)
        wait_once(p['pid'])
    pred=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');assert pred['status']=='complete'and pred['count']==16 and pred['arm_count']==4 and not pred['GT_read']
    for rel,h in pred['bindings'].items():assert sha(ROOT/rel)==h
    write(BASE/'POSTSEAL_STATUS002.json',dict(status='running',stage='postseal_scoring',GT_read=False),mutable=True)
    out=BASE/'execution003/postseal_score';out.mkdir(parents=True,exist_ok=False)
    with(out/'worker.log').open('xb')as f:
        tick=time.time();proc=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/score_stvg_evidence_budget_p0_002.py'],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1'),stdout=f,stderr=subprocess.STDOUT)
        write(out/'LAUNCH.json',dict(status='running',worker_pid=proc.pid,GT_read=False,time=tick));code=proc.wait()
    write(out/'WORKER_EXIT.json',dict(status='exited',exit_code=code,time=time.time(),GT_read=True));assert code==0
    cpu=read(BASE/'complete/CPU_COMPLETION.json');assert cpu['status']=='complete'and cpu['count']==16 and cpu['GT_read']
    write(BASE/'POSTSEAL_STATUS002.json',dict(status='complete_pending_independent_root_view_public_archive',stage='root_handoff',count=16,GT_read=True),mutable=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(BASE/'POSTSEAL_FAILURE002.json',dict(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),GT_read=False),mutable=True);raise
