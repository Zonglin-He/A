"""Dispatch unchanged deployment/CPU implementations under explicit new locks."""
import sys,traceback,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import *

def run(job,*args):
    activate()
    if job=='GPU':
        import scripts.run_stvg_opd_paper_v1 as m
        m.run(args[0],len(args)>1 and args[1]=='qualification')
    elif job=='P0_score':
        import scripts.score_stvg_opd_paper_v1 as m;m.run(args[0])
    elif job=='P0_finalize':
        import scripts.finalize_stvg_opd_p0_v1 as m;m.run()
    elif job=='P1_score':
        import scripts.score_stvg_opd_p1_v1 as m;m.run(args[0])
    elif job=='P1_assemble':
        import scripts.assemble_stvg_opd_table1_v1 as m;m.run()
    elif job=='P1_finalize':
        import scripts.finalize_stvg_opd_table1_v1 as m;m.run()
    else:raise ValueError(job)

if __name__=='__main__':
    try:run(*sys.argv[1:])
    except BaseException:
        d=BASE/'failures'/str(time.time_ns());d.mkdir(parents=True,exist_ok=True);(d/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'STATUS.json',dict(status='failed_preserved',evidence=str(d.relative_to(ROOT)),time=time.time()));raise
