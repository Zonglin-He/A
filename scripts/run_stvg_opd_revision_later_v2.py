"""Dispatch existing matched-arm later implementations into the HC2 revision."""
import sys,traceback,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import *

def run(job,*args):
    activate()
    if job=='GPU':
        import scripts.run_stvg_opd_paper_components_v2 as m;m.run(args[0],len(args)>1 and args[1]=='qualification')
    elif job=='score':
        import scripts.score_stvg_opd_later_phase_v1 as m;m.run(args[0])
    elif job=='finalize':
        import scripts.finalize_stvg_opd_later_phase_v1 as m;m.run(args[0])
    else:raise ValueError(job)

if __name__=='__main__':
    try:run(*sys.argv[1:])
    except BaseException:
        d=BASE/'later_failures'/str(time.time_ns());d.mkdir(parents=True,exist_ok=True);(d/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'LATER_CPU_STAGE.json',dict(status='failed_preserved',evidence=str(d.relative_to(ROOT)),time=time.time()));raise
