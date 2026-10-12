"""Actual corrected-scope Q0 runner and root; old qualifications are not reused."""
import sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_query_q0_native_scope003 import BASE,context,bind_geometry,bind_runner
context()
import scripts.run_stvg_opd_query_q0_v1 as original
from scripts.decota_matrix_common_v1 import status


if __name__=='__main__':
    job=sys.argv[1]
    try:
        bind_geometry();bind_runner(original)
        if job in ['qualify','formal']:original.run(job=='qualify')
        elif job in ['qualification_root','deployment_root']:original.root(job=='qualification_root')
        else:raise ValueError(job)
    except BaseException:
        p=BASE/'failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=True);(p/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'STATUS.json',dict(status='failed_preserved_pending_actual_root',stage=job,evidence=str(p),time=time.time()));raise
