"""Postseal root readback, every source, dense spatial metric and actual cases."""
import os,sys,time
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import activate,BASE,PUB,read,write,sha,verify

def run():
    activate();verify()
    assert read(BASE/'P0_CPU_COMPLETION.json')['P0_postseal_scoring_complete']
    assert read(BASE/'P0_ROOT_BYTE_READBACK.json')['status']=='pass'
    pin=read(BASE/'P0_POSTSEAL_ROOT_RUNTIME.json')
    for f,h in pin['pins'].items():assert sha(ROOT/f)==h
    from scripts import audit_stvg_opd_p0_dense_s_v1 as dense
    from scripts import diagnose_stvg_opd_p0_population_v1 as population
    from scripts import render_stvg_opd_p0_signal_chain_v1 as signals
    from scripts import render_stvg_opd_p0_cases_v1 as cases
    from scripts import audit_stvg_opd_p0_public_v1 as public
    start=time.time();dense.run();population.run();signals.run();cases.run()
    result=public.run(PUB);write(BASE/'P0_PUBLIC_SCALAR_AUDIT.json',result)
    write(BASE/'P0_ROOT_READBACK_COMPLETION.json',dict(status='pending_actual_root_visual_and_publication',
        CPU_seconds=time.time()-start,files={str(p.relative_to(ROOT)):sha(p) for p in PUB.glob('P0*') if p.is_file()},
        private_case_rendered_not_yet_root_viewed=True,paper_suite_complete=False,time=time.time()))

if __name__=='__main__':run()
