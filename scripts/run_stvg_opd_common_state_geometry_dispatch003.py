"""Saved binding repair, original per-stage seed, and full preaudit witness."""
import ast,copy,inspect,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.run_stvg_opd_common_state_geometry_v1 as target
from scripts.decota_matrix_common_v1 import read,write,sha,save,status
BASE=target.BASE;REC=BASE/'GPU_recorder_dispatch003'

def setup():
    runtime=read(REC/'REVISION_RUNTIME.json');original_verify=target.verify
    def verify():
        old=original_verify()
        for p,h in runtime['pins'].items():assert sha(ROOT/p)==h,p
        return old
    target.verify=verify;target.__file__=str(Path(__file__))
    import vg_tta.stvg_opd_common_state_geometry_v1 as old
    from vg_tta.stvg_opd_common_state_geometry_hooks002 import Geometry
    old.Geometry=Geometry
    import scripts.run_decota_paper_main_v1 as common
    model_for=common.model_for
    def stage_model(source):
        import torch,numpy as np
        # Old scientific stages were separate seeded processes. Reset only
        # model construction seed, not a fit/sample RNG or saved state.
        torch.manual_seed(20260920);np.random.seed(20260920)
        return model_for(source)
    common.model_for=stage_model
    # Existing immutable first-launch receipt is retained. A new launch has
    # its own independently hash-bound scheduling receipt.
    s=inspect.getsource(target.priority_lease)
    s=s.replace("BASE/'PRIORITY_GPU_LEASE_RECEIPT.json'","REC/'PRIORITY_GPU_LEASE_RECEIPT.json'")
    ns=dict(target.__dict__);ns['REC']=REC;exec(compile(s,str(Path(__file__)),'exec'),ns)
    target.priority_lease=ns['priority_lease']
    s=inspect.getsource(target.fit_arm)
    s=s.replace('from vg_tta.stvg_opd_p3_scope_precision001 import Installation','from vg_tta.stvg_opd_p4_joint_precision001 import Installation')
    needle="        if arm=='Full':\n";assert s.count(needle)==1
    s=s.replace(needle,"        save(REC/'actual_complete_witnesses'/str(time.time_ns())/(arm+'.pt'),result)\n"+needle)
    ns=dict(target.__dict__);ns['REC']=REC;exec(compile(s,str(Path(__file__)),'exec'),ns)
    target.fit_arm=ns['fit_arm']
    original_save=target.save
    def tagged_save(path,value):
        if isinstance(value,dict) and 'fit' in value:
            value={**value,'execution_revision_runtime_sha256':sha(REC/'REVISION_RUNTIME.json')}
        return original_save(path,value)
    target.save=tagged_save
    return verify()

if __name__=='__main__':
    job=sys.argv[1]
    try:
        setup()
        if job in ['qualify','formal']:target.gpu_run(job=='qualify')
        elif job=='qualification_root':target.qualification_root()
        elif job=='controller':target.controller()
        else:raise ValueError(job)
    except BaseException:
        p=REC/'failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=True);(p/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'CONTROLLER_STATUS.json',dict(status='failed_preserved_pending_actual_root',stage=job,evidence=str(p.relative_to(ROOT)),experiment_complete=False,time=time.time()));raise
