"""Pin-only Student manifest repair; the v1 interfaces and scientific recipe remain intact."""
from scripts.stvg_qwen35_geometry_adapters_v1 import *
from scripts import stvg_qwen35_geometry_adapters_v1 as original
REQUIRED_STUDENT_CODE = (
    'scripts/stvg_soft_ep_coordinate_fit002.py',
    'scripts/audit_stvg_spatial_lite_fit002.py',
    'scripts/review_stvg_lite_complete002.py',
)
def bind_required_student_code(runtime, root=ROOT):
    repaired=copy.deepcopy(runtime)
    for name in REQUIRED_STUDENT_CODE:
        digest=sha(root/name)
        if name in repaired['pins'] and repaired['pins'][name]!=digest:
            raise RuntimeError('existing Student implementation hash mismatch: '+name)
        repaired['pins'][name]=digest
    return repaired

def prepare_student():
    original.prepare_student()
    path=BASE/'arms/QWEN_OPD/trials/G32/RUNTIME.json'
    old=read(path)
    write(BASE/'recovery/student_manifest002/ORIGINAL_RUNTIME.json',old)
    repaired=bind_required_student_code(old)
    for p in [Path(__file__),ROOT/'scripts/stvg_qwen35_geometry_run_v2.py',BASE/'EXECUTION_RUNTIME002.json',BASE/'MATCHED_BASELINES002.json']:
        repaired['pins'][str(p.relative_to(ROOT))]=sha(p)
    write(path,repaired,mutable=True)
    for p,h in repaired['pins'].items():
        assert sha(ROOT/p)==h,p
    write(BASE/'STUDENT_MANIFEST_COMPLETION002.json',dict(status='pass',count=32,
        required_fit_and_audit_modules_bound=True,original_runtime_sha256=sha(BASE/'recovery/student_manifest002/ORIGINAL_RUNTIME.json'),
        runtime_sha256=sha(path),changed_model_or_recipe_or_acceptance=False,new_Student_fits=0,GT_read=False))
