"""Exact same Direct/OPD Full machinery, only prelocked HYP time; reuse identical A inputs."""
from pathlib import Path
import inspect,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_temporal_semantic_full32_v1 as old
from scripts.stvg_temporal_hypothesis32_v1 import BASE as PRIMARY,read,write,sha
BASE=PRIMARY/'full';A=old.BASE

def verify():
    for p,h in read(BASE/'EXECUTION_RUNTIME.json')['pins'].items():assert sha(ROOT/p)==h,p
    if (BASE/'ACTIVE_EXECUTION.json').exists():
        import time
        assert time.time()<read(BASE/'ACTIVE_EXECUTION.json')['deadline_epoch']
def put(p,d,replace=False):
    if Path(p).name=='EXACT_REFERENCE_REUSE_LOCK.json':
        aj={j['job_id']:j for j in read(A/'INPUT_JOB_LOCK.json')['records']};jj=read(BASE/'INPUT_JOB_LOCK.json')['records'];found={r['job_id']:r for r in d['records']}
        def identity(j):return {k:j[k] for k in ['job_id','case_id','dataset','slot','position','physical_frame_id','physical_timestamp','original_image_shape','original_pixel_sha256','query_text','source_case_sha256']}|{'image_sha256':j['image']['sha256'],'context':{k:v for k,v in j['context'].items() if k not in ['path','bytes']}}
        assert read(A/'ROOT_READBACK.json')['status']=='pass'
        for j in jj:
            if j['job_id'] in aj and identity(j)==identity(aj[j['job_id']]):
                p0=A/'teacher'/(j['job_id']+'.json');found[j['job_id']]=dict(job_id=j['job_id'],path=str(p0.relative_to(ROOT)),sha256=sha(p0),complete_input_identity=True,from_A_complete_root=True,GT_read=False)
        d=dict(d,records=list(found.values()),count=len(found),A_full_root_sha256=sha(A/'ROOT_READBACK.json'))
    return write(p,d,mutable=replace)
def run(stage):
    if stage=='prepare':assert read(PRIMARY/'TEACHER_ROOT_READBACK.json')['informative_count']>0
    src=inspect.getsource(getattr(old,stage)).replace('SEM_STUDENT','HYP_STUDENT').replace("['SEM']","['HYP']").replace('semantic_view_full32_','hypothesis_full32_')
    ns=dict(old.__dict__);ns.update(BASE=BASE,PRIMARY=PRIMARY,verify=verify,put=put);exec(compile(src,__file__,'exec'),ns);return ns[stage]()
if __name__=='__main__':
    stage=sys.argv[1]
    if stage in ['worker','root']:
        src=inspect.getsource(old.student).replace('semantic_view_full32_','hypothesis_full32_');ns=dict(old.__dict__);ns.update(BASE=BASE,PRIMARY=PRIMARY,verify=verify,put=put);exec(compile(src,__file__,'exec'),ns);ns['student'](stage)
    else:run(stage)
