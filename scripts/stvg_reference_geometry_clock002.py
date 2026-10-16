"""Audit the actual development Native teacher's exact-reuse provenance.

The Lite Native directory stores references only. Its original job lock is in
the already completed component Native study; this is not a new teacher run.
"""
from pathlib import Path
import inspect, sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_reference_geometry_audit_v1 as audit

def main():
    source=inspect.getsource(audit.audit_clocks)
    needle="jp=base/'TEACHER_JOB_LOCK.json';payload_files[str(jp.relative_to(ROOT))]=sha(jp)"
    replacement="""if cohort=='development' and clock=='native':
                receipt=read(base/'ROOT_REFERENCE_READBACK.json')
                original=ROOT/'artifacts/stvg_component_ablation128_v1/native_clock/revision002'
                assert receipt['status']=='pass' and receipt['count']==128 and receipt['GT_read']is False
                assert receipt['original_job_lock_sha256']==sha(original/'TEACHER_JOB_LOCK.json')
                assert receipt['original_teacher_barrier_sha256']==sha(original/'TEACHER_GLOBAL_BARRIER.json')
                payload_files[str((base/'ROOT_REFERENCE_READBACK.json').relative_to(ROOT))]=sha(base/'ROOT_REFERENCE_READBACK.json')
                base=original
            jp=base/'TEACHER_JOB_LOCK.json';payload_files[str(jp.relative_to(ROOT))]=sha(jp)"""
    assert source.count(needle)==1
    scope=dict(audit.__dict__);exec(compile(source.replace(needle,replacement),str(Path(__file__)), 'exec'),scope)
    scope['audit_clocks']()

if __name__=='__main__':main()
