"""Metadata-only completion of the sealed64 group, no GPU re-execution."""
from pathlib import Path
import json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_instance_evidence32_v1 import BASE as DEV,read,put,sha
BASE=DEV/'confirmation'
def prepare():
    assert read(BASE/'EXPERIMENT_GLOBAL_PREDICTION_BARRIER.json')['count']==64
    for tid in ('E0','E1'):assert read(BASE/'trials'/tid/'ROOT_READBACK.json')['status']=='pass'
    assert read(BASE/'execution/15_evaluate_all/WORKER_EXIT.json')['exit_code']==1
    assert not(BASE/'EXPERIMENT_POSTSEAL_RESULTS.json').exists()
    decision=read(DEV/'DEVELOPMENT_DECISION.json');assert decision['rule_sha256']==sha(DEV/'ADVANCEMENT_RULE_LOCK.json')
    assert read(BASE/'FROZEN_RECIPE_LOCK.json')['development_decision_sha256']==sha(DEV/'DEVELOPMENT_DECISION.json')
    p=BASE/'ADVANCEMENT_RULE_LOCK.json';assert not p.exists();p.write_bytes((DEV/'ADVANCEMENT_RULE_LOCK.json').read_bytes())
    paths=[Path(__file__),BASE/'RUNTIME.json',BASE/'ADVANCEMENT_RULE_LOCK.json',BASE/'FROZEN_RECIPE_LOCK.json',BASE/'EXPERIMENT_GLOBAL_PREDICTION_BARRIER.json',BASE/'execution/15_evaluate_all/WORKER_EXIT.json']
    for tid in ('E0','E1'):paths.extend([BASE/'trials'/tid/'ROOT_READBACK.json',BASE/'trials'/tid/'STUDENT_GLOBAL_BARRIER.json'])
    put(BASE/'recovery/final_binding002/RUNTIME.json',dict(status='locked',pins={str(p.relative_to(ROOT)):sha(p)for p in paths},GT_read=False,scope='copy exact already locked development-rule metadata for inherited CPU reader hash field; frozen64 recipe unchanged; no rule applied to select on confirmation; no source/tolerance/model/fit/prediction change',GPU_reruns=0))
def run():
    b=BASE/'recovery/final_binding002'
    for rel,h in read(b/'RUNTIME.json')['pins'].items():assert sha(ROOT/rel)==h,rel
    from scripts.run_stvg_instance_confirmation64_v1 import stage
    t=time.perf_counter();stage('evaluate','all')
    put(b/'CPU_COMPLETION.json',dict(status='complete',count=64,GT_read=True,wall_seconds=time.perf_counter()-t))
    put(BASE/'MEASUREMENT_COMPLETION.json',dict(status='complete_pending_independent_public_archive',count=64,GT_read=True,result_sha256=sha(BASE/'EXPERIMENT_POSTSEAL_RESULTS.json'),metadata_recovery_sha256=sha(b/'RUNTIME.json'),GPU_reruns=0))
if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
