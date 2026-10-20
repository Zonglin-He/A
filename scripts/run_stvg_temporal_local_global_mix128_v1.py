"""One predeclared cached local/global posterior mixture; no model or GT input."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,time,fcntl,hashlib,signal
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=R/'artifacts/stvg_temporal_local_global_mix128_v1'
P=R/'results/stvg_error_state/2026-10-20/temporal_local_global_mix128'
OLD=R/'artifacts/stvg_temporal_local_motion_completion128_v1'
OP=R/'results/stvg_error_state/2026-10-20/temporal_local_motion_completion128'
BASE=['Native','HiResAuxMix','DenseLocal','DenseLocalMix']
ARMS=BASE+['LocalGlobalMix']

def tensor_sha(x):
    import numpy as np
    a=np.ascontiguousarray(x)
    return hashlib.sha256(a.tobytes()).hexdigest()

def prepare():
    old=read(OLD/'PLAN_LOCK.json');bar=read(OLD/'GLOBAL_PREDICTION_BARRIER.json')
    assert bar['count']==128 and bar['weights_and_flags_unchanged']
    assert all(sha(R/f)==h for f,h in old['pins'].items())
    assert read(OP/'ROOT_AUDIT.json')['status']=='pass'
    own=['scripts/run_stvg_temporal_local_global_mix128_v1.py','scripts/score_stvg_temporal_local_global_mix128_v1.py',
         'scripts/publish_stvg_temporal_local_global_mix128_v1.py','protocols/stvg_temporal_local_global_mix128_v1.md',
         'scripts/run_stvg_joint_event_critic_v1.py','vg_tta/stvg_posterior_decision_tree_v1.py',
         'scripts/score_stvg_where_when_event128_v1.py','scripts/score_stvg_opd_p1_v1.py',
         'vg_tta/tastvg_oracle_event5_v1.py','vg_tta/tastvg_paper48_metrics_v1.py','main_method/runtime.py',
         'methods/stvg_referdino_opd_v1/config.json',str((OLD/'PLAN_LOCK.json').relative_to(R)),
         str((OLD/'GLOBAL_PREDICTION_BARRIER.json').relative_to(R)),str((OP/'RESULTS_PUBLIC.json').relative_to(R)),
         str((OP/'ROOT_AUDIT.json').relative_to(R)),str((OLD/'COST.json').relative_to(R))]
    pins={f:sha(R/f)for f in own}
    for receipt in bar['files']:
        assert sha(R/receipt['path'])==receipt['sha256'];pins[receipt['path']]=receipt['sha256']
    p=dict(records=old['records'],count=128,seeds=[41,42,43],arms=ARMS,primary='LocalGlobalMix',
        mixture=dict(HiResAuxMix=.5,DenseLocal=.5),CPU_seconds_cap=900,GPU_seconds_cap=0,
        created_epoch=time.time(),deadline_epoch=time.time()+14400,pins=pins,
        source_checkpoint=old['source_checkpoint'],source_checkpoint_sha256=old['source_checkpoint_sha256'],
        spatial_frozen=True,no_new_model_forward=True,no_OPD=True,old_GPU_cost_not_deployment_latency=read(OLD/'COST.json')['GPU_seconds'])
    put(D/'PLAN_LOCK.json',p)
    put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','created_epoch','deadline_epoch']})
    print('locked one .5 Strong + .5 Dense combination; cached128',flush=True)

def measure():
    import numpy as np,torch
    from main_method.runtime import resource_guard
    from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
    torch.set_num_threads(4)
    def reject(event,args):
        if event=='open'and args and isinstance(args[0],(str,bytes,Path))and str(args[0]).endswith('/data/hcstvg2_official_metadata/val_v2.json'):
            raise PermissionError('cached mixture prediction forbids GT')
    sys.addaudithook(reject)
    p=read(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items())
    beg=time.perf_counter();files=[]
    for r in p['records']:
        resource_guard();assert time.perf_counter()-beg<p['CPU_seconds_cap']and time.time()<p['deadline_epoch']
        a=r['alias'];f=OLD/'predictions'/f'{a}.pt';assert sha(f)==p['pins'][str(f.relative_to(R))]
        old=load(f);q=.5*old['outputs']['HiResAuxMix']['q']+.5*old['outputs']['DenseLocal']['q']
        assert np.isfinite(q).all()and(q>=0).all()and abs(q.sum()-1)<1e-12 and not np.tril(q).any()
        out={k:old['outputs'][k]for k in BASE};out['LocalGlobalMix']=dict(q=q,interval=final_mbr(old['frame_ids'],q)['interval'])
        files.append(save(D/'predictions'/f'{a}.pt',dict(alias=a,frame_ids=old['frame_ids'],outputs=out,
            source_prediction=str(f.relative_to(R)),source_sha256=sha(f),
            frozen_spatial_hashes={k:tensor_sha(v)for k,v in old['trajectories'].items()},GT_read=False,spatial_updated=False)))
        put(D/'STATUS.json',dict(status='predicting',done=len(files),pid=os.getpid(),epoch=time.time()),True)
    assert all(sha(R/f)==h for f,h in p['pins'].items())
    put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=128,files=files,plan_sha256=sha(D/'PLAN_LOCK.json'),all_before_GT=True))
    put(D/'COST.json',dict(CPU_prediction_seconds=time.perf_counter()-beg,GPU_seconds=0,new_model_forwards=0,new_RGB=0,new_experts=0,network_training=0,spatial_updates=0,student_updates=0,not_deployment_latency=True))
    print('all128 sealed, no GT/model',flush=True)

if __name__=='__main__':
    def timeout(signum,frame):raise TimeoutError('finite cached mixture phase')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
    D.mkdir(parents=True,exist_ok=True);lock=(D/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:globals()[sys.argv[1]]()
    except BaseException as e:put(D/f'FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e)));raise
    finally:lock.close()
