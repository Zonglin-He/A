"""Four independent CPU domain jobs; unchanged frozen MAP inference and metrics."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,time,argparse,subprocess,traceback,hashlib,json
from pathlib import Path
S=Path(__file__).resolve().parent
sys.path.insert(0,str(S))
import run as frozen
from run import D,P,R,CG,STRICT,COUNTS,BACKBONES,read,sha,atomic,identity,exited
JOBS=[(b,ds)for b in BACKBONES for ds in COUNTS]

def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
def available_memory():
    return next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))
def prepare():
    frozen.validate();assert not(D/'parallel_v2_plan.json').exists()
    p=dict(schema='map_full_parallel_controller_v2',authorization='Latest user asks CPU parallelism for the already authorized complete MAP-DP evaluation. Preserve wait-until-current-Vid-full-completes gate.',
        jobs=JOBS,max_concurrent_domain_jobs=4,compute_threads_each=1,minimum_available_memory_before_launch_bytes=8*2**30,
        science='Invoke unchanged frozen run.py predict/score/report in separate subprocesses for each domain. No algorithm, posterior, DP, metrics, parameters, seed, cohort or scientific fingerprint changes. Prediction guard never leaks into separate scoring subprocess.',
        order='After original both-CG-full seal/score/audit/named-source/ownedexit gate and original8querypreflight, four domains run concurrently. Each entire domain seals before its own official labels. All four must score/report successfully before original strict deletion queue starts.',
        failure='Keep every domain log/partial prediction/failed state; no retry, no skipped query or fallback. Independent domains finish so their evidence is preserved; any failed domain prevents strict launch.',
        max_new_Source=0,max_new_expert=0,max_new_GPU=0,scientific_plan_sha256=sha(D/'plan.json'),scientific_fingerprint=read(D/'plan.json')['fingerprint'],
        controller_code_sha256=sha(Path(__file__)),hardware=dict(logical_CPUs=os.cpu_count(),launch_affinity_CPUs=len(os.sched_getaffinity(0))),
        timing='Parallel scheduling is an engineering change; do not claim a measured speedup or completion time before actual runs. CPU cache times are not uncached deployment latency.')
    p['fingerprint']=digest(p);atomic(D/'parallel_v2_plan.json',p);atomic(P/'parallel_v2_protocol.json',p)
    (D/'source_snapshot/parallel_v2.py').write_bytes(Path(__file__).read_bytes());print(p['fingerprint'])
def validate():
    p=read(D/'parallel_v2_plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
    assert sha(Path(__file__))==p['controller_code_sha256']and sha(D/'plan.json')==p['scientific_plan_sha256'];frozen.validate();return p
def domain(b,ds):
    p=validate();frozen.require_gate();who=identity();statepath=D/f'parallel_v2_{b}_{ds}_state.json'
    try:
        for command,suffix in [('predict','sealed'),('score','score_receipt'),('report','reporting_receipt')]:
            if(D/f'{b}_{ds}_{suffix}.json').exists():continue
            atomic(statepath,dict(status=command,worker=who,controller_fingerprint=p['fingerprint']))
            with(D/f'parallel_v2_{b}_{ds}_{command}.log').open('a')as log:
                c=subprocess.Popen([sys.executable,'-B',str(S/'run.py'),command,'--backbone',b,'--dataset',ds],stdout=log,stderr=subprocess.STDOUT)
                atomic(D/f'parallel_v2_{b}_{ds}_{command}_child.json',identity(c.pid));code=c.wait()
                atomic(D/f'parallel_v2_{b}_{ds}_{command}_exit.json',dict(code=code))
                assert code==0,'Preserve failed stage; no retry/fallback'
        atomic(statepath,dict(status='complete_scored_reported',worker=who,controller_fingerprint=p['fingerprint']))
    except BaseException as e:
        atomic(statepath,dict(status='failed_preserved',error=repr(e),traceback=traceback.format_exc(),worker=who));raise
def queue():
    p=validate()
    if(D/'controller.json').exists():assert exited(read(D/'controller.json')),'Do not duplicate old or current controller'
    who=identity();atomic(D/'controller.json',who);started=time.time()
    atomic(D/'state.json',dict(status='waiting_both_CG_fulls_and_reports',controller=who,parallel_controller_fingerprint=p['fingerprint'],fingerprint=p['scientific_fingerprint'],MAP_predictions=0,new_Source=0,new_expert=0,parallel_domain_jobs=4))
    try:
        while not frozen.gate():
            assert time.time()-started<7*86400 and not read(CG/'state.json')['status'].startswith('failed')
            time.sleep(30)
        if not(D/'preflight_receipt.json').exists():
            with(D/'parallel_v2_preflight.log').open('a')as log:
                c=subprocess.Popen([sys.executable,'-B',str(S/'run.py'),'preflight'],stdout=log,stderr=subprocess.STDOUT)
                atomic(D/'parallel_v2_preflight_child.json',identity(c.pid));assert c.wait()==0,'Preflight failure prevents all prediction jobs'
        running=[]
        for b,ds in JOBS:
            while available_memory()<p['minimum_available_memory_before_launch_bytes']:
                atomic(D/'parallel_v2_resource_wait.json',dict(time=time.time(),reason='available_RAM_below_launch_floor',bytes=available_memory()));time.sleep(30)
            log=(D/f'parallel_v2_{b}_{ds}_domain.log').open('a')
            c=subprocess.Popen([sys.executable,'-B',__file__,'domain','--backbone',b,'--dataset',ds],stdout=log,stderr=subprocess.STDOUT)
            atomic(D/f'parallel_v2_{b}_{ds}_domain_child.json',identity(c.pid));running.append((b,ds,c,log))
        atomic(D/'state.json',dict(status='parallel_MAP_domains_running',controller=who,parallel_controller_fingerprint=p['fingerprint'],fingerprint=p['scientific_fingerprint'],jobs=JOBS))
        failures=[]
        for b,ds,c,log in running:
            code=c.wait();log.close()
            if code:failures.append(dict(backbone=b,dataset=ds,code=code))
        assert not failures,f'Preserved failed domains: {failures}'
        for b,ds in JOBS:
            score=read(D/f'{b}_{ds}_score_receipt.json');report=read(D/f'{b}_{ds}_reporting_receipt.json')
            assert score['queries']==COUNTS[ds]and report['score_sha256']==sha(D/f'{b}_{ds}_score_receipt.json')
        atomic(D/'state.json',dict(status='complete_scored_reported_pending_publication',controller=who,parallel_controller_fingerprint=p['fingerprint'],fingerprint=p['scientific_fingerprint']))
        assert exited(read(STRICT/'controller.json'))and read(STRICT/'state.json')['ablation_predictions']==0
        with(STRICT/'queue_after_MAP.log').open('a')as log:
            c=subprocess.Popen([sys.executable,'-B',str(R/'studies/strict_deletion_v1/run.py'),'queue'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            atomic(STRICT/'launch_after_MAP.json',dict(**identity(c.pid),MAP_plan_sha256=sha(D/'plan.json'),MAP_parallel_plan_sha256=sha(D/'parallel_v2_plan.json'),MAP_reporting_sha256={f'{b}_{ds}':sha(D/f'{b}_{ds}_reporting_receipt.json')for b,ds in JOBS}))
    except BaseException as e:
        atomic(D/'state.json',dict(status='failed_preserved',controller=who,error=repr(e),traceback=traceback.format_exc(),parallel_controller_fingerprint=p['fingerprint'],fingerprint=p['scientific_fingerprint']));raise
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','queue','domain']);a.add_argument('--backbone',choices=BACKBONES);a.add_argument('--dataset',choices=list(COUNTS));v=a.parse_args()
    dict(prepare=prepare,queue=queue,domain=lambda:domain(v.backbone,v.dataset))[v.command]()
