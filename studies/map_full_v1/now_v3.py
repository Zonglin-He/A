"""User-authorized CPU MAP/DP replay now, alongside untouched CG-Vid inference."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,time,json,hashlib,argparse,subprocess,traceback,gc,signal,types,shutil
from pathlib import Path
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S))
import run as frozen
from run import R,RUN,CG,STRICT,COUNTS,BACKBONES,ARMS,read,sha,atomic,identity,exited
D=R/'outputs/map_full_now_v3';P=R/'results/map_full_now_v3'
JOBS=[(b,ds)for b in BACKBONES for ds in COUNTS]

def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
def memory():return next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))
def prepare():
    frozen.validate();assert not(D/'plan.json').exists()
    p=dict(schema='map_full_now_v3',authorization='Latest direct user explicitly authorizes CPU-parallel MAP-DP replay NOW without waiting for CG-Vid, while prioritizing and not interrupting current CG-Vid.',
        scientific_parent_fingerprint=read(frozen.D/'plan.json')['fingerprint'],scientific_parent_plan_sha256=sha(frozen.D/'plan.json'),
        science='Unchanged frozen.calculate/map_core q1/MAP/DP math and frozen score/report code objects, with this independent output namespace and per-domain readiness gate. Same six arms, cohorts, profiles and bootstrap seed. No Source/expert/RGB/GPU/optimizer work.',
        jobs=JOBS,arms=ARMS,domains=COUNTS,max_compute_jobs=4,compute_threads_each=1,
        scheduling=dict(nice=15,ionice='idle class3',CPU_affinity=[12,13,14,15],minimum_available_RAM_bytes=8*2**30,resume_RAM_bytes=10*2**30,slowdown_guard_ratio=1.35,slowdown_new_queries=25,minimum_pause_seconds=60),
        readiness='TA both parent fulls and CG-HC are complete. CG-Vid consumes only atomically committed immutable receipt+compact+prediction SHA triples, never active Source scratch or partial outputs. No CG-Vid GT until all10303 new MAP predictions seal AND originalCG-Vid full seals and scores; then verify all streamed parent SHAs against final original seal.',
        validation='8committed-cache CPU bridges before jobs, FullMBR bitwise parent; all new output arrays cold-read exact. Each complete domain seal before its own labels. Frozen score/reporter source bytes unchanged, rebound only output paths/validate/readiness. All signed positive/negative/severeunion/shortlong/CI/workerCPUcost preserved.',
        resources='No changes/signals/affinity/nice/priority to original CG controller/predictor/teacher or unrelated services. Only own MAP process groups may be temporarily SIGSTOP/CONT on lowRAM or precautionary Vid pipeline slowdown; this is a resource guard, not evidence of causal slowdown. Low priority still cannot guarantee zero measurable shared-host interference.',
        strict='Only after four complete MAP reports and original bothCGfull audit/namedsource/ownedexit gate, launch unchanged strict deletion queue. Never start its GPU reconstruction during CG-Vid.',
        budget=dict(new_Source=0,new_expert=0,new_GPU=0,optimizer_steps=0,max_stage_seconds=259200,min_disk_free_bytes=60*2**30),
        code_sha256={str(Path(__file__).relative_to(R)):sha(Path(__file__))},parent_seals={ds:sha(RUN/ds/'sealed.json')for ds in COUNTS},CG_plan_sha256=sha(CG/'plan.json'),
        exposure='Both full datasets historically exposed. This changes final readout and actual final spatialDP only, not parameter selection or originalSource readout semantics.')
    p['fingerprint']=digest(p);atomic(D/'plan.json',p);atomic(P/'protocol.json',p)
    (D/'source_snapshot').mkdir(parents=True,exist_ok=True);(D/'source_snapshot/now_v3.py').write_bytes(Path(__file__).read_bytes())
    atomic(D/'state.json',dict(status='prepared_CPU_now',fingerprint=p['fingerprint']));print(p['fingerprint'])
def validate():
    p=read(D/'plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
    for n,h in p['code_sha256'].items():assert sha(R/n)==h,n
    assert sha(frozen.D/'plan.json')==p['scientific_parent_plan_sha256']and sha(CG/'plan.json')==p['CG_plan_sha256']
    for ds,h in p['parent_seals'].items():assert sha(RUN/ds/'sealed.json')==h
    frozen.validate();return p
def rows(ds):return frozen.rows(ds)
def parent_entry(b,ds,row,seal=None):
    key=row['id']
    if b=='TA':return frozen.parent(b,ds,key,seal or read(RUN/ds/'sealed.json'))
    rp=CG/ds/'receipts'/(key+'.json')
    if not rp.exists():return None
    raw=rp.read_bytes();rec=json.loads(raw);rh=hashlib.sha256(raw).hexdigest()
    expected=digest(dict(id=key,dataset=row['dataset'],metadata={k:v for k,v in row['metadata'].items()if k!='frame_ids'}))
    assert rec['fingerprint']==read(CG/'plan.json')['fingerprint']and rec['input_binding']==expected and not rec['GT_read']and rec['all_arrays_cold_replay_exact']
    cp=CG/rec['compact_path'];pp=CG/rec['prediction_path'];assert sha(cp)==rec['compact_sha256']and sha(pp)==rec['prediction_sha256']and sha(rp)==rh
    return dict(capture_path=str(cp),capture_sha256=rec['compact_sha256'],prediction_path=str(pp),prediction_sha256=rec['prediction_sha256'],receipt_path=str(rp),receipt_sha256=rh)
def origin_matches_seal(b,ds,files):
    path=RUN/ds/'sealed.json'if b=='TA'else CG/f'{ds}_sealed.json';seal=read(path)
    assert seal['queries']==COUNTS[ds]and not seal['GT_read']
    for key,e in files.items():
        original=frozen.parent(b,ds,key,seal);assert e['parent']==original
    return sha(path)
def score_gate(b,ds):
    path=D/f'{b}_{ds}_sealed.json';assert path.exists()and read(path)['queries']==COUNTS[ds]
    if b=='CG':
        sp=CG/f'{ds}_sealed.json';sc=CG/f'{ds}_score_receipt.json';assert sp.exists()and sc.exists()
        v=read(sc);assert v['whole_domain_sealed_before_GT']and v['queries']==COUNTS[ds]and v['seal_sha256']==sha(sp)
    else:assert(RUN/ds/'score_receipt.json').exists()
def calculate(b,ds,origin):return frozen.calculate(b,ds,origin)
def preflight():
    p=validate()
    import torch,numpy as np
    from stvg.runtime import guard
    sys.addaudithook(guard);torch.set_num_threads(1);proof=[]
    for b,ds in JOBS:
        for row in rows(ds)[:2]:
            o=parent_entry(b,ds,row);assert o is not None
            a=calculate(b,ds,o);again=calculate(b,ds,o);assert all(np.array_equal(a[k],again[k])for k in a)
            if b=='TA':
                from cached import ARMS as names
                cp=R/'outputs/paper_review_suite_v1/cached'/ds/(row['id']+'.npz');rc=read(cp.with_suffix('.json'));assert sha(cp)==rc['sha256']
                with np.load(cp)as z:assert np.array_equal(a['FullMAP_interval'],z['intervals'][names.index('P1MAP')])
            proof.append(dict(backbone=b,dataset=ds,FullMBR_parent_exact=True,MAP_interval_and_DP_replay_exact=True,committed_parent_sha=o))
    assert not torch.cuda.is_initialized()
    atomic(D/'preflight_receipt.json',dict(status='passed',queries=8,proof=proof,fingerprint=p['fingerprint'],new_Source=0,new_expert=0,new_GPU=0,GT_read=False))
    atomic(P/'preflight_receipt.json',dict(status='passed',queries=8,FullMBR_and_MAP_DP_replay_exact=True,TA_MAP_matches_prior_control=True,new_Source=0,new_expert=0,new_GPU=0,GT_read=False,fingerprint=p['fingerprint']))
def predict(b,ds):
    p=validate()
    import torch,numpy as np
    from stvg.runtime import guard
    sys.addaudithook(guard);torch.set_num_threads(1);who=identity();dest=D/b/ds;files={};started=time.perf_counter();rr=rows(ds)
    atomic(D/f'{b}_{ds}_worker.json',who)
    for i,row in enumerate(rr,1):
        assert time.perf_counter()-started<p['budget']['max_stage_seconds']and shutil.disk_usage(R).free>p['budget']['min_disk_free_bytes']
        key=row['id'];rp=dest/'receipts'/(key+'.json');out=dest/'predictions'/(key+'.npz')
        if rp.exists():
            e=read(rp);assert e['fingerprint']==p['fingerprint']and sha(out)==e['sha256'];files[key]=dict(e,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp));continue
        assert not out.exists(),'Preserve unbound output, do not overwrite'
        origin=parent_entry(b,ds,row)
        while origin is None:
            assert b=='CG'and ds=='vidstg'and not read(CG/'state.json')['status'].startswith('failed'),'Parent failure preserved; no speculative Source or fallback'
            atomic(D/f'{b}_{ds}_progress.json',dict(completed=len(files),total=COUNTS[ds],worker=who,status='waiting_next_original_committed_query',GT_read=False));time.sleep(10);origin=parent_entry(b,ds,row)
        tick=time.perf_counter();arrays=calculate(b,ds,origin);out.parent.mkdir(parents=True,exist_ok=True)
        with out.with_suffix('.tmp').open('wb')as f:np.savez_compressed(f,**arrays)
        os.replace(out.with_suffix('.tmp'),out)
        with np.load(out)as z:assert all(np.array_equal(z[k],arrays[k])for k in arrays)
        e=dict(path=str(out.relative_to(D)),sha256=sha(out),parent=origin,worker=who,query_CPU_seconds=time.perf_counter()-tick,FullMBR_parent_bitwise=True,MAP_recomputed_spatialDP=True,new_Source=0,new_expert=0,new_GPU=0,GT_read=False,fingerprint=p['fingerprint'])
        atomic(rp,e);files[key]=dict(e,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp))
        atomic(D/f'{b}_{ds}_progress.json',dict(completed=i,total=COUNTS[ds],worker=who,status='CPU_predicting',seconds=time.perf_counter()-started,GT_read=False));print(b,ds,i,flush=True)
        del arrays;gc.collect()
    while b=='CG'and not(CG/f'{ds}_sealed.json').exists():time.sleep(5)
    parent_sha=origin_matches_seal(b,ds,files);assert len(files)==COUNTS[ds]and not torch.cuda.is_initialized()
    atomic(D/f'{b}_{ds}_sealed.json',dict(queries=len(files),files=files,parent_seal_sha256=parent_sha,worker=who,seconds=time.perf_counter()-started,GT_read=False,fingerprint=p['fingerprint']))
def rebound(command,b,ds):
    p=validate();score_gate(b,ds)
    # Identical original code object; only output namespace/readiness/validation
    # are rebound to the new user-authorized study. No gate/hash is waived.
    original=frozen.score if command=='score'else frozen.reporting
    namespace=dict(frozen.__dict__,D=D,P=P,validate=validate,require_gate=lambda:score_gate(b,ds))
    fn=types.FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    assert fn.__code__ is original.__code__;fn(b,ds)
    atomic(D/f'{b}_{ds}_{command}_binding_receipt.json',dict(frozen_code_object_exact=True,frozen_source_sha256=sha(S/'run.py'),new_namespace=str(D),fingerprint=p['fingerprint']))
def domain(b,ds):
    p=validate();who=identity();path=D/f'{b}_{ds}_state.json'
    try:
        for command,suffix in [('predict','sealed'),('score','score_receipt'),('report','reporting_receipt')]:
            if(D/f'{b}_{ds}_{suffix}.json').exists():continue
            while command!='predict'and b=='CG'and not(CG/f'{ds}_score_receipt.json').exists():
                atomic(path,dict(status='waiting_parent_full_score_no_GT',worker=who));assert not read(CG/'state.json')['status'].startswith('failed');time.sleep(10)
            atomic(path,dict(status=command,worker=who,fingerprint=p['fingerprint']))
            with(D/f'{b}_{ds}_{command}.log').open('a')as log:
                c=subprocess.Popen([sys.executable,'-B',__file__,command,'--backbone',b,'--dataset',ds],stdout=log,stderr=subprocess.STDOUT)
                atomic(D/f'{b}_{ds}_{command}_child.json',identity(c.pid));code=c.wait();atomic(D/f'{b}_{ds}_{command}_exit.json',dict(code=code));assert code==0,'Preserve error; no retry/fallback'
        atomic(path,dict(status='complete_scored_reported',worker=who,fingerprint=p['fingerprint']))
    except BaseException as e:
        atomic(path,dict(status='failed_preserved',worker=who,error=repr(e),traceback=traceback.format_exc()));raise
def vid_rate():
    paths=sorted((CG/'vidstg/receipts').glob('*.json'),key=lambda x:x.stat().st_mtime)[-50:];rr=[read(x)for x in paths]
    vv=[x['pipeline_seconds']for x in rr if not x['acquisition']['acquisition_reused']and x['acquisition']['Source']['Source_forwards']==1]
    return dict(completed=read(CG/'vidstg_progress.json')['completed'],queries=len(vv),mean_pipeline_seconds=sum(vv)/len(vv))
def group_signal(w,sig):
    if not exited(w):
        assert os.getpgid(w['pid'])==w['pid'];os.killpg(w['pid'],sig)
def queue():
    p=validate();assert not(D/'controller.json').exists(),'No duplicate launch'
    who=identity();atomic(D/'controller.json',who);atomic(D/'state.json',dict(status='CPU_preflight_running_alongside_Vid',controller=who,fingerprint=p['fingerprint']))
    baseline=vid_rate();atomic(D/'Vid_before_CPU_baseline.json',baseline);running=[];paused=False;pause_at=0.;slow_last=baseline['completed']
    try:
        with(D/'preflight.log').open('a')as log:
            c=subprocess.Popen([sys.executable,'-B',__file__,'preflight'],stdout=log,stderr=subprocess.STDOUT);atomic(D/'preflight_child.json',identity(c.pid));assert c.wait()==0
        for b,ds in JOBS:
            while memory()<p['scheduling']['minimum_available_RAM_bytes']:time.sleep(10)
            log=(D/f'{b}_{ds}_domain.log').open('a');c=subprocess.Popen([sys.executable,'-B',__file__,'domain','--backbone',b,'--dataset',ds],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            w=identity(c.pid);atomic(D/f'{b}_{ds}_domain_child.json',w);running.append((b,ds,c,log,w))
        atomic(D/'state.json',dict(status='CPU_parallel_running_alongside_Vid',controller=who,fingerprint=p['fingerprint'],jobs=JOBS))
        while any(c.poll()is None for _,_,c,_,_ in running):
            rate=vid_rate();low=memory()<p['scheduling']['minimum_available_RAM_bytes'];slow=rate['completed']-slow_last>=p['scheduling']['slowdown_new_queries']and rate['mean_pipeline_seconds']>baseline['mean_pipeline_seconds']*p['scheduling']['slowdown_guard_ratio']
            if not paused and(low or slow):
                for _,_,_,_,w in running:group_signal(w,signal.SIGSTOP)
                paused=True;pause_at=time.time();slow_last=rate['completed']
                atomic(D/'resource_guard.json',dict(time=time.time(),status='MAP_own_groups_paused',reason='RAM'if low else'Vid_rate_precaution',baseline=baseline,current=rate))
            elif paused and memory()>p['scheduling']['resume_RAM_bytes']and time.time()-pause_at>p['scheduling']['minimum_pause_seconds']:
                for _,_,_,_,w in running:group_signal(w,signal.SIGCONT)
                paused=False;slow_last=rate['completed'];atomic(D/'resource_guard.json',dict(time=time.time(),status='MAP_own_groups_resumed',current=rate))
            if not slow and rate['completed']-slow_last>=p['scheduling']['slowdown_new_queries']:slow_last=rate['completed']
            time.sleep(5)
        failures=[]
        for b,ds,c,log,w in running:
            log.close()
            if c.returncode:failures.append(dict(backbone=b,dataset=ds,code=c.returncode))
        assert not failures,f'Preserved failed domains: {failures}'
        atomic(D/'state.json',dict(status='complete_MAP_waiting_original_CG_audits_before_strict',controller=who,fingerprint=p['fingerprint']))
        while not frozen.gate():time.sleep(15)
        for b,ds in JOBS:
            v=read(D/f'{b}_{ds}_reporting_receipt.json');assert v['queries']==COUNTS[ds]and v['score_sha256']==sha(D/f'{b}_{ds}_score_receipt.json')
        atomic(D/'state.json',dict(status='complete_scored_reported_pending_publication',controller=who,fingerprint=p['fingerprint']))
        assert exited(read(STRICT/'controller.json'))and read(STRICT/'state.json')['ablation_predictions']==0
        with(STRICT/'queue_after_MAP.log').open('a')as log:
            c=subprocess.Popen([sys.executable,'-B',str(R/'studies/strict_deletion_v1/run.py'),'queue'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            atomic(STRICT/'launch_after_MAP.json',dict(**identity(c.pid),MAP_now_plan_sha256=sha(D/'plan.json'),MAP_now_reporting_sha256={f'{b}_{ds}':sha(D/f'{b}_{ds}_reporting_receipt.json')for b,ds in JOBS}))
    except BaseException as e:
        if paused:
            for _,_,_,_,w in running:group_signal(w,signal.SIGCONT)
        atomic(D/'state.json',dict(status='failed_preserved',controller=who,error=repr(e),traceback=traceback.format_exc(),fingerprint=p['fingerprint']));raise
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','queue','preflight','predict','score','report','domain']);a.add_argument('--backbone',choices=BACKBONES);a.add_argument('--dataset',choices=list(COUNTS));v=a.parse_args()
    dict(prepare=prepare,queue=queue,preflight=preflight,predict=lambda:predict(v.backbone,v.dataset),score=lambda:rebound('score',v.backbone,v.dataset),report=lambda:rebound('report',v.backbone,v.dataset),domain=lambda:domain(v.backbone,v.dataset))[v.command]()
