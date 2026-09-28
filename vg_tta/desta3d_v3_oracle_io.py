"""Write-once source oracle registration and measured serial GPU allocations."""
from contextlib import contextmanager
from pathlib import Path
import fcntl,os,shutil,time,traceback
from vg_tta.external_qualification_io import ROOT,PANEL,sha,read,write,check_pins,local_dependencies,tensor_sha
OUT=ROOT/'artifacts/desta3d_v3/latent_oracle_v1'
PROTOCOL=ROOT/'protocols/desta3d_v3_latent_oracle_v1.md'

def total_prior():
    seconds=39977.51307785203
    for name in ('external_privileged_opd_v1','external_privileged_opd_v2','latent_oracle_v1'):
        for p in (ROOT/'artifacts/desta3d_v3'/name).glob('*/RECEIPT.json'):seconds+=read(p)['seconds']
    return seconds

def register_base(name,cfg,rows,paths):
    dest=OUT/name
    if dest.exists():raise ValueError('Oracle run is write-once')
    pre=OUT/'CPU_PREFLIGHT.json';assert read(pre)['status']=='passed';check_pins(read(pre)['pins'])
    write(dest/'CONFIG.json',{**cfg,'queries':len(rows),'parents':len({r['source'] for r in rows}),
        'source_GT_in_worker':True,'source_GT_purpose':'oracle latent masks; not forced decoder tokens','target_input':False,
        'target_GT':False,'optimizer_steps':0,'minimum_free_bytes':8*2**30,'cumulative_cap':None})
    write(dest/'INPUTS.json',rows)
    allpaths=local_dependencies([Path(__file__),PROTOCOL,pre,dest/'CONFIG.json',dest/'INPUTS.json',*paths])
    write(dest/'LOCK.json',{'pins':{str(p):sha(p) for p in allpaths}})
    write(dest/'REGISTRATION.json',{'time':time.time(),'status':'registered_before_GPU','source_training_exposed':True,
        'oracle_not_unlabeled_TTA':True,'no_outcome_selection':True})
    return dest

@contextmanager
def allocation(dest):
    cfg=read(dest/'CONFIG.json');check_pins(read(dest/'LOCK.json')['pins'])
    if (dest/'STARTED.json').exists():raise ValueError('No silent replay')
    lease=(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prior=total_prior();start=time.monotonic();status='running'
    try:
        write(dest/'STARTED.json',{'time':time.time(),'pid':os.getpid(),'prior_seconds':prior})
        def guard():
            assert shutil.disk_usage(ROOT).free>=cfg['minimum_free_bytes'],'8GiB disk floor'
            assert time.monotonic()-start<cfg['phase_seconds'],'Engineering phase limit'
        guard();yield cfg,guard;guard();status='completed'
    except BaseException as exc:
        status='failed';write(dest/'FAILURE.json',{'error':repr(exc),'traceback':traceback.format_exc()});raise
    finally:
        seconds=time.monotonic()-start
        write(dest/'RECEIPT.json',{'status':status,'seconds':seconds,'prior_seconds':prior,'cumulative_seconds':prior+seconds,
            'cap':None,'includes':'imports/loading/inference/finalization; CPU pre-GPU hash validation separate'})
        lease.close()

def seal(dest,predictions):
    files=sorted(p for p in (dest/'episodes').rglob('*') if p.is_file())
    write(dest/'PREDICTIONS_SEAL.json',{'files':{str(p.relative_to(dest)):sha(p) for p in files},'predictions':predictions,
        'source_GT_read_for_oracle':True,'score_computed':False,'target_read':False})
    return sha(dest/'PREDICTIONS_SEAL.json')

def verify_seal(dest):
    done=read(dest/'COMPLETE.json');assert done['seal_sha']==sha(dest/'PREDICTIONS_SEAL.json')
    data=read(dest/'PREDICTIONS_SEAL.json');check_pins({str(dest/p):h for p,h in data['files'].items()})
    return done,data
