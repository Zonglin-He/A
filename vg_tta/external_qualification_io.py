"""Write-once registration, serial leases, hashes and measured allocation receipts."""
from contextlib import contextmanager
from pathlib import Path
import ast,fcntl,hashlib,json,os,shutil,time,traceback
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/desta3d_v3/external_privileged_opd_v2'
DOWNLOAD=ROOT/'artifacts/desta3d_v3/external_privileged_opd_v1'
CK=ROOT/'checkpoints/LLaVA-ST-Qwen2-7B'
OFFICIAL=ROOT/'external/LLaVA-ST'
SIGLIP=ROOT/'checkpoints/llava_official_siglip-so400m-patch14-384'
PANEL=ROOT/'artifacts/desta3d_v2/tta_v2/source_task_control_v2'
PROTOCOL=ROOT/'protocols/desta3d_v3_external_privileged_opd_v2.md'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def check_pins(pins):
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('Pinned input changed: '+p)
def prior_seconds():
    return 39977.51307785203+sum(read(p)['seconds'] for parent in (OUT,DOWNLOAD) for p in parent.glob('*/RECEIPT.json'))
def local_dependencies(paths):
    found={p.resolve() for p in paths};todo=list(found)
    while todo:
        path=todo.pop()
        if path.suffix!='.py' or not path.is_relative_to(ROOT):continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.Import):names=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module:names=[node.module]+[node.module+'.'+a.name for a in node.names]
            else:continue
            for name in names:
                if not name.startswith(('vg_tta.','scripts.')):continue
                dep=ROOT/(name.replace('.','/')+'.py')
                if dep.is_file() and dep not in found:found.add(dep);todo.append(dep)
    return sorted(found)
def register_base(name,config,rows,paths):
    dest=OUT/name
    if dest.exists():raise ValueError('Run is write-once: '+str(dest))
    assert len(rows)==len({r['source'] for r in rows})
    config={**config,'name':name,'queries':len(rows),'parents':len(rows),'optimizer_steps':0,'target_input':False,
        'source_GT_in_worker':False,'target_GT':False,'cumulative_cap':None,'minimum_free_bytes':8*2**30}
    preflight=OUT/'CPU_PREFLIGHT.json';assert read(preflight)['status']=='passed'
    check_pins(read(preflight)['pins'])
    write(dest/'CONFIG.json',config);write(dest/'INPUTS.json',rows)
    allpaths=local_dependencies([Path(__file__),PROTOCOL,preflight,dest/'CONFIG.json',dest/'INPUTS.json',*paths])
    write(dest/'LOCK.json',{'pins':{str(p.resolve()):sha(p) for p in allpaths}})
    write(dest/'REGISTRATION.json',{'time':time.time(),'status':'registered_before_GPU','Q0':'engineering/source-training overlap possible','no_GT_selection':True})
    return dest

@contextmanager
def allocation(dest):
    cfg=read(dest/'CONFIG.json');check_pins(read(dest/'LOCK.json')['pins'])
    if (dest/'STARTED.json').exists():raise ValueError('Never silently replay a started allocation')
    lease=(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prior=prior_seconds();start=time.monotonic();status='running'
    try:
        write(dest/'STARTED.json',{'time':time.time(),'pid':os.getpid(),'prior_seconds':prior})
        def guard():
            assert shutil.disk_usage(ROOT).free>=cfg['minimum_free_bytes'],'8GiB disk floor'
            assert time.monotonic()-start<cfg['phase_seconds'],'Engineering phase limit'
        guard();yield cfg,guard;status='completed'
    except BaseException as e:
        status='failed';write(dest/'FAILURE.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
    finally:
        elapsed=time.monotonic()-start
        write(dest/'RECEIPT.json',{'status':status,'seconds':elapsed,'prior_seconds':prior,'cumulative_seconds':prior+elapsed,
            'cap':None,'includes':'imports/loading/inference/finalization inside allocation; pre-GPU hash validation excluded'})
        lease.close()

def seal(dest,predictions):
    files=sorted((dest/'episodes').rglob('*'));files=[p for p in files if p.is_file()]
    write(dest/'PREDICTIONS_SEAL.json',{'files':{str(p.relative_to(dest)):sha(p) for p in files},'predictions':predictions,
        'source_GT_read':False,'target_read':False})
    return sha(dest/'PREDICTIONS_SEAL.json')

def verify_seal(dest):
    complete=read(dest/'COMPLETE.json');assert complete['seal_sha']==sha(dest/'PREDICTIONS_SEAL.json')
    sealdata=read(dest/'PREDICTIONS_SEAL.json')
    check_pins({str(dest/p):h for p,h in sealdata['files'].items()})
    return complete,sealdata

def tensor_sha(t):
    import torch
    t=t.detach().cpu().contiguous()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()
