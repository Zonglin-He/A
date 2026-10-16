"""Human-authorized reversible process pause after a committed baseline query."""
import json,os,signal,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
BASE=ROOT/'artifacts/stvg_opd_optimization_geometry_v1'
S=ROOT/'artifacts/stvg_opd_supplement_v1/baselines'

def process(pid):
    p=Path('/proc')/str(pid)
    stat=(p/'stat').read_text();rest=stat[stat.rfind(')')+2:].split()
    return dict(pid=pid,state=rest[0],start_ticks=rest[19],
                command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(),
                cwd=str((p/'cwd').resolve()))

def run():
    assert not (BASE/'BASELINE_PAUSE_RECEIPT.json').exists()
    s=read(S/'STATUS.json');assert s['status']=='running' and s['job']=='formal' and s['dataset']=='vidstg'
    cp=int(s['controller_pid']);wp=int(s['worker_pid']);c=process(cp);w=process(wp)
    assert 'continue_stvg_opd_baseline_supplement_v1.py' in c['command']
    assert 'run_stvg_opd_baseline_supplement_v1.py formal vidstg' in w['command']
    dest=S/'formal'/s['dataset']/s['candidate'];old=read(dest/'STATUS.json')
    pins={f:sha(ROOT/f) for f in ['scripts/run_stvg_opd_baseline_supplement_v1.py','scripts/continue_stvg_opd_baseline_supplement_v1.py',
       'methods/CURRENT_METHOD.json','methods/decota_spatial_opd_v1/configs.json']}
    write(BASE/'PRIORITY_AUTHORIZATION.json',dict(status='human_explicit_pause_after_current_query',
      request='当前 query 保存后暂停基线，先跑新对照（推荐）',scope='reversible SIGSTOP of original processes only; no kill/restart/source edit',
      controller=c,worker=w,accepted_before=old['done'],protected=pins,time=time.time()))
    os.kill(cp,signal.SIGSTOP)
    deadline=time.monotonic()+90
    while time.monotonic()<deadline:
        current=read(dest/'STATUS.json')
        if current['done']>old['done']:break
        time.sleep(.01)
    else:
        os.kill(cp,signal.SIGCONT);raise RuntimeError('No committed next query within bounded wait; controller resumed, worker untouched')
    os.kill(wp,signal.SIGSTOP);time.sleep(1)
    assert process(cp)['state']=='T' and process(wp)['state']=='T'
    current=read(dest/'STATUS.json');prefix={};n=0;bytes_=0
    for p in sorted(dest.glob('order*/*.pt')):
        rc=p.with_suffix('.json')
        if not rc.exists():continue
        receipt=read(rc);h=sha(p);assert h==receipt['sha256'] and not receipt['GT_read']
        prefix[str(p.relative_to(ROOT))]=dict(sha256=h,receipt_sha256=sha(rc),bytes=p.stat().st_size);n+=1;bytes_+=p.stat().st_size
    assert n>=current['done']>old['done']
    for f,h in pins.items():assert sha(ROOT/f)==h
    write(BASE/'BASELINE_PAUSE_RECEIPT.json',dict(status='paused_original_processes_preserved',scope='user-priority pause, not experiment failure',
      dataset=s['dataset'],candidate=s['candidate'],controller=process(cp),worker=process(wp),accepted_predictions=n,
      bytes=bytes_,accepted_stage_snapshot=current,prefix=prefix,protected=pins,
      original_shared_GPU_lease_still_owned_by_paused_worker=True,GT_read=False,time=time.time()))
    print('BASELINE_USER_PAUSED',cp,wp,n,bytes_,flush=True)

if __name__=='__main__':run()
