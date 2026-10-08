"""Opaque original/revision receipt readback without arrays or GT."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import *

def run(phase='P0'):
    verify();global_b=read(BASE/(phase+'_PREDICTION_BARRIER.json'));d=read(BASE/'DESIGN_LOCK.json');runtime=sha(BASE/'RUNTIME_LOCK.json')
    count=0;nbytes=0;coverage={}
    for ds in ['hc2','vidstg']:
        name=phase+'_'+ds;p=BASE/'stages'/name/'PREDICTION_BARRIER.json';b=read(p);stage=d['stages'][name]
        assert global_b['barriers'][str(p.relative_to(BASE))]==sha(p)
        assert b['orders']==stage['orders'] and b['config']==d['datasets'][ds]['config'] and not b['GT_read']
        expected={f'stages/{name}/clean/{order}/{arm}/{at:05}.pt' for order,seq in stage['orders'].items() for arm in stage['arms'] for at in range(len(seq))}
        assert set(b['files'])==expected
        origin_runtime=b.get('original_runtime_lock_sha256',runtime)
        if b.get('exact_stage_reused'):
            original=ROOT/b['original_barrier_path'];assert sha(original)==b['original_barrier_sha256']
            assert d['datasets'][ds]['config']==read(OLD/'DESIGN_LOCK.json')['datasets'][ds]['config']
        for f,h in {**b['files'],**b['inputs']}.items():
            p=BASE/f;rc=read(p.with_suffix('.json'));assert not p.is_symlink() and sha(p)==h==rc['sha256'] and not rc['GT_read']
            if f in b['files']:assert p.stat().st_size==rc['bytes']
            # Mixed original frozen inputs and exact unchanged saved prefix have
            # original receipts. New fits always have the revision runtime.
            assert rc['runtime_lock_sha256'] in {runtime,origin_runtime,sha(OLD/'RUNTIME_LOCK.json')}
            assert rc['time']<=b['time']<=global_b['time'];count+=1;nbytes+=p.stat().st_size
        coverage[ds]=dict(adapted_arrivals=len(b['files']),inputs=len(b['inputs']),exact_stage_reused=b.get('exact_stage_reused',False))
    write(BASE/(phase+'_ROOT_BYTE_READBACK.json'),dict(status='pass',files=count,bytes=nbytes,coverage=coverage,
        original_and_revision_receipts_distinguished=True,payload_or_GT_read=False,time=time.time()))
    print('REVISION_OPAQUE_BYTE_PASS',phase,count,nbytes)

if __name__=='__main__':run(sys.argv[1] if len(sys.argv)>1 else 'P0')
