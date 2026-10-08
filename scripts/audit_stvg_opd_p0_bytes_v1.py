"""Independent opaque byte/receipt/coverage/seal-time readback; never unpickle."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_common_v1 import *

def run():
    verify();global_barrier=read(BASE/'P0_PREDICTION_BARRIER.json')
    assert global_barrier['status']=='sealed' and global_barrier['all_deployment_arms_both_directions']
    design=read(BASE/'DESIGN_LOCK.json');runtime=sha(BASE/'RUNTIME_LOCK.json');counts={};totalbytes=0;totalfiles=0
    for ds in DATASETS:
        stage='P0_'+ds;rel=f'stages/{stage}/PREDICTION_BARRIER.json';barrier=read(BASE/rel)
        assert sha(BASE/rel)==global_barrier['barriers'][rel]
        assert barrier['orders']==design['stages'][stage]['orders'] and barrier['config']==design['datasets'][ds]['config']
        assert barrier['source_checkpoint_unchanged'] and not barrier['GT_read'] and barrier['runtime_lock_sha256']==runtime
        expected={f'stages/{stage}/clean/{order}/{arm}/{at:05}.pt' for order,sequence in barrier['orders'].items() for arm in ARMS for at in range(len(sequence))}
        assert set(barrier['files'])==expected and len(expected)==768
        assert len(barrier['inputs'])==128
        checked=0;bytecount=0
        for group in ['files','inputs']:
            for f,h in barrier[group].items():
                path=BASE/f;receipt=read(path.with_suffix('.json'))
                assert not path.is_symlink() and sha(path)==h==receipt['sha256']
                assert not receipt['GT_read']
                if group=='files':assert path.stat().st_size==receipt['bytes']
                else:assert set(receipt)=={'sha256','runtime_lock_sha256','GT_read','time'}
                assert receipt['runtime_lock_sha256']==runtime
                assert receipt['time']<=barrier['time']<=global_barrier['time']
                checked+=1;bytecount+=path.stat().st_size
        counts[ds]=dict(opaque_files=checked,adapted_arrivals=768,Frozen_logical_arrivals=256,
            unique_inputs=128,bytes=bytecount,complete_coverage_and_receipt_before_global_seal=True)
        totalfiles+=checked;totalbytes+=bytecount
    result=dict(status='pass',datasets=counts,opaque_files=totalfiles,bytes=totalbytes,
        barrier_sha256=sha(BASE/'P0_PREDICTION_BARRIER.json'),runtime_sha256=runtime,
        prediction_array_or_GT_or_weights_read=False,not_an_efficacy_audit=True,time=time.time())
    write(BASE/'P0_ROOT_BYTE_READBACK.json',result)
    print('P0_ROOT_OPAQUE_BYTES_PASS',totalfiles,totalbytes)

if __name__=='__main__':run()
