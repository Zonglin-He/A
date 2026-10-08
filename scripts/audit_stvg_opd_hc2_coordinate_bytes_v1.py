"""Postselection opaque bytes/config coverage/receipt-time audit, no tensor loads."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_hc2_coordinate_common_v1 import *

def run():
    verify();selection=read(BASE/'SELECTION_BARRIER.json');assert selection['status']=='all_five_coordinates_locked'
    total=0;size=0;reuse=0;invalid=0;trials={}
    for path in sorted((BASE/'trials').glob('cfg_*/CONFIG.json')):
        dest=path.parent;uid=dest.name;t=read(path)
        if (dest/'INVALID_DISPOSITION.json').exists():
            disposition=read(dest/'INVALID_DISPOSITION.json');assert not disposition['score_read'] and not (dest/'CPU_COMPLETION.json').exists()
            for f,h in disposition['prefix'].items():
                p=ROOT/f;rc=read(p.with_suffix('.json'));assert sha(p)==h==rc['sha256'] and not rc['GT_read']
            invalid+=1;continue
        b=read(dest/'PREDICTION_BARRIER.json');assert b['config_sha256']==sha(path) and b['cells']==64 and not b['GT_read']
        if b['exact_history_reused']:
            source=ROOT/b['origin'];original=read(source/'CONFIG.json')
            assert all(original[k]==t[k] for k in ['config','parents','orders'])
            assert sha(source/'PREDICTION_BARRIER.json')==b['original_barrier_sha256'];reuse+=1
        else:source=dest
        expected={str((source/order/f'{i:05}.pt').relative_to(ROOT)) for order,seq in t['orders'].items() for i in range(len(seq))}
        assert expected==set(b['files']) and len(expected)==64
        nbytes=0
        for f,h in b['files'].items():
            p=ROOT/f;rc=read(p.with_suffix('.json'));assert sha(p)==h==rc['sha256'] and not rc['GT_read'] and not p.is_symlink()
            if 'bytes' in rc:assert p.stat().st_size==rc['bytes']
            assert rc['time']<=b.get('original_seal_time',b['time'])
            nbytes+=p.stat().st_size
        trial_exposure=read(dest/'GT_EXPOSURE.json');index=trial_exposure['coordinate'];_,coord=coordinate_barrier(index)
        assert b['time']<=coord['time']<=trial_exposure['time']<=read(dest/'CPU_COMPLETION.json')['time']
        trials[uid]=dict(arrivals=64,bytes=nbytes,exact_prior_history_reused=b['exact_history_reused'],
            complete_receipt_coverage_and_coordinate_seal_before_GT=True)
        total+=64;size+=nbytes
    write(BASE/'ROOT_BYTE_READBACK.json',dict(status='pass',trials=trials,complete_arrivals=total,bytes=size,
        reused_exact_complete_configurations=reuse,numerical_invalid_not_scored=invalid,
        array_GT_weights_or_gradients_read=False,not_an_efficacy_audit=True,time=time.time()))
    print('HC2_COORDINATE_OPAQUE_BYTES_PASS',total,size,reuse,invalid)

if __name__=='__main__':run()
