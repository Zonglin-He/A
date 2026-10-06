"""Development labels after complete trial seal; never used by GPU worker."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,gzip,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_opd_tuning_common_v1 import *
import numpy as np

def run(ds,trial):
    verify()
    import torch
    torch.set_num_threads(2)
    from scripts.score_decota_spatial_opd_v1 import truths,source_stats
    from scripts.run_decota_paper_main_v1 import unpack_expert
    from vg_tta.tastvg_oracle_event5_v1 import official,DenseTube
    from vg_tta.decota_spatial_opd_tunable_audit_v1 import audit
    dest=BASE/'trials'/ds/trial;t=read(dest/'CONFIG.json');barrier=read(dest/'PREDICTION_BARRIER.json')
    assert barrier['status']=='sealed' and barrier['config_sha256']==sha(dest/'CONFIG.json')
    for p,h in barrier['files'].items():assert sha(BASE/p)==h
    if (dest/'CPU_COMPLETION.json').exists():return
    write(dest/'GT_EXPOSURE.json',dict(prediction_barrier_sha256=sha(dest/'PREDICTION_BARRIER.json'),
          role='development_hyperparameter_selection_only',time=time.time()))
    dense,spans,provenance=truths(ds,t);plan=read(PAPER/ds/'PLAN.json')
    sources=sorted({plan['rows'][p]['source'] for p in t['parents']});ids={s:i for i,s in enumerate(sources)}
    rr=[];checks=0
    for order,seq in t['orders'].items():
        previous=None;previoushash=None
        for at,parent in enumerate(seq):
            p=dest/order/f'{at:05}.pt';z=load(p);fit=z['fit'];row=plan['rows'][parent];inp=load(BASE/z['input']['path'])
            assert sha(BASE/z['input']['path'])==z['input']['sha256'] and not z['GT_read']
            assert z['previous_payload_sha256']==previoushash and fit['config']==t['config']
            assert fit['selected_step']==t['config']['steps'] and len(fit['path'])==t['config']['steps']+1
            if previous is not None:
                assert all(torch.equal(v,torch.zeros_like(v) if n=='spatial.query_residual' else previous[n]) for n,v in fit['initial'].items())
            for n,v in fit['initial'].items():
                expected=torch.zeros_like(v) if n=='spatial.query_residual' else v+(fit['state'][n]-v)*t['config']['writeback']
                assert torch.equal(z['committed'][n],expected)
            assert audit(fit,unpack_expert(inp['expert']))==z['math_audit'];checks+=1
            values={}
            for name,boxes in [('Frozen',inp['native_boxes']),('Before',fit['before']),('After',fit['final'])]:
                score=official(boxes.numpy(),row,dense[parent],spans[parent],z['interval'],ds)
                independent=DenseTube(boxes.numpy(),row,dense[parent],spans[parent],clip=ds=='hc2').score(z['interval'])
                assert abs(score['v']-independent['v'])<2e-10 and abs(score['t']-independent['t'])<2e-10
                values[name]=score
            r=dict(source_id=ids[row['source']],order=order,condition='clean',trial=trial,
                   delta_total_v=values['After']['v']-values['Frozen']['v'],
                   delta_inherited_v=values['Before']['v']-values['Frozen']['v'],
                   delta_current_v=values['After']['v']-values['Before']['v'],
                   delta_total_s=values['After']['s']-values['Frozen']['s'],
                   Frozen_v=values['Frozen']['v'],After_v=values['After']['v'],compute=z['compute'])
            rr.append(r);previous=z['committed'];previoushash=sha(p)
    assert len(rr)==barrier['cells']
    stats=source_stats(rr,['delta_total_v','delta_inherited_v','delta_current_v','delta_total_s','Frozen_v','After_v'])
    out=PUB/ds/trial;out.mkdir(parents=True,exist_ok=True)
    summary=dict(dataset=ds,trial=trial,phase=t['phase'],config=t['config'],statistics=stats,
                 development_selection=True,historical_exposure=True,independent_math_state_checks=checks,
                 GT_provenance=provenance,GPU_fit_seconds=sum(r['compute']['fit_GPU_seconds'] for r in rr),time=time.time())
    write(out/'SUMMARY.json',summary)
    with gzip.open(out/'ROWS.jsonl.gz','wt') as f:
        for r in rr:f.write(json.dumps(r,allow_nan=False)+'\n')
    write(dest/'CPU_COMPLETION.json',dict(status='sealed_development_score_audited',rows=len(rr),
          files={str(p.relative_to(ROOT)):sha(p) for p in out.iterdir()},time=time.time()))
    print('OPD_TUNE_CPU_DONE',ds,trial,stats['metrics']['delta_total_v'],flush=True)

if __name__=='__main__':run(*sys.argv[1:3])
