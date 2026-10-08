"""Root's additional postseal dense spatial audit of every P0 arm/order."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import gzip,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_common_v1 import *

def run():
    import torch
    torch.set_num_threads(2)
    verify();assert read(BASE/'P0_CPU_COMPLETION.json')['P0_postseal_scoring_complete']
    from scripts.score_stvg_opd_paper_v1 import truths
    from vg_tta.tastvg_oracle_event5_v1 import DenseTube
    design=read(BASE/'DESIGN_LOCK.json');checks=0;maximum=0.;counts={};start=time.time()
    for ds in DATASETS:
        stage=design['stages']['P0_'+ds];dense,spans,provenance=truths(ds,stage)
        plan=read(PAPER/ds/'PLAN.json')
        with gzip.open(PUB/('P0_'+ds)/'ROWS.jsonl.gz','rt') as f:
            scored={(r['query_ordinal'],r['order'],r['arm']):r for r in map(json.loads,f)}
        n=0
        for order,sequence in stage['orders'].items():
            for at,q in enumerate(sequence):
                row=plan['rows'][q]
                for arm in ARMS:
                    file=BASE/'stages'/('P0_'+ds)/'clean'/order/arm/f'{at:05}.pt'
                    assert sha(file)==read(file.with_suffix('.json'))['sha256']
                    z=load(file);fit=z['fit'];inp=load(BASE/z['input']['path'])
                    for name,boxes in [('Frozen',inp['native_boxes']),('Before',fit['before']),('After',fit['final'])]:
                        tube=DenseTube(boxes.numpy(),row,dense[q],spans[q],clip=ds=='hc2')
                        error=abs(tube.s-scored[q,order,arm][name+'_s']);maximum=max(maximum,error)
                        assert error<2e-10,(ds,q,arm,name,error)
                        checks+=1;n+=1
        counts[ds]=dict(spatial_readout_comparisons=n,annotation_provenance=provenance)
    result=dict(status='pass',comparisons=checks,maximum_absolute_error=maximum,datasets=counts,
        all_P0_arms_and_orders=True,GT_read_only_after_global_P0_barrier=True,
        decoder_Jacobian_independently_reimplemented=False,CPU_seconds=time.time()-start)
    write(PUB/'P0_ROOT_DENSE_SPATIAL_AUDIT.json',result)
    print(json.dumps(result))

if __name__=='__main__':run()
