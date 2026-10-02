"""Produce S/E/SE maps and raw/QC decisions before reading diagnostic labels."""
from tastvg_temporal_router_common_v1 import *
import numpy as np


def run():
    import torch
    torch.set_num_threads(2);verify();t=time.time();bindings=read(BASE/'INPUT_BINDINGS.json');count=0
    from vg_tta.tastvg_temporal_router_t0_v1 import route
    from vg_tta.tastvg_temporal_qualification_v1 import critic_scores
    from methods.decota_final_simplified_v1.tensors import state_hash
    for ds in DATASETS:
        p=read(PRIOR/ds/'PLAN.json');records=[];previous={};parity=0
        for r in bindings[ds]:
            x=load(ROOT/r['payload']);row=p['rows'][r['parent']];key=r['condition'],r['order']
            assert x['GT_read'] is False and x['parent']==r['parent'] and x['arrival']==r['arrival']
            assert state_hash(x['pre_state'])==x['pre_sha'] and state_hash(x['post_state'])==x['post_sha']
            if r['arrival']>0:assert previous[key]==x['pre_sha']
            previous[key]=x['post_sha']
            z={**r,'pre_sha':x['pre_sha'],'post_sha':x['post_sha'],'pixel_sha256':x['pixel_sha256']}
            if r['expert_scheduled']:
                ev={}
                for stage,rel in r['expert_receipts'].items():
                    er=read(ROOT/rel);assert er['pixel_sha256']==x['pixel_sha256']
                    ev[stage]=load(POOL/ds/'experts'/er['cache'])
                    assert ev[stage]['GT_read'] is False
                td=x['temporal'];assert td['candidates'][0]['indices']==x['slow']['indices']
                rr=route(td['candidates'],row['frame_ids'],ev['temporal']['proposals'],ev['temporal']['proposal_confidence'])
                original=critic_scores([c['physical_interval'] for c in td['candidates']],ev['temporal']['proposals'],ev['temporal']['proposal_confidence'])
                np.testing.assert_allclose(original,rr['current_scores'],atol=1e-14,rtol=0)
                np.testing.assert_allclose(original,td['scores'],atol=1e-14,rtol=0)
                assert rr['current_selected']==td['selected'] and x['final_indices']==td['candidates'][td['selected']]['indices']
                assert len(set(tuple(c['indices']) for c in td['candidates']))==len(td['candidates'])
                z.update(router=rr,current_indices=x['final_indices'],
                    qc_indices=td['candidates'][rr['qc_selected']]['indices'],candidate_count=len(td['candidates']))
                parity+=1
            else:
                assert x['temporal'] is None and x['final_indices']==x['slow']['indices']
                z.update(current_indices=x['final_indices'],qc_indices=x['final_indices'],router=None)
            records.append(z);count+=1
        write(BASE/ds/'OBSERVATIONS.json',records)
        write(BASE/ds/'PREDICTION_BARRIER.json',dict(cells=384,expert_cells=96,raw_critic_parity=parity,
              file_sha256=sha(BASE/ds/'OBSERVATIONS.json'),GT_used=False,GPU_initialized=torch.cuda.is_initialized(),time=time.time()))
    assert count==768 and not torch.cuda.is_initialized()
    write(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(cells=768,expert_cells=192,
        datasets={ds:sha(BASE/ds/'PREDICTION_BARRIER.json') for ds in DATASETS},
        GT_used=False,GPU_initialized=False,seconds=time.time()-t,time=time.time()))
    status(BASE/'STATUS.json',dict(status='sealed_pending_cpu_GT_audit',done=count,total=768,GT_used=False))
    print('SEALED',count,'expert maps',192,'no GPU or new expert inference',flush=True)


if __name__=='__main__':run()
