"""CPU-only label-free generation, exact old critic parity and global seal."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.tastvg_temporal_coverage_common_v1 import *

def run():
    import torch,numpy as np,collections,functools
    from vg_tta.tastvg_temporal_candidate_coverage_v1 import allocate,choose
    sys.addaudithook(guard);verify();torch.set_num_threads(2);tick=time.monotonic();counts=collections.Counter()
    @functools.lru_cache(maxsize=144)
    def frozen(ds,parent,condition):
        c=dict(dataset=ds,parent=parent,condition=condition)
        d,r,f=capture(c)
        # Do not retain large H tensors merely to use two small temporal heads.
        return dict(logits=d['prediction']['logits'],records=d['records'],ids=d['frame_ids'],
            native_indices=d['prediction']['indices'],pixel_sha256=r['pixel_sha256'],cache_sha256=r['sha256'])
    cells=read(BASE/'COHORT.json')['cells'];files={}
    for done,c in enumerate(cells,1):
        budget();a=acell(c);x=donor(c);ids=plan(c['dataset'])['rows'][c['parent']]['frame_ids']
        assert torch.equal(x['A']['boxes'],a['slow']['boxes'])
        assert x['persistent_pre_sha']==a['pre_sha'] and x['persistent_post_sha']==a['post_sha']
        assert x['A']['indices']==a['final_indices'] and a['pixel_sha256']==c['pixel_sha256']
        result=dict(cell_key=key(c),dataset=c['dataset'],split=c['split'],source_id=c['parent'],
            condition=c['condition'],order=c['order'],arrival=c['arrival'],expert_scheduled=c['scheduled'],
            persistent_pre_sha=a['pre_sha'],persistent_post_sha=a['post_sha'],
            A_indices=a['final_indices'],native_indices=a['slow']['indices'],
            old_payload_sha256=sha(oldfile(c)),donor_payload_sha256=sha(donorfile(c)),
            new_expert_calls=0,new_model_calls=0,GT_read=False)
        if c['scheduled']:
            d=frozen(c['dataset'],c['parent'],c['condition']);assert d['ids']==ids and d['pixel_sha256']==c['pixel_sha256']
            e,r=old.expert(c['dataset'],'temporal',c['parent'],c['condition'],c['pixel_sha256'])
            oldpool=x['temporal_candidates']; assert len(oldpool)==8 and oldpool==a['temporal']['candidates']
            assert oldpool[0]['indices']==a['slow']['indices']
            prior=choose(oldpool,e['proposals'],e['proposal_confidence'])
            assert prior['scores']==a['temporal']['scores'] and prior['selected']==a['temporal']['selected']
            assert oldpool[prior['selected']]['indices']==a['final_indices']
            newpool=allocate(a['slow']['indices'],ids,d['logits'],d['records'])
            decision=choose(newpool,e['proposals'],e['proposal_confidence'])
            result.update(old_candidates=oldpool,new_candidates=newpool,old_decision=prior,new_decision=decision,
                new_indices=newpool[decision['selected']]['indices'],expert_receipt=r,
                source_head_cache_sha256=d['cache_sha256'],
                source_native_matches_current=d['native_indices']==a['slow']['indices'])
            counts['expert_pools']+=1;counts['frozen_native_parity']+=result['source_native_matches_current']
            counts['stratum_fallback_slots']+=sum(p['fallback'] for p in newpool)
            counts['native_retained']+=newpool[0]['indices']==oldpool[0]['indices']
            counts['old_critic_exact_parity']+=1
        else:result['new_indices']=a['final_indices'];counts['nonexpert_exact_A']+=1
        f=BASE/c['dataset']/'predictions'/f'{prefix(c)}.json';write(f,result)
        files[str(f.relative_to(BASE))]=sha(f)
        if done%96==0:
            status(BASE/'STATUS.json',dict(status='label_free_generation_running',done=done,total=1152,
                worker_pid=os.getpid(),GT_read=False,time=time.time()))
            print('GENERATE_NO_GT',done,1152,flush=True)
    assert counts['expert_pools']==counts['native_retained']==counts['old_critic_exact_parity']==288
    assert counts['nonexpert_exact_A']==864 and not torch.cuda.is_initialized()
    verify()
    resource=dict(status='pass',arrivals=1152,counts=dict(counts),new_model_calls=0,new_expert_calls=0,
        CUDA_initialized=False,worker_wall_seconds=time.monotonic()-tick,GT_read=False,time=time.time())
    write(BASE/'GENERATION_RESOURCES.json',resource)
    write(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='sealed',arrivals=1152,expert_pools=288,
        GT_read=False,files=files,runtime_sha256=sha(BASE/'RUNTIME_LOCK.json'),time=time.time()))
    status(BASE/'STATUS.json',dict(status='sealed_pending_CPU_GT_score',arrivals=1152,expert_pools=288,
        GT_read=False,time=time.time()))
    archive('1152读出与288新旧八候选对照均已无GT封存，原critic逐分/首选精确复现，待CPU网格oracle评分')
    print('GLOBAL_PREDICTION_SEALED',resource,flush=True)
if __name__=='__main__':run()
