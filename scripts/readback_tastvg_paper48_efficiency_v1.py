"""Root P4 audit: all hashes, state chain, new expert calls, Frozen parity."""
import sys,json,csv,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,load,sha

def run():
    import torch
    torch.set_num_threads(1)
    from methods.decota_final_simplified_v1.tensors import state_hash
    from scripts.tastvg_paper48_common_v1 import verify,BASE
    from scripts.audit_tastvg_paper48_efficiency_public_v1 import run as audit
    verify();out=BASE/'P4';plan=read(BASE/'P4_PLAN.json');p1=read(BASE/'P1/PLAN.json')
    assert read(out/'COMPLETION.json')['status']=='completed'
    timings=[];previous=None;updates=0;calls=0
    for mode in ['Frozen','Ours']:
        for i,parent in enumerate(plan['parents']):
            file=out/mode/f'{i:03}.pt';rr=read(file.with_suffix('.json'));r=read(out/mode/f'{i:03}.timing.json')
            assert all(r[k]==v for k,v in rr.items()) and rr['sha256']==sha(file)
            x=load(file);assert x['parent']==r['parent']==parent and x['index']==i and x['mode']==mode and x['GT_read'] is False
            assert r['uncached_native'] and not r['cached_previous_expert_calls_used'] and r['GT_read'] is False
            scheduled=mode=='Ours' and i%4==0;assert r['expert_scheduled']==scheduled
            if mode=='Frozen':
                a=p1['orders']['order1'].index(parent);ref=BASE/'P1/online/clean/order1'/f'{a:05}.pt';assert sha(ref)==read(ref.with_suffix('.json'))['sha256'];y=load(ref)['source_native']
                assert torch.equal(x['prediction']['boxes'],y['boxes']) and x['prediction']['indices']==y['indices'],('Frozen parity',i)
                assert x['post_state'] is None and not x['updated']
            else:
                if i==0:
                    first=load(BASE/'P1/online/clean/order1/00000.pt');assert x['pre_sha']==first['pre_sha']
                    assert torch.equal(x['prediction']['boxes'],load(out/'Frozen/000.pt')['prediction']['boxes'])
                else:assert x['pre_sha']==previous
                assert state_hash(x['post_state'])==x['post_sha'] and sum(v.numel() for v in x['post_state'].values())==1792
                if not scheduled:assert not x['updated'] and x['pre_sha']==x['post_sha']
                previous=x['post_sha'];updates+=x['updated']
                if scheduled:
                    for stage in ['spatial','temporal']:
                        dst=out/'uncached_experts'/f'{parent:05}'/stage;receipt=read(dst/stage/'clean'/f'{parent:05}.json');stat=read(dst/(stage.upper()+'_STATUS.json'))
                        assert receipt['new_inference'] and receipt['GT_read'] is False and sha(dst/receipt['cache'])==receipt['cache_sha256']
                        assert stat['status']=='completed' and stat['new_inferences']==stat['done']==1
                        calls+=1
            timings.append(r)
    assert calls==50
    fields=['mode','index','expert_scheduled','end_to_end_cold_seconds','native_process_seconds','load_seconds','decode_seconds','native_seconds','initialization_seconds','adapter_seconds','spatial_process_seconds','temporal_process_seconds','logical_expert_calls','peak_vram_including_experts']
    with (out/'TIMINGS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in timings:w.writerow({k:int(r[k]) if isinstance(r[k],bool) else r[k] for k in fields})
    result=audit(out);write(out/'PUBLIC_TIMING_AUDIT.json',result)
    root=dict(status='pass',prediction_hashes=200,timing_receipts=200,Frozen_bitwise_parity_with_P1=100,persistent_state_links=100,persistent_parameters=1792,actual_spatial_updates=int(updates),new_uncached_specialist_calls=50,GT_read=False,new_inference=False,time=time.time())
    write(out/'ROOT_READBACK.json',root);print(root);print(result)
if __name__=='__main__':run()
