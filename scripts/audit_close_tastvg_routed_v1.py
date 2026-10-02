"""CPU root closing checks independent of the inference routing helper."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.tastvg_routed_common_v1 import *
import numpy as np

def independent_route(candidates, ids):
    # Explicit scalar votes, CDF inversion and distinct-frame completion.
    votes=[sum(c['physical_interval'][0]<=t<c['physical_interval'][1] for c in candidates)/len(candidates) for t in ids]
    total=sum(votes);assert total>0
    running=0.;cdf=[]
    for v in votes:running+=v;cdf.append(running/total)
    raw=[next(i for i,v in enumerate(cdf) if v>=q) for q in [.1,.3,.5,.7,.9]]
    chosen=set(raw);middle=[c-v/(2*total) for c,v in zip(cdf,votes)];filled=[]
    while len(chosen)<5:
        choices=[i for i,v in enumerate(votes) if i not in chosen and v>0] or [i for i in range(len(ids)) if i not in chosen]
        i=max(choices,key=lambda j:(min(abs(middle[j]-middle[k]) for k in chosen),votes[j],-j));chosen.add(i);filled.append(i)
    return dict(weights=votes,raw_quantiles=raw,positions=sorted(chosen),filled=filled)

def run():
    import torch
    torch.set_num_threads(2)
    lock=verify();checks=collections.Counter();details={};bar=read(BASE/'GLOBAL_PREDICTION_BARRIER.json')
    assert bar['cells']==1536 and bar['new_R_cells']==bar['reused_A_cells']==768 and not bar['GT_read']
    assert sha(ROOT/'methods/CURRENT_METHOD.json')=='bd75706cf8377823ac6af488b1bd993af5fd7480e7e45a047c6fdde5e81df8d2'
    checks['deployment_unchanged']+=1
    for ds in DATASETS:
        p=read(BASE/ds/'PLAN.json');cost={}
        assert len(p['splits']['search']['orders'])==2 and len(p['conditions'])==6
        for arm in ['A','R']:
            audit=read(BASE/ds/arm/'AUDIT.json');assert audit['status']=='pass' and audit['state_links']==384 and audit['max_parameter_reconstruction_error']==0
            assert audit['GT_after_global_barrier'] and not audit['GPU_initialized'];cost[arm]=audit
            exposure=read(BASE/ds/arm/'GT_EXPOSURE.json');assert exposure['time']>bar['time']
            req=read(BASE/ds/arm/'REQUEST.json');assert req['method']['support_mode']=='full' and req['method']['target_mode']=='rank' and req['method']['actuation']=='rkl'
            checks['GT_after_global_seal']+=1
            seal=read(BASE/ds/arm/'PREDICTION_BARRIER.json');assert len(seal['files'])==384
            for rel,h in seal['files'].items():
                assert sha(BASE/ds/arm/rel)==h
                receipt=read(BASE/ds/arm/rel);assert sha((BASE/ds/arm/rel).with_suffix('.pt'))==receipt['sha256'];checks['payload_hashes']+=1
        smoke=read(BASE/ds/'SMOKE_ROOT_EVIDENCE.json');assert smoke['status']=='pass' and smoke['uniform_specialist_bitwise_parity'] and len(smoke['A_clean_arrivals'])==2
        for c in p['conditions']:
            for order,sequence in p['splits']['search']['orders'].items():
                for at,parent in enumerate(sequence):
                    if at%4:continue
                    f=BASE/ds/'R/online'/c/order/f'{at:05}.pt';x=load(f)
                    route=independent_route(x['temporal']['candidates'],p['rows'][parent]['frame_ids'])
                    for field in ['positions','raw_quantiles','filled']:assert route[field]==x['routing'][field],(ds,c,order,at,field)
                    np.testing.assert_allclose(route['weights'],x['routing']['weights'],atol=0,rtol=0)
                    ev=x['routed_expert'];assert ev['positions']==route['positions'] and ev['pixel_sha256']==x['pixel_sha256']
                    assert sha(ROOT/ev['cache'])==ev['cache_sha256'] and not ev['GT_read'];cached=load(ROOT/ev['cache']);assert cached['input_sha256']==ev['input_sha256'] and cached['positions']==route['positions'] and not cached['GT_read']
                    checks['independent_scalar_route_and_cache_binding']+=1
                    assert all(s.get('event_support') is None for s in x['update_steps']);checks['unweighted_geometry']+=1
        details[ds]=dict(online_audits=cost,resources=read(BASE/ds/'RESOURCES.json'),native_and_uniform_parity=smoke)
    eligibility=read(BASE/'token/ST_ELIGIBILITY.json');assert all(not d['eligible'] for d in eligibility.values())
    for ds in DATASETS:
        assert eligibility[ds]['sealed_T_rows_sha256']==sha(PUBLIC/ds/'TOKEN_ROWS.json')
        assert not (PUBLIC/ds/'ST_ROWS.json').exists();checks['conditional_ST_not_triggered']+=1
    token=read(BASE/'token/ROOT_READBACK.json');assert token['status']=='pass' and token['checks']['native_prediction_parity']==60 and token['max_error']<1e-12 and not token['GPU_initialized']
    assert len(list(BASE.glob('*/attempts/*.json')))==76<=lock['max_new_specialist_calls']
    assert not torch.cuda.is_initialized()
    result=dict(status='pass',checks=dict(checks),details=details,token_readback=token,model_parameter_updates_in_token_P0=0,conditional_ST_and_combined_online='not_triggered_by_locked_qualification',deployment_unchanged=True,new_specialist_calls=76,GPU_initialized=False,time=time.time())
    write(BASE/'ROOT_CLOSING_READBACK.json',result);write(PUBLIC/'ROOT_CLOSING_READBACK.json',result)
    write(PUBLIC/'CONFIG.json',dict(question='Does reference-position-only routing improve future nonexpert online adaptation; does native token binding qualify on fixed candidates?',primary_dataset='hc2',secondary_dataset='vidstg',online_sources_per_dataset=32,queries_per_source=1,orders=2,conditions=['clean','frame_drop_5','frame_freeze_5','motion_blur_5','occlusion_5','exposure_5'],arrivals_per_arm_per_dataset=384,scheduled_expert_fraction=.25,total_online_scored_arrivals=1536,new_R_arrivals=768,exact_reused_A_arrivals=768,exposure='historically exposed development sources, not fresh/full evaluation',params={ds:read(BASE/ds/'PLAN.json')['params'] for ds in DATASETS},token=dict(fixed_cells_per_dataset=30,sources_per_dataset=10,clean_cells_per_dataset=10,corrupt_cells_per_dataset=20,candidates_per_cell=9,parameter_updates=0,external_models=0,GT_used_for_model=False,inside_interval='common sealed A native interval',object_query='native spatial-branch cross-attention weighted post-MM text',event_query='native temporal-branch cross-attention weighted post-MM text',roi='positive cell-area intersection weighted features',binding='motion inside cosine minus same-tube outside cosine',semantic_decomposition_validated=False),global_online_prediction_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),token_prediction_barrier_sha256=sha(BASE/'token/PREDICTION_BARRIER.json'),deployment_unchanged=True))
    print('ROOT closing readback pass',dict(checks))
if __name__=='__main__':run()
