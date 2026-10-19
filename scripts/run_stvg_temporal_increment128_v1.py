"""Finite prior-preserving screen. Export only missing Source logits, no expert calls."""
from pathlib import Path
import sys,time,json,collections,gc,os,random
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=R/'artifacts/stvg_temporal_increment128_v1';P=R/'results/stvg_error_state/2026-10-19/temporal_increment128'
PRE=R/'artifacts/stvg_referdino_condition_hc128_v1';EVENT=R/'artifacts/stvg_where_when_event128_v1'

def prepare():
    import torch
    assert not(D/'PLAN_LOCK.json').exists();D.mkdir(exist_ok=True);P.mkdir(parents=True,exist_ok=True)
    old=read(EVENT/'PLAN_LOCK.json');raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];known={}
    sp=R/'artifacts/stvg_dino_supported_source256_v1'
    for r in read(sp/'PLAN_LOCK.json')['records']:
        if r['dataset']=='hc2':known[r['query_ordinal']]=dict(path=str((sp/'source'/f"{r['alias']}.pt").relative_to(R)),kind='source256')
    # Other known captures are accepted only when their native Source is exact.
    for r in read(R/'artifacts/stvg_posterior_decision_tree_v1/PLAN_LOCK.json')['files']:
        if r['dataset']=='hc2'and r['query_ordinal']not in known:known[r['query_ordinal']]=dict(path=r['path'],kind='P6')
    records=[]
    for r in old['records']:
        a=r['alias'];z=load(PRE/'inputs'/f'{a}.pt');reuse=None
        if r['query_ordinal']in known:
            v=known[r['query_ordinal']];path=R/v['path'];x=load(path)if path.with_suffix('.receipt.json').exists()else torch.load(path,map_location='cpu',weights_only=False);c=x['capture']if v['kind']=='P6'else x;zero=c['zero']if v['kind']=='P6'else c
            b=zero['boxes']if v['kind']=='P6'else c['Source'];assert torch.equal(b.cpu(),z['Source']);assert c['records']==z['records'];reuse=dict(**v,sha256=sha(path))
        records.append(dict(r,reuse=reuse,event_prediction_sha256=sha(EVENT/'predictions'/f'{a}.pt')))
    excluded={r['parent']for r in records}
    previous_panels={}
    for label,rs in [('RH32',read(PRE/'PLAN_LOCK.json')['confirmation_reserved']),('RF32',read(R/'artifacts/stvg_referdino_finalize_v1/PLAN_LOCK.json')['formal_records']),('NH16',[r for r in read(R/'artifacts/stvg_referdino_native32_v1/PLAN_LOCK.json')['records']if r['dataset']=='hc2'])]:previous_panels[label]={r['parent']for r in rs}
    groups={}
    for r in raw:
        if r['source']not in excluded:groups.setdefault(r['source'],[]).append(r)
    parents=sorted(groups);random.Random(2026101911).shuffle(parents);assert len(parents)==109
    confirmation=[dict(alias=f'TC{i+1:03}',parent=parent,query_ordinal=min(groups[parent],key=lambda x:x['ordinal'])['ordinal'],historical_exposure='earlier full Source/old-method exposure; current RC128 excluded; old panel overlap disclosed',prior_panels=[k for k,v in previous_panels.items()if parent in v],event_token_cache_available=False)for i,parent in enumerate(parents[:128])]
    files=['vg_tta/stvg_temporal_increment128_v1.py','scripts/run_stvg_temporal_increment128_v1.py','scripts/score_stvg_temporal_increment128_v1.py','protocols/stvg_temporal_increment128_v1.md','scripts/stvg_temporal_marginal_decode_util256_v1.py','vg_tta/stvg_posterior_decision_tree_v1.py','vg_tta/stvg_where_when_event128_v1.py','methods/stvg_referdino_opd_v1/config.json','methods/stvg_referdino_opd_v1/_source_forward.py','methods/stvg_referdino_opd_v1/runtime.py']
    plan=dict(status='locked_before_new_prior_exports_readouts_GT',records=records,confirmation_reserved=confirmation,count=128,seeds=[41,42,43],lambdas=[0.,.5,1.,2.],primary_space='unchanged saved Live per seed; metric mean within parent',secondary_space='unchanged E',feedback='saved EventLive per seed for primary; EventE for secondary; unchanged old duration inside-minus-outside',selection='highest mean full-v of within-parent seed mean, tolerance1e-12 tie smaller lambda incl0; one common lambda, no per-case/domain/seed choice',confirmation_status='all109 remaining parents prelocked; fewer than requested128; missing token caches; both scope questions pending',frozen_spatial_config=read(R/'methods/stvg_referdino_opd_v1/config.json'),new_experts=0,new_ReferDINO=0,new_DINO=0,new_Student_updates=0,prior_cached=sum(r['reuse']is not None for r in records),new_Source_exports_required=sum(r['reuse']is None for r in records),GPU_seconds_cap=900,deadline_epoch=time.time()+14400,pins={f:sha(R/f)for f in files},created_epoch=time.time(),GT_read=False)
    put(D/'PLAN_LOCK.json',plan);put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in plan.items()if k not in ['records','confirmation_reserved','pins']});put(P/'CONFIRMATION_ROSTER_PUBLIC.json',dict(count=len(confirmation),requested_minimum=128,available_parent_total=237,parent_disjoint_from_RC128=True,prior_panel_overlap_counts={k:sum(k in r['prior_panels']for r in confirmation)for k in previous_panels},roster_sha256=sha(D/'PLAN_LOCK.json'),historical_exposure=confirmation[0]['historical_exposure'],semantics_cached=False,status='locked_below_requested_minimum_pending_scope_answers'))
    print('prior cached',plan['prior_cached'],'Source exports',plan['new_Source_exports_required'],flush=True)

def verify():
    p=read(D/'PLAN_LOCK.json');assert time.time()<p['deadline_epoch'];assert all(sha(R/f)==h for f,h in p['pins'].items());return p

def export():
    import torch
    from methods.stvg_referdino_opd_v1.runtime import setup,lease,load_source,decode
    from methods.stvg_referdino_opd_v1._source_forward import frozen_forward
    from methods.decota_final_simplified_v1.tensors import state_hash
    from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table
    from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
    from main_method.runtime import resource_guard
    setup();p=verify();raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];model=None;lock=None;start=None;files=[];calls=0
    try:
        for r in p['records']:
            verify();resource_guard();a=r['alias'];z=load(PRE/'inputs'/f'{a}.pt');new=False
            if r['reuse']:
                ref=r['reuse'];assert sha(R/ref['path'])==ref['sha256'];x=torch.load(R/ref['path'],map_location='cpu',weights_only=False);c=x['capture']if ref['kind']=='P6'else x;zero=c['zero']if ref['kind']=='P6'else c;logits=zero['logits'];records=c['records'];seconds=0.
            else:
                if model is None:lock=lease();start=time.perf_counter();model=load_source();mh=state_hash(model.state_dict())
                assert time.perf_counter()-start+20<p['GPU_seconds_cap'];assert torch.cuda.mem_get_info()[0]>4*2**30
                row=dict(raw[r['query_ordinal']]);subject=read(R/r['subject_path']);row['parses']=subject['parses'];tick=time.perf_counter();frames,ids=decode(row['input']);batch,records,base=frozen_forward(model,frames,row);assert torch.equal(base.zero['boxes'].cpu(),z['Source']);logits=[v.detach().cpu()for v in base.zero['logits']];seconds=time.perf_counter()-tick;calls+=1;new=True
            assert records==z['records'];ids,q=mass_table(probabilities(records,logits));old=final_mbr(ids,q);assert ids==z['frame_ids']and old['interval']==list(z['interval'])
            files.append(save(D/'priors'/f'{a}.pt',dict(alias=a,ids=ids,q=q,records=records,logits=logits,interval=old['interval'],native_zero_exact=True,distribution='existing full physical min-start/max-end envelope of two independent native offset endpoint posteriors',new_Source_export=new,seconds=seconds,GT_read=False)))
            print('prior',a,'new',new,round(seconds,2),flush=True);put(D/'STATUS.json',dict(stage='prior_export',done=len(files),total=128,new_Source_calls=calls,GT_read=False),True)
            if new:base.restore(base.initial);del frames,batch,base;gc.collect();torch.cuda.empty_cache()
        if model is not None:assert state_hash(model.state_dict())==mh
        put(D/'PRIOR_BARRIER.json',dict(files=files,count=128,new_Source_queries=calls,new_Source_offset_forwards=calls*2,all_lambda0_intervals_exact=True,GT_read=False))
    finally:
        put(D/'COST.json',dict(GPU_seconds=0. if start is None else time.perf_counter()-start,new_Source_queries=calls,new_Source_offset_forwards=calls*2,new_ReferDINO=0,new_DINO=0,new_updates=0))
        if lock is not None:lock.close()

def predict():
    import numpy as np
    from vg_tta.stvg_temporal_increment128_v1 import feedback,tilt
    from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr,brute_utility
    p=verify();assert read(D/'PRIOR_BARRIER.json')['count']==128;files=[]
    for r in p['records']:
        a=r['alias'];prior=load(D/'priors'/f'{a}.pt');assert sha(EVENT/'predictions'/f'{a}.pt')==r['event_prediction_sha256'];old=load(EVENT/'predictions'/f'{a}.pt');q=prior['q'];ids=prior['ids'];outputs={};audits={};support=np.triu_indices(len(ids),1)
        for arm,curve in [('E',old['readouts']['EventE']['curve'])]+[(f'Live{s}',old['readouts'][f'EventLive{s}']['curve'])for s in p['seeds']]:
            f,meta=feedback(ids,old['edges'],curve);audits[arm]=meta
            for lam in p['lambdas']:
                post=tilt(q,f,lam);m=final_mbr(ids,post)
                if lam==0:assert np.array_equal(post,q)and m['interval']==old['original_interval']
                picks=np.unique(np.linspace(0,len(support[0])-1,17).astype(int));direct=brute_utility(ids,post,picks);err=float(abs(direct-m['utility_grid'][support[0][picks],support[1][picks]]).max());assert err<1e-11
                outputs[f'{arm}_L{lam:g}']=dict(interval=m['interval'],utility=m['utility'],KL_posterior_to_prior=float(np.sum(post[post>0]*(np.log(post[post>0])-np.log(q[post>0])))),L1_prior_change=float(abs(post-q).sum()),changed=m['interval']!=old['original_interval'],posterior=post,independent_MBR_error=err)
        files.append(save(D/'predictions'/f'{a}.pt',dict(alias=a,query_ordinal=r['query_ordinal'],frame_ids=old['frame_ids'],positions8=old['positions8'],original_interval=old['original_interval'],trajectories=old['trajectories'],outputs=outputs,feedback_audits=audits,q0=q,GT_read=False)))
    put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=128,files=files,plan_sha256=sha(D/'PLAN_LOCK.json'),all_before_new_GT=True));put(D/'STATUS.json',dict(stage='sealed_before_GT',done=128,total=128),True)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','export','predict']);a=ap.parse_args()
    try:globals()[a.stage]()
    except BaseException as e:
        D.mkdir(exist_ok=True,parents=True);put(D/f'FAILURE_{time.time_ns()}.json',dict(stage=a.stage,error=str(e),type=type(e).__name__,partials_preserved=True));raise
