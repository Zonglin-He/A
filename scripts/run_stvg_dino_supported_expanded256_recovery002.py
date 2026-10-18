"""Original online-state reconstruction followed by frozen E expanded outputs.

No labels are read by lock/qualify/replay/predict. score is a separate CPU process.
"""
from pathlib import Path
import argparse, collections, gc, hashlib, json, os, sys, time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,save,load,sha
D=ROOT/'artifacts/stvg_dino_supported_expanded256_v1'
OLD=ROOT/'artifacts/stvg_opd_paper_hc2_revision_v2'
P=ROOT/'results/stvg_error_state/2026-10-18/expanded256'
ROW_PLANS={}
INPUT_BARRIERS={}


def status(stage,done,total,**kw):
    put(D/'STATUS.json',dict(status='running',stage=stage,done=done,total=total,
        pid=os.getpid(),timestamp_epoch=time.time(),**kw),True)


def lock():
    assert read(D/'CACHE_AUDIT.json')['count']==256
    assert not(D/'RECOVERY_AUTHORIZATION002.json').exists()
    old=read(OLD/'DESIGN_LOCK.json');roster=read(D/'ROSTER_LOCK.json')
    own=['scripts/run_stvg_dino_supported_expanded256_recovery002.py',
        'scripts/continue_stvg_dino_supported_expanded256_recovery002.py',
        'protocols/stvg_dino_supported_expanded256_recovery002.md',
        'vg_tta/decota_spatial_opd_tunable_v1.py','vg_tta/decota_spatial_opd_chart_revision003.py',
        'vg_tta/decota_spatial_opd_boundary_chart_proposal003.py','vg_tta/decota_spatial_opd_precision_audit_revision001.py',
        'vg_tta/decota_spatial_opd_precision_audit_revision002.py','vg_tta/decota_spatial_opd_tunable_audit_v1.py',
        'vg_tta/c1_enabling_tricks_v1.py','vg_tta/spatial_online_state_v1.py',
        'scripts/run_decota_paper_main_v1.py','scripts/run_spatial_regression_alignment_v1.py',
        'scripts/run_final_simplification_v1.py','scripts/stvg_temporal_marginal_decode_util256_v1.py',
        'vg_tta/stvg_posterior_decision_tree_v1.py','main_method/runtime.py']
    own += [str(f.relative_to(ROOT)) for f in (ROOT/'vg_tta').glob('stvg_opd_frozen*.py')]
    own += [str(f.relative_to(ROOT)) for f in (ROOT/'methods/decota_final_simplified_v1').glob('*.py')]
    own += ['vg_tta/stvg_dino_supported_correction32_v1.py','vg_tta/stvg_joint_robust_correction32_v1.py',
        'vg_tta/stvg_error_state_correction_v1.py','scripts/run_stvg_joint_robust_correction32_v1.py',
        'vg_tta/tastvg_oracle_event5_v1.py']
    deps=read(ROOT/'artifacts/stvg_dino_rollback32_v1/RUNTIME_LOCK.json')['pins']
    pins={**deps,**{f:sha(ROOT/f)for f in own}}
    for f,h in deps.items():assert sha(ROOT/f)==h,f
    auth=dict(status='authorized_by_human',authority='Explicit reply permits replay of original historical stream to restore Before, then continue',
        original_no_new_E_Student_updates=True,historical_recovery_Student_updates_authorized=True,
        original_roster_sha256=sha(D/'ROSTER_LOCK.json'),created_epoch=time.time(),deadline_epoch=time.time()+48*3600,
        configurations={ds:old['datasets'][ds]['config']for ds in ['hc2','vidstg']},
        max_arrivals={ds:roster['domains'][ds]['maximum_historical_arrival']for ds in ['hc2','vidstg']},
        new_DINO_calls=0,new_training=0,no_NEW_method_variants=True)
    put(D/'RECOVERY_AUTHORIZATION002.json',auth)
    put(D/'RECOVERY_RUNTIME002.json',dict(pins=pins,authorization_sha256=sha(D/'RECOVERY_AUTHORIZATION002.json'),
        initial_runtime_sha256=sha(D/'RUNTIME_LOCK.json'),torch_version='verify at worker startup',
        resource_limits=dict(memory_high_GiB=16,memory_max_GiB=20,swap_max_GiB=1,disk_floor_GiB=30,host_floor_GiB=6,GPU_floor_GiB=4)))
    status('authorized_original_state_recovery_pending',0,13763,GT_read=False)


def verify():
    a=read(D/'RECOVERY_AUTHORIZATION002.json');assert time.time()<a['deadline_epoch'],'finite recovery deadline reached'
    rt=read(D/'RECOVERY_RUNTIME002.json');assert sha(D/'RECOVERY_AUTHORIZATION002.json')==rt['authorization_sha256']
    for f,h in {**rt['pins'],**read(D/'RUNTIME_LOCK.json')['registries']}.items():assert sha(ROOT/f)==h,f
    return a


def gpu():
    # The existing lease is nonblocking: wait outside CUDA allocation, bounded by the original deadline.
    from scripts.run_decota_paper_main_v1 import gpu as acquire
    while True:
        verify()
        try:return acquire()
        except BlockingIOError:
            status('waiting_shared_GPU_lease',0,13763,GT_read=False)
            time.sleep(15)


def capture(model,ds,q,cache):
    import torch
    from scripts.run_decota_paper_main_v1 import frames_for,unpack_expert
    from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward
    from methods.decota_final_simplified_v1.objectives import prediction
    if ds not in ROW_PLANS:ROW_PLANS[ds]=read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']
    raw=ROW_PLANS[ds][q];subject=ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/subjects/{q:05}.json';ss=read(subject)
    assert ss['ordinal']==q and ss['caption_sha256']==hashlib.sha256(raw['input']['caption'].encode()).hexdigest() and not ss['GT_read']
    row={**raw,'parses':ss['parses'],'subject_sha256':sha(subject)};frames,ids,_=frames_for(ds,row,cache)
    tick=time.perf_counter();batch,records,base=frozen_forward(model,frames,row)
    native=prediction(base.zero['logits'],base.zero['boxes'],records,ids)
    di='vidstg_to_hc2'if ds=='hc2'else'hcstvg2_to_vidstg'
    f=OLD/'inputs'/di/'clean'/f'{q:05}.pt'
    if ds not in INPUT_BARRIERS:INPUT_BARRIERS[ds]=read(OLD/'stages'/('P1_'+ds)/'PREDICTION_BARRIER.json')
    b=INPUT_BARRIERS[ds]
    assert sha(f)==b['inputs'][str(f.relative_to(OLD))]
    old=torch.load(f,map_location='cpu',weights_only=False)
    assert not old['GT_read'] and old['parent']==q and list(ids)==list(old['frame_ids'])
    assert torch.equal(native['boxes'].cpu(),old['native_boxes']),('Source cache exactness',ds,q)
    assert native['physical_interval']==old['interval'] and native['indices']==old['indices']
    expert=unpack_expert(old['expert'])
    return row,records,base,native,expert,dict(path=str(f.relative_to(ROOT)),sha256=sha(f)),time.perf_counter()-tick


def qualify(ds):
    import torch
    from scripts.run_decota_paper_main_v1 import model_for
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.decota_spatial_opd_chart_revision003 import Installation,audit
    from vg_tta.decota_spatial_opd_tunable_v1 import fit as original_fit
    from main_method.runtime import resource_guard
    verify();lease=gpu();installation=None
    try:
        resource_guard();model=model_for('vidstg'if ds=='hc2'else'hcstvg2');mh=state_hash(model.state_dict())
        rec=next(r for r in read(ROOT/'artifacts/stvg_dino_rollback32_v1/PLAN_LOCK.json')['records']if r['dataset']==ds)
        f=ROOT/'artifacts/stvg_dino_rollback32_v1/predictions'/(rec['alias']+'.pt');old=load(f)
        row,records,base,native,expert,inp,elapsed=capture(model,ds,rec['query_ordinal'],collections.OrderedDict())
        installation=Installation();fit=installation.fit(base,old['fit']['initial'],expert,row['frame_ids'],row['key'],'on_policy',config=old['fit']['config'])
        checks=0
        def same(a,b):
            nonlocal checks
            if torch.is_tensor(a):assert torch.is_tensor(b)and a.dtype==b.dtype and torch.equal(a.cpu(),b.cpu());checks+=a.numel()
            elif isinstance(a,dict):
                for k in a:same(a[k],b[k])
            elif isinstance(a,(tuple,list)):
                assert len(a)==len(b)
                for v,w in zip(a,b):same(v,w)
            else:assert a==b;checks+=1
        same(old['fit'],{k:fit[k]for k in old['fit']})
        checked=audit(fit,expert);base.restore(base.initial);assert state_hash(model.state_dict())==mh
        out=save(D/'qualification'/(ds+'.pt'),dict(fit=fit,checks=checks,math=checked,input=inp,GT_read=False))
        put(D/('QUALIFICATION_'+ds+'.json'),dict(status='pass',count=1,source_and_original_saved_fit_exact=True,
            checks=checks,receipt=out,GT_read=False,new_DINO_calls=0,extra_recovery_fit=True))
    finally:
        if installation:installation.uninstall()
        lease.close()


def replay(ds):
    import torch,numpy as np
    from scripts.run_decota_paper_main_v1 import model_for
    from methods.decota_final_simplified_v1.tensors import state_hash,detached
    from vg_tta.spatial_online_state_v1 import arrival
    from vg_tta.c1_enabling_tricks_v1 import TrickReplay
    from vg_tta.decota_spatial_opd_chart_revision003 import Installation,audit
    from main_method.runtime import resource_guard
    a=verify();assert read(D/('QUALIFICATION_'+ds+'.json'))['status']=='pass'
    seq=read(OLD/'DESIGN_LOCK.json')['stages']['P1_'+ds]['orders']['order1'];last=a['max_arrivals'][ds]
    chosen={r['query_ordinal']:r for r in read(D/'ROSTER_LOCK.json')['records']if r['dataset']==ds}
    previous=None;predecessor=None;start=0
    while (D/'replay'/ds/f'{start:05}.receipt.json').exists():
        z=load(D/'replay'/ds/f'{start:05}.pt');assert z['arrival']==start and z['query_ordinal']==seq[start]and z['predecessor_sha256']==predecessor
        previous=z['committed'];predecessor=sha(D/'replay'/ds/f'{start:05}.pt');start+=1
    lease=gpu();installation=None
    try:
        model=model_for('vidstg'if ds=='hc2'else'hcstvg2');mh=state_hash(model.state_dict());cache=collections.OrderedDict()
        installation=Installation();files=[];started=time.perf_counter()
        for at in range(start,last+1):
            verify();resource_guard()
            available=torch.cuda.mem_get_info()[0];assert available>4*2**30,'GPU memory floor'
            q=seq[at];tick=time.perf_counter();row,records,base,native,expert,inp,capture_seconds=capture(model,ds,q,cache)
            initial=arrival(base.initial,previous,'O-split');assert torch.count_nonzero(initial['spatial.query_residual'])==0
            fit=None;math=None
            if at<last:
                fit=installation.fit(base,initial,expert,row['frame_ids'],row['key'],'on_policy',config=a['configurations'][ds])
                math=audit(fit,expert);before=fit['before']
                previous={n:torch.zeros_like(v.cpu())if n=='spatial.query_residual'else v.cpu()+(fit['state'][n]-v.cpu())*a['configurations'][ds]['writeback']for n,v in initial.items()}
            else:
                base.restore(initial)
                with torch.no_grad():before=TrickReplay(base,row['frame_ids'],row['key'],{}).values()['boxes'].detach().cpu()
            if q in chosen:
                r=chosen[q];out=D/'restored_before'/(r['alias']+'.pt')
                assert not out.exists(),'Preserve a precheckpoint capture; explicit recovery required'
                save(out,dict(alias=r['alias'],dataset=ds,query_ordinal=q,arrival=at,frame_ids=row['frame_ids'],
                    Before=before,initial=detached(initial,'cpu'),logits=detached(base.zero['logits'],'cpu'),
                    records=detached(records,'cpu'),expert=detached(expert,'cpu'),input=inp,
                    predecessor_sha256=predecessor,GT_read=False,historical_reconstruction=True))
            base.restore(base.initial)
            if at%50==0 or at==last:assert state_hash(model.state_dict())==mh
            torch.cuda.synchronize();cost=dict(capture_seconds=capture_seconds,query_seconds=time.perf_counter()-tick,
                new_Source_forwards=1,new_DINO_frames=0,recovery_fit=int(at<last),E_Student_updates=0,
                historical_gradient_calls=0 if fit is None else fit['gradient_calls'])
            rc=save(D/'replay'/ds/f'{at:05}.pt',dict(dataset=ds,arrival=at,query_ordinal=q,input=inp,
                predecessor_sha256=predecessor,fit=fit,math=math,committed=previous,compute=cost,GT_read=False))
            predecessor=rc['sha256'];status('recovering_original_order1_'+ds,at+1,last+1,
                selected_Before_captured=sum((D/'restored_before'/(r['alias']+'.receipt.json')).exists()for r in chosen.values()),GT_read=False)
            del row,records,base,native,expert,initial,fit,before;gc.collect();torch.cuda.empty_cache()
        files=[read((D/'restored_before'/(r['alias']+'.pt')).with_suffix('.receipt.json'))for r in chosen.values()]
        put(D/('BEFORE_BARRIER_'+ds+'.json'),dict(status='complete',count=128,files=files,GT_read=False,
            replay_arrivals=last+1,recovery_fits=last,replay_wall_seconds=time.perf_counter()-started))
    finally:
        if installation:installation.uninstall()
        lease.close()


def predict():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch,numpy as np
    from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table,independent_cdf
    from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr,brute_utility
    from scripts.run_stvg_joint_robust_correction32_v1 import input_pool
    from vg_tta.stvg_error_state_correction_v1 import encode
    from vg_tta.stvg_joint_robust_correction32_v1 import apply
    from vg_tta.stvg_dino_supported_correction32_v1 import estimate
    verify();files=[];cal=read(ROOT/'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json')
    assert all(read(D/('BEFORE_BARRIER_'+ds+'.json'))['count']==128 for ds in ['hc2','vidstg'])
    for r in read(D/'ROSTER_LOCK.json')['records']:
        z=load(D/'restored_before'/(r['alias']+'.pt'));ids=z['frame_ids'];base=z['Before'].double().numpy()
        grids=probabilities(z['records'],z['logits']);grid,q=mass_table(grids);assert grid==ids
        assert np.max(abs(q-independent_cdf(grids,ids)))<1e-12
        m=final_mbr(ids,q);ij=np.triu_indices(len(ids),1);best=int(np.flatnonzero((ij[0]==m['indices'][0])&(ij[1]==m['indices'][1]))[0])
        assert abs(brute_utility(ids,q,[best])[0]-m['utility'])<1e-12
        ff=input_pool(z['expert']);pools=[encode(f['boxes'])-encode(base)[f['position']]for f in ff]
        top=[pool[f['top1']]for pool,f in zip(pools,ff)];delta=np.mean(top,axis=0)if top else np.zeros(4)
        source='vidstg'if r['dataset']=='hc2'else'hc2';s=cal['source_domains'][source]
        e=estimate(pools,[f['scores']for f in ff],s['scale'],s['truncation'],1/16)
        outputs=dict(Before=base,A_Top1_mean=apply(base,delta),E_DINO_support=apply(base,e['delta']))
        files.append(save(D/'predictions'/(r['alias']+'.pt'),dict(alias=r['alias'],dataset=r['dataset'],query_ordinal=r['query_ordinal'],
            frame_ids=ids,interval=m['interval'],planned_positions=z['expert']['positions4'],frames=ff,outputs=outputs,
            estimate=e,MBR_utility=m['utility'],GT_read=False,restored_input_sha256=sha(D/'restored_before'/(r['alias']+'.pt')))))
    put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=256,files=files,
        all_outputs_before_new_target_GT=True,GT_read=False,new_E_Student_updates=0))
    assert not torch.cuda.is_initialized()


def score():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np,torch
    from scripts.score_stvg_opd_p1_v1 import truths
    from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official,box_iou
    from scripts.score_stvg_joint_robust_correction32_v1 import paired
    verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==256 and bar['all_outputs_before_new_target_GT']
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    roster=read(D/'ROSTER_LOCK.json')['records'];targets={ds:truths(ds,dict(parents=[r['query_ordinal']for r in roster if r['dataset']==ds]))for ds in ['hc2','vidstg']}
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in targets}
    features=read(ROOT/'artifacts/stvg_evidence_strata_v1/FEATURES_PRIVATE.json');feat={ds:{v['query_ordinal']:v for v in features['rows'][ds]}for ds in targets}
    rows=[];checks=0
    for r in roster:
        ds,q=r['dataset'],r['query_ordinal'];z=load(D/'predictions'/(r['alias']+'.pt'));row=plans[ds][q];gt,span=targets[ds][0][q],targets[ds][1][q]
        ids=np.array(z['frame_ids']);known=np.array([int(i)in gt for i in ids]);unobs=known&~np.isin(np.arange(len(ids)),z['planned_positions'])
        vals={}
        for arm,b in z['outputs'].items():
            v=official(b,row,gt,span,z['interval'],ds);ind=DenseTube(b,row,gt,span,clip=ds=='hc2').score(z['interval'])
            assert max(abs(v[k]-ind[k])for k in ['v','t','s'])<2e-10;checks+=3
            xy=np.column_stack((b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2))*np.array([row['input']['width'],row['input']['height']]*2)
            if ds=='hc2':xy=np.maximum(xy,0)
            iou=np.zeros(len(ids))
            if known.any():iou[known]=box_iou(xy[known],np.array([gt[int(i)]for i in ids[known]]))
            vals[arm]=dict(**v,unobserved_actual_s=float(iou[unobs].mean())if unobs.any()else None,
                unobserved_known_frames=int(unobs.sum()),R03=float(v['v']>.3),R05=float(v['v']>.5))
        assert len({v['t']for v in vals.values()})==1
        rows.append(dict(alias=r['alias'],dataset=ds,query_ordinal=q,values=vals,
            bins={k:feat[ds][q]['bins'][k]for k in ['speed_sizes_sec','target_area','event_fraction']},
            planned_slots=len(z['planned_positions']),support_frames=z['estimate']['support_frames'],rejected_frames=z['estimate']['rejected_frames']))
    def summarize(rr):
        means={a:{ep:float(np.mean([r['values'][a][ep]for r in rr if r['values'][a][ep]is not None]))if any(r['values'][a][ep]is not None for r in rr)else None
            for ep in ['v','t','s','unobserved_actual_s','R03','R05']}for a in ['Before','A_Top1_mean','E_DINO_support']}
        pairs={c:{ep:paired([None if r['values']['E_DINO_support'][ep]is None or r['values'][c][ep]is None else 100*(r['values']['E_DINO_support'][ep]-r['values'][c][ep])for r in rr])for ep in ['v','unobserved_actual_s']}for c in ['Before','A_Top1_mean']}
        return dict(parents=len(rr),queries=len(rr),query_mean=means,parent_mean=means,E_minus=pairs)
    summary={ds:summarize([r for r in rows if r['dataset']==ds])for ds in targets}
    macro={c:{ep:float(np.mean([summary[ds]['E_minus'][c][ep]['mean']for ds in targets]))for ep in ['v','unobserved_actual_s']}for c in ['Before','A_Top1_mean']}
    strata={ds:{feature:{b:summarize([r for r in rows if r['dataset']==ds and r['bins'][feature]==b])for b in sorted({r['bins'][feature]for r in rows if r['dataset']==ds})}for feature in ['speed_sizes_sec','target_area','event_fraction']}for ds in targets}
    put(P/'RESULTS.json',dict(status='complete',count=256,rows=rows,summary=summary,equal_domain_macro_pp=macro,
        strata=strata,shared_full_cache_bins=features['bins'],historically_exposed_expanded_evaluation=True,
        GT_after_all256_output_seal=True,automatic_promotion=False,old_E32_failed_gate_preserved=True,independent_metric_checks=checks))
    put(D/'CPU_COMPLETION.json',dict(status='complete',count=256,GT_read=True,result_sha256=sha(P/'RESULTS.json')))
    put(D/'GROUP_PUBLICATION_HANDOFF.json',dict(status='ready_for_root_independent_audit_and_publication',count=256,
        mandatory_next='Independent E/output/metric/historical compatibility audit, actual fixed pixels, privacy, remote two-ref CAS and archive; no claim of remote completion yet'))
    status('CPU_complete_root_publication_required',256,256,GT_read=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['lock','qualify','replay','predict','score']);p.add_argument('--dataset',choices=['hc2','vidstg']);a=p.parse_args()
    try:globals()[a.stage](a.dataset)if a.stage in ['qualify','replay']else globals()[a.stage]()
    except BaseException as e:
        put(D/('FAILURE_'+a.stage+'_'+str(a.dataset)+'_'+str(time.time_ns())+'.json'),dict(status='failed',stage=a.stage,dataset=a.dataset,error_type=type(e).__name__,error=str(e),outputs_preserved=True))
        raise
