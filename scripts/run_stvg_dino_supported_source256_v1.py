"""Frozen-source capture, three output corrections, then all-seal GT evaluation."""
from pathlib import Path
import argparse,collections,gc,hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,save,load,sha
D=ROOT/'artifacts/stvg_dino_supported_source256_v1'
PRE=ROOT/'artifacts/stvg_dino_supported_expanded256_v1'
OLD=ROOT/'artifacts/stvg_opd_paper_hc2_revision_v2'
P=ROOT/'results/stvg_error_state/2026-10-18/source_expanded256'


def status(stage,done,total=256,**kw):put(D/'STATUS.json',dict(status='running',stage=stage,done=done,total=total,pid=os.getpid(),timestamp=time.time(),**kw),True)


def normalize(ds):return 'hc2'if ds in ['hc2','hcstvg2']else'vidstg'


def prepare():
    import torch
    assert read(PRE/'CANCELLATION_RECEIPT003.json')['controller_and_worker_exited']
    roster=read(PRE/'ROSTER_LOCK.json');wanted={(r['dataset'],r['query_ordinal']):r for r in roster['records']}
    candidates=collections.defaultdict(list)
    for f in sorted((PRE/'restored_before').glob('*.pt')):
        z=load(f);key=z['dataset'],z['query_ordinal']
        if key in wanted:candidates[key].append(dict(kind='recovery_source_clock',path=str(f.relative_to(ROOT)),sha256=sha(f)))
    interval=read(ROOT/'artifacts/stvg_posterior_decision_tree_v1/PLAN_LOCK.json')
    for r in interval['files']:
        key=r['dataset'],r['query_ordinal']
        if key in wanted:candidates[key].append(dict(kind='P6_source_zero',path=r['path'],sha256=r['sha256']))
    core=read(ROOT/'artifacts/stvg_spatial_controls256_v1/ROSTER_LOCK.json')
    barrier=read(ROOT/'artifacts/stvg_spatial_controls256_v1/SOURCE_CAPTURE_BARRIER.json')
    lookup={f['path']:f for f in barrier['files']}
    for r in core['records']:
        key=normalize(r['dataset']),r['query_ordinal'];path=r['source_case_path']
        if key in wanted and path in lookup:candidates[key].append(dict(kind='core_source_zero',path=path,sha256=lookup[path]['sha256']))
    selected={};inputs={};audit=[]
    barriers={ds:read(OLD/'stages'/('P1_'+ds)/'PREDICTION_BARRIER.json')for ds in ['hc2','vidstg']}
    for r in roster['records']:
        ds,q=r['dataset'],r['query_ordinal'];di='vidstg_to_hc2'if ds=='hc2'else'hcstvg2_to_vidstg'
        path=OLD/'inputs'/di/'clean'/f'{q:05}.pt';assert sha(path)==barriers[ds]['inputs'][str(path.relative_to(OLD))]
        old=torch.load(path,map_location='cpu',weights_only=False);assert not old['GT_read'];inputs[str(path.relative_to(ROOT))]=sha(path)
        chosen=None
        for c in candidates[ds,q]:
            f=ROOT/c['path'];assert sha(f)==c['sha256'];z=torch.load(f,map_location='cpu',weights_only=False)
            if c['kind']=='recovery_source_clock':
                assert z['input']['sha256']==sha(path)
                logits=z['logits'];records=z['records'];ids=z['frame_ids'];boxes=old['native_boxes'];iv=old['interval'];pixel=old['pixel_sha256']
            else:
                cap=z['capture']if c['kind']=='P6_source_zero'else z
                logits=cap['zero']['logits'];records=cap['records'];ids=z['frame_ids'];boxes=cap['zero']['boxes']
                iv=z['predictions']['Native']['physical_interval']if c['kind']=='P6_source_zero'else z['Native_interval'];pixel=z['pixel_sha256']
            valid=list(ids)==old['frame_ids'] and torch.equal(boxes.cpu(),old['native_boxes'])and list(iv)==old['interval']and pixel==old['pixel_sha256']
            audit.append(dict(alias=r['alias'],kind=c['kind'],exact_input_compatibility=valid))
            if valid:
                chosen=c;inputs[c['path']]=c['sha256'];break
        selected[r['alias']]=chosen
    own=['scripts/run_stvg_dino_supported_source256_v1.py','scripts/continue_stvg_dino_supported_source256_v1.py',
        'scripts/finalize_stvg_dino_supported_source256_v1.py','scripts/publish_stvg_dino_supported_source256_v1.py',
        'protocols/stvg_dino_supported_source256_v1.md','scripts/run_stvg_dino_supported_expanded256_recovery002.py',
        'vg_tta/stvg_dino_supported_correction32_v1.py','vg_tta/stvg_joint_robust_correction32_v1.py','vg_tta/stvg_error_state_correction_v1.py',
        'scripts/run_stvg_joint_robust_correction32_v1.py','scripts/stvg_temporal_marginal_decode_util256_v1.py',
        'vg_tta/stvg_posterior_decision_tree_v1.py','vg_tta/tastvg_oracle_event5_v1.py']
    rt=read(PRE/'RECOVERY_RUNTIME002.json');pins={**rt['pins'],**{f:sha(ROOT/f)for f in own}}
    for f,h in rt['pins'].items():assert sha(ROOT/f)==h
    put(D/'PLAN_LOCK.json',dict(status='locked_before_capture_and_GT',records=roster['records'],count=256,
        same_roster_sha256=sha(PRE/'ROSTER_LOCK.json'),arms=['Source_no_correction','A_Top1_mean','E_DINO_support'],beta=1/16,
        baseline='frozen Source; no inherited state',clock='same Source final-distribution MBR',
        historically_exposed_expanded_evaluation=True,original_E32_failed_gate_preserved=True,
        source_cache_reuse=selected,new_Student_updates=0,new_DINO=0,new_training=0,
        user_authorized_start_change=True,created_epoch=time.time(),deadline_epoch=time.time()+8*3600))
    put(D/'RUNTIME_LOCK.json',dict(pins=pins,inputs=inputs,registries=read(PRE/'RUNTIME_LOCK.json')['registries'],
        calibration_sha256=sha(ROOT/'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json')))
    put(D/'INPUT_READBACK.json',dict(status='pass',count=256,original_Source_DINO_inputs_hash_exact=True,
        full_Source_cache_reused=sum(c is not None for c in selected.values()),new_frozen_captures_required=sum(c is None for c in selected.values()),
        compatibility=audit,old_recovery_cancelled_and_preserved=True,GT_read=False))
    status('prepared',0,GT_read=False)
    print(json.dumps({k:read(D/'INPUT_READBACK.json')[k]for k in ['count','full_Source_cache_reused','new_frozen_captures_required']}))


def verify():
    p=read(D/'PLAN_LOCK.json');assert time.time()<p['deadline_epoch'],'Original8h deadline reached'
    l=read(D/'RUNTIME_LOCK.json')
    for f,h in {**l['pins'],**l['registries']}.items():assert sha(ROOT/f)==h,f
    assert sha(PRE/'ROSTER_LOCK.json')==p['same_roster_sha256'];return p


def capture(ds):
    import torch
    from scripts.run_stvg_dino_supported_expanded256_recovery002 import capture as fresh
    from scripts.run_decota_paper_main_v1 import gpu,model_for,unpack_expert
    from methods.decota_final_simplified_v1.tensors import state_hash,detached
    from main_method.runtime import resource_guard
    p=verify();rows=[r for r in p['records']if r['dataset']==ds];model=None;lease=None;cache=collections.OrderedDict();started=time.perf_counter();files=[]
    try:
        for i,r in enumerate(rows):
            verify();resource_guard();dest=D/'source'/(r['alias']+'.pt')
            if dest.with_suffix('.receipt.json').exists():files.append(read(dest.with_suffix('.receipt.json')));continue
            di='vidstg_to_hc2'if ds=='hc2'else'hcstvg2_to_vidstg';path=OLD/'inputs'/di/'clean'/f'{r["query_ordinal"]:05}.pt'
            assert sha(path)==read(D/'RUNTIME_LOCK.json')['inputs'][str(path.relative_to(ROOT))]
            original=torch.load(path,map_location='cpu',weights_only=False);reuse=p['source_cache_reuse'][r['alias']];tick=time.perf_counter()
            if reuse:
                assert sha(ROOT/reuse['path'])==reuse['sha256'];z=torch.load(ROOT/reuse['path'],map_location='cpu',weights_only=False)
                c=z if reuse['kind']=='recovery_source_clock'else z['capture']if reuse['kind']=='P6_source_zero'else z
                zero=c if reuse['kind']=='recovery_source_clock'else c['zero'];records=c['records'];logits=zero['logits'];Source=original['native_boxes'];expert=unpack_expert(original['expert']);capture_seconds=0.;new=0
            else:
                if model is None:
                    while lease is None:
                        verify()
                        try:lease=gpu()
                        except BlockingIOError:status('waiting_shared_GPU_lease',len(files),128,dataset=ds,GT_read=False);time.sleep(15)
                    model=model_for('vidstg'if ds=='hc2'else'hcstvg2');mh=state_hash(model.state_dict())
                assert torch.cuda.mem_get_info()[0]>4*2**30
                row,records,base,native,expert,inp,capture_seconds=fresh(model,ds,r['query_ordinal'],cache)
                Source=native['boxes'].cpu();logits=detached(base.zero['logits'],'cpu');records=detached(records,'cpu');new=1
                assert torch.equal(Source,original['native_boxes'])
                base.restore(base.initial);del base,row,native;gc.collect();torch.cuda.empty_cache()
            assert all(z.dtype==torch.float32 for z in logits)
            files.append(save(dest,dict(alias=r['alias'],dataset=ds,query_ordinal=r['query_ordinal'],frame_ids=original['frame_ids'],
                Source=Source,logits=logits,records=records,expert=detached(expert,'cpu'),Native_interval=original['interval'],
                original_input_sha256=sha(path),GT_read=False,source_frozen=True,Student_updates=0,
                compute=dict(reused_full_capture=bool(reuse),new_Source_forwards=new,new_DINO_frames=0,new_Student_updates=0,
                    capture_seconds=capture_seconds,case_seconds=time.perf_counter()-tick),reuse_provenance=reuse)))
            status('Source_capture_'+ds,i+1,128,dataset=ds,GT_read=False)
        if model is not None:assert state_hash(model.state_dict())==mh
        put(D/('SOURCE_BARRIER_'+ds+'.json'),dict(status='complete',count=128,files=files,GT_read=False,wall_seconds=time.perf_counter()-started))
    finally:
        if lease:lease.close()


def predict():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch,numpy as np
    from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table,independent_cdf
    from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr,brute_utility
    from scripts.run_stvg_joint_robust_correction32_v1 import input_pool
    from vg_tta.stvg_error_state_correction_v1 import encode
    from vg_tta.stvg_joint_robust_correction32_v1 import apply
    from vg_tta.stvg_dino_supported_correction32_v1 import estimate
    p=verify();assert all(read(D/('SOURCE_BARRIER_'+ds+'.json'))['count']==128 for ds in ['hc2','vidstg']);files=[]
    cal=read(ROOT/'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json');started=time.perf_counter()
    for i,r in enumerate(p['records']):
        z=load(D/'source'/(r['alias']+'.pt'));base=z['Source'].double().numpy();ids=z['frame_ids'];g=probabilities(z['records'],z['logits']);U,q=mass_table(g)
        assert U==ids and np.max(abs(q-independent_cdf(g,U)))<1e-12
        m=final_mbr(ids,q);si,ei=np.triu_indices(len(ids),1);best=np.flatnonzero((si==m['indices'][0])&(ei==m['indices'][1]))[0]
        assert abs(brute_utility(ids,q,[best])[0]-m['utility'])<1e-12
        frames=input_pool(z['expert']);pools=[encode(f['boxes'])-encode(base)[f['position']]for f in frames]
        top=[v[f['top1']]for f,v in zip(frames,pools)];delta=np.mean(top,0)if top else np.zeros(4)
        setting=cal['source_domains']['vidstg'if r['dataset']=='hc2'else'hc2'];e=estimate(pools,[f['scores']for f in frames],setting['scale'],setting['truncation'],1/16)
        outputs=dict(Source_no_correction=base,A_Top1_mean=apply(base,delta),E_DINO_support=apply(base,e['delta']))
        files.append(save(D/'predictions'/(r['alias']+'.pt'),dict(alias=r['alias'],dataset=r['dataset'],query_ordinal=r['query_ordinal'],frame_ids=ids,
            interval=m['interval'],Native_interval=z['Native_interval'],planned_positions=z['expert']['positions4'],frames=frames,estimate=e,outputs=outputs,
            source_capture_sha256=sha(D/'source'/(r['alias']+'.pt')),GT_read=False,new_Student_updates=0)))
        status('sealing_Source_A_E',i+1,GT_read=False)
    put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=256,files=files,all_outputs_before_GT=True,GT_read=False,
        new_Student_updates=0,new_DINO_frames=0,CPU_wall_seconds=time.perf_counter()-started))
    assert not torch.cuda.is_initialized()


def score():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch,numpy as np
    from scripts.score_stvg_opd_p1_v1 import truths
    from vg_tta.tastvg_oracle_event5_v1 import official,DenseTube,box_iou
    from scripts.score_stvg_joint_robust_correction32_v1 import paired
    p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==256 and bar['all_outputs_before_GT']
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    roster=p['records'];target={ds:truths(ds,dict(parents=[r['query_ordinal']for r in roster if r['dataset']==ds]))for ds in ['hc2','vidstg']}
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in target};feat=read(ROOT/'artifacts/stvg_evidence_strata_v1/FEATURES_PRIVATE.json');features={ds:{f['query_ordinal']:f for f in feat['rows'][ds]}for ds in target}
    rows=[];checks=0
    for r in roster:
        z=load(D/'predictions'/(r['alias']+'.pt'));ds,q=r['dataset'],r['query_ordinal'];row=plans[ds][q];gt,span=target[ds][0][q],target[ds][1][q];ids=np.array(z['frame_ids']);known=np.array([int(i)in gt for i in ids]);unobs=known&~np.isin(np.arange(len(ids)),z['planned_positions']);values={}
        for arm,b in z['outputs'].items():
            v=official(b,row,gt,span,z['interval'],ds);other=DenseTube(b,row,gt,span,clip=ds=='hc2').score(z['interval']);assert max(abs(v[k]-other[k])for k in ['v','t','s'])<2e-10;checks+=3
            xy=np.column_stack((b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2))*np.array([row['input']['width'],row['input']['height']]*2)
            if ds=='hc2':xy=np.maximum(xy,0)
            iou=np.zeros(len(ids))
            if known.any():iou[known]=box_iou(xy[known],np.array([gt[int(fid)]for fid in ids[known]]))
            values[arm]=dict(**v,unobserved_actual_s=float(iou[unobs].mean())if unobs.any()else None,unobserved_known_frames=int(unobs.sum()),R03=float(v['v']>.3),R05=float(v['v']>.5))
        assert len({v['t']for v in values.values()})==1
        rows.append(dict(alias=r['alias'],dataset=ds,query_ordinal=q,values=values,bins={k:features[ds][q]['bins'][k]for k in ['speed_sizes_sec','target_area','event_fraction']},planned_slots=len(z['planned_positions']),support_frames=z['estimate']['support_frames'],rejected_frames=z['estimate']['rejected_frames']))
    arms=p['arms']
    def summary(rr):
        means={a:{ep:float(np.mean([r['values'][a][ep]for r in rr if r['values'][a][ep]is not None]))if any(r['values'][a][ep]is not None for r in rr)else None for ep in ['v','t','s','unobserved_actual_s','R03','R05']}for a in arms}
        dif={c:{ep:paired([None if r['values']['E_DINO_support'][ep]is None or r['values'][c][ep]is None else 100*(r['values']['E_DINO_support'][ep]-r['values'][c][ep])for r in rr])for ep in ['v','unobserved_actual_s']}for c in arms[:2]}
        for c in dif:
            for ep,v in dif[c].items():
                x=np.array(v.get('per_parent_pp',[]));v.update(severe_loss_below_minus20pp=int((x< -20).sum()),severe_loss_fraction=float((x< -5).mean())if len(x)else None)
        return dict(queries=len(rr),parents=len(rr),query_mean=means,parent_mean=means,E_minus=dif)
    groups={ds:summary([r for r in rows if r['dataset']==ds])for ds in target};macro={};rng=np.random.default_rng(20261018)
    for c in arms[:2]:
        macro[c]={}
        for ep in ['v','unobserved_actual_s']:
            arr=[np.array(groups[ds]['E_minus'][c][ep]['per_parent_pp'])for ds in target]
            boot=[v[rng.integers(len(v),size=(10000,len(v)))].mean(1)for v in arr]
            macro[c][ep]=dict(mean_pp=float(np.mean([v.mean()for v in arr])),CI95_pp=np.quantile(np.mean(boot,0),[.025,.975]).tolist(),equal_domain_weight=True)
    strata={ds:{feature:{b:summary([r for r in rows if r['dataset']==ds and r['bins'][feature]==b])for b in sorted({r['bins'][feature]for r in rows if r['dataset']==ds})}for feature in ['speed_sizes_sec','target_area','event_fraction']}for ds in target}
    put(P/'RESULTS.json',dict(status='complete',count=256,baseline='frozen Source, not historical Before',rows=rows,summary=groups,equal_domain_macro=macro,strata=strata,shared_bin_edges=feat['bins'],historically_exposed=True,GT_after_all_outputs_sealed=True,automatic_promotion=False,old_E32_failed_gate_preserved=True,independent_dense_metric_checks=checks))
    put(D/'CPU_COMPLETION.json',dict(status='complete',count=256,GT_read=True,result_sha256=sha(P/'RESULTS.json')))
    status('CPU_complete_root_audit_required',256,GT_read=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','capture','predict','score']);p.add_argument('--dataset',choices=['hc2','vidstg']);a=p.parse_args()
    try:globals()[a.stage](a.dataset)if a.stage=='capture'else globals()[a.stage]()
    except BaseException as e:
        put(D/('FAILURE_'+a.stage+'_'+str(time.time_ns())+'.json'),dict(status='failed',stage=a.stage,error_type=type(e).__name__,error=str(e),all_partials_preserved=True));raise
