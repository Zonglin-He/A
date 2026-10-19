"""All-three finite observation trial; real predictions seal before GT scoring."""
import collections,copy,gc,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=ROOT/'artifacts/stvg_observation_factorial32_v1'
P=ROOT/'results/stvg_error_state/2026-10-19/observation_factorial32'
PRE=ROOT/'artifacts/stvg_dino_endpoint_confirm32_v1'
ARMS=['E_MBR','S_crop','T_dense','ST','T_gridonly']
CODE=['vg_tta/stvg_observation_factorial32_v1.py','scripts/run_stvg_observation_factorial32_v1.py','protocols/stvg_observation_factorial32_v1.md']
VER=None

def verify(full=False):
    global VER
    from main_method.runtime import PinVerifier,resource_guard
    p=read(D/'PLAN_LOCK.json');r=read(D/'RUNTIME_LOCK.json')
    assert sha(D/'PLAN_LOCK.json')==r['plan_sha256']
    pins=dict(r['pins'])
    for f in sorted((D/'amendments').glob('*.json')):
        amendment=read(f);assert amendment['original_runtime_sha256']==sha(D/'RUNTIME_LOCK.json')
        pins.update(amendment['replacement_pins'])
    if VER is None:VER=PinVerifier(pins)
    VER.verify(full=full);resource_guard();return p

def prepare():
    from vg_tta.stvg_observation_factorial32_v1 import dense_ids,properties
    assert not D.exists();D.mkdir();P.mkdir();rs=read(PRE/'PLAN_LOCK.json')['records'];views=[]
    own=CODE+['methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json',
              'scripts/stvg_temporal_marginal_decode_util256_v1.py','vg_tta/stvg_posterior_decision_tree_v1.py',
              'vg_tta/stvg_dino_supported_correction32_v1.py','vg_tta/stvg_joint_robust_correction32_v1.py',
              'vg_tta/stvg_error_state_correction_v1.py','scripts/score_stvg_opd_p1_v1.py',
              'scripts/run_stvg_dino_local_residual32_v1.py','scripts/score_stvg_joint_robust_correction32_v1.py',
              'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json']
    oldpins=read(ROOT/'artifacts/stvg_e_retaining1792_route32_v1/RUNTIME_LOCK.json')['pins']
    own += [f for f in oldpins if f.startswith(('external/TA-STVG/','methods/decota_final_simplified_v1/','checkpoints/'))]
    own += ['scripts/run_decota_paper_main_v1.py','scripts/run_spatial_regression_alignment_v1.py','scripts/run_final_simplification_v1.py',
            'vg_tta/exact_frame_decode_audit_v2.py','vg_tta/tastvg_paper48_hc2_decode_v1.py']
    own += [str(f.relative_to(ROOT))for f in (ROOT/'vg_tta').glob('stvg_opd_frozen*.py')]
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT,EXPERT_SHA256
    own += [str(Path(EXPERT_SNAPSHOT)/'model.safetensors')]
    for r in rs:
        a=r['alias'];e=load(PRE/'E'/(a+'.pt'));ids=e['frame_ids'];new=dense_ids(ids)
        assert len(new[::2])<=200 and len(new[1::2])<=200
        views.append(dict(alias=a,dataset=r['dataset'],old_nodes=len(ids),dense_nodes=len(new),new_physical_nodes=len(set(new)-set(ids))))
        own += [str((PRE/sub/(a+'.pt')).relative_to(ROOT))for sub in ['source','E']]
        own += [f'artifacts/decota_paper_experiments_v1/{r["dataset"]}/subjects/{r["query_ordinal"]:05}.json']
    own += [f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json'for ds in ['hc2','vidstg']]
    put(D/'PLAN_LOCK.json',dict(status='locked_before_new_model_or_GT',count=32,records=rs,views=views,arms=ARMS,
        GPU_seconds_cap=900,wall_deadline_epoch=time.time()+3600,crop_context_factor=2.,new_DINO_cap=128,
        TA_dense_queries_cap=32,TA_original_offset_forwards_cap=64,spatial_rule='replace accepted crop pools, fallback original otherwise; original Source-relative E',
        time_rule='add integer midpoint, same full coverage, same physical final-envelope MBR',new_student_updates=0,OPD_not_run=True,
        no_automatic_followup=True,GT_after_all_outputs=True,development_exposed=True,created_epoch=time.time()))
    put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins={f:sha(ROOT/f)for f in sorted(set(own))}))
    put(D/'CPU_PROPERTIES.json',properties())
    from methods.decota_final_simplified_v1.config import MethodConfig
    put(P/'CONFIG_PUBLIC.json',dict(status='locked',count=32,per_domain=16,arms=ARMS,crop_factor=2.,views=views,
        source_checkpoints={ds:dict(checkpoint=cfg.checkpoint,sha256=cfg.checkpoint_sha256)for ds,cfg in [
            ('hc2',MethodConfig.for_direction('vid_to_hc1')),('vidstg',MethodConfig.for_direction('hc2_to_vid'))]},
        DINO_checkpoint_sha256=EXPERT_SHA256,source_relative_E=True,conditional_prompt_unchanged=True,GT_online=False,
        OPD_not_run=True,new_training_heads_models=0,formal64_excluded=True,gridonly_control=True,
        temporal_readout='same exact independent-offset envelope posterior and physical expected-tIoU MBR'))
    print('LOCKED all3 candidates,32 parents, dense nodes',sum(v['dense_nodes']for v in views),flush=True)

def spatial(expert,row,frames,s,e,check):
    import numpy as np,torch
    from methods.decota_final_simplified_v1.observations import ContextView,probe_from_detection
    from scripts.run_stvg_joint_robust_correction32_v1 import input_pool
    from vg_tta.stvg_observation_factorial32_v1 import crop_bounds,lift_boxes
    from vg_tta.stvg_error_state_correction_v1 import encode
    from vg_tta.stvg_dino_supported_correction32_v1 import estimate
    from vg_tta.stvg_joint_robust_correction32_v1 import apply
    context=row['parses']['context'];fallback=row['parses']['old']
    eligible=context['eligible'] and len(expert.processor.tokenizer(context['context'])['input_ids'])<=256
    old=input_pool(s['expert']);pool={f['position']:f for f in old};original=copy.deepcopy(pool);obs=[];calls=0
    W,H=frames.shape[2],frames.shape[1]
    for pos in e['planned_positions']:
        check();bounds=crop_bounds(e['E'][pos],W,H);x,y,r,d=bounds
        if not eligible and not fallback['phrase']:
            obs.append(dict(position=pos,called=False,reason='original_unresolved_condition',bounds=bounds));continue
        if r<=x or d<=y:
            obs.append(dict(position=pos,called=False,reason='empty_clipped_crop',bounds=bounds));continue
        if ('original',pos)in s['expert']['observations']:
            assert hashlib.sha256(frames[pos].tobytes()).hexdigest()==s['expert']['observations'][('original',pos)]['receipt']['rgb_sha256']
        rgb=np.ascontiguousarray(frames[pos,y:d,x:r]);tick=time.perf_counter()
        if eligible:
            det=ContextView(expert,context)(rgb,context['context'],context['entity'])
            probe=probe_from_detection(det,pos,s['frame_ids'][pos])
        else:
            det=expert(rgb,fallback['phrase'],fallback['entity'])
            probe=dict(accepted=det['accepted'],reason=det['reason'],margin=det['margin'],
                boxes=torch.tensor([det['box']])if det['box']else torch.empty(0,4),
                target_scores=[det['score']]if det['box']else[],candidate_ids=[0]if det['box']else[])
        local=np.asarray(probe['boxes'],np.float64).reshape(-1,4);mapped=lift_boxes(local,bounds,W,H)
        probe={**probe,'boxes':mapped};calls+=1
        if probe['accepted']:
            assert 0<len(mapped)<=3 and np.isfinite(mapped).all() and (mapped[:,2:]>0).all()
            pool[pos]=dict(position=pos,boxes=mapped,scores=np.asarray(probe['target_scores'],float),candidate_ids=probe['candidate_ids'])
        obs.append(dict(position=pos,called=True,bounds=bounds,rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            original_rgb_sha256=hashlib.sha256(frames[pos].tobytes()).hexdigest(),probe=probe,local_boxes=local,
            detector=det,forward_seconds=det['seconds'],seconds=time.perf_counter()-tick,
            accepted_before=pos in original,full_image_crop=(bounds==[0,0,W,H])))
        check()
    settings=read(ROOT/'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json')['source_domains']['vidstg'if row['dataset']=='hc2'else'hc2']
    source=np.asarray(s['Source'],np.float64)
    def estimate_pool(pp):
        fs=[pp[k]for k in sorted(pp)]
        return estimate([encode(f['boxes'])-encode(source)[f['position']]for f in fs],
                        [f['scores']for f in fs],settings['scale'],settings['truncation'],1/16)
    prior=estimate_pool(original);assert np.array_equal(apply(source,prior['delta']),e['E'])
    result=estimate_pool(pool);out=apply(source,result['delta'])
    return dict(boxes=out,estimate=result,observations=obs,new_DINO=calls,pool_positions=sorted(pool),
                fallback_original_pool_positions=[p for p in original if not any(z.get('probe',{}).get('accepted',False)and z['position']==p for z in obs)])

def measure():
    import torch,numpy as np
    from scripts.run_decota_paper_main_v1 import gpu,model_for,read_row,frames_for
    from methods.decota_final_simplified_v1.observations import SpatialExpert
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT,EXPERT_SHA256,MethodConfig
    from methods.decota_final_simplified_v1.tensors import state_hash,detached
    from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward
    from vg_tta.stvg_observation_factorial32_v1 import dense_ids,envelope,embed
    from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
    p=verify(True);assert not (D/'GLOBAL_PREDICTION_BARRIER.json').exists()
    previous=sum(read(f)['GPU_seconds']for f in (D/'allocations').glob('*.json'))
    lease=gpu();start=time.perf_counter();model=None;domain=None;files=[];count=0;dino_calls=0;dense_queries=0;done=False
    def check():
        verify();assert time.time()<p['wall_deadline_epoch'];assert previous+time.perf_counter()-start+3<900
        assert torch.cuda.mem_get_info()[0]>4*2**30
    try:
        check();expert=SpatialExpert(ROOT/EXPERT_SNAPSHOT);eh=state_hash(expert.model.state_dict());cache=collections.OrderedDict()
        provenance={};order=[next(r for r in p['records']if r['alias']==a)for a in ['NH001','NV001']]+[r for r in p['records']if r['alias']not in ['NH001','NV001']]
        for i,r in enumerate(order):
            check();a=r['alias'];ds=r['dataset'];dest=D/'predictions'/(a+'.pt')
            if dest.exists():
                z=load(dest);files.append(read(dest.with_suffix('.receipt.json')));count+=1
                dino_calls+=z['compute']['new_DINO'];dense_queries+=1;print('reuse',a,flush=True);continue
            tick=time.perf_counter();s=load(PRE/'source'/(a+'.pt'));e=load(PRE/'E'/(a+'.pt'))
            row=read_row(ds,r['query_ordinal']);row['dataset']=ds;frames,ids,_=frames_for(ds,row,cache)
            assert ids==s['frame_ids']==e['frame_ids'];oldids,q0,oldstat=envelope(s['records'],s['logits']);m0=final_mbr(oldids,q0)
            assert m0['interval']==e['interval'];new=dense_ids(ids)
            sp=spatial(expert,row,frames,s,e,check);dino_calls+=sp['new_DINO'];spsec=time.perf_counter()-tick
            # Release original decoded-frame cache before dense raw RGB capture.
            del frames;cache.clear();gc.collect()
            if ds!=domain:
                if model is not None:del model;gc.collect();torch.cuda.empty_cache()
                model=model_for('vidstg'if ds=='hc2'else'hcstvg2');domain=ds;mh=state_hash(model.state_dict())
                cfg=MethodConfig.for_direction('vid_to_hc1'if ds=='hc2'else'hc2_to_vid')
                provenance[ds]=dict(checkpoint=cfg.checkpoint,checkpoint_sha256=cfg.checkpoint_sha256,model_state_sha256=mh,
                    MAX_VIDEO_LEN=int(model.cfg.INPUT.MAX_VIDEO_LEN))
            assert all(len(new[o::2])<=model.cfg.INPUT.MAX_VIDEO_LEN for o in [0,1])
            dr=copy.deepcopy(row);dr['frame_ids']=new;dr['input']['frame_ids']=new
            check();dt=time.perf_counter();df,di,_=frames_for(ds,dr,cache)
            # True new RGB, unchanged physical full coverage, original temporal clock.
            assert di==new and di[0]==ids[0]and di[-1]==ids[-1]
            for z in sp['observations']:
                if z['called']:assert hashlib.sha256(df[new.index(ids[z['position']])].tobytes()).hexdigest()==z['original_rgb_sha256']
            with torch.no_grad():batch,recs,base=frozen_forward(model,df,dr)
            logits=detached(base.zero['logits'],'cpu');recs=detached(recs,'cpu');dense_queries+=1
            nd,qt,stat=envelope(recs,logits);assert nd==new;mt=final_mbr(new,qt);mg=final_mbr(new,embed(ids,q0,new))
            temporal=dict(records=recs,logits=logits,posterior=qt,original_posterior=q0,
                          dense_mbr=mt,gridonly_mbr=mg,CDF_statistics=stat,original_CDF_statistics=oldstat,
                          original_MBR_exact=True,seconds=time.perf_counter()-dt)
            del df,batch,base;cache.clear();gc.collect();torch.cuda.empty_cache();check()
            assert state_hash(model.state_dict())==mh and state_hash(expert.model.state_dict())==eh
            outputs=dict(E_MBR=np.asarray(e['E']).copy(),S_crop=sp['boxes'],T_dense=np.asarray(e['E']).copy(),
                         ST=sp['boxes'].copy(),T_gridonly=np.asarray(e['E']).copy())
            intervals=dict(E_MBR=e['interval'],S_crop=e['interval'],T_dense=mt['interval'],ST=mt['interval'],T_gridonly=mg['interval'])
            assert all(np.isfinite(b).all()and(b[:,2:]>0).all()for b in outputs.values())
            cost=dict(new_DINO=sp['new_DINO'],dense_TA_queries=1,dense_physical_offset_full_forwards=2,
                spatial_including_original_decode_seconds=spsec,dense_including_decode_seconds=temporal['seconds'],case_seconds=time.perf_counter()-tick)
            files.append(save(dest,dict(alias=a,dataset=ds,query_ordinal=r['query_ordinal'],frame_ids=ids,dense_frame_ids=new,
                planned_positions=e['planned_positions'],outputs=outputs,intervals=intervals,spatial=sp,temporal=temporal,
                compute=cost,GT_read=False,new_student_updates=0,OPD_not_run=True)))
            count+=1;put(D/'STATUS.json',dict(status='running',done=count,total=32,last=a,pid=os.getpid(),GT_read=False),True)
            print(a,'sealed','DINO',sp['new_DINO'],'frames',len(ids),'->',len(new),'time',e['interval'],'->',mt['interval'],
                  'GPU held',round(time.perf_counter()-start,2),flush=True)
            if count==2:
                predicted=(time.perf_counter()-start)*16
                put(D/'PILOT_COST.json',dict(included_queries=2,measured_GPU_seconds=time.perf_counter()-start,
                    simple_linear_full32_estimate_seconds=predicted,not_a_runtime_guarantee=True))
                assert predicted<850,'pilot extrapolation exceeds finite budget; preserve partials'
        check();verify(True);assert count==32 and dino_calls<=128 and dense_queries==32
        put(P/'MODEL_PROVENANCE.json',dict(TA=provenance,DINO_sha256=EXPERT_SHA256,new_weights_training=False))
        put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,all_outputs_before_GT=True,
            new_DINO=dino_calls,dense_queries=dense_queries,GPU_allocation_seconds=previous+time.perf_counter()-start,
            model_weights_unchanged=True,OPD_not_run=True,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        done=True
    finally:
        elapsed=time.perf_counter()-start;lease.close()
        put(D/'allocations'/f'{time.time_ns()}.json',dict(GPU_seconds=elapsed,complete=done,all_failed_work_included=True))

def score():
    import torch,numpy as np
    torch.set_num_threads(4);p=verify(True);bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32
    assert not torch.cuda.is_initialized()
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    from scripts.score_stvg_opd_p1_v1 import truths
    from scripts.run_stvg_dino_local_residual32_v1 import mean
    from scripts.score_stvg_joint_robust_correction32_v1 import paired
    from scripts.score_stvg_dino_mask_align32_v1 import iou_gt
    from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
    g={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']}
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in g};rows=[];diag=[]
    for r in p['records']:
        a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];z=load(D/'predictions'/(a+'.pt'));row=plans[ds][q];gt=g[ds][0][q];span=g[ds][1][q]
        ids=z['frame_ids'];known=np.array([int(f)in gt for f in ids]);observed=np.isin(np.arange(len(ids)),z['planned_positions']);values={}
        for arm,boxes in z['outputs'].items():
            iv=z['intervals'][arm];tube=DenseTube(boxes,row,gt,span,clip=ds=='hc2');met=official(boxes,row,gt,span,iv,ds)
            assert max(abs(met[k]-tube.score(iv)[k])for k in ['v','t','s'])<2e-10
            si=[iou_gt(boxes[j],gt[int(f)],row,ds)if known[j]else None for j,f in enumerate(ids)]
            values[arm]=dict(**met,observed_actual_s=mean([si[j]for j in range(len(ids))if known[j]and observed[j]]),
                unobserved_actual_s=mean([si[j]for j in range(len(ids))if known[j]and not observed[j]]),
                start_error_frames=int(iv[0]-span[0]),end_error_frames=int(iv[1]-span[1]),length_frames=int(iv[1]-iv[0]),
                zero_temporal_overlap=bool(min(iv[1],span[1])<=max(iv[0],span[0])))
        assert values['E_MBR']['t']==values['S_crop']['t']
        for arm in ['T_dense','T_gridonly']:
            assert values[arm]['s']==values['E_MBR']['s'] and values[arm]['unobserved_actual_s']==values['E_MBR']['unobserved_actual_s']
        assert values['ST']['t']==values['T_dense']['t'] and values['ST']['s']==values['S_crop']['s']
        changes={arm:{ep:100*(values[arm][ep]-values['E_MBR'][ep])if values[arm][ep]is not None and values['E_MBR'][ep]is not None else None
                 for ep in ['v','t','s','unobserved_actual_s']}for arm in ARMS[1:]}
        rows.append(dict(alias=a,dataset=ds,values=values,changes=changes,teacher_calls=z['compute']['new_DINO'],
            original_nodes=len(ids),dense_nodes=len(z['dense_frame_ids']),
            time_changed={arm:z['intervals'][arm]!=z['intervals']['E_MBR']for arm in ARMS},
            interaction_v_pp=100*(values['ST']['v']-values['S_crop']['v']-values['T_dense']['v']+values['E_MBR']['v']),
            known_unobserved_nodes=int((known&~observed).sum()),unknown_unobserved_nodes=int((~known&~observed).sum()),compute=z['compute']))
        for ob in z['spatial']['observations']:
            pos=ob['position'];fid=int(ids[pos]);known_gt=fid in gt;pr=ob.get('probe',{});bb=np.asarray(pr.get('boxes',[])).reshape(-1,4)
            win=int(np.argmax(pr['target_scores']))if len(bb)else None
            diag.append(dict(alias=a,dataset=ds,position=pos,called=ob['called'],accepted=pr.get('accepted',False),
                reason=pr.get('reason',ob.get('reason')),known_GT=known_gt,full_image_crop=ob.get('full_image_crop',False),
                E_IoU=iou_gt(z['outputs']['E_MBR'][pos],gt[fid],row,ds)if known_gt else None,
                detector_top_IoU=iou_gt(bb[win],gt[fid],row,ds)if known_gt and win is not None else None,
                propagated_S_IoU=iou_gt(z['outputs']['S_crop'][pos],gt[fid],row,ds)if known_gt else None,
                crop_contains_GT_center=None if not known_gt else bool(ob['bounds'][0]<=(gt[fid][0]+gt[fid][2])/2<ob['bounds'][2]and ob['bounds'][1]<=(gt[fid][1]+gt[fid][3])/2<ob['bounds'][3])))
    groups={};eps=['v','t','s','observed_actual_s','unobserved_actual_s']
    for ds in g:
        rr=[r for r in rows if r['dataset']==ds]
        groups[ds]=dict(parents=len(rr),metrics={arm:{ep:mean([r['values'][arm][ep]for r in rr])for ep in eps}for arm in ARMS},
            paired={a+'_minus_'+b:{ep:paired([None if r['values'][a][ep]is None or r['values'][b][ep]is None else 100*(r['values'][a][ep]-r['values'][b][ep])for r in rr])for ep in eps}
                    for a,b in [(a,'E_MBR')for a in ARMS[1:]]+[('T_dense','T_gridonly'),('ST','S_crop'),('ST','T_dense')]},
            interaction_v_pp=paired([r['interaction_v_pp']for r in rr]),
            changed_time_counts={a:sum(r['time_changed'][a]for r in rr)for a in ARMS})
    cost=dict(GPU_allocation_seconds=sum(read(f)['GPU_seconds']for f in (D/'allocations').glob('*.json')),
        new_DINO=sum(r['teacher_calls']for r in rows),new_dense_TA_queries=32,new_offset_full_forwards=64,
        new_networks_training_heads_student_updates=0,OPD_not_run=True,
        old_E_Source_DINO_cost_excluded=True,not_first_deployment_latency=True,
        spatial_seconds=sum(r['compute']['spatial_including_original_decode_seconds']for r in rows),
        dense_seconds=sum(r['compute']['dense_including_decode_seconds']for r in rows))
    put(P/'RESULTS_PUBLIC.json',dict(status='completed',count=32,groups=groups,rows=rows,GT_after_full_seal=True,
        exposed_development=True,OPD_not_run=True,joint_is_factorial_not_joint_conditioning=True))
    put(P/'FRAME_DIAGNOSTICS_PUBLIC.json',dict(rows=diag,unknown_preserved=True,no_GT_selection=True))
    put(P/'COST_BREAKDOWN.json',cost)
    print(json.dumps({ds:{a:groups[ds]['paired'][a+'_minus_E_MBR']['v']for a in ARMS[1:]}for ds in g}),flush=True)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','measure','score']);a=ap.parse_args()
    try:globals()[a.stage]()
    except BaseException as e:
        if D.exists():put(D/f'FAILURE_{a.stage}_{time.time_ns()}.json',dict(type=type(e).__name__,error=str(e),partials_preserved=True))
        raise
