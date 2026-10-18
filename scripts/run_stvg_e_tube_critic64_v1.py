"""Budgeted E-neighborhood feedback qualification, no target label reader."""
from pathlib import Path
import sys,os,time,collections,gc,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=ROOT/'artifacts/stvg_e_tube_critic64_v1'
P=ROOT/'results/stvg_error_state/2026-10-18/e_tube_critic64'
CRITIC=ROOT/'artifacts/stvg_joint_event_critic_v1'
VERIFIER=None


def status(stage,done,**kw):put(D/'STATUS.json',dict(status='running',stage=stage,done=done,total=64,pid=os.getpid(),timestamp=time.time(),**kw),True)


def prepare():
    import torch,numpy as np
    from vg_tta.stvg_joint_event_critic_v1 import EventCritic,collate,candidate_tensor
    from vg_tta.stvg_e_tube_critic64_v1 import contract,intervals
    from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
    torch.set_num_threads(4)
    old=read(ROOT/'artifacts/stvg_query_e_opd64_v1/PLAN_LOCK.json');assert len(old['records'])==64
    assert read(CRITIC/'CRITIC_COMPLETION.json')['count']==2
    inputs={};cache=[]
    for r in old['records']:
        for f in ['prior_Source','prior_prediction']:
            path=ROOT/r[f];load(path);inputs[r[f]]=sha(path)
        s=load(ROOT/r['prior_Source']);e=load(ROOT/r['prior_prediction']);iv,ids,q=intervals(s,e['interval'])
        assert final_mbr(ids,q)['interval']==e['interval'];assert 'views'not in s
        cache.append(dict(alias=r['alias'],dataset=r['dataset'],candidate_count=3*len(iv),source_boxes_logits_E_available=True,critic_ROI_maps_in_Source_cache=False,one_Source_recapture_required=True))
    # Freeze optional stage2 scaling BEFORE opening any new target GT.
    calibration=read(ROOT/'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json');scale={};criticpins={}
    source=read(CRITIC/'PLAN_LOCK.json')['source']
    for sd in ['hc2','vidstg']:
        path=CRITIC/'critics'/(sd+'.pt');z=load(path);criticpins[str(path.relative_to(ROOT))]=sha(path)
        model=EventCritic().eval().requires_grad_(False);model.load_state_dict(z['state']);scores=[];files=[]
        for r in source:
            if r['dataset']!=sd or r['split']!='source_validation':continue
            path=CRITIC/'source'/(r['alias']+'.pt');x=load(path);inputs[str(path.relative_to(ROOT))]=sha(path)
            with torch.no_grad():out=model(*collate([candidate_tensor(f,I,x['input']['fps'])for f in x['features']for I in x['candidates']['intervals']]))
            scores.extend(out.tolist());files.append(r['alias'])
        std=float(np.std(scores));assert len(files)==12 and std>0
        scale[sd]=dict(geometry_scale=calibration['source_domains'][sd]['scale'],critic_reward_std=std,source_validation_queries=12,GT_opened_for_calibration=False)
    own=['vg_tta/stvg_e_tube_critic64_v1.py','scripts/run_stvg_e_tube_critic64_v1.py','scripts/score_stvg_e_tube_critic64_v1.py','scripts/continue_stvg_e_tube_critic64_v1.py','protocols/stvg_e_tube_critic64_v1.md']
    oldrt=read(ROOT/'artifacts/stvg_query_e_opd64_v1/RUNTIME_LOCK.json')
    pins={**oldrt['pins'],**inputs,**criticpins,**{f:sha(ROOT/f)for f in own},str((ROOT/'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json').relative_to(ROOT)):sha(ROOT/'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json')}
    put(D/'PLAN_LOCK.json',dict(status='locked_before_new_target_outputs_GT',records=old['records'],count=64,per_domain=32,max_tubes=768,
        spatial=['Source','half_E_delta','E'],temporal='old final MBR plus top3 OTHER distinct positive-mass final physical intervals; no frame-fingerprint restriction',
        raw_critic_frozen=True,old_critic_transfer_failure_preserved=True,source_normalized_feature_maps_cache_missing=64,max_Source_captures=64,
        GPU_seconds_cap=300,wall_seconds_cap=900,created_epoch=time.time(),deadline_epoch=time.time()+900,
        gate='Both domain query-mean temporal and spatial strict-GT-pair accuracies have bootstrap95% lower >.5, both domain Joint-E means >=0 and equal-domain paired macro95% lower >0. Missing/uncertain => no automatic stage2.',
        pair_GT_tie_tolerance=1e-8,critic_ties='exact tie half credit; deterministic first row-major selection',
        source_stage2_scaling=scale,stage2=dict(conditional_only=True,rounds=5,temporal_samples=8,spatial_samples=4,
            time_variables='2 endpoint exponential-tilt coefficients on normalized physical endpoints; full legal q support retained; init0',
            spatial_variables='4 normalized correction means, init E_delta/source_scale, unit covariance Gaussian',
            learning_rate=.05,optimizer='plain SGD maximize detached conditional LOO critic advantages; reward/source validation score std',
            feedback='raw frozen event critic, no geometric reward',final='round5 time full-distribution MBR, spatial corrected mean',
            arms=['E_MBR','T_only','S_only','Joint_OPD','Fixed_initial_rollout','One_shot_search'],
            fixed='same first8x4 actions every round and same160 scoring calls; gradients at current distribution',
            search='initial distribution40 time x4 spatial tubes,160 total; highest critic tube',seed=20261018,network_updates=0,new_safety_gates=0,
            max_queries=64,max_rounds=5,source_only_scaling_locked_before_target_GT=True,two_first_queries_actual_timing_required=True),
        new_DINO_calls=0,new_Qwen_calls=0,new_source_training=0,new_QR_LN_updates=0,old_before_replay=0,independent_confirmation=False))
    put(D/'RUNTIME_LOCK.json',dict(pins=pins,registries=oldrt['registries'],plan_sha256=sha(D/'PLAN_LOCK.json')))
    put(D/'CACHE_AUDIT.json',dict(status='complete',count=64,rows=cache,max_required_captures=64,full_ROI_maps_not_in_Source_cache=True,old32_post_ROI_vectors_not_a_substitute_for_new_trajectories=True))
    put(D/'CPU_CONTRACT_COMPLETION.json',contract());status('prepared',0,GT_read=False)


def verify(allow_closed=False):
    global VERIFIER
    from main_method.runtime import PinVerifier,resource_guard
    p=read(D/'PLAN_LOCK.json');l=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==l['plan_sha256']
    if not allow_closed:assert time.time()<p['deadline_epoch'],'15-minute first-stage wall cap'
    if VERIFIER is None:VERIFIER=PinVerifier({**l['pins'],**l['registries']})
    VERIFIER.verify();resource_guard();return p


def measure():
    import torch,numpy as np
    from methods.decota_final_simplified_v1.tensors import detached,state_hash
    from scripts.run_decota_paper_main_v1 import gpu,model_for
    from scripts.run_stvg_dino_supported_expanded256_recovery002 import capture
    from vg_tta.stvg_joint_event_critic_v1 import EventCritic,extract_visual
    from vg_tta.stvg_e_tube_critic64_v1 import intervals,states,normalized_maps,from_maps,score_grid
    torch.set_num_threads(4);p=verify();lease=None
    while lease is None:
        verify()
        try:lease=gpu()
        except BlockingIOError:status('waiting_shared_GPU_lease',0,GT_read=False);time.sleep(5)
    start=time.perf_counter();model=None;domain=None;cache=collections.OrderedDict();files=[];qualified=[]
    try:
        for i,r in enumerate(p['records']):
            verify();assert time.perf_counter()-start<300,'5-minute GPU stage cap';ds=r['dataset'];status('Source_feature_and_frozen_critic',i,current=r['alias'],GT_read=False)
            if ds!=domain:
                if model is not None:del model;gc.collect();torch.cuda.empty_cache()
                domain=ds;model=model_for('vidstg'if ds=='hc2'else'hcstvg2');mh=state_hash(model.state_dict());cache.clear()
                sd='vidstg'if ds=='hc2'else'hc2';critic=EventCritic().eval().requires_grad_(False);critic.load_state_dict(load(CRITIC/'critics'/(sd+'.pt'))['state']);ch=state_hash(critic.state_dict())
            assert torch.cuda.mem_get_info()[0]>4*2**30;tick=time.perf_counter()
            row,records,base,native,expert,inp,seconds=capture(model,ds,r['query_ordinal'],cache)
            source=load(ROOT/r['prior_Source']);prior=load(ROOT/r['prior_prediction']);assert torch.equal(native['boxes'],source['Source'])
            assert all(torch.equal(a.cpu(),b)for a,b in zip(base.zero['logits'],source['logits']))
            iv,ids,q=intervals(source,prior['interval']);boxes=states(prior['outputs']['Source_no_correction'],prior['outputs']['E_DINO_support'],prior['estimate']['delta'])
            maps=normalized_maps(base);proxy=from_maps(maps);features=[extract_visual(proxy,torch.tensor(b,dtype=torch.float32),ids,row['input']['fps'])for b in boxes]
            scores=score_grid(critic,features,iv,row['input']['fps'])
            if ds not in qualified:
                ref=extract_visual(base,source['Source'],ids,row['input']['fps'])
                assert all(torch.equal(ref[k],features[0][k])for k in ['visual','query','boxes','times']);qualified.append(ds)
            assert state_hash(model.state_dict())==mh and state_hash(critic.state_dict())==ch
            assert time.perf_counter()-start<300,'5-minute GPU stage cap'
            out=dict(alias=r['alias'],dataset=ds,frame_ids=ids,intervals=iv,boxes=boxes,E_delta=prior['estimate']['delta'],q=q,
                scores=scores,selection=list(map(int,np.unravel_index(scores.numpy().argmax(),tuple(scores.shape)))),
                maps=detached(maps,'cpu'),features=features,fps=row['input']['fps'],active=prior['estimate']['support_frames']>0,
                source_logits_exact=True,source_boxes_exact=True,critic_frozen_exact=True,Source_frozen_exact=True,
                GT_read=False,new_Source_captures=1,new_DINO_calls=0,critic_evaluations=int(scores.numel()),
                compute=dict(Source_capture_seconds=seconds,case_seconds=time.perf_counter()-tick))
            files.append(save(D/'predictions'/(r['alias']+'.pt'),out));print(r['alias'],i+1,round(time.perf_counter()-start,2),flush=True)
            if i==1:put(D/'PILOT2_TIMING.json',dict(count=2,elapsed_seconds=time.perf_counter()-start,within_original64=True))
            del out,base,maps,proxy,features,expert
            gc.collect();torch.cuda.empty_cache()
        VERIFIER.verify(full=True);assert time.perf_counter()-start<300
        put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=64,files=files,total_tubes=sum(3*len(load(D/'predictions'/(r['alias']+'.pt'))['intervals'])for r in p['records']),
            GT_read=False,GPU_seconds=time.perf_counter()-start,new_Source_captures=64,new_DINO_calls=0,critic_qualification_domains=qualified))
        status('all_tube_feedback_sealed',64,GT_read=False)
    finally:lease.close()


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','measure']);x=a.parse_args().stage
    try:globals()[x]()
    except BaseException as e:put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',type=type(e).__name__,error=str(e),partials_preserved=True));raise
