"""Cached partial local-motion observation, existing frozen Source suffix only."""
from pathlib import Path
import sys,os,time,gc,fcntl,signal
os.environ['OPENBLAS_NUM_THREADS']='1'
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=R/'artifacts/stvg_temporal_state_observation128_v1';P=R/'results/stvg_error_state/2026-10-20/temporal_state_observation128'
OLD=R/'artifacts/stvg_temporal_local_motion_completion128_v1'
EARLY=R/'artifacts/stvg_temporal_early_fusion128_v1';STAGE=R/'artifacts/stvg_temporal_native_stage128_v1';HR=R/'artifacts/stvg_temporal_hires128_v1'
POLICIES=['Uniform8','InitialState8','CurrentState8','ReflectedState8']
ARMS=['Native','HiResAuxMix','DenseLocalMix']+POLICIES

def prepare():
    from vg_tta.stvg_temporal_state_observation128_v1 import properties
    old=read(OLD/'PLAN_LOCK.json');bar=read(OLD/'GLOBAL_PREDICTION_BARRIER.json')
    assert bar['count']==128 and bar['weights_and_flags_unchanged']and all(sha(R/f)==h for f,h in old['pins'].items())
    own=['vg_tta/stvg_temporal_state_observation128_v1.py','scripts/run_stvg_temporal_state_observation128_v1.py',
         'scripts/score_stvg_temporal_state_observation128_v1.py','scripts/publish_stvg_temporal_state_observation128_v1.py',
         'protocols/stvg_temporal_state_observation128_v1.md','vg_tta/stvg_temporal_native_stage128_v1.py',
         str((OLD/'PLAN_LOCK.json').relative_to(R)),str((OLD/'GLOBAL_PREDICTION_BARRIER.json').relative_to(R)),
         'results/stvg_error_state/2026-10-20/temporal_local_motion_completion128/RESULTS_PUBLIC.json']
    pins=dict(old['pins']);pins.update({f:sha(R/f)for f in own})
    for r in old['records']:
        for path,h in [(EARLY/'early_inputs'/f"{r['alias']}.pt",r['early_sha256']),
                       (STAGE/'predictions'/f"{r['alias']}.pt",r['native_stage_sha256']),
                       (HR/'predictions'/f"{r['alias']}.pt",r['hires_sha256'])]:
            assert sha(path)==h;pins[str(path.relative_to(R))]=h
    for receipt in bar['files']:
        assert sha(R/receipt['path'])==receipt['sha256'];pins[receipt['path']]=receipt['sha256']
    p=dict(records=old['records'],count=128,seeds=[41,42,43],policies=POLICIES,arms=ARMS,primary='CurrentState8',
        rounds=2,reveals_per_round=4,real_local_clip_budget=8,real_RGB_slots=64,hypothetical_B8_padded_slots=128,
        source_checkpoint=old['source_checkpoint'],source_checkpoint_sha256=old['source_checkpoint_sha256'],
        GPU_seconds_cap=900,CPU_seconds_cap=900,created_epoch=time.time(),deadline_epoch=time.time()+14400,
        pins=pins,spatial_frozen=True,no_new_encoding=True,no_training=True,no_OPD=True,
        expected_call_upper_bounds=dict(fusion=2308,decoder=4616,head=2308),
        preserved_Strong='0.5 Source448 raw +0.5 mean6 second-pass layer posteriors; change selected224 motion rows only')
    put(D/'PLAN_LOCK.json',p);put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','created_epoch','deadline_epoch']})
    put(P/'PROPERTIES_PUBLIC.json',properties());print('locked cached local observation128:4 policies,2x4 reveals, no encoding',flush=True)

def measure():
    import torch,numpy as np
    from main_method.runtime import resource_guard
    from methods.stvg_referdino_opd_v1.runtime import setup,lease,load_source
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.stvg_temporal_state_observation128_v1 import choose,reflected,reveal,uniform_order,array_sha
    from vg_tta.stvg_temporal_registered_motion128_v1 import fuse_motion,decode_prefix,tensor_sha
    from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table
    from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
    setup()
    def reject(event,args):
        if event=='open'and args and isinstance(args[0],(str,bytes,Path))and str(args[0]).endswith('/data/hcstvg2_official_metadata/val_v2.json'):
            raise PermissionError('cached acquisition inference forbids GT')
    sys.addaudithook(reject);p=read(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items())
    lock=lease();beg=time.perf_counter();hooks=[];partial=None;files=[];times=[];control_seconds=0.;success=False;cache_hits=0;logical_requests=0;actual_replays=0;empty_checks=0;qual=[]
    names=['fusion','decoder','head','image','motion','text','image_projection','motion_projection']
    counts={k+'_'+s:0 for k in names for s in ['attempts','completed']}
    def guard():
        resource_guard();free,total=torch.cuda.mem_get_info();assert free>5*2**30 and time.time()<p['deadline_epoch']and time.perf_counter()-beg+35<900
    def enter(k):
        def f(m,a):
            counts[k+'_attempts']+=1
            assert counts[k+'_attempts']<=p['expected_call_upper_bounds'].get(k,0)
        return f
    def returned(k):
        def f(m,a,o):
            counts[k+'_completed']+=1
            if k=='head'and partial is not None and partial.get('inflight')is not None:
                partial['inflight'].setdefault('returned_head_logits',[]).append(o.detach().cpu())
        return f
    try:
        resource_guard();free,total=torch.cuda.mem_get_info();assert free>4*2**30
        fraction=min(.8,(free-5*2**30)/total);assert fraction>.4;torch.cuda.set_per_process_memory_fraction(fraction)
        m=load_source();mh=state_hash(m.state_dict());flags={k:v.training for k,v in m.named_modules()};assert not any(flags.values());loading=time.perf_counter()-beg
        for mod,k in [(m.ground_encoder.encoder,'fusion'),(m.ground_decoder,'decoder'),(m.temp_embed,'head'),
                      (m.vis_encoder,'image'),(m.vid,'motion'),(m.text_encoder,'text'),(m.input_proj,'image_projection'),(m.input_proj2,'motion_projection')]:
            hooks.extend([mod.register_forward_pre_hook(enter(k)),mod.register_forward_hook(returned(k))])
        for r in p['records']:
            guard();tick=time.perf_counter();ctrl=0.;a=r['alias']
            old=load(OLD/'predictions'/f'{a}.pt');early=load(EARLY/'early_inputs'/f'{a}.pt');stage=load(STAGE/'predictions'/f'{a}.pt');hr=load(HR/'predictions'/f'{a}.pt')
            ids=old['frame_ids'];q0=old['outputs']['HiResAuxMix']['q'];q448=hr['observation']['q'];nativeparts=[]
            for off in (0,1):
                cap=early['captured_encoder_inputs'][off];h,w=early['original_views'][off]['info']['fea_map_size'];l=h*w
                nativeparts.append(cap['src'][-l:].permute(1,2,0).reshape(len(ids[off::2]),256,h,w))
            native=torch.stack([nativeparts[i%2][i//2]for i in range(65)])
            assert native.shape==old['motion_features']['DenseLocal'].shape
            partial=dict(alias=a,initial_q=q0,replays={},policies={},inflight=None,GT_read=False);cache=partial['replays']
            def infer(selected):
                nonlocal cache_hits,logical_requests,actual_replays
                logical_requests+=1;key=','.join(map(str,sorted(selected)))
                if key in cache:cache_hits+=1;return key,cache[key]
                guard();idx=sorted(selected)
                # Provider exposes values of selected nodes only to reveal().
                local=old['motion_features']['DenseLocal'][idx].clone();motion=reveal(native,idx,local)
                inflight=dict(selected=idx,source='selected local features only',views=[],metadata={},returned_head_logits=[])
                partial['inflight']=inflight
                for off in (0,1):
                    view,meta=fuse_motion(m,early['captured_encoder_inputs'][off],early['original_views'][off],motion[off::2],inflight)
                    inflight['views'].append(view);inflight['metadata'][str(off)]=meta;guard()
                captures=[]
                hook=m.temp_embed.register_forward_hook(lambda mm,aa,oo:captures.append(oo.detach().cpu()))
                try:values=decode_prefix(m,inflight['views'],65)
                finally:hook.remove()
                assert len(captures)==2 and all(x.shape==(6,1,len(ids[off::2]),2)for off,x in enumerate(captures))
                layerqs=[]
                for k in range(6):
                    grid,q=mass_table(probabilities(old['records'],[z[k]for z in captures]));assert grid==ids;layerqs.append(q)
                q=.5*q448+.5*np.mean(layerqs,axis=0);assert abs(q.sum()-1)<1e-12
                value=dict(selected=idx,selected_features_sha256=tensor_sha(local),motion_sha256=tensor_sha(motion),
                    unobserved_motion_exact=True,prefixes=[v['prefix']for v in inflight['views']],metadata=inflight['metadata'],
                    second_pass_all6_logits=captures,layer_qs=layerqs,q=q,final_layer_q=layerqs[-1],
                    interval=final_mbr(ids,q)['interval'],actual_Source_boxes_discarded=True)
                cache[key]=value;actual_replays+=1;partial['inflight']=None;guard();return key,value
            zero_key,zero=infer([])
            assert all(torch.equal(x,y)for x,y in zip(zero['prefixes'],[v['prefix']for v in early['original_views']]))
            assert all(torch.equal(x,y)for x,y in zip(zero['second_pass_all6_logits'],stage['pass_logits'][1]))
            assert np.array_equal(zero['final_layer_q'],old['outputs']['Native']['q'])and np.array_equal(zero['q'],q0)
            assert zero['interval']==old['outputs']['HiResAuxMix']['interval'];empty_checks+=1
            outputs={k:old['outputs'][k]for k in ['Native','HiResAuxMix','DenseLocalMix']}
            for policy in POLICIES:
                selected=[];history=[];current=q0
                for roundno in range(2):
                    if policy=='Uniform8':
                        selected=uniform_order()[:4*(roundno+1)];decision=dict(rule='fixed_even_odd_uniform8',state_q_sha256=array_sha(current))
                    else:
                        state=current if policy=='CurrentState8'else reflected(q0)if policy=='ReflectedState8'else q0
                        selected,decision=choose(ids,state,selected,4)
                    key,out=infer(selected)
                    history.append(dict(round=roundno+1,selected=list(selected),selection=decision,replay_key=key,
                        actual_student_q_before=current.copy(),q_after=out['q'],state_L1_from_initial=float(abs(out['q']-q0).sum())))
                    current=out['q']
                assert len(set(selected))==8
                clips=old['clip_indices']['DenseLocal'][selected]
                partial['policies'][policy]=dict(history=history,final_selected=selected,real_clip_budget=8,real_RGB_slots=64,
                    hypothetical_B8_padded_slots=128,physical_unique_frames=len(set(clips.flatten().tolist())))
                outputs[policy]=dict(q=current,interval=final_mbr(ids,current)['interval'])
            f=partial['policies']['InitialState8']['history'][0];g=partial['policies']['CurrentState8']['history'][0]
            assert f['selected']==g['selected']and np.array_equal(f['q_after'],g['q_after'])
            if len(files)<2:
                ct=time.perf_counter();key,full=infer(list(range(65)))
                assert np.array_equal(full['final_layer_q'],old['outputs']['DenseLocal']['q'])
                assert all(torch.equal(z[-1],v)for z,v in zip(full['second_pass_all6_logits'],old['logits']['DenseLocal']))
                qual.append(dict(alias=a,all65_reveal_final_layer_exact=True,native_all6_and_Strong_exact=True))
                ctrl=time.perf_counter()-ct;control_seconds+=ctrl
            files.append(save(D/'predictions'/f'{a}.pt',dict(alias=a,frame_ids=ids,records=old['records'],outputs=outputs,
                initial_q=q0,hires_q=q448,replays=cache,policies=partial['policies'],qualification_only_full65=len(files)<2,
                trajectories=old['trajectories'],GT_read=False,student_parameter_updates=0,spatial_updated=False)))
            times.append(time.perf_counter()-tick-ctrl)
            put(D/'STATUS.json',dict(stage='cached_suffix',done=len(files),actual_calls=counts,actual_unique_replays=actual_replays,
                logical_requests=logical_requests,deterministic_cache_hits=cache_hits,pid=os.getpid(),epoch=time.time()),True)
            print('cached_observation',a,round(time.perf_counter()-beg,2),flush=True)
            if len(files)==2:
                estimate=loading+control_seconds+128*max(times)+60
                put(D/'PILOT2.json',dict(case_seconds=times,loading=loading,qualification_seconds=control_seconds,
                    estimated_GPU_seconds=estimate,exact_controls=qual,new_GT=False));assert estimate<900
            partial=None;del old,early,stage,hr,native,nativeparts,cache,outputs;gc.collect();torch.cuda.empty_cache()
        guard();assert all(sha(R/f)==h for f,h in p['pins'].items())and state_hash(m.state_dict())==mh
        assert {k:v.training for k,v in m.named_modules()}==flags and all(not v.requires_grad and v.grad is None for v in m.parameters())
        assert all(counts[k+'_attempts']==counts[k+'_completed']for k in names)
        assert counts['fusion_completed']==2*actual_replays and counts['decoder_completed']==4*actual_replays and counts['head_completed']==2*actual_replays
        put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=128,files=files,plan_sha256=sha(D/'PLAN_LOCK.json'),all_before_GT=True,
            weights_flags_grad_unchanged=True,empty_Strong_exact=empty_checks,qualifications=qual));success=True
    finally:
        for h in hooks:h.remove()
        if not success and partial is not None:save(D/'partials'/f"{partial['alias']}_{time.time_ns()}.pt",dict(partial,actual_calls=counts))
        elapsed=time.perf_counter()-beg
        put(D/'COST.json',dict(GPU_seconds=elapsed+3,measured_lease_seconds=elapsed,cleanup_reserve_seconds=3,success=success,
            actual_calls=counts,logical_requests=logical_requests,actual_unique_replays=actual_replays,deterministic_cache_hits=cache_hits,
            new_RGB=0,new_visual_motion_text_encoding=0,new_DINO=0,new_ReferDINO=0,new_training=0,new_student_updates=0,
            spatial_updates=0,not_deployment_latency=True,GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated()if torch.cuda.is_initialized()else 0,
            GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved()if torch.cuda.is_initialized()else 0));lock.close()

if __name__=='__main__':
    def terminated(signum,frame):raise TimeoutError('owned cached-replay900s budget')
    signal.signal(signal.SIGTERM,terminated);D.mkdir(parents=True,exist_ok=True)
    lock=(D/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:globals()[sys.argv[1]]()
    except BaseException as e:put(D/f'FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e)));raise
    finally:lock.close()
