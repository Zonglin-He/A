"""One finite HC-only content-state experiment; all output seal before GT."""
import os, sys, time, gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=ROOT/'artifacts/stvg_hc_anchor_content16_v1'
P=ROOT/'results/stvg_error_state/2026-10-19/hc_anchor_content16'
PRE=ROOT/'artifacts/stvg_dino_endpoint_confirm32_v1'
OBS=ROOT/'artifacts/stvg_observation_factorial32_v1'
CACHE=ROOT/'artifacts/stvg_dino_query_closedloop32_v1/replay_inputs'
ARMS=['E_MBR','Direct','Fixed','Live'];VER=None
CODE=['vg_tta/stvg_hc_anchor_content16_v1.py','scripts/run_stvg_hc_anchor_content16_v1.py','protocols/stvg_hc_anchor_content16_v1.md']


def verify(full=False):
    global VER
    from main_method.runtime import PinVerifier,resource_guard
    p=read(D/'PLAN_LOCK.json');lock=read(D/'RUNTIME_LOCK.json')
    assert sha(D/'PLAN_LOCK.json')==lock['plan_sha256']
    pins=dict(lock['pins'])
    for amendment in sorted((D/'amendments').glob('*.json')):
        a=read(amendment);assert a['original_runtime_sha256']==sha(D/'RUNTIME_LOCK.json');pins.update(a['replacement_pins'])
    if VER is None:VER=PinVerifier(pins)
    VER.verify(full=full);resource_guard();return p


def prepare():
    import torch,numpy as np
    from vg_tta.stvg_hc_anchor_content16_v1 import properties
    assert not D.exists() and not P.exists();D.mkdir();P.mkdir()
    rs=[r for r in read(OBS/'PLAN_LOCK.json')['records']if r['dataset']=='hc2'];assert len(rs)==16
    cfg={k:read(ROOT/'methods/decota_spatial_opd_v1/configs.json')['datasets']['hc2']['config'][k]for k in ['lr','sigma','tau','steps','samples']}
    oldpins=read(ROOT/'artifacts/stvg_e_retaining1792_route32_v1/RUNTIME_LOCK.json')['pins']
    files=list(CODE)+[f for f in oldpins if f.startswith(('external/TA-STVG/','methods/decota_final_simplified_v1/','checkpoints/'))or f in ['scripts/run_decota_paper_main_v1.py','scripts/run_spatial_regression_alignment_v1.py','scripts/run_spatial_ssl_gpu_v1.py','scripts/run_final_simplification_v1.py']]
    files+=['vg_tta/stvg_dino_query_closedloop32_v1.py','vg_tta/decota_spatial_opd_tunable_v1.py','vg_tta/stvg_opd_frozen_capture_cpu007.py','main_method/runtime.py','methods/decota_spatial_opd_v1/configs.json','scripts/run_stvg_joint_event_critic_v1.py','scripts/run_stvg_dino_local_residual32_v1.py','scripts/score_stvg_opd_p1_v1.py','scripts/score_stvg_joint_robust_correction32_v1.py']
    files+=['methods/'+n for n in ['CURRENT_METHOD.json','MAIN_METHOD.json','PAPER_METHOD.json']]
    support=[]
    for r in rs:
        a=r['alias'];s=load(PRE/'source'/(a+'.pt'));e=load(PRE/'E'/(a+'.pt'));o=load(OBS/'predictions'/(a+'.pt'));c=load(CACHE/(a+'.pt'))
        assert torch.equal(s['Source'],c['Source'])and np.array_equal(s['frame_ids'],e['frame_ids'])
        assert np.array_equal(o['outputs']['E_MBR'],e['E'])
        for v in range(2):
            assert s['records'][v]['frame_ids']==s['frame_ids'][v::2]
            assert len(s['records'][v]['frame_ids'])==len(c['spatial_inputs'][v]['query_tgt'])
        original={f['position']:s['expert']['observations'][('original',f['position'])]['probe'] for f in s['expert']['anchors']['single4']}
        obs={f['position']:f for f in o['spatial']['observations']};anchors=[];rows=[]
        for pos in e['planned_positions']:
            cp=obs[pos].get('probe',{});origin='crop'if cp.get('accepted',False)else'original_fallback'if pos in original else'none'
            probe=cp if origin=='crop'else original.get(pos,{})
            if origin!='none':
                scores=np.asarray(probe['target_scores'],float);k=int(scores.argmax());box=np.asarray(probe['boxes'][k],float)
                anchors.append(dict(position=pos,box=box.tolist(),weight=1.,origin=origin,score=float(scores[k]),candidate_index=k))
            rows.append(dict(position=pos,view=pos%2,local=pos//2,origin=origin))
        save(D/'inputs'/(a+'.pt'),dict(alias=a,anchors=anchors,GT_read=False,planned_mapping=rows))
        support.append(dict(alias=a,references=len(anchors),parameters=256*len(anchors),mapping=rows))
        files +=[str(f.relative_to(ROOT))for f in [PRE/'source'/(a+'.pt'),PRE/'E'/(a+'.pt'),OBS/'predictions'/(a+'.pt'),CACHE/(a+'.pt'),D/'inputs'/(a+'.pt')]]
    put(D/'PLAN_LOCK.json',dict(status='locked_before_model_updates_and_GT',records=rs,count=16,arms=ARMS,configuration=cfg,
        source='original opposite-domain Vid-source EMA TA-STVG',teacher='admitted crop top1, original admitted top1 fallback',
        content_state='K independent256 vectors before first spatial temporal-SA, K<=4',
        cost='5L1+2(1-GIoU), mean supported anchors, target-independent',
        OPD_policy='observed native-logit Gaussian; not sampled decoder latent trajectories',
        pilot='NH001',GPU_seconds_cap=900,wall_deadline_epoch=time.time()+3600,created_epoch=time.time(),
        no_GT_support_filter=True,no_new_Source_or_DINO_or_model_training=True,no_parameter_scan=True,no_automatic_followup=True))
    put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins={f:sha(ROOT/f)for f in sorted(set(files))}))
    put(P/'CONFIG_PUBLIC.json',dict(count=16,per_parent_queries=1,exposed_development=True,independent_confirmation=False,
        configuration=cfg,arms=ARMS,all_pretrained_weights_frozen=True,temporary_state_reset=True,reference_support=support,
        teacher_GT_free=True,new_observations=0,new_training_heads=0,sampling='native-logit output Gaussian; current actual decoder mean',
        feedback='negative same 5L1+2(1-GIoU) cost as Direct',normalization='1/K in all losses',
        first_Fixed_Live_identical=True,temporal_fixed=True,GT_after_full16_seal=True))
    put(P/'CPU_PROPERTIES.json',properties());verify(True);print('LOCKED HC16 references',sum(x['references']for x in support),flush=True)


def measure():
    import torch,numpy as np
    from scripts.run_decota_paper_main_v1 import gpu,model_for
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.stvg_hc_anchor_content16_v1 import Replay,fit,inject
    p=verify(True);assert not (D/'GLOBAL_PREDICTION_BARRIER.json').exists()
    lease=gpu();started=time.perf_counter();files=[];rp=None;done=False
    def check():
        verify();assert time.time()<p['wall_deadline_epoch']
        assert time.perf_counter()-started+3<p['GPU_seconds_cap']
        assert torch.cuda.mem_get_info()[0]>4*2**30
    try:
        model=model_for('vidstg');mh=state_hash(model.state_dict())
        for ix,r in enumerate(p['records']):
            check();tick=time.perf_counter();a=r['alias']
            put(D/'STATUS.json',dict(status='running',done=ix,total=16,alias=a,GT_read=False),True)
            inp=load(D/'inputs'/(a+'.pt'));s=load(PRE/'source'/(a+'.pt'));e=load(PRE/'E'/(a+'.pt'));c=load(CACHE/(a+'.pt'))
            pos=[f['position']for f in inp['anchors']];rp=Replay(model,c,s['Source'],e['E'],pos,check)
            qualification=dict(zero_Source_exact=True,zero_E_exact=True,offsets=[dict(view=v,states=[i for i,x in enumerate(pos)if x%2==v],locals=[x//2 for x in pos if x%2==v])for v in range(2)])
            if a==p['pilot'] and pos:
                with torch.no_grad():
                    rp.state[0]=torch.linspace(-.01,.01,256,device='cuda')
                    probe=rp.values();view=pos[0]%2
                    assert torch.equal(probe['native'][1-view::2],rp.source[1-view::2])
                    qualification['single_anchor_unaffected_other_offset_exact']=True
                    qualification['single_anchor_affected_offset_changed_nodes']=int(torch.any(probe['native'][view::2]!=rp.source[view::2],dim=-1).sum())
                    assert qualification['single_anchor_affected_offset_changed_nodes']>0
                    rp.state.zero_()
            fits={arm:fit(rp,inp['anchors'],a,arm,p['configuration'])for arm in ARMS[1:]}
            f,l=fits['Fixed'],fits['Live']
            assert torch.equal(f['path'][0]['state'],l['path'][0]['state'])
            assert torch.equal(f['path'][min(1,len(f['path'])-1)]['state'],l['path'][min(1,len(l['path'])-1)]['state'])
            if f['rounds']:
                for k in ['actions','weights','cost']:assert torch.equal(f['rounds'][0]['rollout'][k],l['rounds'][0]['rollout'][k])
            assert all(not q.requires_grad and q.grad is None for q in model.parameters())and state_hash(model.state_dict())==mh
            assert state_hash(rp.decoder.state_dict())==state_hash(model.ground_decoder.decoder.state_dict())
            outputs=dict(E_MBR=e['E'],**{arm:f['final'].numpy()for arm,f in fits.items()})
            cost=dict(case_seconds=time.perf_counter()-tick,trajectory_decoder_calls=rp.calls,PosDecoder_calls=2*rp.calls,
                decoder_seconds=rp.seconds,backward_calls=rp.backwards,new_Source_DINO_training=0)
            files.append(save(D/'predictions'/(a+'.pt'),dict(alias=a,dataset='hc2',frame_ids=e['frame_ids'],planned_positions=e['planned_positions'],interval=e['interval'],
                outputs=outputs,E=e['E'],positions=pos,targets=inp['anchors'],fits=fits,qualification=qualification,model_hash=mh,
                GT_read=False,time_fixed=True,compute=cost)))
            print(a,ix+1,'refs',len(pos),'seconds',round(cost['case_seconds'],3),'GPU',round(time.perf_counter()-started,2),flush=True)
            if ix==0:
                estimate=time.perf_counter()-started+15*cost['case_seconds']+30
                put(D/'PILOT_TIMING.json',dict(alias=a,seconds=cost['case_seconds'],estimated_total_GPU_seconds=estimate,not_guaranteed=True));assert estimate<900,'pilot budget infeasible'
            del rp,s,e,c,fits,f,l,outputs;rp=None;gc.collect();torch.cuda.empty_cache()
        verify(True);check();done=True
        put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=16,logical_outputs=64,files=files,
            all16_outputs_before_new_GT=True,GPU_seconds=time.perf_counter()-started+3,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        put(D/'STATUS.json',dict(status='all16_sealed',done=16,total=16,GT_read=False),True)
    finally:
        put(D/'GPU_ALLOCATION.json',dict(complete=done,count=len(files),GPU_seconds=time.perf_counter()-started+3));lease.close()


def score():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch,numpy as np
    from scripts.score_stvg_opd_p1_v1 import truths
    from scripts.run_stvg_dino_local_residual32_v1 import metrics,mean
    from scripts.score_stvg_joint_robust_correction32_v1 import paired
    torch.set_num_threads(4);p=verify(True);bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==16 and bar['all16_outputs_before_new_GT']
    for rec in bar['files']:assert sha(ROOT/rec['path'])==rec['sha256']
    gt,extra,_gt_provenance=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]))
    plans=read(ROOT/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows']
    old={r['alias']:r for r in read(ROOT/'results/stvg_error_state/2026-10-19/observation_factorial32/RESULTS_PUBLIC.json')['rows']}
    rows=[]
    for r in p['records']:
        a=r['alias'];z=load(D/'predictions'/(a+'.pt'));m=metrics(z,plans[r['query_ordinal']],gt[r['query_ordinal']],extra[r['query_ordinal']]);di={}
        for key in ['v','t','s','observed_actual_s','unobserved_actual_s']:
            x,y=m['values']['E_MBR'][key],old[a]['values']['E_MBR'][key];assert x is None and y is None or x is not None and y is not None and abs(x-y)<1e-12
        for arm,f in z['fits'].items():
            rr=f['rounds'];di[arm]=dict(parameters=f['parameters'],updates=f['steps'],anchor_loss_initial=f['path'][0]['anchor_loss'],anchor_loss_final=f['path'][-1]['anchor_loss'],
                state_L2=float(f['path'][-1]['state'].norm()),changed_actual_nodes=int(np.any(f['final'].numpy()!=z['E'],axis=1).sum()),
                seconds=f['seconds'],flat_feedback_rounds=sum(x['rollout']is not None and not x['rollout']['informative']for x in rr),
                mean_teacher_KL_from_uniform=None if arm=='Direct'or not rr else float(np.mean([(x['rollout']['weights']*(x['rollout']['weights'].clamp_min(1e-30).log()+np.log(32))).sum(-1).mean()for x in rr])))
            assert m['values'][arm]['t']==m['values']['E_MBR']['t']
        rows.append(dict(alias=a,dataset='hc2',references=len(z['positions']),fit_diagnostics=di,qualification=z['qualification'],compute=z['compute'],**m))
    ep=['v','t','s','observed_actual_s','unobserved_actual_s'];pairs=[(a,'E_MBR')for a in ARMS[1:]]+[('Live','Fixed'),('Live','Direct')]
    group=dict(parents=16,support_queries=sum(r['references']>0 for r in rows),references=sum(r['references']for r in rows),
        means={a:{k:mean([r['values'][a][k]for r in rows])for k in ep}for a in ARMS},
        paired={a+'_minus_'+b:{k:paired([None if r['values'][a][k]is None or r['values'][b][k]is None else 100*(r['values'][a][k]-r['values'][b][k])for r in rows])for k in ep}for a,b in pairs})
    put(P/'RESULTS_PUBLIC.json',dict(status='completed',count=16,groups=dict(hc2=group),rows=rows,development_exposed=True,independent_confirmation=False,GT_after_full16_seal=True))
    put(P/'COST_BREAKDOWN.json',dict(GPU_seconds=read(D/'GPU_ALLOCATION.json')['GPU_seconds'],trajectory_decoder_calls=sum(r['compute']['trajectory_decoder_calls']for r in rows),
        PosDecoder_calls=sum(r['compute']['PosDecoder_calls']for r in rows),backward_calls=sum(r['compute']['backward_calls']for r in rows),
        new_Source_or_DINO_calls=0,new_training_or_heads=0,temporary_state_updates=sum(f['updates']for r in rows for f in r['fit_diagnostics'].values()),historical_cache_cost_excluded=True))
    print('SCORED HC16', {c:group['paired'][c]['v']['mean']for c in group['paired']},flush=True)


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','measure','score']);a=ap.parse_args()
    try:globals()[a.stage]()
    except BaseException as exc:
        D.mkdir(exist_ok=True,parents=True);put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(stage=a.stage,type=type(exc).__name__,error=str(exc),partials_preserved=True));raise
