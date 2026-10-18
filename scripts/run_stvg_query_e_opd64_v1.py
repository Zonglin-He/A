"""Bounded Source-reset dense E experiment; measure has no label reader."""
from pathlib import Path
import sys,os,time,json,hashlib,collections,gc,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,save,load,sha
D=ROOT/'artifacts/stvg_query_e_opd64_v1'
PRE=ROOT/'artifacts/stvg_dino_supported_source256_v1'
P=ROOT/'results/stvg_error_state/2026-10-18/query_e64'
ARMS=['Source_MBR','E_MBR','Sparse_OPD','Dense_OPD','Fixed_rollout','Dense_Direct']
VERIFIER=None


def status(stage,done,**kw):put(D/'STATUS.json',dict(status='running',stage=stage,done=done,total=64,pid=os.getpid(),timestamp=time.time(),**kw),True)


def prepare():
    import numpy as np
    from vg_tta.stvg_query_e_opd64_v1 import contract
    assert not(D/'PLAN_LOCK.json').exists();assert read(PRE/'FINAL_COMPLETION.json')['status']=='complete'
    records=read(ROOT/'artifacts/stvg_dense_e_opd64_v1/PLAN_LOCK.json')['records']
    assert len(records)==64 and all(sum(r['dataset']==d for r in records)==32 for d in ['hc2','vidstg'])
    for r in records:
        assert sha(ROOT/r['prior_prediction'])==r['prediction_sha256']
        assert sha(ROOT/r['prior_Source'])==r['Source_sha256']
    cfg={ds:read(ROOT/'methods/decota_spatial_opd_v1/configs.json')['datasets'][ds]['config']for ds in ['hc2','vidstg']}
    own=['vg_tta/stvg_query_e_opd64_v1.py','scripts/run_stvg_query_e_opd64_v1.py','scripts/continue_stvg_query_e_opd64_v1.py','protocols/stvg_query_e_opd64_v1.md','vg_tta/stvg_dense_e_opd64_v1.py','vg_tta/stvg_opd_paper_ablations_v1.py']
    pins=read(PRE/'RUNTIME_LOCK.json')['pins'];pins={**pins,**{f:sha(ROOT/f)for f in own}}
    put(D/'PLAN_LOCK.json',dict(status='locked_before_new_models_and_GT',records=records,count=64,arms=ARMS,
        selection='Exact already locked predecessor HC32/Vid32, no reselection',
        seed=20261018,initial='same frozen Source every query/arm; QR256 zero and fresh Adam; all pretrained and LN frozen',configs=cfg,
        E_teacher='exact prior sealed E once, fixed entire adaptation; shared float32 loss reference no clamp',
        final_clock='exact prior frozen Source final-distribution MBR interval, all arms',
        loss_support='all actual decoder frames, no GT visibility, no extra sparse loss',
        max_fit_slots=256,max_optimizer_round_slots=6400,GPU_lease_seconds_cap=3600,
        source_prefix_captures_required=64,created_epoch=time.time(),deadline_epoch=time.time()+10800,
        historical_development_exposure=True,independent_confirmation=False,GT_before_all_seal=False,
        new_DINO_calls=0,new_Temporal_updates=0,new_source_training=0,LN_writeback=0,
        no_automatic_expansion=True,query_only=True,active_parameters=256,all_pretrained_parameters_frozen=True))
    put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=pins,registries=read(PRE/'RUNTIME_LOCK.json')['registries']))
    put(D/'CPU_CONTRACT_COMPLETION.json',contract());status('prepared',0,GT_read=False)
    print(json.dumps(dict(status='locked',count=64,per_domain=32,fit_slots=256)))


def verify():
    global VERIFIER
    from main_method.runtime import PinVerifier,resource_guard
    p=read(D/'PLAN_LOCK.json');l=read(D/'RUNTIME_LOCK.json');assert time.time()<p['deadline_epoch'];assert sha(D/'PLAN_LOCK.json')==l['plan_sha256']
    if VERIFIER is None:VERIFIER=PinVerifier({**l['pins'],**l['registries']})
    VERIFIER.verify();resource_guard();return p


def measure():
    import torch,numpy as np
    from scripts.run_decota_paper_main_v1 import gpu,model_for
    from scripts.run_stvg_dino_supported_expanded256_recovery002 import capture
    from methods.decota_final_simplified_v1.tensors import detached,state_hash
    from vg_tta.stvg_query_e_opd64_v1 import fit
    from vg_tta.stvg_dense_e_opd64_v1 import NativeReplay
    p=verify();lease=None
    while lease is None:
        verify()
        try:lease=gpu()
        except BlockingIOError:status('waiting_shared_GPU_lease',0,GT_read=False);time.sleep(15)
    started=time.perf_counter();put(D/'GPU_RUNTIME.json',dict(status='running',pid=os.getpid(),start_epoch=time.time(),shared_lease=True))
    model=None;domain=None;cache=collections.OrderedDict();files=[];quals=[]
    try:
        for i,r in enumerate(p['records']):
            verify();elapsed=time.perf_counter()-started;assert elapsed<p['GPU_lease_seconds_cap'],'60min GPU lease cap'
            ds=r['dataset'];dest=D/'predictions'/(r['alias']+'.pt')
            if dest.exists():load(dest);files.append(read(dest.with_suffix('.receipt.json')));continue
            if ds!=domain:
                if model is not None:del model;gc.collect();torch.cuda.empty_cache()
                domain=ds;model=model_for('vidstg'if ds=='hc2'else'hcstvg2');mh=state_hash(model.state_dict());cache.clear()
            assert torch.cuda.mem_get_info()[0]>4*2**30
            assert sha(ROOT/r['prior_prediction'])==r['prediction_sha256']and sha(ROOT/r['prior_Source'])==r['Source_sha256']
            prior=load(ROOT/r['prior_prediction']);oldsource=load(ROOT/r['prior_Source'])
            status('frozen_Source_prefix_capture',i,current=r['alias'],GT_read=False)
            torch.cuda.synchronize();tick=time.perf_counter();row,records,base,native,expert,inp,capture_seconds=capture(model,ds,r['query_ordinal'],cache)
            source=native['boxes'];assert torch.equal(source,oldsource['Source'])
            assert np.array_equal(source.double().numpy(),prior['outputs']['Source_no_correction'])
            assert row['frame_ids']==prior['frame_ids']
            assert all(torch.equal(a.cpu(),b)for a,b in zip(base.zero['logits'],oldsource['logits']))
            teacher=torch.tensor(prior['outputs']['E_DINO_support'],dtype=torch.float64)
            cast=(teacher-teacher.float().double()).abs().max().item();assert torch.isfinite(teacher).all()
            active=prior['estimate']['support_frames']>0;initial=detached(base.initial,'cpu')
            assert sum(v.numel()for v in initial.values())==1792
            with torch.no_grad():assert torch.equal(NativeReplay(base,row['frame_ids'],row['key'],{}).values()['boxes'].cpu(),source)
            fits={};fit_times={}
            for arm in ARMS[2:]:
                verify();assert time.perf_counter()-started<p['GPU_lease_seconds_cap'],'60min GPU lease cap'
                status(arm,i,current=r['alias'],GT_read=False);ft=time.perf_counter()
                z=fit(base,row['frame_ids'],row['key'],teacher,prior['planned_positions'],arm,p['configs'][ds],active)
                assert torch.equal(z['before'],source)and all(torch.equal(z['initial'][n],initial[n])for n in initial)
                base.restore(z['state'])
                with torch.no_grad():full=base.reference_values(capture=False)
                assert torch.equal(full['boxes'].cpu(),z['final']),('Full two-pass readout differs',r['alias'],arm)
                z['full_two_pass_readout_exact']=True;torch.cuda.synchronize();fit_times[arm]=time.perf_counter()-ft;fits[arm]=z;base.restore(base.initial)
            assert state_hash(model.state_dict())==mh
            if ds not in [q['dataset']for q in quals]:quals.append(dict(status='pass',dataset=ds,count=1,Source_boxes_and_logits_exact=True,all1792_initials_exact=True,only256_updated_all_LN_frozen=True,full_two_pass_readout_exact_four_arms=True,mean_gradient_checked_all_OPD_rounds=True,qualification_within_original_fit_slots=True))
            out=dict(alias=r['alias'],dataset=ds,query_ordinal=r['query_ordinal'],frame_ids=prior['frame_ids'],interval=prior['interval'],
                planned_positions=prior['planned_positions'],E_teacher_float64=teacher,E_training_teacher_float32=teacher.float(),teacher_cast_max_error=cast,
                E_support_frames=prior['estimate']['support_frames'],active=active,fits=fits,
                outputs=dict(Source_MBR=prior['outputs']['Source_no_correction'],E_MBR=prior['outputs']['E_DINO_support'],**{a:z['final'].double().numpy()for a,z in fits.items()}),
                GT_read=False,new_DINO_calls=0,initial_Source_state_SHA=mh,prior_prediction_sha256=r['prediction_sha256'],
                compute=dict(Source_capture_seconds=capture_seconds,fit_seconds=fit_times,case_seconds=time.perf_counter()-tick,new_Source_prefix_captures=1))
            files.append(save(dest,out))
            if i==1:
                costs=[load(D/'predictions'/(v['alias']+'.pt'))['compute']for v in p['records'][:2]]
                total=time.perf_counter()-started
                put(D/'PILOT2_TIMING.json',dict(status='measured',count=2,within_original64=True,elapsed_seconds=total,case_seconds=[v['case_seconds']for v in costs],fit_seconds=[v['fit_seconds']for v in costs],rough_64_seconds=32*total,not_a_guarantee=True,HC_only_pilot=True))
            print(r['alias'],i+1,'of64',round(time.perf_counter()-started,2),flush=True)
            del out,fits,z,base,row,records,native,expert,source,teacher,full,prior,oldsource;gc.collect();torch.cuda.empty_cache()
        VERIFIER.verify(full=True);put(D/'QUALIFICATION_COMPLETION.json',dict(status='pass',count=2,rows=quals))
        put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=64,fit_slots=256,files=files,GT_read=False,all_outputs_before_GT=True,GPU_lease_seconds=time.perf_counter()-started,new_Source_prefix_captures=64,new_DINO_calls=0))
        status('all64_four_outputs_sealed_before_GT',64,GT_read=False)
    finally:lease.close()


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','measure']);args=a.parse_args()
    try:globals()[args.stage]()
    except BaseException as e:
        put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',type=type(e).__name__,error=str(e),partials_preserved=True));raise
