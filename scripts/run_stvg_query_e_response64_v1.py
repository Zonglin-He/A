"""Finite matched decoder-response measurement. No target label reader."""
from pathlib import Path
import sys,os,time,gc,collections,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,save,load,sha
OLD=ROOT/'artifacts/stvg_query_e_opd64_v1'
D=ROOT/'artifacts/stvg_query_e_response64_v1'
P=ROOT/'results/stvg_error_state/2026-10-18/query_e_response64'
VERIFIER=None


def status(stage,done,**kw):put(D/'STATUS.json',dict(status='running',stage=stage,done=done,total=64,pid=os.getpid(),timestamp=time.time(),**kw),True)


def prepare():
    old=read(OLD/'PLAN_LOCK.json');assert read(OLD/'FINAL_COMPLETION.json')['status']=='complete'
    own=['vg_tta/stvg_query_e_response64_v1.py','scripts/run_stvg_query_e_response64_v1.py','protocols/stvg_query_e_response64_v1.md','external/TA-STVG/models/grounding_model/query_decoder.py']
    pins={**read(OLD/'RUNTIME_LOCK.json')['pins'],**{f:sha(ROOT/f)for f in own}}
    for f in read(OLD/'GLOBAL_PREDICTION_BARRIER.json')['files']:pins[f['path']]=f['sha256']
    for f,h in pins.items():assert sha(ROOT/f)==h,f
    put(D/'PLAN_LOCK.json',dict(status='locked_before_new_forward_and_GT',records=old['records'],configs=old['configs'],
        count=64,per_domain=32,local_rounds=dict(hc2=[0,19,39],vidstg=[0,4,9]),local_slots=192,
        fixed_extra_case=dict(alias='H008',arm='Sparse_OPD',rounds=[2,5]),extra_not_in_primary_summary=True,
        local_factors=dict(amplitude=[1.,.5],first_moment=['history','current_only'],reference_backward=['detach','full_chain']),
        local_metric='all directions share original Adam second moment and original proposal QR norm before amplitude',
        FD_steps=[.001,.0005],FD_numerical_diagnostic_not_parameter_search=True,
        control_arms=['Sparse_control','Dense_control'],control_scales=[1.,.5,.25,.125],
        control_accept='same frozen rollout FP64 surrogate <= before + 1e-7*(1+abs(before)); log physical displacement; no GT gate',
        control_reject='restore QR and complete Adam state, next original scheduled round; no extra rounds',
        control_accept_optimizer='commit original Adam proposed moments even when QR step is scaled',
        pretrained_frozen=True,query_only_256=True,E_fixed=True,clock='original final MBR interval',
        local_not_full_adaptation_performance=True,historical_development=True,new_DINO_calls=0,new_Temporal_updates=0,
        max_fits=128,GPU_seconds_cap=2700,deadline_epoch=time.time()+7200,no_automatic_expansion=True))
    put(D/'RUNTIME_LOCK.json',dict(pins=pins,registries=read(OLD/'RUNTIME_LOCK.json')['registries'],plan_sha256=sha(D/'PLAN_LOCK.json')))
    status('prepared',0,GT_read=False)


def verify():
    global VERIFIER
    from main_method.runtime import PinVerifier,resource_guard
    p=read(D/'PLAN_LOCK.json');l=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==l['plan_sha256'];assert time.time()<p['deadline_epoch']
    if VERIFIER is None:VERIFIER=PinVerifier({**l['pins'],**l['registries']})
    VERIFIER.verify();resource_guard();return p


def measure():
    import torch,numpy as np
    from scripts.run_decota_paper_main_v1 import gpu,model_for
    from scripts.run_stvg_dino_supported_expanded256_recovery002 import capture
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.stvg_query_e_response64_v1 import local,controlled_fit
    from vg_tta.stvg_dense_e_opd64_v1 import NativeReplay
    torch.set_num_threads(4);p=verify();lease=None
    while lease is None:
        verify()
        try:lease=gpu()
        except BlockingIOError:status('waiting_shared_GPU_lease',0,GT_read=False);time.sleep(15)
    start=time.perf_counter();model=None;domain=None;cache=collections.OrderedDict();files=[]
    put(D/'GPU_RUNTIME.json',dict(status='running',pid=os.getpid(),started_epoch=time.time()))
    try:
        for i,r in enumerate(p['records']):
            verify();assert time.perf_counter()-start<p['GPU_seconds_cap'];ds=r['dataset'];dest=D/'predictions'/(r['alias']+'.pt')
            if dest.exists():load(dest);files.append(read(dest.with_suffix('.receipt.json')));continue
            if ds!=domain:
                if model is not None:del model;gc.collect();torch.cuda.empty_cache()
                domain=ds;model=model_for('vidstg'if ds=='hc2'else'hcstvg2');mh=state_hash(model.state_dict());cache.clear()
            assert torch.cuda.mem_get_info()[0]>4*2**30
            tick=time.perf_counter();status('Source_capture_and_local_response',i,current=r['alias'],GT_read=False)
            row,records,base,native,expert,inp,secs=capture(model,ds,r['query_ordinal'],cache)
            old=load(OLD/'predictions'/(r['alias']+'.pt'));assert torch.equal(native['boxes'],old['fits']['Dense_OPD']['before'])
            oldsource=load(ROOT/r['prior_Source']);assert all(torch.equal(a.cpu(),b)for a,b in zip(base.zero['logits'],oldsource['logits']))
            local_rows=[dict(arm='Dense_OPD',primary=True,**local(base,row['frame_ids'],row['key'],old['fits']['Dense_OPD'],k))for k in p['local_rounds'][ds]]
            if r['alias']=='H008':local_rows +=[dict(arm='Sparse_OPD',primary=False,**local(base,row['frame_ids'],row['key'],old['fits']['Sparse_OPD'],k))for k in [2,5]]
            fits={};times={}
            for arm in p['control_arms']:
                verify();assert time.perf_counter()-start<p['GPU_seconds_cap'];status(arm,i,current=r['alias'],GT_read=False)
                ft=time.perf_counter();z=controlled_fit(base,row['frame_ids'],row['key'],old['E_training_teacher_float32'],old['planned_positions'],p['configs'][ds],old['active'],arm=='Sparse_control')
                base.restore(z['state'])
                with torch.no_grad():assert torch.equal(base.reference_values(capture=False)['boxes'].cpu(),z['final'])
                base.restore(base.initial);fits[arm]=z;times[arm]=time.perf_counter()-ft
            assert state_hash(model.state_dict())==mh
            out=dict(alias=r['alias'],dataset=ds,frame_ids=old['frame_ids'],interval=old['interval'],planned_positions=old['planned_positions'],active=old['active'],
                local=local_rows,fits=fits,GT_read=False,compute=dict(Source_capture_seconds=secs,fit_seconds=times,case_seconds=time.perf_counter()-tick))
            files.append(save(dest,out));print(r['alias'],i+1,round(time.perf_counter()-start,1),flush=True)
            if i==1:put(D/'PILOT2_TIMING.json',dict(count=2,elapsed_seconds=time.perf_counter()-start,within_original64=True))
            del base,out,z,fits,old,oldsource,local_rows;gc.collect();torch.cuda.empty_cache()
        VERIFIER.verify(full=True)
        put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=64,local_slots=194,fit_slots=128,files=files,GT_read=False,GPU_seconds=time.perf_counter()-start))
        status('all_local_and_control_outputs_sealed',64,GT_read=False)
    finally:lease.close()


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','measure']);args=a.parse_args()
    try:globals()[args.stage]()
    except BaseException as e:put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',type=type(e).__name__,error=str(e),partials_preserved=True));raise
