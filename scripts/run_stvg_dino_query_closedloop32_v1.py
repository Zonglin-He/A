"""Finite32 E-retaining QR closed loop: seal all predictions before scoring."""
from pathlib import Path
import os,sys,time,collections,gc,argparse,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=ROOT/'artifacts/stvg_dino_query_closedloop32_v1';P=ROOT/'results/stvg_error_state/2026-10-18/dino_query_closedloop32';PRE=ROOT/'artifacts/stvg_dino_endpoint_confirm32_v1'
CODE=['vg_tta/stvg_dino_query_closedloop32_v1.py','scripts/run_stvg_dino_query_closedloop32_v1.py','scripts/score_stvg_dino_query_closedloop32_v1.py','scripts/audit_stvg_dino_query_closedloop32_v1.py','scripts/report_stvg_dino_query_closedloop32_v1.py','scripts/continue_stvg_dino_query_closedloop32_v1.py','protocols/stvg_dino_query_closedloop32_v1.md']
VERIFIER=None
PRIOR_GPU_SECONDS=0.

def status(stage,count,**kw):put(D/'STATUS.json',dict(status='running',stage=stage,count=count,total=32,pid=os.getpid(),epoch=time.time(),**kw),True)
def verify(live=False,full=False):
    global VERIFIER
    p=read(D/'PLAN_LOCK.json');l=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==l['plan_sha256']
    if live:assert time.time()<p['deadline_epoch']
    from main_method.runtime import PinVerifier,resource_guard
    if VERIFIER is None:VERIFIER=PinVerifier(l['pins'])
    VERIFIER.verify(full=full);resource_guard();return p

def prepare():
    from vg_tta.stvg_dino_query_closedloop32_v1 import properties,ARMS
    assert read(PRE/'FINAL_COMPLETION.json')['status']=='complete';old=read(PRE/'PLAN_LOCK.json')
    files=CODE+['vg_tta/stvg_query_e_opd64_v1.py','vg_tta/stvg_dense_e_opd64_v1.py','vg_tta/c1_enabling_tricks_v1.py','methods/decota_spatial_opd_v1/configs.json']
    for r in old['records']:
        files += [str((PRE/k/(r['alias']+'.pt')).relative_to(ROOT))for k in ['source','E','predictions']]
    pins={**read(PRE/'RUNTIME_LOCK.json')['pins'],**{f:sha(ROOT/f)for f in files}}
    cfg=read(ROOT/'methods/decota_spatial_opd_v1/configs.json')['datasets'];lr={ds:cfg[ds]['config']['lr']for ds in ['hc2','vidstg']}
    put(D/'PLAN_LOCK.json',dict(status='locked',records=old['records'],count=32,per_domain=16,arms=ARMS,seed=20261018,
        lr=lr,scale=old['scale'],threshold=.35,rounds=2,candidates=4,loss='masked diagonal Gaussian forward KL at four anchors, normalization16',
        current_E_baseline_exact=True,first_fixed_dynamic_round_exact=True,noise_indices=[[0,1,2,3],[4,5,6,7]],
        GPU_seconds_cap=900,created_epoch=time.time(),deadline_epoch=time.time()+3600,MBR_fixed=True,GT_after_full32_seal=True,
        no_parameter_search=True,cohort_development_exposed=True,independent_confirmation=False,read_final_step=2))
    put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=pins));put(D/'CPU_PROPERTIES.json',properties());status('prepared',0,GT_read=False)


def measure():
    global PRIOR_GPU_SECONDS
    import torch,numpy as np
    from scripts.run_decota_paper_main_v1 import gpu,model_for,frames_for,read_row
    from scripts.run_stvg_dino_supported_expanded256_recovery002 import capture
    from methods.decota_final_simplified_v1.tensors import state_hash,detached
    from methods.decota_final_simplified_v1.observations import SpatialExpert,target_tokens
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
    from vg_tta.stvg_dino_mask_align32_v1 import MaskDecoder
    from vg_tta.stvg_dino_query_closedloop32_v1 import fit,FIT_ARMS
    p=verify(True);PRIOR_GPU_SECONDS=read(D/'RECOVERY_ENGINEERING_RECEIPT001.json')['prior_GPU_seconds_upper_bound'];lease=None
    while lease is None:
        verify(True)
        try:lease=gpu()
        except BlockingIOError:status('waiting_GPU_lease',0,GT_read=False);time.sleep(15)
    start=time.perf_counter();files=[];model=None;domain=None;cache=collections.OrderedDict();qualified=set();checks=[]
    def check():
        verify(True);assert PRIOR_GPU_SECONDS+time.perf_counter()-start<p['GPU_seconds_cap'],'15min total GPU cap';assert torch.cuda.mem_get_info()[0]>4*2**30
    try:
        expert=SpatialExpert(ROOT/EXPERT_SNAPSHOT);eh=state_hash(expert.model.state_dict())
        put(D/'GPU_RUNTIME.json',dict(status='active',pid=os.getpid(),start_epoch=time.time(),shared_lease=True))
        for ix,r in enumerate(p['records']):
            check();a=r['alias'];ds=r['dataset'];tick=time.perf_counter();status('query_capture_and_closedloop',ix,alias=a,GT_read=False)
            if domain!=ds:
                if model is not None:del model;gc.collect();torch.cuda.empty_cache()
                domain=ds;model=model_for('vidstg'if ds=='hc2'else'hcstvg2');mh=state_hash(model.state_dict());cache.clear()
            old=load(PRE/'predictions'/(a+'.pt'));s=load(PRE/'source'/(a+'.pt'));e=load(PRE/'E'/(a+'.pt'))
            row,records,base,native,_,inp,capture_seconds=capture(model,ds,r['query_ordinal'],cache)
            assert torch.equal(native['boxes'].cpu(),s['Source'])and native['physical_interval']==s['Native_interval']
            assert all(torch.equal(x.cpu(),y)for x,y in zip(base.zero['logits'],s['logits']))
            save(D/'replay_inputs'/(a+'.pt'),dict(spatial_inputs=detached(base.spatial_inputs,'cpu'),initial=detached(base.initial,'cpu'),Source=s['Source'],GT_read=False))
            source=s['Source'];E=torch.as_tensor(old['outputs']['E_MBR'],dtype=torch.float64);caches=[];quals=[]
            if old['active']:
                frames,ids,_=frames_for(ds,row,cache);assert list(ids)==old['frame_ids']
                for pos in old['planned_positions']:
                    rc=s['expert']['observations'][('original',pos)]['receipt'];assert hashlib.sha256(frames[pos].tobytes()).hexdigest()==rc['rgb_sha256']
                    text=rc['text'];inputs=expert.processor(images=frames[pos],text=text,return_tensors='pt').to('cuda');offs=expert.processor.tokenizer(text,return_offsets_mapping=True)['offset_mapping']
                    if rc['context_active']:ent=target_tokens(text,offs,rc['target_span'])
                    else:
                        entity=row['parses']['old']['entity'];k=text.find(entity);assert k>=0;ent=target_tokens(text,offs,[k,k+len(entity)])
                    c=MaskDecoder(expert.model,inputs,ent);tr=c.native_trace();saved=old['native'][pos]
                    assert np.max(abs(tr['layer_boxes']-saved['layer_boxes']))<1e-5 and np.max(abs(tr['layer_scores']-saved['layer_scores']))<1e-5 and tr['high_slot']==saved['high_slot']
                    if ds not in qualified:quals.append(c.qualify_mask())
                    caches.append(c)
                qualified.add(ds)
            fits={};fit_seconds={}
            for arm in FIT_ARMS:
                check();ft=time.perf_counter();f=fit(base,row['frame_ids'],row['key'],source,E,old,e['estimate']['delta'],caches,arm,p['lr'][ds],check);torch.cuda.synchronize();fit_seconds[arm]=time.perf_counter()-ft;fits[arm]=f;save(D/'fits'/(a+'_'+arm+'.pt'),f)
            f,g=fits['Fixed_sample_KL'],fits['On_policy_KL'];assert torch.equal(f['path'][1]['boxes'],g['path'][1]['boxes'])and torch.equal(f['path'][1]['state']['spatial.query_residual'],g['path'][1]['state']['spatial.query_residual'])
            assert torch.equal(f['rounds'][0]['QR_step'],g['rounds'][0]['QR_step'])and torch.equal(f['rounds'][0]['targets'],g['rounds'][0]['targets'])and torch.equal(f['rounds'][0]['mask'],g['rounds'][0]['mask'])
            if f['rounds'][0]['updated']:assert torch.equal(f['rounds'][0]['QR_gradient'],g['rounds'][0]['QR_gradient'])
            assert state_hash(model.state_dict())==mh and state_hash(expert.model.state_dict())==eh
            outputs=dict(E_MBR=old['outputs']['E_MBR'],Center_full_local=old['outputs']['Center_full_local'],**{k:v['final'].numpy()for k,v in fits.items()})
            cost=dict(case_seconds=time.perf_counter()-tick,Source_capture_seconds=capture_seconds,new_Source_captures=1,DINO_encoding_frames=len(caches),
                DINO_native_encoding_seconds=sum(c.encoding_full_forward_seconds for c in caches),fit_seconds=fit_seconds,
                new_conditional_queries=sum(v['count']['conditional_logical_queries']for v in fits.values()),new_conditional_decoder_seconds=sum(v['count']['conditional_decoder_seconds']for v in fits.values()))
            files.append(save(D/'predictions'/(a+'.pt'),dict(alias=a,dataset=ds,frame_ids=old['frame_ids'],planned_positions=old['planned_positions'],interval=old['interval'],active=old['active'],
                outputs=outputs,fits=fits,qualifications=quals,compute=cost,GT_read=False,first_fixed_dynamic_round_exact=True,all_pretrained_weights_frozen=True)))
            if ix==1:put(D/'PILOT2_TIMING.json',dict(status='measured',count=2,elapsed_seconds=time.perf_counter()-start,fit_seconds=fit_seconds,rough32_seconds=16*(time.perf_counter()-start),not_a_guarantee=True))
            print(a,ix+1,round(time.perf_counter()-start,2),flush=True)
            del base,native,caches,fits,f,g,source,E,old,s,e,records;gc.collect();torch.cuda.empty_cache()
        check();verify(full=True);put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,all_outputs_before_GT=True,GT_read=False,
            GPU_allocation_seconds=PRIOR_GPU_SECONDS+time.perf_counter()-start,final_worker_GPU_seconds=time.perf_counter()-start,prior_engineering_GPU_seconds_upper_bound=PRIOR_GPU_SECONDS,peak_allocated_bytes=torch.cuda.max_memory_allocated(),qualified_domains=sorted(qualified)))
        status('all32_sealed_before_GT',32,GT_read=False)
    finally:lease.close()
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','measure']);args=parser.parse_args()
    try:globals()[args.stage]()
    except BaseException as e:put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',type=type(e).__name__,error=str(e),partials_preserved=True));raise
