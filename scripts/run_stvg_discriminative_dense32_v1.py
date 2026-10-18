"""Capture, frozen-teacher seal, offline gate, and conditional two-step center KL."""
from pathlib import Path
import os,sys,time,gc,collections,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=ROOT/'artifacts/stvg_discriminative_dense32_v1';P=ROOT/'results/stvg_error_state/2026-10-18/discriminative_dense32';PRE=ROOT/'artifacts/stvg_dino_endpoint_confirm32_v1'
CODE=['vg_tta/stvg_discriminative_dense32_v1.py','scripts/run_stvg_discriminative_dense32_v1.py','scripts/continue_stvg_discriminative_dense32_v1.py','scripts/audit_stvg_discriminative_dense32_v1.py','protocols/stvg_discriminative_dense32_v1.md']
VERIFIER=None

def status(stage,count,**kw):put(D/'STATUS.json',dict(status='running',stage=stage,count=count,total=32,pid=os.getpid(),epoch=time.time(),**kw),True)
def verify(live=False,full=False):
    global VERIFIER
    p=read(D/'PLAN_LOCK.json');l=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==l['plan_sha256']
    from main_method.runtime import PinVerifier,resource_guard
    if VERIFIER is None:VERIFIER=PinVerifier(l['pins'])
    VERIFIER.verify(full=full);resource_guard()
    if live:assert time.time()<p['deadline_epoch']
    return p

def prepare():
    from vg_tta.stvg_discriminative_dense32_v1 import contract
    old=read(PRE/'PLAN_LOCK.json');assert read(PRE/'FINAL_COMPLETION.json')['status']=='complete'
    files=CODE+['vg_tta/stvg_soft_correspondence_v1.py','vg_tta/stvg_shared_text_query32_v1.py','methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json']
    for r in old['records']:
        files +=[str((PRE/k/(r['alias']+'.pt')).relative_to(ROOT))for k in ['source','E','predictions']]
    p=dict(status='locked',count=32,per_domain=16,records=old['records'],configuration=dict(ridge=.01,temperature=.1,center_scale=.25,step_size=.5,rounds=2,candidates_per_round=17,background_grid=9,background_limit=64),
        GPU_seconds_cap=900,CPU_stage_seconds_cap=3600,created_epoch=time.time(),deadline_epoch=time.time()+3600,
        gate='both-domain unobserved parent means: ridge-minus-prior>0 AND ridge-minus-cosine>0 AND ridge-minus-shuffled>0; CI descriptive',
        feature='frozen input_proj before text fusion',current_Cohort_exposed=True,independent_confirmation=False,MBR_fixed=True,E_fixed=True,new_DINO_calls=0,no_parameter_search=True,source_baseline_commit='77d139a9595c5dcfabf67754123d5f5a4cc6e729')
    put(D/'PLAN_LOCK.json',p);pins={**read(PRE/'RUNTIME_LOCK.json')['pins'],**{f:sha(ROOT/f)for f in files}};put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=pins));put(D/'CPU_PROPERTIES.json',contract());status('prepared',0,GT_read=False)

def capture_features():
    import torch,numpy as np
    from scripts.run_decota_paper_main_v1 import gpu,model_for
    from scripts.run_stvg_dino_supported_expanded256_recovery002 import capture
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.stvg_shared_text_query32_v1 import input_pool
    p=verify(True);lease=None
    while lease is None:
        verify(True)
        try:lease=gpu()
        except BlockingIOError:status('waiting_GPU_lease',0,GT_read=False);time.sleep(15)
    tick=time.perf_counter();files=[];model=None;domain=None;cache=collections.OrderedDict()
    try:
        put(D/'GPU_RUNTIME001.json',dict(pid=os.getpid(),start_epoch=time.time(),GPU_cap_seconds=p['GPU_seconds_cap']))
        for ix,r in enumerate(p['records']):
            verify(True);assert time.perf_counter()-tick<p['GPU_seconds_cap'];a=r['alias'];ds=r['dataset'];status('capture_pre_text_appearance',ix,alias=a,GT_read=False)
            if domain!=ds:
                if model is not None:del model;gc.collect();torch.cuda.empty_cache()
                domain=ds;model=model_for('vidstg'if ds=='hc2'else'hcstvg2');mh=state_hash(model.state_dict());cache.clear()
            appearance=[];masks=[]
            def apphook(module,args,out):appearance.append(out.detach().float().cpu())
            def maskhook(module,args,out):
                _,m,*_=out[0].decompose();masks.append((~m).detach().cpu())
            hooks=[model.input_proj.register_forward_hook(apphook),model.vis_encoder.register_forward_hook(maskhook)]
            try:row,records,base,native,expert,inp,seconds=capture(model,ds,r['query_ordinal'],cache)
            finally:
                for h in hooks:h.remove()
            s=load(PRE/'source'/(a+'.pt'));e=load(PRE/'E'/(a+'.pt'));old=load(PRE/'predictions'/(a+'.pt'));assert torch.equal(native['boxes'].cpu(),s['Source'])and all(torch.equal(x.cpu(),y)for x,y in zip(base.zero['logits'],s['logits']))
            assert len(appearance)==len(masks)==2;maps=torch.empty(len(row['frame_ids']),*appearance[0].shape[1:]);valid=torch.empty(len(row['frame_ids']),*masks[0].shape[1:],dtype=torch.bool)
            for off in [0,1]:maps[off::2]=appearance[off];valid[off::2]=masks[off]
            assert torch.isfinite(maps).all()and state_hash(model.state_dict())==mh;frames=input_pool(expert);assert len(frames)==len(e['estimate']['selected'])
            files.append(save(D/'features'/(a+'.pt'),dict(alias=a,dataset=ds,maps=maps,valid=valid,frame_ids=old['frame_ids'],E=torch.as_tensor(old['outputs']['E_MBR']),frames=frames,selected=e['estimate']['selected'],planned_positions=old['planned_positions'],interval=old['interval'],Source_exact=True,source_sha=inp['sha256'],GT_read=False,capture_seconds=seconds,new_Source_captures=1,new_DINO_calls=0)))
            print(a,ix+1,round(time.perf_counter()-tick,2),flush=True);del base,native,s,e,old,maps,valid,appearance,masks,expert;gc.collect();torch.cuda.empty_cache()
            if ix==1:put(D/'PILOT2_TIMING.json',dict(count=2,elapsed_seconds=time.perf_counter()-tick,rough32_seconds=16*(time.perf_counter()-tick),not_a_guarantee=True))
        verify(full=True);put(D/'FEATURE_BARRIER.json',dict(status='complete',count=32,files=files,GPU_allocation_seconds=time.perf_counter()-tick,peak_allocated_bytes=torch.cuda.max_memory_allocated(),GT_read=False,exact_Source_all32=True))
    finally:lease.close()

def feedback():
    os.environ['CUDA_VISIBLE_DEVICES']='';import torch
    from vg_tta.stvg_discriminative_dense32_v1 import train_teachers,noise,initial_feedback
    p=verify();torch.set_num_threads(4);assert read(D/'FEATURE_BARRIER.json')['count']==32;files=[];start=time.perf_counter()
    for ix,r in enumerate(p['records']):
        verify();assert time.perf_counter()-start<p['CPU_stage_seconds_cap'];a=r['alias'];z=load(D/'features'/(a+'.pt'));cfg=p['configuration'];teachers,diag=train_teachers(z['maps'],z['valid'],z['frames'],z['selected'],cfg['ridge'])
        eps=noise(a,len(z['frame_ids']));all_E_anchors=[(frame['position'],torch.as_tensor(frame['boxes'][k],dtype=torch.float64))for frame,k in zip(z['frames'],z['selected'])if k>=0];f=initial_feedback(z['maps'],z['valid'],z['E'],teachers,eps,cfg,z['planned_positions'],all_E_anchors)
        files.append(save(D/'feedback'/(a+'.pt'),dict(alias=a,dataset=r['dataset'],teachers=teachers,training_diagnostics=diag,feedback=f,outputs=dict(E_MBR=z['E'].numpy(),Teacher_readout=f['teacher_readout'].numpy()),frame_ids=z['frame_ids'],planned_positions=z['planned_positions'],interval=z['interval'],GT_read=False)))
        print(a,'teachers',len(teachers),'positions',len(z['frame_ids']),flush=True);status('sealed_teacher_feedback',ix+1,GT_read=False)
    verify(full=True);put(D/'FEEDBACK_BARRIER.json',dict(status='complete',count=32,files=files,all32_before_GT=True,phase1_outputs=['E_MBR','Teacher_readout'],CPU_feedback_seconds=time.perf_counter()-start))

def score_phase1():
    os.environ['CUDA_VISIBLE_DEVICES']='';import torch,numpy as np
    torch.set_num_threads(4)
    from scripts.score_stvg_opd_p1_v1 import truths
    from scripts.run_stvg_dino_local_residual32_v1 import metrics,mean
    from scripts.score_stvg_joint_robust_correction32_v1 import paired
    from vg_tta.tastvg_oracle_event5_v1 import box_iou
    p=verify();bar=read(D/'FEEDBACK_BARRIER.json');assert bar['count']==32 and bar['all32_before_GT']
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt}
    strata=read(ROOT/'artifacts/stvg_evidence_strata_v1/FEATURES_PRIVATE.json')['rows'];strata={ds:{v['query_ordinal']:v for v in vv}for ds,vv in strata.items()};rows=[]
    for r in p['records']:
        a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];z=load(D/'feedback'/(a+'.pt'));feat=load(D/'features'/(a+'.pt'));f=z['feedback'];row=plans[ds][q];gg=gt[ds][0][q];ids=np.array(z['frame_ids']);known=np.array([int(i)in gg for i in ids]);unobs=~np.isin(np.arange(len(ids)),z['planned_positions']);distance=np.min(abs(ids[:,None]-ids[np.array(z['planned_positions'])][None,:]),axis=1);far=distance>=.25*(ids[-1]+1-ids[0]);b=f['candidates'].numpy();xy=np.concatenate((b[...,:2]-b[...,2:]/2,b[...,:2]+b[...,2:]/2),-1)*np.array([row['input']['width'],row['input']['height']]*2)
        if ds=='hc2':xy=np.maximum(xy,0)
        truth=np.zeros(b.shape[:2])
        for j in np.flatnonzero(known):truth[j]=box_iou(xy[j],gg[int(ids[j])])
        probs=dict(ridge=f['real']['prob'].numpy(),shuffled=f['shuffled']['prob'].numpy(),cosine=f['cosine'].numpy(),cosine_matched=f['cosine_matched'].numpy(),prior=f['real']['prior'].numpy());quality={k:(v*truth).sum(1)for k,v in probs.items()};uniform=truth.mean(1);supports={}
        for name,mask in dict(unobserved=known&unobs,observed=known&~unobs,far=known&unobs&far,all_known=known).items():
            supports[name]=dict(known_frames=int(mask.sum()),rank_vs_uniform={k:None if not mask.any()else float((v-uniform)[mask].mean())for k,v in quality.items()},rank_vs_prior={k:None if not mask.any()else float((v-quality['prior'])[mask].mean())for k,v in quality.items()},weighted_iou={k:None if not mask.any()else float(v[mask].mean())for k,v in quality.items()})
        m=metrics(z,row,gg,gt[ds][1][q]);s=strata[ds][q];rows.append(dict(alias=a,dataset=ds,active=bool(z['teachers']),E_supported=bool(feat['frames']),usable_teachers=len(z['teachers']),selected_E_anchors=sum(k>=0 for k in feat['selected']),background_count=sum(t['background_count']for t in z['training_diagnostics']if t['usable']),
            target_area_bin=s['bins']['target_area'],motion_bin=s['bins']['speed_sizes_sec'],query_form=s['query_form'],supports=supports,unknown_GT_actual_frames=int((~known).sum()),
            training=[{k:v for k,v in t.items()if k not in ['position','candidate_id']}for t in z['training_diagnostics']],ESS={k:float(np.mean(1/(v*v).sum(1)))for k,v in probs.items()},**m))
    def aggregate(rr):
        supports={}
        for support in ['unobserved','observed','far','all_known']:
            ranks={}
            for key in ['rank_vs_uniform','rank_vs_prior']:
                ranks[key]={k:paired([None if r['supports'][support][key][k]is None else 100*r['supports'][support][key][k]for r in rr])for k in ['ridge','cosine','cosine_matched','shuffled','prior']}
            paired_controls={k:paired([None if r['supports'][support]['weighted_iou']['ridge']is None else 100*(r['supports'][support]['weighted_iou']['ridge']-r['supports'][support]['weighted_iou'][k])for r in rr])for k in ['cosine','cosine_matched','shuffled']}
            supports[support]=dict(known_frames=sum(r['supports'][support]['known_frames']for r in rr),**ranks,ridge_minus_control=paired_controls)
        return dict(count=len(rr),active=sum(r['active']for r in rr),E_anchor_selected=sum(r['selected_E_anchors']for r in rr),usable_teachers=sum(r['usable_teachers']for r in rr),supports=supports,
            metrics={a:{k:mean([r['values'][a][k]for r in rr])for k in ['v','t','s','observed_actual_s','unobserved_actual_s']}for a in ['E_MBR','Teacher_readout']},
            readout_minus_E={k:paired([None if r['values']['Teacher_readout'][k]is None else 100*(r['values']['Teacher_readout'][k]-r['values']['E_MBR'][k])for r in rr])for k in ['v','s','observed_actual_s','unobserved_actual_s']})
    groups={ds:aggregate([r for r in rows if r['dataset']==ds])for ds in ['hc2','vidstg']};gate={}
    for ds,g in groups.items():
        u=g['supports']['unobserved'];conditions=dict(ridge_minus_prior_positive=u['rank_vs_prior']['ridge']['mean']is not None and u['rank_vs_prior']['ridge']['mean']>0,
            ridge_beats_cosine=u['ridge_minus_control']['cosine']['mean']is not None and u['ridge_minus_control']['cosine']['mean']>0,
            ridge_beats_shuffle=u['ridge_minus_control']['shuffled']['mean']is not None and u['ridge_minus_control']['shuffled']['mean']>0)
        gate[ds]=dict(conditions=conditions,passed=all(conditions.values()))
    passed=all(v['passed']for v in gate.values());subgroups={ds:{key:{v:aggregate([r for r in rows if r['dataset']==ds and r[key]==v])for v in sorted({r[key]for r in rows if r['dataset']==ds},key=str)}for key in ['target_area_bin','motion_bin','active']}for ds in groups}
    put(P/'PHASE1_RESULTS_PUBLIC.json',dict(status='complete',count=32,groups=groups,rows=rows,strata=subgroups,phase1_all_feedback_and_readouts_sealed_before_GT=True,unknown_preserved=True,independent_confirmation=False,cohort_development_exposed=True))
    put(P/'GATE.json',dict(status='complete',groups=gate,passed=passed,rule=p['gate'],action='run_locked_two_rounds'if passed else'stop_this_teacher_no_optimizer_search'));put(D/'PHASE1_SCORE_COMPLETION.json',dict(status='complete',count=32,passed=passed));print(dict(passed=passed,groups={ds:g['supports']['unobserved']for ds,g in groups.items()}),flush=True)

def adapt():
    os.environ['CUDA_VISIBLE_DEVICES']='';import torch
    from vg_tta.stvg_discriminative_dense32_v1 import distill
    p=verify();assert read(P/'GATE.json')['passed'];torch.set_num_threads(4);files=[];start=time.perf_counter()
    for ix,r in enumerate(p['records']):
        verify();assert time.perf_counter()-start<p['CPU_stage_seconds_cap'];a=r['alias'];feat=load(D/'features'/(a+'.pt'));old=load(D/'feedback'/(a+'.pt'));fits={}
        for name,dynamic in [('Fixed_dense',False),('On_policy_dense',True)]:fits[name]=distill(feat['maps'],feat['valid'],feat['E'],old['teachers'],old['feedback']['noise'],p['configuration'],feat['planned_positions'],dynamic)
        assert torch.equal(fits['Fixed_dense']['path'][1]['boxes'],fits['On_policy_dense']['path'][1]['boxes'])
        files.append(save(D/'predictions'/(a+'.pt'),dict(alias=a,dataset=r['dataset'],frame_ids=feat['frame_ids'],planned_positions=feat['planned_positions'],interval=feat['interval'],
            outputs={**old['outputs'],**{k:v['final'].numpy()for k,v in fits.items()}},fits=fits,GT_read_in_online_process=False,Phase1_offline_GT_global_gate_exposure=True)))
        print(a,ix+1,flush=True)
    put(D/'ADAPTATION_BARRIER.json',dict(status='complete',count=32,files=files,CPU_adaptation_seconds=time.perf_counter()-start,all_phase2_predictions_before_phase2_GT=True,Phase1_GT_previously_used_for_global_gate=True))

def score_phase2():
    os.environ['CUDA_VISIBLE_DEVICES']='';import torch
    from scripts.score_stvg_opd_p1_v1 import truths
    from scripts.run_stvg_dino_local_residual32_v1 import metrics,mean
    from scripts.score_stvg_joint_robust_correction32_v1 import paired
    from vg_tta.stvg_discriminative_dense32_v1 import ARMS
    p=verify();bar=read(D/'ADAPTATION_BARRIER.json');assert bar['count']==32
    gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};rows=[]
    for r in p['records']:
        a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];o=load(D/'predictions'/(a+'.pt'));m=metrics(o,plans[ds][q],gt[ds][0][q],gt[ds][1][q]);rows.append(dict(alias=a,dataset=ds,**m))
    groups={}
    for ds in ['hc2','vidstg']:
        rr=[r for r in rows if r['dataset']==ds];eps=['v','t','s','observed_actual_s','unobserved_actual_s'];groups[ds]=dict(count=16,metrics={a:{k:mean([r['values'][a][k]for r in rr])for k in eps}for a in ARMS},
            paired={a+'_minus_'+b:{k:paired([None if r['values'][a][k]is None else 100*(r['values'][a][k]-r['values'][b][k])for r in rr])for k in eps if k!='t'}for a,b in [('Fixed_dense','E_MBR'),('On_policy_dense','E_MBR'),('On_policy_dense','Fixed_dense'),('On_policy_dense','Teacher_readout')]})
    put(P/'PHASE2_RESULTS_PUBLIC.json',dict(status='complete',count=32,groups=groups,rows=rows,MBR_fixed=True,only_external_center_state=True,pretrained_weights_frozen=True,phase1_GT_global_gate_exposure=True));put(D/'PHASE2_SCORE_COMPLETION.json',dict(status='complete',count=32))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','capture_features','feedback','score_phase1','adapt','score_phase2']);args=ap.parse_args()
    try:globals()[args.stage]()
    except BaseException as e:put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',stage=args.stage,type=type(e).__name__,error=str(e),all_partials_preserved=True));raise
