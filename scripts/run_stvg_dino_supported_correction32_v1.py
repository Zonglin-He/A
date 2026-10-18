"""One score-supported correction arm, frozen source choice then target seal."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
from scripts.run_stvg_joint_robust_correction32_v1 import input_pool,CRITIC
from vg_tta.stvg_error_state_correction_v1 import encode
from vg_tta.stvg_joint_robust_correction32_v1 import apply
from vg_tta.stvg_dino_supported_correction32_v1 import estimate,contract
OLD=ROOT/'artifacts/stvg_joint_robust_correction32_v1'
OLDP=ROOT/'results/stvg_error_state/2026-10-18/joint_robust32'
D=ROOT/'artifacts/stvg_dino_supported_correction32_v1'
P=ROOT/'results/stvg_error_state/2026-10-18/dino_supported32'
BASES=['Before','Source'];GRID=[0.,1/16,1/8,1/4,1/2,1.,2.]


def prepare():
    import torch
    assert not(D/'PLAN_LOCK.json').exists()
    old=read(OLD/'PLAN_LOCK.json');source=read(CRITIC/'PLAN_LOCK.json')['source'];assert len(source)==96
    inputs={str((OLD/f).relative_to(ROOT)):sha(OLD/f)for f in ['RUNTIME_LOCK.json','PLAN_LOCK.json','SOURCE_CALIBRATION.json','GLOBAL_PREDICTION_BARRIER.json']}
    inputs[str((OLDP/'RESULTS.json').relative_to(ROOT))]=sha(OLDP/'RESULTS.json')
    for r in old['records']:
        f=OLD/'predictions'/(r['alias']+'.pt');load(f);inputs[str(f.relative_to(ROOT))]=sha(f)
    for r in source:
        for f in [CRITIC/'source'/(r['alias']+'.pt'),ROOT/'artifacts/stvg_joint_posterior_v1/source_supervision'/(r['alias']+'.pt')]:
            load(f);inputs[str(f.relative_to(ROOT))]=sha(f)
    for ds in ['hc2','vidstg']:
        assert len([r for r in source if r['dataset']==ds and r['split']=='source_validation'])==12
        assert not({r['parent']for r in source if r['dataset']==ds}&{r['parent']for r in old['records']if r['dataset']==ds})
    own=['vg_tta/stvg_dino_supported_correction32_v1.py','scripts/run_stvg_dino_supported_correction32_v1.py',
        'scripts/score_stvg_dino_supported_correction32_v1.py','protocols/stvg_dino_supported_correction32_v1.md',
        'vg_tta/stvg_joint_robust_correction32_v1.py','vg_tta/stvg_error_state_correction_v1.py','vg_tta/tastvg_oracle_event5_v1.py']
    put(D/'PLAN_LOCK.json',dict(status='locked_before_source_beta_calibration_and_new_target_outputs',count=32,
        records=old['records'],source=source,new_method_groups=['E_DINO_support'],bases=BASES,primary='Before/final_MBR',
        beta_grid=GRID,eligible_positive_betas=GRID[1:],beta_shared_both_domains=True,
        beta_selection='source-validation24 pooled equal-domain unobserved actual s gain; tie full Native v then smaller beta',
        transform='log(max score after1e-12 floor)-log(candidate score after1e-12 floor)',
        score_penalty_units='beta*old cutoff*relative log cost',source_train_transform_diagnostics_only=72,
        source_validation_beta_selection=24,old_scale_cutoff_unchanged=True,new_target_groups=1,
        target_GT_for_prediction_or_calibration=False,diagnostic_factorial='candidate indices x acceptance masks, C/E, not selectable method groups',
        CPU_cap_seconds=3600,new_models_experts_training_Student_updates=0,new_pixel_or_identity_labels=0,created_epoch=time.time()))
    put(D/'RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f)for f in own},inputs=inputs,
        registries={f:sha(ROOT/f)for f in ['methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json']}))
    put(D/'CPU_CONTRACT_COMPLETION.json',contract());assert not torch.cuda.is_initialized()


def verify(inputs=False):
    p=read(D/'PLAN_LOCK.json');l=read(D/'RUNTIME_LOCK.json')
    for f,h in {**l['pins'],**l['registries'],**(l['inputs']if inputs else{})}.items():assert sha(ROOT/f)==h,f
    return p


def pools_for(b,frames):return [encode(f['boxes'])-encode(b)[f['position']]for f in frames]


def actual_iou(boxes,ids,truth,width,height,ds):
    from vg_tta.tastvg_oracle_event5_v1 import box_iou
    known=np.array([int(fid)in truth for fid in ids]);out=np.zeros(len(ids))
    xy=np.concatenate((boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2),axis=1)*np.array([width,height]*2)
    if ds=='hc2':xy=np.maximum(xy,0)
    out[known]=box_iou(xy[known],np.array([truth[int(fid)]for fid in np.asarray(ids)[known]],float))
    return out,known


def calibrate():
    import torch
    from vg_tta.tastvg_oracle_event5_v1 import DenseTube
    from vg_tta.stvg_dino_supported_correction32_v1 import relative_cost
    p=verify(inputs=True);cal=read(OLD/'SOURCE_CALIBRATION.json');started=time.perf_counter();train=[];files=[]
    for r in p['source']:
        if r['split']!='source_train':continue
        z=load(CRITIC/'source'/(r['alias']+'.pt'));ff=input_pool(z['expert'])
        train.append(dict(alias=r['alias'],dataset=r['dataset'],frames=len(ff),candidates=sum(len(f['scores'])for f in ff),
            score_min=min((float(f['scores'].min())for f in ff),default=None),
            score_max=max((float(f['scores'].max())for f in ff),default=None),
            max_relative_log_cost=max((float(relative_cost(f['scores']).max())for f in ff),default=None)))
    # Generate ALL source validation alternatives before any new source metric.
    for r in p['source']:
        if r['split']!='source_validation':continue
        z=load(CRITIC/'source'/(r['alias']+'.pt'));b=z['boxes'][0].double().numpy();frames=input_pool(z['expert']);pp=pools_for(b,frames)
        setting=cal['source_domains'][r['dataset']];outputs={}
        for beta in GRID:
            e=estimate(pp,[f['scores']for f in frames],setting['scale'],setting['truncation'],beta,False)
            outputs[str(beta)]=dict(boxes=apply(b,e['delta']),estimate=e)
        files.append(save(D/'source_validation_predictions'/(r['alias']+'.pt'),dict(alias=r['alias'],dataset=r['dataset'],
            frame_ids=z['frame_ids'],input=z['input'],Native_interval=z['native_interval'],planned_positions=z['expert']['positions4'],
            Base=b,outputs=outputs,source_GT_used_for_generation=False)))
    put(D/'SOURCE_GRID_PREDICTION_BARRIER.json',dict(status='complete',count=24,files=files,betas=GRID,
        all_source_grid_outputs_before_source_evaluation=True,target_GT_read=False))
    rows=[]
    for r in p['source']:
        if r['split']!='source_validation':continue
        z=load(D/'source_validation_predictions'/(r['alias']+'.pt'));y=load(ROOT/'artifacts/stvg_joint_posterior_v1/source_supervision'/(r['alias']+'.pt'))
        assert y['split']=='source_validation'and y['source_GT'];gt=y['dense_source_GT'];ids=np.array(z['frame_ids']);base=z['Base']
        baseline,known=actual_iou(base,ids,gt,z['input']['width'],z['input']['height'],r['dataset'])
        unobs=known&~np.isin(np.arange(len(ids)),z['planned_positions']);assert unobs.any()
        row=dict(frame_ids=ids,input=z['input']);bv=DenseTube(base,row,gt,y['span'],clip=r['dataset']=='hc2').score(z['Native_interval'])['v'];vals={}
        for beta in GRID:
            b=z['outputs'][str(beta)]['boxes'];iou,_=actual_iou(b,ids,gt,z['input']['width'],z['input']['height'],r['dataset'])
            met=DenseTube(b,row,gt,y['span'],clip=r['dataset']=='hc2').score(z['Native_interval'])
            vals[str(beta)]=dict(unobserved_gain_pp=float(100*np.mean(iou[unobs]-baseline[unobs])),full_v_gain_pp=100*(met['v']-bv),
                source_known_unobserved_frames=int(unobs.sum()))
        rows.append(dict(alias=r['alias'],dataset=r['dataset'],values=vals))
    groups={str(beta):{ds:{ep:float(np.mean([r['values'][str(beta)][ep]for r in rows if r['dataset']==ds]))for ep in ['unobserved_gain_pp','full_v_gain_pp']}for ds in ['hc2','vidstg']}for beta in GRID}
    pooled={str(beta):{ep:float(np.mean([groups[str(beta)][ds][ep]for ds in ['hc2','vidstg']]))for ep in ['unobserved_gain_pp','full_v_gain_pp']}for beta in GRID}
    selected=max(GRID[1:],key=lambda b:(pooled[str(b)]['unobserved_gain_pp'],pooled[str(b)]['full_v_gain_pp'],-b))
    put(D/'BETA_CALIBRATION.json',dict(status='locked_before_target_outputs',selected_beta=selected,
        grid=GRID,positive_grid_only_eligible=True,zero_reference_not_eligible=True,groups=groups,pooled=pooled,source_rows=rows,
        source_train_diagnostics=train,source_validation_count=24,source_train_count=72,source_grid_barrier_sha256=sha(D/'SOURCE_GRID_PREDICTION_BARRIER.json'),
        target_GT_used=False,one_beta_both_domains=True,old_scale_cutoff_sha256=sha(OLD/'SOURCE_CALIBRATION.json'),
        CPU_calibration_seconds=time.perf_counter()-started))
    assert not torch.cuda.is_initialized();print('LOCKED_BETA',selected,'source_validation_pooled',pooled,flush=True)


def measure():
    import torch
    from vg_tta.stvg_joint_robust_correction32_v1 import estimate as geometric
    p=verify(inputs=True);bc=read(D/'BETA_CALIBRATION.json');cal=read(OLD/'SOURCE_CALIBRATION.json')
    assert bc['status']=='locked_before_target_outputs'and not bc['target_GT_used'];beta=bc['selected_beta'];files=[];checks=0;started=time.perf_counter()
    for r in p['records']:
        z=load(OLD/'predictions'/(r['alias']+'.pt'));setting=cal['source_domains'][cal['target_to_source'][r['dataset']]]
        frames=z['frames'];scores=[f['scores']for f in frames];bases={}
        for base in BASES:
            previous=z['bases'][base];b=previous['Base'];pp=pools_for(b,frames)
            zero=estimate(pp,scores,setting['scale'],setting['truncation'],0.,False);cc=previous['estimate_C']
            assert np.array_equal(zero['delta'],cc['delta'])and zero['selected']==cc['selected'];checks+=1
            e=estimate(pp,scores,setting['scale'],setting['truncation'],beta);out=apply(b,e['delta'])
            oldlatent=[int(np.square((v-cc['delta'])/np.asarray(setting['scale'])).sum(1).argmin())for v in pp]
            c_mask=[k>=0 for k in cc['selected']];e_mask=[k>=0 for k in e['selected']];cross={}
            for candidate in range(2):
                for mask in range(2):
                    choice=[oldlatent,e['latent_best']][candidate];accept=[c_mask,e_mask][mask]
                    active=[v[k]for v,k,yes in zip(pp,choice,accept)if yes];delta=np.mean(active,axis=0)if active else np.zeros(4)
                    cross[str(candidate*2+mask)]=apply(b,delta)
            assert np.array_equal(cross['0'],previous['outputs']['C_joint'])
            assert np.array_equal(cross['3'],out);checks+=2
            bases[base]=dict(E=out,estimate=e,old_latent=oldlatent,old_selected=cc['selected'],
                candidate_acceptance_diagnostics=cross)
        files.append(save(D/'predictions'/(r['alias']+'.pt'),dict(alias=r['alias'],dataset=r['dataset'],
            frame_ids=z['frame_ids'],interval=z['interval'],planned_positions=z['planned_positions'],frames=frames,bases=bases,
            target_GT_read=False,new_formal_method_groups=1)))
        assert time.perf_counter()-started<p['CPU_cap_seconds'];print('SEALED_E',r['alias'],flush=True)
    verify(inputs=True);assert not torch.cuda.is_initialized()
    put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,
        beta_calibration_sha256=sha(D/'BETA_CALIBRATION.json'),all_E_and_diagnostics_before_target_GT=True,
        beta_zero_C_and_factorial_endpoint_exact_checks=checks,new_formal_method_groups=1,
        CPU_generation_seconds=time.perf_counter()-started,new_models_experts_training_Student_updates=0))


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','calibrate','measure']);s=ap.parse_args().stage
    try:globals()[s]()
    except BaseException as e:
        put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',stage=s,type=type(e).__name__,error=str(e),partial_preserved=True));raise
