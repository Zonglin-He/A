"""Seal selected source outputs, independently verify gates and public scalars."""
from pathlib import Path
import argparse, collections, json, os, sys, time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_posterior_repair_v1 import BASE,OLD,SEED,read,put,load,save,sha,condition,metric,verify
OUT=ROOT/'results/stvg_joint_posterior/2026-10-18/source_repair'

def seal_selected():
    import torch,math
    from vg_tta.stvg_joint_posterior_repair_v1 import PreconditionedJointPrior,rollout
    from scripts.run_decota_paper_main_v1 import gpu
    assert read(BASE/'TRAINING_COMPLETION.json')['count']==2
    assert not(BASE/'SELECTED_PREDICTION_BARRIER.json').exists()
    p=read(BASE/'PLAN_LOCK.json');prior_seconds=read(BASE/'GPU_BUDGET.json')['reserved_seconds'];lease=gpu();start=time.monotonic();files=[]
    try:
        def budget():
            elapsed=prior_seconds+time.monotonic()-start
            assert elapsed<14400;verify()
            put(BASE/'GPU_ROOT_BUDGET.json',dict(status='active',cumulative_reserved_seconds=elapsed,limit_seconds=14400),True)
        gen=torch.Generator().manual_seed(SEED);noise={}
        for ds in ['hc2','vidstg']:
            for r in [r for r in p['source']if r['dataset']==ds and r['split']=='source_validation']:
                noise[r['alias']]=torch.randn((1,33,4),generator=gen)
        levels=torch.logspace(math.log10(2.),math.log10(.03),16)
        for ds in ['hc2','vidstg']:
            stats=load(BASE/'statistics'/(ds+'.pt'));net=PreconditionedJointPrior(stats['mean'],stats['scale']).cuda().eval().requires_grad_(False);net.load_state_dict(load(BASE/'models'/(ds+'.pt'))['state'])
            rows=[r for r in p['source']if r['dataset']==ds and r['split']=='source_validation']
            for i,r in enumerate(rows):
                budget();z=load(OLD/'features/source'/(r['alias']+'.pt'));y=load(OLD/'source_supervision'/(r['alias']+'.pt'));c=condition(z);cc={k:v.expand((4,)+v.shape[1:])for k,v in c.items()}
                with torch.no_grad():
                    states,trace=rollout(net,cc,SEED+1000+i)
                    probes={}
                    for level in [15,8,0]:
                        sig=levels[level].cuda();x=y['state'][None].cuda()+sig*noise[r['alias']].cuda();x[:,0,2:]=0
                        probes[str(level)]=dict(input=x.cpu(),output=net(x,sig[None],c).cpu(),sigma=float(sig))
                files.append(save(BASE/'selected_predictions'/(r['alias']+'.pt'),dict(alias=r['alias'],dataset=ds,samples=states.cpu(),probes=probes,trace=trace,source_only=True,target_GT=False)))
            del net;torch.cuda.empty_cache()
        put(BASE/'SELECTED_PREDICTION_BARRIER.json',dict(status='complete',count=24,files=files,no_latent_average=True,selected_same_checkpoint=True,new_training_steps=0,new_target_predictions=0))
    finally:
        torch.cuda.synchronize();elapsed=prior_seconds+time.monotonic()-start
        put(BASE/'GPU_ROOT_COMPLETION.json',dict(status='complete',additional_reserved_seconds=elapsed-prior_seconds,cumulative_reserved_seconds=elapsed,limit_seconds=14400),True);lease.close()

def audit_export():
    import numpy as np,torch
    from scipy.stats import t
    from vg_tta.tastvg_oracle_event5_v1 import official,DenseTube
    assert not torch.cuda.is_initialized()
    p=read(BASE/'PLAN_LOCK.json');train=read(BASE/'TRAINING_COMPLETION.json');q=read(BASE/'QUALIFICATION_COMPLETION.json');assert len(train['models'])==2
    assert read(BASE/'SELECTED_PREDICTION_BARRIER.json')['count']==24
    checks=0
    for f,h in read(BASE/'RUNTIME_LOCK.json')['pins'].items():assert sha(ROOT/f)==h,f;checks+=1
    old_diag=read(ROOT/'artifacts/stvg_joint_posterior_gap_diagnosis_v1/DIAGNOSIS002.json')
    original={(r['alias'],r['level']):r['results']['GT_centered_training_kernel']for r in old_diag['source_only_probes']}
    # Recompute train-only statistics using independently collected values.
    for ds in ['hc2','vidstg']:
        yy=[load(OLD/'source_supervision'/(r['alias']+'.pt'))for r in p['source']if r['dataset']==ds and r['split']=='source_train'];st=load(BASE/'statistics'/(ds+'.pt'))
        for j in range(6):
            if j<2:a=np.array([float(y['state'][0,j])for y in yy]);mu=st['mean'][0,j];sd=st['scale'][0,j]
            else:a=np.concatenate([y['state'][1:,j-2][y['mask'][1:,j-2].bool()].numpy()for y in yy]);mu=st['mean'][1,j-2];sd=st['scale'][1,j-2]
            assert abs(float(mu)-a.mean())<2e-6 and abs(float(sd)-max(a.std(),.05))<2e-6;checks+=2
    rows=[]
    for r in [r for r in p['source']if r['split']=='source_validation']:
        ds=r['dataset'];z=load(OLD/'features/source'/(r['alias']+'.pt'));y=load(OLD/'source_supervision'/(r['alias']+'.pt'));pr=load(BASE/'selected_predictions'/(r['alias']+'.pt'));saved=next(v for v in read(BASE/'qualification'/(ds+'.json'))['rows']if v['alias']==r['alias'])
        rec=dict(alias=r['alias'],dataset=ds,Source=saved['Source'],Native_codec_control=saved['Native_codec_control'],samples=[],probes={})
        for v in pr['samples']:
            m=metric(v,z,y);decoded=__import__('scripts.run_stvg_joint_posterior_v1',fromlist=['decode_state']).decode_state(v,z)
            mm=DenseTube(decoded['boxes'],dict(frame_ids=z['frame_ids'],input=z['input']),y['dense_source_GT'],y['span'],clip=ds=='hc2').score(decoded['interval'])
            for k in ['v','t','s']:assert abs(m[k]-mm[k])<2e-9;checks+=1
            rec['samples'].append(m)
        for j,m in enumerate(rec['samples']):
            for k,v in m.items():assert abs(v-saved['samples'][j][k])<2e-6;checks+=1
        for level,v in pr['probes'].items():
            rec['probes'][level]={io:metric(v[io][0],z,y)for io in ['input','output']}
            for io in ['input','output']:
                for k,vv in rec['probes'][level][io].items():assert abs(vv-saved['probes'][level][io][k])<2e-6;checks+=1
            for k in ['v','t','s']:
                field={'v':'input_vIoU','t':'input_tIoU','s':'input_sIoU'}[k]
                assert abs(rec['probes'][level]['input'][k]-original[r['alias'],int(level)][field])<2e-6;checks+=1
        rows.append(rec)
    means={};contrasts={};gates={}
    for ds in ['hc2','vidstg']:
        rr=[r for r in rows if r['dataset']==ds];keys=['v','t','s','duration_fraction'];m={}
        for arm in ['Source','Native_codec_control','sample0','all_samples_metric_mean']:
            m[arm]={k:float(np.mean([r[arm][k]if arm in ['Source','Native_codec_control']else(r['samples'][0][k]if arm=='sample0'else np.mean([s[k]for s in r['samples']]))for r in rr]))for k in keys}
        m['probes']={l:{io:{k:float(np.mean([r['probes'][l][io][k]for r in rr]))for k in keys}for io in ['input','output']}for l in ['15','8','0']}
        g=dict(low_noise=m['probes']['15']['output']['v']-m['probes']['15']['input']['v']>=-.01,medium_denoising=m['probes']['8']['output']['v']-m['probes']['8']['input']['v']>1e-6)
        for arm in ['sample0','all_samples_metric_mean']:
            g[arm+'_v_gain']=m[arm]['v']-m['Source']['v']>1e-6;g[arm+'_time_retention']=m[arm]['t']-m['Source']['t']>=-.01;g[arm+'_duration_retention']=m[arm]['duration_fraction']>=.5*m['Source']['duration_fraction']
        saved=read(BASE/'qualification'/(ds+'.json'));assert g==saved['gates'];assert all(g.values())==saved['pass_all'];checks+=9
        candidates=[read(BASE/'validation'/ds/('epoch%02d.json'%ep))for ep in p['checkpoints']]
        ranking=[(int(v['pass_all']),v['means']['sample0']['v']if v['pass_all']else-v['weighted_validation_loss'],-ep)for ep,v in zip(p['checkpoints'],candidates)];ep=p['checkpoints'][max(range(6),key=lambda i:ranking[i])];assert ep==saved['selected_epoch'];checks+=1
        means[ds]=m;gates[ds]=dict(selected_epoch=ep,checks=g,pass_all=all(g.values()))
        vv=np.array([r['samples'][0]['v']-r['Source']['v']for r in rr]);contrasts[ds]=dict(values=vv.tolist(),mean=float(vv.mean()),CI95=[float(vv.mean()+sign*t.ppf(.975,11)*vv.std(ddof=1)/np.sqrt(12))for sign in [-1,1]],positive=int((vv>1e-8).sum()),negative=int((vv< -1e-8).sum()),neutral=int((np.abs(vv)<=1e-8).sum()))
    decision=all(g['pass_all']for g in gates.values());assert decision==(q['status']=='pass')
    budget=read(BASE/'GPU_ROOT_COMPLETION.json');assert budget['cumulative_reserved_seconds']<14400
    obs=read(BASE/'OBSERVATION_ISOLATION_COMPLETION.json');assert decision or obs['status']=='not_triggered'
    # Training/conditions are cached. No claim this equals first online E2E cost.
    upstream=read(OLD/'TRAINING_COMPLETION.json')
    result=dict(status='complete',source_only=True,count=24,means=means,gates=gates,passed=decision,rows=rows,paired_Native_sample0_minus_Source=contrasts,observation_isolation=obs,new_target_inferences=0,new_Teacher_Student_calls=0,new_association_training=0,new_prior_models=2,total_optimizer_steps=720,source_validation_not_established_unseen_to_pretrained_Student=True,selected_on_source_validation=True,not_independent_generalization_evidence=True,GPU_budget=budget,old_prior_training_time_seconds=upstream['seconds'])
    OUT.mkdir(exist_ok=True,parents=True);put(OUT/'RESULTS.json',result)
    put(BASE/'ROOT_SCALAR_COMPLETION.json',dict(status='pass',independent_checks=checks,original_physical_noise_probe_regression=True,all_old_source_and_target_model_prediction_seals_unchanged=True,source_train_only_statistics=True,selection_reproduced=True,selected_outputs_GPU_replay_matches=True,no_target_training_selection_or_new_inference=True,old_diagnosis_source_probe_fields_only_used=True))
    put(OUT/'ROOT_SCALAR_COMPLETION.json',read(BASE/'ROOT_SCALAR_COMPLETION.json'))
    history={ds:{k:v for k,v in load(BASE/'models'/(ds+'.pt')).items()if k!='state'}for ds in ['hc2','vidstg']};put(OUT/'SOURCE_TRAINING.json',history)
    publicplan={k:v for k,v in p.items()if k not in ['source','created_epoch']};put(OUT/'PROTOCOL.json',publicplan)
    put(OUT/'CPU_CONTRACT_COMPLETION.json',read(BASE/'CPU_CONTRACT_COMPLETION.json'))
    lines=['# Bounded source-only diffusion denoiser repair','','Two repaired joint priors, existing source36train/12val per domain. Source validation comes from official train and is held out only for these auxiliary models, not established unseen to pretrained Student. No target inference, no association training, no Teacher/Student calls. Original negative pilot/diagnosis and production registries preserved.','','## Prelocked qualification','','Values in percent; low and medium probes are source-GT-assisted interface diagnostics, not deployable TTA results. Native uses the prelocked first path; all four separate paths are also reported, never latent-averaged.','','| Source | Epoch | Low input/output vIoU | Medium input/output vIoU | Native Source/rollout vIoU | Source/rollout tIoU | Domain gate |','|---|---:|---:|---:|---:|---:|---|']
    for ds,m in means.items():
        pct=lambda x:f'{100*x:.6f}'
        lines.append('| '+ds+' | '+str(gates[ds]['selected_epoch'])+' | '+pct(m['probes']['15']['input']['v'])+' / '+pct(m['probes']['15']['output']['v'])+' | '+pct(m['probes']['8']['input']['v'])+' / '+pct(m['probes']['8']['output']['v'])+' | '+pct(m['Source']['v'])+' / '+pct(m['sample0']['v'])+' | '+pct(m['Source']['t'])+' / '+pct(m['sample0']['t'])+' | '+str(gates[ds]['pass_all'])+' |')
    lines+=['','Decision: '+('BOTH SOURCE GATES PASS; same-checkpoint source observation isolation executed.'if decision else'STOP: source qualification failed. Observation isolation NOT triggered; no target evaluation, identity-axis redesign, TopK or dense OPD.'),'','All six checkpoints and all12 cases/domain remain in results. If none passes, selected minimum weighted validation loss is a negative-report checkpoint, not a qualified deployment model. Source validation selection creates development exposure, not an independent confirmation.','','## What changed and what did not','','Diagonal source-train-only EDM preconditioning in RAW coordinates, matched inverse-c_out-squared loss, and actual Native-rollout validation. Original raw GT+sigma Gaussian law, .03..2 range and original .03 regression inputs retained. ROI reads physical noisy state; preconditioned inputs are not passed as physical boxes. Original fixed60 epochs,360 optimizer steps/domain and original small joint architecture retained. No extra epochs, seed search or target-dependent selection.','','Sampler remains a finite Native-warm-start VE approximation, not an exact draw from a generative prior. The16-step sampler and event-relative trajectory representation remain; physical-axis identity modeling is NOT implemented. Each of four paths is preserved; a mean of per-path scores is descriptive, not a coordinate-mean prediction.','','## Cost and audit','','Cumulative GPU reservation (including validation, selected-output replay and CPU scoring waits while holding the lease): '+f"{budget['cumulative_reserved_seconds']:.6f}"+' seconds, against14400s cap. CPU preparation/publication excluded. This is cached-feature component cost, NOT first-use end-to-end deployment latency; no new upstream capture was executed. Original feature generation and DINO costs cannot be erased by caching.','',f'Independent scalar/codec/selection/protected-seal checks: {checks}. Original raw-noise input metrics match prior CPU diagnosis; selected GPU replay matches checkpoint metrics. All private per-path states sealed locally, omitted from public export. Source visual checks are engineering view only, not semantic identity annotation.','','References: [EDM coefficients](https://github.com/NVlabs/edm/blob/main/training/networks.py), [matched EDM loss](https://github.com/NVlabs/edm/blob/main/training/loss.py), [SDA clean-estimate observation guidance](https://github.com/francois-rozet/sda/blob/master/sda/score.py). Diagonal coordinate scaling is this prototype extension, not a full EDM/SDA reproduction.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');print('ROOT_EXPORT',checks,'passed',decision,json.dumps(means))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['seal_selected','audit_export']);a=ap.parse_args();globals()[a.command]()
