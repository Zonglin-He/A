"""At most two source-only repaired denoisers, strict cumulative 4h GPU cap."""
from pathlib import Path
import argparse, gc, json, math, os, sys, time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_posterior_v1 import BASE as OLD,read,put,load,save,sha,condition,batch_conditions,decode_state
BASE=ROOT/'artifacts/stvg_joint_posterior_repair_v1'
SEED=20261018
EPOCHS=[10,20,30,40,50,60]

def status(stage, **kw):
    put(BASE/'STATUS.json',dict(status='running',stage=stage,pid=os.getpid(),**kw),True)

def prepare():
    import torch
    assert not(BASE/'PLAN_LOCK.json').exists()
    p=read(OLD/'PLAN_LOCK.json');source=p['source'];assert len(source)==96
    pins={}
    for r in source:
        for folder in ['features/source','source_supervision']:
            f=OLD/folder/(r['alias']+'.pt');load(f);pins[str(f.relative_to(ROOT))]=sha(f)
    for f in list((OLD/'models').glob('*.pt'))+list((OLD/'predictions').glob('*.pt'))+list((OLD/'source_validation_predictions').glob('*.pt')):
        pins[str(f.relative_to(ROOT))]=sha(f)
    for f in ['methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json','vg_tta/stvg_joint_posterior_v1.py','scripts/run_stvg_joint_posterior_v1.py','vg_tta/stvg_joint_posterior_repair_v1.py','scripts/run_stvg_joint_posterior_repair_v1.py','protocols/stvg_joint_posterior_repair_v1.md']:
        pins[f]=sha(ROOT/f)
    plan=dict(status='locked_before_new_training',source=source,parent_plan_sha256=sha(OLD/'PLAN_LOCK.json'),source_train_per_domain=36,source_validation_per_domain=12,
        source_validation_origin='official source train; held out only for new auxiliary network, not established unseen to pretrained Student',
        new_models_max=2,new_association_models=0,new_target_inferences=0,new_Teacher_Student_calls=0,seed=SEED,
        epochs=60,batch_size=6,lr=.0003,weight_decay=.0001,gradient_clip=1.,checkpoints=EPOCHS,
        raw_sigma_min=.03,raw_sigma_max=2.,training_kernel='raw GT + isotropic raw-logit Gaussian; no Native training',
        scale='source_train only, two temporal coordinate stats plus four shared spatial-coordinate stats, std floor.05; diagonal EDM in raw coordinates',
        precondition_loss='1/c_out^2; per-query equal temporal/spatial block weight',
        qualification=dict(low_raw_sigma03_vIoU_delta_min=-.01,medium_original_level=8,medium_vIoU_delta_min=1e-6,
            native_vIoU_gain_min=1e-6,native_tIoU_delta_min=-.01,native_mean_duration_ratio_min=.5,
            native_gate_both_sample0_and_mean_per_sample=True),
        selection='highest Native sample0 vIoU among all within-domain gates passing; earliest tie; if none minimum multi-noise EDM val loss for negative report only',
        observations_if_both_pass=['none','fixed_inlier_prior_0.5_with_outlier','old_frozen_association_with_outlier'],
        source_only=True,target_GT_access=False,created_epoch=time.time(),gpu_reservation_limit_seconds=14400,
        sampler='old16level raw VE Native warm-start,4 distinct samples, no latent averaging, sample0 primary; approximate not unconditional prior sampling',
        target_or_OPD_or_identity_axis_implementation='not_authorized_in_this_stage')
    put(BASE/'PLAN_LOCK.json',plan);put(BASE/'RUNTIME_LOCK.json',dict(status='locked',pins=pins))
    # No model initialization or optimizer here; these are source-train statistics.
    for ds in ['hc2','vidstg']:
        rr=[r for r in source if r['dataset']==ds and r['split']=='source_train']
        yy=[load(OLD/'source_supervision'/(r['alias']+'.pt'))for r in rr]
        mean=torch.zeros(33,4);scale=torch.ones(33,4)
        for j in range(2):
            a=torch.stack([y['state'][0,j]for y in yy]);mean[0,j]=a.mean();scale[0,j]=a.std(unbiased=False).clamp_min(.05)
        for j in range(4):
            a=torch.cat([y['state'][1:,j][y['mask'][1:,j].bool()]for y in yy]);mean[1:,j]=a.mean();scale[1:,j]=a.std(unbiased=False).clamp_min(.05)
        save(BASE/'statistics'/(ds+'.pt'),dict(mean=mean,scale=scale,source_train_only=True,source_count=36))
    put(BASE/'PREPARATION_COMPLETION.json',dict(status='complete',count=96,source_train=72,source_val=24,target_payloads_loaded=0))
    status('prepared',done=0,total=2)

def verify():
    from main_method.runtime import resource_guard
    resource_guard()
    for f,h in read(BASE/'RUNTIME_LOCK.json')['pins'].items():
        # Large immutable input seals are checked at preparation and final audit.
        if not f.startswith('artifacts/'):
            assert sha(ROOT/f)==h,f

def metric(x,z,y):
    from vg_tta.tastvg_oracle_event5_v1 import official
    d=decode_state(x.detach().cpu(),z)
    m=official(d['boxes'],dict(frame_ids=z['frame_ids'],input=z['input']),y['dense_source_GT'],y['span'],d['interval'],z['dataset'])
    m['duration_fraction']=(d['interval'][1]-d['interval'][0])/(z['frame_ids'][-1]+1-z['frame_ids'][0])
    return m

def evaluation(net,rr,zs,ys,noise,budget,full=True):
    import torch,numpy as np
    from vg_tta.stvg_joint_posterior_repair_v1 import rollout
    net.eval();rows=[];noise_levels=torch.logspace(math.log10(2.),math.log10(.03),16)
    for i,r in enumerate(rr):
        budget();z,y=zs[i],ys[i];c=condition(z);truth=y['state'][None].cuda();mask=y['mask'][None].cuda();eps=noise[r['alias']].cuda()
        probes={};weighted=[]
        for level in [15,8,0]:
            s=noise_levels[level].cuda();x=truth+s*eps;x[:,0,2:]=0
            with torch.no_grad():pred=net(x,s[None],c);vl=net.weighted_loss(pred,truth,s[None],mask)
            probes[str(level)]=dict(sigma=float(s),input=metric(x[0],z,y),output=metric(pred[0],z,y))
            weighted.append(float(vl))
        with torch.no_grad():
            cc={k:v.expand((4,)+v.shape[1:])for k,v in c.items()}
            xx,trace=rollout(net,cc,SEED+1000+i)
        sample_metrics=[metric(v,z,y)for v in xx]
        source=metric(c['native'][0],z,y)
        from vg_tta.tastvg_oracle_event5_v1 import official
        raw=official(z['boxes'][0].numpy(),dict(frame_ids=z['frame_ids'],input=z['input']),y['dense_source_GT'],y['span'],z['native_interval'],z['dataset'])
        raw['duration_fraction']=(z['native_interval'][1]-z['native_interval'][0])/(z['frame_ids'][-1]+1-z['frame_ids'][0])
        rows.append(dict(alias=r['alias'],dataset=r['dataset'],Source=raw,Native_codec_control=source,probes=probes,samples=sample_metrics,weighted_validation_loss=float(np.mean(weighted))))
    keys=['v','t','s','duration_fraction']
    means={k:{m:float(np.mean([r[k][m]for r in rows]))for m in keys}for k in ['Source','Native_codec_control']}
    means['sample0']={m:float(np.mean([r['samples'][0][m]for r in rows]))for m in keys}
    means['all_samples_metric_mean']={m:float(np.mean([s[m]for r in rows for s in r['samples']]))for m in keys}
    pm={str(l):{io:{m:float(np.mean([r['probes'][str(l)][io][m]for r in rows]))for m in keys}for io in ['input','output']}for l in [15,8,0]}
    gates=dict(low_noise=pm['15']['output']['v']-pm['15']['input']['v']>=-.01,
        medium_denoising=pm['8']['output']['v']-pm['8']['input']['v']>1e-6)
    for key in ['sample0','all_samples_metric_mean']:
        gates[key+'_v_gain']=means[key]['v']-means['Source']['v']>1e-6
        gates[key+'_time_retention']=means[key]['t']-means['Source']['t']>=-.01
        gates[key+'_duration_retention']=means[key]['duration_fraction']>=.5*means['Source']['duration_fraction']
    return dict(rows=rows,means=means,probe_means=pm,gates=gates,pass_all=all(gates.values()),weighted_validation_loss=float(np.mean([r['weighted_validation_loss']for r in rows])))

def train():
    import torch,numpy as np
    from vg_tta.stvg_joint_posterior_repair_v1 import PreconditionedJointPrior
    from scripts.run_decota_paper_main_v1 import gpu
    verify();p=read(BASE/'PLAN_LOCK.json');assert not(BASE/'TRAINING_STARTED.json').exists(),'no automatic restart or third model'
    put(BASE/'TRAINING_STARTED.json',dict(status='started',pid=os.getpid(),time=time.time(),max_models=2))
    status('waiting_shared_GPU_lease',done=0,total=2);lease=gpu();started=time.monotonic();models=[]
    def budget():
        torch.cuda.synchronize();elapsed=time.monotonic()-started
        put(BASE/'GPU_BUDGET.json',dict(status='active',reserved_seconds=elapsed,limit_seconds=14400),True)
        assert elapsed<14400,'cumulative GPU-reservation limit reached'
        verify()
    try:
        put(BASE/'EXECUTION_RUNTIME.json',dict(status='running',pid=os.getpid(),starttick=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19],device=torch.cuda.get_device_name(),GPU_lease_acquired_epoch=time.time()))
        torch.cuda.set_per_process_memory_fraction(.5)
        for ds in ['hc2','vidstg']:
            rr=[r for r in p['source']if r['dataset']==ds];zs=[load(OLD/'features/source'/(r['alias']+'.pt'))for r in rr];ys=[load(OLD/'source_supervision'/(r['alias']+'.pt'))for r in rr]
            ti=[i for i,r in enumerate(rr)if r['split']=='source_train'];vi=[i for i,r in enumerate(rr)if r['split']=='source_validation'];assert len(ti)==36 and len(vi)==12
            stats=load(BASE/'statistics'/(ds+'.pt'));torch.manual_seed(SEED);net=PreconditionedJointPrior(stats['mean'],stats['scale']).cuda();opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.0001)
            rng=np.random.default_rng(SEED);history=[];candidates=[];best=None;best_state=None;valrows=[rr[i]for i in vi];valzs=[zs[i]for i in vi];valys=[ys[i]for i in vi]
            # Reproduce EXACT old CPU generator/order, including preceding HC domain.
            gen=torch.Generator().manual_seed(SEED);noises={}
            for domain in ['hc2','vidstg']:
                for r in [r for r in p['source']if r['dataset']==domain and r['split']=='source_validation']:
                    noises[r['alias']]=torch.randn((1,33,4),generator=gen)
            updates=0
            for epoch in range(1,61):
                budget();net.train();losses=[];order=rng.permutation(ti)
                for start in range(0,36,6):
                    idx=order[start:start+6];c=batch_conditions([zs[i]for i in idx]);truth=torch.stack([ys[i]['state']for i in idx]).cuda();mask=torch.stack([ys[i]['mask']for i in idx]).cuda()
                    sig=torch.empty(len(idx),device='cuda').uniform_(math.log(.03),math.log(2.)).exp();x=truth+sig[:,None,None]*torch.randn_like(truth);x[:,0,2:]=0
                    opt.zero_grad();pred=net(x,sig,c);loss=net.weighted_loss(pred,truth,sig,mask);assert torch.isfinite(loss)
                    loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();updates+=1;losses.append(float(loss.detach()))
                h=dict(epoch=epoch,train_loss=float(np.mean(losses)))
                if epoch in EPOCHS:
                    ev=evaluation(net,valrows,valzs,valys,noises,budget)
                    put(BASE/'validation'/ds/('epoch%02d.json'%epoch),ev)
                    h.update(validation_loss=ev['weighted_validation_loss'],Native_sample0_vIoU=ev['means']['sample0']['v'],qualification=ev['pass_all'])
                    rank=(int(ev['pass_all']),ev['means']['sample0']['v']if ev['pass_all']else-ev['weighted_validation_loss'],-epoch)
                    candidates.append(dict(epoch=epoch,rank=rank,gates=ev['gates']))
                    if best is None or rank>best:
                        best=rank;best_epoch=epoch;best_state={k:v.detach().cpu().clone()for k,v in net.state_dict().items()};selected=ev
                    print('CHECKPOINT',ds,epoch,json.dumps(h),flush=True)
                history.append(h);status('source_training',dataset=ds,epoch=epoch,optimizer_steps=updates,done=len(models),total=2)
            receipt=save(BASE/'models'/(ds+'.pt'),dict(state=best_state,selected_epoch=best_epoch,history=history,candidates=candidates,optimizer_steps=updates,parameter_count=sum(v.numel()for v in net.parameters()),source_only=True,qualification=selected['pass_all']))
            put(BASE/'qualification'/(ds+'.json'),dict(dataset=ds,selected_epoch=best_epoch,**selected))
            models.append(dict(dataset=ds,selected_epoch=best_epoch,optimizer_steps=updates,parameter_count=sum(v.numel()for v in net.parameters()),pass_all=selected['pass_all'],model_receipt=receipt))
            del net,opt,zs,ys,best_state;gc.collect();torch.cuda.empty_cache()
        gate=all(m['pass_all']for m in models)
        put(BASE/'TRAINING_COMPLETION.json',dict(status='complete',count=2,models=models,optimizer_steps=sum(m['optimizer_steps']for m in models),GPU_reserved_seconds=time.monotonic()-started))
        put(BASE/'QUALIFICATION_COMPLETION.json',dict(status='pass'if gate else'fail',count=24,domains=models,observation_isolation='permitted_next'if gate else'not_triggered',target_or_OPD='not_started'))
        if gate:
            observation_isolation(budget)
        else:
            put(BASE/'OBSERVATION_ISOLATION_COMPLETION.json',dict(status='not_triggered',reason='at least one source qualification gate failed',new_models=0,new_inference_conditions=0))
        put(BASE/'GPU_COMPLETION.json',dict(status='complete',reserved_seconds=time.monotonic()-started,limit_seconds=14400,allocation_includes_validation_CPU_wait=True,new_models=2,new_association_training=0,new_target_inferences=0))
        status('complete',done=2,total=2,qualification='pass'if gate else'fail',model_work_finished=True)
    finally:
        torch.cuda.synchronize();put(BASE/'GPU_BUDGET.json',dict(status='released',reserved_seconds=time.monotonic()-started,limit_seconds=14400),True);lease.close()

def observation_isolation(budget):
    import torch,numpy as np
    from vg_tta.stvg_joint_posterior_v1 import Association,roi
    from vg_tta.stvg_joint_posterior_repair_v1 import PreconditionedJointPrior,rollout
    p=read(BASE/'PLAN_LOCK.json');rows=[]
    for ds in ['hc2','vidstg']:
        stats=load(BASE/'statistics'/(ds+'.pt'));net=PreconditionedJointPrior(stats['mean'],stats['scale']).cuda().eval().requires_grad_(False);net.load_state_dict(load(BASE/'models'/(ds+'.pt'))['state'])
        old=load(OLD/'models'/(ds+'_association.pt'));assoc=Association().cuda().eval().requires_grad_(False);assoc.load_state_dict(old['state'])
        rr=[r for r in p['source']if r['dataset']==ds and r['split']=='source_validation']
        for i,r in enumerate(rr):
            budget();z=load(OLD/'features/source'/(r['alias']+'.pt'));y=load(OLD/'source_supervision'/(r['alias']+'.pt'));c=condition(z);lo,hi=z['frame_ids'][0],z['frame_ids'][-1]+1;aa=z['expert']['anchors']['single4']
            obs=dict(time=torch.tensor([(z['frame_ids'][a['position']]-lo)/(hi-lo)for a in aa],device='cuda'),boxes=torch.tensor([a['box']for a in aa],device='cuda').reshape(-1,4))
            with torch.no_grad():
                pp=assoc(roi(c['maps'],obs['boxes'][None],obs['time'][None])[0],c['query'].mean(1).expand(len(aa),-1)).sigmoid()if aa else torch.zeros(0,device='cuda')
            cc={k:v.expand((4,)+v.shape[1:])for k,v in c.items()};arms={}
            for name,prior in [('none',None),('fixed',torch.full_like(pp,.5)),('learned',pp)]:
                x,tr=rollout(net,cc,SEED+1000+i,None if name=='none'else obs,prior,old['variance'])
                arms[name]=dict(samples=[metric(v,z,y)for v in x],trace=tr)
            save(BASE/'observation_states'/(r['alias']+'.pt'),dict(arms=arms,source_only=True))
            rows.append(dict(alias=r['alias'],dataset=ds,arms=arms))
        del net,assoc;gc.collect();torch.cuda.empty_cache()
    put(BASE/'OBSERVATION_ISOLATION_COMPLETION.json',dict(status='complete',count=24,rows=rows,new_models=0,new_training=0,target_GT_used=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','train']);args=parser.parse_args()
    try:globals()[args.command]()
    except BaseException as e:
        put(BASE/'FAILURE.json',dict(status='failed',stage=args.command,error=type(e).__name__,message=str(e),pid=os.getpid()),True);raise
