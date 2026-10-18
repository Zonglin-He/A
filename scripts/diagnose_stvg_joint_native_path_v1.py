"""Frozen source-only initialization/denoiser/sampler diagnosis; no optimizers."""
from pathlib import Path
import argparse,math,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_posterior_repair_v1 import BASE as REPAIR,OLD,SEED,read,put,load,save,sha,condition,metric
BASE=ROOT/'artifacts/stvg_joint_native_path_diagnosis_v1'
OUT=ROOT/'results/stvg_joint_posterior/2026-10-18/native_path_diagnosis'

def prepare():
    import torch
    assert not(BASE/'PLAN_LOCK.json').exists()
    p=read(REPAIR/'PLAN_LOCK.json');rr=[r for r in p['source']if r['split']=='source_validation'];assert len(rr)==24
    pins={}
    for f in ['scripts/diagnose_stvg_joint_native_path_v1.py','protocols/stvg_joint_native_path_diagnosis_v1.md','vg_tta/stvg_joint_posterior_repair_v1.py','vg_tta/stvg_joint_posterior_v1.py','scripts/run_stvg_joint_posterior_repair_v1.py','methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json']:
        pins[f]=sha(ROOT/f)
    for ds in ['hc2','vidstg']:
        for folder in ['models','statistics']:
            f=REPAIR/folder/(ds+'.pt');load(f);pins[str(f.relative_to(ROOT))]=sha(f)
    for r in rr:
        for f in [OLD/'features/source'/(r['alias']+'.pt'),OLD/'source_supervision'/(r['alias']+'.pt'),REPAIR/'selected_predictions'/(r['alias']+'.pt')]:
            load(f);pins[str(f.relative_to(ROOT))]=sha(f)
    put(BASE/'PLAN_LOCK.json',dict(status='locked_before_frozen_replay',source=rr,source_count=24,source_GT_diagnostic_only=True,new_models=0,new_optimizers=0,new_target_inferences=0,new_Teacher_Student_calls=0,source_validation_not_established_unseen_to_Student=True,protocol_sha256=sha(ROOT/'protocols/stvg_joint_native_path_diagnosis_v1.md'),previous_GPU_reserved_seconds=read(REPAIR/'GPU_ROOT_COMPLETION.json')['cumulative_reserved_seconds'],new_GPU_lease_cap_seconds=300,original_total_GPU_cap_seconds=14400,selection='fixed existing two checkpoints; no source or target reselection'))
    put(BASE/'RUNTIME_LOCK.json',dict(pins=pins))
    x=torch.randn(4,33,4);d=torch.randn_like(x);s=2.;sn=.3
    assert torch.allclose(d+(sn/s)*(x-d),x+(sn-s)*(x-d)/s,atol=5e-7,rtol=1e-6)
    assert torch.equal(x+(sn/s)*(x-x),x)
    put(BASE/'CPU_CONTRACT_COMPLETION.json',dict(status='pass',Euler_formula_equivalence=True,identity_denoiser_preserves_state=True,GPU_initialized=False,new_training=0))
    assert not torch.cuda.is_initialized()

def replay():
    import torch
    from vg_tta.stvg_joint_posterior_repair_v1 import PreconditionedJointPrior
    from scripts.run_decota_paper_main_v1 import gpu
    from main_method.runtime import resource_guard
    p=read(BASE/'PLAN_LOCK.json');pins=read(BASE/'RUNTIME_LOCK.json')['pins']
    for f,h in pins.items():assert sha(ROOT/f)==h,f
    assert not(BASE/'REPLAY_STARTED.json').exists();put(BASE/'REPLAY_STARTED.json',dict(status='started',pid=os.getpid(),new_training=0))
    lease=gpu();started=time.monotonic();files=[]
    try:
        torch.cuda.set_per_process_memory_fraction(.5)
        levels=torch.logspace(math.log10(2.),math.log10(.03),16,device='cuda')
        def budget():
            torch.cuda.synchronize();elapsed=time.monotonic()-started;resource_guard()
            assert elapsed<300 and elapsed+p['previous_GPU_reserved_seconds']<14400
            put(BASE/'GPU_BUDGET.json',dict(status='active',seconds=elapsed,prior_seconds=p['previous_GPU_reserved_seconds']),True)
        def path(net,c,initial):
            x=initial.clone();steps=[]
            with torch.no_grad():
                for j,s in enumerate(levels):
                    x=x.clone();x[:,0,2:]=0;clean=net(x,s.expand(len(x)),c)
                    sn=levels[j+1]if j+1<len(levels)else s*0
                    proposal=clean+(sn/s)*(x-clean)
                    steps.append(dict(level=j,sigma=float(s),input=x.cpu(),clean=clean.cpu(),proposal=proposal.cpu()))
                    x=proposal
            return dict(initial=initial.cpu(),steps=steps,final=x.cpu())
        for ds in ['hc2','vidstg']:
            st=load(REPAIR/'statistics'/(ds+'.pt'));net=PreconditionedJointPrior(st['mean'],st['scale']).cuda().eval().requires_grad_(False);net.load_state_dict(load(REPAIR/'models'/(ds+'.pt'))['state'])
            rr=[r for r in p['source']if r['dataset']==ds]
            for i,r in enumerate(rr):
                budget();z=load(OLD/'features/source'/(r['alias']+'.pt'));y=load(OLD/'source_supervision'/(r['alias']+'.pt'));c=condition(z);cc={k:v.expand((4,)+v.shape[1:])for k,v in c.items()};truth=y['state'][None].cuda().expand(4,-1,-1);native=cc['native']
                gen=torch.Generator(device='cuda').manual_seed(SEED+1000+i);eps=torch.randn(native.shape,generator=gen,device='cuda')
                initial=native+levels[0]*eps;gt_initial=truth+levels[0]*eps;initial[:,0,2:]=0;gt_initial[:,0,2:]=0
                arms=dict(Native_noisy=path(net,cc,initial),GT_noisy=path(net,cc,gt_initial),Native_no_initial_noise=path(net,cc,native))
                old=load(REPAIR/'selected_predictions'/(r['alias']+'.pt'))['samples'];assert (arms['Native_noisy']['final']-old).abs().max()<2e-5
                probes={}
                for level in [15,8,0]:
                    sig=levels[level];pair={}
                    for name,center in [('GT',truth),('Native',native)]:
                        x=center+sig*eps;x=x.clone();x[:,0,2:]=0
                        with torch.no_grad():d=net(x,sig.expand(4),cc)
                        pair[name]=dict(input=x.cpu(),clean=d.cpu())
                    probes[str(level)]=pair
                files.append(save(BASE/'states'/(r['alias']+'.pt'),dict(alias=r['alias'],dataset=ds,epsilon=eps.cpu(),arms=arms,probes=probes,source_only=True,no_observations=True,original_terminal_max_abs_error=float((arms['Native_noisy']['final']-old).abs().max()))))
                put(BASE/'STATUS.json',dict(status='running',stage='frozen_source_replay',done=len(files),total=24,pid=os.getpid()),True)
            del net;torch.cuda.empty_cache()
        put(BASE/'PREDICTION_BARRIER.json',dict(status='complete',count=24,files=files,all_original_paths_reproduced=True,new_training=0,new_target_inference=0))
    finally:
        torch.cuda.synchronize();seconds=time.monotonic()-started
        put(BASE/'GPU_COMPLETION.json',dict(status='complete',new_reserved_seconds=seconds,cumulative_with_previous_repair_seconds=seconds+p['previous_GPU_reserved_seconds'],cap_seconds=14400,new_training=0));lease.close()

def audit_export():
    import torch,numpy as np
    from scripts.run_stvg_joint_posterior_v1 import decode_state
    from vg_tta.tastvg_oracle_event5_v1 import DenseTube
    assert not torch.cuda.is_initialized();assert read(BASE/'PREDICTION_BARRIER.json')['count']==24
    checks=0
    for f,h in read(BASE/'RUNTIME_LOCK.json')['pins'].items():assert sha(ROOT/f)==h,f;checks+=1
    rows=[]
    for r in read(BASE/'PLAN_LOCK.json')['source']:
        z=load(OLD/'features/source'/(r['alias']+'.pt'));y=load(OLD/'source_supervision'/(r['alias']+'.pt'));pr=load(BASE/'states'/(r['alias']+'.pt'));st=load(REPAIR/'statistics'/(r['dataset']+'.pt'));native=condition(z,'cpu')['native'][0];mask=y['mask'];truth=y['state'];sample_rows=[]
        def score(x):
            nonlocal checks
            m=metric(x,z,y);d=decode_state(x,z);ind=DenseTube(d['boxes'],dict(frame_ids=z['frame_ids'],input=z['input']),y['dense_source_GT'],y['span'],clip=r['dataset']=='hc2').score(d['interval'])
            for k in ['v','t','s']:assert abs(m[k]-ind[k])<2e-9;checks+=1
            e=(x-truth).square()*mask;ne=((x-truth)/st['scale']).square()*mask
            m.update(temporal_state_MSE=float(e[0,:2].mean()),spatial_state_MSE=float(e[1:].sum()/mask[1:].sum()),equal_block_state_loss=float((e[0,:2].mean()+e[1:].sum()/mask[1:].sum())/2),normalized_equal_block_state_loss=float((ne[0,:2].mean()+ne[1:].sum()/mask[1:].sum())/2))
            return m
        raw=read(REPAIR/'qualification'/(r['dataset']+'.json'));original=next(v for v in raw['rows']if v['alias']==r['alias'])
        base=score(native);gtbase=score(truth)
        for sample in range(4):
            a=pr['arms']['Native_noisy'];stages=dict(Native_codec=base,GT_codec=gtbase,Native_noisy_initial=score(a['initial'][sample]),first_clean=score(a['steps'][0]['clean'][sample]),first_proposal=score(a['steps'][0]['proposal'][sample]),final=score(a['final'][sample]),GT_noisy_initial=score(pr['arms']['GT_noisy']['initial'][sample]),GT_first_clean=score(pr['arms']['GT_noisy']['steps'][0]['clean'][sample]),GT_final=score(pr['arms']['GT_noisy']['final'][sample]),no_initial_first_clean=score(pr['arms']['Native_no_initial_noise']['steps'][0]['clean'][sample]),no_initial_final=score(pr['arms']['Native_no_initial_noise']['final'][sample]))
            for k in ['v','t','s']:assert abs(stages['final'][k]-original['samples'][sample][k])<2e-6;checks+=1
            decomp={}
            for k in ['v','t','s','equal_block_state_loss']:
                v=stages;parts=dict(initial_noise=v['Native_noisy_initial'][k]-v['Native_codec'][k],first_clean_denoising=v['first_clean'][k]-v['Native_noisy_initial'][k],multistep_vs_first_clean=v['final'][k]-v['first_clean'][k],total=v['final'][k]-v['Native_codec'][k])
                assert abs(parts['initial_noise']+parts['first_clean_denoising']+parts['multistep_vs_first_clean']-parts['total'])<1e-12;checks+=1;decomp[k]=parts
            curves={name:[dict(level=step['level'],sigma=step['sigma'],input=score(step['input'][sample]),clean=score(step['clean'][sample]),proposal=score(step['proposal'][sample]))for step in arm['steps']]for name,arm in pr['arms'].items()}
            probes={l:{name:{io:score(state[sample])for io,state in pair.items()}for name,pair in pp.items()}for l,pp in pr['probes'].items()}
            sample_rows.append(dict(sample=sample,stages=stages,decomposition=decomp,curves=curves,paired_probes=probes))
        rows.append(dict(alias=r['alias'],dataset=r['dataset'],raw_Source=original['Source'],samples=sample_rows))
    means={};step_means={};probe_means={};counts={}
    keys=list(rows[0]['samples'][0]['stages']['Native_codec'])
    for ds in ['hc2','vidstg']:
        rr=[r for r in rows if r['dataset']==ds];means[ds]={};step_means[ds]={};probe_means[ds]={};counts[ds]={}
        for support in ['sample0','all_samples']:
            ss=[s for r in rr for s in (r['samples'][:1]if support=='sample0'else r['samples'])]
            means[ds][support]={stage:{k:float(np.mean([s['stages'][stage][k]for s in ss]))for k in keys}for stage in ss[0]['stages']}
            step_means[ds][support]={arm:[dict(level=i,sigma=ss[0]['curves'][arm][i]['sigma'],**{which:{k:float(np.mean([s['curves'][arm][i][which][k]for s in ss]))for k in keys}for which in ['input','clean','proposal']})for i in range(16)]for arm in ss[0]['curves']}
            probe_means[ds][support]={l:{name:{io:{k:float(np.mean([s['paired_probes'][l][name][io][k]for s in ss]))for k in keys}for io in ['input','clean']}for name in ['GT','Native']}for l in ['15','8','0']}
            counts[ds][support]={stage:dict(positive=sum(s['stages'][stage]['v']-s['stages']['Native_codec']['v']>1e-8 for s in ss),negative=sum(s['stages'][stage]['v']-s['stages']['Native_codec']['v']< -1e-8 for s in ss),neutral=sum(abs(s['stages'][stage]['v']-s['stages']['Native_codec']['v'])<=1e-8 for s in ss))for stage in ['first_clean','final','no_initial_final']}
    result=dict(status='complete',source_only=True,count=24,new_models=0,new_training_steps=0,new_target_inferences=0,no_observations=True,means=means,curves=step_means,paired_probes=probe_means,counts=counts,rows=rows,caveats=['source GT centered controls are diagnostic, not deployable','Native no-initial-noise with sigma2 schedule is an off-kernel counterfactual','telescoping metric changes are ordered accounting, not unique causal percentages','source validation exposed and not proven unseen to pretrained Student','no checkpoint/noise/method selection or production change'])
    put(BASE/'RESULTS.json',result);OUT.mkdir(exist_ok=True,parents=True);put(OUT/'RESULTS.json',result)
    root=dict(status='pass',independent_checks=checks,old_repaired_weights_and_predictions_unchanged=True,original_four_paths_reproduced=True,official_DenseTube_agree=True,metric_decomposition_telescope=True,no_target_training_or_evaluation=True,new_training=0)
    put(BASE/'ROOT_COMPLETION.json',root);put(OUT/'ROOT_COMPLETION.json',root)
    put(OUT/'GPU_COMPLETION.json',read(BASE/'GPU_COMPLETION.json'))
    put(OUT/'PROTOCOL.json',{k:v for k,v in read(BASE/'PLAN_LOCK.json').items()if k!='source'})
    print('ROOT',checks)
    for ds in ['hc2','vidstg']:
        print(ds,json.dumps({s:{k:round(v,6)for k,v in m.items()if k in ['v','t','s','equal_block_state_loss']}for s,m in means[ds]['sample0'].items()}))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','replay','audit_export']);args=a.parse_args();globals()[args.command]()
