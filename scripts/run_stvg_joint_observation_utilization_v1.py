"""Bounded frozen source-only observation utilization. No train/target route."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_posterior_repair_v1 import BASE as REPAIR,OLD,read,put,load,save,sha,condition
PARENT=ROOT/'artifacts/stvg_joint_sigma213_source24_v1'
BASE=ROOT/'artifacts/stvg_joint_observation_utilization_v1'
OUT=ROOT/'results/stvg_joint_posterior/2026-10-18/observation_utilization'
ARMS=['none','DINO_all','DINO_common','GT_common','GT_planned']

def prepare():
    import torch,numpy as np
    assert not(BASE/'PLAN_LOCK.json').exists();rr=read(PARENT/'PLAN_LOCK.json')['source'];pins={}
    for f in ['scripts/run_stvg_joint_observation_utilization_v1.py','vg_tta/stvg_joint_observation_utilization_v1.py','vg_tta/stvg_joint_posterior_repair_v1.py','vg_tta/stvg_joint_posterior_v1.py','protocols/stvg_joint_observation_utilization_v1.md','methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json']:
        pins[f]=sha(ROOT/f)
    for ds in ['hc2','vidstg']:
        for f in [REPAIR/'models'/(ds+'.pt'),REPAIR/'statistics'/(ds+'.pt'),OLD/'models'/(ds+'_association.pt')]:load(f);pins[str(f.relative_to(ROOT))]=sha(f)
    for r in rr:
        for f in [OLD/'features/source'/(r['alias']+'.pt'),OLD/'source_supervision'/(r['alias']+'.pt'),PARENT/'predictions'/(r['alias']+'.pt')]:load(f);pins[str(f.relative_to(ROOT))]=sha(f)
    put(BASE/'PLAN_LOCK.json',dict(status='locked_before_inference',source=rr,count=24,clocks=['fixed','free'],conditions=ARMS,source_only=True,source_oracle_sparse_GT_exposed=True,new_models=0,new_training=0,new_Student_Teacher_calls=0,new_target_inference=0,learned_association_head=False,inlier_prior=.5,outlier='original density1',observation_step=.02,component_cap=.2,sigma_start=.213,sigma_end=.03,levels=16,samples=4,checkpoints={'hc2':30,'vidstg':20},primary_propagation_support='GT annotated Native decoder frames inside fixed Native interval excluding all4 planned anchors',fixed_native_membership='integer physical Native interval, not floating decoded interval',GT_common='same positions as DINO_common where precise source GT exists, no quality filter',GT_planned='precise source GT at original planned4 positions; no exterior extrapolation',previous_GPU_reserved_seconds=read(PARENT/'GPU_COMPLETION.json')['cumulative_reserved_seconds'],new_GPU_limit_seconds=300,cumulative_GPU_limit_seconds=14400,no_automatic_followup=True,source_validation_not_established_unseen_to_pretrained_Student=True))
    put(BASE/'RUNTIME_LOCK.json',dict(pins=pins));counts=[];receipts=[]
    for r in rr:
        z=load(OLD/'features/source'/(r['alias']+'.pt'));y=load(OLD/'source_supervision'/(r['alias']+'.pt'));ids=z['frame_ids'];lo,hi=ids[0],ids[-1]+1;iv=z['native_interval'];gt=y['dense_source_GT'];aa=z['expert']['anchors']['single4'];positions=sorted(set(z['expert']['positions4']));assert len(positions)==4
        assert all(a['position']in positions for a in aa)
        common=[a for a in aa if ids[a['position']]in gt]
        def GTbox(pos):
            x,y,xx,yy=gt[ids[pos]];w,h=z['input']['width'],z['input']['height'];return [(x+xx)/(2*w),(y+yy)/(2*h),(xx-x)/w,(yy-y)/h]
        def make(rows):
            pp=[int(a['position'])for a in rows];b=torch.tensor([a['box']for a in rows],dtype=torch.float32).reshape(-1,4);assert torch.isfinite(b).all()and bool((b[:,2:]>0).all())
            return dict(time=torch.tensor([(ids[p]-lo)/(hi-lo)for p in pp],dtype=torch.float32),boxes=b,fixed_inside=torch.tensor([iv[0]<=ids[p]<iv[1]for p in pp],dtype=torch.bool),positions=pp)
        obs=dict(none=make([]),DINO_all=make(aa),DINO_common=make(common),GT_common=make([dict(position=a['position'],box=GTbox(a['position']))for a in common]),GT_planned=make([dict(position=p,box=GTbox(p))for p in positions if ids[p]in gt]))
        assert obs['DINO_common']['positions']==obs['GT_common']['positions']
        for o in obs.values():assert bool(o['fixed_inside'].all())
        receipts.append(save(BASE/'observations'/(r['alias']+'.pt'),dict(alias=r['alias'],dataset=r['dataset'],observations=obs,planned_positions=positions,source_GT_role='sparse boxes/availability only in oracle/matched diagnosis; no full GT model input')))
        counts.append(dict(alias=r['alias'],dataset=r['dataset'],planned=4,DINO_all=len(aa),DINO_common=len(common),GT_common=len(common),GT_planned=len(obs['GT_planned']['time']),DINO_unknown_GT=len(aa)-len(common),no_DINO=len(aa)==0,no_common_GT=len(common)==0,no_planned_GT=len(obs['GT_planned']['time'])==0))
    put(BASE/'OBSERVATION_INPUT_BARRIER.json',dict(status='complete',count=24,files=receipts,counts=counts));put(BASE/'STATUS.json',dict(status='prepared',stage='CPU_contract_pending',done=0,total=48))

def contract():
    import torch
    from vg_tta.stvg_joint_posterior_repair_v1 import PreconditionedJointPrior
    from vg_tta.stvg_joint_observation_utilization_v1 import project_time,observation_terms,guided_rollout
    from vg_tta.stvg_joint_posterior_v1 import observation_loglik
    from vg_tta.stvg_joint_sigma_start_v1 import frozen_rollout
    torch.set_num_threads(4);torch.manual_seed(193);net=PreconditionedJointPrior(torch.zeros(33,4),torch.ones(33,4)*.7).double().eval().requires_grad_(False)
    c=dict(maps=torch.randn(1,512,32,8,8,dtype=torch.float64),query=torch.randn(1,5,256,dtype=torch.float64),qmask=torch.zeros(1,5,dtype=torch.bool),context=torch.randn(1,32,512,dtype=torch.float64),unit=torch.tensor([.002],dtype=torch.float64),native=torch.randn(1,33,4,dtype=torch.float64)*.1);c['native'][:,0,2:]=0
    obs=dict(time=torch.tensor([.6,.7],dtype=torch.float64),boxes=torch.tensor([[.55,.45,.52,.48],[.5,.5,.5,.5]],dtype=torch.float64),fixed_inside=torch.ones(2,dtype=torch.bool));sigma=torch.tensor([.15],dtype=torch.float64);fds=[]
    for fixed in [False,True]:
        x=c['native'].clone().requires_grad_(True);clean=project_time(net(project_time(x,c,fixed),sigma,c),c,fixed);ll,st=observation_terms(clean,c,obs,.01,fixed);g,=torch.autograd.grad(ll.sum(),x);assert g.norm()>0
        if fixed:assert float(g[:,0].abs().max())==0
        else:assert torch.allclose(ll,observation_loglik(clean,c,obs,torch.full((2,),.5,dtype=torch.float64),.01),atol=1e-12,rtol=0)
        for idx in [(0,0,0),(0,0,1),(0,20,0),(0,25,2)]:
            a=c['native'].clone();b=a.clone();a[idx]+=1e-5;b[idx]-=1e-5
            with torch.no_grad():
                aa=project_time(net(project_time(a,c,fixed),sigma,c),c,fixed);bb=project_time(net(project_time(b,c,fixed),sigma,c),c,fixed);f=float((observation_terms(aa,c,obs,.01,fixed)[0]-observation_terms(bb,c,obs,.01,fixed)[0])/(2e-5))
            v=float(g[idx]);assert abs(f-v)<2e-5+.002*abs(v),(fixed,idx,f,v);fds.append(dict(fixed=fixed,coordinate=list(idx),autograd=v,finite_difference=f))
    gen=torch.Generator().manual_seed(27);eps=torch.randn(c['native'].shape,generator=gen);empty=dict(time=torch.empty(0),boxes=torch.empty(0,4),fixed_inside=torch.empty(0,dtype=torch.bool))
    a=guided_rollout(net,c,eps,empty,.01,False);b=frozen_rollout(net,c,27)
    assert torch.equal(a['final'],b['final'])
    fixed=guided_rollout(net,c,eps,empty,.01,True);assert torch.equal(fixed['final'][:,0],c['native'][:,0])
    put(BASE/'CPU_CONTRACT_COMPLETION.json',dict(status='pass',finite_differences=fds,old_free_mixture_exact=True,fixed_clock_zero_temporal_gradient=True,empty_free_guidance_exact_baseline=True,new_training=0,GPU_initialized=False));assert not torch.cuda.is_initialized()

def infer():
    import torch
    from vg_tta.stvg_joint_posterior_repair_v1 import PreconditionedJointPrior
    from vg_tta.stvg_joint_observation_utilization_v1 import guided_rollout
    from scripts.run_decota_paper_main_v1 import gpu
    from main_method.runtime import resource_guard
    p=read(BASE/'PLAN_LOCK.json');assert read(BASE/'CPU_CONTRACT_COMPLETION.json')['status']=='pass'
    for f,h in read(BASE/'RUNTIME_LOCK.json')['pins'].items():assert sha(ROOT/f)==h,f
    assert not(BASE/'INFERENCE_STARTED.json').exists();put(BASE/'INFERENCE_STARTED.json',dict(status='started',pid=os.getpid(),new_training=0));lease=gpu();beg=time.monotonic();nexecuted=0;nreused=0;qual=[]
    try:
        torch.cuda.set_per_process_memory_fraction(.5)
        def budget():
            torch.cuda.synchronize();s=time.monotonic()-beg;resource_guard();assert s<300 and s+p['previous_GPU_reserved_seconds']<14400
        models={};variances={}
        for ds in ['hc2','vidstg']:
            st=load(REPAIR/'statistics'/(ds+'.pt'));net=PreconditionedJointPrior(st['mean'],st['scale']).cuda().eval().requires_grad_(False);net.load_state_dict(load(REPAIR/'models'/(ds+'.pt'))['state']);models[ds]=net;variances[ds]=load(OLD/'models'/(ds+'_association.pt'))['variance']
        for clock in ['fixed','free']:
            if clock=='free':assert read(BASE/'FIXED_PREDICTION_BARRIER.json')['count']==24
            files=[];first=set()
            for n,r in enumerate(p['source']):
                budget();ds=r['dataset'];z=load(OLD/'features/source'/(r['alias']+'.pt'));data=load(BASE/'observations'/(r['alias']+'.pt'));obs=data['observations'];old=load(PARENT/'predictions'/(r['alias']+'.pt'))['path'];eps=old['epsilon'].cuda();c=condition(z);cc={k:v.expand((4,)+v.shape[1:])for k,v in c.items()};net=models[ds];arms={};reuse={};cache={}
                if clock=='free':
                    arms['none']=dict(initial=old['initial'],final=old['final'],trace=[],fixed_clock=False,reference_count=0,exact_old_sigma213_reuse=True);cache['empty']='none';nreused+=1
                    if ds not in first:
                        probe=guided_rollout(net,cc,eps,{k:v.cuda()for k,v in obs['none'].items()if k!='positions'},variances[ds],False);err=float((probe['final']-old['final']).abs().max());assert err<2e-5;qual.append(dict(dataset=ds,free_none_max_error=err));first.add(ds)
                for name in ARMS:
                    if name in arms:continue
                    o=obs[name];key='empty'if not len(o['time'])else hashlib.sha256(o['time'].numpy().tobytes()+o['boxes'].numpy().tobytes()).hexdigest()
                    if key in cache:reuse[name]=cache[key];arms[name]=arms[cache[key]];nreused+=1;continue
                    oo={k:v.cuda()for k,v in o.items()if k!='positions'};arms[name]=guided_rollout(net,cc,eps,oo,variances[ds],clock=='fixed');cache[key]=name;nexecuted+=1
                files.append(save(BASE/'predictions'/clock/(r['alias']+'.pt'),dict(alias=r['alias'],dataset=ds,clock=clock,arms=arms,reused_from=reuse,variance=variances[ds],source_sparse_GT_only_in_oracle=True,full_GT_tube_input=False,learned_association_executed=False,new_training=0)))
                put(BASE/'STATUS.json',dict(status='running',stage=clock+'_observation_inference',done=n+1,total=24,pid=os.getpid()),True)
            put(BASE/(clock.upper()+'_PREDICTION_BARRIER.json'),dict(status='complete',count=24,files=files))
        put(BASE/'GPU_QUALIFICATION_COMPLETION.json',dict(status='pass',count=2,rows=qual,original_free_none_reproduced=True,model_parameter_gradients=False))
        put(BASE/'PREDICTION_BARRIER.json',dict(status='complete',count=24,clocks=2,conditions_per_clock=5,logical_predictions=240,paths_per_prediction=4,new_rollouts_executed=nexecuted,exact_logical_reuses=nreused,source_GT_sparse_oracle=True,full_GT_metrics_after_seal=True,new_training=0,new_target_inference=0))
    finally:
        torch.cuda.synchronize();s=time.monotonic()-beg;put(BASE/'GPU_COMPLETION.json',dict(status='complete',new_reserved_seconds=s,cumulative_reserved_seconds=s+p['previous_GPU_reserved_seconds'],limit_seconds=14400,new_training=0,learned_association_head_executed=False));lease.close()

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','contract','infer']);args=a.parse_args();globals()[args.command]()
