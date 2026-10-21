"""Fixed16 decoder Jacobian probe, not a new adaptation method or GT selection."""
import argparse,json,os,shlex,subprocess,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from stvg.runtime import sha,setup,lease
from stvg.tube_runner import write
from tools.layout_trial.worker import checked_plan,records,reference,original,load_feature

PUBLIC=ROOT/'results/tubedetr_observability';OUT=ROOT/'outputs/tubedetr_observability';PLAN=PUBLIC/'plan.json';CACHE=ROOT/'outputs/tubedetr_layout_trial'

def inputs():
    p=json.loads(PLAN.read_text());assert sha(Path(__file__))==p['code_sha256']
    for folder,h in p['parent_final_seals'].items():
        f=ROOT/folder/'final_seal.json';assert sha(f)==h;s=json.loads(f.read_text());assert s['status']=='complete'
        for name,v in s['files'].items():assert sha(ROOT/name)==v,name
    assert sha(ROOT/'outputs/tubedetr_reward_mechanism/network_seal.json')==p['reference_network_seal_sha256']
    old=checked_plan(ROOT/'results/tubedetr_layout_trial/plan.json');return p,old,{r['id']:r for r in records(old)}

def worker():
    p,old,rows=inputs();setup();lock=lease()
    from stvg.tubedetr import load_source,native_forward,TubeReplay
    from stvg.tube_decode import decode
    from stvg.inputs import validate_input
    from stvg.tensors import state_hash
    from tools.layout_trial.reward import LayoutReward
    from stvg import policy
    model=load_source(old['configuration']);before=state_hash(model.state_dict());root,ref=reference(old)
    fs=json.loads((CACHE/'features_seal.json').read_text());tau=json.loads((CACHE/'temperature_calibration.json').read_text())['layout3']['tau']
    ns=json.loads((ROOT/'outputs/tubedetr_reward_mechanism/network_seal.json').read_text());entries={}
    for index,q in enumerate(p['query_ids']):
        row=rows[q];value,_=original(row,root,ref);frames,fids=decode(row['metadata'],'vidstg');validate_input(frames,fids,row['metadata'])
        native=native_forward(model,frames,fids,row['metadata']['caption'],224);assert np.array_equal(native['source'].numpy(),value['Source'])
        f=load_feature(row,CACHE,fs);pos=value['positions8'].astype(int);assert np.array_equal(np.array(fids)[pos],f['observed_frame_ids'])
        reward=LayoutReward(torch.from_numpy(f['features']).cuda(),torch.from_numpy(f['boxes']).cuda().double(),3,torch.from_numpy(f['valid_ratio']).cuda())
        rp=TubeReplay(model,native,value['E_Ref'],pos);v=rp.values();b=v['boxes'][pos];mu=policy.chart(b);D=policy.diagonal(b);u=(mu/D).reshape(-1)
        J=torch.stack([torch.autograd.grad(u[j],rp.state,retain_graph=True)[0].reshape(-1)for j in range(32)]).double()
        _,singular,V=torch.linalg.svd(J,full_matrices=False);rank=int((singular>singular[0]*p['relative_rank_tolerance']).sum());basis=V[:rank]
        gd=torch.autograd.grad(reward.mean_loss(b,None).mean(),rp.state,retain_graph=True)[0].detach()
        rr=reward.feedback(mu.detach(),D,None,row['key'],41,0,old['configuration']['scales'],tau)
        go=torch.autograd.grad(policy.mixture_loss(mu,rr['actions'],D,rr['weights'],old['configuration']['scales']),rp.state)[0].detach()
        e=ns['files'][q];refpath=ROOT/'outputs/tubedetr_reward_mechanism'/e['path'];assert sha(refpath)==e['sha256']
        with np.load(refpath)as z:prior_steps={a:z[a+'_after_one_step'].copy()for a in ['Direct','OPD']}
        arrays=dict(frame_ids=np.array(fids),positions8=pos,Strong_interval=value['Strong_interval'],E_Ref=value['E_Ref'],rank=np.array(rank),singular_values=singular.cpu().numpy())
        for arm,g in [('Direct',gd),('OPD',go)]:
            assert torch.isfinite(g).all()
            opt=torch.optim.Adam([rp.state],lr=old['configuration']['lr'],betas=(.9,.999),eps=1e-8);opt.zero_grad(set_to_none=True);rp.state.grad=g.clone();opt.step();full=rp.state.detach().clone()
            with torch.no_grad():rp.state.zero_()
            rp.state.grad=None;flat=full.flatten().double();parallel=(basis.T@(basis@flat));null=flat-parallel
            arrays[arm+'_null_energy_fraction']=np.array(float(null.square().sum()/flat.square().sum().clamp_min(1e-30)))
            arrays[arm+'_linear_observation_null_ratio']=np.array(float((J@null).norm()/(J@flat).norm().clamp_min(1e-20)))
            for part,d in [('full',flat),('parallel',parallel),('null',null)]:
                with torch.no_grad():rp.state.copy_(d.reshape_as(rp.state).to(rp.state));after=rp.values()['boxes'];arrays[arm+'_'+part]=after.cpu().numpy();rp.state.zero_()
            assert np.array_equal(arrays[arm+'_full'],prior_steps[arm]),'One-step reference differs'
        assert not rp.state.count_nonzero() and all(not x.requires_grad and x.grad is None for x in model.parameters())
        path=OUT/'responses'/f'{q}.npz';path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();np.savez(path,**arrays);entries[q]=dict(path=str(path.relative_to(OUT)),sha256=sha(path))
        del native,frames,reward,rp,v,b,mu,D,u,J,singular,V,basis,gd,go,rr,g,full,flat,parallel,null,after,opt
        write(OUT/'worker_progress.json',dict(complete=index+1,total=16,labels_read=False));print('Observability probe',index+1,flush=True)
    assert state_hash(model.state_dict())==before
    write(OUT/'sealed.json',dict(status='complete',queries=16,files=entries,plan_sha256=sha(PLAN),labels_read=False,network_weights_unchanged=True,full_steps_bitexact=True));lock.close()

def score():
    p,old,rows=inputs();seal=json.loads((OUT/'sealed.json').read_text());assert seal['status']=='complete' and seal['queries']==16 and seal['plan_sha256']==sha(PLAN)
    for e in seal['files'].values():assert sha(OUT/e['path'])==e['sha256']
    lp=ROOT/old['labels'];assert sha(lp)==old['labels_sha256'];labels={v['id']:v for v in map(json.loads,lp.read_text().splitlines())}
    from stvg.metrics import DenseTube
    details=[]
    for q in p['query_ids']:
        row=rows[q];t=labels[q];gt={int(k):v for k,v in t['boxes'].items()};record=dict(id=q)
        with np.load(OUT/seal['files'][q]['path'])as z:
            meta=dict(frame_ids=z['frame_ids'].tolist(),input=row['metadata']);base=DenseTube(z['E_Ref'],meta,gt,t['interval']).score(z['Strong_interval']);observed=np.zeros(len(z['frame_ids']),bool);observed[z['positions8'].astype(int)]=True;record['rank']=int(z['rank'])
            for arm in ['Direct','OPD']:
                record[arm]=dict(null_energy_fraction=float(z[arm+'_null_energy_fraction']),linear_observation_null_ratio=float(z[arm+'_linear_observation_null_ratio']),responses={})
                for part in ['full','parallel','null']:
                    pred=z[arm+'_'+part];m=DenseTube(pred,meta,gt,t['interval']).score(z['Strong_interval']);move=np.linalg.norm(pred[:,:2]-z['E_Ref'][:,:2],axis=-1)
                    record[arm]['responses'][part]=dict(m_vIoU_change_pp=100*(m['v']-base['v']),m_sIoU_change_pp=100*(m['s']-base['s']),observed_center_displacement=float(move[observed].mean()),unobserved_center_displacement=float(move[~observed].mean()))
        details.append(record)
    result=dict(status='complete_mechanistic_probe',queries=16,mean_observation_Jacobian_rank=float(np.mean([r['rank']for r in details])),arms={},
        no_GT_component_selection=True,one_step_not_multistep_evidence=True,not_new_method_performance=True,GT_only_after_complete_response_seal=True)
    for arm in ['Direct','OPD']:
        result['arms'][arm]={k:float(np.mean([r[arm][k]for r in details]))for k in ['null_energy_fraction','linear_observation_null_ratio']}
        result['arms'][arm]['responses']={part:{k:float(np.mean([r[arm]['responses'][part][k]for r in details]))for k in details[0][arm]['responses'][part]}for part in ['full','parallel','null']}
    write(OUT/'details_private.json',details);write(PUBLIC/'summary.json',result)
    (PUBLIC/'report.md').write_text('# TubeDETR 观测Jacobian与Adam更新分解探针\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n\n固定16条、仅初始单步局部线性分解；不按GT选择分量，不证明投影后多步收益或OPD优势。\n')
    write(PUBLIC/'final_seal.json',dict(status='complete',files={str(f.relative_to(ROOT)):sha(f)for f in [PLAN,PUBLIC/'summary.json',PUBLIC/'report.md']},response_seal_sha256=sha(OUT/'sealed.json'),publication_pending=True))

def controller():
    inputs()
    for stage in ['worker','score']:
        command=[sys.executable,'-B','-u','tools/tubedetr_observability_probe.py','--stage',stage];log=ROOT/'logs'/f'tubedetr_observability_{stage}.log';env=dict(os.environ)
        if stage=='score':env['CUDA_VISIBLE_DEVICES']=''
        write(OUT/'status.json',dict(status=stage,no_parameter_search=True))
        with log.open('a')as f:
            process=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env);path=ROOT/'runtime/tubedetr_active_jobs.json';r=json.loads(path.read_text());r['jobs']['observability_worker']=dict(pid=process.pid,command=shlex.join(command),log=str(log.relative_to(ROOT)),stage=stage);write(path,r)
            if process.wait():raise RuntimeError('Observability stage failed: '+stage)
    write(OUT/'status.json',dict(status='complete',queries=16,old_queues_still_deferred=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['worker','score']);a=p.parse_args()
    try:globals()[a.stage]()if a.stage else controller()
    except Exception as e:
        v=dict(status='failed',error=type(e).__name__,message=str(e));write(OUT/'failure.json',v);write(OUT/'status.json',v);raise
