"""Finite TubeDETR implementation of interface-based trajectory search.

Same16 input-hash queries, frozen time, real LayerNorm-parameter candidates.
Search, reranking and multi-start Direct share the same differentiable evidence.
"""
import argparse,hashlib,json,os,shlex,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from stvg.runtime import sha,setup,lease
from stvg.tube_runner import write,digest
from tools.layout_trial.worker import checked_plan,records,reference,original
from tools.universal_trial.core import VideoEvidence,chart,evidence_subspace,initial_candidates,component_update
PUBLIC=ROOT/'results/tubedetr_universal_trial';OUT=ROOT/'outputs/tubedetr_universal_trial';PLAN=PUBLIC/'plan.json'

def inputs():
    p=json.loads(PLAN.read_text())
    for f,h in p['code_sha256'].items():assert sha(ROOT/f)==h,f
    for folder,h in p['parent_final_seals'].items():
        path=ROOT/folder/'final_seal.json';assert sha(path)==h;s=json.loads(path.read_text());assert s['status']=='complete'
        for name,v in s['files'].items():assert sha(ROOT/name)==v,name
    assert sha(ROOT/'outputs/tubedetr_mask_reward_qualification/mask_seal.json')==p['teacher_mask_seal_sha256']
    old=checked_plan(ROOT/'results/tubedetr_layout_trial/plan.json');return p,old,{r['id']:r for r in records(old)}

def worker():
    p,old,rows=inputs();setup();lock=lease()
    from stvg.tubedetr import load_source
    from stvg.tube_decode import decode
    from stvg.inputs import validate_input
    from stvg.tensors import state_hash
    from tools.universal_trial.tubedetr_adapter import encode,ParameterInterface,teacher_seed_mass
    model=load_source(old['configuration']);before=state_hash(model.state_dict());root,ref=reference(old);ms=json.loads((ROOT/'outputs/tubedetr_mask_reward_qualification/mask_seal.json').read_text());entries={}
    for index,q in enumerate(p['query_ids']):
        row=rows[q];start=time.perf_counter();torch.cuda.reset_peak_memory_stats();v,_=original(row,root,ref);frames,fids=decode(row['metadata'],'vidstg');validate_input(frames,fids,row['metadata'])
        native,features,cells,feature_info=encode(model,frames,fids,row['metadata']['caption']);assert np.array_equal(native['source'].numpy(),v['Source'])
        me=ms['files'][q];mp=ROOT/'outputs/tubedetr_mask_reward_qualification'/me['path'];assert sha(mp)==me['sha256']
        with np.load(mp)as z:mask={k:z[k].copy()for k in z.files}
        positions=v['positions8'].astype(int).tolist();assert np.array_equal(np.array(fids)[positions],mask['observed_frame_ids'])
        interface=ParameterInterface(model,native);zero=interface.zero().requires_grad_(True);base=interface.execute(zero);assert torch.equal(base.detach().cpu(),native['source'])
        teacher=torch.from_numpy(mask['boxes'][:,0]).to(base);D=torch.cat([teacher[:,2:],torch.ones_like(teacher[:,2:])],-1).clamp_min(1e-5);grads=[]
        for a,pos in enumerate(positions):
            # Geometry is disclosed only as the common evidence-gradient basis
            # generator; no coordinate/GIoU term enters trajectory feedback.
            loss=.5*((chart(base[pos])-chart(teacher[a]))/D[a]).square().sum();g=torch.autograd.grad(loss,zero,retain_graph=a<7)[0];assert torch.isfinite(g).all();grads.append(g.detach())
        U,gram=evidence_subspace(grads,p['parameter_RMS'],p['parameter_max_abs']);source=base.detach();del base,zero,grads
        masks=torch.from_numpy(mask['masks'][:,:1]).to(features);ratio=torch.from_numpy(mask['valid_ratio']).to(features);seed=teacher_seed_mass(masks,ratio,cells)
        fps=float(row['metadata']['fps']);assert fps>0;times=torch.tensor(fids,device=source.device,dtype=torch.float64)/fps;window=torch.tensor(v['Strong_interval'],device=source.device,dtype=torch.float64)/fps
        evidence=VideoEvidence(features.detach(),cells,times,positions,seed,source,window,p['feature_temperature'])
        def reward(boxes):return evidence.score(boxes,p['association_weight'],p['change_weight'],p['hypothesis_temperature'])
        coeff=initial_candidates(8,U.device,U.dtype);boxes=[source.cpu().numpy()];scores=[float(reward(source)['total'])];components=[]
        for c in coeff[1:]:
            with torch.no_grad():b=interface.execute(U@c);r=reward(b);boxes.append(b.cpu().numpy());scores.append(float(r['total']))
        initial=torch.tensor(scores,device=U.device,dtype=U.dtype);centers,weights=component_update(coeff,initial,p['policy_temperature']);postboxes=[];postscores=[]
        for c in centers:
            with torch.no_grad():b=interface.execute(U@c);postboxes.append(b.cpu().numpy());postscores.append(float(reward(b)['total']))
        directboxes=[];directscores=[];directcoeff=[];gradnorm=[]
        for a in range(8):
            u=coeff[1+a].detach().clone().requires_grad_(True);opt=torch.optim.Adam([u],lr=p['Direct_coefficient_lr'],betas=(.9,.999),eps=1e-8)
            for k in range(p['Direct_steps']):
                b=interface.execute(U@u);loss=-reward(b)['total'];g=torch.autograd.grad(loss,u)[0];assert torch.isfinite(g).all();gradnorm.append(float(g.norm()));opt.zero_grad(set_to_none=True);u.grad=g.detach();opt.step()
                with torch.no_grad():u.div_(u.abs().sum().clamp_min(1.))
            with torch.no_grad():b=interface.execute(U@u);directboxes.append(b.cpu().numpy());directscores.append(float(reward(b)['total']));directcoeff.append(u.detach().cpu().numpy())
            del b,loss,g,u,opt
        path=OUT/'predictions'/f'{q}.npz';path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
        np.savez(path,Source=source.cpu().numpy(),frame_ids=np.array(fids),positions8=np.array(positions),Strong_interval=v['Strong_interval'],initial_coefficients=coeff.cpu().numpy(),initial_boxes=np.stack(boxes),initial_rewards=np.array(scores),policy_centers=centers.cpu().numpy(),policy_component_weights=weights.cpu().numpy(),policy_boxes=np.stack(postboxes),policy_rewards=np.array(postscores),Direct_coefficients=np.stack(directcoeff),Direct_boxes=np.stack(directboxes),Direct_rewards=np.array(directscores),basis_Gram=gram.cpu().numpy(),basis_U=U.cpu().numpy(),confidence=evidence.confidence.cpu().numpy(),Direct_gradient_norm=np.array(gradnorm))
        receipt=dict(parameter_count=interface.count,parameter_names=interface.names,feature_interface=feature_info,frame_time_unit='seconds',timestamp_fps=fps,ordinary_feature_sha256=hashlib.sha256(features.cpu().numpy().tobytes()).hexdigest(),teacher_cache_sha256=me['sha256'],Source_bitexact=True,frozen_time=True,new_teacher_forwards=0,no_query_residual=True,seconds=time.perf_counter()-start,peak_GPU_bytes=torch.cuda.max_memory_allocated(),initial_candidate_forwards=20,policy_center_forwards=8,Direct_forwards=8*(p['Direct_steps']+1),common_basis_backwards=8,Direct_backwards=8*p['Direct_steps'],GT_read=False)
        rp=path.with_suffix('.json');write(rp,receipt);entries[q]=dict(path=str(path.relative_to(OUT)),sha256=sha(path),receipt_path=str(rp.relative_to(OUT)),receipt_sha256=sha(rp));assert state_hash(model.state_dict())==before
        assert all(not x.requires_grad and x.grad is None for x in model.parameters())
        del native,features,cells,interface,U,gram,source,masks,ratio,seed,evidence,coeff,centers,weights,frames,teacher,D,initial,c,r
        write(OUT/'worker_progress.json',dict(complete=index+1,total=16,labels_read=False));print('Universal TubeDETR trial',index+1,flush=True)
    write(OUT/'sealed.json',dict(status='complete',queries=16,files=entries,plan_sha256=sha(PLAN),labels_read=False,model_weights_unchanged=True,new_teacher_forwards=0));lock.close()

def score():
    p,old,rows=inputs();s=json.loads((OUT/'sealed.json').read_text());assert s['status']=='complete'and s['queries']==16 and s['plan_sha256']==sha(PLAN)
    for e in s['files'].values():assert sha(OUT/e['path'])==e['sha256']and sha(OUT/e['receipt_path'])==e['receipt_sha256']
    lp=ROOT/old['labels'];assert sha(lp)==old['labels_sha256'];labels={v['id']:v for v in map(json.loads,lp.read_text().splitlines())}
    from stvg.metrics import DenseTube,box_iou
    from stvg.metrics_base import xyxy
    from tools.layout_trial.diagnostics import paired_ranking,mean
    from tools.tubedetr_opd_direct_diagnosis import cluster_difference
    details=[];receipts=[]
    for q in p['query_ids']:
        t=labels[q];row=rows[q];e=s['files'][q];receipts.append(json.loads((OUT/e['receipt_path']).read_text()))
        with np.load(OUT/e['path'])as z:
            meta=dict(frame_ids=z['frame_ids'].tolist(),input=row['metadata']);gt={int(k):v for k,v in t['boxes'].items()};span=z['Strong_interval'];bank=z['initial_boxes'];rewards=z['initial_rewards'];window=(z['frame_ids']>=span[0])&(z['frame_ids']<span[1]);pairs=[]
            def metric(b):return DenseTube(b,meta,gt,t['interval']).score(span)
            allm=[metric(b)for b in bank];selection=int(np.argmax(rewards));pi=int(np.argmax(z['policy_rewards']));di=int(np.argmax(z['Direct_rewards']))
            values=dict(SourceStrong=metric(z['Source']),CandidateRerank=allm[selection],PolicyCenter=metric(z['policy_boxes'][pi]),MultiStartDirect=metric(z['Direct_boxes'][di]))
            for i in range(len(bank)):
                for j in range(i):pairs.append(float(box_iou(xyxy(bank[i],1,1),xyxy(bank[j],1,1))[window].mean()))
            gram=z['basis_Gram'];eigen=np.linalg.eigvalsh(gram);rank=int((eigen>max(eigen.max(),1e-20)*1e-5).sum())
            modes=[]
            for b in bank:
                if all(float(box_iou(xyxy(b,1,1),xyxy(c,1,1))[window].mean())<p['trajectory_mode_IoU_threshold']for c in modes):modes.append(b)
            ranks=paired_ranking(rewards,np.array([m['v']for m in allm]));details.append(dict(id=q,parent=t['parent'],values=values,candidate_oracle_vIoU=max(m['v']for m in allm),reward_rank=ranks,trajectory_clusters=len(modes),mean_pairwise_trajectory_IoU=float(np.mean(pairs)),subspace_rank=rank,mean_confidence=float(z['confidence'].mean()),unobserved_confidence=float(z['confidence'][:,[i for i in range(len(z['frame_ids']))if i not in z['positions8']]].mean()),Direct_gradient_norm=float(z['Direct_gradient_norm'].mean())))
    names=list(details[0]['values']);result=dict(status='complete_interface_and_finite_search_trial',queries=16,metrics={n:{k:100*float(np.mean([r['values'][n][k]for r in details]))for k in ['v','s','t']}for n in names},mean_subspace_rank=float(np.mean([r['subspace_rank']for r in details])),mean_trajectory_clusters=float(np.mean([r['trajectory_clusters']for r in details])),mean_pairwise_trajectory_IoU=float(np.mean([r['mean_pairwise_trajectory_IoU']for r in details])),mean_evidence_confidence=float(np.mean([r['mean_confidence']for r in details])),mean_unobserved_confidence=float(np.mean([r['unobserved_confidence']for r in details])),candidate_reward_pair_accuracy=mean([r['reward_rank']['pair_accuracy']for r in details]),candidate_reward_spearman=mean([r['reward_rank']['spearman']for r in details]),candidate_oracle_m_vIoU=100*float(np.mean([r['candidate_oracle_vIoU']for r in details])),parameter_count=receipts[0]['parameter_count'],feature_interfaces=[r['feature_interface']for r in receipts],frozen_Strong_time=True,no_new_teacher=True,actual_parameter_executions=True,no_query_residual_or_ASA=True,not_full_population_or_independent_confirmation=True,Direct_not_tuned=True,policy_is_reward_weighted_parameter_search_not_claimed_new_distillation=True,differences={})
    for a,b in [('PolicyCenter','CandidateRerank'),('PolicyCenter','MultiStartDirect'),('MultiStartDirect','SourceStrong')]:result['differences'][a+'-'+b]=cluster_difference([r['values'][a]['v']-r['values'][b]['v']for r in details],[r['parent']for r in details])
    result['execution_cost']=dict(common_basis_backwards=8,initial_candidate_forwards=20,policy_center_forwards=8,Direct_forwards=8*(p['Direct_steps']+1),Direct_backwards=8*p['Direct_steps'],not_equal_compute=True,mean_seconds=float(np.mean([r['seconds']for r in receipts])),peak_GPU_bytes=max(r['peak_GPU_bytes']for r in receipts))
    write(OUT/'details_private.json',details);write(PUBLIC/'summary.json',result);(PUBLIC/'report.md').write_text('# 通用轨迹接口：TubeDETR初试\n\n固定原输入哈希16条，时间固定Strong。SourceStrong是原生Source空间+固定Strong时间，未使用旧E_Ref输出修正。所有候选真实执行decoder LayerNorm affine；普通学生视觉特征传播、奖励、梯度子空间和分量更新独立于模型query结构。八个证据分量不是已知不同身份。Direct同奖励同U和约束，8起点2步，仅初试未公平调参，计算不等额。完整预测封存后独立CPU评分，不写主表，不自动全量。\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    write(PUBLIC/'final_seal.json',dict(status='complete',files={str(f.relative_to(ROOT)):sha(f)for f in [PLAN,PUBLIC/'summary.json',PUBLIC/'report.md']},prediction_seal_sha256=sha(OUT/'sealed.json'),publication_pending=True))

def controller():
    inputs()
    for stage in ['worker','score']:
        command=[sys.executable,'-B','-u','tools/tubedetr_universal_trial.py','--stage',stage];log=ROOT/'logs'/f'tubedetr_universal_trial_{stage}.log';env=dict(os.environ)
        if stage=='score':env['CUDA_VISIBLE_DEVICES']=''
        write(OUT/'status.json',dict(status=stage,queries=16,no_full_evaluation=True))
        with log.open('a')as f:
            process=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env);path=ROOT/'runtime/tubedetr_active_jobs.json';r=json.loads(path.read_text());r['jobs']['universal_trial_worker']=dict(pid=process.pid,command=shlex.join(command),log=str(log.relative_to(ROOT)),stage=stage);write(path,r)
            if process.wait():raise RuntimeError('Universal trial failed: '+stage)
    write(OUT/'status.json',dict(status='complete',queries=16,no_full_evaluation=True))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['worker','score']);args=a.parse_args()
    try:globals()[args.stage]()if args.stage else controller()
    except Exception as e:
        v=dict(status='failed',error=type(e).__name__,message=str(e));write(OUT/'failure.json',v);write(OUT/'status.json',v);raise
