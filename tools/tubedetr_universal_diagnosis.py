"""Frozen-evidence decomposition and stronger same-bank Direct follow-up.

No reward/parameter-profile changes. Original16, all21 common candidate starts,
same two Direct steps, all checkpoints retained; old eight endpoints regress.
"""
import argparse,hashlib,json,os,shlex,subprocess,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from stvg.runtime import sha,setup,lease
from stvg.tube_runner import write
from tools.tubedetr_universal_trial import inputs as original_inputs
PUBLIC=ROOT/'results/tubedetr_universal_diagnosis';OUT=ROOT/'outputs/tubedetr_universal_diagnosis';PLAN=PUBLIC/'plan.json';PARENT=ROOT/'outputs/tubedetr_universal_trial'

def inputs():
    p=json.loads(PLAN.read_text());assert sha(Path(__file__))==p['code_sha256']
    f=ROOT/'results/tubedetr_universal_trial/final_seal.json';assert sha(f)==p['parent_final_seal_sha256'];s=json.loads(f.read_text());assert s['status']=='complete'
    for name,v in s['files'].items():assert sha(ROOT/name)==v,name
    assert sha(PARENT/'sealed.json')==p['parent_prediction_seal_sha256'];old,source,rows=original_inputs();return p,old,source,rows

def worker():
    p,old,cfg,rows=inputs();setup();lock=lease()
    from stvg.tubedetr import load_source
    from stvg.tube_decode import decode
    from stvg.inputs import validate_input
    from stvg.tensors import state_hash
    from tools.universal_trial.core import VideoEvidence
    from tools.universal_trial.tubedetr_adapter import encode,ParameterInterface,teacher_seed_mass
    model=load_source(cfg['configuration']);before=state_hash(model.state_dict());parent=json.loads((PARENT/'sealed.json').read_text());mseal=json.loads((ROOT/'outputs/tubedetr_mask_reward_qualification/mask_seal.json').read_text());entries={}
    for index,q in enumerate(old['query_ids']):
        row=rows[q];e=parent['files'][q];assert sha(PARENT/e['path'])==e['sha256']and sha(PARENT/e['receipt_path'])==e['receipt_sha256']
        prior_receipt=json.loads((PARENT/e['receipt_path']).read_text())
        with np.load(PARENT/e['path'])as z:prior={k:z[k].copy()for k in z.files}
        frames,fids=decode(row['metadata'],'vidstg');validate_input(frames,fids,row['metadata']);native,features,cells,info=encode(model,frames,fids,row['metadata']['caption']);assert np.array_equal(native['source'].numpy(),prior['Source'])
        assert hashlib.sha256(features.cpu().numpy().tobytes()).hexdigest()==prior_receipt['ordinary_feature_sha256']
        interface=ParameterInterface(model,native);assert interface.names==prior_receipt['parameter_names'];assert np.array_equal(interface.execute(interface.zero()).detach().cpu().numpy(),prior['Source'])
        me=mseal['files'][q];mp=ROOT/'outputs/tubedetr_mask_reward_qualification'/me['path'];assert sha(mp)==me['sha256']
        with np.load(mp)as z:masks=torch.from_numpy(z['masks'][:,:1]).to(features);ratio=torch.from_numpy(z['valid_ratio']).to(features)
        seed=teacher_seed_mass(masks,ratio,cells);source=torch.from_numpy(prior['Source']).to(features);fps=float(row['metadata']['fps']);times=torch.tensor(fids,device=source.device,dtype=torch.double)/fps;window=torch.tensor(prior['Strong_interval'],device=source.device,dtype=torch.double)/fps
        evidence=VideoEvidence(features.detach(),cells,times,prior['positions8'].tolist(),seed,source,window,old['feature_temperature']);U=torch.from_numpy(prior['basis_U']).to(features)
        def score(b):return evidence.score(b,old['association_weight'],old['change_weight'],old['hypothesis_temperature'])
        parts=[];alts=[]
        for b,oldR in zip(prior['initial_boxes'],prior['initial_rewards']):
            r=score(torch.from_numpy(b).to(features));assert abs(float(r['total'])-oldR)<p['reward_replay_atol'];w=r['anchor_posterior'];entropy=-(w*w.clamp_min(1e-30).log()).sum()
            values=torch.stack([(w*r['support']).sum(),old['association_weight']*(w*r['association']).sum(),-old['change_weight']*r['change'],old['hypothesis_temperature']*(entropy-np.log(8))]);assert abs(float(values.sum()-r['total']))<1e-6;parts.append(values.detach().cpu().numpy())
            sr=old['hypothesis_temperature']*(torch.logsumexp(r['support']/old['hypothesis_temperature'],0)-np.log(8));ar=old['hypothesis_temperature']*(torch.logsumexp(r['association']/old['hypothesis_temperature'],0)-np.log(8));nr=old['hypothesis_temperature']*(torch.logsumexp((r['support']+old['association_weight']*r['association'])/old['hypothesis_temperature'],0)-np.log(8));alts.append([float(sr),float(ar),-float(r['change']),float(nr)])
        allboxes=[];allrewards=[];norms=[]
        for j,start in enumerate(prior['initial_coefficients']):
            u=torch.from_numpy(start).to(features).clone().requires_grad_(True);opt=torch.optim.Adam([u],lr=old['Direct_coefficient_lr'],betas=(.9,.999),eps=1e-8);pathboxes=[];pathrewards=[]
            for k in range(old['Direct_steps']):
                b=interface.execute(U@u);r=score(b)['total'];pathboxes.append(b.detach().cpu().numpy());pathrewards.append(float(r.detach()));g=torch.autograd.grad(-r,u)[0];assert torch.isfinite(g).all();norms.append(float(g.norm()));opt.zero_grad(set_to_none=True);u.grad=g.detach();opt.step()
                with torch.no_grad():u.div_(u.abs().sum().clamp_min(1.))
            with torch.no_grad():b=interface.execute(U@u);pathboxes.append(b.cpu().numpy());pathrewards.append(float(score(b)['total']))
            assert np.array_equal(pathboxes[0],prior['initial_boxes'][j])
            if 1<=j<=8:assert np.array_equal(pathboxes[-1],prior['Direct_boxes'][j-1])
            allboxes.append(pathboxes);allrewards.append(pathrewards);del b,r,g,u,opt
        path=OUT/'predictions'/f'{q}.npz';path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();np.savez(path,Direct_path_boxes=np.asarray(allboxes),Direct_path_rewards=np.asarray(allrewards),Direct_gradient_norm=np.array(norms),reward_parts=np.stack(parts),diagnostic_scores=np.array(alts),density=evidence.density.cpu().numpy(),confidence=evidence.confidence.cpu().numpy(),cells=cells.cpu().numpy(),frame_ids=np.array(fids),positions8=prior['positions8'],Strong_interval=prior['Strong_interval'])
        entries[q]=dict(path=str(path.relative_to(OUT)),sha256=sha(path));assert state_hash(model.state_dict())==before;assert all(not v.requires_grad and v.grad is None for v in model.parameters())
        del native,features,cells,interface,masks,ratio,seed,source,evidence,U,frames,w,values,sr,ar,nr
        write(OUT/'worker_progress.json',dict(complete=index+1,total=16,labels_read=False));print('Universal diagnostics',index+1,flush=True)
    write(OUT/'sealed.json',dict(status='complete',queries=16,files=entries,plan_sha256=sha(PLAN),labels_read=False,old_features_bitexact=True,old_eight_Direct_endpoints_bitexact=True,unchanged_reward=True,model_weights_unchanged=True));lock.close()

def score():
    p,old,cfg,rows=inputs();s=json.loads((OUT/'sealed.json').read_text());assert s['status']=='complete'and s['queries']==16 and s['plan_sha256']==sha(PLAN)
    for e in s['files'].values():assert sha(OUT/e['path'])==e['sha256']
    lp=ROOT/cfg['labels'];assert sha(lp)==cfg['labels_sha256'];labels={v['id']:v for v in map(json.loads,lp.read_text().splitlines())};parent=json.loads((PARENT/'sealed.json').read_text())
    from stvg.metrics import DenseTube
    from tools.universal_trial.core import cell_occupancy
    from tools.layout_trial.diagnostics import paired_ranking,mean
    from tools.tubedetr_opd_direct_diagnosis import cluster_difference
    details=[];names=['support_only','association_only','protection_only','without_protection','full_reward']
    for q in old['query_ids']:
        t=labels[q];row=rows[q]
        with np.load(PARENT/parent['files'][q]['path'])as a,np.load(OUT/s['files'][q]['path'])as z:
            gt={int(k):v for k,v in t['boxes'].items()};meta=dict(frame_ids=z['frame_ids'].tolist(),input=row['metadata']);span=z['Strong_interval'];metric=lambda b:DenseTube(b,meta,gt,t['interval']).score(span)
            bank=a['initial_boxes'];target=np.array([metric(b)['v']for b in bank]);ranks={n:paired_ranking(z['diagnostic_scores'][:,i]if i<4 else a['initial_rewards'],target)for i,n in enumerate(names)}
            j,k=np.unravel_index(z['Direct_path_rewards'].argmax(),z['Direct_path_rewards'].shape);values=dict(SourceStrong=metric(a['Source']),CandidateRerank=metric(bank[a['initial_rewards'].argmax()]),PolicyCenter=metric(a['policy_boxes'][a['policy_rewards'].argmax()]),AllStartDirect=metric(z['Direct_path_boxes'][j,k]));quality=dict(observed=[],unobserved=[])
            for index,fid in enumerate(z['frame_ids']):
                if int(fid)not in gt or not(span[0]<=fid<span[1]):continue
                x0,y0,x1,y1=np.array(gt[int(fid)])/[row['metadata']['width'],row['metadata']['height'],row['metadata']['width'],row['metadata']['height']];b=torch.tensor([[(x0+x1)/2,(y0+y1)/2,x1-x0,y1-y0]],dtype=torch.float64)
                cells=torch.from_numpy(z['cells']);occ=cell_occupancy(b,cells)[0].numpy();area=((z['cells'][:,2:]-z['cells'][:,:2]).prod(-1)*occ).sum();mass=z['density'][:,index]@occ;group='observed'if index in z['positions8']else'unobserved';c=z['confidence'][:,index];enrichment=mass-area
                quality[group].append(dict(mean_enrichment=float(enrichment.mean()),weighted_enrichment=float(c@enrichment/c.sum())if c.sum()>0 else None,positive_hypothesis_fraction=float((enrichment>0).mean()),confidence=float(c.mean())))
            details.append(dict(id=q,parent=t['parent'],values=values,reward_ranks=ranks,parts_std=z['reward_parts'].std(0).tolist(),parts_names=['support','weighted_association','negative_protection','hypothesis_entropy'],reward_improvement=float(z['Direct_path_rewards'].max()-a['initial_rewards'].max()),field_quality={g:{k:mean([v[k]for v in vs])for k in ['mean_enrichment','weighted_enrichment','positive_hypothesis_fraction','confidence']}for g,vs in quality.items()}))
    result=dict(status='complete_frozen_reward_diagnosis',queries=16,metrics={n:{k:100*float(np.mean([r['values'][n][k]for r in details]))for k in ['v','s','t']}for n in details[0]['values']},reward_ranks={n:{k:mean([r['reward_ranks'][n][k]for r in details])for k in ['pair_accuracy','spearman']}for n in names},field_quality={g:{k:mean([r['field_quality'][g][k]for r in details])for k in details[0]['field_quality'][g]}for g in ['observed','unobserved']},mean_candidate_parts_std=np.mean([r['parts_std']for r in details],axis=0).tolist(),mean_Direct_reward_improvement=float(np.mean([r['reward_improvement']for r in details])),differences={},same_reward_and_U=True,all21_common_candidate_starts=True,initial_and_both_step_checkpoints_available=True,old_eight_Direct_bitexact=True,Direct_not_tuned=True,not_equal_compute=True,posthoc_diagnostic_not_independent_confirmation=True,GT_only_after_complete_followup_seal=True,no_new_reward_or_teacher=True)
    for a,b in [('PolicyCenter','AllStartDirect'),('AllStartDirect','SourceStrong')]:result['differences'][a+'-'+b]=cluster_difference([r['values'][a]['v']-r['values'][b]['v']for r in details],[r['parent']for r in details])
    write(OUT/'details_private.json',details);write(PUBLIC/'summary.json',result);(PUBLIC/'report.md').write_text('# TubeDETR通用原型：奖励分解与完整候选起点Direct\n\n同16条、同冻结证据/U/奖励/两步/lr。Direct覆盖全部21共同候选起点，保留起点及两步所有实际输出，只按相同奖励选取；原8起点终点逐位回归。不调参、不做新奖励或全量；此处Direct更多计算，不能称等算力方法比较。分项排序与传播GT矩形支持仅离线原因诊断，不用于选门控。\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    write(PUBLIC/'final_seal.json',dict(status='complete',files={str(f.relative_to(ROOT)):sha(f)for f in [PLAN,PUBLIC/'summary.json',PUBLIC/'report.md']},followup_seal_sha256=sha(OUT/'sealed.json'),publication_pending=True))

def controller():
    inputs()
    for stage in ['worker','score']:
        command=[sys.executable,'-B','-u','tools/tubedetr_universal_diagnosis.py','--stage',stage];log=ROOT/'logs'/f'tubedetr_universal_diagnosis_{stage}.log';env=dict(os.environ)
        if stage=='score':env['CUDA_VISIBLE_DEVICES']=''
        write(OUT/'status.json',dict(status=stage,queries=16))
        with log.open('a')as f:
            process=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env);path=ROOT/'runtime/tubedetr_active_jobs.json';r=json.loads(path.read_text());r['jobs']['universal_diagnosis_worker']=dict(pid=process.pid,command=shlex.join(command),log=str(log.relative_to(ROOT)),stage=stage);write(path,r)
            if process.wait():raise RuntimeError('Universal diagnosis failed: '+stage)
    write(OUT/'status.json',dict(status='complete',queries=16))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['worker','score']);args=a.parse_args()
    try:globals()[args.stage]()if args.stage else controller()
    except Exception as e:
        v=dict(status='failed',error=type(e).__name__,message=str(e));write(OUT/'failure.json',v);write(OUT/'status.json',v);raise
