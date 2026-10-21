"""Finite cached-region sequential policy: features -> coverage -> fit -> score.

Only features uses GPU/heavy model. Coverage reads sealed candidates with GT,
then supplies one predeclared whole-panel routing bit, never per-query gates.
"""
import argparse,hashlib,json,os,shlex,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from stvg.runtime import sha,setup,lease
from stvg.tube_runner import write
from tools.layout_trial.worker import checked_plan,records,reference,original
PUBLIC=ROOT/'results/tubedetr_propagation_policy';OUT=ROOT/'outputs/tubedetr_propagation_policy';PLAN=PUBLIC/'plan.json'


def inputs():
    p=json.loads(PLAN.read_text())
    for f,h in p['code_sha256'].items():assert sha(ROOT/f)==h,f
    for folder,h in p['parent_final_seals'].items():
        path=ROOT/folder/'final_seal.json';assert sha(path)==h;s=json.loads(path.read_text());assert s['status']=='complete'
        for f,v in s['files'].items():assert sha(ROOT/f)==v,f
    assert sha(ROOT/'outputs/tubedetr_mask_reward_qualification/mask_seal.json')==p['teacher_mask_seal_sha256']
    old=checked_plan(ROOT/'results/tubedetr_layout_trial/plan.json');return p,old,{r['id']:r for r in records(old)}


def features():
    p,old,rows=inputs();setup();lock=lease()
    from stvg.tubedetr import load_source
    from stvg.tube_decode import decode
    from stvg.inputs import validate_input
    from stvg.tensors import state_hash
    from tools.universal_trial.tubedetr_adapter import encode,teacher_seed_mass
    from tools.propagation_policy.core import rollout,initial_weights,graph_path
    from tools.propagation_policy.candidates import build
    model=load_source(old['configuration']);before=state_hash(model.state_dict());root,ref=reference(old);ms=json.loads((ROOT/'outputs/tubedetr_mask_reward_qualification/mask_seal.json').read_text());entries={}
    for i,q in enumerate(p['query_ids']):
        row=rows[q];v,_=original(row,root,ref);start=time.perf_counter();frames,fids=decode(row['metadata'],'vidstg');validate_input(frames,fids,row['metadata'])
        decoder_calls=[];h=model.transformer.decoder.register_forward_hook(lambda *a:decoder_calls.append(1));tick=time.perf_counter()
        try:native,f,cells,info=encode(model,frames,fids,row['metadata']['caption'])
        finally:h.remove()
        torch.cuda.synchronize();source_seconds=time.perf_counter()-tick
        assert len(decoder_calls)==1 and np.array_equal(native['source'].numpy(),v['Source'])
        me=ms['files'][q];mp=ROOT/'outputs/tubedetr_mask_reward_qualification'/me['path'];assert sha(mp)==me['sha256']
        with np.load(mp)as z:m={k:z[k].copy()for k in z.files}
        pos=v['positions8'].astype(int).tolist();assert np.array_equal(np.array(fids)[pos],m['observed_frame_ids']);source=native['source'].to(f)
        tick=time.perf_counter();seed=teacher_seed_mass(torch.from_numpy(m['masks'][:,:1]).to(f),torch.from_numpy(m['valid_ratio']).to(f),cells)
        fps=float(row['metadata']['fps']);times=torch.tensor(fids,device=f.device,dtype=torch.double)/fps;window=torch.tensor(v['Strong_interval'],device=f.device,dtype=torch.double)/fps
        valid_pixels=torch.from_numpy(m['valid_ratio'].prod(-1)*m['masks'].shape[-2]*m['masks'].shape[-1]).to(f);area=(cells[:,2:]-cells[:,:2]).prod(-1);seeds=(seed/valid_pixels[:,None]/area[None]).clamp(0,1)
        cache=build(f,cells,times,pos,seeds,source,torch.from_numpy(m['boxes'][:,0]).to(f),window,p['confidence_threshold']);torch.cuda.synchronize();evidence_seconds=time.perf_counter()-tick
        cache={k:x.detach().cpu()for k,x in cache.items()};phi=initial_weights(cache)
        with torch.no_grad():
            initial=rollout(cache,phi,greedy=True);assert initial['choices'].count_nonzero()==0
            cache['feedback_source']=initial['logq'];cache['EvidenceMemory_path']=rollout(cache,phi,greedy=True,teacher_path=True)['choices'];cache['EvidenceNoMemory_path']=rollout(cache,phi,greedy=True,teacher_path=True,use_memory=False)['choices'];cache['FixedGraph_path']=graph_path(cache)
        cache.update(source=source.cpu(),frame_ids=torch.tensor(fids),Strong_interval=torch.tensor(v['Strong_interval']))
        path=OUT/'cache'/f'{q}.pt';path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();torch.save(cache,path)
        receipt=dict(Source_bitexact=True,decoder_calls_common_source=1,heavy_adaptation_calls=0,new_teacher_calls=0,source_seconds=source_seconds,evidence_candidate_seconds=evidence_seconds,total_prepare_seconds=time.perf_counter()-start,feature_interface=info,teacher_cache_sha256=me['sha256'],valid_candidate_counts=cache['valid'].sum(-1).tolist(),eligible_frames=int(cache['active'].sum()),GT_read=False)
        rp=path.with_suffix('.json');write(rp,receipt);entries[q]=dict(path=str(path.relative_to(OUT)),sha256=sha(path),receipt_path=str(rp.relative_to(OUT)),receipt_sha256=sha(rp));assert state_hash(model.state_dict())==before
        del native,f,cells,frames,source,seed,seeds,cache
        write(OUT/'worker_progress.json',dict(stage='features',complete=i+1,total=16));print('features',i+1,flush=True)
    write(OUT/'features_seal.json',dict(status='complete',queries=16,files=entries,plan_sha256=sha(PLAN),labels_read=False,frozen_network_unchanged=True,new_teacher_calls=0));lock.close()


def caches():
    s=json.loads((OUT/'features_seal.json').read_text());assert s['status']=='complete'and s['queries']==16 and s['plan_sha256']==sha(PLAN)
    for e in s['files'].values():assert sha(OUT/e['path'])==e['sha256']and sha(OUT/e['receipt_path'])==e['receipt_sha256']
    return s


def labels(old):
    lp=ROOT/old['labels'];assert sha(lp)==old['labels_sha256'];return{x['id']:x for x in map(json.loads,lp.read_text().splitlines())}


def coverage():
    p,old,rows=inputs();s=caches();truth=labels(old)
    from stvg.metrics import box_iou,DenseTube
    from tools.propagation_policy.core import output
    from stvg.metrics_base import xyxy
    from tools.layout_trial.diagnostics import paired_ranking,mean
    details=[]
    for q in p['query_ids']:
        c=torch.load(OUT/s['files'][q]['path'],map_location='cpu',weights_only=True);t=truth[q];gt={int(k):v for k,v in t['boxes'].items()};pos=set(c['positions'].tolist());base=[];oracle=[];eligible=0;improved=0;total=0;ranks=[];memory=[];nomemory=[]
        for j,f in enumerate(c['frame_ids'].tolist()):
            if j in pos or f not in gt:continue
            vals=box_iou(xyxy(c['boxes'][j].numpy(),rows[q]['metadata']['width'],rows[q]['metadata']['height']),np.array(gt[f]));vals[~c['valid'][j].numpy()]=-1
            use=c['valid'][j].numpy();ranks.append(paired_ranking(c['feedback_source'][j].numpy()[use],vals[use]));memory.append(float(vals[int(c['EvidenceMemory_path'][j])]));nomemory.append(float(vals[int(c['EvidenceNoMemory_path'][j])]))
            base.append(float(vals[0]));oracle.append(float(vals.max()));eligible+=int(c['active'][j]);improved+=int(vals.max()>vals[0]+.05);total+=1
        meta=dict(frame_ids=c['frame_ids'].tolist(),input=rows[q]['metadata']);statics={'SourceStrong':c['source'].numpy(),'EvidenceMemory':output(c,c['EvidenceMemory_path']).numpy(),'EvidenceNoMemory':output(c,c['EvidenceNoMemory_path']).numpy(),'FixedGraph':output(c,c['FixedGraph_path']).numpy()};static_metrics={name:DenseTube(b,meta,gt,t['interval']).score(c['Strong_interval'].tolist())for name,b in statics.items()}
        details.append(dict(id=q,static_metrics=static_metrics,unobserved_GT_frames=total,eligible_unobserved_frames=eligible,source_mean=float(np.mean(base))if base else None,oracle_mean=float(np.mean(oracle))if oracle else None,improvable_by_5pp_frames=improved,reward_pair_accuracy=mean([x['pair_accuracy']for x in ranks]),reward_spearman=mean([x['spearman']for x in ranks]),memory_mean=mean(memory),no_memory_mean=mean(nomemory)))
    vals=[d for d in details if d['unobserved_GT_frames']];gain=100*float(np.mean([d['oracle_mean']-d['source_mean']for d in vals]))
    pair=mean([d['reward_pair_accuracy']for d in vals]);corr=mean([d['reward_spearman']for d in vals]);proceed=gain>=p['coverage_gate_oracle_gain_pp']and sum(d['eligible_unobserved_frames']for d in vals)>0 and pair is not None and pair>.5 and corr is not None and corr>0
    result=dict(status='complete',queries=16,eligible_queries=len(vals),unobserved_GT_frames=sum(d['unobserved_GT_frames']for d in vals),eligible_unobserved_frames=sum(d['eligible_unobserved_frames']for d in vals),candidate_oracle_minus_Source_unobserved_pp=gain,improvable_by_5pp_frames=sum(d['improvable_by_5pp_frames']for d in vals),reward_pair_accuracy=pair,reward_spearman=corr,memory_minus_no_memory_unobserved_pp=100*mean([d['memory_mean']-d['no_memory_mean']for d in vals]),static_metrics={name:{k:100*mean([d['static_metrics'][name][k]for d in details])for k in ['v','s','t']}for name in ['SourceStrong','EvidenceMemory','EvidenceNoMemory','FixedGraph']},proceed_to_all16_adaptation=proceed,criterion='Predeclared whole-panel scientific routing only: oracle gain>=1pp; reliable unobserved frames exist; feedback pair>0.5 and Spearman>0. No GT in method stop/revert/candidate selection' ,oracle_unrealizable_upper_bound=True,GT_offline_only=True,not_independent_confirmation=True)
    write(OUT/'coverage_private.json',details);write(PUBLIC/'coverage.json',result);write(OUT/'coverage_seal.json',dict(status='complete',features_seal_sha256=sha(OUT/'features_seal.json'),coverage_sha256=sha(PUBLIC/'coverage.json'),whole_panel_routing_only=True))


def adapt():
    p,old,rows=inputs();s=caches();routing=json.loads((PUBLIC/'coverage.json').read_text());assert routing['proceed_to_all16_adaptation'];assert sha(PUBLIC/'coverage.json')==json.loads((OUT/'coverage_seal.json').read_text())['coverage_sha256']
    setup();torch.set_num_threads(1)
    from tools.propagation_policy.core import initial_weights,rollout,fixed_loss,graph_path,output
    entries={}
    for i,q in enumerate(p['query_ids']):
        start=time.perf_counter();cache=torch.load(OUT/s['files'][q]['path'],map_location='cpu',weights_only=True);cache={k:x.double()if x.is_floating_point()else x for k,x in cache.items()};T=len(cache['boxes']);init=initial_weights(cache);traces={};predictions={'SourceStrong':cache['source'].numpy()};cost={}
        noises=[]
        for r in range(3):
            seed=int(hashlib.sha256((q+'|41|'+str(r)).encode()).hexdigest()[:15],16);g=torch.Generator().manual_seed(seed);u=torch.rand(2,T,3,generator=g,dtype=torch.double).clamp(1e-12,1-1e-12);noises.append(-(-u.log()).log())
        fixed=[rollout(cache,init,n,teacher_path=True)for n in noises[0]]
        with torch.no_grad():
            initial=rollout(cache,init,greedy=True);predictions['InitialPolicy']=output(cache,initial['choices']).numpy();gp=graph_path(cache);predictions['FixedGraph']=output(cache,gp).numpy();predictions['EvidenceMemory']=output(cache,cache['EvidenceMemory_path']).numpy();predictions['EvidenceNoMemory']=output(cache,cache['EvidenceNoMemory_path']).numpy();assert np.array_equal(predictions['InitialPolicy'],predictions['SourceStrong']);traces['InitialPolicy']=dict(choices=initial['choices'].tolist(),proxy=initial['proxy'],writes=initial['writes']);traces['FixedGraph']=dict(choices=gp.tolist())
        for kind in p['adaptation_arms']:
            tick=time.perf_counter();phi=init.clone().requires_grad_(True);opt=torch.optim.Adam([phi],lr=p['learning_rate'],betas=(.9,.999),eps=1e-8);history=[];accepted=initial;best_proxy=initial['proxy']
            for r in range(p['rounds']):
                if kind=='FixedPathKD':loss=fixed_loss(cache,phi,fixed);runs=[]
                else:
                    runs=[rollout(cache,phi,n,kind=kind)for n in noises[r]];loss=torch.stack([a['loss']for a in runs]).mean()
                previous_phi=phi.detach().clone();opt.zero_grad(set_to_none=True);loss.backward();assert phi.grad is not None and torch.isfinite(phi.grad).all();gn=float(phi.grad.norm());torch.nn.utils.clip_grad_norm_([phi],1.);opt.step()
                with torch.no_grad():
                    check=rollout(cache,phi,greedy=True)
                    accept=check['proxy']>=best_proxy-1e-10
                    if accept:accepted=check;best_proxy=check['proxy']
                    else:phi.copy_(previous_phi)
                history.append(dict(round=r+1,loss=float(loss.detach()),gradient_norm=gn,weights=phi.detach().tolist(),actual_greedy_proxy=check['proxy'],accepted=accept,actual_choices=check['choices'].tolist(),sampled_choices=[a['choices'].tolist()for a in runs],sampled_memory_writes=[a['writes']for a in runs],sampled_logq=[a['logq'].tolist()for a in runs],sampled_policy=[a['probabilities'].tolist()for a in runs]))
                if not accept:break
            predictions[kind]=output(cache,accepted['choices']).numpy();traces[kind]=history;cost[kind]=dict(seconds=time.perf_counter()-tick,attempted_updates=len(history),accepted_updates=sum(x['accepted']for x in history),sampled_paths_per_update=2,heavy_decoder_calls=0,backwards=len(history),trainable_parameters=phi.numel(),full_available_Direct_gradients=kind=='DirectSoft',soft_history_relaxation=kind=='DirectSoft')
            del phi,opt,loss,runs
        path=OUT/'predictions'/f'{q}.npz';path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();np.savez(path,**predictions,frame_ids=cache['frame_ids'].numpy(),positions8=cache['positions'].numpy(),Strong_interval=cache['Strong_interval'].numpy());rp=path.with_suffix('.json');write(rp,dict(cost=cost,traces=traces,total_seconds=time.perf_counter()-start,new_teacher_calls=0,heavy_calls_adaptation=0,labels_read=False));entries[q]=dict(path=str(path.relative_to(OUT)),sha256=sha(path),receipt_path=str(rp.relative_to(OUT)),receipt_sha256=sha(rp))
        write(OUT/'worker_progress.json',dict(stage='adapt',complete=i+1,total=16));print('adapt',i+1,flush=True)
    write(OUT/'sealed.json',dict(status='complete',queries=16,files=entries,features_seal_sha256=sha(OUT/'features_seal.json'),plan_sha256=sha(PLAN),labels_read=False,heavy_calls_adaptation=0))


def score():
    p,old,rows=inputs();fs=caches();s=json.loads((OUT/'sealed.json').read_text());assert s['status']=='complete'and s['queries']==16 and s['plan_sha256']==sha(PLAN)
    for e in s['files'].values():assert sha(OUT/e['path'])==e['sha256']and sha(OUT/e['receipt_path'])==e['receipt_sha256']
    truth=labels(old)
    from stvg.metrics import DenseTube,box_iou
    from stvg.metrics_base import xyxy
    from tools.layout_trial.diagnostics import mean
    from tools.tubedetr_opd_direct_diagnosis import cluster_difference
    details=[];costs=[];preparation=[];legacy_root=ROOT/'outputs/tubedetr_layout_trial/predictions/Layout3Direct';legacy_seal=json.loads((legacy_root/'sealed.json').read_text());assert sha(legacy_root/'sealed.json')==p['legacy_Direct_prediction_seal_sha256']
    for q in p['query_ids']:
        t=truth[q];gt={int(k):v for k,v in t['boxes'].items()};e=s['files'][q];receipt=json.loads((OUT/e['receipt_path']).read_text());costs.append(receipt);preparation.append(json.loads((OUT/fs['files'][q]['receipt_path']).read_text()));values={};unseen={};observations={};far={};continuation={}
        with np.load(OUT/e['path'])as z:
            names=['SourceStrong','InitialPolicy','FixedGraph','EvidenceMemory','EvidenceNoMemory']+p['adaptation_arms'];meta=dict(frame_ids=z['frame_ids'].tolist(),input=rows[q]['metadata']);observed=set(z['frame_ids'][z['positions8']].tolist());le=legacy_seal['files'][q];assert sha(legacy_root/le['path'])==le['sha256']
            with np.load(legacy_root/le['path'])as prev:legacy=prev['Live'].copy();assert np.array_equal(prev['frame_ids'],z['frame_ids'])
            names+=['LegacyLayout3Direct']
            for n in names:
                b=legacy if n=='LegacyLayout3Direct'else z[n];metric=DenseTube(b,meta,gt,t['interval']);values[n]=metric.score(z['Strong_interval']);mask=np.array([int(f)not in observed for f in metric.fids]);unseen[n]=float(metric.iou[mask].mean())if mask.any()else None;observations[n]=float(metric.iou[~mask].mean())if(~mask).any()else None;dist=np.abs(metric.fids[:,None]-np.array(sorted(observed))[None]).min(-1);far_mask=mask&(dist>=np.median(np.diff(sorted(observed)))/2);far[n]=float(metric.iou[far_mask].mean())if far_mask.any()else None
                # Sequential sampled-frame low-IoU continuation, only a box-GT
                # proxy; not a semantic identity-switch annotation.
                ids=[j for j,f in enumerate(z['frame_ids'])if int(f)in gt and int(f)not in observed];ious=[float(box_iou(xyxy(b[j],rows[q]['metadata']['width'],rows[q]['metadata']['height']),np.array(gt[int(z['frame_ids'][j])])))for j in ids];bad=np.array(ious)<.5
                continuation[n]=dict(unobserved_low_IoU_fraction=float(bad.mean())if len(bad)else None,bad_after_bad=float(np.mean(bad[1:][bad[:-1]]))if len(bad)>1 and bad[:-1].any()else None)
        details.append(dict(id=q,parent=t['parent'],values=values,unobserved_dense=unseen,observed=observations,far=far,low_IoU_continuation=continuation))
    result=dict(status='complete_finite_cached_propagation_policy_trial',queries=16,coverage=json.loads((PUBLIC/'coverage.json').read_text()),metrics={n:{k:100*mean([r['values'][n][k]for r in details])for k in ['v','s','t']}for n in names},unobserved_dense_sIoU={n:100*mean([r['unobserved_dense'][n]for r in details])for n in names},observed_sIoU={n:100*mean([r['observed'][n]for r in details])for n in names},far_unobserved_sIoU={n:100*mean([r['far'][n]for r in details])for n in names},unobserved_degraded_query_fraction={n:mean([float(r['unobserved_dense'][n]<r['unobserved_dense']['SourceStrong'])for r in details if r['unobserved_dense'][n]is not None])for n in names},low_IoU_continuation={n:{k:mean([r['low_IoU_continuation'][n][k]for r in details])for k in ['unobserved_low_IoU_fraction','bad_after_bad']}for n in names},differences={},execution_cost={n:dict(mean_seconds=mean([r['cost'][n]['seconds']for r in costs]),median_seconds=float(np.median([r['cost'][n]['seconds']for r in costs])),P90_seconds=float(np.percentile([r['cost'][n]['seconds']for r in costs],90)),mean_attempted_updates=mean([r['cost'][n]['attempted_updates']for r in costs]),mean_accepted_updates=mean([r['cost'][n]['accepted_updates']for r in costs]),paths_per_update=2,max_backwards=3,heavy_calls=0)for n in p['adaptation_arms']},common_preparation=dict(mean_heavy_source_seconds=mean([r['source_seconds']for r in preparation]),mean_evidence_candidate_seconds=mean([r['evidence_candidate_seconds']for r in preparation]),source_decoder_calls_per_query=1,new_teacher_calls=0,teacher_and_Strong_are_exact_cached_not_new_end_to_end=True),frozen_time=True,original_plus_new_output_module=True,not_original_algorithm_equivalent_acceleration=True,no_GT_memory_gating=True,development_exposure_disclosed=True,no_parameter_search=True,Direct_not_fairly_tuned=True,DirectSoft_full_gradients_and_relaxation_disclosed=True,not_independent_confirmation=True,not_full_population=True)
    for a,b in [('OPD','FixedPathKD'),('OPD','DirectSoft'),('OPD','FixedGraph'),('OPD','SourceStrong')]:result['differences'][a+'-'+b]=cluster_difference([r['values'][a]['v']-r['values'][b]['v']for r in details],[r['parent']for r in details])
    write(OUT/'details_private.json',details);write(PUBLIC/'summary.json',result);(PUBLIC/'report.md').write_text('# TubeDETR＋轻量序列传播策略：固定16条初试\n\n先封存最多4候选，再离线核验未观测候选上界；只以事前总体条件接续全部16条，不按GT筛query或调参。重网络仅公共准备一次；v0小模块6零初始权重、Source先验.7、2路径×最多3更新、lr.1/clip1/先验KL.05；固定不含记忆自奖励的代理分数下降则回退停止。OPD在硬学生历史上逐状态枚举四个动作的forward KL；固定路径KD在冻结伪教师历史；DirectSoft为完整梯度soft-history forward KL，不声称松弛是离散历史的精确梯度。固定图精确最大化共同固定检查评分，不奖励记忆自匹配。身份漂移仅以低IoU延续代理诊断，不冒充身份标注。\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n');finish()


def finish():
    files=[PLAN,PUBLIC/'coverage.json',PUBLIC/'summary.json',PUBLIC/'report.md'];write(PUBLIC/'final_seal.json',dict(status='complete',files={str(f.relative_to(ROOT)):sha(f)for f in files},features_seal_sha256=sha(OUT/'features_seal.json'),prediction_seal_sha256=sha(OUT/'sealed.json')if(OUT/'sealed.json').exists()else None,publication_pending=True))


def controller():
    inputs()
    for stage in ['features','coverage','adapt','score']:
        if stage=='adapt'and not json.loads((PUBLIC/'coverage.json').read_text())['proceed_to_all16_adaptation']:
            write(PUBLIC/'summary.json',dict(status='stopped_at_v0_candidate_feedback_qualification',coverage=json.loads((PUBLIC/'coverage.json').read_text()),no_policy_fit=True,no_parameter_search=True,not_independent_confirmation=True));(PUBLIC/'report.md').write_text('# v0候选/反馈资格未通过，未做策略适应\n');finish();break
        command=[sys.executable,'-B','-u','tools/tubedetr_propagation_policy.py','--stage',stage];log=ROOT/'logs'/f'tubedetr_propagation_policy_{stage}.log';env=dict(os.environ)
        if stage!='features':env['CUDA_VISIBLE_DEVICES']=''
        write(OUT/'status.json',dict(status=stage,queries=16))
        with log.open('a')as f:
            process=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env);path=ROOT/'runtime/tubedetr_active_jobs.json';r=json.loads(path.read_text());r['jobs']['propagation_policy_worker']=dict(pid=process.pid,command=shlex.join(command),log=str(log.relative_to(ROOT)),stage=stage);write(path,r)
            if process.wait():raise RuntimeError('Propagation policy failed: '+stage)
    write(OUT/'status.json',dict(status='complete',queries=16,no_automatic_full=True))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['features','coverage','adapt','score']);args=a.parse_args()
    try:globals()[args.stage]()if args.stage else controller()
    except Exception as e:
        v=dict(status='failed',error=type(e).__name__,message=str(e));write(OUT/'failure.json',v);write(OUT/'status.json',v);raise
