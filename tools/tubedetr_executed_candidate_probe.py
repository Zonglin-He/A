"""Same-forward-budget executed-state candidates versus Direct direction search.

Sixteen existing input-hash queries; no new teacher, training or GT selection.
Both candidate sets include zero and16 actual decoder executions. Offline GT
oracle ceilings are explicitly diagnostic and never used to choose predictions.
"""
import argparse,json,os,shlex,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from stvg.runtime import sha,setup,lease
from stvg.tube_runner import write
from tools.layout_trial.worker import checked_plan,records,reference,original,load_feature

PUBLIC=ROOT/'results/tubedetr_executed_candidates';OUT=ROOT/'outputs/tubedetr_executed_candidates';PLAN=PUBLIC/'plan.json'
CACHE=ROOT/'outputs/tubedetr_layout_trial';PILOT=ROOT/'results/tubedetr_layout_trial'

def inputs():
    p=json.loads(PLAN.read_text());assert sha(Path(__file__))==p['code_sha256']
    for folder,h in p['parent_final_seals'].items():
        path=ROOT/folder/'final_seal.json';assert sha(path)==h;seal=json.loads(path.read_text());assert seal['status']=='complete'
        for name,v in seal['files'].items():assert sha(ROOT/name)==v,name
    assert sha(ROOT/'outputs/tubedetr_reward_mechanism/network_seal.json')==p['reference_network_seal_sha256']
    prior=checked_plan(PILOT/'plan.json');return p,prior,{r['id']:r for r in records(prior)}

def worker():
    p,prior,rows=inputs();setup();lock=lease()
    from stvg.tubedetr import load_source,native_forward,TubeReplay
    from stvg.tube_decode import decode
    from stvg.inputs import validate_input
    from stvg.tensors import state_hash
    from tools.layout_trial.reward import LayoutReward
    from stvg import policy
    model=load_source(prior['configuration']);before=state_hash(model.state_dict());root,oldseal=reference(prior)
    fs=json.loads((CACHE/'features_seal.json').read_text());tau=json.loads((CACHE/'temperature_calibration.json').read_text())['layout3']['tau']
    ns=json.loads((ROOT/'outputs/tubedetr_reward_mechanism/network_seal.json').read_text());entries={}
    for index,q in enumerate(p['query_ids']):
        row=rows[q];old,_=original(row,root,oldseal);frames,fids=decode(row['metadata'],'vidstg');validate_input(frames,fids,row['metadata'])
        native=native_forward(model,frames,fids,row['metadata']['caption'],224);assert np.array_equal(native['source'].numpy(),old['Source'])
        f=load_feature(row,CACHE,fs);pos=old['positions8'].astype(int);assert np.array_equal(np.array(fids)[pos],f['observed_frame_ids'])
        reward=LayoutReward(torch.from_numpy(f['features']).cuda(),torch.from_numpy(f['boxes']).cuda().double(),3,torch.from_numpy(f['valid_ratio']).cuda())
        rp=TubeReplay(model,native,old['E_Ref'],pos);v=rp.values();loss=reward.mean_loss(v['boxes'][pos],None).mean();gd=torch.autograd.grad(loss,rp.state)[0].detach()
        del v,loss
        gradients=[]
        for seed in p['OPD_seeds']:
            v=rp.values();mu=policy.chart(v['boxes'][pos]);D=policy.diagonal(v['boxes'][pos]);rr=reward.feedback(mu.detach(),D,None,row['key'],seed,0,prior['configuration']['scales'],tau)
            loss=policy.mixture_loss(mu,rr['actions'],D,rr['weights'],prior['configuration']['scales']);g=torch.autograd.grad(loss,rp.state)[0].detach();assert torch.isfinite(g).all();gradients.append(g)
        def adam_delta(g):
            with torch.no_grad():rp.state.zero_()
            opt=torch.optim.Adam([rp.state],lr=prior['configuration']['lr'],betas=(.9,.999),eps=1e-8);opt.zero_grad(set_to_none=True);rp.state.grad=g.clone();opt.step();d=rp.state.detach().clone()
            with torch.no_grad():rp.state.zero_()
            rp.state.grad=None;return d
        direct=adam_delta(gd);opd=[adam_delta(g)for g in gradients]
        candidates=dict(DirectSearch=[torch.zeros_like(direct),*[scale*direct for scale in p['Direct_scales']]],
            OPDSearch=[torch.zeros_like(direct),*[scale*d for d in opd for scale in p['OPD_scales']]])
        arrays=dict(frame_ids=np.array(fids),positions8=pos,Strong_interval=old['Strong_interval'])
        e=ns['files'][q];reference_path=ROOT/'outputs/tubedetr_reward_mechanism'/e['path'];assert sha(reference_path)==e['sha256']
        with np.load(reference_path)as z:one_step={a:z[a+'_after_one_step'].copy()for a in ['Direct','OPD']}
        for arm,deltas in candidates.items():
            assert len(deltas)==17;boxes=[];scores=[];linear=[];norms=[];start=time.perf_counter()
            for d in deltas:
                with torch.no_grad():
                    rp.state.copy_(d);value=rp.values()['boxes'];boxes.append(value.cpu().numpy());scores.append(float(reward(value[pos,None]).mean()));linear.append(float(-(gd*d).sum()));norms.append(float(d.norm()));rp.state.zero_()
            refindex=1+p['Direct_scales'].index(1.) if arm=='DirectSearch' else 1+p['OPD_scales'].index(1.)
            assert np.array_equal(boxes[refindex],one_step['Direct'if arm=='DirectSearch'else'OPD']),arm+' one-step reference differs'
            scores=np.array(scores);ties=np.flatnonzero(np.isclose(scores,scores.max(),atol=1e-12,rtol=0));chosen=int(min(ties,key=lambda i:(norms[i],i)))
            arrays.update({arm+'_boxes':np.stack(boxes),arm+'_rewards':scores,arm+'_linear_gain':np.array(linear),arm+'_state_norms':np.array(norms),arm+'_selected_index':np.array(chosen),arm+'_execution_seconds':np.array(time.perf_counter()-start)})
        assert not rp.state.count_nonzero() and all(not x.requires_grad and x.grad is None for x in model.parameters())
        path=OUT/'candidates'/f'{q}.npz';path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();np.savez(path,**arrays);entries[q]=dict(path=str(path.relative_to(OUT)),sha256=sha(path))
        del native,frames,reward,rp,gradients,gd,opd,direct,deltas,d,loss,v,mu,rr,g
        write(OUT/'worker_progress.json',dict(complete=index+1,total=16,labels_read=False));print('Executed candidate probe',index+1,flush=True)
    assert state_hash(model.state_dict())==before
    write(OUT/'sealed.json',dict(status='complete',queries=16,files=entries,plan_sha256=sha(PLAN),network_weights_unchanged=True,labels_read=False,actual_decodes_per_query_per_set=17,one_step_references_bitexact=True));lock.close()

def score():
    p,prior,rows=inputs();seal=json.loads((OUT/'sealed.json').read_text());assert seal['status']=='complete' and seal['queries']==16 and seal['plan_sha256']==sha(PLAN)
    for e in seal['files'].values():assert sha(OUT/e['path'])==e['sha256']
    lp=ROOT/prior['labels'];assert sha(lp)==prior['labels_sha256'];labels={r['id']:r for r in map(json.loads,lp.read_text().splitlines())}
    from stvg.metrics import DenseTube
    from tools.tubedetr_opd_direct_diagnosis import cluster_difference
    details=[]
    for q in p['query_ids']:
        row=rows[q];t=labels[q];gt={int(k):v for k,v in t['boxes'].items()};record=dict(id=q,parent=t['parent'])
        with np.load(OUT/seal['files'][q]['path'])as z:
            meta=dict(frame_ids=z['frame_ids'].tolist(),input=row['metadata'])
            for arm in ['DirectSearch','OPDSearch']:
                metric=[DenseTube(b,meta,gt,t['interval']).score(z['Strong_interval'])for b in z[arm+'_boxes']];selected=int(z[arm+'_selected_index']);rewards=z[arm+'_rewards'];linear=z[arm+'_linear_gain']
                record[arm]=dict(selected=metric[selected],zero=metric[0],oracle_v=max(v['v']for v in metric),oracle_s=max(v['s']for v in metric),
                    zero_selected=selected==0,finite_candidate_crosses_negative_local_prediction=bool(((linear[1:]<0)&(rewards[1:]>rewards[0]+1e-8)).any()),
                    selected_dense_below_zero=metric[selected]['s']<metric[0]['s']-1e-12,selected_volume_below_zero=metric[selected]['v']<metric[0]['v']-1e-12)
        details.append(record)
    result=dict(status='complete_mechanistic_probe',queries=16,actual_decodes_per_query_per_set=17,arms={},
        no_new_teacher=True,no_parameter_search=True,GT_only_after_complete_candidate_seal=True,
        GT_oracle_is_not_deployable=True,same_forward_count_not_same_backward_cost=True,OPD_backwards_per_query=4,Direct_backwards_per_query=1,
        not_independent_confirmation=True,not_a_new_method_or_full_evaluation=True)
    for arm in ['DirectSearch','OPDSearch']:
        result['arms'][arm]=dict(m_vIoU=100*float(np.mean([r[arm]['selected']['v']for r in details])),m_sIoU=100*float(np.mean([r[arm]['selected']['s']for r in details])),
            diagnostic_oracle_v=100*float(np.mean([r[arm]['oracle_v']for r in details])),diagnostic_oracle_s=100*float(np.mean([r[arm]['oracle_s']for r in details])),
            zero_selected_queries=sum(r[arm]['zero_selected']for r in details),reward_selection_hurts_dense_queries=sum(r[arm]['selected_dense_below_zero']for r in details),
            reward_selection_hurts_volume_queries=sum(r[arm]['selected_volume_below_zero']for r in details),queries_with_negative_local_prediction_but_positive_finite_reward=sum(r[arm]['finite_candidate_crosses_negative_local_prediction']for r in details))
    result['OPD_minus_Direct_pp']={k:cluster_difference([r['OPDSearch']['selected'][key]-r['DirectSearch']['selected'][key]for r in details],[r['parent']for r in details])for k,key in [('m_vIoU','v'),('m_sIoU','s')]}
    write(OUT/'details_private.json',details);write(PUBLIC/'summary.json',result)
    (PUBLIC/'report.md').write_text('# TubeDETR 执行后候选与Direct方向搜索机制探针\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n\n各17次真实decoder候选执行，OPD四次backward对Direct一次，不能称等算力。候选按相同原Layout3奖励选、GT仅完整seal后诊断；oracle不可部署，16条不是新方法成绩。\n')
    write(PUBLIC/'final_seal.json',dict(status='complete',files={str(f.relative_to(ROOT)):sha(f)for f in [PLAN,PUBLIC/'summary.json',PUBLIC/'report.md']},candidate_seal_sha256=sha(OUT/'sealed.json'),publication_pending=True))

def controller():
    inputs()
    for stage in ['worker','score']:
        command=[sys.executable,'-B','-u','tools/tubedetr_executed_candidate_probe.py','--stage',stage];log=ROOT/'logs'/f'tubedetr_executed_candidates_{stage}.log';env=dict(os.environ)
        if stage=='score':env['CUDA_VISIBLE_DEVICES']=''
        write(OUT/'status.json',dict(status=stage,no_student_training=True))
        with log.open('a')as f:
            process=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env);path=ROOT/'runtime/tubedetr_active_jobs.json';r=json.loads(path.read_text());r['jobs']['executed_candidate_worker']=dict(pid=process.pid,command=shlex.join(command),log=str(log.relative_to(ROOT)),stage=stage);write(path,r)
            if process.wait():raise RuntimeError('Executed candidate stage failed: '+stage)
    write(OUT/'status.json',dict(status='complete',queries=16,old_queues_still_deferred=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['worker','score']);a=p.parse_args()
    try:globals()[a.stage]()if a.stage else controller()
    except Exception as e:
        v=dict(status='failed',error=type(e).__name__,message=str(e));write(OUT/'failure.json',v);write(OUT/'status.json',v);raise
