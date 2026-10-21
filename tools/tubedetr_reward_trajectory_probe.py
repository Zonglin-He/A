"""Exact16-query replay of the sealed ROI methods, with full optimization paths.

This is a mechanistic replay, not parameter selection or a new method result.
The worker never reads GT; an independent score process follows the full seal.
"""
import argparse,json,os,shlex,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from stvg.runtime import sha,setup,lease
from stvg.tube_runner import write
from tools.layout_trial.worker import checked_plan,records,reference,original,load_feature

PUBLIC=ROOT/'results/tubedetr_reward_trajectory';OUT=ROOT/'outputs/tubedetr_reward_trajectory';PLAN=PUBLIC/'plan.json'
CACHE=ROOT/'outputs/tubedetr_layout_trial';PILOT=ROOT/'results/tubedetr_layout_trial'

def inputs():
    p=json.loads(PLAN.read_text());assert sha(Path(__file__))==p['code_sha256']
    assert sha(ROOT/'results/tubedetr_reward_mechanism/final_seal.json')==p['mechanism_final_seal_sha256']
    prior=checked_plan(PILOT/'plan.json');return p,prior,{r['id']:r for r in records(prior)}

def worker():
    p,prior,rows=inputs();setup();lock=lease()
    from stvg.tubedetr import load_source,native_forward,TubeReplay
    from stvg.tube_decode import decode
    from stvg.inputs import validate_input
    from stvg.tensors import state_hash
    from tools.layout_trial.reward import LayoutReward,fit_layout,fit_direct
    model=load_source(prior['configuration']);before=state_hash(model.state_dict());root,oldseal=reference(prior)
    fs=json.loads((CACHE/'features_seal.json').read_text());cal=json.loads((CACHE/'temperature_calibration.json').read_text());entries={}
    for i,q in enumerate(p['query_ids']):
        row=rows[q];value,_=original(row,root,oldseal);frames,fids=decode(row['metadata'],'vidstg');validate_input(frames,fids,row['metadata'])
        native=native_forward(model,frames,fids,row['metadata']['caption'],224)
        assert np.array_equal(native['source'].numpy(),value['Source']) and native['interval']==value['Native_interval'].tolist()
        f=load_feature(row,CACHE,fs);positions=value['positions8'].astype(int);assert np.array_equal(np.array(fids)[positions],f['observed_frame_ids'])
        target=torch.from_numpy(f['boxes']).cuda().double();reward=LayoutReward(torch.from_numpy(f['features']).cuda(),target,3,torch.from_numpy(f['valid_ratio']).cuda())
        arrays=dict(frame_ids=np.array(fids),positions8=positions,Strong_interval=value['Strong_interval'])
        for arm in ['Layout3OPD','Layout3Direct']:
            cfg=dict(prior['configuration']);cfg['tau']=cal['layout3']['tau'];rp=TubeReplay(model,native,value['E_Ref'],positions)
            result=fit_layout(rp,target,row['key'],41,cfg,reward) if arm.endswith('OPD')else fit_direct(rp,target,cfg,reward)
            cache_root=CACHE/'predictions'/arm;cache_seal=json.loads((cache_root/'sealed.json').read_text());old,_=original(row,cache_root,cache_seal)
            if not np.array_equal(result['final'].numpy(),old['Live']):raise ValueError('Exact60-step replay differs: '+arm)
            assert not rp.state.count_nonzero() and all(not v.requires_grad and v.grad is None for v in model.parameters())
            arrays[arm+'_boxes']=np.stack([step['boxes'].numpy()for step in result['path']])
            arrays[arm+'_pseudo_loss']=np.array([step['anchor_loss']for step in result['path']])
            del result,rp
        path=OUT/'paths'/f'{q}.npz';path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();np.savez(path,**arrays)
        entries[q]=dict(path=str(path.relative_to(OUT)),sha256=sha(path));del native,frames,reward,target
        write(OUT/'worker_progress.json',dict(complete=i+1,total=16,labels_read=False));print('Exact trajectory replay',i+1,flush=True)
    assert state_hash(model.state_dict())==before
    write(OUT/'sealed.json',dict(status='complete',queries=16,files=entries,plan_sha256=sha(PLAN),all_endpoints_bitexact=True,labels_read=False,network_weights_unchanged=True));lock.close()

def score():
    p,prior,rows=inputs();seal=json.loads((OUT/'sealed.json').read_text());assert seal['status']=='complete' and seal['queries']==16 and seal['plan_sha256']==sha(PLAN)
    for e in seal['files'].values():assert sha(OUT/e['path'])==e['sha256']
    label_path=ROOT/prior['labels'];assert sha(label_path)==prior['labels_sha256'];labels={v['id']:v for v in map(json.loads,label_path.read_text().splitlines())}
    from stvg.metrics import DenseTube
    curves={a:[]for a in ['Layout3OPD','Layout3Direct']};losses={a:[]for a in curves}
    for q in p['query_ids']:
        row=rows[q];truth=labels[q];gt={int(k):v for k,v in truth['boxes'].items()}
        with np.load(OUT/seal['files'][q]['path'])as z:
            meta=dict(frame_ids=z['frame_ids'].tolist(),input=row['metadata'])
            for arm in curves:
                vals=[DenseTube(b,meta,gt,truth['interval']).score(z['Strong_interval'])for b in z[arm+'_boxes']]
                curves[arm].append([[v['v'],v['s']]for v in vals]);losses[arm].append(z[arm+'_pseudo_loss'])
    result=dict(status='complete',queries=16,endpoints_bitexact_with_old_ROI=True,mechanistic_replay_not_new_method=True,
        no_step_selection_or_GT_early_stopping=True,annotation_only_after_all_paths_sealed=True,arms={})
    for arm in curves:
        c=np.array(curves[arm]);loss=np.array(losses[arm]);mean=c.mean(0)
        result['arms'][arm]=dict(steps={str(k):dict(m_vIoU=100*float(mean[k,0]),m_sIoU=100*float(mean[k,1]),pseudo_loss=float(loss[:,k].mean()))for k in p['reported_steps']},
            pseudo_loss_down_but_dense_down_queries=int(((loss[:,-1]<loss[:,0])&(c[:,-1,1]<c[:,0,1])).sum()),
            pseudo_loss_down_but_volume_down_queries=int(((loss[:,-1]<loss[:,0])&(c[:,-1,0]<c[:,0,0])).sum()))
    write(PUBLIC/'summary.json',result);write(OUT/'curves_private.json',dict(values={a:np.array(v).tolist()for a,v in curves.items()}))
    (PUBLIC/'report.md').write_text('# TubeDETR 真实decoder全60步路径复现\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n\n仅16条机制探针；不按GT选步、不代表128或全量收益。原终点须逐位复现，未修改原已封存结果。\n')
    write(PUBLIC/'final_seal.json',dict(status='complete',files={str(f.relative_to(ROOT)):sha(f)for f in [PLAN,PUBLIC/'summary.json',PUBLIC/'report.md']},trajectory_seal_sha256=sha(OUT/'sealed.json'),publication_pending=True))

def controller():
    inputs()
    for stage in ['worker','score']:
        command=[sys.executable,'-B','-u','tools/tubedetr_reward_trajectory_probe.py','--stage',stage];log=ROOT/'logs'/f'tubedetr_reward_trajectory_{stage}.log';env=dict(os.environ)
        if stage=='score':env['CUDA_VISIBLE_DEVICES']=''
        write(OUT/'status.json',dict(status=stage,new_parameter_search=False))
        with log.open('a')as f:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env)
            path=ROOT/'runtime/tubedetr_active_jobs.json';j=json.loads(path.read_text());j['jobs']['reward_trajectory_worker']=dict(pid=proc.pid,command=shlex.join(command),log=str(log.relative_to(ROOT)),stage=stage);write(path,j)
            if proc.wait():raise RuntimeError('Trajectory stage failed: '+stage)
    write(OUT/'status.json',dict(status='complete',queries=16,old_queues_still_deferred=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['worker','score']);a=p.parse_args()
    try:globals()[a.stage]() if a.stage else controller()
    except Exception as e:
        v=dict(status='failed',error=type(e).__name__,message=str(e));write(OUT/'failure.json',v);write(OUT/'status.json',v);raise
