"""Bounded cached reward and actual decoder interventions, never parameter search.

The CPU worker probes output-chart gradients on all128 cached queries; the GPU
worker probes one Adam step per method on16 input-hash-selected queries. Separate
CPU scoring opens labels only after both complete probe seals.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from stvg.runtime import sha,setup,lease
from stvg.tube_runner import digest,write
from stvg import policy
from tools.layout_trial.reward import LayoutReward
from tools.layout_trial.worker import checked_plan,records,reference,original,load_feature

PUBLIC=ROOT/'results/tubedetr_reward_mechanism';OUT=ROOT/'outputs/tubedetr_reward_mechanism';PLAN=PUBLIC/'plan.json'
PILOT=ROOT/'results/tubedetr_layout_trial';CACHE=ROOT/'outputs/tubedetr_layout_trial'


def cosine(a,b):
    a,b=np.asarray(a).ravel(),np.asarray(b).ravel();n=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/n) if n>1e-20 else None


def verified(path):
    value=json.loads(path.read_text());assert value['status']=='complete'
    for name,h in value['files'].items():
        assert sha(ROOT/name)==h, name
    return value


def inputs():
    p=json.loads(PLAN.read_text());assert sha(Path(__file__))==p['code_sha256']
    assert sha(ROOT/'tools/complete_tubedetr_reward_mechanism.py')==p['controller_code_sha256']
    assert sha(PILOT/'final_seal.json')==p['pilot_final_seal_sha256'];verified(PILOT/'final_seal.json')
    old=checked_plan(PILOT/'plan.json');rows=records(old)
    assert [r['id'] for r in sorted(rows,key=lambda r:digest({'seed':41,'input':r}))[:16]]==p['network_query_ids']
    return p,old,rows


def cached_prediction(arm,row):
    root=CACHE/'predictions'/arm;seal=json.loads((root/'sealed.json').read_text());assert seal['queries']==128
    return original(row,root,seal)[0]


def finish(stage,entries):
    write(OUT/f'{stage}_seal.json',dict(status='complete',queries=len(entries),files=entries,
        plan_sha256=sha(PLAN),labels_read=False,new_network_training=False))


def entry(path):
    return dict(path=str(path.relative_to(OUT)),sha256=sha(path))


def cpu():
    p,old,rows=inputs();setup();torch.set_num_threads(4)
    features=json.loads((CACHE/'features_seal.json').read_text());tau=json.loads((CACHE/'temperature_calibration.json').read_text())['layout3']['tau']
    entries={}
    for i,row in enumerate(rows):
        f=load_feature(row,CACHE,features);feature=torch.from_numpy(f['features']);teacher=torch.from_numpy(f['boxes']).double()
        reward=LayoutReward(feature,teacher,3,torch.from_numpy(f['valid_ratio']));arrays={}
        for state in ['initial','Layout3OPD','Layout3Direct']:
            value=cached_prediction('GeometryOPD' if state=='initial' else state,row)
            boxes=value['E_Ref'] if state=='initial' else value['Live']
            mu=policy.chart(torch.from_numpy(boxes[value['positions8'].astype(int)]).double()).requires_grad_(True)
            D=policy.diagonal(policy.unchart(mu));loss=reward.mean_loss(policy.unchart(mu),teacher).mean()
            gd=torch.autograd.grad(loss,mu)[0].detach();gs=[]
            for seed in p['reward_probe_seeds']:
                rr=reward.feedback(mu.detach(),D,teacher,row['key'],seed,0,old['configuration']['scales'],tau)
                g=policy.mean_gradient(mu.detach(),rr['actions'],D,rr['weights'],old['configuration']['scales'])
                assert torch.isfinite(g).all();gs.append(g)
            arrays[state+'_direct_gradient']=(D*gd).numpy()
            arrays[state+'_OPD_gradients']=torch.stack([D*g for g in gs]).numpy()
            # A fixed1% normalized output-chart direction; not a realizable decoder update.
            before=float(reward(policy.unchart(mu.detach())[:,None]).mean());gains=[]
            for g in [gd,*gs]:
                normalized=D*g;denom=normalized.norm().clamp_min(1e-20)
                after=mu.detach()-p['chart_probe_radius']*D*normalized/denom
                gains.append(float(reward(policy.unchart(after)[:,None]).mean())-before)
            arrays[state+'_reward_gains']=np.array(gains)
        arrays['observed_frame_ids']=f['observed_frame_ids']
        path=OUT/'cpu'/f"{row['id']}.npz";path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
        np.savez(path,**arrays);entries[row['id']]=entry(path)
        write(OUT/'worker_progress.json',dict(stage='cpu',complete=i+1,total=128,labels_read=False));print('CPU reward probe',i+1,flush=True)
    finish('cpu',entries)


def network():
    p,old,rows=inputs();setup();lock=lease()
    from stvg.tubedetr import load_source,native_forward,TubeReplay
    from stvg.tube_decode import decode
    from stvg.inputs import validate_input
    from stvg.tensors import state_hash
    features=json.loads((CACHE/'features_seal.json').read_text());tau=json.loads((CACHE/'temperature_calibration.json').read_text())['layout3']['tau']
    model=load_source(old['configuration']);before=state_hash(model.state_dict());entries={};root,seal=reference(old)
    selected={r['id']:r for r in rows}
    for i,q in enumerate(p['network_query_ids']):
        row=selected[q];value,_=original(row,root,seal);frames,fids=decode(row['metadata'],'vidstg');validate_input(frames,fids,row['metadata'])
        native=native_forward(model,frames,fids,row['metadata']['caption'],224)
        assert np.array_equal(native['source'].numpy(),value['Source']) and native['interval']==value['Native_interval'].tolist()
        f=load_feature(row,CACHE,features);positions=value['positions8'].astype(int)
        assert np.array_equal(np.array(fids)[positions],f['observed_frame_ids'])
        reward=LayoutReward(torch.from_numpy(f['features']).cuda(),torch.from_numpy(f['boxes']).cuda().double(),3,torch.from_numpy(f['valid_ratio']).cuda())
        arrays=dict(E_Ref=value['E_Ref'],frame_ids=np.array(fids),positions8=positions,Strong_interval=value['Strong_interval'])
        for arm in ['OPD','Direct']:
            replay=TubeReplay(model,native,value['E_Ref'],positions);v=replay.values();mu=policy.chart(v['boxes'][positions]);D=policy.diagonal(v['boxes'][positions])
            if arm=='OPD':
                rr=reward.feedback(mu.detach(),D,None,row['key'],41,0,old['configuration']['scales'],tau)
                loss=policy.mixture_loss(mu,rr['actions'],D,rr['weights'],old['configuration']['scales'])
            else:loss=reward.mean_loss(v['boxes'][positions],None).mean()
            g=torch.autograd.grad(loss,replay.state)[0];assert torch.isfinite(g).all()
            arrays[arm+'_state_gradient']=g.detach().cpu().numpy()
            opt=torch.optim.Adam([replay.state],lr=old['configuration']['lr'],betas=(.9,.999),eps=1e-8)
            opt.zero_grad(set_to_none=True);replay.state.grad=g.detach();opt.step()
            with torch.no_grad():arrays[arm+'_after_one_step']=replay.values()['boxes'].cpu().numpy();replay.state.zero_()
            assert not replay.state.count_nonzero() and all(not v.requires_grad and v.grad is None for v in model.parameters())
            del replay,opt,g,v,loss
        path=OUT/'network'/f'{q}.npz';path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
        np.savez(path,**arrays);entries[q]=entry(path);del frames,native,reward
        write(OUT/'worker_progress.json',dict(stage='network',complete=i+1,total=16,labels_read=False));print('Network probe',i+1,flush=True)
    assert state_hash(model.state_dict())==before
    finish('network',entries);lock.close()


def score():
    p,old,rows=inputs();seals={stage:json.loads((OUT/f'{stage}_seal.json').read_text()) for stage in ['cpu','network']}
    for stage,s in seals.items():
        assert s['status']=='complete' and s['queries']==(128 if stage=='cpu' else 16) and s['plan_sha256']==sha(PLAN)
        for e in s['files'].values():assert sha(OUT/e['path'])==e['sha256']
    labelpath=ROOT/old['labels'];assert sha(labelpath)==old['labels_sha256']
    labels={r['id']:r for r in map(json.loads,labelpath.read_text().splitlines())};details=[]
    from stvg.metrics import official
    from tools.layout_trial.diagnostics import mean
    for row in rows:
        q=row['id'];t=labels[q];record=dict(id=q,network=None,chart={})
        with np.load(OUT/seals['cpu']['files'][q]['path']) as z:
            outside=np.array([not(t['interval'][0]<=fid<t['interval'][1]) for fid in z['observed_frame_ids']])
            record['observations_outside_event']=int(outside.sum())
            for state in ['initial','Layout3OPD','Layout3Direct']:
                d=z[state+'_direct_gradient'];gs=z[state+'_OPD_gradients'];gain=z[state+'_reward_gains']
                record['chart'][state]=dict(OPD_Direct_cosine=mean([cosine(g,d)for g in gs]),
                    OPD_seed_cosine=mean([cosine(gs[a],gs[b])for a,b in [(0,1),(0,2),(1,2)]]),
                    direct_reward_gain=float(gain[0]),OPD_reward_gain=float(gain[1:].mean()),
                    direct_gradient_fraction_outside_event=float(np.linalg.norm(d[outside],axis=-1).sum()/max(np.linalg.norm(d,axis=-1).sum(),1e-20)),
                    OPD_gradient_fraction_outside_event=mean([float(np.linalg.norm(g[outside],axis=-1).sum()/max(np.linalg.norm(g,axis=-1).sum(),1e-20))for g in gs]))
        if q in seals['network']['files']:
            with np.load(OUT/seals['network']['files'][q]['path']) as z:
                truth={int(k):v for k,v in t['boxes'].items()};meta=dict(frame_ids=z['frame_ids'].tolist(),input=row['metadata'])
                base=official(z['E_Ref'],meta,truth,t['interval'],z['Strong_interval'],'vidstg')
                observed=np.zeros(len(z['frame_ids']),bool);observed[z['positions8'].astype(int)]=True
                record['network']=dict(state_gradient_cosine=cosine(z['OPD_state_gradient'],z['Direct_state_gradient']))
                for arm in ['OPD','Direct']:
                    after=z[arm+'_after_one_step'];m=official(after,meta,truth,t['interval'],z['Strong_interval'],'vidstg')
                    displacement=np.linalg.norm(after[:,:2]-z['E_Ref'][:,:2],axis=-1)
                    record['network'][arm]=dict(m_vIoU_change_pp=100*(m['v']-base['v']),m_sIoU_change_pp=100*(m['s']-base['s']),
                        observed_center_displacement=float(displacement[observed].mean()),unobserved_center_displacement=float(displacement[~observed].mean()))
        details.append(record)
    result=dict(status='complete',cpu_queries=128,network_queries=16,chart={},network={},
        annotation_only_after_probe_seals=True,one_step_not_60_step_evidence=True,not_a_new_reward_or_full_evaluation=True,
        chart_gradients_not_decoder_state_gradients=True,GT_groups_not_adaptation_gates=True)
    for state in ['initial','Layout3OPD','Layout3Direct']:
        result['chart'][state]={k:mean([r['chart'][state][k]for r in details])for k in details[0]['chart'][state]}
    nets=[r['network']for r in details if r['network']is not None]
    result['network']['state_gradient_cosine']=mean([r['state_gradient_cosine']for r in nets])
    for arm in ['OPD','Direct']:result['network'][arm]={k:mean([r[arm][k]for r in nets])for k in nets[0][arm]}
    write(OUT/'details_private.json',details);write(PUBLIC/'summary.json',result)
    (PUBLIC/'report.md').write_text('# TubeDETR 奖励与真实decoder单步机制探针\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n\n128条缓存输出空间探针和16条输入哈希选择的真实decoder单步探针；不是60步方法评价，不按结果选择query，不证明新奖励收益。\n')
    write(PUBLIC/'final_seal.json',dict(status='complete',files={str(f.relative_to(ROOT)):sha(f)for f in [PLAN,PUBLIC/'summary.json',PUBLIC/'report.md']},
        probe_seals={stage:sha(OUT/f'{stage}_seal.json')for stage in ['cpu','network']},publication_pending=True))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['cpu','network','score'],required=True);args=a.parse_args()
    try:globals()[args.stage]()
    except Exception as e:
        write(OUT/'failure.json',dict(stage=args.stage,error=type(e).__name__,message=str(e)));raise
