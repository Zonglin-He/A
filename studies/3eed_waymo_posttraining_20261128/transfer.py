"""Lock Waymo training, predict all transfers blind, and score sealed rows offline."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from bridge import CFG, HERE, JOINT, REGISTRY, ROOT, RUN, backend, digest, geometry_frame, publish, read, readout, replies, setup, source_bank, source_guard, truth_iou, valid_shape
from source_data import stable

ADAPTERS=('warm','source_sft','grpo','shape','graph')
MODEL_ARMS=('joint_frozen','warm','source_sft','grpo','shape_frozen','shape','shape_wrong','graph_frozen','graph_sft')
GEOMETRY_ARMS=('frozen','joint_frozen','warm','source_sft','grpo','shape_frozen','shape','shape_wrong')
GRAPH_ARMS=tuple(f'{arm}_{option}' for arm in ('graph_frozen','graph_sft') for option in ('som','true','wrong'))
EXTERNAL=ROOT/'runs/2d-pose-finalization-20261128'
ORDINARY=ROOT/'runs/ordinary-full-input'


def lock():
    source_guard()
    setup()
    from safetensors.torch import load_file
    records={}
    snapshots={}
    for name in ADAPTERS:
        path=RUN/f'{name}-receipt.json'
        report=read(path)
        assert not report['target_read'] and report['source']==digest(RUN/'source-seal.json')
        assert report['protocol']==digest(HERE/'protocol.json') and report['model_code']==digest(HERE/'model_ops.py')
        assert report['script']==digest(HERE/'train.py') and digest(RUN/report['file'])==report['sha256']
        records[name]=dict(receipt=digest(path),weights=report['sha256'])
        snapshots[name]=load_file(str(RUN/report['file']))
    for name in ('source_sft','grpo'):
        assert read(RUN/f'{name}-receipt.json')['warm_start']==records['warm']['weights']
        assert any(not (x==snapshots['warm'][k]).all() for k,x in snapshots[name].items())
    receipt=read(RUN/'conditions-receipt.json')
    assert not receipt['target_read'] and receipt['source']==digest(RUN/'source-seal.json')
    assert receipt['code']==digest(HERE/'conditioning.py')
    hashes={p.name:digest(p) for p in HERE.glob('*.py')}
    publish(RUN/'training-lock.json',dict(protocol=digest(HERE/'protocol.json'),source=digest(RUN/'source-seal.json'),
        registered_method=digest(REGISTRY),source_conditions=digest(RUN/'conditions-receipt.json'),
        checkpoints=records,code=hashes,terminal=True,source_only=True,no_target_read=True))
    print('All five source checkpoints sealed. Target access begins in a new process.',flush=True)


def training_lock():
    stamp=read(RUN/'training-lock.json')
    assert stamp['terminal'] and stamp['source_only'] and stamp['no_target_read']
    assert stamp['protocol']==digest(HERE/'protocol.json') and stamp['registered_method']==digest(REGISTRY)
    assert stamp['source']==digest(RUN/'source-seal.json')
    assert all(digest(HERE/f)==h for f,h in stamp['code'].items())
    for name,v in stamp['checkpoints'].items():
        assert digest(RUN/f'{name}.safetensors')==v['weights']
    return stamp


def unique_roundrobin(grouped, limit, used, name):
    groups={k:sorted(v,key=lambda r:stable(name+r['uid'])) for k,v in grouped.items()}
    order=sorted(groups,key=lambda k:stable(name+str(k)))
    out=[]
    for i in range(max(map(len,groups.values()))):
        for group in order:
            if i>=len(groups[group]):
                continue
            row=groups[group][i]
            sha=digest(Path(row['frame']['rgb']))
            if sha not in used:
                used.add(sha)
                out.append(dict(row,rgb_sha=sha))
            if len(out)==limit:
                return out
    raise RuntimeError((name,len(out),limit))


def target_frame(frame, path):
    return geometry_frame(dict(frame,**{k:str((path/frame[k]).resolve(strict=True)) for k in ('rgb','points','calibration')}))


def plan():
    training_lock()
    val=source_bank('val')
    used={v['rgb_sha'] for v in (*source_bank('train'),*val)}
    rows=[{k:r[k] for k in ('uid','frame','recording','rgb_sha')}|{'target':'waymo'} for r in val]
    ordinary=read(ORDINARY/'manifest.json')
    for domain in ('drone','quad'):
        groups=defaultdict(list)
        for frame in ordinary:
            if frame['target']==domain:
                recording=Path(frame['rgb']).parent.parent.name
                groups[recording].append(dict(uid=frame['uid'],frame=target_frame(frame,ORDINARY),target=domain,recording=recording))
        rows.extend(unique_roundrobin(groups,CFG['target_queries_per_platform'],used,domain))
    ext=read(EXTERNAL/'mono-input-seal.json')
    assert ext['manifest']==digest(EXTERNAL/'mono-inputs.json') and not ext['target_labels_read']
    strict=read(EXTERNAL/'mono-strict-roster-seal.json')
    strictpath=EXTERNAL/'mono-strict-roster-private.json'
    assert strict['source_input_seal']==digest(EXTERNAL/'mono-input-seal.json') and strict['file']==digest(strictpath)
    included=set(read(strictpath))
    index_path=EXTERNAL/'mono-paired-index-private.json'
    assert ext['private_index']==digest(index_path)
    records={d['queries'][0]['uid']:d['recording'] for d in read(index_path)}
    groups=defaultdict(list)
    for f in read(EXTERNAL/'mono-inputs.json'):
        if f['uid'] in included:
            record=records[f['uid']]
            groups[record].append(dict(uid=f['uid'],frame=target_frame(f,EXTERNAL),target='mono',recording=record))
    rows.extend(unique_roundrobin(groups,CFG['external_queries'],used,'mono'))
    assert len(rows)==len({v['uid'] for v in rows})==len({v['rgb_sha'] for v in rows})
    external_sensors=EXTERNAL/'mono-sensor-hashes.json'
    assert ext['sensor_bank']==digest(external_sensors)
    external_hashes=read(external_sensors)
    for row in rows:
        frame=row['frame']
        row['sensor_hashes']={k:digest(Path(frame[k])) for k in ('points','calibration')}
        if row['target']=='mono':
            expected={k:external_hashes[frame[k]] for k in ('rgb','points','calibration')}
            assert expected==dict(row['sensor_hashes'],rgb=row['rgb_sha'])
    path=RUN/'test-plan.json'
    publish(path,rows)
    publish(RUN/'test-plan-seal.json',dict(file=digest(path),weights=digest(RUN/'training-lock.json'),labels_read=False,
                description_count=dict(Counter(r['target'] for r in rows)),
                recordings={p:len({r['recording'] for r in rows if r['target']==p}) for p in ('waymo','drone','quad','mono')},
                external_reference=digest(EXTERNAL/'mono-strict-roster-seal.json')))
    print('Target plan',read(RUN/'test-plan-seal.json'),flush=True)


def blind_targets(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,Path)):
        from os import fsdecode
        name=Path(fsdecode(args[0])).name
        if name in ('test.parquet','data_object_label_2.zip','labels.json','meta_info.json','source-train.json','source-val.json') or name.startswith('scores-'):
            raise PermissionError(f'Annotation blocked during target inference: {name}')


def test_rows():
    training_lock()
    path=RUN/'test-plan.json'
    stamp=read(RUN/'test-plan-seal.json')
    assert not stamp['labels_read'] and stamp['file']==digest(path) and stamp['weights']==digest(RUN/'training-lock.json')
    return read(path)


def physical(job):
    row,raw=job
    from conditioning import condition, proposals
    return dict(uid=row['uid'],measured=condition(row['frame'],raw['2d']),bank=proposals(row['frame'],raw['3d']))


def graph_jobs(frame, base_image, bank):
    from conditioning import graph_prompt, metric_pair, marked

    if not bank:
        return [],[]
    img=marked(base_image,bank,list(range(1,len(bank)+1)))
    rotated=bank[1:]+bank[:1]
    prompts=[graph_prompt(frame,bank,with_graph=False),graph_prompt(frame,bank),graph_prompt(frame,rotated)]
    choices=[list(range(len(bank)+1))]*3
    pair=metric_pair(frame['uid'],bank)
    if pair:
        prompts.extend(pair['prompts'])
        choices.extend([pair['mark_set']]*2)
    return [(img,p) for p in prompts],choices


def shape_evidence(measure, mode):
    if mode!='shape_wrong':
        return measure
    return dict(measure,center=[round(1.4*x,2) for x in measure['center']],
                span=[round(1.4*x,2) for x in measure['span']])


def predict():
    sys.addaudithook(blind_targets)
    setup()
    import cv2
    import torch
    from conditioning import shape_prompt
    from infer import PROMPTS
    from model_ops import choices, generate, image, load, make_model, restore, state

    cv2.setNumThreads(1)
    rows=test_rows()
    for row in rows:
        hashes=dict(row['sensor_hashes'],rgb=row['rgb_sha'])
        assert all(digest(Path(row['frame'][name]))==expected for name,expected in hashes.items())
    path=RUN/'test-model-outputs.json'
    plan_hash=digest(RUN/'test-plan-seal.json')
    bank=read(path) if path.exists() else dict(plan=plan_hash,mode={},seconds={})
    assert bank['plan']==plan_hash
    ids={r['uid'] for r in rows}
    pr,model,limit,parameters=make_model(trainable=True)
    zero=state(parameters)

    def batched(jobs,maximum):
        if not jobs:
            return []
        try:
            return generate(pr,model,jobs,maximum)[0]
        except torch.cuda.OutOfMemoryError:
            if len(jobs)==1:
                raise
        torch.cuda.empty_cache()
        mid=len(jobs)//2
        return batched(jobs[:mid],maximum)+batched(jobs[mid:],maximum)

    def choose_batched(jobs, valid):
        if not jobs:
            return []
        try:
            return choices(pr,model,jobs,valid)
        except torch.cuda.OutOfMemoryError:
            if len(jobs)==1:
                raise
        torch.cuda.empty_cache()
        mid=len(jobs)//2
        return choose_batched(jobs[:mid],valid[:mid])+choose_batched(jobs[mid:],valid[mid:])

    def loops(name, infer, batch_size):
        result=bank['mode'].setdefault(name,{})
        bank['seconds'].setdefault(name,0.)
        remaining=[v for v in rows if v['uid'] not in result]
        for i in range(0,len(remaining),batch_size):
            sub=remaining[i:i+batch_size]
            when=perf_counter()
            responses=infer(sub)
            torch.cuda.synchronize()
            bank['seconds'][name]+=perf_counter()-when
            result.update((r['uid'],a) for r,a in zip(sub,responses,strict=True))
            publish(path,bank)
            if i==0 or i+batch_size>=len(remaining) or i%(batch_size*4)==0:
                print(name,len(result),'/',len(rows),round(bank['seconds'][name],1),flush=True)
        assert ids==set(result)

    for task in ('2d','3d'):
        loops('frozen_'+task,lambda batch:[readout(a,task) for a in batched([(image(r['frame']['rgb']),PROMPTS[task]+r['frame']['text']) for r in batch],limit)],16)
    evidence_path=RUN/'test-evidence.json'
    if not evidence_path.exists():
        work=[(r,{t:bank['mode']['frozen_'+t][r['uid']] for t in ('2d','3d')}) for r in rows]
        with ProcessPoolExecutor(10) as pool:
            ev={a['uid']:a for a in pool.map(physical,work,chunksize=3)}
        publish(evidence_path,dict(plan=plan_hash,evidence=ev))
    saved=read(evidence_path)
    assert saved['plan']==plan_hash and ids==set(saved['evidence'])
    ev=saved['evidence']
    for mode in MODEL_ARMS:
        if mode in ('joint_frozen','shape_frozen','graph_frozen'):
            restore(parameters,zero)
        else:
            checkpoint={'graph_sft':'graph','shape_wrong':'shape'}.get(mode,mode)
            load(parameters,checkpoint)
        if mode.startswith('graph'):
            def answer_group(sub):
                work=[graph_jobs(r['frame'],image(r['frame']['rgb']),ev[r['uid']]['bank']) for r in sub]
                ids=choose_batched([job for jobs,_ in work for job in jobs],[v for _,opts in work for v in opts])
                it=iter(ids)
                return [[next(it) for _ in jobs] for jobs,_ in work]
            loops(mode,answer_group,4)
        elif mode.startswith('shape'):
            def shapes(sub):
                work=[r for r in sub if ev[r['uid']]['measured'] is not None]
                jobs=[(image(r['frame']['rgb']),shape_prompt(r['frame'],shape_evidence(ev[r['uid']]['measured'],mode))) for r in work]
                predictions=dict(zip((r['uid'] for r in work),batched(jobs,64),strict=True))
                return [predictions.get(r['uid']) for r in sub]
            loops(mode,shapes,16)
        else:
            loops(mode,lambda sub:batched([(image(r['frame']['rgb']),JOINT+r['frame']['text']) for r in sub],120),16)
    assert all(set(x)==ids for x in bank['mode'].values())
    publish(RUN/'test-model-seal.json',dict(plan=plan_hash,payload=digest(path),evidence=digest(evidence_path),labels_read=False,
        weights=digest(RUN/'training-lock.json'),seconds=bank['seconds'],peak_cuda=torch.cuda.max_memory_allocated(),target_parameter_updates=0))


def raw_shape(frame, raw):
    from geometry import camera,to_sensor
    from score import canonical_box
    if raw['3d']['value'] is None:
        return None
    b=canonical_box(to_sensor(np.asarray(raw['3d']['value']),camera(frame)[2])[0])
    return np.r_[b[3:6],b[6]/np.pi].tolist()


def infer_boxes(task):
    row, model, evidence=task
    from conditioning import metric_pair,shape_into_base
    frame=row['frame']; uid=row['uid']
    raw={t:model['frozen_'+t] for t in ('2d','3d')}
    result={'frozen':backend(frame,raw)}
    size={'frozen':raw_shape(frame,raw)}
    coverage={}
    for a in ('joint_frozen','warm','source_sft','grpo'):
        parsed=replies(model[a])
        result[a]=backend(frame,parsed)
        coverage[a]=all(t['status']=='ok' for t in parsed.values())
    for a in ('shape_frozen','shape','shape_wrong'):
        parsed=shape_into_base(frame,model[a],raw) if model[a] is not None else None
        result[a]=backend(frame,parsed) if parsed is not None else result['frozen']
        v=valid_shape(model[a]) if model[a] is not None else None
        size[a]=None if v is None else v.tolist()
        coverage[a]=parsed is not None
    box={k:v['box'] for k,v in result.items()}
    bank=evidence['bank']
    pair=metric_pair(uid,bank)
    picks={}
    update={}
    for a in ('graph_frozen','graph_sft'):
        values=model[a]
        for i,b in enumerate(('som','true','wrong')):
            mark=values[i] if bank else 0
            picks[a+'_'+b]=mark
            box[a+'_'+b]=bank[mark-1]['completed'] if mark else None
        update[a]=dict(predicted=values[3:],expected=pair['correct']) if pair else None
    return dict(uid=uid,recording=row['recording'],target=row['target'],boxes=box,
                routes={k:v['route'] for k,v in result.items()},coverage=coverage,shape=size,
                picks=picks,pair=update,proposal_bank=bank)


def geometry():
    sys.addaudithook(blind_targets)
    setup()
    import cv2
    cv2.setNumThreads(1)
    rows=test_rows()
    receipt=read(RUN/'test-model-seal.json')
    assert not receipt['labels_read'] and receipt['plan']==digest(RUN/'test-plan-seal.json')
    assert receipt['payload']==digest(RUN/'test-model-outputs.json') and receipt['evidence']==digest(RUN/'test-evidence.json')
    outputs=read(RUN/'test-model-outputs.json')['mode']
    ev=read(RUN/'test-evidence.json')['evidence']
    jobs=[(r,{k:v[r['uid']] for k,v in outputs.items()},ev[r['uid']]) for r in rows]
    with ProcessPoolExecutor(10) as pool:
        pred=list(pool.map(infer_boxes,jobs,chunksize=2))
    p=RUN/'test-boxes.json'
    publish(p,pred)
    publish(RUN/'test-boxes-seal.json',dict(payload=digest(p),model=digest(RUN/'test-model-seal.json'),labels_read=False,n=len(pred)))
    print('All geometry and counterfactual outputs sealed',len(pred),flush=True)


def score():
    setup()
    members=test_rows()
    box_file=RUN/'test-boxes.json'
    receipt=read(RUN/'test-boxes-seal.json')
    assert not receipt['labels_read'] and receipt['payload']==digest(box_file) and receipt['model']==digest(RUN/'test-model-seal.json')
    sys.path.insert(0,str(HERE.parent/'3eed_2d_pose_finalization_20261128'))
    from run_mono import labels_for_panel
    sys.path.insert(0,str(HERE.parent/'3eed_hypothesis_verification_20261128'))
    sys.path.insert(0,str(HERE.parent/'3eed_motivation_next_20261128'))
    from score_experiment import get_truth
    from score import canonical_box
    from conditioning import target_mark
    from inference_stats import cluster_weights, intervals
    external,_=labels_for_panel(read(EXTERNAL/'mono-input-seal.json'))
    waymo={r['uid']:r['box'] for r in source_bank('val')}
    standard={x['uid']:dict(x,metadata=str((ORDINARY/x['metadata']).resolve())) for x in read(ORDINARY/'labels.json')}
    src={r['uid']:r for r in members}
    scored=[]
    for row in read(box_file):
        uid=row['uid']; domain=row['target']; frame=src[uid]['frame']
        if domain=='mono':
            gt=external[uid]['box']
            assert external[uid]['recording']==row['recording']
        elif domain=='waymo':
            gt=canonical_box(waymo[uid])
        else:
            _,_,gt=get_truth(frame,standard[uid])
        iou=truth_iou(row['boxes'],gt)
        truth_shape=gt[3:6]
        details={}
        for k,v in row['shape'].items():
            if v is not None:
                estimate=canonical_box(np.r_[np.zeros(3),v[:3],v[3]*np.pi])
                details[k]=[float(np.mean(abs(estimate[3:6]/truth_shape-1))),float(np.rad2deg(abs((estimate[6]-gt[6]+np.pi/2)%np.pi-np.pi/2)))]
        bank=row['proposal_bank']
        hits=[bool(target_mark([v],gt)) for v in bank]
        correct={k:bool(index and hits[index-1]) for k,index in row['picks'].items()}
        oracle_scores=truth_iou({str(i):v['completed'] for i,v in enumerate(bank)},gt)
        iou['proposal_oracle']=max(oracle_scores.values(),default=0.)
        scored.append(dict(uid=uid,target=domain,recording=row['recording'],iou=iou,coverage=row['coverage'],shape=details,
            graph_hits=correct,graph_coverage=any(hits),pair=row['pair'],routes=row['routes']))
    assert len(scored)==len(members)==receipt['n']

    def panel(group):
        weights=cluster_weights(group)
        metrics={}
        base=np.array([r['iou']['frozen'] for r in group]); base_ok=base>.5
        for name in group[0]['iou']:
            v=np.array([r['iou'][name] for r in group]); hit=v>.5
            metrics[name]=dict(miou=float(100*v.mean()),acc50=float(100*hit.mean()),
                delta_miou=float(100*(v-base).mean()),miou_ci=intervals(100*(v-base),weights),
                delta_acc50=float(100*(hit.astype(float)-base_ok).mean()),acc50_ci=intervals(100*(hit.astype(float)-base_ok),weights))
        contrasts={}
        for a,b in [('grpo','source_sft'),('grpo','warm'),('grpo','joint_frozen'),('shape','shape_frozen'),('shape','shape_wrong'),
                    ('graph_sft_true','graph_frozen_true'),('graph_sft_true','graph_sft_wrong'),('graph_sft_true','graph_sft_som')]:
            v=np.array([r['iou'][a] for r in group]); u=np.array([r['iou'][b] for r in group]);
            d=(v-u)*100; e=((v>.5).astype(float)-(u>.5))*100
            contrasts[a+'_vs_'+b]=dict(delta_miou=float(d.mean()),miou_ci=intervals(d,weights),
                    delta_acc50=float(e.mean()),acc50_ci=intervals(e,weights))
        shape={}
        for base_shape in ('frozen','shape_frozen','shape_wrong'):
            matched=[r for r in group if {base_shape,'shape'}<=r['shape'].keys()]
            if matched:
                y=np.array([r['shape'][base_shape] for r in matched]); x=np.array([r['shape']['shape'] for r in matched]); d=x-y
                w=cluster_weights(matched)
                shape[base_shape]=dict(n=len(matched),size_relative_delta=float(d[:,0].mean()),yaw_deg_delta=float(d[:,1].mean()),
                    size_ci=intervals(d[:,0],w),yaw_ci=intervals(d[:,1],w),size_medians=[float(np.median(v[:,0])) for v in (y,x)],
                    yaw_medians=[float(np.median(v[:,1])) for v in (y,x)])
        logic={}
        for a in ('graph_frozen','graph_sft'):
            pairs=[r['pair'][a] for r in group if r['pair'][a] is not None]
            logic[a]=dict(n=len(pairs),both_correct=sum(v['predicted']==v['expected'] for v in pairs),
                flips=sum(len(set(v['predicted']))==2 for v in pairs),first_correct=sum(v['predicted'][0]==v['expected'][0] for v in pairs),
                second_correct=sum(v['predicted'][1]==v['expected'][1] for v in pairs))
        return dict(n=len(group),recordings=len({(r['target'],r['recording']) for r in group}),metrics=metrics,contrasts=contrasts,
            available={a:sum(r['coverage'].get(a,False) for r in group) for a in GEOMETRY_ARMS[1:]},
            shape=shape, graph=dict(covered=sum(r['graph_coverage'] for r in group), selected={a:sum(r['graph_hits'][a] for r in group) for a in GRAPH_ARMS},logic=logic),
            routes={a:dict(Counter(r['routes'][a] for r in group)) for a in GEOMETRY_ARMS})
    out=RUN/'scores-private.json'
    publish(out,scored)
    panels={p:panel([r for r in scored if r['target']==p]) for p in ('waymo','drone','quad','mono')}
    panels['transfer']=panel([r for r in scored if r['target']!='waymo'])
    train={a:{k:v for k,v in read(RUN/f'{a}-receipt.json').items() if k not in ('trace','file')} for a in ADAPTERS}
    rollouts=read(RUN/'grpo-receipt.json')['trace']
    reward_all=[y for group in rollouts for y in group['rewards']]
    reward=dict(n=len(reward_all),valid=sum(x['valid'] for x in reward_all),
      nonzero_iou=sum(x['iou']>0 for x in reward_all),routes=dict(Counter(x['route'] for x in reward_all)),
      mean_iou=float(np.mean([x['iou'] for x in reward_all])))
    summary=dict(panels=panels,training=train,checkpoints_sealed_before_targets=True,source=read(RUN/'source-seal.json'),
        condition_examples=read(RUN/'conditions-receipt.json'),source_reward=reward,plan=read(RUN/'test-plan-seal.json'),
        prediction=read(RUN/'test-model-seal.json'),geometry=receipt,private_scores_sha=digest(out))
    publish(RUN/'results.json',summary)
    for p,group in panels.items():
        print(p,group['n'],{key:tuple(round(group['metrics'][key][k],2) for k in ('miou','acc50')) for key in (*GEOMETRY_ARMS,'graph_sft_true')},flush=True)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('lock','plan','predict','geometry','score'))
    globals()[p.parse_args().stage]()
