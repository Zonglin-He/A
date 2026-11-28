"""Physical conditions, natural SoM examples, and paired metric interventions."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from bridge import CFG, DATA, HERE, RUN, SHAPE, compact, digest, publish, read, readout, setup, source_bank, source_guard, valid_shape
from model_ops import generate, image, make_model
from source_data import camera_raw, stable

sys.path.insert(0,str(HERE.parent/'3eed_inference_time_vlm_20261128'))
from prepare import FrameEvidence, KMAX, proposal_bank
from infer_controls import describe_graph, marked


def condition(frame, response):
    from geometry import camera, cell_points, raw_returns, rigid_xyz

    if response['value'] is None:
        return None
    spec,size,transform=camera(frame)
    uv,returns=raw_returns(frame,spec,size)
    window=np.asarray(response['value'])*np.tile(size,2)/1000
    fit,_=cell_points(uv,returns,window)
    if len(fit)<5:
        return None
    depth=np.median(fit[:,2])
    keep=abs(fit[:,2]-depth)<=max(.5,min(2.,1.4826*np.median(abs(fit[:,2]-depth))))
    support=rigid_xyz(fit[keep],np.linalg.inv(transform))
    if len(support)<5:
        return None
    center=np.median(support,axis=0)
    span=np.diff(np.quantile(support,[.05,.95],axis=0),axis=0)[0]
    return dict(center=center.round(2).tolist(),span=span.round(2).tolist(),points=len(support))


def shape_prompt(frame, measured):
    return SHAPE+frame['text']+'\nLiDAR XYZ median of visible returns and XYZ visible span in metres: '+compact(measured)


def shape_into_base(frame, value, base):
    from geometry import camera, to_sensor
    from score import canonical_box

    shape=valid_shape(value)
    if shape is None or base['3d']['value'] is None:
        return None
    transform=camera(frame)[2]
    native=np.asarray(base['3d']['value'],float)
    center=to_sensor(native,transform)[0][:3]
    sensor=canonical_box(np.r_[center,shape[:3],shape[3]*np.pi])
    camera_box=camera_raw(sensor,transform)
    camera_box[:3]=native[:3]
    import json
    from bridge import FENCE
    entry=json.loads(FENCE.sub('',base['3d']['raw'].strip()))
    label=(entry[0] if isinstance(entry,list) else entry).get('label','object')
    return dict(base,**{'3d':readout(compact([dict(bbox_3d=camera_box.tolist(),label=label)]),'3d')})


def proposals(frame, base3=None):
    evidence=FrameEvidence(frame)
    bank,_=proposal_bank(dict(raw={'3d':base3 or {'value':None}}),evidence)
    return bank[:KMAX]


def graph_prompt(frame, bank, *, with_graph=True):
    msg=('Select the numeric mark for the single described object. Return one number only, '
         '0 if none matches. Available marks: '+', '.join(map(str,range(1,len(bank)+1)))+'.')
    if with_graph:
        msg+='\n'+describe_graph(bank)
    return msg+'\nDescription: '+frame['text']


def metric_pair(uid, bank):
    pairs=[(i,j) for i,x in enumerate(bank) for j,y in enumerate(bank) if i<j and abs(x['center_camera'][2]-y['center_camera'][2])>=1.]
    if not pairs:
        return None
    i,j=min(pairs,key=lambda pair:stable(f'metric-pair|{uid}|{pair}'))
    again=list(bank)
    again[i],again[j]=again[j],again[i]
    direction='farther' if int(stable('relation'+uid)[0],16)%2 else 'closer'
    expected=(i if bank[i]['center_camera'][2]>bank[j]['center_camera'][2] else j) if direction=='farther' else (i if bank[i]['center_camera'][2]<bank[j]['center_camera'][2] else j)
    wrong=j if expected==i else i
    def prompt(graph):
        return (f'Using the recorded depth in the scene graph, choose mark {i+1} or {j+1} '
                f'that is {direction} from the camera. Return the chosen number only.\n'+describe_graph(graph))
    return dict(mark_set=[i+1,j+1],prompts=[prompt(bank),prompt(again)],correct=[expected+1,wrong+1])


def target_mark(bank, truth):
    xyz0=np.asarray(truth[:3]); dims=np.asarray(truth[3:6])*.5
    c,s=np.cos(truth[6]),np.sin(truth[6]); matches=[]
    for i,p in enumerate(bank,1):
        xyz=np.asarray(p['points'])-xyz0
        local=np.c_[c*xyz[:,0]+s*xyz[:,1],-s*xyz[:,0]+c*xyz[:,1],xyz[:,2]]
        inside=(abs(local)<=dims+1e-8).all(1)
        if inside.sum()>=5 and inside.mean()>=.5:
            matches.append((inside.sum(),inside.mean(),i))
    return max(matches)[2] if matches else 0


def prepare_examples(task):
    row, raw2=task
    frame=row['frame']
    examples={'shape':[],'graph':[]}
    measure=condition(frame,raw2)
    if measure is not None:
        truth=row['box']
        answer=compact(dict(shape=np.round([*truth[3:6],truth[6]/np.pi],2).tolist()))
        examples['shape'].append(dict(uid=row['uid'],rgb=frame['rgb'],prompt=shape_prompt(frame,measure),answer=answer))
    bank=proposals(frame)
    if bank:
        marks=[v['mark'] for v in bank]
        examples['graph'].append(dict(uid=row['uid'],rgb=frame['rgb'],marks=marks,
                         prompt=graph_prompt(frame,bank),answer=str(target_mark(bank,row['box'])),kind='natural'))
        pair=metric_pair(row['uid'],bank)
        if pair is not None:
            for prompt,answer in zip(pair['prompts'],pair['correct'],strict=True):
                examples['graph'].append(dict(uid=row['uid'],rgb=frame['rgb'],marks=marks,prompt=prompt,
                                               answer=str(answer),kind='metric_counterfactual'))
    return examples


def conditioned_example(row):
    img=image(row['rgb'])
    if 'marks' in row:
        img=marked(img,[dict(mark=x) for x in row['marks']],list(range(1,len(row['marks'])+1)))
    return img,row['prompt'],row['answer']


def source_examples(name):
    assert name in ('shape','graph')
    receipt=read(RUN/'conditions-receipt.json')
    assert receipt['source']==digest(RUN/'source-seal.json') and receipt['code']==digest(Path(__file__))
    path=RUN/f'examples-{name}.json'
    assert digest(path)==receipt[name]['file']
    return read(path)


def prepare_source():
    source_guard()
    setup()
    import cv2
    cv2.setNumThreads(1)
    from infer import PROMPTS
    rows=source_bank('train')
    processor,model,maximum,_=make_model(trainable=False)
    base_path=RUN/'source-frozen-2d.json'
    base=read(base_path) if base_path.exists() else {}
    start=perf_counter()
    for offset in range(0,len(rows),8):
        part=[r for r in rows[offset:offset+8] if r['uid'] not in base]
        if part:
            jobs=[(image(r['frame']['rgb']),PROMPTS['2d']+r['frame']['text']) for r in part]
            generated,_=generate(processor,model,jobs,maximum)
            base.update((r['uid'],readout(a,'2d')) for r,a in zip(part,generated,strict=True))
            publish(base_path,base)
        if offset%32==0 or offset+8>=len(rows):
            print('frozen source 2D',len(base),'/',len(rows),round(perf_counter()-start,1),flush=True)
    assert {r['uid'] for r in rows}==set(base)
    with ProcessPoolExecutor(10) as pool:
        data=list(pool.map(prepare_examples,[(r,base[r['uid']]) for r in rows],chunksize=4))
    budget=CFG['sft_steps']*CFG['sft_batch']
    natural=[v for r in data for v in r['graph'] if v['kind']=='natural']
    pairs=[r['graph'][1:] for r in data if len(r['graph'])==3]
    natural.sort(key=lambda v:stable('graph-natural|'+v['uid']))
    pairs.sort(key=lambda v:stable('graph-counterfactual|'+v[0]['uid']))
    # One complete pass includes both answers for every selected counterfactual pair.
    graph=natural[:budget//2]+[item for pair in pairs[:budget//4] for item in pair]
    assert len(graph)==budget
    result={}
    for name in ('shape','graph'):
        items=[v for r in data for v in r['shape']] if name=='shape' else graph
        assert items and all(Path(v['rgb']).is_relative_to(DATA/'waymo') for v in items)
        path=RUN/f'examples-{name}.json'
        publish(path,items)
        result[name]=dict(n=len(items),file=digest(path),kinds=dict(Counter(r.get('kind',name) for r in items)))
    result['graph'].update(available_natural=len(natural),available_pairs=len(pairs),used_pairs=budget//4)
    publish(RUN/'conditions-receipt.json',dict(**result,source=digest(RUN/'source-seal.json'),
                code=digest(Path(__file__)),target_read=False,frozen_2d=digest(base_path),seconds=perf_counter()-start))
    print(result,flush=True)


if __name__=='__main__':
    prepare_source()
