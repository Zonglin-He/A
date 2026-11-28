"""Seal method readouts before evaluating against the fixed benchmark references."""
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import re

import cv2
import numpy as np

from prepare import HERE,RUN,PLATFORMS,KMAX,FrameEvidence,check,digest,load_blind,publish,read
from infer_controls import ORDER,PREFIX,PREFIX_MODES,ICL_MODES,SELECT_MODES,start_number
from shared import canonical,score_boxes
from evaluate_structure import stat_panel
from inference_stats import cluster_weights, intervals
from truth_score_inputs import truth_data

GEN_MODES=(*PREFIX_MODES,*ICL_MODES,'graph_full')
CHOOSE_MODES=('window_pick',*SELECT_MODES)
RELATION=re.compile(r'\b(closest|nearest|farthest|furthest|behind|in front of|next to|left|right|between|beside|ahead|opposite|near|further|farther)\b',re.I)


def sealed_models():
    r=check(RUN/'gpu-state.json','artifact',RUN/'gpu-seal.json')
    p=check(RUN/'plan-private.json','input',RUN/'plan-seal.json')
    assert r['pin']['plan']==p['input'] and r['pin']['code']==digest(HERE/'infer_controls.py')
    assert p['protocol']==digest(HERE/'PROTOCOL_ZH.md')
    assert not r['target_labels_read'] and not r['checkpoint_training'] and not p['labels_read']
    return r,p,read(RUN/'gpu-state.json'),read(RUN/'plan-private.json')


def assemble(job):
    unit,row,model=job
    import live_2d

    ev=FrameEvidence(row['frame'])
    raw=row['raw']
    original = raw['3d']['value']
    native=canonical(ev.module.to_sensor(np.asarray(original),ev.mat)[0]) if original is not None else None
    formal=unit['formal']
    k=len(unit['proposals'][:KMAX])
    choices={'window_pick':unit['window_pick']}
    raw_variants={'original_native':None if native is None else native.tolist()}
    formulas={'formal':formal,'original_native':raw_variants['original_native']}
    parsed={}
    for name in GEN_MODES:
        result=model[name]
        value=result['value']
        parsed[name]=result['status']
        if value is None:
            formulas[name+'_native']=None
            formulas[name+'_backend']=formal if result['status']=='inapplicable' and name!='graph_full' else None
            raw_variants[name]=None
            continue
        upright=canonical(ev.module.to_sensor(np.asarray(value),ev.mat)[0]).tolist()
        raw_variants[name]=upright
        formulas[name+'_native']=upright
        if name=='graph_full':
            # This readout deliberately carries the selected physical instance forward.
            formulas[name+'_backend']=upright
        else:
            nxt=live_2d.predict(row['frame'],dict(raw,**{'3d':result}))
            formulas[name+'_backend']=nxt['box']
    for name in SELECT_MODES:
        choice=model[name]['selected']
        assert 0<=choice<=k
        choices[name]=choice
    for name in CHOOSE_MODES:
        choice=choices[name]
        p=unit['proposals'][choice-1] if choice else None
        formulas[name]=p['completed'] if p else None
        formulas[name+'_observed']=p['measured'] if p else None
    centers={}
    for name in ('tree_uniform','tree_physics'):
        response=model[name]
        if response['status']=='inapplicable':
            continue
        opts=unit['center_forms']['grid']
        matched=[i for i,opt in enumerate(opts) if response['raw'].startswith(PREFIX+start_number(opt))]
        assert len(matched)==1,(name,response['status'])
        i=matched[0]
        if response['value'] is not None:
            assert np.allclose(response['value'][:3],opts[i],rtol=0.,atol=1e-8)
        centers[name]=dict(grid_index=i,
            distance_m=float(np.linalg.norm(np.asarray(opts[i])-unit['center_forms']['metric'])))
    rgb=row['frame']['text']
    return dict(uid=unit['uid'],target=row['target'],recording=row['recording'],formulas=formulas,
                raw=raw_variants,parsed=parsed,choices=choices,center_choices=centers,
                relation=bool(RELATION.search(rgb)),prefix_eligible=unit['center_forms'] is not None)


def seal_geometry():
    gpu,pin,bank,plan=sealed_models()
    src,_,_,_,_=load_blind()
    cv2.setNumThreads(1)
    work=[(unit,src[unit['uid']],{name:bank['answers'][name][unit['uid']] for name in ORDER}) for unit in plan['units']]
    with ProcessPoolExecutor(8) as pool:
        rows=list(pool.map(assemble,work,chunksize=3))
    assert len(rows)==96
    path=RUN/'geometry-private.json'
    publish(path,rows)
    publish(RUN/'geometry-seal.json',dict(artifact=digest(path),code=digest(Path(__file__)),gpu=gpu['artifact'],
                    donor_and_selection_plan=pin['input'],n=len(rows),target_labels_read=False))
    print('Sealed',len(rows),'rows with all geometry readouts',flush=True)


def membership(proposal,truth):
    xyz=np.array(proposal['points'],dtype=float)-truth[:3]
    c,s=np.cos(truth[6]),np.sin(truth[6])
    local=np.c_[c*xyz[:,0]+s*xyz[:,1],-s*xyz[:,0]+c*xyz[:,1],xyz[:,2]]
    selected=(np.abs(local)<=(truth[3:6]/2+1e-8)).all(axis=1)
    return bool(selected.sum()>=5 and selected.mean()>=.5)


def describe(xs):
    q=list(xs)
    if not q:
        return dict(n=0)
    v=np.asarray(q)
    return dict(n=len(v),mean=float(v.mean()),median=float(np.median(v)))


def paired(rows, value, a, b):
    have=[r for r in rows if a in r[value] and b in r[value]]
    if not have:
        return dict(n=0)
    dl=100*np.array([r[value][a]-r[value][b] for r in have])
    return dict(n=len(have),delta_pp=float(dl.mean()),ci=intervals(dl,cluster_weights(have)))


def paired_iou(rows, a, b):
    weights=cluster_weights(rows)
    x=np.array([r['iou'][a] for r in rows])
    y=np.array([r['iou'][b] for r in rows])
    iou=100*(x-y)
    acc=100*((x>.50).astype(float)-(y>.50))
    return dict(n=len(rows),delta_pp=float(iou.mean()),ci=intervals(iou,weights),
                acc50_delta_pp=float(acc.mean()),acc50_ci=intervals(acc,weights))


def rank_report(rows):
    result={}
    all_matched=[x for x in rows if x['prefix_eligible']]
    references={
        'prefix_format':'icl_zero', 'prefix_native':'prefix_format', 'prefix_metric':'prefix_native',
        'prefix_wrong':'prefix_native', 'tree_uniform':'prefix_format', 'tree_physics':'tree_uniform',
        'icl_zero':'original_native', 'icl_random':'icl_zero', 'icl_verified':'icl_zero',
        'icl_cross':'icl_zero', 'graph_full':'icl_zero',
    }
    for name,base in references.items():
        eval_rows=all_matched if name in PREFIX_MODES else rows
        joint=[r for r in eval_rows if name in r['shape'] and base in r['shape']]
        result[name]=dict(available=sum(name in r['shape'] for r in eval_rows),eligible=len(eval_rows),
                          size_error=describe(r['shape'][name]['size'] for r in eval_rows if name in r['shape']),
                          yaw_deg=describe(r['shape'][name]['yaw'] for r in eval_rows if name in r['shape']),
                          center_m=describe(r['shape'][name]['center'] for r in eval_rows if name in r['shape']),
                          versus=base,matched=len(joint))
        if joint:
            weight=cluster_weights(joint)
            metrics={}
            for m in ('size','yaw','center'):
                d=np.array([r['shape'][name][m]-r['shape'][base][m] for r in joint])
                metrics[m]=dict(mean_delta=float(d.mean()),ci=intervals(d,weight),
                    arm_median=float(np.median([r['shape'][name][m] for r in joint])),
                    reference_median=float(np.median([r['shape'][base][m] for r in joint])))
            result[name]['paired_errors']=metrics
    return result


def panel(rows):
    if not rows:
        return dict(n=0)
    names=list(rows[0]['iou'])
    base=stat_panel(rows,names,'formal')
    base['rereads']=rank_report(rows)
    base['rank_selection']=dict(
        oracle_identity_complete=sum(r['coverage']['all'] for r in rows),
        oracle_identity_displayed=sum(r['coverage']['displayed'] for r in rows),
        complete_count=sum(r['coverage']['count'] for r in rows),
        readable_proposals=describe(r['coverage']['shown'] for r in rows),
        selected={k:dict(correct=sum(r['identity'][k] for r in rows),
                         correct_given_recall=sum(r['identity'][k] for r in rows if r['coverage']['displayed']),
                         abstention=sum(r['choices'][k]==0 for r in rows))
                  for k in CHOOSE_MODES},
        permuted_same=sum(r['choices']['som']==r['choices']['som_permuted'] for r in rows),
        paired_graph_vs_som=paired(rows,'identity','graph','som'),
        paired_graph_vs_shuffled=paired(rows,'identity','graph','graph_shuffled'),
        paired_som_vs_window=paired(rows,'identity','som','window_pick'))
    base['parses']={m:dict(Counter(r['parsed'][m] for r in rows)) for m in GEN_MODES}
    pairs=dict(
       p1_metric_native=('prefix_metric','prefix_native'), p1_metric_wrong=('prefix_metric','prefix_wrong'),
       p1_tree_physics=('tree_physics','tree_uniform'),
       p3_verified_zero=('icl_verified','icl_zero'),p3_verified_random=('icl_verified','icl_random'),
       p3_verified_cross=('icl_verified','icl_cross'),
       p2_som_window=('som','window_pick'),p4_graph_som=('graph','som'),
       p4_graph_counterfactual=('graph','graph_shuffled'),p4_full_readout=('graph_full_backend','graph'))
    for tag,(a,b) in pairs.items():
        if a in GEN_MODES and b in GEN_MODES:
            base.setdefault('paired_arms',{})[tag]={'native':paired_iou(rows,a+'_native',b+'_native'),
                'backend':paired_iou(rows,a+'_backend',b+'_backend')}
        else:
            base.setdefault('paired_arms',{})[tag]=paired_iou(rows,a,b)
    both=[r for r in rows if set(r['tree'])=={'tree_uniform','tree_physics'}]
    base['tree_center']=dict(matched=len(both),changed=sum(r['tree']['tree_uniform']['grid_index']!=r['tree']['tree_physics']['grid_index'] for r in both),
      offset={m:describe(r['tree'][m]['distance_m'] for r in both) for m in ('tree_uniform','tree_physics')})
    return base


def score():
    from metric_image_common import inputs,setup

    model, pin, gpu,plan=sealed_models()
    labelsafe=RUN/'geometry-private.json'
    geom=check(labelsafe,'artifact',RUN/'geometry-seal.json')
    assert geom['gpu']==model['artifact'] and geom['donor_and_selection_plan']==pin['input']
    assert geom['code']==digest(Path(__file__)) and not geom['target_labels_read']
    setup()
    source=inputs()
    requested={v['uid']:source[v['uid']] for v in plan['units']}
    gt,_=truth_data(requested)
    units={r['uid']:r for r in plan['units']}
    previous=RUN.parent/'2d-relocalization-20261128'
    p=check(previous/'scored.json','scored',previous/'score-seal.json')
    old={v['uid']:v for v in read(previous/'scored.json') if v['group']=='main'}
    rows=[]
    for r in read(labelsafe):
        uid=r['uid']
        truth=gt[uid]['box']
        props=units[uid]['proposals']
        n=min(KMAX,len(props))
        flags=[membership(v,truth) for v in props]
        values=score_boxes(r['formulas'],truth)
        iou={k:v['iou'] for k,v in values.items()}
        assert abs(iou['formal']-old[uid]['pool']['cray']['iou']['cached'])<1e-10
        potential=score_boxes({str(c+1):q['completed'] for c,q in enumerate(props)},truth)
        scores=[v['iou'] for v in potential.values()]
        iou.update(oracle_pool=max(scores,default=0.),oracle_displayed=max(scores[:n],default=0.))
        shape={name:{k:values[name+'_native'][k] for k in ('size','yaw','center')}
               for name in GEN_MODES if 'size' in values[name+'_native']}
        if 'size' in values['original_native']:
            shape['original_native']={k:values['original_native'][k] for k in ('size','yaw','center')}
        identity={arm:float(mark>0 and flags[mark-1]) for arm,mark in r['choices'].items()}
        rows.append(dict(uid=uid,target=r['target'],recording=r['recording'],relation=r['relation'],
            iou=iou,shape=shape,choices=r['choices'],identity=identity,parsed=r['parsed'],tree=r['center_choices'],
            prefix_eligible=r['prefix_eligible'],coverage=dict(all=any(flags),displayed=any(flags[:n]),shown=n,count=len(props))))
    assert len(rows)==96 and Counter(r['target'] for r in rows)=={p:32 for p in PLATFORMS}
    scope={'all':rows,'relation':[r for r in rows if r['relation']],
           'not_relation':[r for r in rows if not r['relation']],
           'prefix_eligible':[r for r in rows if r['prefix_eligible']]}
    for platform in PLATFORMS:
        scope[platform]=[r for r in rows if r['target']==platform]
    score_file=RUN/'scores-private.json'
    publish(score_file,rows)
    donor={}
    for platform,v in plan['donors'].items():
        donor[platform]={k:{'n':len(v[k]),'strict_n':sum(x['strict'] for x in v[k]),
            'members':[{q:x[q] for q in ('uid','platform','strict','iou2d','audit','source')} for x in v[k]]}
                          for k in ('random','verified','cross')}
        donor[platform].update(eligible=v['eligible'],strict_eligible=v['strict_eligible'])
    result=dict(n=len(rows),panels={k:panel(v) for k,v in scope.items()},donors=donor,
                compute=gpu['bench'],model_seal=model,geometry=geom,score=digest(score_file),
                no_external_model=True,labels_read_only_after_geometry=True,training=False)
    publish(RUN/'aggregate.json',result)
    print('All four directions scored for',len(rows),'queries.',flush=True)
    all=result['panels']['all']
    print('identity',all['rank_selection'],flush=True)
    for m in ('formal','original_native','prefix_native_native','prefix_metric_native','tree_uniform_native','tree_physics_native','icl_zero_native','icl_random_native','icl_verified_native','icl_cross_native','window_pick','som','graph','graph_full_backend'):
        print(m,all['metrics'][m]['miou'],all['metrics'][m]['acc50'],flush=True)


if __name__=='__main__':
    ap=ArgumentParser()
    ap.add_argument('stage',choices=('geometry','score'))
    {'geometry':seal_geometry,'score':score}[ap.parse_args().stage]()
