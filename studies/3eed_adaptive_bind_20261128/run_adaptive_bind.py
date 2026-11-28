"""Label-blind matched A/B/C binding replay followed by separate offline scoring."""
import argparse
from collections import Counter
from contextlib import ExitStack
from itertools import permutations
from pathlib import Path
import sys
from time import perf_counter_ns
from unittest.mock import patch

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent/'3eed_minimal_verifier_20261128'
sys.path.insert(0,str(PAPER))
from provenance import ROOT, RUN as OLD_RUN, digest, read, retain, runtime, write
from cache_mono import guard
from eval_minimal_verifier import dev_jobs,development_labels,external_labels,mono_jobs
from lean_geometry import describe,reduced_pool
from prepare_blind import prepare
from score_paradigms import ious
from adaptive_regions import ARMS,ranked

RUN = ROOT/'runs/adaptive-bind-20261128'
FILES = ('adaptive_regions.py','run_adaptive_bind.py','PROTOCOL_ZH.md')
ORDERS = tuple(permutations(ARMS))


def code_stamp():
    return {n:digest(HERE/n) for n in FILES}


class Trace:
    """Memoize one graph per actual binding input and record Fit memberships."""
    def __init__(self,arm,optical,window):
        self.arm = arm
        self.optical = optical
        self.window = window
        self.stages = {}
        self.elapsed = 0

    def rank(self,sample,terrain,scales,returns,window,selection=None):
        stage = 'H' if selection is None else 'C_ray'
        if stage not in self.stages:
            assert np.array_equal(window,self.window)
            start = perf_counter_ns()
            result = ranked(self.arm,sample,terrain,scales,returns,window,selection,self.optical)
            self.elapsed += perf_counter_ns()-start
            self.stages[stage] = (result,sample,selection)
        previous,points,held = self.stages[stage]
        assert sample is points and (selection is None or np.array_equal(held,selection))
        return previous.regions

    def choose(self,*args):
        regions = self.rank(*args)
        return (regions[0].points,regions[0].selection) if regions else (None,None)

    def summary(self,branch):
        used = self.stages.get(branch)
        return dict(members=used[0].members if used else [],
                    seed_sizes=[len(r.points) for r in used[0].regions] if used else [],
                    attempts={s:dict(components=v.count,eligible=v.admitted) for s,(v,_,_) in self.stages.items()},
                    bind_ns=self.elapsed)


def cache_geometry(stack,g):
    # A frame is held fixed while all three generators are evaluated.
    for name in ('camera','raw_returns','fit_sample','model_sample'):
        fn,cache = getattr(g,name),{}
        def one(*args,_fn=fn,_cache=cache):
            if not _cache:
                _cache[0] = _fn(*args)
            return _cache[0]
        stack.enter_context(patch.object(g,name,one))
    terrain = g.regions.terrain
    ground = {}
    def one_terrain(sample):
        key = id(sample)
        if key not in ground:
            ground[key] = terrain(sample)
        return ground[key]
    stack.enter_context(patch.object(g.regions,'terrain',one_terrain))


def common_fit(g,frame,replies,spec,size,transform):
    value = replies['2d']['value']
    if value is None:
        return np.empty((0,3)),np.empty(0,bool)
    uv,returns = g.raw_returns(frame,spec,size)
    ids,cells = g.z_buffer(uv,returns,np.arange(len(uv)))
    ids = ids[g.reserve(cells)==0]
    window = np.asarray(value)*np.tile(size,2)/1000.
    return g.rigid_xyz(returns[ids],np.linalg.inv(transform)),g.inside(uv[ids],window)


def regenerate(frame,replies,target,trace,cap):
    import compact_pool
    import minimal

    with patch.object(minimal,'choose_region',trace.choose),patch.object(compact_pool,'ranked_regions',trace.rank):
        # Audit mode fixes the requested held-return list across the three calls.
        anchor,_ = minimal.predict(frame,replies,audit=True)
        row = dict(frame=frame,raw=replies,target=target)
        case = prepare((frame['uid'],row,dict(outputs=dict(axis_2d=anchor))))
        pool,evidence = reduced_pool(case,frame,replies,cap)
    return pool,evidence,anchor


def run_query(row,original,case,order,cap,choose):
    from minimal import geometry as g

    frame,replies = row['frame'],row['raw']
    outputs = {}
    with ExitStack() as stack:
        cache_geometry(stack,g)
        spec,size,transform = g.camera(frame)
        optical = g.model_to_camera_matrix(spec)[2]
        xyz,window = common_fit(g,frame,replies,spec,size,transform)
        for arm in order:
            trace = Trace(arm,optical,window)
            pool,evidence,anchor = regenerate(frame,replies,row['target'],trace,cap)
            assert len(pool)<=cap
            selected = choose(pool,evidence)
            outputs[arm] = dict(pool=pool,choice=selected,anchor=anchor,
                                binding=trace.summary(anchor['readout']))
    control = outputs['C_four']
    previous_pool = original['pool']
    previous_kept = np.flatnonzero(retain(previous_pool)[:,1])
    assert control['pool']==[previous_pool[i]['box'] for i in previous_kept],frame['uid']
    assert control['pool'][control['choice']]==previous_pool[original['choice']['S5-pass']]['box'],frame['uid']
    if case is not None:
        assert control['anchor']==case['base']
    used = np.unique(np.r_[np.flatnonzero(window),np.asarray(
        [i for value in outputs.values() for members in value['binding']['members'] for i in members],int)])
    for value in outputs.values():
        value['binding']['members'] = [np.searchsorted(used,ids).tolist() for ids in value['binding']['members']]
    return dict(uid=frame['uid'],target=row['target'],fit=xyz[used].tolist(),window=window[used].tolist(),arms=outputs)


def predict():
    sys.addaudithook(guard)
    runtime()
    from run_qwen_3eed import PAPER_REGISTRATION,checked_file,registered_predictor
    from s5_paper_entry import _verifier

    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    cv2.setNumThreads(1)
    method = read(PAPER_REGISTRATION)
    assert method['method']=='s5_pass'
    predict_locked = registered_predictor(method)
    config = predict_locked.keywords
    model = _verifier(config['checkpoint'],config['cap'])
    cap,threshold = config['cap'],config['threshold']

    @torch.inference_mode()
    def choose(pool,evidence):
        count = len(pool)
        if count==1:
            return 0
        x,e = describe(evidence,pool)
        nodes = np.zeros((1,cap,14),np.float32)
        edges = np.zeros((1,cap,cap,8),np.float32)
        nodes[0,:count],edges[0,:count,:count] = x,e
        mask = (np.arange(cap)<count)[None]
        inp = (torch.as_tensor(a,device='cuda') for a in (nodes,edges,mask))
        utility = model(*inp)[0,1:count]
        best,index = utility.max(0)
        return int(index)+1 if float(best)>threshold else 0

    RUN.mkdir(parents=True,exist_ok=True)
    lock = read(checked_file(method['method_lock']))
    for scope,(jobs,pins) in (('development',dev_jobs()),('mono',mono_jobs())):
        previous = OLD_RUN/'passed-stack'/f'{scope}-predictions-private.json'
        stamp = read(OLD_RUN/'passed-stack'/f'{scope}-seal.json')
        assert stamp['payload']==digest(previous) and not stamp['labels_read']
        expected = {r['uid']:r for r in read(previous)}
        out = []
        started = perf_counter_ns()
        for j,(case,row,_) in enumerate(jobs):
            uid = row['frame']['uid']
            out.append(run_query(row,expected[uid],case,ORDERS[j%len(ORDERS)],cap,choose))
            if (j+1)%100==0 or j+1==len(jobs):
                print(scope,j+1,'/',len(jobs),flush=True)
        file = RUN/f'{scope}-predictions-private.json'
        write(file,out)
        write(RUN/f'{scope}-seal.json',dict(n=len(out),payload=digest(file),inputs=pins,
              previous=stamp['payload'],method=digest(PAPER_REGISTRATION),checkpoint=lock['checkpoint']['sha256'],
              labels_read=False,code=code_stamp(),seconds=(perf_counter_ns()-started)/1.e9))


def support_measures(members,inside,window):
    denom = (inside&window).sum()
    rows = []
    for indices in members:
        return_ids = np.asarray(indices,int)
        positives = inside[return_ids]
        rows.append(dict(purity=float(positives.mean()),coverage=float((positives&window[return_ids]).sum()/denom)
                         if denom else None))
    return rows


def query_score(entry,truth):
    from oracle_geometry import inside

    box = np.asarray(truth['box'])
    valid = inside(np.asarray(entry['fit']).reshape(-1,3),box)
    window = np.asarray(entry['window'],bool)
    result = dict(uid=entry['uid'],target=entry['target'],recording=truth['recording'],
                  roi_eligible=bool((valid&window).any()),iou={},anchor={},oracle={},center={},
                  supports={},branches={},counts={},switches={},bind_ns={},attempted={})
    for name,arm in entry['arms'].items():
        pool,choice = arm['pool'],arm['choice']
        values = ious(pool,box)
        result['iou'][name] = float(values[choice])
        result['anchor'][name] = float(values[0])
        result['oracle'][name] = float(values.max())
        chosen = pool[choice]
        result['center'][name] = None if chosen is None else float(np.linalg.norm(np.asarray(chosen[:3])-box[:3]))
        bind = arm['binding']
        result['supports'][name] = support_measures(bind['members'],valid,window)
        result['branches'][name] = arm['anchor']['readout']
        result['counts'][name] = len(pool)
        result['switches'][name] = choice!=0
        result['bind_ns'][name] = bind['bind_ns']
        result['attempted'][name] = bind['attempts']
    return result


def summarize(rows):
    from evaluate_structure import stat_panel

    summary = stat_panel(rows,ARMS,'C_four')
    for key in ('anchor','oracle'):
        summary[key] = stat_panel([dict(r,iou=r[key]) for r in rows],ARMS,'C_four')['metrics']
    common = [r for r in rows if all(r['center'][arm] is not None for arm in ARMS)]
    summary['valid_box_center_error_m'] = dict(common=len(common),arms={
        a:dict(n=sum(r['center'][a] is not None for r in rows),
               median=float(np.median([r['center'][a] for r in common])) if common else None)
        for a in ARMS})
    valid = [r for r in rows if r['roi_eligible']]
    quality = {}
    for a in ARMS:
        observed = [r['supports'][a] for r in valid]
        matched = [p[0] for p in observed if p]
        good = lambda p:p['purity']>=.70 and p['coverage']>=.70
        quality[a] = dict(n_with_support=len(matched),top1_qualified=sum(good(p) for p in matched),
                          top3_qualified=sum(any(good(v) for v in r) for r in observed),
                          mean_purity=float(np.mean([r['purity'] for r in matched])) if matched else None,
                          mean_coverage=float(sum(r['coverage'] for r in matched)/len(valid)) if valid else None)
    summary['fit_cell_support'] = dict(roi_eligible=len(valid),arms=quality)
    summary['routes'] = {a:dict(Counter(r['branches'][a] for r in rows)) for a in ARMS}
    summary['replacements'] = {a:sum(r['switches'][a] for r in rows) for a in ARMS}
    summary['pool_size_mean'] = {a:float(np.mean([r['counts'][a] for r in rows])) for a in ARMS}
    # Every attempted stage is decided by the frozen frontend; pair on all three reaching Bind.
    paired = [r for r in rows if all(r['attempted'][a] for a in ARMS)]
    summary['bind_ms'] = dict(n=len(paired),arms={a:dict(
        median=float(np.median([r['bind_ns'][a] for r in paired]))/1.e6 if paired else None,
        mean=float(np.mean([r['bind_ns'][a] for r in paired]))/1.e6 if paired else None,
        invocations=sum(len(r['attempted'][a]) for r in paired),
        mean_components=float(np.mean([sum(v['components'] for v in r['attempted'][a].values())
                                       for r in paired])) if paired else None) for a in ARMS})
    return summary


def score():
    from shared import setup
    setup()
    sys.path.insert(0,str(HERE.parent/'3eed_qwen_cray_identifiability_audit_20261121'))
    torch.set_num_threads(1)
    inputs = {}
    for scope in ('development','mono'):
        file = RUN/f'{scope}-predictions-private.json'
        stamp = read(RUN/f'{scope}-seal.json')
        assert stamp['payload']==digest(file) and stamp['code']==code_stamp() and not stamp['labels_read']
        assert stamp['method']==digest(PAPER.parents[1]/'QWEN_PAPER_METHOD.json')
        inputs[scope] = read(file),stamp
    output = {}
    primary = read(OLD_RUN/'passed-stack/results.json')
    for scope,truth in (('development',development_labels()),('mono',external_labels())):
        rows,stamp = inputs[scope]
        detail = [query_score(r,truth[r['uid']]) for r in rows if r['uid'] in truth]
        assert len(detail)==len(truth)==(665 if scope=='development' else 148)
        by_target = {'overall':detail}
        for name in sorted({r['target'] for r in detail}):
            by_target[name] = [r for r in detail if r['target']==name]
        panels = {name:summarize(group) for name,group in by_target.items()}
        control = primary['target'][scope]['panel']['overall']['metrics']['S5-pass']
        assert all(abs(panels['overall']['metrics']['C_four'][k]-control[k])<1.e-10 for k in ('miou','acc50'))
        file = RUN/f'{scope}-scores-private.json'
        write(file,detail)
        output[scope] = dict(panels=panels,prediction=stamp,scores_sha256=digest(file),labels_after_seal=True)
        for name,d in panels.items():
            print(scope,name,len(by_target[name]),[(k,round(v['miou'],3),round(v['acc50'],3))
                                       for k,v in d['metrics'].items()],flush=True)
    write(RUN/'results.json',dict(target=output,code=code_stamp(),no_training=True))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=('predict','score'))
    dict(predict=predict,score=score)[parser.parse_args().phase]()
