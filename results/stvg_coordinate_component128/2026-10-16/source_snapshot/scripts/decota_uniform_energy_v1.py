"""GT-free collection and prediction for uniform/absolute and minimax/energy.

Reuses sealed temporal episodes and frozen expert proposals only when source,
frame grid, prompt and content hashes match. No temporal hyperparameter changes.
"""
import argparse
import copy
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,save,load,sha,status
from scripts import decota_heuristic_study_v1 as old
from scripts.tune_decota_refine_v1 import configure,anchors,SPATIAL
from methods.decota_refine_uniform_v1.api import select_positions,minimax_positions,coverage_radius,reconstruct

OUT=ROOT/'artifacts/decota_uniform_energy_v1'
BS=old.BS;GS=old.GS;SPLITS=old.SPLITS
FAMILIES=[s+'_'+m for s in ['current','uniform','minimax'] for m in ['residual','absolute']]


def prepare():
    if (OUT/'lock.json').exists():return
    p=old.plan()
    files=['methods/decota_refine_uniform_v1/__init__.py','methods/decota_refine_uniform_v1/api.py',
        'methods/decota_refine_uniform_v1/predictor.py','scripts/decota_uniform_energy_v1.py',
        'scripts/score_decota_uniform_energy_v1.py','tests/test_decota_refine_uniform_v1.py']
    temporal=[]
    for b in BS:
        for g in GS:
            for z in read(old.OUT/'temporal'/b/g/'barrier.json')['receipts']:
                temporal.append(dict(backbone=b,group=g,**z))
    write(OUT/'lock.json',dict(version='decota_refine_uniform_v1',created=time.time(),
        requested_default='uniform_absolute_fixed',experimental='minimax_absolute; numerical minimum_energy equivalent',
        previous_lock=str(old.OUT/'lock.json'),previous_sha256=sha(old.OUT/'lock.json'),
        historical_current=read(ROOT/'methods/CURRENT_METHOD.json'),
        pins={f:sha(ROOT/f) for f in files},temporal_receipts=temporal,
        families=FAMILIES,spatial_default=SPATIAL,temporal_configs=p['temporal_configs'],
        gate_grid=p['grids'],gate_sweeps=2,gate_selection='source-macro development corrected vIoU',
        oracle='GT boxes only at the accepted-and-GT-annotated positions; diagnostic, not legal TTA',
        source_counts={s:{g:len(p['rows'][s][g]) for g in GS} for s in SPLITS},
        historically_exposed=True,untouched_confirmation=False,
        no_temporal_retuning=True,no_corruption=True,spatial_parameter_updates=False,
        selection_objective='minimax physical-frame distance over the discrete available grid inside adapted T',
        minimax_tie_break='rightmost feasible center greedy, leftover budget farthest point; no labels',
        energy='sum squared cxcywh differences / physical frame gap, Dirichlet expert anchors, anchor hull only',
        coordinates='normalized cxcywh; empty/single/outside-hull fallback unchanged',
        statistics=dict(bootstrap=10000,seed=20260910,unit='original source video',neutral_pp=.1)))
    print('LOCKED',read(OUT/'lock.json')['source_counts'],flush=True)


def plan():
    p=read(OUT/'lock.json')
    assert sha(p['previous_lock'])==p['previous_sha256']
    for f,h in p['pins'].items():assert sha(ROOT/f)==h,f
    previous=old.plan()
    for g in GS:
        for key in ['source','video_sha256']:
            assert not {r['input'][key] for r in previous['rows']['development'][g]} & {r['input'][key] for r in previous['rows']['evaluation'][g]}
    return p,previous


def temporal(p,b,g,split,j):
    rr=next(r for r in p['temporal_receipts'] if r['backbone']==b and r['group']==g and r['split']==split and r['ordinal']==j)
    assert sha(rr['path'])==rr['sha256']
    x=load(rr['path']);assert not x['GT_used'];return x


def positions(ids,ij,ni,ev,selector,k=8):
    return old.select_positions(ids,ij,ni,ev,k,'current') if selector=='current' else select_positions(ids,ij,k,selector)


def expert_path(g,split,j):return OUT/'expert'/g/split/f'{j:03d}.pt'


def cache_expert():
    import torch
    from vg_tta.tg_spatial_tta_v1 import SpatialExpert
    from vg_tta.dense_expansion_data_v1 import decode_raw
    from vg_tta.foreground_runtime import state_digest
    if (OUT/'expert_barrier.json').exists():return
    p,prior=plan();configure();expert=None;before=None;receipts=[];counts=dict(new=0,reused=0)
    assert sha(Path(prior['expert_snapshot'])/'model.safetensors')==prior['expert_sha256']
    for g in GS:
        previous_receipts={z['path']:z['sha256'] for z in read(old.OUT/'expert'/g/'barrier.json')['receipts']}
        for split in SPLITS:
            for r in prior['rows'][split][g]:
                j=r['ordinal'];q=r['input'];f=expert_path(g,split,j)
                if not f.exists():
                    need=set()
                    for b in BS:
                        t=temporal(p,b,g,split,j);n=old.native_row(r,b);ev=old.evidence(r,b,n)
                        for selector in ['current','uniform','minimax']:
                            need.update(positions(q['frame_ids'],t['intervals']['decota'],t['intervals']['frozen'],ev,selector))
                    phrase,entity=old.text_spec(r,'parsed');olddev={};olddevpath=None
                    if split=='development':
                        olddevpath=old.PREV/'expert'/g/f"{q['index']:06d}.pt"
                        db=read(old.PREV/'expert_barrier.json')
                        rr=next(z for z in db['receipts'] if z['group']==g and z['index']==q['index'])
                        assert sha(olddevpath)==rr['sha256'];dd=load(olddevpath)
                        assert not dd['GT_used'] and dd['frame_ids']==q['frame_ids']
                        olddev={z['position']:z for z in dd['probes']}
                    probes=[];raw=None
                    if phrase and entity in phrase:
                        for pos in sorted(need):
                            of=old.epath(g,split,j,'parsed',pos);z=None;origin=None
                            if str(of) in previous_receipts:
                                assert sha(of)==previous_receipts[str(of)]
                                z=load(of);origin=dict(path=str(of),sha256=sha(of))
                            elif pos in olddev:
                                z=olddev[pos];origin=dict(path=str(olddevpath),sha256=sha(olddevpath))
                            if z is not None:
                                z=copy.deepcopy(z);counts['reused']+=1
                                assert z['text']==phrase.lower().strip()+'.' and z['frame_id']==q['frame_ids'][pos]
                            else:
                                if expert is None:
                                    expert=SpatialExpert(prior['expert_snapshot']);before=state_digest(expert.model)
                                if raw is None:
                                    assert sha(q['video_path'])==q['video_sha256'];raw,ids=decode_raw(q);assert ids==q['frame_ids']
                                z=expert(raw[pos],phrase,entity)
                                z.update(position=pos,frame_id=q['frame_ids'][pos]);counts['new']+=1
                            z.update(origin=origin,GT_used=False,video_sha256=q['video_sha256']);probes.append(z)
                    save(f,dict(probes=probes,required_positions=sorted(need),frame_ids=q['frame_ids'],
                        source=q['source'],GT_used=False,visual_query=r['visual_query']))
                    del raw
                x=load(f)
                receipts.append(dict(group=g,split=split,ordinal=j,path=str(f),sha256=sha(f)))
                status(OUT/'progress.json',dict(stage='expert',group=g,split=split,done=j+1,total=len(prior['rows'][split][g]),**counts))
                print('EXPERT',g,split,j+1,len(prior['rows'][split][g]),counts,flush=True)
    if expert is not None:assert state_digest(expert.model)==before
    write(OUT/'expert_barrier.json',dict(receipts=receipts,expert_unchanged=True,GT_used=False,
        new_forward_receipts=sum(sum(z['origin'] is None for z in load(r['path'])['probes']) for r in receipts),
        reused_receipts=sum(sum(z['origin'] is not None for z in load(r['path'])['probes']) for r in receipts)))


def pack(p,prior,r,b,g,split):
    n=old.native_row(r,b);t=temporal(p,b,g,split,r['ordinal'])
    f=expert_path(g,split,r['ordinal']);bar=read(OUT/'expert_barrier.json')
    rr=next(z for z in bar['receipts'] if z['path']==str(f));assert sha(f)==rr['sha256'];ex=load(f)
    assert not ex['GT_used'] and ex['frame_ids']==n['frame_ids']
    return dict(row=r,native=n,temporal=t,ids=n['frame_ids'],base=n['predictions']['frozen']['boxes'].float().cpu(),
        ni=t['intervals']['frozen'],ij=t['intervals']['decota'],evidence=old.evidence(r,b,n),probes=ex['probes'],nms={})


def spatial(x,selector,reconstruction,cfg):
    pos=positions(x['ids'],x['ij'],x['ni'],x['evidence'],selector,cfg['keyframes'])
    bypos={z['position']:z for z in x['probes']};phrase,entity=old.text_spec(x['row'],'parsed')
    if phrase and entity in phrase:assert set(pos)<=set(bypos)
    pseudo=anchors(x['probes'],pos,cfg,x['nms'])
    started=time.perf_counter();boxes,audit=reconstruct(x['base'],pseudo,x['ids'],reconstruction)
    reconstruction_seconds=time.perf_counter()-started
    return dict(boxes=boxes,indices=list(x['ij']),positions=pos,called=[i for i in pos if i in bypos],pseudo=pseudo,
        audit=audit,selector=selector,reconstruction=reconstruction,config=dict(cfg),
        radius=coverage_radius(x['ids'],x['ij'],pos),
        accepted_radius=coverage_radius(x['ids'],x['ij'],[q['position'] for q in pseudo]),
        radius_fraction=coverage_radius(x['ids'],x['ij'],pos)/max(1,x['ids'][x['ij'][1]]-x['ids'][x['ij'][0]]),
        reconstruction_seconds=reconstruction_seconds,
        expert_seconds=sum(bypos[i]['seconds'] for i in pos if i in bypos))


def predict(b,g):
    import torch
    torch.set_num_threads(2);p,prior=plan();sel=read(OUT/'selection_barrier.json')
    rr=next(z for z in sel['receipts'] if z['backbone']==b and z['group']==g)
    assert sha(rr['path'])==rr['sha256'];selected=read(rr['path']);receipts=[]
    for r in prior['rows']['evaluation'][g]:
        j=r['ordinal'];f=OUT/'predictions'/b/g/f'{j:03d}.pt'
        if not f.exists():
            x=pack(p,prior,r,b,g,'evaluation');pred={}
            for name in FAMILIES:
                selector,mode=name.split('_')
                for variant,cfg in [('fixed',SPATIAL),('retuned',selected['families'][name]['config'])]:
                    pred[name+'_'+variant]=spatial(x,selector,mode,cfg)
                    if mode=='absolute':
                        v=spatial(x,selector,'minimum_energy',cfg)
                        assert v['pseudo']==pred[name+'_'+variant]['pseudo']
                        assert torch.max(abs(v['boxes']-pred[name+'_'+variant]['boxes']))<=2e-7
                        pred[selector+'_energy_'+variant]=v
            for name,ij in [('frozen',x['ni']),('time_only',x['ij'])]:
                pred[name]=dict(boxes=x['base'],indices=list(ij),called=[],pseudo=[],positions=[])
            previous=load(old.OUT/'predictions'/b/g/f'{j:03d}.pt')['predictions']
            for now,then in [('current_residual','main'),('current_absolute','absolute'),('uniform_residual','uniform')]:
                for variant in ['fixed','retuned']:
                    a=pred[now+'_'+variant];z=previous[then+'_'+variant]
                    assert a['indices']==z['indices'] and torch.equal(a['boxes'],z['boxes']),(now,variant,b,g,j)
            u=pred['uniform_absolute_fixed'];m=pred['minimax_absolute_fixed']
            assert m['radius']<=u['radius']
            certificate=minimax_positions(x['ids'],x['ij'],SPATIAL['keyframes'])[1]
            save(f,dict(predictions=pred,input=r['input'],frame_ids=x['ids'],GT_used=False,old_parity=True,
                minimax_certificate=certificate,selection_sha256=sha(OUT/'selection_barrier.json')))
        receipts.append(dict(backbone=b,group=g,ordinal=j,path=str(f),sha256=sha(f),source=r['input']['source']))
        print('PREDICT',b,g,j+1,len(prior['rows']['evaluation'][g]),flush=True)
    write(OUT/'predictions'/b/g/'barrier.json',dict(receipts=receipts,GT_used=False))


def seal():
    p,prior=plan();receipts=[]
    for b in BS:
        for g in GS:
            bar=read(OUT/'predictions'/b/g/'barrier.json');assert not bar['GT_used']
            assert len(bar['receipts'])==len(prior['rows']['evaluation'][g])
            for r in bar['receipts']:assert sha(r['path'])==r['sha256']
            receipts.extend(bar['receipts'])
    write(OUT/'prediction_barrier.json',dict(receipts=receipts,created=time.time(),GT_used=False,
        selection_sha256=sha(OUT/'selection_barrier.json')))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','expert','predict','seal']);ap.add_argument('--backbone',choices=BS);ap.add_argument('--group',choices=GS)
    a=ap.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='expert':cache_expert()
    elif a.stage=='predict':predict(a.backbone,a.group)
    else:seal()
