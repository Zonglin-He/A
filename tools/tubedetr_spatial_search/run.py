"""Sealed CPU spatial development; labels live in separate scoring processes."""
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from stvg.runtime import ROOT, sha
from stvg.tube_runner import write, digest
from .numeric import predictions, dense_scores, finish

PUBLIC=ROOT/'results/tubedetr_spatial_search'
OUT=ROOT/'outputs/tubedetr_spatial_search'
PLAN=PUBLIC/'plan.json'
GAMMAS=[0.,.0125,.025,.05,.1,.2,.35,.5,.75,1.,1.5,2.]
GRIDS=dict(gamma=GAMMAS,observed_weight=[0.,.1,.25,.5,1.,2.,4.],
 unobserved_weight=[0.,.1,.25,.5,1.,2.,4.],link_weight=[0.,.03125,.125,.25,.5,1.,2.,4.],
 guard=[0.,.005,.02,.1,.4,1.,4.,16.],admission_margin=[-2.,0.,.01,.03,.1,.25],
 alpha_observed=[0.,.025,.05,.1,.25,.5,.75,1.],
 alpha_unobserved=[0.,.025,.05,.1,.25,.5,.75,1.],
 width_scale=[.8,.85,.9,.95,1.,1.05,1.1,1.15,1.2],
 height_scale=[.8,.85,.9,.95,1.,1.05,1.1,1.15,1.2])


def variants():
    return [dict(topk=k,temperature=t) for k,t in [(20,.1),(5,.1),(10,.1),(40,.1),(20,.03),(20,.06),(20,.2),(20,.4)]]


def profiles(ds):
    base=dict(family='simplepath',geometry=0,gamma=1. if ds=='vidstg' else .5,
       observed_weight=.5,unobserved_weight=.5,link_weight=.25,guard=0.,
       admission_margin=-2.,alpha_observed=1.,alpha_unobserved=1.,width_scale=1.,height_scale=1.)
    values=[];seen=set()
    def add(p):
        key=digest(p)
        if key not in seen:values.append(p);seen.add(key)
    add({**base,'family':'source_control','alpha_observed':0.,'alpha_unobserved':0.})
    add({**base,'original_baseline':True})
    for w in GRIDS['width_scale']:
        for h in GRIDS['height_scale']:
            add({**base,'family':'source_calibration','width_scale':w,'height_scale':h})
    for gamma in GAMMAS:
        for alpha in [0.,.05,.1,.25,.5,1.]:
            add({**base,'family':'initialization','gamma':gamma,'alpha_observed':alpha,'alpha_unobserved':alpha})
    for a in GRIDS['alpha_observed']:
        for b in GRIDS['alpha_unobserved']:
            add({**base,'family':'sealed_full_blend','alpha_observed':a,'alpha_unobserved':b})
    for k,grid in GRIDS.items():
        for v in grid:add({**base,k:v})
    for v in range(len(variants())):add({**base,'geometry':v})
    rng=np.random.default_rng(20261024+(ds=='hc2'))
    while len(values)<2048:
        p={**base,'geometry':int(rng.integers(len(variants())))}
        for k,g in GRIDS.items():p[k]=g[int(rng.integers(len(g)))]
        # Retain many uncalibrated candidates, and jointly search numeric geometry.
        if rng.random()<.5:p.update(width_scale=1.,height_scale=1.)
        if rng.random()<.25:p.update(alpha_observed=1.,alpha_unobserved=1.)
        add(p)
    return values


def checked():
    p=json.loads(PLAN.read_text())
    for f,h in p['code_sha256'].items():assert sha(ROOT/f)==h,f
    return p


def prepare():
    assert not PLAN.exists(),'Never overwrite a scored search plan'
    parent=ROOT/'results/simplepath_tubedetr/run_plan.json';old=json.loads(parent.read_text())
    files=[*sorted((ROOT/'tools/tubedetr_spatial_search').glob('*.py')),
           *sorted((ROOT/'tools/simplepath_tube/core').glob('*.py')),
           ROOT/'stvg/metrics.py',ROOT/'stvg/predictor.py',ROOT/'stvg/runtime.py']
    p=dict(status='frozen_before_spatial_search_predictions_and_scores',
       authorization='2026-10-24 user explicitly requests broad TubeDETR parameter tuning to make spatial sIoU increment positive',
       objective='maximize complete original128 equal-query dense sIoU relative to unchanged Source, independently for both domains',
       exposure='development selection; original128 and earlier full research already exposed; not independent test confirmation',
       source='untouched original backbone outputs, native MAP time; NativeMBR is only separate readout ablation',
       temporal='each query keeps previously sealed SimplePath Selected_interval; no temporal tuning, no Global6 switching',
       neural_calls=0,new_teacher_calls=0,network_updates=0,
       scope='independent numerical SimplePath search; optional Source protection, output mixing and extent calibration are explicit added numeric postprocessing, not claimed unchanged upstream method',
       grids=GRIDS,geometry_variants=variants(),profiles={ds:profiles(ds) for ds in ['vidstg','hc2']},
       profile_budget_per_domain=2048,dataset_order=['vidstg','hc2'],
       selection='largest complete128 dense sIoU; exact ties prefer no extent calibration, lower output modification, lower index; preserve every positive and negative configuration',
       controls=['unchanged Source','original frozen SimplePath replay','Source-only extent calibration; explicitly separate from propagation improvement'],
       next_step='lock both128 selections, independently evaluate complete full population if selected family can reuse sealed outputs; new GPU geometry work must wait until active LLaVA completes and acquire shared lock',
       full_not_used_to_select_parameters=True,never_choose_per_query_using_GT=True,
       parent_plan_sha256=sha(parent),datasets=copy.deepcopy(old['datasets']),
       code_sha256={str(f.relative_to(ROOT)):sha(f) for f in files})
    for ds in p['dataset_order']:
        d=p['datasets'][ds]
        for label in ['manifest','full_manifest']:
            assert sha(ROOT/d[label])==d[label+'_sha256']
        for name,rel in [('captures','outputs/tubedetr_boxfixed_development'),('fixed','outputs/simplepath_tubedetr')]:
            f=ROOT/rel/ds/('captures' if name=='captures' else 'dev128')/'sealed.json'
            d[name+'_seal']=str(f.relative_to(ROOT));d[name+'_seal_sha256']=sha(f)
        d['fixed_full_seal']=f'outputs/simplepath_tubedetr/{ds}/full/sealed.json'
        d['fixed_full_seal_sha256']=sha(ROOT/d['fixed_full_seal'])
    write(PLAN,p);print('frozen plan',sha(PLAN),'profiles2048/domain',flush=True)


def verify_seal(path,expected=None):
    if expected:assert sha(path)==expected
    s=json.loads(path.read_text());assert s['status']=='complete'
    for e in s['files'].values():
        assert sha(path.parent/e['path'])==e['sha256']
        assert sha(path.parent/e['receipt_path'])==e['receipt_sha256']
    return s


def predict(ds):
    # The scorer is a separate process. This process cannot open annotation files.
    from stvg.runtime import setup
    setup();torch.set_num_threads(1);p=checked();d=p['datasets'][ds]
    caps_path=ROOT/d['captures_seal'];caps=verify_seal(caps_path,d['captures_seal_sha256'])
    fixed_path=ROOT/d['fixed_seal'];fixed=verify_seal(fixed_path,d['fixed_seal_sha256'])
    rows=[json.loads(x) for x in (ROOT/d['manifest']).read_text().splitlines()]
    assert set(caps['files'])==set(fixed['files'])=={r['id'] for r in rows} and len(rows)==128
    target=OUT/ds/'development';target.mkdir(parents=True,exist_ok=True);entries={}
    for j,row in enumerate(rows):
        q=row['id'];e=caps['files'][q];c=torch.load(caps_path.parent/e['path'],map_location='cpu',weights_only=False)
        with np.load(fixed_path.parent/fixed['files'][q]['path'],allow_pickle=False) as f:
            a={k:f[k].copy() for k in f.files}
        for key in ['Source','frame_ids','positions8','Native_interval']:assert np.array_equal(np.asarray(c[key]),a[key]),(q,key)
        tick=time.perf_counter();out=predictions(c,a,p['profiles'][ds],p['geometry_variants'])
        path=target/(q+'.npz');tmp=path.with_suffix('.pending')
        with tmp.open('xb') as stream:np.savez_compressed(stream,predictions=out,Source=c['Source'],frame_ids=c['frame_ids'],positions8=c['positions8'],Native_interval=c['Native_interval'],Selected_interval=a['Selected_interval'])
        tmp.replace(path);rp=path.with_suffix('.json')
        write(rp,dict(plan_sha256=sha(PLAN),request_sha256=digest(row),payload_sha256=sha(path),labels_read=False,
                      profiles=len(out),original_SimplePath_replay_bitexact=True,Source_unchanged=True,
                      capture_sha256=e['sha256'],seconds=time.perf_counter()-tick,new_neural_calls=0,new_teacher_calls=0))
        entries[q]=dict(path=path.name,sha256=sha(path),receipt_path=rp.name,receipt_sha256=sha(rp))
        write(OUT/'worker_progress.json',dict(dataset=ds,stage='predict2048',complete=j+1,total=128,labels_read=False))
        print(ds,j+1,'/128, profiles',len(out),flush=True)
    write(target/'sealed.json',dict(status='complete',queries=128,profiles=2048,files=entries,plan_sha256=sha(PLAN),labels_read=False))


def score(ds):
    p=checked();d=p['datasets'][ds];target=OUT/ds/'development';s=verify_seal(target/'sealed.json')
    assert s['queries']==128 and s['plan_sha256']==sha(PLAN) and s['profiles']==2048
    # All candidate predictions have been sealed and verified before labels open.
    assert sha(ROOT/d['labels'])==d['labels_sha256']
    labels={r['id']:r for r in map(json.loads,(ROOT/d['labels']).read_text().splitlines())}
    rows=[json.loads(x) for x in (ROOT/d['manifest']).read_text().splitlines()]
    assert set(labels)==set(s['files'])=={r['id'] for r in rows}
    all_s=[];all_u=[];all_o=[];parents=[];ids=[]
    for row in rows:
        q=row['id'];gt=labels[q];parents.append(gt['parent']);ids.append(q)
        with np.load(target/s['files'][q]['path'],allow_pickle=False) as a:
            fids,iou=dense_scores(a['predictions'],a['frame_ids'],row['metadata']['width'],row['metadata']['height'],gt['boxes'],ds=='hc2')
            seen=np.isin(fids,a['frame_ids'][a['positions8']])
            ss=iou.mean(1);uu=iou[:,~seen].mean(1) if (~seen).any() else np.full(2048,np.nan)
            oo=iou[:,seen].mean(1) if seen.any() else np.full(2048,np.nan)
            # Independent reference for Source, original method, and a random config.
            from stvg.metrics import DenseTube
            for ix in [0,1,137]:
                ref=DenseTube(a['predictions'][ix],dict(frame_ids=a['frame_ids'],input=row['metadata']),{int(k):v for k,v in gt['boxes'].items()},gt['interval'],clip=ds=='hc2')
                assert abs(ref.s-ss[ix])<1e-12,(q,ix,ref.s,ss[ix])
        all_s.append(ss);all_u.append(uu);all_o.append(oo)
    S=np.array(all_s);U=np.array(all_u);O=np.array(all_o);means=S.mean(0)*100
    def rank(i):
        a=p['profiles'][ds][i]
        return (means[i],-(abs(a['width_scale']-1)+abs(a['height_scale']-1)),
                -(a['alpha_observed']+a['alpha_unobserved']),-i)
    winner=max(range(2048),key=rank)
    result=[dict(profile_id=digest(a)[:16],index=i,parameters=a,sIoU=float(means[i]),delta_sIoU_pp=float(means[i]-means[0]),
                 unobserved_sIoU=float(np.nanmean(U[:,i])*100),observed_sIoU=float(np.nanmean(O[:,i])*100),
                 declines_over5pp=int((S[:,i]<S[:,0]-.05).sum())) for i,a in enumerate(p['profiles'][ds])]
    from tools.tubedetr_opd_direct_diagnosis import cluster_difference
    selection=dict(status='locked_after_complete128_search_before_any_new_full',dataset=ds,queries=128,profiles_tested=2048,
      winner=result[winner],Source_sIoU=float(means[0]),original_fixed_sIoU=float(means[1]),
      positive_development_increment=bool(means[winner]>means[0]),
      descriptive_parent_CI=cluster_difference((S[:,winner]-S[:,0]).tolist(),parents),
      exposure='selected on these128 and uncorrected for search; not independent confirmation',
      best_by_family={family:result[max([i for i,x in enumerate(p['profiles'][ds]) if x['family']==family],key=rank)] for family in sorted({x['family'] for x in p['profiles'][ds]})},
      plan_sha256=sha(PLAN),prediction_seal_sha256=sha(target/'sealed.json'))
    write(PUBLIC/ds/'all_profiles.json',result);write(PUBLIC/ds/'selected_profile.json',selection)
    np.savez_compressed(target/'scores_private.npz',sIoU=S,unobserved=U,observed=O)
    anon=[dict(query=hashlib.sha256(q.encode()).hexdigest(),parent=hashlib.sha256(str(parent).encode()).hexdigest(),sIoU=row.tolist()) for q,parent,row in zip(ids,parents,S)]
    path=PUBLIC/ds/'all_signed_development_scores.json.gz'
    with gzip.open(path,'wt') as stream:json.dump(anon,stream,separators=(',',':'))
    print(ds,'locked',json.dumps(selection['winner']),flush=True)


def child(stage,ds):
    cmd=[sys.executable,'-B','-u','-m','tools.tubedetr_spatial_search.run',stage,'--dataset',ds]
    log=ROOT/'logs'/f'tubedetr_spatial_search_{ds}_{stage}.log'
    write(OUT/'status.json',dict(stage=stage,dataset=ds,CPU_only=True))
    with log.open('a') as f:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
                              env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        write(ROOT/'runtime/tubedetr_spatial_search.json',dict(controller_pid=os.getpid(),worker_pid=proc.pid,command=cmd,log=str(log.relative_to(ROOT))))
        if proc.wait():raise RuntimeError(f'{stage} {ds} failed; inspect {log}')


def run():
    p=checked()
    for ds in p['dataset_order']:child('predict',ds);child('score',ds)
    locks={ds:json.loads((PUBLIC/ds/'selected_profile.json').read_text()) for ds in p['dataset_order']}
    write(PUBLIC/'selected_profiles.json',dict(status='both128_locked_before_new_full',datasets=locks,plan_sha256=sha(PLAN)))
    write(OUT/'status.json',dict(status='both2048_profile_searches_complete',both_positive=all(x['positive_development_increment'] for x in locks.values()),
         next='Root verify and publish all configurations; evaluate selected complete populations with same Source and no further selection'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run','predict','score']);parser.add_argument('--dataset',choices=['vidstg','hc2']);a=parser.parse_args()
    try:
        if a.stage=='prepare':prepare()
        elif a.stage=='run':run()
        elif a.stage=='predict':predict(a.dataset)
        else:score(a.dataset)
    except Exception as exc:
        write(OUT/'failure.json',dict(stage=a.stage,dataset=a.dataset,error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));raise
