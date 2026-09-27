"""Bounded development-only cache interventions. Predictions precede GT scoring."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.decota_uniform_energy_v1 import plan, pack, spatial, BS, GS, SPATIAL
from scripts.decota_matrix_common_v1 import read, write, save, load, sha
from vg_tta.decota_local_iteration_v1 import (
    ALPHAS, RHOS, blend, extent_amount, local_placement, leave_one_anchor_alpha)

OUT=ROOT/'artifacts/decota_local_iteration_v1'
SEEDS=(20260911,20260912,20260913)
EPS=.001
METRICS=['vIoU_corrected','sIoU','tIoU','recall','precision','span','uncalled_sIoU']
FAMILIES={
    'temporal_locality':[f'local_{r}' for r in RHOS],
    'temporal_amount':[f'amount_{a}' for a in ALPHAS],
    'spatial_blend':[f'spatial_{a}' for a in ALPHAS],
}
BASELINES={'temporal_locality':'local_None','temporal_amount':'local_None','spatial_blend':'spatial_1.0'}


def protect():
    files=list((ROOT/'methods').rglob('*.py'))+[ROOT/'methods/CURRENT_METHOD.json']
    files += list((ROOT/'methods').rglob('configs.json'))
    return {str(f.relative_to(ROOT)):sha(f) for f in files}


def verify_protected(lock):
    for f,h in lock['protected'].items():
        assert sha(ROOT/f)==h, f
    for f,h in lock['code'].items():
        assert sha(ROOT/f)==h, f


def predict():
    torch.set_num_threads(2)
    p,prior=plan()
    if (OUT/'prediction_barrier.json').exists():
        verify_protected(read(OUT/'lock.json'));return
    if not (OUT/'lock.json').exists():
        folds={g:{r['input']['source']:i%2 for i,r in enumerate(sorted(prior['rows']['development'][g],
                      key=lambda r:hashlib.sha256(r['input']['source'].encode()).hexdigest()))} for g in GS}
        write(OUT/'lock.json',dict(created=time.time(),protocol_sha256=sha(ROOT/'protocols/decota_local_iteration_v1.md'),
            prior_lock_sha256=sha(ROOT/'artifacts/decota_uniform_energy_v1/lock.json'),
            protected=protect(),code={f:sha(ROOT/f) for f in ['vg_tta/decota_local_iteration_v1.py','scripts/run_decota_local_iteration_v1.py']},
            split='development',historically_exposed=True,independent_confirmation=False,
            folds=folds,spatial_config=SPATIAL,temporal_configs=p['temporal_configs'],
            families=FAMILIES,GT_used_for_predictions=False,new_neural_calls=0,old_queues_resumed=False))
    lock=read(OUT/'lock.json');verify_protected(lock);receipts=[]
    for b in BS:
        for g in GS:
            xs=[]
            for r in prior['rows']['development'][g]:
                f=OUT/'predictions'/b/g/f"{r['ordinal']:03d}.pt"
                if f.exists():
                    xs.append(load(f));continue
                x=pack(p,prior,r,b,g,'development');raw=x['temporal']['intervals']['coupled_coverage']
                z=x['native']['native_logits'];ids=x['ids'];ni=x['ni'];current=x['ij']
                temporal={'frozen':ni,'raw_literal':raw}
                for rho in RHOS:
                    temporal[f'local_{rho}']=local_placement(z,ids,raw,ni,rho)
                for eta in ALPHAS:
                    temporal[f'amount_{eta}']=extent_amount(z,ids,raw,ni,eta)
                assert tuple(temporal['local_None'])==tuple(current)
                assert tuple(temporal['amount_1.0'])==tuple(current)
                v=spatial(x,'uniform','absolute',SPATIAL)
                a,loo=leave_one_anchor_alpha(x['base'],v['pseudo'],ids)
                spatial_boxes={f'spatial_{a}':blend(x['base'],v['boxes'],a) for a in ALPHAS}
                spatial_boxes['spatial_loo']=blend(x['base'],v['boxes'],a)
                assert torch.equal(spatial_boxes['spatial_0.0'],x['base'])
                assert torch.equal(spatial_boxes['spatial_1.0'],v['boxes'])
                record=dict(backbone=b,group=g,ordinal=r['ordinal'],source=r['input']['source'],ids=ids,
                    native_indices=ni,current_indices=current,temporal=temporal,base=x['base'],spatial=spatial_boxes,
                    accepted=v['pseudo'],called=v['called'],positions=v['positions'],loo_alpha=a,loo=loo,
                    GT_used=False,fold=lock['folds'][g][r['input']['source']],
                    input_receipts=dict(native=r['native'][b],temporal_source=x['temporal']['source']))
                save(f,record);xs.append(record)
                print('PREDICT',b,g,r['ordinal']+1,len(prior['rows']['development'][g]),flush=True)
            # Same marginal alpha distribution; no source is its own donor.
            permutations={}
            for seed in SEEDS:
                rng=np.random.default_rng(seed)
                while True:
                    perm=rng.permutation(len(xs))
                    if np.all(perm!=np.arange(len(xs))):break
                permutations[str(seed)]=perm.tolist()
                for j,x in enumerate(xs):
                    a=xs[int(perm[j])]['loo_alpha']
                    x['spatial'][f'spatial_loo_shuffle{seed}']=blend(x['base'],x['spatial']['spatial_1.0'],a)
            for x in xs:
                f=OUT/'sealed'/b/g/f"{x['ordinal']:03d}.pt"
                if not f.exists():save(f,x)
                receipts.append(dict(path=str(f),sha256=sha(f),backbone=b,group=g,ordinal=x['ordinal'],source=x['source']))
            pf=OUT/'permutations'/f'{b}_{g}.json'
            if not pf.exists():write(pf,permutations)
    verify_protected(lock)
    write(OUT/'prediction_barrier.json',dict(receipts=receipts,GT_used=False,records=len(receipts),
        source_count=len({r['source'] for r in receipts}),protocol_sha256=lock['protocol_sha256']))
    print('SEALED',len(receipts),'model/source records; no labels read',flush=True)


def stats(values):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    if not len(a):return dict(n=0,mean=None,ci95=None,median=None)
    ix=np.random.default_rng(SEEDS[0]).integers(len(a),size=(10000,len(a)))
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),
                ci95=np.percentile(a[ix].mean(1),[2.5,97.5]).tolist())


def contrast(rows,method,baseline):
    d=np.array([r['metrics'][method]['vIoU_corrected']-r['metrics'][baseline]['vIoU_corrected'] for r in rows])
    return dict(**stats(d),win_neutral_loss=[int((d>EPS).sum()),int((abs(d)<=EPS).sum()),int((d< -EPS).sum())])


def choose(rr,methods,baseline):
    # Ties retain current when possible; otherwise choose closest grid order.
    scores={m:np.mean([r['metrics'][m]['vIoU_corrected'] for r in rr]) for m in methods}
    best=max(scores.values());candidates=[m for m in methods if scores[m]>=best-1e-12]
    return baseline if baseline in candidates else candidates[-1]


def score():
    from scripts.score_decota_heuristic_study_v1 import gt_payload,gt_array,metric
    from scripts.audit_decota_heuristic_study_v1 import independent_quality
    torch.set_num_threads(2)
    lock=read(OUT/'lock.json');verify_protected(lock);bar=read(OUT/'prediction_barrier.json')
    assert not bar['GT_used']
    p,prior=plan();rows=[];max_error=0.
    for rr in bar['receipts']:
        assert sha(rr['path'])==rr['sha256']
        x=load(rr['path']);assert not x['GT_used']
        b,g,j=x['backbone'],x['group'],x['ordinal'];r=prior['rows']['development'][g][j]
        targets,gt=gt_payload(prior,r,g,'development');truth,valid=gt_array(targets,x['ids'],gt)
        predictions={k:(x['base'],ij) for k,ij in x['temporal'].items()}
        predictions.update({k:(v,x['current_indices']) for k,v in x['spatial'].items()})
        metrics={};grid=np.array(x['ids']);uncalled=valid.copy();uncalled[x['called']]=False
        for name,(boxes,ij) in predictions.items():
            m,q=metric(boxes,truth,valid,x['ids'],gt,ij)
            independent=independent_quality(boxes,truth,valid)
            a,c=x['ids'][ij[0]],x['ids'][ij[1]]+1
            inter=valid&(grid>=a)&(grid<c);union=(grid>=min(a,gt[0]))&(grid<max(c,gt[1]))
            vv=float(independent[inter].sum()/max(1,int(union.sum())))
            err=abs(m['vIoU_corrected']-vv);assert err<1e-10;max_error=max(max_error,err)
            m['uncalled_sIoU']=float(np.mean(q[uncalled])) if uncalled.any() else None
            metrics[name]=m
        for name in x['temporal']:
            assert metrics[name]['sIoU']==metrics['frozen']['sIoU']
        for name in x['spatial']:
            assert metrics[name]['tIoU']==metrics['local_None']['tIoU']
        rows.append(dict(backbone=b,group=g,ordinal=j,source=x['source'],fold=x['fold'],metrics=metrics,
            accepted=len(x['accepted']),called=len(x['called']),uncalled_frames=int(uncalled.sum()),
            loo_alpha=x['loo_alpha'],loo=x['loo'],gt_interval=gt,
            temporal_intervals={k:[x['ids'][ij[0]],x['ids'][ij[1]]+1] for k,ij in x['temporal'].items()}))
    assert len(rows)==128 and len({(r['backbone'],r['group'],r['source']) for r in rows})==128
    summary={}
    for b in BS:
        for g in GS:
            rr=sorted([r for r in rows if r['backbone']==b and r['group']==g],key=lambda r:r['source'])
            assert len(rr)==len({r['source'] for r in rr})==32
            selected={};crossfit={}
            for family,methods in FAMILIES.items():
                baseline=BASELINES[family];allbest=choose(rr,methods,baseline)
                folds={}
                for heldout in [0,1]:
                    train=[r for r in rr if r['fold']!=heldout];test=[r for r in rr if r['fold']==heldout]
                    name=choose(train,methods,baseline);folds[str(heldout)]=name
                    for r in test:r['metrics']['crossfit_'+family]=r['metrics'][name]
                selected[family]=dict(full_dev_best=allbest,fold_selected=folds)
                crossfit[family]=contrast(rr,'crossfit_'+family,baseline)
            absolute={m:{k:stats([r['metrics'][m][k] for r in rr if r['metrics'][m][k] is not None]) for k in METRICS}
                      for m in rr[0]['metrics']}
            contrasts={m:contrast(rr,m,'spatial_1.0' if m.startswith('spatial') else 'local_None')
                       for m in rr[0]['metrics'] if not m.startswith('crossfit')}
            slices={}
            for component,baseline,working,family in [
                ('temporal','frozen','local_None','temporal_locality'),
                ('amount','frozen','local_None','temporal_amount'),
                ('spatial','local_None','spatial_1.0','spatial_blend')]:
                slices[component]={}
                for tag in ['good','neutral','failure']:
                    def inside(r):
                        d=r['metrics'][working]['vIoU_corrected']-r['metrics'][baseline]['vIoU_corrected']
                        return d>EPS if tag=='good' else d< -EPS if tag=='failure' else abs(d)<=EPS
                    sub=[r for r in rr if inside(r)]
                    slices[component][tag]=dict(n=len(sub),
                        crossfit_delta=contrast(sub,'crossfit_'+family,working),
                        candidates={m:contrast(sub,m,working) for m in FAMILIES[family]})
            loo_compare={f'vs_{m}':contrast(rr,'spatial_loo',m) for m in
                         ['spatial_1.0','crossfit_spatial_blend']+[f'spatial_loo_shuffle{s}' for s in SEEDS]}
            eligible=[r for r in rr if r['loo']['eligible']]
            summary[b+'_'+g]=dict(n=32,selected=selected,absolute=absolute,contrasts=contrasts,crossfit=crossfit,
                slices=slices,loo=dict(eligible=len(eligible),alpha_counts={str(a):sum(r['loo_alpha']==a for r in rr) for a in ALPHAS},comparisons=loo_compare))
            print('RESULT',b,g,'selected',selected,'crossfit_delta_pp',
                  {k:round(v['mean']*100,4) for k,v in crossfit.items()},flush=True)
    write(OUT/'scored_rows.json',rows);write(OUT/'summary.json',summary)
    verify_protected(lock)
    write(OUT/'QA_RESULTS.json',dict(records=128,unique_sources=len({r['source'] for r in rows}),
        development_only=True,evaluation_labels_used=False,predictions_before_labels=True,
        independent_metric_max_error=max_error,temporal_spatial_invariants=True,
        locked_method_files_unchanged=True,no_new_neural_calls=True,no_new_GT_oracle=True,
        stopped_queues_resumed=False,caveat='historically exposed development; crossfit is exploratory'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['predict','score']);args=ap.parse_args()
    predict() if args.stage=='predict' else score()
