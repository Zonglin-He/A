"""Full-split scoring and tracking diagnostics; partial snapshots labeled.

This process alone combines labels with sealed label-free predictions.
Intervention outputs are always diagnostic oracles, never TTA baselines.
"""
import argparse,collections,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read,write,status,load,sha
from scripts.run_stvg_fullscale_v1 import plan,label_payload,OUT
from vg_tta.box_stability_diagnostics_v1 import overlap,trajectory,smooth
from vg_tta.st_component_diagnostics_v1 import mean_ci
from vg_tta.metrics import temporal_iou


def score(boxes,gt,ids,indices):
    boxes=np.asarray(boxes,float);truth=np.asarray(gt['boxes'],float);valid=np.asarray(gt['valid'],bool);ids=np.asarray(ids)
    assert boxes.shape==truth.shape and np.isfinite(boxes).all()
    quality=overlap(boxes,truth);g,h=gt['interval'];event=(ids>=g)&(ids<h)
    s=float(quality[valid].mean()) if valid.any() else None
    if indices is None:return dict(vIoU_corrected=0.,sIoU=s,tIoU=0.,span=0.,temporal_recall=0.,temporal_precision=0.),quality
    a,b=ids[indices[0]],ids[indices[1]]+1
    union=(ids>=min(a,g))&(ids<max(b,h));selected=(ids>=a)&(ids<b);intersection=max(0,min(b,h)-max(a,g))
    # Missing annotation is not silently hallucinated or charged to a model.
    v=None if (event&~valid).any() else float(quality[valid&selected].sum()/max(int(union.sum()),1))
    return dict(vIoU_corrected=v,sIoU=s,tIoU=temporal_iou((a,b),(g,h)),span=(b-a)/(ids[-1]+1-ids[0]),
        temporal_recall=intersection/(h-g),temporal_precision=intersection/(b-a)),quality


def sub_gt(gt,pos):
    return {**gt,**{k:np.asarray(gt[k])[pos].tolist() for k in ['boxes','valid','event_mask']}}


def summarize(values,sources):
    x=np.asarray([v for v in values if v is not None and np.isfinite(v)],float)
    r=mean_ci(values,sources,seed=20260911)
    r['query_mean']=float(x.mean()) if len(x) else None
    r['query_median']=float(np.median(x)) if len(x) else None
    if len(x):
        trim=int(.1*len(x));s=np.sort(x);r['query_trimmed10']=float(s[trim:len(s)-trim].mean())
    return r


def delta(a,b):return None if a is None or b is None else a-b


def run(stage,b,cohort,partial=False):
    p=plan();labels=label_payload(p);dest=OUT/stage/b/cohort;results=[];unavailable=[]
    complete=(dest/'barrier.json').exists()
    if not complete and not partial:raise RuntimeError('Full barrier missing; request an explicitly partial snapshot')
    manifests={r['key']:r for r in p['rows'][cohort]}
    for path in sorted(dest.glob('[0-9]*.json')):
        rr=read(path);assert rr['lock_sha256']==sha(OUT/'lock.json') and sha(rr['path'])==rr['sha256']
        x=load(rr['path']);row=manifests[x['key']]
        if x.get('status')=='input_unavailable':
            unavailable.append(dict(key=x['key'],source=x['input']['source'],reason=x['input_unavailable']));continue
        gt=labels[x['key']];ids=x['frame_ids'];q=x['input']
        if b=='ptd':gt=sub_gt(gt,x['parent_positions'])
        valid=np.asarray(gt['valid'],bool);truth=np.asarray(gt['boxes']);preds=x['predictions'];baseline='native' if stage=='causal' else 'frozen';base=preds[baseline]
        metrics={};tracking={};spatial={};layers=[];sm={}
        for name,z in preds.items():
            metrics[name],quality=score(z['boxes'],gt,ids,z['indices'])
            if stage=='causal' and name!='native_sparse' and valid.any():
                tr,_=trajectory(np.asarray(z['boxes']),truth,valid,ids,q['fps'],[q['width'],q['height']],np.asarray(z.get('present',np.ones(len(ids))),bool))
                tracking[name]=tr
        for name,z in x.get('spatial',{}).items():spatial[name]=score(base['boxes'],gt,ids,z['indices'])[0]
        if stage=='causal':
            if 'layer_boxes' in base:
                for lb in base['layer_boxes']:
                    layers.append(score(lb,gt,ids,base['indices'])[0])
            # GT-free smoothers are diagnostics, not proposed TTA or tuned candidates.
            times=np.array(ids)/q['fps'];present=np.asarray(base.get('present',np.ones(len(ids))),bool)
            for name,kw in [(f'mean{w}',dict(window=w)) for w in [3,5,9]]+[(f'median{w}',dict(window=w,median=True)) for w in [3,5]]+[(f'triangular_{s}s',dict(seconds=s)) for s in [.2,.5,1.]]:
                bb=smooth(np.asarray(base['boxes']),times,present,**kw)
                sm[name]=score(bb,gt,ids,base['indices'])[0]
        metrics_base=metrics[baseline];quality=overlap(np.asarray(base['boxes']),truth)
        grid=np.asarray(ids);g,h=gt['interval'];native_mask=np.zeros(len(ids),bool)
        if base['indices'] is not None:native_mask[base['indices'][0]:base['indices'][1]+1]=True
        event=(grid>=g)&(grid<h);good=valid&(quality>=.5)
        accounting=dict(event_fraction=(h-g)/(grid[-1]+1-grid[0]),valid_GT_frames=int(valid.sum()),
            good_spatial_frames_excluded=int((good&~native_mask).sum()),good_spatial_frames=int(good.sum()),
            incorrect_boxes_inside_native=int((valid&native_mask&(quality<.3)).sum()),
            GT_time_fixed_boxes_sIoU=float(quality[valid].mean()) if valid.any() else None,
            double_GT_sIoU=float(overlap(truth[valid],truth[valid]).mean()) if valid.any() else None,
            missing_event_GT=int((event&~valid).sum()))
        if valid.any():assert abs(accounting['double_GT_sIoU']-1)<1e-10
        # These are output-substitution diagnostics, not claims that one final
        # head causes the other, and GT time is not a strict utility upper bound.
        known_complete=not (event&~valid).any()
        substitutions=dict(B0_T0=metrics_base,BGT_T0=score(truth,gt,ids,base['indices'])[0],
            B0_TGT=dict(vIoU_corrected=accounting['GT_time_fixed_boxes_sIoU'] if known_complete else None,
                sIoU=accounting['GT_time_fixed_boxes_sIoU'],tIoU=1.),
            BGT_TGT=dict(vIoU_corrected=1. if valid.any() and known_complete else None,
                sIoU=accounting['double_GT_sIoU'],tIoU=1.))
        if 'temporal_only' in preds:
            substitutions['BGT_TTTA']=score(truth,gt,ids,preds['temporal_only']['indices'])[0]
        comparisons={}
        if 'time_GT_all' in metrics:
            comparisons['time_GT_minus_native_sIoU']=delta(metrics['time_GT_all']['sIoU'],metrics_base['sIoU'])
            random=[metrics.get(f'time_control_{s}',{}).get('sIoU') for s in p['seeds']]
            if all(v is not None for v in random):comparisons['time_GT_minus_random_sIoU']=delta(metrics['time_GT_all']['sIoU'],np.mean(random))
        for prefix in ['space_normalized_','space_pixels_']:
            pool=spatial if b=='ptd' else metrics
            if prefix+'GT_direction' in pool:
                a=pool[prefix+'GT_direction']['tIoU'];other=[pool[prefix+k]['tIoU'] for k in ['anti_GT','orthogonal_plus','orthogonal_minus']]
                comparisons[prefix+'GT_minus_controlmean_tIoU']=a-float(np.mean(other))
                comparisons[prefix+'GT_minus_anti_tIoU']=a-other[0]
                comparisons[prefix+'GT_minus_native_tIoU']=a-metrics_base['tIoU']
        results.append(dict(key=x['key'],source=q['source'],query_type=row['query_type'],historical_source_listed=row['historical_source_listed'],
            metrics=metrics,tracking=tracking,smoothing=sm,layers_fixed_native_time=layers,spatial=spatial,accounting=accounting,comparisons=comparisons,
            substitutions=substitutions,seconds=x['seconds'],receipt_sha256=rr['sha256']))
    if not results:raise RuntimeError('No completed receipts')
    output={}
    scopes={'all':results,**{t:[r for r in results if r['query_type']==t] for t in ['caption','question']}}
    scopes['previously_listed']=[r for r in results if r['historical_source_listed']]
    scopes['not_in_scanned_locks']=[r for r in results if not r['historical_source_listed']]
    for tag,rr in scopes.items():
        if not rr:continue
        c=dict(queries=len(rr),sources=len({r['source'] for r in rr}),methods={},comparisons={},tracking={},length_groups={})
        for name in sorted(set.union(*(set(r['metrics']) for r in rr))):
            xx=[r for r in rr if name in r['metrics']];ss=[r['source'] for r in xx]
            c['methods'][name]={key:summarize([r['metrics'][name][key] for r in xx],ss) for key in ['vIoU_corrected','sIoU','tIoU','span','temporal_recall','temporal_precision']}
        for name in sorted(set.union(*(set(r['comparisons']) for r in rr))):
            xx=[r for r in rr if name in r['comparisons']];c['comparisons'][name]=summarize([r['comparisons'][name] for r in xx],[r['source'] for r in xx])
        for name in sorted(set.union(*(set(r['tracking']) for r in rr))):
            xx=[r for r in rr if name in r['tracking']]
            c['tracking'][name]={key:summarize([r['tracking'][name].get(key) for r in xx],[r['source'] for r in xx]) for key in
                ['pred_step_px','GT_step_px','abs_IoU_step','pred_curvature_px','GT_curvature_px','large_IoU_step_given_GT_stable','pred_jump_given_GT_stable','center_SSE_offset_parity']}
        for label,lo,hi in [('short',0,.25),('medium',.25,.75),('long',.75,2.)]:
            xx=[r for r in rr if lo<=r['accounting']['event_fraction']<hi];ss=[r['source'] for r in xx]
            c['length_groups'][label]=dict(queries=len(xx),sources=len(set(ss)),
                native_span=summarize([r['metrics'][baseline]['span'] for r in xx],ss),
                native_sIoU=summarize([r['metrics'][baseline]['sIoU'] for r in xx],ss),
                GT_time_delta_sIoU=summarize([r['comparisons'].get('time_GT_minus_native_sIoU') for r in xx],ss))
        for section in ['smoothing']:
            c[section]={}
            for name in sorted(set.union(*(set(r[section]) for r in rr))):
                xx=[r for r in rr if name in r[section]]
                c[section][name]=summarize([delta(r[section][name]['sIoU'],r['metrics'][baseline]['sIoU']) for r in xx],[r['source'] for r in xx])
        c['substitutions']={}
        for name in sorted(set.union(*(set(r['substitutions']) for r in rr))):
            xx=[r for r in rr if name in r['substitutions']];ss=[r['source'] for r in xx]
            c['substitutions'][name]={m:summarize([r['substitutions'][name][m] for r in xx],ss) for m in ['vIoU_corrected','sIoU','tIoU']}
        output[tag]=c
    summary=dict(complete=complete and not unavailable,all_available_complete=complete,unavailable=unavailable,
        stage=stage,backbone=b,cohort=cohort,finished=len(results),total=len(p['rows'][cohort]),
        updated=time.time(),descriptive_test_replication=True,configuration_selection=False,all_test_rows_retained=True,
        partially_annotated_query_count=sum(r['accounting']['missing_event_GT']>0 for r in results),groups=output)
    if complete:
        if not (dest/'summary.json').exists():write(dest/'summary.json',summary);write(dest/'analysis_rows.json',results)
    else:status(dest/'PARTIAL_summary.json',summary)
    print('SCORED',stage,b,cohort,len(results),'complete',complete,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);ap.add_argument('--backbone',required=True);ap.add_argument('--cohort',required=True);ap.add_argument('--partial',action='store_true');a=ap.parse_args();run(a.stage,a.backbone,a.cohort,a.partial)
