"""Fixed-grid temporal/spatial error accounting and controlled attention masks."""
from collections import defaultdict
import numpy as np
import torch
from vg_tta.metrics import aligned_box_iou_cxcywh, temporal_iou


def frame_iou(boxes, targets, frame_ids, gt_interval):
    ids=np.asarray(frame_ids,dtype=np.int64)
    valid=(ids>=gt_interval[0]) & (ids<gt_interval[1])
    quality=np.full(len(ids),np.nan)
    for i, target in enumerate(targets):
        present=len(target['boxes'])>0
        assert present==bool(valid[i]), 'Missing/extra GT annotation on fixed grid'
        if present:
            assert len(target['boxes'])==1
            assert bool((target['boxes'][0,2:]>0).all()), 'Degenerate GT box'
            quality[i]=float(aligned_box_iou_cxcywh(boxes[i:i+1].float().cpu(),target['boxes'][:1].float().cpu())[0])
    assert valid.any() and np.isfinite(quality[valid]).all()
    return quality, valid


def physical_score(quality, frame_ids, gt_interval, prediction_interval):
    """Same corrected sampled-grid vIoU, with exact physical output interval.

    Undefined GT boxes outside the event are never invented/evaluated. They
    still incur the same temporal-union penalty as the native evaluator.
    """
    ids=np.asarray(frame_ids,dtype=np.int64)
    a,b=map(int,prediction_interval);g,h=map(int,gt_interval)
    assert a<b and g<h
    valid=(ids>=g)&(ids<h)
    assert np.isfinite(quality[valid]).all()
    intersection=valid&(ids>=a)&(ids<b)
    union=(ids>=min(a,g))&(ids<max(b,h))
    numerator=float(quality[intersection].sum())
    denominator=int(union.sum())
    return {'vIoU_corrected':numerator/max(denominator,1),'tIoU':temporal_iou((a,b),(g,h)),
            'sIoU':float(quality[valid].mean()),'numerator':numerator,'denominator':denominator,
            'prediction_interval':[a,b],'gt_interval':[g,h]}


def inclusion(frame_ids, interval):
    ids=np.asarray(frame_ids)
    return (ids>=interval[0])&(ids<interval[1])


def ratio(n,d): return float(n/d) if d else None


def error_accounting(quality, frame_ids, gt_interval, native_interval, adapted_interval):
    gt=inclusion(frame_ids,gt_interval);before=inclusion(frame_ids,native_interval);after=inclusion(frame_ids,adapted_interval)
    gained=after&~before;lost=before&~after
    inside_before=gt&before;inside_after=gt&after
    result={'gt_sample_count':int(gt.sum()),'newly_included_count':int(gained.sum()),
        'newly_excluded_count':int(lost.sum()),'newly_included_GT_count':int((gained&gt).sum()),
        'newly_included_outside_GT_count':int((gained&~gt).sum()),
        'added_outside_GT_fraction':ratio((gained&~gt).sum(),gained.sum()),
        'native_inside_GT_mean_IoU':float(quality[inside_before].mean()) if inside_before.any() else None,
        'adapted_inside_GT_mean_IoU':float(quality[inside_after].mean()) if inside_after.any() else None,
        'native_temporal_GT_recall':ratio(inside_before.sum(),gt.sum()),
        'adapted_temporal_GT_recall':ratio(inside_after.sum(),gt.sum()),
        'native_intersection_low_IoU_fraction':ratio((inside_before&(quality<.3)).sum(),inside_before.sum()),
        'adapted_intersection_low_IoU_fraction':ratio((inside_after&(quality<.3)).sum(),inside_after.sum()),
        'B_any_low_IoU_in_native_intersection':bool((inside_before&(quality<.3)).any()),
        'C_any_added_outside_GT':bool((gained&~gt).any())}
    for threshold in [.3,.5,.7]:
        good=gt&(quality>=threshold);excluded=good&~before;recovered=excluded&after
        prefix=f'good_{threshold:g}'
        result.update({prefix+'_GT_count':int(good.sum()),prefix+'_excluded_native_count':int(excluded.sum()),
            prefix+'_recovered_count':int(recovered.sum()),prefix+'_recovery_fraction':ratio(recovered.sum(),excluded.sum()),
            prefix+'_native_coverage':ratio((good&before).sum(),good.sum()),
            prefix+'_adapted_coverage':ratio((good&after).sum(),good.sum()),
            prefix+'_lost_count':int((good&lost).sum()),prefix+'_fraction_of_added':ratio((good&gained).sum(),gained.sum())})
    result['A_any_good_GT_excluded_native']=result['good_0.5_excluded_native_count']>0
    result['A_any_recovered_good_GT']=result['good_0.5_recovered_count']>0
    return result


def mean_ci(values, sources, seed=20260909):
    grouped=defaultdict(list)
    for v,s in zip(values,sources):
        if v is not None and np.isfinite(v):grouped[s].append(float(v))
    if not grouped:return {'mean':None,'ci95':None,'eligible_queries':0,'sources':0}
    array=np.array([np.mean(v) for s,v in sorted(grouped.items())])
    boot=array[np.random.default_rng(seed).integers(0,len(array),(10000,len(array)))].mean(1)
    return {'mean':float(array.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),
        'eligible_queries':sum(map(len,grouped.values())),'sources':len(array)}


def contiguous_controls(gt_mask, frame_ids, seeds=(20260909,20260910,20260911)):
    """Match key count and index span; closest physical span, no circular wrap."""
    gt=np.asarray(gt_mask,bool);ids=np.asarray(frame_ids);positions=np.where(gt)[0]
    assert len(positions) and np.all(np.diff(positions)==1)
    k=len(positions);wanted=ids[positions[-1]]-ids[positions[0]]
    starts=[s for s in range(len(ids)-k+1) if s!=positions[0]]
    if not starts: starts=[int(positions[0])]
    distance=min(abs((ids[s+k-1]-ids[s])-wanted) for s in starts)
    starts=[s for s in starts if abs((ids[s+k-1]-ids[s])-wanted)==distance]
    outputs={}
    for seed in seeds:
        start=int(np.random.default_rng(seed).choice(starts));mask=np.zeros(len(ids),bool);mask[start:start+k]=True
        outputs[f'control_{seed}']={'mask':mask,'start_index':start,'key_count':k,
            'physical_span_mismatch':int(distance),'same_as_GT':bool(np.array_equal(mask,gt)),
            'GT_mask_IoU':float((mask&gt).sum()/(mask|gt).sum())}
    return outputs
