"""Area-aware per-frame complete-set accuracy, distinct from native FSLA.
Copied unchanged metric functions from independent root audit, with source SHA provenance.
"""
from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment
from pycocotools import mask as masks
def rle(seg,h,w):
    if seg is None or seg==[]:return None
    if isinstance(seg,dict) and isinstance(seg['counts'],(str,bytes)):
        value=seg
    else:
        value=masks.frPyObjects(seg,h,w)
        if isinstance(value,list):value=masks.merge(value)
    assert value['size']==[h,w]
    return value if float(masks.area(value))>0 else None


def accuracy(ground,pred,frame_indices,h,w):
    levels=np.arange(.05,1.,.05)
    good=np.zeros(len(levels),int)
    by_type={k:{'n':0,'good':np.zeros(len(levels),int)} for k in ['none','single','multi']}
    empty_removed=0
    for local,original in enumerate(frame_indices):
        gt=[];pr=[]
        for t in ground:
            seg=rle(t['segmentations'][original],h,w)
            if seg is not None:gt.append((t['category_id'],seg))
        for t in pred:
            seg=rle(t['segmentations'][local],h,w)
            if seg is not None:pr.append((t['category_id'],seg))
            elif t['segmentations'][local] is not None:empty_removed+=1
        kind='none' if not gt else 'single' if len(gt)==1 else 'multi'
        by_type[kind]['n']+=1
        passed=np.zeros(len(levels),int)
        if Counter(x[0] for x in gt)==Counter(x[0] for x in pr):
            if not gt:passed[:]=1
            else:
                overlaps=masks.iou([x[1] for x in gt],[x[1] for x in pr],[0]*len(pr))
                compatible=np.equal.outer([x[0] for x in gt],[x[0] for x in pr])
                for i,level in enumerate(levels):
                    edges=(overlaps>level)&compatible
                    rows,cols=linear_sum_assignment(~edges)
                    passed[i]=int(edges[rows,cols].all())
        good+=passed;by_type[kind]['good']+=passed
    return {'n_frames':len(frame_indices),'mean_threshold_accuracy':float(good.mean()/len(frame_indices)),
            'correct_counts_per_threshold':good.tolist(),'empty_prediction_detections_removed':empty_removed,
            'strata':{k:{'n':v['n'],'correct_counts_per_threshold':v['good'].tolist(),
                         'mean_threshold_accuracy':float(v['good'].mean()/v['n']) if v['n'] else None}
                      for k,v in by_type.items()}}

