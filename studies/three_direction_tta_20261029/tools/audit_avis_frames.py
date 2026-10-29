"""Independent per-frame complete-instance-set accuracy, not native FSLA.

Drop empty masks by decoded area, require identical category multiplicities and
an IoU-threshold-feasible perfect matching. This diagnostic deliberately has no
global trajectory assignment. Keep its name distinct from released FSLA.
"""
import hashlib
import json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment
from pycocotools import mask as masks

ROOT=Path(__file__).resolve().parents[1]
STUDIES=['avis_prefix_dev_v3','avis_full_context_v2']


def sha(path):return hashlib.file_digest(Path(path).open('rb'),'sha256').hexdigest()


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


def main():
    recovered=ROOT/'research/avis_tail_recovery/test.json.recovered'
    proof=json.loads((recovered.parent/'RECOVERY_RECEIPT.json').read_text())
    assert sha(recovered)==proof['payload_sha256']
    # Verify all14 arm seals before opening target annotations.
    specifications=[]
    for name in STUDIES:
        study=ROOT/'avis/studies'/name
        plan=json.loads((study/'PLAN_LOCK.json').read_text())
        assert sha(study/'PLAN_LOCK.json')==(study/'PLAN_LOCK.json.sha256').read_text().split()[0]
        inputs=json.loads((study/'inputs.json').read_text())
        assert sha(study/'inputs.json')==plan['manifest_sha256']
        arms=[]
        for p in sorted((study/'runs').iterdir()):
            if not (p/'protocol.json').is_file():continue
            protocol=json.loads((p/'protocol.json').read_text())
            assert protocol['state']=='sealed'
            assert sha(p/'predictions.json')==protocol['prediction_sha256']
            assert protocol['manifest_sha256']==sha(study/'inputs.json')
            arms.append((p.name,p/'predictions.json'))
        assert len(arms)==len(plan['arms'])
        specifications.append((name,inputs,arms))
    gt=json.loads(recovered.read_text())
    videos={Path(v['file_names'][0]).parts[0]:v for v in gt['videos']}
    rows=[]
    for name,inputs,arms in specifications:
        for arm,path in arms:
            predictions=json.loads(path.read_text())
            for v in inputs['videos']:
                reference=videos[v['video_id']]
                assert (reference['height'],reference['width'])==(v['height'],v['width'])
                truth=[a for a in gt['annotations'] if a['video_id']==reference['id']]
                pred=[a for a in predictions if a['video_id']==v['video_id']]
                value=accuracy(truth,pred,v['frame_indices'],v['height'],v['width'])
                rows.append({'study':name,'arm':arm,'video_key':hashlib.sha256(v['video_id'].encode()).hexdigest()[:16],**value})
    summaries=[]
    for study in STUDIES:
        for arm in sorted({r['arm'] for r in rows if r['study']==study}):
            subset=[r for r in rows if r['study']==study and r['arm']==arm]
            total=sum(x['n_frames'] for x in subset)
            correct=np.sum([r['correct_counts_per_threshold'] for r in subset],axis=0)
            summaries.append({'study':study,'arm':arm,'n_frames':total,
                              'frame_weighted_accuracy':float(correct.mean()/total)})
    output={'prediction_seals_verified':True,'target_payload_sha256':sha(recovered),
            'metric':'per-frame complete-set accuracy averaged overIoU>.05:.05:.95',
            'not_native_FSLA':True,'empty_masks':'area==0 is absent for bothGT and predictions',
            'matching':'per-threshold class-compatible perfect matching; no global trajectory assignment',
            'scope':'fixed8 convenience videos; labels only accessed after14 arm prediction seals',
            'rows':rows,'summaries':summaries}
    path=ROOT/'research/avis_frame_exact_audit.json'
    with path.open('x') as f:json.dump(output,f,indent=2)
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
