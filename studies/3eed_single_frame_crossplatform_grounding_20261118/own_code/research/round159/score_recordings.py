"""Score the sealed single-frame transfer on other target recordings, using two OBB implementations."""
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = ROOT/'crossplatform/round159'
sys.path.insert(0, str(ROOT/'research/round154'))
from score154 import DATA, ground_truth, many_iou, oriented_iou
from summarize import csv_file, summarize

ARMS = ('old_axis','old_pca','r158_pca')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotation(angle):
    c,s=math.cos(angle),math.sin(angle)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])


def main():
    evaluator=read(HERE/'experiment/EVALUATOR_LOCK.json')
    for name,expected in evaluator['sha256'].items():
        if digest(ROOT/name)!=expected:
            raise ValueError('Locked evaluator or forward runner changed: '+name)
    pred_seal=read(HERE/'RECORDING_PREDICTION_SEAL.json')
    input_seal=read(HERE/'RECORDING_INPUT_SEAL.json')
    manifest=WORK/'RECORDING_INPUTS_PRIVATE.json'
    assert pred_seal['lock_sha256']==input_seal['lock_sha256']==digest(HERE/'experiment/RECORDING_CHECK.json')
    assert input_seal['input_sha256']==digest(manifest) and not pred_seal['target_grounding_labels_read']
    rows=read(manifest)
    indices={r['id']:(r['meta_path'],r['object_index']) for r in read(ROOT/'research/round69/input/SCORING_INDEX_PRIVATE.json')}
    indices.update({r['id']:(r['metapath'],r['object_index']) for r in read(ROOT/'research/round76/input/SOURCE_TRAIN512_PRIVATE.json')})
    root_data=ROOT/'crossplatform/dataset_full/3eed_dataset'
    outcome=[]; checks=0; err=0.
    for source,seal in pred_seal['sources'].items():
        p=WORK/(source+'_PRIVATE.npz')
        assert digest(p)==seal['sha256'] and seal['input_sha256']==input_seal['input_sha256']
        with np.load(p,allow_pickle=False) as saved:
            ids,box,win=saved['rows'],saved['boxes'],saved['winners']
        assert list(ids)==[i for i,r in enumerate(rows) if r['platform']!=source]
        assert len(ids)==seal['n'] and box.shape==(len(ids),3,7)
        for k,i in enumerate(ids):
            r=rows[i];meta=Path(r['metapath']);record,object_idx=indices[r['id']]
            if Path(record).resolve()!=meta.resolve():
                raise ValueError('Caption scoring index has a different frame')
            project_view=DATA/meta.relative_to(root_data)
            if not project_view.samefile(meta):
                raise ValueError('Native model and scoring view refer to different inputs')
            # R76 retained native comma spacing. The shared join compares against tokenized captions.
            expected_caption = ' '.join(r['caption'].replace(',', ' ,').split())
            gt=ground_truth(r['platform'],str(project_view),int(object_idx),expected_caption)
            uniq,inv=np.unique(box[k].astype(np.float64),axis=0,return_inverse=True)
            a=many_iou(uniq,gt)
            b=np.array([oriented_iou(g[:3],g[3:6],rotation(g[6]),gt[:3],gt[3:6],rotation(gt[6])) for g in uniq])
            err=max(err,float(np.abs(a-b).max())); checks+=len(a)
            if err>1e-7 or not (np.array_equal(a>.25,b>.25) and np.array_equal(a>.5,b>.5)):
                raise ValueError('Independent metrics disagree')
            outcome.append(dict(ordinal=len(outcome)+1,source=source,target=r['platform'],
                                scene_cluster=r['scene_cluster'],point_content_cluster=r['point_content_cluster'],
                                changed_query=int(win[k,0]!=win[k,1]),**dict(zip(ARMS,map(float,a[inv])))))
    assert len(outcome)==pred_seal['total']==2*len(rows) and len(outcome)>0
    ds,mean,paired=summarize('other_recordings_seed0',outcome,dict(zip(ARMS,ARMS)),list(zip(ARMS,ARMS[1:])))
    groups=defaultdict(list)
    for r in outcome:
        groups[r['source'],r['target'],r['scene_cluster']].append(r)
    scene_rows=[]
    for (s,t,sc),group in sorted(groups.items()):
        for arm in ARMS:
            y=np.array([r[arm] for r in group])
            scene_rows.append(dict(source=s,target=t,scene_cluster=sc,n=len(y),arm=arm,mIoU=100*y.mean(),
                                   Acc25=100*(y>.25).mean(),Acc50=100*(y>.5).mean()))
    for name,items in [('all_rows',outcome),('directions',ds),('overall',mean),('changes',paired),('scenes',scene_rows)]:
        csv_file('recording_'+name+'.csv',items)
    audit=dict(status='PASS',direction_records=len(outcome),target_expressions=len(rows),
               target_recordings=input_seal['recording_ids'],scored_after_prediction_seal=digest(HERE/'RECORDING_PREDICTION_SEAL.json'),
               independent_oriented_IoU_pairs=checks,maximum_IoU_difference=err,threshold_agreement=True,
               source_training_runs=0,target_parameter_updates=0,all_negative_rows_preserved=True,FINAL_access=False)
    (HERE/'RECORDING_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(mean,indent=2),flush=True)


if __name__=='__main__':
    main()
