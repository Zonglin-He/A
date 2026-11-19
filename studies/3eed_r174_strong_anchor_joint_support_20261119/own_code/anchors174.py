"""Candidate preparation with a genuine R158 winner and original A256 PCA boxes."""
from functools import lru_cache
import sys

import numpy as np
import torch

from scope174 import R97_DATA, ROOT, PLATFORMS, read, sha
sys.path.insert(0,str(ROOT/'research/round158'))
from grounding158 import select_grounding_box
from reader158 import overlaps
from crossplatform.ground_single_frame import orient_box


def diversify(a, utilities, first):
    """Stable R158-utility order with at most moderate overlap among preferred candidates."""
    order=np.argsort(-utilities,kind='stable')
    assert order[0]==first
    # Use the same axis-aligned IoU convention as R158 on the shortlisted boxes.
    kernel=overlaps(a)
    keep=[int(first)]
    for j in order[1:]:
        if (kernel[j,keep]<=.35).all():
            keep.append(int(j))
            if len(keep)==4:
                break
    if len(keep)<4:
        keep.extend(int(j) for j in order if int(j) not in keep)
    return np.asarray(keep[:4],np.int64)


def prepare(a, prob, sampled, ref_choice):
    first, utility=select_grounding_box(a,prob)
    strong=diversify(a,utility,first)
    scores=utility[strong]
    log=np.log(np.maximum(scores,1e-80))
    log=(log-log[0]).clip(-20.,0.).astype(np.float32)
    sub=np.stack([orient_box(sampled,a[k]) for k in strong])
    ref=np.stack([orient_box(sampled,a[k]) for k in ref_choice])
    assert len(set(strong))==len(set(ref_choice))==4
    return dict(subject_ids=strong, reference_ids=np.asarray(ref_choice,np.int64),
                base_logit=log, subject_boxes=sub, reference_boxes=ref)


@lru_cache(maxsize=6)
def r97_bundle(source,target):
    master=read(R97_DATA/'MATRIX_SEALED.json')
    assert not master['FINAL_access'] and not master['target_task_GT_read']
    p=R97_DATA/f'{source}_to_{target}'
    item=read(p/'SEALED.json')
    assert sha(p/'SEALED.json')==master['seals_sha256'][p.name]
    assert not item['FINAL_access'] and not item['target_task_GT_read']
    return p,item['files']


def original_target_score(source, row, a, ids):
    path,files=r97_bundle(source,row['platform'])
    n=f'samples_PRIVATE/{row["id"]}.npz'
    assert sha(path/n)==files[n]
    with np.load(path/n,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['source_boxes'],a)
        np.testing.assert_array_equal(z['point_indices'],ids)
        assert int(z['anno_index'])==row['anno_index']
        return z['scores'][z['arm_names'].tolist().index('prepared_frozen')]
