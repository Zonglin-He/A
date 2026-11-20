"""Evaluate the unchanged floor-only component on every frozen ordinary v1 endpoint."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import csv
from pathlib import Path

import numpy as np

from shared_paths import HERE, ROOT, WORK, OUT, read, save, digest, archive_npz, anonymous
from common_lex import sample
from scope183 import ordinary,frame,check_input
from scene_partition import terrain
from scene_faces import split_returns,bottom
from crossplatform.round7.camera_projection import WAYMO_ANGLES,z_rotation

BASE=HERE.parent/'3eed_v1_subject_margin_attribution_20261120/run_PRIVATE'
FOLLOW=OUT/'terrain_followup_protocol.json'
DST=WORK/'terrain_full'
ARMS=('v1','local_two_halves','global_same_eligible','wrong_recording_location')


def boxes_in_model(b,target,view):
    x=b.copy()
    if target=='waymo':
        angle=WAYMO_ANGLES[view];x[:,:3]=x[:,:3]@z_rotation(angle).T
        x[:,6]=(x[:,6]+np.deg2rad(angle)+np.pi/2)%np.pi-np.pi/2
    elif target=='drone':x[:,2]+=1.8
    return x


def read_winners():
    cert=read(BASE/'SEALED.json')
    assert digest(BASE/'SEALED.json')==read(FOLLOW)['parent']
    assert cert['n']==18620 and not cert['FINAL_access'] and not cert['target_labels_used']
    result={}
    for direction,sha in cert['directions'].items():
        folder=BASE/direction
        assert sha==digest(folder/'SEALED.json')
        block=read(folder/'SEALED.json')
        winners=np.full((block['n'],7),np.nan);ids=np.full(block['n'],-1,int)
        for f,v in block['files'].items():
            assert v==digest(folder/f)
            with np.load(folder/f) as z:
                q=z['slots'][np.arange(len(z['ordinals'])),z['choices'][:,0]]
                winners[z['ordinals']]=z['pool'][np.arange(len(q)),q]
                ids[z['ordinals']]=q
        assert np.isfinite(winners).all() and (ids>=0).all()
        result[direction]=(winners,ids)
    return result


def one(task):
    row,boxes=task;f=frame(row)
    check_input(row)
    xyz=np.unique(sample(f).astype(float),axis=0)
    plane=terrain(xyz);split=split_returns(xyz)
    anchor=boxes_in_model(boxes,row['platform'],row['suffix'])
    values=[]
    for box in anchor:
        original=box[2]-box[5]/2
        loc=bottom(box,xyz,plane,split)
        delta=loc-original
        global_z=float(box[:2]@plane[:2]+plane[2])
        gd=global_z-original if delta!=0 and np.isfinite(global_z) and global_z<original+box[5] else 0.
        values.append([delta,gd])
    return values


def capture():
    DST.mkdir(exist_ok=True,mode=0o700)
    roster=ordinary();winners=read_winners()
    counts=defaultdict(int)
    for target,rows in roster.items():
        dirs=sorted(k for k in winners if k.endswith('_to_'+target))
        assert len(dirs)==2 and all(len(rows)==len(winners[k][0]) for k in dirs)
        jobs=[(v,np.stack([winners[k][0][i] for k in dirs])) for i,v in enumerate(rows)]
        with ProcessPoolExecutor(max_workers=10) as ex:
            moves=np.asarray(list(ex.map(one,jobs,chunksize=12)))
        for d,direction in enumerate(dirs):
            base,q=winners[direction];frac=moves[:,d,0]/base[:,5];wrong=frac.copy()
            bins=defaultdict(list)
            for i,x in enumerate(rows):
                if frac[i]!=0:bins[x['recording']].append(i)
            for v in bins.values():
                v=sorted(v,key=lambda i:anonymous((direction,int(i))))
                if len(v)==1:counts['singletons']+=1;continue
                other=np.roll(v,len(v)//2)
                counts['permuted']+=len(v)
                counts['same_physical_path']+=sum(rows[i]['pointpath']==rows[j]['pointpath'] for i,j in zip(v,other))
                wrong[v]=frac[other]
                np.testing.assert_array_equal(np.sort(wrong[v]),np.sort(frac[v]))
            boxes=np.repeat(base[:,None,:],len(ARMS),axis=1)
            change=np.column_stack((np.zeros(len(rows)),moves[:,d,:],wrong*base[:,5]))
            boxes[:,:,2]+=change/2;boxes[:,:,5]-=change
            assert (boxes[:,:,3:6]>0).all() and np.isfinite(boxes).all()
            np.testing.assert_allclose(boxes[:,:,2]+boxes[:,:,5]/2,np.repeat((base[:,2]+base[:,5]/2)[:,None],4,axis=1),atol=1.e-10,rtol=0)
            archive_npz(DST/(direction+'.npz'),boxes=boxes,ordinal=np.arange(len(rows)),winner=q)
        print('ordinary',target,len(rows)*2,flush=True)
    save(DST/'SEALED.json',dict(protocol=digest(FOLLOW),files={k:digest(DST/(k+'.npz')) for k in winners},
        geometry_code={n:digest(HERE/n) for n in ('scene_partition.py','scene_faces.py','terrain_followup.py')},
        negative_control=dict(counts),labels_used=False,FINAL_access=False,source_training=0,target_updates=0))


def geometry_predictions():
    p=read(DST/'SEALED.json')
    assert not p['labels_used'] and not p['FINAL_access'] and p['protocol']==digest(FOLLOW)
    assert all(digest(HERE/n)==v for n,v in p['geometry_code'].items())
    for d,v in sorted(p['files'].items()):
        path=DST/(d+'.npz')
        assert digest(path)==v
        with np.load(path) as z:yield d,z['boxes'],z['winner']


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['predict','score'])
    if p.parse_args().stage=='predict':capture()
    else:
        from terrain_score import score
        score()
