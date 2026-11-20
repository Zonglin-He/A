"""Check actual sparse construction and the interaction of two proposed trims."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
import numpy as np

from common import ARMS, HERE, WORK, OUT, catalog, digest, prior, row_id, save, read, archive_npz
from crossplatform.physical_regions import regions, selection
from predict_delivery import to_model, compare_pose
from score_roles import csv_file


def code():
    from pathlib import Path
    return {p.name:digest(p) for p in [HERE/'COMPRESSION.json',Path(__file__),Path(regions.__file__),Path(selection.__file__)]}


def job(item):
    fid,frame,rows=item
    data={};timing={}
    with np.load(frame['lex']) as f, np.load(frame['word']) as w:
        assert digest(frame['lex'])==frame['lh'] and digest(frame['word'])==frame['wh']
        xyz,logit,anchor,ext=f['sample'],f['logits'],f['xyz'],dict(w)
    order=[('eight',tuple(range(8))),('four',(0,2,5,7))]
    if int(fid[-1],16)%2:order.reverse()
    for name,levels in order:
        start=perf_counter()
        seg=regions.build(xyz,logit,anchor,frame['f'],levels)
        data[name]=regions.assemble(seg,ext)
        timing[name]=perf_counter()-start
    counts={k:len(v['size']) for k,v in data.items()}
    d=data['four'];points=regions.Returns(d['xyz'])
    files={}
    maxerr=0.
    for row in rows:
        assert digest(row['prior'])==row['ph'] and digest(row['role'])==row['rh']
        with np.load(row['prior']) as old,np.load(row['role']) as z,np.load(WORK/'boxes'/(row_id(row)+'.npz')) as stored:
            base=to_model(row,old)
            emitted_s=np.tile(base['anchored'][0],(2,1));emitted_r=np.tile(base['ref_anchored'][0],(2,1))
            idx=np.full((2,2),-1,int)
            for name in ('eight','four'):
                g=data[name]
                if not len(g['size']):continue
                side,depth,nearest=int(z['side']),int(z['depth']),bool(z['immediate'])
                words=row['words']
                m,K=selection.structure(g['parent'],g['stop'],g['size']);valid=K==0
                if depth or nearest:
                    J,_=selection.shared(g['parent'],valid,g['stop'],g['p'],words,g['view'],side,depth,nearest,'physical')
                else:J=selection.closed_joint(g['parent'],g['stop'],m,g['p'],words,g['view'],valid,side)
                us=selection.overlap_prior(g,z,'t',points);ur=selection.overlap_prior(g,z,'r',points)
                if name=='eight':
                    for key,node,c in zip(('subject','reference'),selection.medoid(J,K,us,ur,valid)[:2],words):
                        if node>=0:maxerr=max(maxerr,compare_pose(g['completed'][node,c],stored[key][0]))
                    continue
                for k,refprior in enumerate((ur,np.ones_like(ur))):
                    idx[k]=selection.medoid(J,K,us,refprior,valid)[:2]
                    for dest,node,c in zip((emitted_s,emitted_r),idx[k],words):
                        if node>=0:dest[k]=g['completed'][node,c]
            target=ARMS.index('four_scales')
            four_err=max(compare_pose(emitted_s[0],stored['subject'][target]),compare_pose(emitted_r[0],stored['reference'][target]))
            fn=WORK/'compact'/(row_id(row)+'.npz')
            archive_npz(fn,subject=emitted_s,reference=emitted_r,four_replay_error=np.array(four_err))
            files[row_id(row)]=digest(fn)
    assert maxerr<1e-5
    return files,dict(full_nodes=counts['eight'],four_nodes=counts['four'],full_sec=timing['eight'],four_sec=timing['four'],full_replay=maxerr)


def run():
    seal=read(WORK/'SEALED.json');c=catalog();r=defaultdict(list)
    for row in c['rows']:
        if row['scope'] in ('ordinary','noun'):r[row['frame']].append(row)
    (WORK/'compact').mkdir(exist_ok=True,mode=0o700)
    files={};timings=[]
    for row in c['rows']:
        if row['scope'] in ('ordinary','noun'):
            key=row_id(row);assert digest(WORK/'boxes'/(key+'.npz'))==seal['files'][key]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for block,dt in pool.map(job,[(f,c['frames'][f],v) for f,v in sorted(r.items())],chunksize=2):
            files.update(block);timings.append(dt)
    csv_file(WORK/'construction_time.csv',timings)
    save(WORK/'COMPACT_SEAL.json',dict(code=code(),base=digest(WORK/'SEALED.json'),files=files,frames=len(r),
         time=digest(WORK/'construction_time.csv'),labels_used=False,FINAL_access=False))


if __name__=='__main__':run()
