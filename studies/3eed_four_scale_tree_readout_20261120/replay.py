"""Verify the four-scale tree readout against pinned decisions without opening target labels."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent/'3eed_four_scale_subject_readout_20261120'
EIGHT = HERE.parent/'3eed_eight_scale_component_ablation_20261120'
sys.path.insert(0,str(EIGHT))
from common import ROOT, catalog, digest, read, row_id, save
from predict_delivery import to_model
from crossplatform.physical_regions import regions, selection, tree
from research.round158.reader158 import overlaps

CFG = read(HERE/'PROTOCOL.json')
OUT = HERE/'results'
PRIVATE = HERE/'private'


def pins():
    files=[__file__,regions.__file__,selection.__file__,tree.__file__,ROOT/'crossplatform/ground.py',
           ROOT/'crossplatform/ground_regions.py']
    return {p.name:digest(p) for p in map(Path,files)}


def previous(package):
    seal = read(BASE/'private/SEAL.json')
    assert digest(BASE/'private/SEAL.json') == CFG['previous_decisions_seal']
    def load(name):
        path = package/'crossplatform/physical_regions'/(name+'.py')
        assert digest(path)==seal['code'][path.name]
        spec = spec_from_file_location('crossplatform.physical_regions.baseline_'+name,path)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return load('regions'),load('selection')


def initialize(package):
    global old_regions, old_selection
    old_regions,old_selection = previous(package)


def close(a,b):
    np.testing.assert_allclose(a,b,rtol=1e-10,atol=2e-13)
    return float(np.max(abs(a-b),initial=0.))


def check_frame(task):
    frame,rows = task
    for path,key in [('lex','lh'),('word','wh')]:
        assert digest(frame[path])==frame[key]
    with np.load(frame['lex']) as lex, np.load(frame['word']) as word:
        args = lex['sample'],lex['logits'],lex['xyz'],frame['f']
        old_seg = old_regions.build(*args)
        new_seg = regions.build(*args,native_logits=True)
        for field in ('xyz','cuts','labels','plane'):
            np.testing.assert_array_equal(new_seg[field],old_seg[field])
        old = old_regions.build_inventory(old_seg,word)
        data = regions.build_inventory(new_seg,word)
    for k in ('parent','level','size','boxes','dimensions','view','stop','assign'):
        np.testing.assert_array_equal(data[k],old[k])
    qerr = close(data['p'],old['p'])
    parent,size,stop = data['parent'],data['size'],data['stop']
    occupancy,K = old_selection.structure(parent,stop,size)
    valid = K==0
    np.testing.assert_array_equal(tree.disjoint_pairs(parent),valid)
    pts = regions.Returns(data['xyz'])
    output = []
    for row,es,er,old_ids in rows:
        key = row_id(row)
        assert digest(row['role'])==row['rh'] and digest(row['prior'])==row['ph']
        with np.load(row['role']) as role, np.load(row['prior']) as fallback:
            words = tuple(row['words'])
            side,depth,nearest = int(role['side']),int(role['depth']),bool(role['immediate'])
            cached = {k:role[k] for k in role.files if k.startswith('t_')}
            w0 = old_selection.overlap_prior(old,cached,'t',pts)
            # PCA changes only yaw. Reconstruct the axis-aligned overlap used by
            # the native nominations; archived invalid sizes were floored for IoU.
            native_boxes=cached['t_box'][:,:6].copy()
            invalid=(native_boxes[:,3:]<=float(np.float32(1e-6))).any(1)
            native_boxes[invalid,3:]=0.
            raw=np.maximum(cached['t_p'],1e-300)/np.square(overlaps(native_boxes)).sum(1)
            nomination_error=close(raw/raw.sum(),cached['t_m'])
            props={**cached,'t_m':raw}
            w = selection.overlap_prior(data,props,'t',pts)
            werr = close(w,w0)
            if depth or nearest:
                j0,_ = old_selection.shared(parent,valid,stop,old['p'],words,old['view'],side,depth,nearest,'physical')
                j,_ = selection.shared(parent,valid,stop,data['p'],words,data['view'],side,depth,nearest,'physical')
                h = j.sum(1)
            else:
                j0 = old_selection.closed_joint(parent,stop,occupancy,old['p'],words,old['view'],valid,side)
                field = tree.SideEvidence(parent,stop,data['view'])
                h = field.marginal(data['p'][:,words[0]],data['p'][:,words[1]],side)
            h0 = j0.sum(1)
            np.testing.assert_array_equal(h>0,h0>0,err_msg=key)
            herr = close(h,h0)
            uerr = close(tree.overlap_sum(parent,size,w*h),K@(w0*h0))
            s,r = selection.select(data,props,words,side,depth,nearest,pts)
            assert selection.select(data,props,words,side,depth,nearest,pts,read_reference=False)==(s,-1)
            assert (s,r)==old_ids,(key,(s,r),old_ids)
            prior = to_model(row,fallback)
            def box(i,c,name):
                return regions.complete(data['boxes'][i],data['dimensions'][i,c]) if i>=0 else prior[name][0]
            sb,rb = box(s,words[0],'anchored'),box(r,words[1],'ref_anchored')
            np.testing.assert_array_equal(sb,es,err_msg=key)
            np.testing.assert_array_equal(rb,er,err_msg=key)
            output.append((key,sb,rb,(s,r,len(parent),np.count_nonzero(h>0),int(bool(depth or nearest))),
                           (qerr,nomination_error,werr,herr,uerr)))
    return output


def run(package,workers,limit):
    source_code = pins()
    assert digest(BASE/'private/decisions.npz')==CFG['previous_decisions']
    with np.load(BASE/'private/decisions.npz') as f:
        locked = {k:(s[0],r[0],tuple(map(int,d[3:5]))) for k,s,r,d in zip(f['keys'],f['subjects'],f['refs'],f['details'])}
    c = catalog()
    groups = defaultdict(list)
    for row in c['rows']:
        if row['scope'] in ('ordinary','noun'):
            groups[row['frame']].append((row,*locked[row_id(row)]))
    tasks = [(c['frames'][k],v) for k,v in sorted(groups.items())]
    if limit:
        tasks = tasks[:limit]
    entries = []
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(package,)) as pool:
        for i,batch in enumerate(pool.map(check_frame,tasks,chunksize=1),1):
            entries.extend(batch)
            if i%100==0:
                print(f'{i}/{len(tasks)} source-frame inventories matched',flush=True)
    if limit:
        print('Smoke test matched',len(entries),'rows',flush=True)
        return
    keys,subjects,refs,details,errors = zip(*sorted(entries))
    assert set(keys)==set(locked) and len(keys)==len(locked)==1422 and len(tasks)==1358
    PRIVATE.mkdir(exist_ok=True,mode=0o700)
    path=PRIVATE/'decisions.npz'
    tmp=PRIVATE/'decisions.tmp.npz'
    np.savez_compressed(tmp,keys=keys,subjects=subjects,refs=refs,details=details)
    tmp.replace(path)
    assert pins()==source_code
    save(PRIVATE/'SEAL.json',dict(previous=CFG['previous_decisions_seal'],decisions=digest(path),
          source_frames=len(tasks),rows=len(keys),protocol=digest(HERE/'PROTOCOL.json'),
          code=source_code,
          max_abs_error=dict(zip(('noun','proposal_mass','votes','joint_marginal','subject_utility'),np.max(errors,axis=0))),
          labels_used=False,FINAL_access=False))
    print('Sealed',len(keys),'exact subjects and diagnostic references',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('previous_package',type=Path)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--limit',type=int)
    a=p.parse_args()
    run(a.previous_package,a.workers,a.limit)
