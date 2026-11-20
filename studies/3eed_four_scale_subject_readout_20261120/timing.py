"""Paired single-thread CPU time from the same four partitions to the subject box."""
import argparse
from collections import defaultdict
from hashlib import sha1
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch

from replay import BASE, OUT, catalog, digest, regions, row_id, save, selection
from score_roles import csv_file


def old_module(package,name,expected):
    p = package/'crossplatform/physical_regions'/(name+'.py')
    raw = p.read_bytes()
    assert sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==expected
    spec = spec_from_file_location('crossplatform.physical_regions.old_'+name,p)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def timing(package):
    torch.set_num_threads(1)
    r0 = old_module(package,'regions','80822dabb1614872929e3a90eb9d5ad71473c091')
    s0 = old_module(package,'selection','78a730a310847c40749f68214079094fe658dab3')
    catalog_data = catalog()
    by_direction = defaultdict(list)
    for row in catalog_data['rows']:
        if row['scope']=='ordinary':
            by_direction[row['source'],row['target']].append(row)
    timings = []
    for (src,tgt),rows in sorted(by_direction.items()):
        for index,row in enumerate(sorted(rows,key=row_id)[:4]):
            frame = catalog_data['frames'][row['frame']]
            assert digest(frame['lex'])==frame['lh'] and digest(frame['word'])==frame['wh'] and digest(row['role'])==row['rh']
            with np.load(frame['lex']) as lex, np.load(frame['word']) as w, np.load(row['role']) as z:
                part = regions.build(lex['sample'],lex['logits'],lex['xyz'],frame['f'])
                word = dict(w)
                proposals = {k:z[k] for k in z.files if k.startswith('t_')}
                args = proposals,row['words'],int(z['side']),int(z['depth']),bool(z['immediate'])
            def earlier():
                d = r0.assemble(part,word)
                i,_ = s0.select(d,*args,r0.Returns(d['xyz']))
                return None if i<0 else d['completed'][i,row['words'][0]]
            def current():
                d = regions.build_inventory(part,word)
                i,_ = selection.select(d,*args,regions.Returns(d['xyz']),read_reference=False)
                return None if i<0 else regions.complete(d['boxes'][i],d['dimensions'][i,row['words'][0]])
            np.testing.assert_array_equal(earlier(),current())  # warm both paths
            times = [[],[]]
            for rep in range(6):
                order = (0,1) if (index+rep)%2==0 else (1,0)
                for j in order:
                    start = perf_counter_ns()
                    (earlier,current)[j]()
                    times[j].append((perf_counter_ns()-start)/1e6)
            before,after = map(np.median,times)
            timings.append(dict(source=src,target=tgt,sample=index,
                                previous_ms=before,subject_only_ms=after,saved_ms=before-after))
    assert len(timings)==24
    csv_file(OUT/'timing.csv',timings)
    save(OUT/'timing.json',dict(rows=len(timings),repetitions_per_version=6,warmup_runs_per_version=1,
        selection='first four ordinary registered rows by label-free digest in each transfer direction',
        timed='region inventory, cached subject nominations to overlap votes, joint decision, box completion',
        excludes='point and language networks, phrase forward, raw frame IO, segmentation and fallback',
        median_previous_ms=float(np.median([r['previous_ms'] for r in timings])),
        median_subject_only_ms=float(np.median([r['subject_only_ms'] for r in timings])),
        median_paired_saved_ms=float(np.median([r['saved_ms'] for r in timings])),
        positive_paired_savings=sum(int(r['saved_ms']>0) for r in timings)))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous_package',type=Path)
    timing(parser.parse_args().previous_package)
