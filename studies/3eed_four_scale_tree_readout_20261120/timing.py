"""Time the graph and the subject decision independently on fixed cached inputs."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from replay import OUT, catalog, digest, previous, regions, row_id, save, selection


def times(before,after,index):
    results=[[],[]]
    for rep in range(8):
        for j in ((0,1) if (rep+index)%2==0 else (1,0)):
            start=perf_counter_ns()
            (before,after)[j]()
            results[j].append((perf_counter_ns()-start)/1e6)
    return tuple(map(float,map(np.median,results)))


def run(package):
    r0,s0=previous(package)
    c=catalog()
    groups=defaultdict(list)
    for r in c['rows']:
        if r['scope']=='ordinary':groups[r['source'],r['target']].append(r)
    rows=[]
    for (src,tgt),records in sorted(groups.items()):
        for idx,row in enumerate(sorted(records,key=row_id)[:4]):
            f=c['frames'][row['frame']]
            assert digest(f['lex'])==f['lh'] and digest(f['word'])==f['wh'] and digest(row['role'])==row['rh']
            with np.load(f['lex']) as lex,np.load(f['word']) as words,np.load(row['role']) as role:
                args=lex['sample'],lex['logits'],lex['xyz'],f['f']
                old_seg=r0.build(*args)
                seg=regions.build(*args,native_logits=True)
                old=r0.build_inventory(old_seg,words)
                new=regions.build_inventory(seg,words)
                proposals={k:role[k] for k in role.files if k.startswith('t_')}
                params=(proposals,row['words'],int(role['side']),int(role['depth']),bool(role['immediate']),
                        regions.Returns(new['xyz']))
            xyz=seg['xyz'].astype(float)
            plane=seg['plane'];fg=xyz[:,2]-xyz[:,:2]@plane[:2]-plane[2]>.20
            def old_graph():return r0.levels(xyz,fg)
            def new_graph():return regions.levels(xyz,fg)
            def old_read():return s0.select(old,*params,read_reference=False)
            def new_read():return selection.select(new,*params,read_reference=False)
            assert old_read()==new_read()
            np.testing.assert_array_equal(old_graph()[1],new_graph()[1])
            g0,g1=times(old_graph,new_graph,idx)
            r0_ms,r1_ms=times(old_read,new_read,idx)
            rows.append(dict(source=src,target=tgt,index=idx,regions=len(new['parent']),
                             graph_previous_ms=g0,graph_tree_ms=g1,graph_saved_ms=g0-g1,
                             readout_previous_ms=r0_ms,readout_tree_ms=r1_ms,readout_saved_ms=r0_ms-r1_ms))
    assert len(rows)==24
    OUT.mkdir(exist_ok=True)
    with (OUT/'timing.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    summary={name:dict(previous_ms=float(np.median([r[name+'_previous_ms'] for r in rows])),
                       tree_ms=float(np.median([r[name+'_tree_ms'] for r in rows])),
                       paired_saved_ms=float(np.median([r[name+'_saved_ms'] for r in rows])),
                       faster=sum(r[name+'_saved_ms']>0 for r in rows)) for name in ('graph','readout')}
    save(OUT/'timing.json',dict(rows=24,timed_repeats_per_case_per_version=8,
        units='ms',cpu_threads=1,case_selection='first four registered ordinary digests per direction',
        graph='foreground through four partitions including neighbors and quantiles',
        readout='fixed region inventory and subject proposals through subject decision including nomination votes',
        summary=summary))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous_package',type=Path)
    run(parser.parse_args().previous_package)
