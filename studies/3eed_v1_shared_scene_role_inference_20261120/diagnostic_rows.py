"""Descriptive case attribution after task scoring. Recount escapes by raw LiDAR content."""
from collections import Counter,defaultdict
import csv
import gzip
import numpy as np

from shared_paths import NEW,OLD,OUT,read,save,anonymous,digest,inputs
from scope183 import ordinary,VIEWS
from panel_witness import nearest,inventory,projection
from score_roles import csv_file


def diagnostic():
    index={(t,anonymous((t,i))):r for t,v in ordinary().items() for i,r in enumerate(v)}
    with (OUT/'terrain_certified.csv').open() as f:c=list(csv.DictReader(f))
    scans={digest(index[r['target'],r['case']]['pointpath']) for r in c}
    with (OUT/'terrain_paired.csv').open() as f:
        deltas=[r for r in csv.DictReader(f) if r['arm']=='local_two_halves' and r['control']=='v1' and r['direction']!='macro']
    save(OUT/'geometry_concentration.json',dict(escapes=len(c),physical_cloud_content_hashes=len(scans),
        recordings=len({r['recording'] for r in c}),direction=dict(Counter(r['source']+'_to_'+r['target'] for r in c)),
        same_cases_global_also_pass=sum(float(r['global_iou'])>.5 for r in c),
        same_cases_permuted_also_pass=sum(float(r['permuted_iou'])>.5 for r in c),
        median_before=float(np.median([float(r['v1_iou']) for r in c])),
        median_after=float(np.median([float(r['local_iou']) for r in c])),
        median_best_original=float(np.median([max(float(r['best_original_PCA']),float(r['best_original_zero'])) for r in c])),
        mean_other_four_directions=float(np.mean([float(r['delta50']) for r in deltas if not r['direction'].endswith('_to_waymo')])),
        exposed_ordinary_evaluation=True))
    queries=read(NEW/'roster.json');labels=read(NEW/'labels.json');physical={}
    for r in queries:
        f=r['frame'];e=labels[r['uid']];k=e['pair'],e['half']
        if k in physical:continue
        view=VIEWS.index(f['waymo_view']);target=f['target']
        cls,b,conflict=inventory(dict(metapath=f['calibration'],platform=target,suffix=view))
        assert not conflict and np.array_equal(b,e['truth'])
        px=projection(b,target,view,f['calibration'],f['rgb']);t=e['subject'];ref=e['reference']
        side='right' if e['half'] else 'left'
        synthetic=px.copy();synthetic[ref,[0,2,3]]=.5
        physical[k]=(nearest(cls[t],ref,side,cls,synthetic)==t,float(px[t,0]-.5))
    def properties(p):
        a,b=physical[p,0],physical[p,1]
        return ('camera_center_suffices' if a[0] and b[0] else 'camera_center_insufficient',
                'both_on_one_side' if a[1]*b[1]>0 else 'straddle_image_center')
    allgroups=defaultdict(list)
    for f,keys in [('endpoint_rows',('both_subjects','both_refs','four')),
                   ('contrast_rows',('available','both_correct','either_tie','both_reversed'))]:
        with gzip.open(OUT/(f+'.csv.gz'),'rt') as src:
            for r in csv.DictReader(src):
                if r['panel']!='new_side':continue
                for g in properties(r['pair']):allgroups[f,g,r['arm'],r['rule']].append(r)
        data=[]
        for (q,g,arm,mode),rs in sorted(allgroups.items()):
            if q==f:data.append(dict(subset=g,arm=arm,rule=mode,pairs=len(rs),frames=len({x['frame'] for x in rs}),
                **{k:sum(int(r[k]) for r in rs) for k in keys}))
        csv_file(OUT/('center_'+f.removesuffix('_rows')+'.csv'),data)


if __name__=='__main__':diagnostic()
