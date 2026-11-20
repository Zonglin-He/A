"""Descriptive geometry diagnostic, run after the fixed experiment has been scored."""
import csv
import gzip

import numpy as np

from score_roles import ARMS, OUT, PANELS, PRIVATE, SOURCES, anon, csv_file, digest, inventory, many_iou, owners, read
from roles_joint import direction_tensor


def coordinates(box):
    return np.r_[box[:3], np.log(box[5]), direction_tensor(box[None])[0]]


def from_coordinates(value, original_yaw):
    xx, xy, yy = value[4:]
    eigenvalues, vectors = np.linalg.eigh([[xx, xy], [xy, yy]])
    if eigenvalues[0] <= 0:
        raise ValueError('The weighted horizontal shape is not positive definite')
    yaw = np.arctan2(vectors[1, 1], vectors[0, 1])
    if eigenvalues[1] - eigenvalues[0] <= 1e-12 * max(eigenvalues[1], 1.):
        yaw = original_yaw
    return np.r_[value[:3], np.sqrt(eigenvalues[::-1]), np.exp(value[3]), (yaw+np.pi/2)%np.pi-np.pi/2]


def diagnostic_boxes(raw, prediction, weight):
    # Extremely small weights cannot stably identify the unaveraged direction from saved floats.
    if weight < 1e-7:
        return np.stack([raw, raw]), False
    assert weight <= .5 + 1e-12
    base = coordinates(raw)
    difference = (coordinates(prediction) - base) / weight
    alternatives = [from_coordinates(base + a*difference, raw[6]) for a in (.5, 1.)]
    recovered = from_coordinates((1-weight)*base + weight*coordinates(alternatives[-1]), raw[6])
    assert abs(many_iou(recovered[None], prediction)[0]-1.) < 1e-7
    return np.stack(alternatives), True


def run():
    groups = {}
    with gzip.open(OUT/'all_rows.csv.gz','rt') as stream:
        for row in csv.DictReader(stream):
            if row['arm']=='ordered_full':
                groups[row['panel'],row['case'],row['source']]=row
    seal=read(PRIVATE/'SEAL.json')
    results=[]
    for panel, path in PANELS.items():
        for i,entry in enumerate(read(path)):
            row=entry['row']
            target=row['platform']
            objects,_=inventory(row['metapath'],target,int(row['suffix'] or 0),int(row['object_index']))
            for source in SOURCES:
                if panel!='source_proxy' and source==target:
                    continue
                case=anon(f'{panel}|{i}')
                primary=groups[panel,case,source]
                file=PRIVATE/f'{panel}_{i:03d}_{source}.npz'
                assert digest(file)==seal['files'][file.name]
                with np.load(file,allow_pickle=False) as data:
                    selected=ARMS.index('ordered_full')
                    raw=data['raw'][selected]
                    pair,valid=diagnostic_boxes(raw,data['boxes'][selected],float(data['fraction'][selected]))
                ious=many_iou(pair,objects[0])
                who=owners(pair,objects)
                results.append(dict(group=primary['group'],source=source,target=target,case=case,path_identified=int(valid),
                    both_original_subjects_correct=int(primary['raw_subject']==primary['baseline_owner']=='T'),
                    weight=float(primary['shape_weight']),raw_iou=float(primary['raw_iou']),deployed_iou=float(primary['iou']),
                    half_iou=float(ious[0]),end_iou=float(ious[1]),half_still_correct=int(who[0]==0),
                    end_still_correct=int(who[1]==0),best_all=float(primary['best_all_axis_and_pca'])))
    assert len(results)==len(groups)==501
    with gzip.open(OUT/'shape_path_rows.csv.gz','wt',newline='') as f:
        w=csv.DictWriter(f,results[0]);w.writeheader();w.writerows(results)
    totals=[]
    for group in ('own_proxy','cross_proxy','natural170','natural171'):
        all_rows=[r for r in results if r['group']==group]
        fixed=[r for r in all_rows if r['both_original_subjects_correct']]
        for name in ('deployed','half','end','best_from_three'):
            vals=[max(r['raw_iou'],r['half_iou'],r['end_iou']) if name=='best_from_three' else r[name+'_iou'] for r in fixed]
            totals.append(dict(group=group,comparison=name,all_rows=len(all_rows),same_original_correct=len(fixed),
                identified_direction=sum(r['path_identified'] for r in fixed),
                rescue50=sum(r['raw_iou']<=.5<v for r,v in zip(fixed,vals)),
                harm50=sum(v<=.5<r['raw_iou'] for r,v in zip(fixed,vals)),
                outside_all_original=sum(r['best_all']<=.5<v for r,v in zip(fixed,vals)),
                lost_identifiable_subject=sum(not r[name+'_still_correct'] for r in fixed) if name in ('half','end') else None))
    csv_file(OUT/'shape_path.csv',totals)


if __name__=='__main__':
    run()
