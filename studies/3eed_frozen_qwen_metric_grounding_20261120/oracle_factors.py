"""Post-seal attribution of the 7D prediction gap to four GT substitutions."""
import csv
from itertools import permutations
import numpy as np
from paths import OUT,PRIV,PLATFORMS,read,digest
from geometry import camera,rigid_xyz
from score import canonical_box,truth_for_slot,score_boxes,save_csv

FACTORS=('depth','bearing','size','yaw')

def run():
    assert read(OUT/'geometry_seal.json')['predictions']==digest(PRIV/'predictions.json')
    raw=read(PRIV/'predictions.json')
    with (OUT/'all_rows.csv').open() as stream:scored={r['uid']:r for r in csv.DictReader(stream)}
    rows=[]
    for r in read(PRIV/'panel.json'):
        uid=r['uid'];t=r['target'];record=raw[uid]
        if not record['sensor_boxes']:continue
        orig=canonical_box(record['sensor_boxes']['raw'])
        gt=canonical_box(truth_for_slot(r,read(r['calibration'])['ground_info']))
        _,_,mat=camera(r);inv=np.linalg.inv(mat)
        z0=rigid_xyz(orig[:3],mat);z1=rigid_xyz(gt[:3],mat)
        assert min(z0[2],z1[2])>0
        boxes=[]
        for mask in range(16):
            z=(z1 if mask&1 else z0)[2]
            bearing=z1 if mask&2 else z0
            center=rigid_xyz(bearing/bearing[2]*z,inv)
            dims=gt[3:6] if mask&4 else orig[3:6]
            angle=gt[6] if mask&8 else orig[6]
            boxes.append(np.r_[center,dims,angle])
        ious,_=score_boxes(np.stack(boxes),gt)
        for mask,k in [(0,'raw'),(1,'gt_z'),(2,'gt_bearing'),(3,'gt_center'),(4,'gt_size'),(8,'gt_yaw'),(12,'gt_geometry')]:
            assert abs(ious[mask]-float(scored[uid][k]))<1e-7
        assert abs(ious[-1]-1)<1e-7
        metrics=np.column_stack([ious,ious>.25,ious>.5]);contribution=np.zeros((4,3))
        for order in permutations(range(4)):
            mask=0
            for k in order:
                new=mask|(1<<k)
                contribution[k]+=metrics[new]-metrics[mask]
                mask=new
        contribution/=24
        np.testing.assert_allclose(contribution.sum(0),metrics[-1]-metrics[0],atol=1e-8)
        for k,factor in enumerate(FACTORS):
            vals=contribution[k]*100
            rows.append(dict(uid=uid,target=t,identity=scored[uid]['id2d'],factor=factor,miou_pp=vals[0],
                             acc25_pp=vals[1],acc50_pp=vals[2]))
    save_csv(OUT/'oracle_attribution_rows.csv',rows)
    out=[]
    for t in (*PLATFORMS,'all'):
        for identity in ('all','target','other','unknown'):
            for f in FACTORS:
                subset=[r for r in rows if (t=='all' or r['target']==t) and (identity=='all' or r['identity']==identity) and r['factor']==f]
                if subset:out.append(dict(target=t,identity=identity,factor=f,n=len(subset),
                    **{k:np.mean([r[k] for r in subset]) for k in ('miou_pp','acc25_pp','acc50_pp')}))
    save_csv(OUT/'oracle_attribution.csv',out)

if __name__=='__main__':run()
