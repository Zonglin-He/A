"""Reproduce the blind primary readout without reader marks, then audit frozen scores."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import numpy as np
from common import BASE,HERE,OUT,PRIVATE,digest,inputs,read,write
from predict_span import ARMS,process
from span import TOL,axes,center_interval,offset,read_endpoint
from geometry import CameraSpec

def synthetic():
    # Exact oblique construction. The two points have different optical depths.
    K=np.array([[1000,0,400],[0,1000,300],[0,0,1.]],float)
    spec=CameraSpec('quad',K,np.eye(4),np.zeros(5))
    n=np.array([.6,0,.8]);p=np.array([-.9,0,18.8])
    for sign in (-1,1):
        e=read_endpoint(p,sign*n,0.,[400,300],spec,np.eye(4),sign*n,1.8)
        assert e['status']=='feasible' and abs(e['nominal_s']-1.5*sign)<1.e-12
        assert e['endpoint'][0] <= 1.5*sign <= e['endpoint'][1]
        assert abs(e['nominal_z']-20)<1.e-12 and e['copy_z']==18.8
    assert read_endpoint(np.array([0.,0,19]),np.array([0,0,1.]),0.,[400,300],spec,np.eye(4),n,2.)['status']=='parallel'
    assert np.allclose(center_interval(-.6,.6,1.8),[-.3,.3])
    assert offset(center_interval(-1.2,0.,1.8))<0 and center_interval(-1.,1.,1.8) is None

def replay():
    panel,frozen=inputs()
    hashes=read(PRIVATE/'reply_hashes.json')
    assert digest(PRIVATE/'reply_hashes.json')==read(OUT/'landmark_seal.json')['sha256']
    jobs=[]
    for row in panel:
        uid=row['uid'];reply=None
        if frozen[uid]['selected']:
            path=PRIVATE/'replies'/(uid+'.json')
            assert hashes[uid]==digest(path)
            reply=read(path)
        jobs.append((row,frozen[uid],reply,None))
    with ProcessPoolExecutor(max_workers=5) as pool:
        fresh={r['uid']:v for r,v in zip(panel,pool.map(process,jobs,chunksize=4),strict=True)}
    write(PRIVATE/'primary_replay.json',fresh)
    return fresh

def audit():
    from score import canonical_box,many_iou,truth_for_slot
    from score_region import save
    synthetic()
    panel,frozen=inputs()
    replay()
    fresh=read(PRIVATE/'primary_replay.json')
    saved=read(PRIVATE/'predictions.json')
    assert saved.keys()==fresh.keys() and all(fresh[k]=={a:v for a,v in e.items() if a!='independent'} for k,e in saved.items())
    with (OUT/'rows.csv').open() as f:
        rows={r['uid']:r for r in csv.DictReader(f)}
    with (OUT/'metrics.csv').open() as f:
        metrics=list(csv.DictReader(f))
    max_error=0.;dense_violation=0.;coverage=[]
    for r in panel:
        uid=r['uid'];rec=rows[uid];e=fresh[uid];bb=frozen[uid]['boxes']['H']
        if bb is None:
            assert all(float(rec[a])==0 for a in ARMS)
            continue
        gt=canonical_box(truth_for_slot(r,read(r['calibration'])['ground_info']))
        box=np.array(bb);n,w=axes(box)
        shifts=np.array([e['offsets'][a] for a in ARMS])
        predictions=np.repeat(box[None],len(shifts),axis=0)
        predictions[:,:3]+=shifts[:,None]*n
        zz=many_iou(predictions,gt)
        max_error=max(max_error,max(abs(zz[j]-float(rec[a])) for j,a in enumerate(ARMS)))
        if e.get('real_interval') is None:
            continue
        low,high=e['real_interval']
        trial=np.repeat(box[None],257,axis=0)
        trial[:,:3]+=np.linspace(low,high,len(trial))[:,None]*n
        dense_violation=max(dense_violation,float(many_iou(trial,gt).max())-float(rec['span_oracle']))
        source=np.array(frozen[uid]['support_raw'])
        x=(source-box[:3])@n;center=float((gt[:3]-box[:3])@n)
        cos=abs(axes(gt)[0]@n)
        gt_width=float(gt[4]*cos+gt[3]*np.sqrt(max(0.,1.-cos*cos)))
        enclosed=bool(x.min()>=center-gt_width/2-TOL and x.max()<=center+gt_width/2+TOL)
        true_in=bool(low-TOL<=center<=high+TOL)
        assert true_in==(rec['true_center_in_real_interval']=='True')
        if enclosed and w>=gt_width+TOL:
            assert true_in
        coverage.append(dict(target=r['target'],true_in=true_in,projected_support_contained=enclosed,
                             width_covers=w>=gt_width-TOL))
        if e.get('interval') is not None:
            low,high=e['interval']
            if low<=center<=high: assert abs(e['offsets']['pair']-center)<=abs(center)+TOL
    assert max_error<1.e-8 and dense_violation<1.e-7
    scalars=0
    for m in metrics:
        def belongs(r):
            return (m['panel']=='all' or m['panel']==r['panel'] or
                    (m['panel']=='recording_separated' and r['separation']=='new_recording'))
        subset=[r for r in rows.values() if belongs(r) and m['platform'] in ('macro',r['target'])]
        assert len(subset)==int(m['n'])
        per=[]
        for t in set(r['target'] for r in subset):
            x=np.array([float(r[m['arm']]) for r in subset if r['target']==t])
            per.append([np.mean(x>.25),np.mean(x>.5),np.mean(x)])
        for k,val in zip(('acc25','acc50','miou'),100*np.mean(per,axis=0),strict=True):
            assert abs(val-float(m[k]))<1.e-8
            scalars+=1
    table=[]
    for t in ('all','waymo','drone','quad'):
        g=[r for r in coverage if t in ('all',r['target'])]
        for condition,partial in [('all',g),('all_support_in_true_projected_slab',[r for r in g if r['projected_support_contained']]),
                 ('estimated_width_covers_true_projection',[r for r in g if r['width_covers']]),
                 ('both_sufficient_conditions',[r for r in g if r['width_covers'] and r['projected_support_contained']])]:
            table.append(dict(target=t,condition=condition,n=len(partial),true_center_in_interval=sum(r['true_in'] for r in partial)))
    save(OUT/'width_coverage.csv',table)
    write(OUT/'verification.json',dict(frames=len(fresh),identical_label_free_replays=len(fresh),
        primary_arm_frame_scores_checked=len(rows)*len(ARMS),primary_scorer_gap=max_error,
        independent_metric_scalars=scalars,maximum_dense_oracle_violation=dense_violation,
        analytic_oblique_parallel_and_interval_checks=True,code=digest(__file__),
        primary=digest(OUT/'prediction_seal.json'),score=digest(OUT/'evaluation_receipt.json'),
        diagnostics=digest(OUT/'diagnostics.json'),width_coverage=digest(OUT/'width_coverage.csv'),FINAL_access=False))
    print('Verified',len(fresh),'blind replays and',scalars,'aggregate scores.',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=('replay','audit'))
    {'replay':replay,'audit':audit}[ap.parse_args().action]()
