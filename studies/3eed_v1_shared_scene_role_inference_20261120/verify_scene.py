"""Independent scalar enumeration, full paired metric recalc and geometric certificates."""
from collections import Counter,defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import gzip

import numpy as np

from shared_paths import HERE, OUT, OLD, NEW, LEX, PRED, SPEC, save, read, digest, inputs
from scene_roles import Role,probability
from scene_faces import bottom,split_returns,only_ground
from terrain_followup import BASE,DST,ARMS as GEO,geometry_predictions,boxes_in_model
from scope183 import ordinary
from score_roles import score_boxes, sensor_truth, sensor_frame_box, csv_file
from score_lex import lexical_has_pass
from batch_yaw_iou import many_iou
from predict_scene import ARMS
from score_swap import best
from tables_scene import grouped


def enumeration():
    rng=np.random.default_rng(7891)
    views=np.array([[.08,.1,.12,14.,1.],[.47,.5,.54,7.,1.],[.75,.8,.83,21.,1.],[.75,.8,.83,27.,2.]])
    identity=np.eye(4,dtype=bool); draws=np.arange(128)
    p=rng.dirichlet(np.ones(4));q=rng.dirichlet(np.ones(4))
    a=rng.integers(0,3,(128,4));b=rng.integers(0,3,(128,4))
    t=Role(np.arange(4),np.eye(4),p,views,a==0,b==0)
    r=Role(np.arange(4),np.eye(4),q,views,a==1,b==1)
    error=0.;checks=0
    for mode in ('joint_raw','separate_worlds','wrong_reference','image_center'):
        for side in [-1,1]:
            for depth in [-1,0,1]:
                for immediate in [True,False]:
                    expected=np.zeros((4,4))
                    for win in draws:
                        subset=np.flatnonzero(t.observed[win])
                        ref=(r.independent if mode=='separate_worlds' else r.observed)[win]
                        if mode=='wrong_reference':ref=ref[np.roll(np.arange(4),2)]
                        rr=np.flatnonzero(ref)
                        if len(rr) and depth:rr=rr[depth*views[rr,3]==(depth*views[rr,3]).min()]
                        for j in rr:
                            u=views[j,:3] if mode!='image_center' else [.5,.5,.5]
                            got=[i for i in subset if i!=j and ((views[i,2]<u[0]) if side==-1 else (views[i,0]>u[2]))]
                            if not got:continue
                            if immediate:
                                dist=np.array([abs(views[i,1]-u[1]) for i in got])
                                got=np.asarray(got)[dist==dist.min()]
                            for i in got:expected[i,j]+=q[j]/sum(q[rr])*p[i]/sum(p[got])/128
                    x=probability(t,r,identity,side,depth,immediate,mode,draws)
                    error=max(error,float(abs(x-expected).max()));checks+=1
                    assert np.allclose(x,expected,atol=5.e-15,rtol=0) and np.diag(x).sum()==0
    return checks,error


def terrain_cases():
    with gzip.open(OUT/'terrain_task_rows.csv.gz','rt') as f:rows=list(csv.DictReader(f))
    summary=read(OUT/'terrain_audit.json');assert len(rows)==18620*4
    wins=grouped(rows,['source','target'])
    published={(r['direction'],r['arm']):r for r in csv.DictReader((OUT/'terrain_metrics.csv').open())}
    roster=ordinary();new_cases=[];top_error=0.;n=0
    for d,boxes,q in geometry_predictions():
        src,target=d.split('_to_');rr=wins[src,target]
        lookup=grouped(rr,['arm'])
        for i,a in enumerate(GEO):
            col=lookup[(a,)]
            assert len(col)==len(boxes) and np.array_equal(boxes[:,0,[0,1,3,4,6]],boxes[:,i,[0,1,3,4,6]])
            v=np.array([float(r['iou']) for r in col])
            for tag,t in [('acc25',.25),('acc50',.5)]:
                assert abs(100*np.mean(v>t)-float(published[d,a][tag]))<1.e-9
            dv=abs((boxes[:,i,2]+boxes[:,i,5]/2)-(boxes[:,0,2]+boxes[:,0,5]/2))
            top_error=max(top_error,float(dv.max()))
        assert top_error<1.e-10
        for i,row in enumerate(lookup[('local_two_halves',)]):
            if row['beyond_a256']!='1':continue
            info=roster[target][i];gt=sensor_truth(target,info['metapath'],info['caption'],info['suffix'])
            part=i//128*128
            with np.load(BASE/d/f'{part:05d}.npz') as s:
                assert s['ordinals'][i-part]==i
                original=s['pool'][i-part]
            native=boxes_in_model(original,target,info['suffix'])
            native[:,6]=0.
            zero=np.array([sensor_frame_box(b,target,info['suffix']) for b in native])
            scores,err=score_boxes(np.concatenate((original,zero,boxes[i])),gt)
            assert scores[:512].max()<=.5 and scores[513]>.5>=scores[512]
            assert all(row[k]=='1' for k in ('iou_before','iou_after','return_before','return_after'))
            new_cases.append(dict(case=row['case'],source=src,target=target,v1_iou=scores[512],
                local_iou=scores[513],global_iou=scores[514],permuted_iou=scores[515],
                best_original_PCA=scores[:256].max(),best_original_zero=scores[256:512].max(),
                both_ownership_checks=1,recording=row['recording'],frame=row['frame']))
        n+=len(boxes)
    if new_cases:csv_file(OUT/'terrain_certified.csv',new_cases)
    return dict(verified_frames=n,verified_direction_arms=len(published)-len(GEO),
        beyond_A256_verified=len(new_cases),distinct_frames=len({r['frame'] for r in new_cases}),
        max_fixed_top_error=top_error,checked_boxes_for_each_certificate=512)


def verify_core():
    scalar,err=enumeration()
    import verify_bound
    with TemporaryDirectory() as d:
        verify_bound.OUT=Path(d);verify_bound.run()
        assert (Path(d)/'bound_recheck.csv').read_bytes()==(LEX/'results/bound_recheck.csv').read_bytes()
    audited=read(OUT/'audit.json');assert not audited['FINAL_access'] and not audited['target_updates']
    # Read the two newly generated positives and their same-observation shape controls by original case.
    with gzip.open(OUT/'task_rows.csv.gz','rt') as f:
        outcomes=[r for r in csv.DictReader(f) if r['arm']=='joint_shape' and r['beyond_all']=='1']
    positives={(r['case'],r['source']):r for r in outcomes}
    verified=[]
    labels={**read(OLD/'labels.json'),**read(NEW/'labels.json')}
    for row,rpath,_,lpath,_,_,_,_,_ in inputs():
        key=row['uid'],row['source']
        if key not in positives:continue
        entry=labels[row['uid']];truth=np.array(entry['truth'])[entry['subject']]
        with np.load(rpath) as z,np.load(lpath) as h,np.load(PRED/(key[0]+'_'+key[1]+'.npz')) as pr:
            assert max(best(z[k],truth) for k in ('a','t_box','r_box'))<=.5
            assert not lexical_has_pass(h['native_boxes'],h['sample'],truth)
            b=pr['boxes'];y=score_boxes(b,truth)[0]
            record=positives[key]
            v=dict(case=key[0],source=key[1],group=record['group'],raw=float(record['raw_iou']),
                joint=y[ARMS.index('joint_shape')],ground=y[ARMS.index('ground_only')],
                marginal=y[ARMS.index('marginal_shape')],v1_ground=y[ARMS.index('v1_ground')],
                tagged_by_iou=int(record['iou_raw']==record['iou_final']=='T'),
                tagged_by_seeds=int(record['seed_raw']==record['seed_final']=='T'),reference_iou=record['iou_r'])
            assert v['joint']>.5
            verified.append(v)
    assert len(verified)==len(positives)
    if verified:csv_file(OUT/'panel_certified.csv',verified)
    return dict(paired_scene_controls_enumerated=scalar,scalar_max_error=err,margin_records_rechecked=18620,
        panel_outside_all=len(verified),both_ownership_tests=sum(r['tagged_by_iou'] and r['tagged_by_seeds'] for r in verified))


if __name__=='__main__':
    save(OUT/'verification.json',dict(**verify_core(),**terrain_cases(),source_training=0,
            target_updates=0,FINAL_access=False))
