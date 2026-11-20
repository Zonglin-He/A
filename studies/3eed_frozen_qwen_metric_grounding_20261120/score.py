"""Offline scoring. Task ground truth is opened only after both prediction paths seal."""
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import sys
from pathlib import Path
import numpy as np
from paths import ROOT, HERE, PRIV, OUT, SPEC, digest, read, write, PLATFORMS
from geometry import (ARMS, camera, raw_returns, cell_points, extent_corners, cook_rotation, to_sensor,
                      axial, rigid_xyz, sensor_pixels, upright, pixel_rays, front_offset)
sys.path.insert(0,str(ROOT/'studies/3eed_v1_component_ablation_20261119'))
from score_ablation import (rotation, sensor_truth, score_boxes, normalize, oriented_iou,
                            many_iou, save_csv)

def truth_for_slot(row, items):
    target=row['target']; slot=items[row['slot']]
    assert normalize(slot['caption'])==normalize(row['text'])
    raw=np.asarray(slot['bbox_3d'],float)
    assert raw.shape==((7,) if target=='waymo' else (9,)) and np.isfinite(raw).all() and min(raw[3:6])>0
    box=raw[:7].copy()
    from geometry import WAYMO_ANGLES,z_rotation
    angle=WAYMO_ANGLES[row['suffix']] if target=='waymo' else 0.
    if target=='waymo':
        box[:3]=z_rotation(angle)@box[:3];box[6]+=np.deg2rad(angle)
    if target=='drone':box[2]+=1.8
    box=box.astype(np.float32).astype(float)
    if target=='waymo':
        box[:3]=box[:3]@z_rotation(angle);box[6]-=np.deg2rad(angle)
    if target=='drone':box[2]-=1.8
    matches=[i for i in items if normalize(i['caption'])==normalize(row['text'])]
    if all(i['bbox_3d']==slot['bbox_3d'] for i in matches):
        np.testing.assert_allclose(box,sensor_truth(target,row['calibration'],row['text'],row['suffix']),rtol=0,atol=1e-12)
    return box

def canonical_box(b):
    out=np.array(b,float)
    if out[4]>out[3]:
        out[[3,4]]=out[[4,3]]
        out[6]+=np.pi/2
    out[6]=axial(out[6])
    return out

def rectangle(xy,size):
    if xy is None:return None
    shape=np.tile(size,2)
    r=np.clip(np.r_[xy.min(0),xy.max(0)],0,shape)
    return r if np.all(r[:2]<r[2:]) else None

def given_rect(raw,size):
    if raw is None:return None
    r=np.asarray(raw,float)
    if r.shape!=(4,) or not np.isfinite(r).all():return None
    r=np.clip(r,0,np.tile(size,2))
    return r if (r[:2]<r[2:]).all() else None

def projected(box,spec,mat,size):
    pts=extent_corners(box[:3],box[3:6],rotation(box[6]))
    return rectangle(sensor_pixels(rigid_xyz(pts,mat),spec),size)

def iou2(r,s):
    if r is None or s is None:return 0.
    a=np.maximum(0,np.minimum(r[2:],s[2:])-np.maximum(r[:2],s[:2]))
    inter=np.prod(a)
    return float(inter/(np.prod(r[2:]-r[:2])+np.prod(s[2:]-s[:2])-inter))

def owners(rect,windows):
    if rect is None or windows[0] is None:return 'unknown',0.,0.
    scores=np.array([iou2(rect,w) for w in windows])
    k=np.argsort(-scores,kind='stable')
    competitor=scores[k[1]] if len(k)>1 else 0.
    if scores[k[0]]>=.50 and scores[k[0]]-competitor>.10:
        owner='target' if k[0]==0 else 'other'
    else:owner='unknown'
    return owner,float(scores[0]),float(max(scores[1:],default=0.))

def inventory(item,row,spec,mat,size,tar):
    # Prefer a dataset-supplied image rectangle when available. A duplicate
    # physical annotation appearing under another caption is not a distractor.
    gtbox=canonical_box(tar['bbox_3d'][:7])
    group=[gtbox];windows=[given_rect(tar.get('bbox_2d_proj'),size)]
    all_boxes=[(obj['bbox_3d'],obj.get('bbox_2d_proj')) for obj in item]
    all_boxes.extend((other.get('bbox_3d_other'),None) for obj in item for other in obj.get('others',[]))
    for raw,pixel in all_boxes:
        if raw is None or len(raw)<7:continue
        b=canonical_box(raw[:7])
        if not np.isfinite(b).all() or min(b[3:6])<=0:continue
        if np.max(many_iou(np.stack(group),b))>.96:continue
        group.append(b)
        rect=given_rect(pixel,size)
        windows.append(projected(b,spec,mat,size) if rect is None else rect)
    return windows

def aggregate(rows):
    ious=np.asarray(rows,float)
    return dict(acc25=100*float(np.mean(ious>.25)),acc50=100*float(np.mean(ious>.50)),miou=100*float(np.mean(ious)))

def delta(rows,new,ref):
    a=np.array([r[new] for r in rows]);b=np.array([r[ref] for r in rows])
    return dict(improved=int(np.sum(a>b+1e-10)),worsened=int(np.sum(b>a+1e-10)),
        rescue25=int(np.sum((a>.25)&(b<=.25))),harm25=int(np.sum((b>.25)&(a<=.25))),
        rescue50=int(np.sum((a>.5)&(b<=.5))),harm50=int(np.sum((b>.5)&(a<=.5))),
        delta_miou_pp=100*float(np.mean(a-b)),delta25_pp=100*float(np.mean((a>.25).astype(int)-(b>.25))),
        delta50_pp=100*float(np.mean((a>.5).astype(int)-(b>.5))))

def recorder_ci(rows,new,ref):
    rng=np.random.default_rng(20261120);buckets=[]
    for t in PLATFORMS:
        by=defaultdict(list)
        for r in rows:
            if r['target']!=t:continue
            a,b=r[new],r[ref]
            by[r['recording']].append([int(a>.25)-int(b>.25),int(a>.5)-int(b>.5),a-b])
        vals=[np.asarray(v,float) for v in by.values()]
        totals=np.array([v.sum(0) for v in vals]);ns=np.array([len(v) for v in vals])
        draws=rng.integers(len(vals),size=(3000,len(vals)))
        buckets.append(totals[draws].sum(1)/ns[draws].sum(1)[:,None])
    lo,hi=np.quantile(100*np.mean(buckets,axis=0),[.025,.975],axis=0)
    return {k:[float(l),float(h)] for k,l,h in zip(('acc25','acc50','miou'),lo,hi)}

def run():
    assert digest(PRIV/'predictions.json')==read(OUT/'geometry_seal.json')['predictions']
    assert digest(PRIV/'official.json')==read(OUT/'official_seal.json')['pred']
    assert digest(PRIV/'panel.json')==read(OUT/'panel_seal.json')['panel']
    roster=read(PRIV/'panel.json');pred=read(PRIV/'predictions.json');official=read(PRIV/'official.json')
    base_dir=ROOT/'studies/3eed_four_scale_direct_subject_20261120/results'
    bp=base_dir/'ordinary_rows.csv.gz'
    assert digest(bp)==read(base_dir/'verification.json')['ordinary_rows_sha256']
    with gzip.open(bp,'rt') as stream:
        registered={(r['source'],r['case']):float(r['separate_direct']) for r in csv.DictReader(stream)}
    all_rows=[];pairs=[];held_errors=[];err_max=0.
    oracle=('gt_z','gt_bearing','gt_center','gt_size','gt_yaw','gt_geometry','image_ray_gt_z','gt_window_returns')
    for r in roster:
        uid=r['uid'];t=r['target'];entry=pred[uid]
        item=read(r['calibration'])['ground_info'];tar=item[r['slot']]
        assert normalize(tar['caption'])==normalize(r['text'])
        gt=canonical_box(truth_for_slot(r,item))
        spec,size,mat=camera(r);gtc=rigid_xyz(gt[:3],mat)
        windows=inventory(item,r,spec,mat,size,tar)
        given=windows[0]
        actual2=None if entry['pixel'] is None else np.array(entry['pixel'])*np.tile(size,2)/1000.
        aux,ai,ad=owners(actual2,windows)
        row=dict(uid=uid,target=t,recording=r['recording'],status2=read(PRIV/'replies'/(uid+'_2d.json'))['status'],
                 status3=read(PRIV/'replies'/(uid+'_3d.json'))['status'],id2d=aux,iou2_target=ai,iou2_other=ad,
                 id3d_projection='unknown',iou_proj_target=0.,iou_proj_other=0.,available_distractors=len(windows)-1,
                 projection_agreement=iou2(given,projected(gt,spec,mat,size)),GT_image_rect=int(given is not None),
                 **entry['support'])
        known={**{a:None for a in ARMS},**{a:None for a in oracle}}
        if entry['camera_boxes']:
            boxes={a:canonical_box(entry['sensor_boxes'][a]) for a in ARMS}
            known.update(boxes)
            b=np.array(entry['camera_boxes']['raw']);q=boxes['raw']
            for a in ARMS:
                qb=boxes[a];qb_cam=np.array(entry['camera_boxes'][a]);jc=rigid_xyz(qb[:3],mat)
                row['depth_error_'+a]=float(jc[2]-gtc[2]);row['center_error_'+a]=float(np.linalg.norm(qb[:3]-gt[:3]))
            pi=rectangle(sensor_pixels(extent_corners(b[:3],b[3:6],cook_rotation(b)),spec),size)
            row['id3d_projection'],row['iou_proj_target'],row['iou_proj_other']=owners(pi,windows)
            row.update(target_optical_depth=float(gtc[2]),abs_yaw_deg=float(np.rad2deg(abs(axial(q[6]-gt[6])))),
                       gt_aspect=float(gt[3]/gt[4]),size_relative_pct=float(100*np.mean(abs(q[3:6]/gt[3:6]-1.))),
                       size_l_abs=float(abs(q[3]-gt[3])),size_w_abs=float(abs(q[4]-gt[4])),size_h_abs=float(abs(q[5]-gt[5])))
            _,r9=to_sensor(b,mat)
            for a in ('raw','ray_depth'):
                row[a+'_full9_iou']=oriented_iou(boxes[a][:3],b[3:6],r9,gt[:3],gt[3:6],rotation(gt[6]))
            for a,center in [('gt_z',b[:3]/b[2]*gtc[2]),('gt_bearing',gtc/gtc[2]*b[2]),
                             ('image_ray_gt_z',np.array(entry['camera_boxes']['image_ray'][:3])/b[2]*gtc[2])]:
                bb=b.copy();bb[:3]=center;known[a]=to_sensor(bb,mat)[0]
            for a,sl in [('gt_center',slice(0,3)),('gt_size',slice(3,6)),('gt_yaw',slice(6,7)),('gt_geometry',slice(3,7))]:
                qb=q.copy();qb[sl]=gt[sl];known[a]=qb
            uv,cam=raw_returns(r,spec,size)
            known['gt_window_returns']=q
            if given is not None:
                fit,_=cell_points(uv,cam,given)
                rr=pixel_rays((given[:2]+given[2:])/2,spec)[0]
                bb=b.copy();bb[:3]=rr*((np.median(fit[:,2])+front_offset(b,rr)) if len(fit)>=5 else b[2])
                known['gt_window_returns']=to_sensor(bb,mat)[0]
            if actual2 is not None:
                fit,_=cell_points(uv,cam,actual2)
                pts=rigid_xyz(fit,np.linalg.inv(mat))
                inside=(abs((pts-gt[:3])@rotation(gt[6]))<=gt[3:6]/2+.05).all(1)
                if len(inside):row['fit_inside_gt_fraction']=float(inside.mean())
            for arm,e in entry['surface'].items():
                for k,val in enumerate(e):
                    held_errors.append(dict(uid=uid,target=t,ray=k,arm=arm,error=val))
        else:row.update(raw_full9_iou=0.,ray_depth_full9_iou=0.)
        ready=[(k,v) for k,v in known.items() if v is not None]
        row.update({k:0. for k in known})
        incoming=[s for s in PLATFORMS if s!=t]
        tests=[v for _,v in ready]+[official[s+'_'+uid] for s in incoming]
        scored,gap=score_boxes(np.stack(tests),gt);err_max=max(gap,err_max)
        for k,i in zip([k for k,v in ready],scored[:len(ready)]):row[k]=float(i)
        anon=hashlib.sha256(('shared-role-scene/1|'+str((t,r['ordinal']))).encode()).hexdigest()[:22]
        for s,i in zip(incoming,scored[len(ready):]):
            pairs.append(dict(uid=uid,source=s,target=t,official=float(i),registered=registered[s,anon],**{a:row[a] for a in ARMS}))
        all_rows.append(row)
    fields=list(dict.fromkeys(k for r in all_rows for k in r))
    save_csv(PRIV/'diagnostic_rows.csv',[{k:r.get(k,'') for k in fields} for r in all_rows])
    public_fields=[k for k in fields if k not in ('target_optical_depth','gt_aspect')]
    save_csv(OUT/'all_rows.csv',[{k:r.get(k,'') for k in public_fields} for r in all_rows])
    save_csv(OUT/'baseline_rows.csv',pairs)
    metrics=[];comparisons=[]
    for t in (*PLATFORMS,'macro'):
        rows=[r for r in all_rows if t=='macro' or r['target']==t]
        pp=[r for r in pairs if t=='macro' or r['target']==t]
        for a in (*ARMS,'raw_full9_iou','ray_depth_full9_iou',*oracle):
            metrics.append(dict(target=t,arm=a,unique_frames=len(rows),**aggregate([r[a] for r in rows])))
        for a in ('official','registered'):
            metrics.append(dict(target=t,arm=a,unique_frames=len(rows),**aggregate([r[a] for r in pp])))
        for a,b in [('image_ray','raw'),('depth_only','raw'),('ray_depth','image_ray'),('ray_depth','mismatched_depth'),('ray_depth','raw')]:
            comparisons.append(dict(target=t,arm=a,reference=b,n=len(rows),**delta(rows,a,b)))
    baseline_deltas=[]
    for label,block in [('all_incoming',pairs)]+[(s+'_to_'+t,[r for r in pairs if r['source']==s and r['target']==t])
              for s in PLATFORMS for t in PLATFORMS if s!=t]:
        for a in ('raw','image_ray','ray_depth'):
            for ref in ('official','registered'):
                baseline_deltas.append(dict(direction=label,n_source_pairs=len(block),unique_frames=len({r['uid'] for r in block}),
                    arm=a,reference=ref,**delta(block,a,ref)))
    save_csv(OUT/'baseline_paired.csv',baseline_deltas)
    save_csv(OUT/'metrics.csv',metrics)
    save_csv(OUT/'paired.csv',comparisons)
    sub=[]
    for group,rr in [('2d_target',[r for r in all_rows if r['id2d']=='target']),
             ('2d_other',[r for r in all_rows if r['id2d']=='other']),('2d_unknown',[r for r in all_rows if r['id2d']=='unknown']),
             ('both_supports',[r for r in all_rows if min(r['fit_bins'],r['wrong_fit_bins'])>=5])]:
        for a,b in [('ray_depth','image_ray'),('ray_depth','mismatched_depth')]:
            if rr:sub.append(dict(group=group,n=len(rr),arm=a,reference=b,**delta(rr,a,b)))
    save_csv(OUT/'subgroups.csv',sub)
    surfaces=[]
    for t in (*PLATFORMS,'all'):
        selected=[p for p in held_errors if t=='all' or p['target']==t]
        for arm in ARMS:
            dat=[p for p in selected if p['arm']==arm]
            both=[(p['uid'],p['ray']) for p in dat if p['error'] is not None]
            v={(p['uid'],p['ray']):p['error'] for p in selected if p['arm']=='image_ray' and p['error'] is not None}
            less=[p for p in dat if p['error'] is not None and (p['uid'],p['ray']) in v]
            med=lambda x:float(np.median(x)) if len(x) else None
            surfaces.append(dict(target=t,arm=arm,heldout_rays=len(dat),hit_rays=len(both),miss_fraction=1-len(both)/len(dat) if dat else None,
                common_hits=len(less),error_median_common=med([p['error'] for p in less]),reference_median_common=med([v[p['uid'],p['ray']] for p in less])))
    save_csv(OUT/'surfaces.csv',surfaces)
    ci={a+'_vs_'+b:recorder_ci(all_rows,a,b) for a,b in [('ray_depth','raw'),('ray_depth','image_ray'),('ray_depth','mismatched_depth')]}
    write(OUT/'audit.json',dict(frames=len(all_rows),baseline_records=len(pairs),max_iou_two_scorer_gap=err_max,
            clustered_95pct_intervals_pp=ci,labels_after_geometry_seal=digest(OUT/'geometry_seal.json'),
            official_after_seal=digest(OUT/'official_seal.json'),original_rows_sha256=digest(bp),
            all_rows=digest(OUT/'all_rows.csv'),source_training=0,target_updates=0,FINAL_access=False))

if __name__=='__main__':run()
