"""Score sealed predictions and characterize the physical center domains offline."""
import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from feasible_geometry import CameraRays, bounds, inside, rotation
from oabc import ARMS, BASE, HERE, OUT, POSES, PREVIOUS, PRIVATE, SPEC, digest, read, write
from evaluate_surface import groups as panel_groups, stats
from score import canonical_box, many_iou, score_boxes, truth_for_slot
from score_confirmation import cluster_interval, paired_stats, ray_residual
from score_region import save
from surface_grounding import read_panel


AUDIT_ARMS = ('anchor_pair','plain_pair','feasible_pair','D_no_yaw_prior','D')
COMPARISONS = (('B','anchor_pca'),('C','anchor_qwen'),('B','feasible_pca'),('C','feasible_qwen'),
    ('plain_pair','anchor_pair'),('feasible_pair','plain_pair'),('feasible_pair','anchor_pair'),
    ('D_no_yaw_prior','feasible_pair'),('D','D_no_yaw_prior'),('D','A'),('D','anchor_pair'),
    ('D','plain_pair'),('D','feasible_pair'),('D_L7','D'),('L7','A'))


def groups(rows):
    yield ('all','macro'), rows
    yield from panel_groups(rows)


def inside_truth(support, center, yaw, dims, cfg):
    rot = rotation(yaw)
    l,h = bounds(support@rot,dims,cfg['trim_quantile'],cfg['tolerance_m'])
    return bool(inside(center@rot,l,h))


def domain_oracle(est, pose, gt, cfg):
    if not pose['feasible']:
        return None,None,False
    theta = pose['theta']
    axes = rotation(theta)
    dims = np.asarray(est['dims'])
    l,h = np.asarray(pose['domain'])
    gt_local = gt[:3]@axes
    anchor = np.clip(gt_local,l,h)
    candidates = np.vstack((np.asarray(pose.get('witness_centers', [])).reshape(-1,3),anchor))
    boxes = np.column_stack((candidates@axes.T,np.tile(dims,(len(candidates),1)),np.full(len(candidates),theta)))
    scores = many_iou(boxes,gt)
    best = float(scores.max())
    if not pose['solved']:
        return best,None,False
    cap = pose['ray_caps_m']
    rays = CameraRays(np.asarray(est['fit_rays']),np.asarray(est['transform']),theta,dims,cap)
    permitted = est['near_minimum']+cfg['tolerance_m']+1.e-10
    truth_near = bool(inside(gt_local,l,h) and rays.losses(gt_local)[0] <= permitted)
    valid = np.r_[np.asarray(pose['witness_losses'])<=permitted,rays.losses(anchor)[0]<=permitted]
    best_near = float(scores[valid].max()) if valid.any() else None
    return best,best_near,truth_near


def domain_records(est,gt,identity,cfg):
    support = np.asarray(est['support'])
    dims = np.asarray(est['dims'])
    tol = cfg['tolerance_m']
    located = (np.abs((support-gt[:3])@rotation(gt[6])) <= gt[3:6]/2.+tol).all(1)
    rows = []
    for name,pose in est['poses'].items():
        th = pose['theta']
        rot = rotation(th)
        gt_loc = gt[:3]@rot
        l,h = np.asarray(pose['domain'])
        strict_l,strict_h = np.asarray(pose['strict'])
        oracle,near,truth_near = domain_oracle(est,pose,gt,cfg)
        public = dict(**identity,pose=name,point_count=len(support), fit_raw_count=est['fit_raw_count'],
            strict_feasible=bool((strict_l<=strict_h).all()), robust_feasible=pose['feasible'], solved=pose['solved'],
            strict_truth=bool(inside(gt_loc,strict_l,strict_h)), robust_truth=bool(inside(gt_loc,l,h)), near_truth=truth_near,
            gt_size=inside_truth(support,gt[:3],th,gt[3:6],cfg),
            gt_yaw=inside_truth(support,gt[:3],gt[6],dims,cfg),
            gt_both=inside_truth(support,gt[:3],gt[6],gt[3:6],cfg),
            support_truth_fraction=float(located.mean()), domain_oracle=oracle, near_oracle=near,
            size_iou_upper_bound=float(min(np.prod(dims),np.prod(gt[3:6]))/max(np.prod(dims),np.prod(gt[3:6]))),
            domain_distance_to_gt_m=float(np.linalg.norm(gt_loc-np.clip(gt_loc,l,h))) if pose['feasible'] else None)
        # Dict fields encode distances and widths, with no raw center/GT coordinates public.
        for axis in range(3):
            public[f'strict_width_{axis}']=strict_h[axis]-strict_l[axis]
            public[f'width_{axis}']=h[axis]-l[axis]
        if pose['solved']:
            before,after = np.asarray(pose['geometry']),np.asarray(pose['oabc'])
            error = np.abs((after[:3]-gt[:3])@rot)
            public.update(weak_axes=sum(not x for x in pose['strong']), face_rays=sum(pose['face_counts']),
               prior_move_m=float(np.linalg.norm(pose['prior_move'])), near_witnesses=pose['near_count'],
               center_geometry_m=float(np.linalg.norm(before[:3]-gt[:3])), center_oabc_m=float(np.linalg.norm(after[:3]-gt[:3])),
               raw_qwen_center_m=float(np.linalg.norm(np.asarray(est['raw_qwen_center'])-gt[:3])),
               fit_geometry_loss_m=pose['fit_min'],fit_oabc_loss_m=pose['fit_oabc'],
               actual_loose_containment=pose['support_fraction'])
            for i in range(3):
                public[f'center_error_{i}']=error[i]
                public[f'near_width_{i}']=pose['near_width'][i] if pose['near_width'] is not None else None
        rows.append(public)
    return rows


def audit_frame(est,identity):
    rays = np.asarray(est['audit'],float).reshape(-1,3)
    if not len(rays):
        return []
    transform = np.asarray(est['transform'])
    cap = max(.1,.5*min(est['dims'][:2]))
    output = []
    reference = ray_residual(np.asarray(est['boxes']['anchor_pair']),rays,transform)
    for arm in AUDIT_ARMS:
        hit,dist=ray_residual(np.asarray(est['boxes'][arm]),rays,transform)
        common = hit & reference[0]
        cost = np.where(hit,np.minimum(dist,cap),cap)
        output.append(dict(**identity,arm=arm,n=len(rays),hits=int(hit.sum()),both=int(common.sum()),
            sum_cost_m=float(cost.sum()),common_sum_m=float(dist[common].sum()),reference_common_sum_m=float(reference[1][common].sum())))
    return output


def evaluate():
    seal = read(OUT/'prediction_seal.json')
    spec=read(SPEC)
    raw=PRIVATE/'predictions.json'
    assert digest(SPEC)==seal['protocol'] and digest(raw)==seal['predictions']
    assert digest(PREVIOUS/'results/prediction_seal.json')==seal['previous_prediction_seal']
    assert digest(BASE/'private/panel.json')==seal['panel'] and not seal['target_labels_read'] and not seal['FINAL_access']
    assert seal['ray_isolation']['retained_withheld_raw']==0 and seal['frames']==416
    assert all(digest(HERE/f)==h for f,h in seal['implementation'].items())
    panel=read_panel()
    predictions=read(raw)
    with (PREVIOUS/'results/rows.csv').open(newline='') as f:
        previous={r['uid']:r for r in csv.DictReader(f)}
    assert set(predictions)==set(previous)=={r['uid'] for r in panel}
    rows,poses,audit = [],[],[]
    iou_gap, previous_gap=0.,0.
    for frame in panel:
        uid=frame['uid']
        est=predictions[uid]
        identity={k:frame[k] for k in ('uid','target','panel','recording','separation')}
        gt=canonical_box(truth_for_slot(frame,read(frame['calibration'])['ground_info']))
        valid = {arm:canonical_box(est['boxes'][arm]) for arm in ARMS if est['boxes'][arm] is not None}
        if valid:
            scores,gap=score_boxes(np.stack(tuple(valid.values())),gt)
            iou_gap=max(iou_gap,gap)
        else:
            scores=[]
        scores=dict.fromkeys(ARMS,0.) | dict(zip(valid,map(float,scores)))
        for a,b in (('A','P'),('L7','L7'),('anchor_pca','C_pca'),('anchor_qwen','C_qwen'),('anchor_pair','C_ray')):
            previous_gap=max(previous_gap,abs(scores[a]-float(previous[uid][b])))
        info = dict(**identity,**scores,fit_node=est['fit_node'],status=est['status'])
        for name in ('A','anchor_pair','plain_pair','feasible_pair','B','C','D_no_yaw_prior','D','D_L7'):
            info[name+'_center_m']=float(np.linalg.norm(valid[name][:3]-gt[:3])) if name in valid else None
        if est['fit_node']:
            local=domain_records(est,gt,identity,spec)
            poses.extend(local)
            audit.extend(audit_frame(est,identity))
            axes = [r for r in local if r['solved']]
            wins = [r for r in axes if r['near_witnesses']]
            yaw_delta = est['poses']['pca']['theta']-est['poses']['qwen']['theta']
            info.update(D_pose=est['choices']['D']['pose'], geometric_pose=est['choices']['D_no_yaw_prior']['pose'],
                single_node=est['single_node'], selected_rays=est['selection_count'], solved_poses=len(axes),
                fit_near_poses=len(wins), robust_oracle=max((r['domain_oracle'] for r in local if r['domain_oracle'] is not None),default=None),
                near_oracle=max((r['near_oracle'] for r in axes if r['near_oracle'] is not None),default=None),
                yaw_separation_deg=float(np.rad2deg(abs((yaw_delta+np.pi/2.)%np.pi-np.pi/2.))),
                size_iou_upper_bound=local[0]['size_iou_upper_bound'], support_truth_fraction=local[0]['support_truth_fraction'])
            for field in ('strict_feasible','robust_feasible','strict_truth','robust_truth','near_truth','gt_size','gt_yaw','gt_both'):
                info[field]=any(r[field] for r in local)
        rows.append(info)
    assert iou_gap<1.e-8 and previous_gap<1.e-8
    return rows,poses,audit,dict(max_iou_check_gap=iou_gap, max_five_replayed_arms_gap=previous_gap,
                              previous_matches=416*5, prediction_seal=digest(OUT/'prediction_seal.json'))


def report_coverage(rows):
    entries=[]
    for (panel,target),group in groups(rows):
        matched=[r for r in group if r['fit_node']]
        chosen=[r for r in matched if r['D_pose'] is not None]
        wrong=[r for r in matched if r['D']<=.50]
        solved=[r for r in matched if r['solved_poses']]
        near=[r for r in solved if r['fit_near_poses']==2]
        finite=[r for r in matched if r['robust_oracle'] is not None]
        entries.append(dict(panel=panel,target=target,n=len(group),fit_node=len(matched), solved=len(solved),
            chose=len(chosen),choose_qwen=sum(r['D_pose']=='qwen' for r in chosen),
            two_near_witness_poses=len(near),two_near_and_15deg=sum(r['yaw_separation_deg']>=15. for r in near),
            missed_inside_domain=sum(r['robust_truth'] for r in wrong),bad_D_at_fit_node=len(wrong),
            missed_inside_near_fit=sum(r['near_truth'] for r in wrong),
            domain_can_rescue=sum(r['robust_oracle'] is not None and r['robust_oracle']>.50 for r in wrong),
            near_can_rescue=sum(r['near_oracle'] is not None and r['near_oracle']>.50 for r in wrong),
            robust_empty=len(matched)-len(finite),size_precludes_acc50=sum(r['size_iou_upper_bound']<=.5 for r in matched),
            rescued_cover_by_gt_size=sum(not r['robust_truth'] and r['gt_size'] for r in matched),
            rescued_cover_by_gt_yaw=sum(not r['robust_truth'] and r['gt_yaw'] for r in matched),
            **{f:sum(r[f] for r in matched) for f in ('strict_feasible','robust_feasible','strict_truth',
                'robust_truth','near_truth','gt_size','gt_yaw','gt_both')}))
    return entries


def summaries(rows,poses,audit):
    metrics,comparisons,width_report,physics = [],[],[],[]
    intervals={}
    for (panel,target),group in groups(rows):
        sal=dict(panel=panel,target=target)
        for arm in (*ARMS,'robust_oracle','near_oracle'):
            sub=[r for r in group if r.get(arm) is not None]
            if not sub:
                continue
            ctr=[r[arm+'_center_m'] for r in sub if r.get(arm+'_center_m') is not None]
            metrics.append(dict(**sal,arm=arm,n=len(sub),**stats(sub,arm),center_n=len(ctr),
                                center_median_m=float(np.median(ctr)) if ctr else None))
        for a,b in COMPARISONS:
            comparisons.append(dict(**sal,arm=a,reference=b,n=len(group),**paired_stats(group,a,b)))
            if a=='D':
                intervals[f'{panel}/{target}/{a}-{b}']=cluster_interval(group,a,b)
    for (panel,target),group in groups(poses):
        for name in POSES:
            use=[r for r in group if r['pose']==name and r['solved']]
            for ax in range(3):
                if not use:
                    continue
                w,x=np.array([(r[f'width_{ax}'],r[f'center_error_{ax}']) for r in use]).T
                near=[r[f'near_width_{ax}'] for r in use if r[f'near_width_{ax}'] is not None]
                rho=spearmanr(w,x).statistic if len(use)>1 and np.ptp(w)>0 and np.ptp(x)>0 else None
                width_report.append(dict(panel=panel,target=target,pose=name,axis=ax,n=len(use),
                    width_median_m=float(np.median(w)), error_median_m=float(np.median(x)),
                    rho_width_vs_error=None if rho is None else float(rho),
                    median_sampled_near_width_m=float(np.median(near)) if near else None))
    for (panel,target),group in groups(audit):
        for arm in AUDIT_ARMS:
            sub=[r for r in group if r['arm']==arm]
            n=sum(r['n'] for r in sub)
            both=sum(r['both'] for r in sub)
            if n:
                physics.append(dict(panel=panel,target=target,arm=arm,frames=len(sub),n=n,
                    hits=sum(r['hits'] for r in sub),clipped_mae_m=sum(r['sum_cost_m'] for r in sub)/n,both=both,
                    common_hit_mae_m=sum(r['common_sum_m'] for r in sub)/both if both else None,
                    reference_common_mae_m=sum(r['reference_common_sum_m'] for r in sub)/both if both else None))
    return metrics,comparisons,width_report,physics,intervals


def run():
    rows,poses,audit,quality=evaluate()
    task,paired,width,ray,intervals=summaries(rows,poses,audit)
    for name,data in dict(rows=rows,poses=poses,metrics=task,paired=paired,width_error=width,
        ray_rows=audit,ray_audit=ray,coverage=report_coverage(rows)).items():
        save(OUT/f'{name}.csv',data)
    write(OUT/'evaluation_receipt.json',dict(score_script=digest(__file__),frames=len(rows),pose_rows=len(poses),
          audit_rows=len(audit),quality=quality,recording_intervals=intervals,gt_offline_only=True,FINAL_access=False))
    print('Scored',len(rows),'frames',len(poses),'pose-domain records.',quality,flush=True)


if __name__=='__main__':
    run()
