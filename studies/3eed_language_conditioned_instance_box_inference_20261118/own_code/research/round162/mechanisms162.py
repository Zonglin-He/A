"""Post-seal support, identity, visibility and geometric diagnostics on the fixed panel."""
from collections import defaultdict
from functools import lru_cache
import re
import sys

import numpy as np
from scipy.sparse.csgraph import connected_components

from evidence162 import Evidence
from geometry162 import CameraX, axial, in_box
from manifest162 import HERE, WORK, atomic_json, read, sensor_row, target_rows
from score162 import matrix, put_table, table, threshold_cases, verified
from support162 import SceneSupport
from audit160 import transform_gt
from decode160 import WAYMO_SYNONYMS, M3ED_SYNONYMS
from score154 import many_iou

SUPPORT_METHODS = ('uniform_crop','geometry_component','native_attention','r91_local',
                   'no_rgb_graph','lcib','swap_cross','uniform_cross')
QUOTAS = (8,16,32,64,128)


def read_evidence(pred):
    return Evidence(**{k:pred['ev_'+k].item() if pred['ev_'+k].ndim==0 else pred['ev_'+k]
                       for k in Evidence.__dataclass_fields__})


@lru_cache(maxsize=300)
def annotation_inventory(path, platform, view, index):
    """Build an independent box inventory. Slot zero is the addressed annotation."""
    data = read(path)['ground_info']
    current = data[index]
    targets = [(current['class'],current['bbox_3d'])]
    for annotation in data:
        targets.append((annotation['class'],annotation['bbox_3d']))
        targets.extend((a['class_other'],a['bbox_3d_other']) for a in annotation.get('others',[]))
    names, boxes = [], []
    seen = set()
    for name, box in targets:
        if box is None or len(box)<7:
            continue
        val = tuple(float(v) for v in box[:7])
        if val in seen or not np.isfinite(val).all() or min(val[3:6])<=0:
            continue
        # Other objects are stored as seven yaw-only values on all platforms.
        # The Drone and Quad main annotation adds two unused rotation fields.
        raw = np.asarray(val if platform=='waymo' else (*val,0.,0.))
        model = transform_gt(raw,platform,view).astype(float)
        # The same labelled physical box can recur as an 'other' in several descriptions.
        if boxes and many_iou(np.stack(boxes),model).max()>.96:
            continue
        seen.add(val)
        boxes.append(model)
        names.append(name.casefold())
    return names,np.stack(boxes)


def unique_assignment(mass):
    if not len(mass) or mass.sum() <= 1e-12:
        return -1
    rank = np.argsort(-mass)
    best = rank[0]
    second = mass[rank[1]] if len(rank)>1 else 0.
    if mass[best]<.55*mass.sum() or mass[best]-second<.15*mass.sum():
        return -1
    return int(best)


def seed_owners(points, boxes):
    occupancy = np.stack([in_box(points,b) for b in boxes],axis=1)
    return np.where(occupancy.sum(1)==1,occupancy.argmax(1),-1)


def support_owner(weight, owners, n_objects):
    unknown = np.where(owners < 0,n_objects,owners)
    mass = np.bincount(unknown,weights=weight,minlength=n_objects+1)
    idx = unique_assignment(mass)
    return idx if 0<=idx<n_objects else -1


def box_owner(box, boxes):
    if not np.isfinite(box).all():
        return -1
    v = many_iou(boxes,box)
    order = np.argsort(-v)
    if v[order[0]]<.20 or (len(v)>1 and v[order[0]]-v[order[1]]<.08):
        return -1
    return int(order[0])


def find_expected_ref(ev, names, boxes, phrase, platform, role):
    edges = [(a,b,sign) for a,b,sign in ev.edges if set((a,b))=={0,role}]
    if len(edges)!=1:
        return -1
    a,b,sign=edges[0]
    s,e=ev.role_spans[role]
    words=phrase[s:e].casefold()
    mapping=WAYMO_SYNONYMS if platform=='waymo' else M3ED_SYNONYMS
    hits=[]
    for i,name in enumerate(names):
        if i and any(re.search(r'\b'+re.escape(str(k))+r'\b',words) for k in mapping.get(name,[name])):
            hits.append(i)
    if len(hits)!=1:
        return -1
    cam = CameraX(ev.camera_intrinsic,ev.model_to_camera,ev.distortion,ev.width)
    x = cam.project(boxes[0,:3])[0]
    y = cam.project(boxes[hits[0],:3])[0]
    rel = (y-x) * sign * (1 if a==0 else -1)
    return hits[0] if 0<=x<=1 and 0<=y<=1 and rel>.03 else -1


def r91_evidence(ev, role, old_inside):
    if not len(ev.role_mask):
        return np.zeros(len(ev.seeds))
    if role==0 and ev.subject_weights.shape[-1] == 4:
        return ev.subject_weights[ev.old_winner] @ np.vstack((old_inside,ev.supports[0]))
    return ev.route_weights[role] @ ev.supports[role]


def geometric_region(scene, box, old_inside):
    seeds = scene.seeds
    near = (np.abs(seeds - box[:3]) <= np.maximum(box[3:6] * .85,.25)).all(1)
    ids = np.flatnonzero(near)
    weight = np.zeros(len(seeds))
    if len(ids):
        _, label = connected_components(scene.geometry[ids][:,ids],directed=False)
        quality = np.bincount(label,weights=old_inside[ids]+.01)
        weight[ids[label==quality.argmax()]]=1.
    return weight


def support_samples(ev, p, scene, role, primary_box):
    old_inside=in_box(scene.seeds,primary_box)
    uniform=in_box(scene.points,primary_box).astype(float)
    baseline=r91_evidence(ev,role,old_inside)
    versions = {
        'geometry_component': geometric_region(scene,primary_box,old_inside),
        'native_attention': ev.affinity[role],
        'r91_local': baseline,
    }
    for name in ('no_rgb_graph','lcib','swap_cross','uniform_cross'):
        versions[name] = p[name+'_old_support'] if role==0 else p[name+'_support'][role]
    dense = {'uniform_crop':uniform}
    for name,weight in versions.items():
        dense[name]=(scene.interpolation * weight[scene.neighbor]).sum(1)
    return dense,baseline


def matched_points(support, membership, role, common, scene_entry, out):
    hits=np.asarray(membership,bool)
    maximum = int(hits.sum())
    if maximum==0:
        return
    n_valid = {arm:int(np.count_nonzero(s>1e-7)) for arm,s in support.items()}
    matched = min(n_valid.values())
    for k in (*QUOTAS,0):
        quota = matched if k==0 else k
        for arm,s in support.items():
            eligible = quota>0 and matched>=quota
            score = int(hits[np.argsort(-s,kind='stable')[:quota]].sum()) if eligible else 0
            out.append(dict(**common, role=role, arm=arm, requested_points=k, joint_coverage=eligible,
                            used_points=quota if eligible else 0, available=n_valid[arm], gt_box_points=maximum,
                            box_hits=score, in_box_fraction=score/quota if eligible else '',
                            box_point_recall=score/maximum if eligible else '',
                            scene_cluster=scene_entry))


def main():
    panel=read(WORK/'AUDIT_PANEL_PRIVATE.json')
    assert set(panel)<=set(verified('DEV'))
    records=target_rows()
    dest=HERE/'results/DEV'
    task={int(r['ordinal']):r for r in table(dest/'rows.csv')}
    owners, measures, errors, role_pairs, source_cases, origins = [],[],[],[],[],[]
    for i in panel:
        e=records[i]
        r=e['r']
        common={k:e[k] for k in ('ordinal','source','target')}
        pos=int(r['object_index'])
        names, labeled=annotation_inventory(r['metapath'],e['target'],int(r['suffix'] or 0),pos)
        gt=labeled[0]
        current=task[e['ordinal']]
        with np.load(WORK/'DEV'/f'{i:05d}.npz',allow_pickle=False) as p:
            ev=read_evidence(p)
            scene=SceneSupport(ev)
            old=np.r_[ev.boxes[ev.old_winner],0.]
            known=seed_owners(ev.seeds,labeled)
            anchor=ev.anchors[ev.old_winner]
            pixels=ev.p2i[anchor]
            cross=np.zeros(len(ev.seeds))
            for u,w in zip(pixels,ev.p2i_learned[anchor]):
                if u>=0 and w>0:
                    keys=ev.i2p[u]
                    valid=keys>=0
                    np.add.at(cross,keys[valid],w*ev.i2p_learned[u][valid])
            box_old=box_owner(ev.old_pca,labeled)
            a=int(known[anchor])
            box_new=box_owner(p['boxes'][p['arms'].tolist().index('lcib')],labeled)
            owners.append(dict(**common,role='subject',kind='old_output',expected=0,assigned=box_old))
            owners.append(dict(**common,role='subject',kind='new_output',expected=0,assigned=box_new))
            owners.append(dict(**common,role='subject',kind='query_anchor',expected=0,assigned=a))
            owners.append(dict(**common,role='subject',kind='anchor_RGB_reverse',expected=0,
                               assigned=support_owner(cross,known,len(labeled))))
            local_seeds = in_box(ev.seeds,old)
            local_points = scene.points[in_box(scene.points,old)]
            on_one_face = single_side_proxy(local_points,old)
            source_cases.append(dict(**common, prior_success=float(current['r158_full'])>.25,
                   anchor_outside_predicted_box=not bool(in_box(ev.seeds[anchor:anchor+1],old)[0]),
                   anchor_and_box_both_known=min(a,box_old)>=0,
                   assigned_anchor_differs_from_box=min(a,box_old)>=0 and a!=box_old,
                   role_parser_count=len(ev.role_mask), distinct_relation_count=len(ev.edges), box_inventory=len(labeled),
                   sampled_inside_old_box=len(local_points), seeds_inside_old_box=int(local_seeds.sum()),
                   seed_visibility_inside_old_box=float(ev.visible[local_seeds].mean()) if local_seeds.any() else -1.,
                   single_side_proxy=on_one_face))
            for name in ('bounded_origins','expanded_origins','lcib'):
                old_miss=float(current['oracle_pca'])
                pool=p[name+'_pool']
                values=many_iou(pool,gt)
                record=dict(**common,arm=name,old_PCA256=old_miss,new_pool=float(values.max(initial=0.)),
                            n=len(pool),n_scene=int((p[name+'_origins']<0).sum()),actual=float(current[name]))
                for threshold in (.25,.50):
                    q=str(int(100*threshold))
                    record['adds_coverage_'+q]=old_miss<=threshold and record['new_pool']>threshold
                    record['actual_rescue_'+q]=old_miss<=threshold and record['actual']>threshold
                origins.append(record)
            assignment_r91,assignment_lcib={},{}
            for role in range(len(ev.role_mask)):
                expected = 0 if role==0 else find_expected_ref(ev,names,labeled,r['caption'],e['target'],role)
                query = int(ev.hypotheses[role,ev.route_weights[role].argmax()]) if role else ev.old_winner
                reference = (np.r_[ev.boxes[query],0.] if query>=0 else ev.old_pca)
                dens, baseline = support_samples(ev,p,scene,role,reference)
                values = {
                    'r91_local_initializer':support_owner(baseline,known,len(labeled)),
                    'lcib_support':support_owner(p['lcib_support'][role],known,len(labeled)),
                    'lcib_role_box':box_owner(p['lcib_role_boxes'][role],labeled)
                }
                for name,k in values.items():
                    owners.append(dict(**common,role='subject' if role==0 else 'reference',kind=name,
                                       expected=expected,assigned=k))
                assignment_r91[role]=values['r91_local_initializer']
                assignment_lcib[role]=values['lcib_support']
                if expected>=0:
                    matched_points(dens,in_box(scene.points,labeled[expected]),
                                   'subject' if role==0 else 'reference',common,int(current['recording']),measures)
            for a,b,_ in ev.edges:
                for label,d in [('r91',assignment_r91),('lcib',assignment_lcib)]:
                    x,y = d.get(a,-1),d.get(b,-1)
                    role_pairs.append(dict(**common,kind=label,known=min(x,y)>=0,merged=min(x,y)>=0 and x==y))
            if box_old==0 and box_new==0:
                arms = dict(zip(p['arms'].tolist(),p['boxes']))
                arms.update(r158_full=ev.old_pca)
                for name in ('r158_full','lcib','uniform_fit','fixed_winner_support','serial'):
                    b=arms[name]
                    if box_owner(b,labeled)==0:
                        yaw,b_dims=aligned_error(b,gt)
                        errors.append(dict(**common,arm=name,center_m=np.linalg.norm(b[:3]-gt[:3]),
                                           absolute_log_size=np.abs(np.log(b_dims/gt[3:6])).mean(),axial_yaw_deg=yaw))
    put_table(dest/'anchor_audit.csv',source_cases)
    put_table(dest/'initialization_oracles.csv',origins)
    aggregate_identity(dest,owners,role_pairs)
    put_table(dest/'support_curves.csv',summarize_curves(measures))
    put_table(dest/'support_matched_inputs.csv',measures)
    if errors:
        put_table(dest/'same_annotated_target_geometry.csv',errors)
    summarize_visibility(dest,[dict(task[r['ordinal']],**{k:r[k] for k in (
        'sampled_inside_old_box','seed_visibility_inside_old_box','single_side_proxy')}) for r in source_cases])
    atomic_json(dest/'MECHANISMS.json',dict(panel=len(panel), native_point_identity_annotations_available=False,
                identity_measurement='independent human-labelled box inventory with conservative unknown; no mask or visually adjudicated instance identity',
                relations_with_labelled_reference=sum(r['role']=='reference' and r['kind']=='lcib_support' and r['expected']>=0 for r in owners),
                no_target_labels_used_to_select_predictions=True,
                anchor_box_assignment_known=sum(r['anchor_and_box_both_known'] for r in source_cases),
                anchor_box_assignment_differs=sum(r['assigned_anchor_differs_from_box'] for r in source_cases),
                anchor_outside=sum(r['anchor_outside_predicted_box'] for r in source_cases), FINAL_access=False))


def aligned_error(box, gt):
    direct = abs(axial(box[6]-gt[6]))
    swapped = abs(axial(box[6]+np.pi/2-gt[6]))
    if swapped < direct and abs(np.log(box[4]/gt[3]))+abs(np.log(box[3]/gt[4])) < abs(np.log(box[3]/gt[3]))+abs(np.log(box[4]/gt[4])):
        return float(np.degrees(swapped)), box[[4,3,5]]
    return float(np.degrees(direct)), box[3:6]


def single_side_proxy(points, box):
    """Describe domination by one nearest original box face, without claiming actual occlusion."""
    from geometry162 import local_points
    if len(points)<8 or np.any(box[3:6]<=0):
        return False
    local=local_points(points,box)
    face=(box[3:6]/2-abs(local)).argmin(1)
    label=2*face+(local[np.arange(len(local)),face]>0)
    return bool(np.bincount(label,minlength=6).max() >= .65*len(local))


def aggregate_identity(dest,rows,relations):
    groups=defaultdict(list)
    for row in rows:
        groups[row['role'],row['kind']].append(row)
    out=[]
    for (role,kind),rr in groups.items():
        eligible=[r for r in rr if r['expected']>=0]
        known=[r for r in eligible if r['assigned']>=0]
        out.append(dict(role=role,readout=kind, all_occurrences=len(rr), label_verifiable=len(eligible),
                        assigned=len(known), model_unknown=len(eligible)-len(known),
                        assignment_agrees=sum(r['assigned']==r['expected'] for r in known)))
    put_table(dest/'identity_proxy.csv',out)
    if relations:
        put_table(dest/'role_overlap_proxy.csv',[dict(readout=label,pairs=len(g),known=sum(r['known'] for r in g),
              merged=sum(r['merged'] for r in g)) for label in ('r91','lcib')
              for g in ([r for r in relations if r['kind']==label],)])


def summarize_curves(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[r['role'],r['arm'],r['requested_points']].append(r)
    out=[]
    for (role,arm,k),g in sorted(groups.items()):
        valid=[r for r in g if r['joint_coverage']]
        out.append(dict(role=role,arm=arm,quota=k,panel_roles=len(g),matched_roles=len(valid),
                        mean_available=np.mean([r['available'] for r in g]),
                        mean_used=np.mean([r['used_points'] for r in valid]) if valid else '',
                        gt_box_fraction=np.mean([r['in_box_fraction'] for r in valid]) if valid else '',
                        gt_box_point_recall=np.mean([r['box_point_recall'] for r in valid]) if valid else ''))
    return out


def summarize_visibility(dest, panel):
    summary=[]
    for name,rows in [('all',panel),('low_rgb',[r for r in panel if float(r['visibility'])<.4]),
                      ('high_rgb',[r for r in panel if float(r['visibility'])>=.4]),
                      ('old_region_low_rgb',[r for r in panel if 0 <= r['seed_visibility_inside_old_box']<.5]),
                      ('old_region_unknown_rgb',[r for r in panel if r['seed_visibility_inside_old_box']<0]),
                      ('old_region_sparse',[r for r in panel if r['sampled_inside_old_box']<16]),
                      ('old_region_one_face_proxy',[r for r in panel if r['single_side_proxy']]),
                      ('matched_wrong_binding',[r for r in panel if int(r['swapped_seeds'])>0]),
                      ('camera_relation',[r for r in panel if int(r['known_edges'])>0])]:
        if not rows:
            continue
        for arm in ('lcib','missing_as_negative','fixed_dimensions','visible_surface_only','no_rgb_graph','swap_cross'):
            comparable=[dict(r,weight=float(r['weight'])) for r in rows]
            pp,gain,loss,_=threshold_cases(comparable,'r158_full',arm)
            summary.append(dict(stratum=name,n=len(rows),arm=arm,dIoU=pp[0],d25=pp[1],d50=pp[2],
                                acc25_rescues=int(gain[1]),acc25_harms=int(loss[1]),
                                acc50_rescues=int(gain[2]),acc50_harms=int(loss[2]),
                                main_fallbacks=sum(r['fallback']=='True' for r in rows)))
    put_table(dest/'visibility_strata.csv',summary)


if __name__=='__main__':
    main()
