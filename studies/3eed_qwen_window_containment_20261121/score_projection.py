"""Post-seal audit of whether the true upright box satisfies the proposed image constraint."""
import csv
import numpy as np
from scipy.spatial.transform import Rotation
from projection import (BASE, HERE, PUBLIC, PRIVATE, PROTOCOL, TOL, SIDES, camera,
                        EDGE_PAIRS, digest, edges, given_rect, inputs, projection, read, write)
from geometry import axial, extent_corners, rigid_xyz, sensor_pixels
from score import canonical_box, iou2, oriented_iou, rectangle, rotation, score_boxes, truth_for_slot


def save_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def edge_fields(row, tag, window, p):
    row[tag+'_status'] = p['status']
    if p['status'] != 'ok' or window is None:
        return
    m = edges(window, p['rect'])
    dense = edges(window, p['dense'])
    row.update({tag+'_'+k:v for k,v in m.items() if k != 'signed'})
    sides = np.asarray(m['signed'])
    row.update({tag+'_exceeds_'+side:bool(x>TOL) for side,x in zip(SIDES,sides)})
    row[tag+'_energy_px2'] = float(np.square(np.maximum(sides-TOL,0.)).sum())
    row[tag+'_oversize_2d'] = bool(max(sides[0]+sides[2],sides[1]+sides[3]) > 2*TOL)
    row[tag+'_pass'] = m['worst_px'] <= TOL
    row[tag+'_pass_zero'] = m['worst_px'] <= 1e-9
    row[tag+'_dense_pass'] = dense['worst_px'] <= TOL
    row[tag+'_dense_edge_gap_px'] = float(np.max(np.abs(np.asarray(p['rect']) - p['dense'])))


def native_projection(raw, spec, size, mat):
    yaw, pitch, roll = (list(raw[6:]) + [0.,0.])[:3]
    rot = Rotation.from_euler('ZYX', [yaw, pitch, roll]).as_matrix()
    corners = rigid_xyz(extent_corners(raw[:3], raw[3:6], rot), mat)
    depth = corners[:,2]
    if depth.min() <= .1:
        return rot, dict(status='behind_camera' if depth.max() <= .1 else 'crosses_near_plane')
    r = rectangle(sensor_pixels(corners,spec),size)
    if r is None:
        return rot, dict(status='off_image')
    t = np.linspace(0,1,33)[:,None]
    samples = np.concatenate([corners[i]+t*(corners[j]-corners[i]) for i,j in EDGE_PAIRS])
    dense = rectangle(sensor_pixels(samples,spec),size)
    return rot, dict(status='ok',rect=r.tolist(),dense=dense.tolist())


def run():
    seal = read(PUBLIC/'observation_seal.json')
    assert not seal['target_labels_read'] and seal['protocol'] == digest(PROTOCOL)
    assert seal['code'] == digest(HERE/'projection.py') and seal['observations'] == digest(PRIVATE/'observations.json')
    assert seal['parent_seal'] == digest(BASE/'results/prediction_seal.json')
    panel, frozen = inputs()
    pred = read(PRIVATE/'observations.json')
    with (BASE/'results/rows.csv').open() as h:
        previous = {r['uid']:float(r['H']) for r in csv.DictReader(h)}
    assert set(pred) == set(frozen) == set(previous)
    recording_keys = sorted({(r['target'],r['recording']) for r in panel})
    record_ids = {key:'R%02d' % i for i, key in enumerate(recording_keys, 1)}
    rows = []
    max_iou_gap = max_corner_gap = max_frozen_gap = 0.
    for i, f in enumerate(sorted(panel, key=lambda r:r['uid']), 1):
        uid = f['uid']
        p = pred[uid]
        win = p['window']
        spec, size, mat = camera(f)
        metadata = read(f['calibration'])['ground_info']
        box = canonical_box(truth_for_slot(f, metadata))
        item = metadata[f['slot']]
        gt = projection(box, f, spec, size, mat)
        raw = np.asarray(item['bbox_3d'],float)
        rot_native, native = native_projection(raw,spec,size,mat)
        image_label = given_rect(item.get('bbox_2d_proj'), size)
        decision = p['projection']
        rf = {k:f[k] for k in ('panel','target','separation')}
        rf.update(case='F%03d' % i, recording=record_ids[(f['target'], f['recording'])],
                  selected=p['selected'], valid_window=win is not None, dist_assumed_zero=p['distortion_assumed_zero'],
                  image_width=int(size[0]), image_height=int(size[1]), iou=0., native_iou=0.,
                  nonzero_extra_box_fields=bool(np.any(np.abs(item['bbox_3d'][7:]) > 1e-8)))
        edge_fields(rf, 'frozen', win, decision)
        edge_fields(rf, 'truth', win, gt)
        edge_fields(rf, 'native', win, native)
        if win is not None:
            rf['window_border'] = bool(min(win[0],win[1],size[0]-win[2],size[1]-win[3]) <= TOL)
            rf['window_min_side_px'] = float(min(win[2]-win[0],win[3]-win[1]))
        if image_label is not None and gt['status']=='ok':
            rf['annot_vs_truth_iou2'] = iou2(image_label, np.asarray(gt['rect']))
        if image_label is not None and native['status']=='ok':
            rf['annot_vs_native_iou2'] = iou2(image_label, np.asarray(native['rect']))
        if win is not None and image_label is not None:
            rf['annot_encloses_window'] = edges(win, image_label)['worst_px'] <= TOL
            rf['annot_vs_window_iou2'] = iou2(image_label,np.asarray(win))
        if 'truth_pass' in rf and 'frozen_pass' in rf:
            if not rf['frozen_pass']:
                flags = [side for side in SIDES if rf['frozen_exceeds_'+side]]
                rf['flagged_sides'] = len(flags)
                rf['flagged_sides_covered_by_gt'] = sum(not rf['truth_exceeds_'+side] for side in flags)
        if frozen[uid]['boxes']['H'] is not None:
            b = canonical_box(frozen[uid]['boxes']['H'])
            ious, err = score_boxes(b[None],box)
            rf['native_iou'] = oriented_iou(raw[:3], raw[3:6], rot_native, b[:3], b[3:6], rotation(b[6]))
            rf.update(iou=float(ious[0]), center_m=float(np.linalg.norm(b[:3]-box[:3])),
                      dim_relative=float(np.mean(np.abs(b[3:6]/box[3:6]-1))),
                      yaw_error_deg=float(np.rad2deg(abs(axial(b[6]-box[6])))))
            max_iou_gap = max(max_iou_gap, err)
        max_corner_gap = max(max_corner_gap, gt.get('crosscheck_px',0.), decision.get('crosscheck_px',0.))
        max_frozen_gap = max(max_frozen_gap, abs(rf['iou'] - previous[uid]))
        rows.append(rf)
    assert len(rows) == 416 and sum(r['selected'] for r in rows) == 320
    assert max_frozen_gap < 1e-8
    save_csv(PUBLIC/'rows.csv', rows)
    write(PUBLIC/'scoring_seal.json',dict(frames=len(rows),recordings=len(recording_keys),
          observation_seal=digest(PUBLIC/'observation_seal.json'),outcome_rows=digest(PUBLIC/'rows.csv'),
          score_code=digest(__file__), historical_iou_max_gap=max_frozen_gap,
          independent_oriented_iou_max_gap=max_iou_gap, independent_projection_max_gap_px=max_corner_gap,
          label_stage_only=True,FINAL_access=False))
    return rows


if __name__=='__main__':
    run()
