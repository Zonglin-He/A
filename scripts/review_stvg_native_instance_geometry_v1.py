"""Independent saved-mask geometry and reliability readback; no labels/model."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_native_instance_common_v1 import BASE, read, sha, write, guard

def main():
    import numpy as np
    from pycocotools import mask as coco
    sys.addaudithook(guard)
    refs = read(BASE/'REFERENCE_LOCK.json')
    direct = {x['case_id']: x for x in read(BASE/'DIRECT_TEACHER_READOUT.json')['records']}
    roster = read(BASE/'ROSTER_LOCK.json')['records']
    assert refs['status'] == 'locked' and refs['GT_read'] is False
    checks = 0
    outputs = []
    for rec, ref in zip(roster, refs['records']):
        cid = rec['case_id']; assert ref['case_id'] == cid
        req = read(ROOT/rec['request_path']); assert sha(ROOT/rec['request_path']) == rec['request_sha256']
        inv = read(BASE/'TRACKING'/cid/'MASK_INVENTORY.json')
        receipt = read(BASE/'teacher'/(cid+'.receipt.json'))
        teacher = read(BASE/'teacher'/(cid+'.json'))
        assert sha(BASE/'teacher'/(cid+'.json')) == receipt['teacher_record_sha256']
        valid = direct[cid]['valid_uncapped_selection']
        if not valid:
            assert not ref['anchors'] and ref['no_update'] and not direct[cid]['dense_frames']
            outputs.append(dict(case_id=cid,valid=False,frames_checked=0,anchors_checked=0)); continue
        candidate = ref['candidate_id']; assert candidate == teacher['parsed']['candidate_id']
        saved_frames = direct[cid]['dense_frames']
        assert len(saved_frames) == req['local_frame_count'] == len(inv['files'])
        windows = {j for c in rec['centers'] for j in c['local_frame_indices']}
        retained = {}; geometry = {}; present = 0
        for item, frame in zip(inv['files'], saved_frames):
            j = item['index']; p = ROOT/item['path']
            assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256']
            obj = read(p)['objects'].get(str(candidate)); mask = None; xy = cx = box = None; area = 0
            if obj is not None:
                mask = np.asarray(coco.decode(obj['rle']), dtype=bool).squeeze()
                assert mask.ndim == 2
                yy, xx = np.nonzero(mask); area = len(xx)
                if area:
                    h,w = mask.shape; box = [int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())]
                    xy = [box[0]/w,box[1]/h,(box[2]+1)/w,(box[3]+1)/h]
                    cx = [(xy[0]+xy[2])/2,(xy[1]+xy[3])/2,xy[2]-xy[0],xy[3]-xy[1]]
                    present += 1
                    if j in windows: retained[j] = mask
            assert frame['local_frame_index'] == j and frame['physical_frame_id'] == req['frame_map'][j]['physical_frame_id']
            assert frame['candidate_id'] == candidate and frame['present'] == bool(area)
            assert frame['mask_area_px'] == area and frame['xyxy_inclusive_pixels'] == box
            assert frame['xyxy_normalized_continuous'] == xy and frame['cxcywh_normalized_continuous'] == cx
            geometry[j] = (xy,cx); checks += 8
        anchors = {a['slot']:a for a in ref['anchors']}
        assert len(anchors) == len(ref['anchors'])
        physical = {x['physical_frame_id']:x['local_frame_index'] for x in req['frame_map']}
        for center in rec['centers']:
            j = physical[center['center_physical_frame_id']]; window = center['local_frame_indices']
            assert window == list(range(max(0,j-2),min(req['local_frame_count'],j+3)))
            if j not in retained:
                assert center['slot'] not in anchors; continue
            a = anchors[center['slot']]
            assert a['position'] == center['source_grid_position'] and a['physical_frame_id'] == center['center_physical_frame_id']
            assert a['box_xyxy_normalized'] == geometry[j][0] and a['box_cxcywh'] == geometry[j][1]
            overlap=[]
            for left,right in zip(window,window[1:]):
                if left in retained and right in retained:
                    intersection=np.count_nonzero(retained[left]&retained[right]); union=np.count_nonzero(retained[left]|retained[right])
                    overlap.append(intersection/union)
            fraction=sum(k in retained for k in window)/len(window)
            mean=sum(overlap)/len(overlap) if overlap else 0.
            assert abs(a['weight']-fraction*mean) <= 1e-15
            assert a['reliability'] == dict(window_frame_count=len(window),present_frame_count=sum(k in retained for k in window),presence_fraction=fraction,valid_adjacent_pair_count=len(overlap),mean_adjacent_mask_iou=mean)
            checks += 7
        assert ref['no_update'] == (not bool(anchors))
        outputs.append(dict(case_id=cid,valid=True,frames_checked=len(saved_frames),present_frames=present,anchors_checked=len(anchors)))
    write(BASE/'ROOT_INDEPENDENT_GEOMETRY_READBACK.json',dict(status='pass',scope='all selected saved RLE geometry and fixed local reliability; no semantic identity claim',case_count=8,checks=checks,records=outputs,GT_read=False,CPU_only=True,model_replayed=False,reference_lock_sha256=sha(BASE/'REFERENCE_LOCK.json'),direct_readout_sha256=sha(BASE/'DIRECT_TEACHER_READOUT.json'),reader_sha256=sha(Path(__file__))))
    print(dict(status='pass',case_count=8,checks=checks))

if __name__=='__main__': main()
