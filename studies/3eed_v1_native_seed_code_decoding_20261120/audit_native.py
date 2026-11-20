"""After prediction seal, compare every GT and hypothesis with the actual upstream training call."""
import sys
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from predict import (HERE, NATIVE, OUT, ROOT, PROTOCOL, WAYMO_VIEWS, atomic, digest,
                     pinned, prior, read)
from seed_code import FIELDS, focal_costs, label_frame_boxes, mask_in_label_frame, native_codes
from evaluate import score_boxes
from roles import inventory

sys.path.insert(0, str(NATIVE))
from src.joint_det_dataset import Joint3DDataset
from models.losses import SigmoidFocalClassificationLoss, compute_points_obj_cls_loss_hard_topk
from utils.transform_waymo import transform_to_front_view

PRIVATE = HERE / 'run_PRIVATE/postseal'
RESULTS = HERE / 'results'


def validated_predictions():
    parent, roster = pinned()
    seal = read(OUT/'SEALED.json')
    assert seal['n'] == len(seal['files']) == 192 and not seal['labels_read']
    lock = prior.checked(OUT/'LOCK.json',seal['lock'])
    assert all(digest(ROOT/p) == h for p,h in lock['files'].items())
    assert all(digest(OUT/p) == h for p,h in seal['files'].items())
    assert seal['native_state_sha256'] == parent['native_state_hashes']
    return roster


def official_input(platform, raw_box, points, view, object_id):
    boxes, valid, labels = Joint3DDataset._get_3eed_target_boxes(
        SimpleNamespace(visualize=False),
        dict(dataset=platform, gt_bbox=raw_box, target_id=object_id), points)
    if platform == 'waymo':
        _, boxes = transform_to_front_view(np.empty((0,3)), boxes, WAYMO_VIEWS[view])
    assert np.array_equal(np.flatnonzero(valid), [0])
    return boxes,valid,labels


def capture_targets(box_banks, valid_masks, memberships, z):
    batch = len(box_banks)
    def tensor(x,dtype):
        return torch.as_tensor(np.asarray(x),dtype=dtype,device='cuda')
    def repeat(x,dtype):
        return tensor(x,dtype).unsqueeze(0).repeat((batch,)+(1,)*np.ndim(x))
    boxes = tensor(box_banks,torch.float32)
    args = dict(center_label=boxes[:,:,:3],size_gts=boxes[:,:,3:6],
                box_label_mask=tensor(valid_masks,torch.float32),seed_inds=repeat(z['input_index'],torch.long),
                point_instance_label=tensor(memberships,torch.long),seed_xyz=repeat(z['xyz'],torch.float32),
                seeds_obj_cls_logits=repeat(z['logits'][None],torch.float32))
    records = {}
    original = SigmoidFocalClassificationLoss.forward
    def instrument(self, input, target, weights):
        assert self.alpha == .25 and self.gamma == 2. and torch.all(weights == 1/1024)
        losses = original(self,input,target,weights)
        records['targets'] = target[:,:,0].cpu().numpy().astype(bool)
        records['row_loss'] = losses.sum(1).cpu().numpy()
        return losses
    with patch.object(SigmoidFocalClassificationLoss,'forward',instrument):
        original_loss = compute_points_obj_cls_loss_hard_topk(args,4)
    np.testing.assert_allclose(records['row_loss'].mean(),original_loss.item(),atol=2e-7)
    return records


@torch.inference_mode()
def audit():
    roster = validated_predictions()
    PRIVATE.mkdir(mode=0o700,exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    counts, maxima, files = Counter(),Counter(),{}
    by_platform = {}
    for platform in PROTOCOL['source']:
        local = Counter()
        inputs = prior.source_inputs(platform)
        archived = prior.sealed_arrays(prior.R181,'SEALED.json','source_scores',platform,['ious'])['ious']
        iou_bank,gt_bank = [],[]
        for ordinal in roster[platform]:
            row = inputs[ordinal]
            raw_gt = read(row['metapath'])['ground_info'][row['object_index']]['bbox_3d']
            with np.load(OUT/platform/f'{ordinal:05}.npz',allow_pickle=False) as z:
                points, idx, boxes = z['pre_view'],z['input_index'],z['boxes']
                view = int(z['view'])
                raw_candidates = label_frame_boxes(boxes,platform,view).copy()
                if platform == 'drone':
                    raw_candidates[:,2] -= 1.8
                if platform != 'waymo':
                    raw_candidates = np.pad(raw_candidates,((0,0),(0,2)))
                examples = [official_input(platform,b,points,view,row['object_index'])
                            for b in [raw_gt,*raw_candidates]]
                box_banks,valid,labels = map(np.asarray, zip(*examples))
                np.testing.assert_allclose(box_banks[1:,0,:6],boxes[:,:6],rtol=0,atol=2e-6)
                np.testing.assert_array_equal(labels[1:,idx]==0,z['inside'])
                native = capture_targets(box_banks,valid,labels,z)
                gt = box_banks[0,0].copy()
                raw_model_mask = mask_in_label_frame(points[idx],label_frame_boxes(gt[None],platform,view))
                np.testing.assert_array_equal(raw_model_mask[0],labels[0,idx]==0)
                code_gt = native_codes(z['xyz'],gt[None],raw_model_mask,'cuda')[0]
                np.testing.assert_array_equal(native['targets'][0],code_gt)
                codes = z['codes']
                np.testing.assert_array_equal(native['targets'][1:],codes)
                encoded = native['targets']
                pos,neg = focal_costs(z['logits'])
                direct = np.where(encoded,pos,neg).sum(1)
                sparse = (encoded*(neg-pos)).sum(1)
                signed_projection = np.minimum(neg,pos).sum() + (np.abs(neg-pos)*(encoded!=(neg>pos))).sum(1)
                errors = dict(original_fp32_mean_loss=float(np.max(np.abs(direct/1024-native['row_loss']))),
                              sparse_expansion=float(np.max(np.abs(direct-(neg.sum()-sparse)))),
                              hamming_projection=float(np.max(np.abs(signed_projection-direct))),
                              sealed_score=float(np.max(np.abs(sparse[1:]-z['fields'][FIELDS.index('native_focal')]))))
                assert errors['original_fp32_mean_loss']<1e-6 and max(v for k,v in errors.items() if k!='original_fp32_mean_loss')<1e-10
                gt[6] = (gt[6]+np.pi/2)%np.pi-np.pi/2
                reference = gt.astype(np.float32)
                np.testing.assert_array_equal(inventory(row)[0][0],reference)
                ious, geometry_err = score_boxes(boxes.astype(float),reference.astype(float))
                archive_err = float(np.max(np.abs(ious-archived[ordinal])))
                assert archive_err<1e-5 and geometry_err<1e-5
                for cutoff in (.25,.50):
                    np.testing.assert_array_equal(ious>cutoff,archived[ordinal]>cutoff)
                errors.update(iou_two_engines=float(geometry_err),iou_vs_archived=archive_err,
                              index_xyz_max_m=float(z['index_distance'].max()))
                for k,v in errors.items():
                    maxima[k] = max(maxima[k],v)
                wrong = z['index_xyz_diff']
                different = (codes!=z['physical_codes']).any(1)
                physical_gt_mask = mask_in_label_frame(points[z['composed']],label_frame_boxes(box_banks[0,:1],platform,view))
                physical_gt = native_codes(z['xyz'],box_banks[0,:1],physical_gt_mask,'cuda')[0]
                local.update(rows=1,gt_code_bits=len(idx),candidate_code_bits=codes.size,candidate_boxes=len(boxes),
                             official_idx_different_seed=int(wrong.sum()),official_idx_different_rows=int(wrong.any()),
                             index_different_same_xyz_seed=int(np.count_nonzero((idx!=z['composed'])&~wrong)),
                             physical_variant_changed_candidate_codes=int(different.sum()),
                             physical_variant_changed_candidate_rows=int(different.any()),
                             physical_variant_changed_gt_codes=int(np.any(code_gt!=physical_gt)))
                gt_bank.append(code_gt)
                iou_bank.append(ious)
        counts.update(local)
        by_platform[platform]=dict(local)
        named = PRIVATE/(platform+'.npz')
        np.savez_compressed(named,ordinal=roster[platform],ious=iou_bank,gt_codes=gt_bank)
        files[named.name]=digest(named)
        print('Official labels and IoU verified',platform,len(iou_bank),flush=True)
    audit_info=dict(platforms=by_platform,totals=dict(counts),max_errors=dict(maxima),
                    all_gt_and_candidate_bits_identical=True,logit_dtype='float32',all_seed_count=1024,
                    private_prediction_manifest=digest(OUT/'SEALED.json'),private_prediction_lock=digest(OUT/'LOCK.json'),
                    training_source_commit='843a901f5367a6be5cc484b97c6ecad462961dfa',
                    official_functions=['Joint3DDataset._get_3eed_target_boxes','compute_points_obj_cls_loss_hard_topk'],
                    original_focal_forward_called=True,prediction_network_calls=192,additional_deployment_network_calls=0,
                    all_model_fingerprints_preserved=True)
    atomic(RESULTS/'native_fidelity.json',audit_info)
    atomic(PRIVATE/'SEALED.json',dict(files=files,predictions=digest(OUT/'SEALED.json'),
                                     native_fidelity=digest(RESULTS/'native_fidelity.json')))


if __name__=='__main__':
    audit()
