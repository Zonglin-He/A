"""Synthetic CPU tests; no model, media, real query, GT, or dataset scoring."""
import copy

import pytest
import torch

from scripts.trajectory_opd.evidence import TrajectoryEvidence, extract_visual_grids


def fixture(ambiguous=False, padding=False, positions=None, interval=None):
    height, width, frames, channels = 4, 8, 10, 256
    physical = [0, 4, 9, 15, 22, 30, 41, 55, 72, 91]
    anchors = [0, 1, 2, 3, 4, 5, 8, 9] if positions is None else positions
    features = []; masks = []; boxes = []; targets = []
    for node in range(frames):
        shift = node % 3
        feature = torch.zeros(height, width, channels)
        for row in range(height):
            for col in range(width):
                if ambiguous:
                    feature[row, col, 0] = 1
                else:
                    feature[row, col, row * width + ((col - shift) % width)] = 1
        valid = torch.ones(height, width, dtype=torch.bool)
        if padding:
            valid[:, -1] = False; feature[:, -1] = float('nan')
        features.append(feature.flatten(0, 1)); masks.append(~valid.flatten())
        target = torch.zeros(height, width)
        target[1:3, 1 + shift:3 + shift] = 1
        targets.append(target)
        boxes.append([(2 + shift) / width, 2 / height, 2 / width, 2 / height])
    features, masks = torch.stack(features), torch.stack(masks)
    pos = torch.zeros(height, width, channels)
    for row in range(height):
        pos[row, :, :128] = (row + 1) / (height + 1)
    for col in range(width):
        pos[:, col, 128:] = (col + 1) / (width + 1)
    pos = pos.flatten(0, 1)
    inputs = []
    for view in (0, 1):
        local = features[view::2]; length = len(local)
        inputs.append(dict(encoded_feature=torch.cat((local.transpose(0, 1), torch.ones(2, length, channels)), 0),
                           encoded_pos=torch.cat((pos[:, None].expand(-1, length, -1), torch.zeros(2, length, channels)), 0),
                           encoded_mask=torch.cat((masks[view::2], torch.zeros(length, 2, dtype=torch.bool)), 1),
                           query_tgt=torch.zeros(length, 1, channels)))
    teacher_boxes = torch.tensor([boxes[x] for x in anchors], dtype=torch.float32)
    teacher = dict(selected_query=1, boxes=teacher_boxes, input_shape=[8, 3, height, width])
    logits = torch.full((1, 8, 2, height, width), -30.)
    for j, node in enumerate(anchors):
        logits[0, j, 1] = torch.where(targets[node] > 0, 30., -30.)
    cache = dict(spatial_inputs=inputs, frame_ids=physical, planned_positions=anchors, teacher=teacher,
                 interval=[0, 92] if interval is None else interval, GT_read=False)
    pack = dict(raw_out=dict(pred_masks=logits), teacher=teacher, selected_query=1,
                positions8=anchors, frame_ids=physical, input_shape=[8, 3, height, width],
                nested_tensor_shape=[1, 8, 3, height, width], nested_mask_shape=[1, 8, height, width],
                nested_mask_all_false=True, GT_read=False)
    return cache, pack, torch.tensor(boxes, dtype=torch.float64), torch.stack(targets).double()


def test_translation_and_independent_past_future_transport():
    cache, pack, boxes, expected = fixture()
    evidence = TrajectoryEvidence(pack, cache)
    assert evidence.targets.shape == (10, 4, 8)
    assert torch.max(torch.abs(evidence.targets[6:8] - expected[6:8])) < .002
    assert bool((evidence.confidence[6:8] > .97).all())
    assert evidence.metadata['native_mask_coarsened']
    assert evidence.metadata['propagation'].startswith('independent')
    assert torch.equal(evidence.targets[cache['planned_positions']], evidence.seeds)
    assert not torch.cuda.is_initialized()


def test_identity_link_prefers_consistent_trajectory_to_switch():
    cache, pack, boxes, _ = fixture()
    evidence = TrajectoryEvidence(pack, cache)
    switched = boxes.clone(); switched[7, 0] += 3 / 8
    consistent = evidence.utility(boxes, boxes)
    wrong = evidence.utility(switched, boxes)
    assert consistent['link'] > wrong['link'] + .05
    assert 0 <= consistent['link'] <= 1


def test_projection_partial_full_and_image_range():
    cache, pack, boxes, _ = fixture()
    evidence = TrajectoryEvidence(pack, cache)
    full = evidence.projection_scores(boxes)
    partial = boxes.clone(); partial[:, 2] /= 2
    entire = boxes.new_tensor([.5, .5, 1., 1.])[None].expand_as(boxes)
    p = evidence.projection_scores(partial); image = evidence.projection_scores(entire)
    # Original observation masks retain exact almost-binary rectangle geometry.
    assert abs(float(full[0]) - 1) < 1e-10
    assert abs(float(p[0]) - (1 + 2 / 3) / 2) < 1e-10
    assert abs(float(image[0]) - (.4 + 2 / 3) / 2) < 1e-10
    assert bool(((full >= 0) & (full <= 1 + 1e-12)).all())


def test_ambiguous_features_do_not_amplify_low_confidence():
    cache, pack, boxes, _ = fixture(ambiguous=True)
    evidence = TrajectoryEvidence(pack, cache)
    unobserved = ~evidence.observed
    assert float(evidence.confidence[unobserved].max()) < .2
    assert torch.allclose(evidence.W[unobserved].sum(1), evidence.confidence[unobserved])
    output = evidence.utility(boxes, boxes)
    assert output['unobs'] <= evidence.confidence[unobserved].mean() + 1e-12
    assert evidence.metadata['confidence_normalized_across_nodes'] is False


def test_direct_gradient_is_finite_and_matches_finite_difference_away_from_knots():
    cache, pack, E, _ = fixture()
    evidence = TrajectoryEvidence(pack, cache)
    boxes = E.clone(); boxes[:, 0] += .037; boxes[:, 2] *= 1.11
    boxes.requires_grad_(); utility = evidence.utility(boxes, E)['U']
    gradient = torch.autograd.grad(utility, boxes)[0]
    assert bool(torch.isfinite(gradient).all()) and abs(float(gradient[6, 0])) > 1e-6
    step = 1e-6; plus = boxes.detach().clone(); minus = plus.clone()
    plus[6, 0] += step; minus[6, 0] -= step
    numeric = (evidence.utility(plus, E)['U'] - evidence.utility(minus, E)['U']) / (2 * step)
    assert abs(float(numeric - gradient[6, 0])) < 1e-6


def test_physical_offset_and_graph_mapping_and_prior():
    cache, pack, _, _ = fixture()
    evidence = TrajectoryEvidence(pack, cache)
    features, _, ids, shape, meta = extract_visual_grids(cache)
    assert shape == (4, 8) and meta['offset_frames'] == [5, 5]
    for node in range(10):
        item = cache['spatial_inputs'][node % 2]
        assert torch.equal(features[node], item['encoded_feature'][:32, node // 2])
    weight = evidence.W[6] / evidence.W[6].sum()
    assert abs(float(weight[5]) - (72 - 41) / (72 - 30)) < 1e-12
    assert abs(float(weight[6]) - (41 - 30) / (72 - 30)) < 1e-12
    # Observed rows stay onehot even outside Strong. Unobserved outside Strong is zero.
    cache, pack, _, _ = fixture(positions=[0, 1, 2, 3, 4, 5, 6, 9], interval=[9, 72])
    e = TrajectoryEvidence(pack, cache)
    assert e.W[9, 7] == 1 and e.W[8].sum() == 0 and e.W[7].sum() > 0


def test_padding_excluded_from_features_targets_and_both_affinity_directions():
    cache, pack, boxes, _ = fixture(padding=True)
    evidence = TrajectoryEvidence(pack, cache)
    invalid = ~evidence.valid
    assert bool(torch.isfinite(evidence.features).all())
    assert torch.count_nonzero(evidence.targets.flatten(1)[invalid]) == 0
    for node in range(1, 10):
        f, r = evidence.adjacent_forward[node - 1], evidence.adjacent_reverse[node - 1]
        assert torch.count_nonzero(f[invalid[node]]) == torch.count_nonzero(f[:, invalid[node - 1]]) == 0
        assert torch.count_nonzero(r[invalid[node - 1]]) == torch.count_nonzero(r[:, invalid[node]]) == 0
        assert torch.allclose(f[evidence.valid[node]].sum(1), torch.ones(int(evidence.valid[node].sum()), dtype=torch.float64))
    assert bool(torch.isfinite(evidence.utility(boxes, boxes)['U']))


def test_empty_prior_groups_are_zero_without_confidence_normalization():
    cache, pack, boxes, _ = fixture(interval=[100, 110])
    evidence = TrajectoryEvidence(pack, cache)
    result = evidence.utility(boxes, boxes)
    assert result['obs'] == result['unobs'] == result['link'] == result['U'] == 0


def test_rejects_wrong_native_identity_or_noncontiguous_visual_layout():
    cache, pack, _, _ = fixture()
    wrong = copy.deepcopy(pack); wrong['teacher']['selected_query'] = 0
    with pytest.raises(ValueError, match='selected object query'):
        TrajectoryEvidence(wrong, cache)
    bad = copy.deepcopy(cache); bad['spatial_inputs'][0]['encoded_pos'][0] = 0
    with pytest.raises(ValueError, match='nonzero prefix'):
        TrajectoryEvidence(pack, bad)
