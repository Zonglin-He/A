"""Synthetic CPU checks; no real query, annotation, RGB or model reads."""
import inspect

import pytest
import torch
import torch.nn.functional as F

from scripts.score_audit.cross_feedback import (
    CrossAffinity,box_occupancy,held_feedback,predict_held_support,source_only_affinity)
from scripts.propagation_v0.bank import transport


def identity_case():
    feature = torch.eye(16,dtype=torch.float32)
    valid = torch.ones(16,dtype=torch.bool)
    affinity = source_only_affinity(feature,feature,valid,valid,source_node=0,held_node=4)
    return affinity


def test_source_affinity_matches_existing_FP32_transport_operation_order():
    generator = torch.Generator().manual_seed(12)
    features = torch.randn(2,6,32,dtype=torch.float32,generator=generator)
    valid = torch.tensor([[1,1,0,1,1,1],[1,0,1,1,1,1]],dtype=torch.bool)
    features[~valid] = 0
    affinity = source_only_affinity(features[0],features[1],valid[0],valid[1],
                                    source_node=2,held_node=5)
    normalized = F.normalize(features,dim=-1,eps=1e-12)
    columns = []
    for source_cell in range(6):
        occupancy = torch.zeros(6,dtype=torch.float64)
        occupancy[source_cell] = 1
        columns.append(transport(normalized,valid,occupancy,1,0))
    expected_matrix = torch.stack(columns,dim=1)
    assert torch.equal(affinity.matrix,expected_matrix)
    boxes = torch.tensor([[.5,.5,.3,.5],[.2,.2,.4,.4],[.5,.5,1.,1.]],dtype=torch.float64)
    prediction = predict_held_support(boxes,affinity,source_shape=(2,3),held_shape=(2,3))
    for box,result in zip(boxes,prediction):
        occupancy = box_occupancy(box[None],(2,3)).flatten()*valid[0]
        expected = transport(normalized,valid,occupancy,1,0)
        torch.testing.assert_close(result.flatten(),expected,atol=2e-16,rtol=2e-15)


def test_top20_row_normalization_and_padding_exclusion():
    generator = torch.Generator().manual_seed(7)
    source = torch.randn(25,32,generator=generator,dtype=torch.float32)
    held = torch.randn(9,32,generator=generator,dtype=torch.float32)
    sv = torch.ones(25,dtype=torch.bool);sv[2] = False
    hv = torch.ones(9,dtype=torch.bool);hv[3] = False
    affinity = source_only_affinity(source,held,sv,hv,source_node=3,held_node=7)
    assert affinity.matrix.dtype==torch.float64
    assert torch.equal((affinity.matrix>0).sum(-1)[hv],torch.full((8,),20))
    assert not bool(affinity.matrix[:,~sv].any()) and not bool(affinity.matrix[~hv].any())
    torch.testing.assert_close(affinity.matrix.sum(-1)[hv],torch.ones(8,dtype=torch.float64),
                               atol=3e-16,rtol=0)


def test_discriminative_source_grid_penalizes_shrinking_full_image_and_empty_support():
    affinity = identity_case()
    boxes = torch.tensor([[.125,.125,.25,.25],[.125,.125,.125,.125],
                          [.5,.5,1.,1.],[-1.,-1.,.1,.1]],dtype=torch.float64)
    target = torch.zeros(4,4,dtype=torch.float64);target[0,0] = 1
    prediction = predict_held_support(boxes,affinity,source_shape=(4,4),held_shape=(4,4))
    scores = held_feedback(prediction,target,affinity)
    assert scores.shape==(4,) and scores.dtype==torch.float64
    assert scores[0]>scores[1]>scores[2]>scores[3]==0
    assert scores[0]>.99
    empty_target = torch.zeros_like(target)
    assert torch.equal(held_feedback(prediction,empty_target,affinity),torch.zeros(4,dtype=torch.float64))


def test_uniform_feature_counterexample_can_still_prefer_full_image_to_correct_small_box():
    # A necessary limitation: with aliased Source descriptors, every held cell
    # receives the same average occupancy. 2D Dice then does not ensure correct
    # extent and, here, rewards the full image more than the correct one-cell box.
    features = torch.ones(16,8,dtype=torch.float32)
    valid = torch.ones(16,dtype=torch.bool)
    affinity = source_only_affinity(features,features,valid,valid,source_node=0,held_node=1)
    boxes = torch.tensor([[.125,.125,.25,.25],[.5,.5,1.,1.]],dtype=torch.float64)
    target = torch.zeros(4,4,dtype=torch.float64);target[0,0] = 1
    prediction = predict_held_support(boxes,affinity,source_shape=(4,4),held_shape=(4,4))
    scores = held_feedback(prediction,target,affinity)
    assert scores[1]>scores[0]
    torch.testing.assert_close(scores,torch.tensor([1/16,2/17],dtype=torch.float64),
                               atol=2e-16,rtol=0)


def test_changing_held_target_cannot_change_cached_affinity_or_candidate_prediction():
    assert 'held_target' not in inspect.signature(source_only_affinity).parameters
    assert 'held_target' not in inspect.signature(predict_held_support).parameters
    affinity = identity_case()
    boxes = torch.tensor([[.125,.125,.25,.25],[.875,.875,.25,.25],[.5,.5,1.,1.]],dtype=torch.float64)
    original_boxes,original_matrix = boxes.clone(),affinity.matrix.clone()
    prediction = predict_held_support(boxes,affinity,source_shape=(4,4),held_shape=(4,4))
    original_prediction = prediction.clone()
    target_a = torch.zeros(4,4,dtype=torch.float64);target_a[0,0] = 1
    target_b = torch.zeros_like(target_a);target_b[-1,-1] = 1
    score_a = held_feedback(prediction,target_a,affinity)
    score_b = held_feedback(prediction,target_b,affinity)
    assert not torch.equal(score_a,score_b)
    assert torch.equal(boxes,original_boxes)
    assert torch.equal(affinity.matrix,original_matrix)
    assert torch.equal(prediction,original_prediction)
    repeated = predict_held_support(boxes,affinity,source_shape=(4,4),held_shape=(4,4))
    assert torch.equal(repeated,prediction)


def test_self_scoring_is_rejected_before_affinity_or_feedback():
    feature = torch.eye(4,dtype=torch.float32);valid = torch.ones(4,dtype=torch.bool)
    with pytest.raises(ValueError):
        source_only_affinity(feature,feature,valid,valid,source_node=2,held_node=2)
    forged = CrossAffinity(torch.eye(4,dtype=torch.float64),valid,valid,3,3)
    with pytest.raises(ValueError):
        predict_held_support(torch.tensor([[.5,.5,.2,.2]]),forged,
                             source_shape=(2,2),held_shape=(2,2))
    with pytest.raises(ValueError):
        held_feedback(torch.zeros(1,2,2),torch.zeros(2,2),forged)


def test_empty_valid_support_and_invalid_held_cells_never_create_a_match():
    feature = torch.eye(4,dtype=torch.float32)
    none,all_cells = torch.zeros(4,dtype=torch.bool),torch.ones(4,dtype=torch.bool)
    affinity = source_only_affinity(feature,feature,none,all_cells,source_node=0,held_node=1)
    boxes = torch.tensor([[.5,.5,1.,1.],[.1,.1,0.,0.],[2.,2.,.2,.2]],dtype=torch.float64)
    prediction = predict_held_support(boxes,affinity,source_shape=(2,2),held_shape=(2,2))
    assert torch.equal(prediction,torch.zeros_like(prediction))
    assert torch.equal(held_feedback(prediction,torch.ones(2,2),affinity),torch.zeros(3,dtype=torch.float64))
    affinity = source_only_affinity(feature,feature,all_cells,none,source_node=0,held_node=1)
    prediction = predict_held_support(boxes,affinity,source_shape=(2,2),held_shape=(2,2))
    assert torch.equal(held_feedback(prediction,torch.ones(2,2),affinity),torch.zeros(3,dtype=torch.float64))


def test_batch_three_candidates_supports_different_source_and_held_grid_sizes():
    generator = torch.Generator().manual_seed(22)
    source = torch.randn(6,8,generator=generator,dtype=torch.float32)
    held = torch.randn(4,8,generator=generator,dtype=torch.float32)
    affinity = source_only_affinity(source,held,torch.ones(6,dtype=torch.bool),
        torch.ones(4,dtype=torch.bool),source_node=5,held_node=6)
    boxes = torch.tensor([[.5,.5,1.,1.],[.2,.3,.2,.2],[.8,.7,.2,.2]],dtype=torch.float64)
    predicted = predict_held_support(boxes,affinity,source_shape=(2,3),held_shape=(2,2))
    assert predicted.shape==(3,2,2)
    assert held_feedback(predicted,torch.ones(2,2),affinity).shape==(3,)
    with pytest.raises(ValueError):
        predict_held_support(boxes,affinity,source_shape=(2,2),held_shape=(2,2))
