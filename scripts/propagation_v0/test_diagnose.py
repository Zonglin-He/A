"""Pure synthetic gate/ranking tests: no assets, real GT or model calls."""
import numpy as np

from scripts.propagation_v0.diagnose import (average_ranks, spearman, rank_state, query_ranking,
    oracle_unobserved, distance_bins, bootstrap_mean, bootstrap_ratio, gate_decision)


def test_average_rank_ties_and_constant_spearman():
    assert np.array_equal(average_ranks([4, 2, 2, 9]), [3, 1.5, 1.5, 4])
    assert abs(spearman([4, 2, 2, 9], [4, 2, 2, 9]) - 1) < 1e-12
    assert abs(spearman([1, 2, 3], [3, 2, 1]) + 1) < 1e-12
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    assert spearman([1], [1]) is None


def test_GT_ties_are_excluded_and_feedback_ties_receive_half_credit():
    result = rank_state([0, 0, 2, 999], [.1, .4, .4, -.5], [True, True, True, False])
    assert result['valid_candidates'] == 3 and result['GT_tied_pairs'] == 1
    assert result['comparable_pairs'] == 2 and result['correct_pairs'] == 1.5
    assert result['pair_accuracy'] == .75 and result['tied_score_pairs'] == 1


def test_fallback_nodes_remain_in_oracle_denominator_and_are_not_rank_fabricated():
    iou = np.array([[.2, .8, .9], [.3, .8, .9], [.4, .9, .8]])
    valid = np.array([[True, True, False], [True, False, False], [True, True, True]])
    result = oracle_unobserved(iou, valid, [False, False, True], [True, True, True])
    assert result['nodes'] == 2 and result['E_only_nodes'] == 1
    assert abs(result['mean_improvement'] - .3) < 1e-12
    assert result['over5pp_fraction'] == .5
    assert np.array_equal(result['selected_slots'], [1, 0, 0])
    states = [rank_state([0, 1, 2], row, mask) for row, mask in zip(iou, valid)]
    summary = query_ranking(states)
    assert summary['states'] == 3 and summary['states_with_pairs'] == 2
    assert states[1]['pair_accuracy'] is None and states[1]['Spearman'] is None


def test_distance_bins_are_in_physical_units_and_include_fixed_boundary():
    ids = [0, 2, 4, 5, 10, 20, 30, 40, 50, 60, 75, 90]
    result = distance_bins(ids, [0, 4, 5, 6, 7, 8, 9, 11])
    assert result['median_anchor_gap'] == 10
    assert result['near'][1] and result['normalized_distance'][1] == .2
    assert result['middle'][2] and result['middle'][3]
    assert result['far'][10] and result['normalized_distance'][10] == 1.5
    assert not any(result[name][0] for name in ['near', 'middle', 'far'])


def test_bootstrap_parent_means_and_micro_ratios_keep_missing_coverage():
    mean = bootstrap_mean([1, None, 3], draws=100, seed=4)
    ratio = bootstrap_ratio([1, 0, 3], [2, 0, 3], draws=100, seed=4)
    assert mean['mean'] == 2 and mean['valid_parents'] == 2 and mean['all_parents'] == 3
    assert ratio['mean'] == .8 and ratio['valid_parents'] == 2 and ratio['all_parents'] == 3
    assert bootstrap_mean([None, None], draws=10)['mean'] is None
    assert bootstrap_ratio([0, 0], [0, 0], draws=10)['mean'] is None


def test_gate_uses_registered_point_estimates_and_both_memory_endpoints():
    good = gate_decision(.01, .51, .02, 0., 1e-5)
    assert good['passed'] and not good['query_selection']
    assert not good['confidence_interval_sign_required']
    assert not gate_decision(0., .6, .2, .01, .01)['passed']
    assert not gate_decision(.01, .5, .2, .01, .01)['passed']
    assert not gate_decision(.01, .6, 0., .01, .01)['passed']
    assert not gate_decision(.01, .6, .2, .01, -.001)['passed']
    assert not gate_decision(.01, .6, .2, 0., 0.)['passed']
    assert gate_decision(.01, .6, .2, -1e-13, .01)['passed']
    assert not gate_decision(.01, None, None, .01, .01)['passed']
