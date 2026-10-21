"""Synthetic timer conservation and original-callable transparency only."""
import pytest
import torch

from scripts.profile_evidence_policy import ExclusiveTimer,FunctionProfiler,profile_fit,assert_replay_match
from scripts.evidence_policy import policy
from scripts.evidence_policy.test_policy import FakeBank,CFG


class Clock:
    def __init__(self):self.now=0
    def __call__(self):return self.now
    def advance(self,n):self.now+=n


def test_nested_exclusive_wall_time_conservation_exactly_once():
    clock=Clock();timer=ExclusiveTimer(clock)
    with timer.scope('bookkeeping_and_acceptance'):
        clock.advance(11)
        with timer.scope('history_construction'):
            clock.advance(5)
            with timer.scope('feedback_Q'):clock.advance(19)
            clock.advance(7)
        with timer.scope('backward'):clock.advance(23)
        clock.advance(3)
    out=timer.report()
    assert out['wall_nanoseconds']==68==out['exclusive_sum_nanoseconds']
    assert out['exclusive_nanoseconds']['bookkeeping_and_acceptance']==14
    assert out['exclusive_nanoseconds']['history_construction']==12
    assert out['exclusive_nanoseconds']['feedback_Q']==19
    assert out['exclusive_nanoseconds']['backward']==23
    assert out['conservation_residual_nanoseconds']==0


def test_same_category_nesting_is_not_double_charged_and_exception_restores_scope():
    clock=Clock();timer=ExclusiveTimer(clock)
    with pytest.raises(RuntimeError):
        with timer.scope('bookkeeping_and_acceptance'):
            clock.advance(3)
            with timer.scope('bookkeeping_and_acceptance'):
                clock.advance(7);raise RuntimeError('synthetic')
    out=timer.report()
    assert out['wall_nanoseconds']==out['exclusive_nanoseconds']['bookkeeping_and_acceptance']==10


@pytest.mark.parametrize('arm',policy.ARMS)
def test_wrappers_preserve_every_numerical_update_and_restore_original_objects(arm):
    torch.set_num_threads(1);bank=FakeBank(T=3)
    original=policy.fit(bank,'synthetic wrapper',arm,CFG,variant='B')
    originals=(policy.rollout,policy._query,torch.Tensor.backward,torch.optim.Adam.step,torch.optim.Optimizer.zero_grad)
    actual,timing=profile_fit(bank,'synthetic wrapper',arm,CFG)
    proof=assert_replay_match(actual,original)
    assert proof['actual_counts_identical'] and proof['all_round_weights_gradients_losses_acceptance_and_Adam_hashes_identical']
    assert originals==(policy.rollout,policy._query,torch.Tensor.backward,torch.optim.Adam.step,torch.optim.Optimizer.zero_grad)
    assert timing['wall_nanoseconds']==sum(timing['exclusive_nanoseconds'].values())
    assert timing['exclusive_nanoseconds']['feedback_Q']>0
    assert timing['exclusive_scope_calls']['backward']==len(actual['rounds'])
    assert not torch.cuda.is_initialized()


def test_wrapper_restoration_on_failure_and_strict_comparison_detects_changed_parameters():
    originals=(policy.rollout,torch.Tensor.backward)
    timer=ExclusiveTimer()
    with pytest.raises(RuntimeError):
        with FunctionProfiler(policy,timer):
            with timer.scope('bookkeeping_and_acceptance'):raise RuntimeError('synthetic')
    assert originals==(policy.rollout,torch.Tensor.backward)
    result=policy.fit(FakeBank(T=1),'comparison','OPD',CFG,variant='B')
    changed={**result,'finaltheta':result['finaltheta']+1e-12}
    with pytest.raises(AssertionError,match='finaltheta'):assert_replay_match(changed,result)
