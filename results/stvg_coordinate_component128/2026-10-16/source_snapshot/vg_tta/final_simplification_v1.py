"""Final bounded subtractive ablations. No labels, scoring or dataset choices.

Sealed A4/F44 implementations are imported, never patched. Plain Adam variants
retain actual best and last states. Removing backtracking does not impose a
hidden acceptance gate. Nonfinite failure is recorded and source-restored.
"""
import copy
import math
import time

import torch

from vg_tta.dense_support_tuning_v1 import move, state_hash
from vg_tta.dense_support_temporal_v1 import norm
from vg_tta.decota_tastvg_episode_v1 import fitted_merge, native_view_indices
from vg_tta.simplification_partial_v1 import fit as sealed_space, weighted_spatial
from vg_tta.structured_temporal_calibration_v1 import fit as sealed_time, project, objective


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Meter:
    def __init__(self, inner):
        self.inner, self.forwards, self.seconds = inner, 0, 0.

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def values(self):
        start = time.perf_counter()
        value = self.inner.values()
        sync()
        self.forwards += 1
        self.seconds += time.perf_counter()-start
        return value


def costs(result, meter, elapsed, backtracking, selection):
    path = result['path']
    active = [s for s in path if 'accepted' in s]
    result.update(optimizer_variant=dict(backtracking=backtracking, selection=selection),
        measured_forwards=meter.forwards, measured_suffix_seconds=meter.seconds,
        measured_fit_seconds=elapsed,
        trial_forwards=sum(len(s.get('trials', [])) for s in path),
        accepted_updates=sum(bool(s.get('accepted')) for s in active),
        backtracking_trigger_steps=sum(len(s.get('trials', [])) > 1 for s in active),
        optimizer_steps=len(active), numerical_failure=result.get('failure'))
    return result


def _plain(it, records, ids, *, lr, steps, selection, anchors=None, teacher=None, config=None):
    temporal = teacher is not None
    assert selection in ('best', 'last') and steps >= 0 and lr >= 0
    it.restore(it.initial)
    params = [p for _, p in it.named]
    opt_class = torch.optim.AdamW if temporal else torch.optim.Adam
    eps = 1e-4 if temporal else 1e-8
    opt = opt_class(params, lr=lr, betas=(.9, .999), eps=eps, weight_decay=0.)
    with torch.no_grad():
        zero = it.values()
    evidence = project(zero['logits'], teacher, records, config) if temporal else None
    teacher_hash = state_hash({str(i): t['raw_logits'] for i,t in enumerate(teacher)}) if temporal else None
    skipped = not temporal and not anchors
    path, best, best_step, saved, last, failure, backwards = [], math.inf, 0, it.state(), it.state(), None, 0

    def loss_fn(value):
        if temporal:
            return objective(value['logits'], evidence)
        data = weighted_spatial(value['boxes'], anchors, 4, None)
        return data, dict(data=float(data.detach()), regularizer=0.)

    try:
        for step in range((0 if skipped else steps)+1):
            opt.zero_grad(set_to_none=True)
            value = it.values()
            loss, parts = loss_fn(value)
            lv = float(loss.detach())
            if not math.isfinite(lv):
                failure = 'nonfinite'; break
            if temporal:
                assert torch.equal(value['boxes'], zero['boxes'])
            else:
                assert all(torch.equal(a,b) for a,b in zip(value['logits'], zero['logits']))
            last = it.state()
            tol = 1e-12 if temporal else 0.
            if lv < best-tol:
                best, best_step, saved = lv, step, last
            pair = list(fitted_merge(value['logits'], records, ids))
            row = dict(step=step, loss=lv, parts=parts,
                logits=move(value['logits'], 'cpu'), boxes=move(value['boxes'], 'cpu'),
                indices=pair, physical_interval=[ids[pair[0]],ids[pair[1]]+1],
                state_sha256=state_hash(last), state_delta=norm([p-it.initial[n] for n,p in it.named]))
            if temporal:
                row['raw_offset_indices'] = [list(native_view_indices(z)) for z in value['logits']]
                row['raw_physical_intervals'] = [[r['frame_ids'][s],r['frame_ids'][e]+1]
                    for r,(s,e) in zip(records, row['raw_offset_indices'])]
            path.append(row)
            if skipped or step == steps:
                break
            loss.backward(); backwards += 1
            row['unused_parameters'] = [n for n,p in it.named if p.grad is None]
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
                failure = 'nonfinite_gradient'; break
            if temporal:
                assert not row['unused_parameters']
            row['gradient_norm'] = norm([p.grad for p in params if p.grad is not None])
            before = it.state()
            opt.step()
            # Deliberately unconditional: the next evaluated state may have a
            # worse loss. Best-state selection is a separate optional step.
            row.update(accepted=True, trials=[], optimizer_restored_on_reject=False,
                proposal_norm=norm([p-before[n] for n,p in it.named]))
        assert path, 'Invalid step zero is an implementation error, not a successful no-op'
        last_step = len(path)-1
        selected = best_step if selection == 'best' else last_step
        chosen = saved if selection == 'best' else last
        if failure:
            selected, chosen = 0, it.initial
        it.restore(chosen)
        with torch.no_grad():
            final = it.values()
        assert torch.equal(final['boxes'].cpu(), path[selected]['boxes'])
        assert all(torch.equal(a.cpu(),b) for a,b in zip(final['logits'], path[selected]['logits']))
        result = dict(path=path, state=move(chosen,'cpu'), initial_state=move(it.initial,'cpu'),
            last_state=move(last,'cpu'), last_step=last_step,
            minimum_loss_step=best_step, best_step=selected, final=path[selected],
            state_delta=path[selected]['state_delta'], lr=lr, steps=steps, eps=eps,
            failure=failure, skipped=bool(skipped), backwards=backwards,
            parameter_count=sum(p.numel() for p in params), GT_online=False,
            restore_exact=True, student_output_only=True)
        if temporal:
            assert teacher_hash == state_hash({str(i):t['raw_logits'] for i,t in enumerate(teacher)})
            result.update(evidence=move(evidence,'cpu'), config=evidence['config'],
                parameters=result['parameter_count'], parameter_TTA=True, output_control=False,
                teacher_frozen=True, spatial_invariant=True, source_restored=True)
        else:
            result.update(anchors=copy.deepcopy(anchors), planned=4, gamma=0., kappa=None,
                parameter_names={n:p.numel() for n,p in it.named})
        return result
    finally:
        it.restore(it.initial)
        for _,p in it.named:
            p.grad = None
        assert all(torch.equal(p,it.initial[n]) for n,p in it.named)


def spatial_fit(it, records, ids, anchors, *, lr, steps=10, backtracking=True, selection='best'):
    meter = Meter(it); sync(); tick = time.perf_counter()
    if backtracking:
        assert selection == 'best', 'Only the sequentially authorized variant is allowed'
        result = sealed_space(meter,records,ids,anchors=anchors,planned=4,kappa=None,gamma=0.,lr=lr,steps=steps)
    else:
        result = _plain(meter,records,ids,lr=lr,steps=steps,selection=selection,anchors=anchors)
    sync()
    return costs(result,meter,time.perf_counter()-tick,backtracking,selection)


def temporal_fit(it, records, ids, teacher, config, *, lr, steps=5, backtracking=True, selection='best'):
    meter = Meter(it); sync(); tick = time.perf_counter()
    if backtracking:
        assert selection == 'best'
        result = sealed_time(meter,records,ids,teacher,config,lr=lr,steps=steps)
    else:
        result = _plain(meter,records,ids,lr=lr,steps=steps,selection=selection,teacher=teacher,config=config)
    sync()
    return costs(result,meter,time.perf_counter()-tick,backtracking,selection)


def use_last(result):
    """Optional sequential ablation from the same actual saved Adam trajectory."""
    assert not result['optimizer_variant']['backtracking']
    out = copy.deepcopy(result)
    idx = 0 if result.get('failure') else result['last_step']
    out.update(state=copy.deepcopy(result['initial_state'] if idx == 0 else result['last_state']),
        best_step=idx, final=copy.deepcopy(result['path'][idx]),
        state_delta=result['path'][idx]['state_delta'], selected_from_saved_trajectory=True)
    out['optimizer_variant']['selection'] = 'last'
    return out
