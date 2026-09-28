"""Research-only budget and measurement wrappers; the released method is read-only.

No labels, dataset lookup, state selection by utility, or production defaults.
The K experiment changes only the uniform observation budget and planned loss
denominator. Generalization executes the unchanged compact C+D computation.
"""
import copy
import time

import numpy as np
import torch

from methods.decota_final_simplified_v1 import observations as obs_module
from methods.decota_final_simplified_v1.backbone import (
    capture, full_prediction, make_batch, query_subject, validate_input,
)
from methods.decota_final_simplified_v1.objectives import SpatialLoss, prediction
from methods.decota_final_simplified_v1.optim import _fit, fit_spatial, fit_temporal
from methods.decota_final_simplified_v1.predictor import _EpisodeParser
from methods.decota_final_simplified_v1.replay import SpatialReplay, TemporalReplay
from methods.decota_final_simplified_v1.tensors import detached, state_hash


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def budget_loss(anchors, boxes, planned):
    if isinstance(planned, bool) or planned not in (1, 2, 4):
        raise ValueError("Only the predeclared K1/2/4 panel is authorized")
    loss = SpatialLoss(anchors, boxes)
    # The unchanged function is sum/4. K1/2 use an exact power-of-two scale;
    # there is no renormalization by accepted count.
    return loss, lambda value: loss(value["boxes"]) * (4. / planned)


@torch.no_grad()
def observe_budget(expert, parses, frames, ids, interval, planned):
    context, s1 = parses['context'], parses['old']
    eligible = context['eligible'] and len(expert.processor.tokenizer(context['context'])['input_ids']) <= 256
    positions = obs_module.uniform_positions(ids, interval, planned)
    observed, anchors = {}, []
    for pos in positions:
        if not eligible and not s1['phrase']:
            continue
        sync(); start = time.perf_counter()
        rgb = frames[pos]
        d = (obs_module.ContextView(expert, context)(rgb, context['context'], context['entity'])
             if eligible else expert(rgb, s1['phrase'], s1['entity']))
        sync()
        if eligible:
            z = obs_module.probe_from_detection(d, pos, ids[pos])
        else:
            z = dict(position=pos, frame_id=ids[pos], accepted=d['accepted'],
                     reason=d['reason'], margin=d['margin'],
                     boxes=torch.tensor([d['box']]) if d['box'] else torch.empty(0, 4),
                     target_scores=[d['score']] if d['box'] else [], candidate_ids=[0] if d['box'] else [])
        observed[pos] = dict(probe=z, seconds=time.perf_counter()-start,
                             forward_seconds=d['seconds'], text=d['text'], context_active=eligible)
        if z['accepted']:
            at = int(np.argmax(z['target_scores']))
            anchors.append(dict(position=pos, frame_id=ids[pos], box=torch.as_tensor(z['boxes'][at]).tolist(),
                                score=float(z['target_scores'][at]), weight=1.))
    return dict(positions=positions, observations=observed, anchors=anchors,
                actual_observation_positions=sorted(observed), planned=planned,
                new_DINO=len(observed), GT_online=False)


def _parses(predictor, metadata):
    parser = _EpisodeParser(predictor.parser)
    return dict(subject=parser(metadata['caption'])['subject'],
                context=obs_module.parse_context(parser, metadata['caption']),
                old=obs_module.visual_query(parser, metadata['caption']))


def generalization_prediction(predictor, frames, ids, metadata):
    """Same operations as released predictor, with frozen output and cost counters."""
    validate_input(frames, ids, metadata)
    before = (state_hash(predictor.model.state_dict()), state_hash(predictor.expert.model.state_dict()))
    sync(); start = time.perf_counter(); cost = {}
    parses = _parses(predictor, metadata)
    batch = make_batch(frames, ids, metadata, predictor.model)
    with query_subject(predictor.model, batch, parses['subject']):
        sync(); tick = time.perf_counter()
        views, records = capture(predictor.model, batch)
        spatial = SpatialReplay(predictor.model, views, len(ids))
        zero = spatial.zero
        native = prediction(zero['logits'], zero['boxes'], records, ids)
        sync(); cost['native_capture'] = time.perf_counter()-tick
        tick = time.perf_counter()
        expert = obs_module.observations(predictor.expert, parses, frames, ids, native['indices'])
        sync(); cost['DINO'] = time.perf_counter()-tick
        tick = time.perf_counter()
        space = fit_spatial(spatial, zero, expert['anchors']['single4'], predictor.config)
        sync(); cost['spatial_fit'] = time.perf_counter()-tick
        tick = time.perf_counter()
        temporal = TemporalReplay(spatial.head, spatial.temporal_inputs,
                                  {**zero, 'boxes': space['final']['boxes']})
        timeres = fit_temporal(temporal, records, predictor.config)
        sync(); cost['temporal_fit'] = time.perf_counter()-tick
        tick = time.perf_counter()
        final = full_prediction(predictor.model, batch, ids, records,
                                {**space['state'], **timeres['shrunk_state']}, timeres['shrunk'])
        sync(); cost['final_forward'] = time.perf_counter()-tick
    after = (state_hash(predictor.model.state_dict()), state_hash(predictor.expert.model.state_dict()))
    if before != after:
        raise RuntimeError('Source/expert state changed')
    return dict(predictions=dict(Frozen=native, Full_DeCoTA=final), records=records, parses=parses,
                spatial=detached(space, 'cpu'), temporal=detached(timeres, 'cpu'), expert=expert,
                config=predictor.config.to_dict(), source_hashes=before,
                source_restored=True, GT_online=False, costs=cost,
                backwards=space['backwards']+timeres['backwards'],
                actual_DINO_calls=len(expert['actual_observation_positions']),
                seconds=time.perf_counter()-start, full_model_forward=True)


def budget_prediction(predictor, frames, ids, metadata, parent, verify_k4=False):
    validate_input(frames, ids, metadata)
    before = (state_hash(predictor.model.state_dict()), state_hash(predictor.expert.model.state_dict()))
    start = time.perf_counter()
    batch = make_batch(frames, ids, metadata, predictor.model)
    parses = parent['parses']; results = {}
    with query_subject(predictor.model, batch, parses['subject']):
        views, records = capture(predictor.model, batch)
        spatial = SpatialReplay(predictor.model, views, len(ids))
        zero = spatial.zero
        native = prediction(zero['logits'], zero['boxes'], records, ids)
        if not torch.equal(native['boxes'], parent['predictions']['Frozen']['boxes']):
            raise RuntimeError('Compact native differs from sealed panel')
        if any(not torch.equal(a, b) for a, b in zip(native['logits'], parent['native_logits'])):
            raise RuntimeError('Compact native temporal parity failure')
        # The temporal path is private and fixed across K. It is not fitted again.
        tstate = parent['anchored_temporal']['state']
        expected_time = parent['predictions']['Full_DeCoTA']
        for k in ([4] if verify_k4 else []) + [1, 2]:
            sync(); tick = time.perf_counter()
            if k == 4:
                expert = dict(anchors=copy.deepcopy(parent['spatial']['anchors']), planned=4,
                              actual_observation_positions=parent['actual_observation_positions'], new_DINO=0,
                              old_K4_engineering_parity=True)
            else:
                expert = observe_budget(predictor.expert, parses, frames, ids, native['indices'], k)
            sync(); observation_seconds = time.perf_counter()-tick; tick = time.perf_counter()
            loss, loss_fn = budget_loss(expert['anchors'], zero['boxes'], k)
            space = _fit(spatial, zero, loss_fn, lr=predictor.config.spatial_lr, steps=10,
                         temporal=False, empty=loss.empty)
            expected = {**expected_time, 'boxes': space['final']['boxes']}
            final = full_prediction(predictor.model, batch, ids, records, {**space['state'], **tstate}, expected)
            sync()
            if final['indices'] != expected_time['indices']:
                raise RuntimeError('K changed the supposedly fixed temporal output')
            if k == 4:
                if not torch.equal(final['boxes'], expected_time['boxes']):
                    raise RuntimeError('Compact K4 does not reproduce saved selected method')
                if any(not torch.equal(v.cpu(), parent['spatial']['state'][n].cpu()) for n,v in space['state'].items()):
                    raise RuntimeError('K4 learned spatial parameters differ')
            results[str(k)] = dict(prediction=final, spatial=detached(space, 'cpu'), expert=expert,
                                   planned_denominator=k, observation_seconds=observation_seconds,
                                   fit_and_final_seconds=time.perf_counter()-tick)
    if before != (state_hash(predictor.model.state_dict()), state_hash(predictor.expert.model.state_dict())):
        raise RuntimeError('Source state changed during K panel')
    return dict(arms=results, Frozen=native, fixed_temporal_indices=expected_time['indices'],
                parent_K4=expected_time, same_source_state=True, source_restored=True,
                GT_online=False, method_selection=False, seconds=time.perf_counter()-start)
