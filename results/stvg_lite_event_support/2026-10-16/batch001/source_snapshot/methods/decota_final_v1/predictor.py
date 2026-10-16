"""Composition of sealed A4 and F44/F45 algorithms; no new adaptation rule.

The A4 observation/loss/optimizer implementations are imported unchanged.
One original capture supplies the independent spatial and temporal replays.
Final output is read from a full original-model forward, never teacher boxes.
Prediction accepts only the inherited label-free input schema.
"""
import copy
import json
import time
from pathlib import Path

import torch

from methods.decota_spatial4_v1.predictor import Spatial4Predictor, observations
from methods.decota_spatial10_v1.predictor import sha256
from vg_tta.dense_support_tuning_v1 import move

ROOT = Path(__file__).resolve().parents[2]
METHOD = Path(__file__).resolve().parent


def temporal_config(direction):
    config = json.loads((METHOD/'configs.json').read_text())
    if direction not in config['temporal']:
        raise ValueError(direction)
    return {**config['fixed'], **config['temporal'][direction], 'eta': config['eta']}


class FinalDeCoTAPredictor(Spatial4Predictor):
    def __init__(self, model, expert, parser, direction):
        super().__init__(model, expert, parser, direction)
        self.temporal_config = temporal_config(direction)
        manifest = METHOD/'FINAL_METHOD.json'
        if manifest.exists():
            for name, expected in json.loads(manifest.read_text())['code_pins'].items():
                if sha256(ROOT/name) != expected:
                    raise RuntimeError('Frozen final-method dependency changed: '+name)

    def _predict(self, frames, ids, metadata):
        from scripts.audit_parametric_reinsertion_v1 import full_prediction
        from vg_tta.decota_spatial_extension_v1 import parse_context
        from vg_tta.decota_tastvg_episode_v1 import make_batch
        from vg_tta.dense_support_temporal_v1 import teacher_from_logits
        from vg_tta.dense_support_tuning_v1 import HeadReplay
        from vg_tta.foreground_runtime import state_digest
        from vg_tta.parametric_observation_v1 import ObservationReplay
        from vg_tta.shared_state_v1 import capture_shared
        from vg_tta.simplification_partial_v1 import FullInputHeadReplay, fit as spatial_fit
        from vg_tta.structured_temporal_calibration_v1 import fit as temporal_fit, projected_indices
        from vg_tta.structured_temporal_shrinkage_v1 import path_point, prediction
        from vg_tta.tg_spatial_tta_v1 import visual_query

        began = time.perf_counter()
        torch.cuda.synchronize()
        source_digest = state_digest(self.model)
        expert_digest = state_digest(self.expert.model)
        cost = {}
        tick = time.perf_counter()
        parses = dict(subject=self.parser(metadata['caption'])['subject'],
                      context=parse_context(self.parser, metadata['caption']),
                      old=visual_query(self.parser, metadata['caption']))
        batch = make_batch(frames, ids, metadata, parses['subject'], self.model)
        base16, _, records, _, _, views = capture_shared(self.model, batch)
        spatial = ObservationReplay(self.model, views, len(ids), 'spatial')
        with torch.no_grad():
            zero = spatial.values()
        native = prediction(zero['logits'], zero['boxes'], records, ids)
        torch.cuda.synchronize(); cost['native_capture_and_FP32_replay'] = time.perf_counter()-tick

        # Exact A4: original four observations in the native interval; no new choices.
        tick = time.perf_counter()
        ex = observations(self.expert, parses, frames, ids, native['indices'])
        torch.cuda.synchronize(); cost['expert_four_original_observations'] = time.perf_counter()-tick
        config = self._config
        assert {n:p.numel() for n,p in spatial.named} == config['trainable_parameters']
        tick = time.perf_counter()
        adapted_space = spatial_fit(spatial, records, ids,
            anchors=copy.deepcopy(ex['anchors']['single4']), planned=4,
            kappa=None, gamma=0., lr=config['lr'], steps=config['steps'])
        torch.cuda.synchronize(); cost['spatial_adaptation'] = time.perf_counter()-tick
        assert all(torch.equal(a.cpu(),b) for a,b in zip(zero['logits'],adapted_space['final']['logits']))

        # The full-input teacher is captured from the unchanged source temporal path.
        tick = time.perf_counter()
        captured = FullInputHeadReplay(ObservationReplay(self.model, views, len(ids), 'head'))
        assert torch.equal(captured.zero['boxes'], zero['boxes'])
        assert all(torch.equal(a,b) for a,b in zip(captured.zero['logits'],zero['logits']))
        teacher = teacher_from_logits(captured.zero['actions'], records)
        cache = dict(inputs=move(captured.inputs,'cpu'), head_state=move(captured.initial,'cpu'),
            zero=dict(boxes=adapted_space['final']['boxes'],logits=native['logits']),
            teacher=move(teacher,'cpu'), records=records, frame_ids=ids, GT_online=False)
        head = HeadReplay(cache)
        cfg = self.temporal_config
        hard = temporal_fit(head, records, ids, head.teacher,
                            {k:cfg[k] for k in ('center_fraction','prior_weight','epsilon','temperature','margin','gamma','beta')},
                            lr=cfg['lr'],steps=cfg['steps'])
        point = path_point(head, hard, records, ids, cfg['eta'])
        torch.cuda.synchronize(); cost['temporal_capture_fit_and_shrink'] = time.perf_counter()-tick

        # Both actual parameter groups are inserted together. Expected is an assertion,
        # not the returned model prediction. All original source modules are restored.
        tick = time.perf_counter()
        final = full_prediction(self.model, frames, ids, metadata, parses['subject'],
                                {**adapted_space['state'],**point['state']},point['parameter'])
        torch.cuda.synchronize(); cost['final_full_model_inference'] = time.perf_counter()-tick
        assert torch.equal(final['boxes'], adapted_space['final']['boxes'])
        assert state_digest(self.model) == source_digest
        assert state_digest(self.expert.model) == expert_digest
        preds = {
            'Frozen': native,
            'A4': {**native,'boxes':final['boxes']},
            'Temporal_final': {**point['parameter'],'boxes':native['boxes']},
            'Final_DeCoTA': {**point['parameter'],**final},
            'Direct_projection': {'boxes':final['boxes'],
                                  'indices':projected_indices(hard['evidence'],ids),
                                  'projected_raw_intervals':[t['interval'] for t in hard['evidence']['offsets']],
                                  'temporal_network_update':False},
            'Full_update_eta1': {**hard['final'],'boxes':final['boxes']},
            'Logit_damping_eta025': point['logit'],
        }
        # Explicit full physical intervals for every arm, including direct projection.
        for pred in preds.values():
            a,b = pred['indices']; pred['physical_interval'] = [ids[a],ids[b]+1]
        torch.cuda.synchronize(); cost['pipeline_seconds'] = time.perf_counter()-began
        return dict(method='decota_final_v1',direction=self.direction,
            boxes=final['boxes'],indices=final['indices'],frame_ids=ids,
            I_seed=native['indices'],I_out=final['indices'],predictions=preds,
            spatial=adapted_space,temporal=hard,anchored_temporal=point,
            expert=ex,parses=parses,teacher=move(teacher,'cpu'),records=records,
            # Small cached inputs support auditable supplementary controls, not fitting GT.
            temporal_cache=cache, native_fp16_indices=list(base16['predicted_indices']),
            native_logits=native['logits'],native_boxes=native['boxes'],
            full_forward_audit=final['audit'],cost=cost,
            spatial_parameter_changed=adapted_space['state_delta']>0,
            temporal_parameter_changed=point['state_delta']>0,
            source_restored=True,expert_frozen=True,GT_online=False,
            student_output_only=True,interpolation_output=False,
            eta=.25,temporal_parameters=66306,spatial_parameters=1792)
